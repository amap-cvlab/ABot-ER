#!/usr/bin/env python
"""Launch and aggregate local full evaluation shards for the three TP datasets."""

from __future__ import annotations

import argparse
import contextlib
import concurrent.futures as futures
import io
import json
import os
import socket
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from task_planning_config import (
        AI2THOR_RELEASES_ROOT,
        DEFAULT_ALFRED_SHARD_TIMEOUT,
        DEFAULT_ET_COMMAND_TIMEOUT,
        DEFAULT_HABITAT_SHARD_TIMEOUT,
        job_output_dir,
        job_scratch_dir,
    )
except ModuleNotFoundError:  # Support importing as scripts.run_full_eval_local.
    from scripts.task_planning_config import (
        AI2THOR_RELEASES_ROOT,
        DEFAULT_ALFRED_SHARD_TIMEOUT,
        DEFAULT_ET_COMMAND_TIMEOUT,
        DEFAULT_HABITAT_SHARD_TIMEOUT,
        job_output_dir,
        job_scratch_dir,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBODIED_ROOT = PROJECT_ROOT / "EmbodiedBench"
ET_RUNNER = PROJECT_ROOT / "scripts" / "et_plan_navigation_sample_runner.py"

ET_BENCHMARKS = [
    ("navigation_tasks", "navigation_tasks.json", "navigation"),
    ("navigation_tasks_with_spatial_constraint", "navigation_tasks_with_spatial_constraint.json", "navigation"),
    ("navigation_and_manipulation_tasks", "navigation_and_manipulation_tasks.json", "nav_manip_single"),
    (
        "navigation_and_manipulation_tasks_with_spatial_constraint",
        "navigation_and_manipulation_tasks_with_spatial_constraint.json",
        "nav_manip_single",
    ),
    (
        "navigation_and_manipulation_tasks_with_temporal_constraint",
        "navigation_and_manipulation_tasks_with_temporal_constraint.json",
        "nav_manip_multi",
    ),
    ("navi_and_mani_multi_two_arm", "navi_and_mani_multi_two_arm.json", "nav_manip_multi"),
    ("navi_and_mani_multi_one_arm", "navi_and_mani_multi_one_arm.json", "nav_manip_multi"),
]

ALFRED_EVAL_SETS = ["base", "common_sense", "complex_instruction", "spatial", "visual_appearance", "long_horizon"]
HABITAT_EVAL_SETS = [
    "base",
    "common_sense",
    "complex_instruction",
    "spatial_relationship",
    "visual_appearance",
    "long_horizon",
]

OFFICIAL_EB_MAX_EPISODE_STEPS = 30
OFFICIAL_EB_MAX_INVALID_ACTIONS = 10


def load_run_id(args: argparse.Namespace) -> str:
    return args.run_id


def jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def balanced_chunks(items: list[int], shard_count: int) -> list[list[int]]:
    """Split items into exactly shard_count non-empty chunks, differing by at most one item."""
    if shard_count <= 0 or shard_count > len(items):
        raise ValueError(f"shard_count must be in [1, {len(items)}], got {shard_count}")
    small, extra = divmod(len(items), shard_count)
    result: list[list[int]] = []
    offset = 0
    for index in range(shard_count):
        size = small + (1 if index < extra else 0)
        result.append(items[offset : offset + size])
        offset += size
    return result


def balanced_shards_per_eval_set(total_shards: int, eval_set_count: int) -> list[int]:
    if total_shards < eval_set_count:
        raise ValueError(f"total EB shards must be >= {eval_set_count}, got {total_shards}")
    small, extra = divmod(total_shards, eval_set_count)
    return [small + (1 if index < extra else 0) for index in range(eval_set_count)]


def run_shell(command: str, cwd: Path, status_path: Path) -> dict[str, Any]:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if existing.get("status") == "success":
            return {
                **existing,
                "skipped_existing_success": True,
                "discarded_excp": purge_exception_files(cwd),
            }
    started = time.time()
    status: dict[str, Any] = {
        "command": command,
        "cwd": str(cwd),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "status": "running",
        "hostname": socket.gethostname(),
    }
    print(json.dumps({"message": "shard_start", **status}, ensure_ascii=False), flush=True)
    proc = subprocess.Popen(command, cwd=str(cwd), shell=True, executable="/bin/bash")
    status["pid"] = proc.pid
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    rc = proc.wait()
    discarded_excp = purge_exception_files(cwd)
    ended = time.time()
    status.update(
        {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended)),
            "elapsed_sec": round(ended - started, 3),
            "discarded_excp": discarded_excp,
        }
    )
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"message": "shard_done", **status}, ensure_ascii=False), flush=True)
    return status


def purge_exception_files(root: Path) -> int:
    """Delete Unity exception sidecars instead of persisting or archiving them."""
    deleted = 0
    if not root.exists():
        return deleted
    for path in root.rglob("*.excp"):
        if not path.is_file():
            continue
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted


def common_api_env(args: argparse.Namespace) -> str:
    env = {
        "OPENAI_API_KEY": args.api_key,
        "OPENAI_API_BASE": args.api_base,
        "OPENAI_BASE_URL": args.api_base,
        "remote_url": args.api_base,
    }
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


def common_api_exports(args: argparse.Namespace) -> str:
    env = {
        "OPENAI_API_KEY": args.api_key,
        "OPENAI_API_BASE": args.api_base,
        "OPENAI_BASE_URL": args.api_base,
        "remote_url": args.api_base,
        "EB_DISABLE_RESPONSE_FORMAT": "0",
    }
    return " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items())


def model_basename(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def eb_budget_mode() -> str:
    return "official-default-30-10"


def write_eval_config(args: argparse.Namespace, root: Path, jobs: list[dict[str, Any]], manifest: Path) -> Path:
    config = {
        "run_id": args.run_id,
        "suite": args.suite,
        "model": args.model,
        "model_basename": model_basename(args.model),
        "api_base": args.api_base,
        "api_key_set": bool(args.api_key),
        "jobs": len(jobs),
        "max_jobs": args.max_jobs,
        "manifest": str(manifest),
        "budget_mode": eb_budget_mode(),
        "official_eb_budget": {
            "max_episode_steps": OFFICIAL_EB_MAX_EPISODE_STEPS,
            "max_invalid_actions": OFFICIAL_EB_MAX_INVALID_ACTIONS,
            "source": "EmbodiedBench default evaluator/env configuration",
        },
        "eb": {
            "shard_size": args.eb_shard_size,
            "shards_per_dataset": args.eb_shards_per_dataset,
            "alfred_timeout_sec": args.alfred_timeout,
            "habitat_timeout_sec": args.habitat_timeout,
            "max_completion_tokens": args.eb_max_completion_tokens,
            "response_format": "json_schema_enabled",
            "alfred_resolution": args.alfred_resolution,
            "habitat_resolution": args.habitat_resolution,
        },
        "et": {
            "shard_size": args.et_shard_size,
            "task_retries": args.et_task_retries,
            "port_base": args.et_port_base,
            "command_timeout_sec": args.et_command_timeout,
        },
        "workers": {
            "requested": args.workers,
            "all_workers": args.all_workers,
            "et_workers": args.et_workers,
            "eb_workers": args.eb_workers,
        },
    }
    path = root / "full" / f"eval_config_{args.suite}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_et_jobs(args: argparse.Namespace, root: Path, scratch_root: Path | None = None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    scratch_root = scratch_root or job_scratch_dir(args.run_id)
    runtime_dir = scratch_root / "et_plan"
    context_cache_dir = scratch_root / "et_plan_context_cache"
    work_dir = scratch_root / "et_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)
    for bench_name, bench_file, task_type in ET_BENCHMARKS:
        benchmark = PROJECT_ROOT / "ET-Plan-Bench" / "benchmark" / bench_file
        task_ids = list(range(jsonl_count(benchmark)))
        for shard_idx, shard_ids in enumerate(chunks(task_ids, args.et_shard_size)):
            shard_name = f"{bench_name}_shard{shard_idx:03d}"
            out_dir = root / "full" / "ET-Plan_Bench" / bench_name / shard_name
            status_path = root / "status" / f"full_et_{shard_name}.json"
            retry_port_stride = args.et_shard_size * 2 + 2
            shard_port_stride = retry_port_stride * (args.et_task_retries + 1)
            port_base = args.et_port_base + len(jobs) * shard_port_stride
            cmd = (
                f"{exception_cleanup_fallback_command()}"
                f"{common_api_env(args)} "
                "NOTEBOOK_CONTAINER=1 "
                f"ET_PLAN_RUNTIME_DIR={shlex.quote(str(runtime_dir))} "
                f"ET_PLAN_WORK_DIR={shlex.quote(str(work_dir))} "
                f"{shlex.quote(sys.executable)} {shlex.quote(str(ET_RUNNER))} "
                f"--benchmark {shlex.quote(str(benchmark))} "
                f"--task-type {shlex.quote(task_type)} "
                f"--task-ids {shlex.quote(','.join(map(str, shard_ids)))} "
                f"--model {shlex.quote(args.model)} "
                f"--port-base {port_base} "
                f"--task-retries {args.et_task_retries} "
                f"--retry-port-stride {retry_port_stride} "
                f"--command-timeout {args.et_command_timeout} "
                f"--output-dir {shlex.quote(str(out_dir))} "
                f"--context-cache-dir {shlex.quote(str(context_cache_dir))} "
                f"--reuse-existing; "
                "eval_rc=$?; "
                f"{exception_cleanup_fallback_command()}"
                "exit \"$eval_rc\""
            )
            jobs.append({"kind": "et", "name": shard_name, "command": cmd, "cwd": work_dir, "status": status_path})
    return jobs


def exception_cleanup_fallback_command() -> str:
    # Evaluators run from a job-scoped work directory. Never delete scratch
    # files from the repository or another job.
    roots = [Path(".")]
    commands: list[str] = []
    seen: set[str] = set()
    for root in roots:
        root_text = str(root)
        if root_text in seen:
            continue
        seen.add(root_text)
        commands.append(
            "find "
            f"{shlex.quote(root_text)} "
            "-name '*.excp' -type f "
            "-exec rm -f {} + 2>/dev/null || true"
        )
    return "; ".join(commands) + "; "


def xvfb_prefix(display: int, log_path: Path, clean_exception_fallback: bool = False) -> str:
    exception_cleanup = ""
    if clean_exception_fallback:
        exception_cleanup = exception_cleanup_fallback_command()
    quoted_log_path = shlex.quote(str(log_path))
    display_probe = 'timeout 1s xdpyinfo -display "$DISPLAY" >/dev/null 2>&1'
    return (
        f"export DISPLAY=:{display}; "
        "display_num=${DISPLAY#:}; "
        f'if [ -e "/tmp/.X${{display_num}}-lock" ] && ! {display_probe}; then '
        'rm -f "/tmp/.X${display_num}-lock" "/tmp/.X11-unix/X${display_num}"; '
        "fi; "
        f"{exception_cleanup}"
        f"mkdir -p {shlex.quote(str(log_path.parent))}; "
        f"Xvfb :{display} -screen 0 1024x768x24 +extension GLX +render -noreset >{quoted_log_path} 2>&1 & "
        "xvfb_pid=$!; "
        f"cleanup(){{ kill -9 \"$xvfb_pid\" 2>/dev/null || true; {exception_cleanup}}}; "
        "trap cleanup EXIT; "
        f'for _ in $(seq 1 20); do {display_probe} && break; sleep 0.5; done; '
        f'if ! {display_probe}; then cat '
        f"{quoted_log_path}; exit 86; fi; "
    )


def alfred_ai2thor_home_prefix(root: Path, exp_name: str) -> str:
    """Isolate simulator caches without changing HOME or deleting user files.

    ABot-ER portability patch: ThorEnv consumes this dedicated cache variable.
    Scoring, simulator builds and episode inputs are unchanged.
    """
    if Path(exp_name).name != exp_name or exp_name in {".", ".."}:
        raise ValueError("Unsafe simulator shard name")
    cache_dir = (root / "runtime" / "ai2thor_cache" / exp_name).resolve()
    if not cache_dir.is_relative_to(root.resolve()):
        raise ValueError("Simulator cache escaped runtime root")
    cache_dir.mkdir(parents=True, exist_ok=True)
    releases = cache_dir / "releases"
    if not AI2THOR_RELEASES_ROOT.is_dir():
        raise FileNotFoundError(f"Frozen AI2-THOR releases missing: {AI2THOR_RELEASES_ROOT}")
    if not releases.exists() and not releases.is_symlink():
        releases.symlink_to(AI2THOR_RELEASES_ROOT, target_is_directory=True)
    elif releases.resolve() != AI2THOR_RELEASES_ROOT.resolve():
        raise ValueError("Existing AI2-THOR release link points to a different build")
    return f"export ABOT_AI2THOR_BASE_DIR={shlex.quote(str(cache_dir))}; "


def eb_direct_run_root(root: Path, dataset: str) -> Path:
    if dataset == "alfred":
        return root / "full" / "EB-ALFRED" / "runs"
    if dataset == "habitat":
        return root / "full" / "EB-Habitat" / "runs"
    raise ValueError(f"Unknown EB dataset: {dataset}")


def prepare_embodiedbench_workdir(scratch_root: Path, root: Path | None = None) -> Path:
    work_dir = scratch_root / "embodiedbench"
    work_dir.mkdir(parents=True, exist_ok=True)
    package_link = work_dir / "embodiedbench"
    if not package_link.exists():
        package_link.symlink_to(EMBODIED_ROOT / "embodiedbench", target_is_directory=True)
    if root is not None:
        running_dir = work_dir / "running"
        running_dir.mkdir(parents=True, exist_ok=True)
        for dataset, name in [("alfred", "eb_alfred"), ("habitat", "eb_habitat")]:
            target = eb_direct_run_root(root, dataset)
            target.mkdir(parents=True, exist_ok=True)
            link = running_dir / name
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    link.unlink()
                    link.symlink_to(target, target_is_directory=True)
            elif not link.exists():
                link.symlink_to(target, target_is_directory=True)
            elif link.resolve() != target.resolve():
                raise RuntimeError(f"EmbodiedBench running directory is not the NAS output target: {link}")
    return work_dir


def build_alfred_jobs(args: argparse.Namespace, root: Path, scratch_root: Path | None = None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    scratch_root = scratch_root or job_scratch_dir(args.run_id)
    work_dir = prepare_embodiedbench_workdir(scratch_root, root)
    shard_counts = balanced_shards_per_eval_set(args.eb_shards_per_dataset, len(ALFRED_EVAL_SETS))
    for eval_set, shard_count in zip(ALFRED_EVAL_SETS, shard_counts):
        for shard_idx, shard_ids in enumerate(balanced_chunks(list(range(50)), shard_count)):
            shard_name = f"{eval_set}_shard{shard_idx:03d}"
            display = args.display_base + len(jobs)
            exp = f"{args.run_id}_full_alfred_{eval_set}_shard{shard_idx:03d}"
            status_path = root / "status" / f"full_eb_alfred_{shard_name}.json"
            cmd = (
                f"{common_api_exports(args)} "
                f"export PYTHONPATH={shlex.quote(str(EMBODIED_ROOT))}:\"${{PYTHONPATH:-}}\"; "
                f"export EB_MAX_COMPLETION_TOKENS={args.eb_max_completion_tokens}; "
                "export LIBGL_ALWAYS_SOFTWARE=1; "
                "export GALLIUM_DRIVER=llvmpipe; "
                "export MESA_GL_VERSION_OVERRIDE=3.3; "
                "export MESA_GLSL_VERSION_OVERRIDE=330; "
                f"export EB_ALFRED_X_DISPLAY={display}; "
                f"{alfred_ai2thor_home_prefix(scratch_root, exp)}"
                f"{xvfb_prefix(display, scratch_root / 'xvfb' / f'alfred_{display}.log', clean_exception_fallback=True)}"
                f"timeout {args.alfred_timeout}s {shlex.quote(sys.executable)} -u -m embodiedbench.main "
                f"env=eb-alf model_name={shlex.quote(args.model)} model_type=remote "
                f"exp_name={shlex.quote(exp)} eval_sets='[{eval_set}]' "
                f"+selected_indexes='[{','.join(map(str, shard_ids))}]' "
                "n_shots=0 language_only=0 chat_history=0 multistep=0 +env_feedback=1 "
                f"resolution={args.alfred_resolution} down_sample_ratio=1 log_level=INFO"
            )
            run_dir = eb_direct_run_root(root, "alfred") / f"{model_basename(args.model)}_{exp}" / eval_set
            jobs.append({"kind": "alfred", "name": shard_name, "command": cmd, "cwd": work_dir, "status": status_path, "run_dir": run_dir})
    return jobs


def build_habitat_jobs(args: argparse.Namespace, root: Path, scratch_root: Path | None = None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    scratch_root = scratch_root or job_scratch_dir(args.run_id)
    work_dir = prepare_embodiedbench_workdir(scratch_root, root)
    shard_counts = balanced_shards_per_eval_set(args.eb_shards_per_dataset, len(HABITAT_EVAL_SETS))
    for eval_set, shard_count in zip(HABITAT_EVAL_SETS, shard_counts):
        for shard_idx, shard_ids in enumerate(balanced_chunks(list(range(50)), shard_count)):
            start = shard_ids[0]
            shard_size = len(shard_ids)
            shard_name = f"{eval_set}_shard{shard_idx:03d}"
            display = args.display_base + 100 + len(jobs)
            exp = f"{args.run_id}_full_habitat_{eval_set}_shard{shard_idx:03d}"
            status_path = root / "status" / f"full_eb_habitat_{shard_name}.json"
            cmd = (
                f"{common_api_exports(args)} "
                f"export PYTHONPATH={shlex.quote(str(EMBODIED_ROOT))}:\"${{PYTHONPATH:-}}\"; "
                f"export EB_MAX_COMPLETION_TOKENS={args.eb_max_completion_tokens}; "
                "export LIBGL_ALWAYS_SOFTWARE=1; "
                f"{xvfb_prefix(display, scratch_root / 'xvfb' / f'habitat_{display}.log', clean_exception_fallback=True)}"
                f"timeout {args.habitat_timeout}s {shlex.quote(sys.executable)} -u -m embodiedbench.main "
                f"env=eb-hab model_name={shlex.quote(args.model)} model_type=remote "
                f"exp_name={shlex.quote(exp)} eval_sets='[{eval_set}]' +start_epi_index={start} "
                f"+max_episodes={shard_size} "
                "n_shots=0 language_only=0 chat_history=0 multistep=0 +env_feedback=1 "
                f"resolution={args.habitat_resolution} down_sample_ratio=1 log_level=INFO"
            )
            run_dir = eb_direct_run_root(root, "habitat") / f"{model_basename(args.model)}_{exp}" / eval_set
            jobs.append({"kind": "habitat", "name": shard_name, "command": cmd, "cwd": work_dir, "status": status_path, "run_dir": run_dir})
    return jobs


def launch(args: argparse.Namespace) -> None:
    args.run_id = load_run_id(args)
    root = job_output_dir(args.run_id)
    scratch_root = job_scratch_dir(args.run_id)
    selected: list[dict[str, Any]] = []
    if args.suite in {"all", "et"}:
        selected.extend(build_et_jobs(args, root, scratch_root))
    if args.suite in {"all", "eb", "alfred"}:
        selected.extend(build_alfred_jobs(args, root, scratch_root))
    if args.suite in {"all", "eb", "habitat"}:
        selected.extend(build_habitat_jobs(args, root, scratch_root))
    if args.max_jobs is not None:
        selected = selected[: args.max_jobs]

    manifest = root / "full" / f"manifest_{args.suite}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(selected, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    eval_config = write_eval_config(args, root, selected, manifest)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "suite": args.suite,
                "jobs": len(selected),
                "manifest": str(manifest),
                "eval_config": str(eval_config),
                "budget_mode": eb_budget_mode(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return

    max_workers = args.workers
    if args.suite == "et" and args.workers is None:
        max_workers = args.et_workers
    elif args.suite in {"eb", "alfred", "habitat"} and args.workers is None:
        max_workers = args.eb_workers
    elif args.workers is None:
        max_workers = args.all_workers

    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(run_shell, job["command"], job["cwd"], job["status"]): job
            for job in selected
        }
        for fut in futures.as_completed(future_to_job):
            job = future_to_job[fut]
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover - operational guard
                result = {"status": "failed", "error": repr(exc)}
            print(json.dumps({"job": job["name"], **result}, ensure_ascii=False), flush=True)
    if not args.no_aggregate:
        aggregate(args)


def official_average_json_values():
    if str(EMBODIED_ROOT) not in sys.path:
        sys.path.insert(0, str(EMBODIED_ROOT))
    from embodiedbench.evaluator.summarize_result import average_json_values

    return average_json_values


def mirror_episode_results(paths: list[Path], dest_root: Path) -> int:
    copied = 0
    for path in paths:
        try:
            exp_dir = path.parents[2].name
        except IndexError:
            exp_dir = "unknown_exp"
        dest = dest_root / exp_dir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        copied += 1
    return copied


def mirror_model_outputs(paths: list[Path], dest_root: Path) -> int:
    copied = 0
    for path in paths:
        try:
            exp_dir = path.parents[2].name
        except IndexError:
            exp_dir = "unknown_exp"
        dest = dest_root / exp_dir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        copied += 1
    return copied


def eb_episode_paths(run_id: str, model: str, dataset: str, root: Path | None = None) -> list[Path]:
    model_name = model_basename(model)
    bases: list[Path] = []
    if dataset == "alfred":
        pattern = f"{model_name}_{run_id}_full_alfred_*/*/results/episode_*_final_res.json"
        fallback = f"*_{run_id}_full_alfred_*/*/results/episode_*_final_res.json"
        if root is not None:
            bases.append(eb_direct_run_root(root, "alfred"))
        bases.append(job_scratch_dir(run_id) / "embodiedbench" / "running" / "eb_alfred")
    elif dataset == "habitat":
        pattern = f"{model_name}_{run_id}_full_habitat_*/*/results/episode_*_final_res.json"
        fallback = f"*_{run_id}_full_habitat_*/*/results/episode_*_final_res.json"
        if root is not None:
            bases.append(eb_direct_run_root(root, "habitat"))
        bases.append(job_scratch_dir(run_id) / "embodiedbench" / "running" / "eb_habitat")
    else:
        raise ValueError(f"Unknown EB dataset: {dataset}")
    for base in bases:
        paths = sorted(base.glob(pattern))
        if paths:
            return paths
        paths = sorted(base.glob(fallback))
        if paths:
            return paths
    return []


def eb_model_output_paths(run_id: str, model: str, dataset: str, root: Path | None = None) -> list[Path]:
    model_name = model_basename(model)
    bases: list[Path] = []
    if dataset == "alfred":
        pattern = f"{model_name}_{run_id}_full_alfred_*/*/model_outputs/episode_*_model_outputs.jsonl"
        fallback = f"*_{run_id}_full_alfred_*/*/model_outputs/episode_*_model_outputs.jsonl"
        if root is not None:
            bases.append(eb_direct_run_root(root, "alfred"))
        bases.append(job_scratch_dir(run_id) / "embodiedbench" / "running" / "eb_alfred")
    elif dataset == "habitat":
        pattern = f"{model_name}_{run_id}_full_habitat_*/*/model_outputs/episode_*_model_outputs.jsonl"
        fallback = f"*_{run_id}_full_habitat_*/*/model_outputs/episode_*_model_outputs.jsonl"
        if root is not None:
            bases.append(eb_direct_run_root(root, "habitat"))
        bases.append(job_scratch_dir(run_id) / "embodiedbench" / "running" / "eb_habitat")
    else:
        raise ValueError(f"Unknown EB dataset: {dataset}")
    for base in bases:
        paths = sorted(base.glob(pattern))
        if paths:
            return paths
        paths = sorted(base.glob(fallback))
        if paths:
            return paths
    return []


def archived_eb_episode_paths(root: Path, dataset_name: str) -> list[Path]:
    return sorted((root / "full" / dataset_name / "episode_results").glob("*/episode_*_final_res.json"))


def archived_eb_model_output_paths(root: Path, dataset_name: str) -> list[Path]:
    return sorted((root / "full" / dataset_name / "model_outputs").glob("*/episode_*_model_outputs.jsonl"))


def eb_exp_dir_from_result_path(path: Path) -> str:
    if path.parent.name in {"results", "model_outputs"} and len(path.parents) >= 3:
        return path.parents[2].name
    return path.parent.name


def official_eb_summary(paths: list[Path], root: Path, dataset_name: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"tp_summary_{dataset_name}_") as temp_dir:
        staging_results = Path(temp_dir)
        staging_running = staging_results / "running"
        staging_running.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(sorted(paths)):
            exp_dir = eb_exp_dir_from_result_path(path)
            dest = staging_running / f"episode_{index:04d}_{exp_dir}_{path.name}"
            try:
                dest.hardlink_to(path)
            except OSError:
                shutil.copyfile(path, dest)

        average_json_values = official_average_json_values()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            average_json_values(str(staging_results), target_file="episode_*_final_res.json", output_file="summary.json")
        summary = json.loads((staging_results / "summary.json").read_text(encoding="utf-8"))
    out_summary = root / "full" / dataset_name / "summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "episode_count": len(paths),
        **summary,
        "summary_source": "EmbodiedBench.average_json_values",
        "official_summary_path": str(out_summary),
    }


def aggregate(args: argparse.Namespace) -> None:
    run_id = load_run_id(args)
    root = job_output_dir(run_id)
    result: dict[str, Any] = {"run_id": run_id}

    et_strict = root / "strict_rescore" / "et_strict_summary.json"
    if et_strict.exists():
        strict = json.loads(et_strict.read_text(encoding="utf-8"))
        result["ET-Plan_Bench"] = {
            "task_count": strict.get("task_count", 0),
            "summary_files": strict.get("summary_files", 0),
            "success_rate": strict.get("task_success_rate", strict.get("success_rate", 0)),
            "task_success_rate": strict.get("task_success_rate", 0),
            "format_success_rate": strict.get("format_success_rate", 0),
            "action_execution_success_rate": strict.get("action_execution_success_rate", 0),
            "average_steps": strict.get("average_steps", 0),
            "average_lcs": strict.get("average_lcs", 0),
            "average_lcs_ratio": strict.get("average_lcs_ratio", 0),
        }
    else:
        et_summaries = sorted((root / "full" / "ET-Plan_Bench").glob("*/*/summary.json"))
        et_rows = [json.loads(path.read_text(encoding="utf-8")) for path in et_summaries]
        et_total = sum(int(row.get("task_count", 0)) for row in et_rows)
        if et_total:
            def weighted(key: str, default: float = 0.0) -> float:
                return (
                    sum(float(row.get(key, default) or 0.0) * int(row.get("task_count", 0)) for row in et_rows)
                    / et_total
                )

            result["ET-Plan_Bench"] = {
                "task_count": et_total,
                "summary_files": len(et_summaries),
                "scoring_mode": "official-output",
                "success_rate": weighted("success_rate"),
                "task_success_rate": weighted("task_success_rate"),
                "format_success_rate": weighted("format_success_rate"),
                "action_execution_success_rate": None,
                "average_steps": weighted("average_steps"),
                "average_lcs": weighted("average_lcs"),
                "average_lcs_ratio": weighted("average_lcs_ratio"),
            }

    alfred_raw_paths = eb_episode_paths(run_id, args.model, "alfred", root)
    habitat_raw_paths = eb_episode_paths(run_id, args.model, "habitat", root)
    alfred_model_outputs = eb_model_output_paths(run_id, args.model, "alfred", root)
    habitat_model_outputs = eb_model_output_paths(run_id, args.model, "habitat", root)
    if alfred_raw_paths and not all(path.is_relative_to(root) for path in alfred_raw_paths):
        mirror_episode_results(alfred_raw_paths, root / "full" / "EB-ALFRED" / "episode_results")
        mirror_model_outputs(alfred_model_outputs, root / "full" / "EB-ALFRED" / "model_outputs")
    if habitat_raw_paths and not all(path.is_relative_to(root) for path in habitat_raw_paths):
        mirror_episode_results(habitat_raw_paths, root / "full" / "EB-Habitat" / "episode_results")
        mirror_model_outputs(habitat_model_outputs, root / "full" / "EB-Habitat" / "model_outputs")

    alfred_paths = alfred_raw_paths or archived_eb_episode_paths(root, "EB-ALFRED")
    habitat_paths = habitat_raw_paths or archived_eb_episode_paths(root, "EB-Habitat")
    if alfred_paths:
        result["EB-ALFRED"] = official_eb_summary(alfred_paths, root, "EB-ALFRED")
        result["EB-ALFRED"]["mirrored_episode_results"] = len(alfred_paths)
        result["EB-ALFRED"]["mirrored_model_output_files"] = len(
            archived_eb_model_output_paths(root, "EB-ALFRED")
        )
    if habitat_paths:
        result["EB-Habitat"] = official_eb_summary(habitat_paths, root, "EB-Habitat")
        result["EB-Habitat"]["mirrored_episode_results"] = len(habitat_paths)
        result["EB-Habitat"]["mirrored_model_output_files"] = len(
            archived_eb_model_output_paths(root, "EB-Habitat")
        )

    status_files = list((root / "status").glob("full_*.json"))
    if status_files:
        statuses = [json.loads(path.read_text(encoding="utf-8")) for path in status_files]
        result["runtime"] = {
            "shards_total": len(statuses),
            "shards_success": sum(1 for item in statuses if item.get("status") == "success"),
            "shards_failed": sum(1 for item in statuses if item.get("status") == "failed"),
            "elapsed_sec_sum": sum(float(item.get("elapsed_sec") or 0) for item in statuses),
        }

    out = root / "full" / "aggregate_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--job-name", dest="run_id", required=True)
    common.add_argument("--model", default="qwen35-vl")
    common.add_argument("--api-base", default="http://127.0.0.1:29199/v1")
    common.add_argument("--api-key", default="EMPTY")

    launch_p = sub.add_parser("launch", parents=[common])
    launch_p.add_argument("--suite", choices=["all", "et", "eb", "alfred", "habitat"], default="all")
    launch_p.add_argument("--dry-run", action="store_true")
    launch_p.add_argument("--workers", type=int)
    launch_p.add_argument("--all-workers", type=int, default=4)
    launch_p.add_argument("--et-workers", type=int, default=3)
    launch_p.add_argument("--eb-workers", type=int, default=2)
    launch_p.add_argument("--et-shard-size", type=int, default=25)
    launch_p.add_argument("--et-task-retries", type=int, default=2)
    launch_p.add_argument("--et-port-base", type=int, default=31000)
    launch_p.add_argument("--et-command-timeout", type=int, default=DEFAULT_ET_COMMAND_TIMEOUT)
    launch_p.add_argument("--eb-shard-size", type=int, default=10)
    launch_p.add_argument(
        "--eb-shards-per-dataset",
        type=int,
        default=80,
        help="Exact balanced shard count for each EB dataset; default 80 for the 8-GPU service.",
    )
    launch_p.add_argument("--display-base", type=int, default=130)
    launch_p.add_argument("--alfred-timeout", type=int, default=DEFAULT_ALFRED_SHARD_TIMEOUT)
    launch_p.add_argument("--habitat-timeout", type=int, default=DEFAULT_HABITAT_SHARD_TIMEOUT)
    launch_p.add_argument("--eb-max-completion-tokens", type=int, default=512)
    launch_p.add_argument("--alfred-resolution", type=int, default=300)
    launch_p.add_argument("--habitat-resolution", type=int, default=128)
    launch_p.add_argument("--no-aggregate", action="store_true")
    launch_p.add_argument(
        "--max-jobs",
        type=int,
        help="Launch only the first N generated shard jobs. Intended for throughput probes; unset for official full runs.",
    )
    launch_p.set_defaults(func=launch)

    agg_p = sub.add_parser("aggregate", parents=[common])
    agg_p.set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
