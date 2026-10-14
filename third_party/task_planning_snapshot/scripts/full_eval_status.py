#!/usr/bin/env python
"""Report full-evaluation progress and retry gaps without scanning heavy artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from task_planning_config import job_output_dir, job_scratch_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_RUNNING_GRACE_SEC = 6 * 3600


def parse_utc(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_id(args: argparse.Namespace) -> str:
    return args.run_id


def load_manifest(root: Path) -> list[dict[str, Any]]:
    manifests = sorted((root / "full").glob("manifest_*.json"))
    if not manifests:
        return []
    merged: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        try:
            jobs = load_json(manifest)
        except Exception:
            continue
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if isinstance(job, dict) and job.get("status"):
                merged[status_name_for_job(job)] = job
    return [merged[key] for key in sorted(merged)]


def load_statuses(root: Path) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for path in (root / "status").glob("full_*.json"):
        try:
            item = load_json(path)
        except Exception:
            continue
        item["_file"] = path.name
        statuses[path.name] = item
    return statuses


def status_name_for_job(job: dict[str, Any]) -> str:
    return Path(str(job["status"])).name


def status_process_is_active(status: dict[str, Any]) -> bool:
    """Keep fresh remote running records active; verify local PIDs directly."""
    if status.get("status") != "running":
        return False
    if not status.get("hostname") or not status.get("pid"):
        # Backward compatibility for status files produced before PID tracking:
        # avoid duplicating a legitimately long shard, but recover it after six hours.
        started = parse_utc(status.get("started_at"))
        return bool(started and time.time() - started < REMOTE_RUNNING_GRACE_SEC)
    if status.get("hostname") != socket.gethostname():
        # NAS status is also read from submission hosts, where the container PID
        # cannot be probed. Treat a fresh record as active instead of scheduling
        # a duplicate retry while the remote shard is still running.
        started = parse_utc(status.get("started_at"))
        return bool(started and time.time() - started < REMOTE_RUNNING_GRACE_SEC)
    try:
        pid = int(status["pid"])
        os.kill(pid, 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def classify_statuses(manifest: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    running: list[str] = []
    running_details: list[dict[str, Any]] = []
    failed: list[str] = []
    missing: list[str] = []
    elapsed_by_kind: dict[str, list[float]] = {}
    started_values: list[float] = []
    ended_values: list[float] = []
    now = time.time()

    for job in manifest:
        kind = job.get("kind", "unknown")
        status_file = status_name_for_job(job)
        status = statuses.get(status_file)
        state = "pending" if status is None else status.get("status", "unknown")
        if state == "running" and not status_process_is_active(status):
            state = "stale_running"
        by_state[state] = by_state.get(state, 0) + 1
        by_kind.setdefault(kind, {})
        by_kind[kind][state] = by_kind[kind].get(state, 0) + 1
        if state == "running":
            running.append(status_file)
            started = parse_utc(status.get("started_at")) if status else None
            running_details.append(
                {
                    "status_file": status_file,
                    "kind": kind,
                    "elapsed_running_sec": round(now - started, 3) if started else None,
                    "started_at": status.get("started_at") if status else None,
                }
            )
        elif state == "failed":
            failed.append(status_file)
        elif state == "pending":
            missing.append(status_file)

        if status:
            started = parse_utc(status.get("started_at"))
            ended = parse_utc(status.get("ended_at"))
            if started:
                started_values.append(started)
            if ended:
                ended_values.append(ended)
            if status.get("status") == "success" and status.get("elapsed_sec") is not None:
                elapsed_by_kind.setdefault(kind, []).append(float(status["elapsed_sec"]))

    first_started = min(started_values) if started_values else None
    has_active_jobs = any(by_state.get(key, 0) > 0 for key in ("running", "pending", "stale_running"))
    last_ended = max(ended_values) if ended_values else None
    if first_started and has_active_jobs:
        wall_elapsed = now - first_started
    elif first_started and last_ended:
        wall_elapsed = last_ended - first_started
    else:
        wall_elapsed = None
    avg_success_elapsed = {
        kind: round(mean(values), 3)
        for kind, values in elapsed_by_kind.items()
        if values
    }

    running_count = max(by_state.get("running", 0), 1)
    success_count = by_state.get("success", 0)
    known_avg = mean([item for values in elapsed_by_kind.values() for item in values]) if elapsed_by_kind else None
    remaining = by_state.get("pending", 0) + by_state.get("running", 0) + by_state.get("stale_running", 0)
    eta_sec = (remaining * known_avg / running_count) if known_avg else None
    progress_ratio = success_count / len(manifest) if manifest else 0.0
    throughput_jobs_per_hour = (success_count / wall_elapsed * 3600) if wall_elapsed and wall_elapsed > 0 else None
    estimated_total_sec = (wall_elapsed / progress_ratio) if wall_elapsed and progress_ratio > 0 else None

    return {
        "total_jobs": len(manifest),
        "by_state": by_state,
        "by_kind": by_kind,
        "running": sorted(running),
        "running_details": sorted(running_details, key=lambda item: item["status_file"]),
        "failed": sorted(failed),
        "pending": sorted(missing),
        "first_started_at": datetime.fromtimestamp(first_started, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        if first_started
        else None,
        "wall_elapsed_sec": round(wall_elapsed, 3) if wall_elapsed else None,
        "progress_ratio": round(progress_ratio, 6),
        "progress_percent": round(progress_ratio * 100, 2),
        "throughput_jobs_per_hour": round(throughput_jobs_per_hour, 3) if throughput_jobs_per_hour else None,
        "estimated_total_sec": round(estimated_total_sec, 3) if estimated_total_sec else None,
        "avg_success_elapsed_sec_by_kind": avg_success_elapsed,
        "rough_eta_sec": round(eta_sec, 3) if eta_sec else None,
    }


def et_task_details(root: Path) -> dict[str, Any]:
    strict_summary = root / "strict_rescore" / "et_strict_summary.json"
    base = root / "strict_rescore" / "ET-Plan_Bench" if strict_summary.exists() else root / "full" / "ET-Plan_Bench"
    result: dict[str, Any] = {}
    if not base.exists():
        return {"_note": "ET-Plan_Bench details are omitted until full/ET-Plan_Bench summaries exist."}
    for bench_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        shard_summaries = sorted(bench_dir.glob("*/summary.json"))
        task_count = 0
        task_errors = 0
        score_success = 0.0
        for summary in shard_summaries:
            try:
                data = load_json(summary)
            except Exception:
                continue
            rows = data.get("results", [])
            if isinstance(rows, list):
                task_count += len(rows)
                task_errors += sum(1 for row in rows if row.get("error"))
                score_success += sum(float(row.get("success_rate", 0)) for row in rows)
            else:
                count = int(data.get("task_count", 0))
                task_count += count
                score_success += float(data.get("success_rate", 0)) * count
        standalone_errors = [path for path in sorted(bench_dir.glob("*/task_*_error.json")) if not task_error_resolved(path)]
        missing_predictions = sorted(bench_dir.glob("*/predictions/task_*"))
        result[bench_dir.name] = {
            "shard_summaries": len(shard_summaries),
            "tasks_in_shard_summaries": task_count,
            "task_errors_in_summaries": task_errors,
            "standalone_task_error_files": len(standalone_errors),
            "average_task_success_rate_over_recorded_tasks": round(score_success / task_count, 6) if task_count else None,
            "scoring_source": "strict_rescore" if strict_summary.exists() else "official-output",
            "latest_error_files": [str(path.relative_to(root)) for path in standalone_errors[-10:]],
            "prediction_dirs": len(missing_predictions),
        }
    return result


def root_artifact_counts(root: Path) -> dict[str, int]:
    work_dir = job_scratch_dir(root.name) / "et_workdir"
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    return {
        "port_txt": len(list(work_dir.glob("port_*.txt"))),
        "player_log": len(list(work_dir.glob("Player_*.log"))),
        "excp": len(list(work_dir.glob("*.excp"))),
        "uuid_dirs": sum(1 for path in work_dir.iterdir() if path.is_dir() and uuid_pattern.match(path.name)) if work_dir.exists() else 0,
    }


def eb_artifact_details(root: Path, run_id: str) -> dict[str, Any]:
    scratch_root = job_scratch_dir(run_id)

    def count_running(dataset: str, subdir: str, pattern: str) -> int:
        base = scratch_root / "embodiedbench" / "running" / dataset
        return len(list(base.glob(f"*{run_id}*/*/{subdir}/{pattern}"))) if base.exists() else 0

    def count_mirrored(dataset_name: str, subdir: str, pattern: str) -> int:
        base = root / "full" / dataset_name / subdir
        return len(list(base.glob(f"*/{pattern}"))) if base.exists() else 0

    return {
        "EB-ALFRED": {
            "running_episode_results": count_running("eb_alfred", "results", "episode_*_final_res.json"),
            "running_model_output_files": count_running("eb_alfred", "model_outputs", "episode_*_model_outputs.jsonl"),
            "mirrored_episode_results": count_mirrored("EB-ALFRED", "episode_results", "episode_*_final_res.json"),
            "mirrored_model_output_files": count_mirrored("EB-ALFRED", "model_outputs", "episode_*_model_outputs.jsonl"),
        },
        "EB-Habitat": {
            "running_episode_results": count_running("eb_habitat", "results", "episode_*_final_res.json"),
            "running_model_output_files": count_running("eb_habitat", "model_outputs", "episode_*_model_outputs.jsonl"),
            "mirrored_episode_results": count_mirrored("EB-Habitat", "episode_results", "episode_*_final_res.json"),
            "mirrored_model_output_files": count_mirrored("EB-Habitat", "model_outputs", "episode_*_model_outputs.jsonl"),
        },
    }


def eb_job_eval_set(job: dict[str, Any]) -> str:
    name = str(job.get("name", ""))
    match = re.match(r"(.+)_shard\d+$", name)
    return match.group(1) if match else name


def eb_expected_episodes(job: dict[str, Any]) -> int | None:
    command = str(job.get("command", ""))
    if job.get("kind") == "alfred":
        match = re.search(r"\+selected_indexes='([^']*)'", command)
        if not match:
            match = re.search(r'\+selected_indexes="([^"]*)"', command)
        if match:
            try:
                selected = ast.literal_eval(match.group(1))
            except Exception:
                selected = []
            if isinstance(selected, list):
                return len(selected)
    if job.get("kind") == "habitat":
        match = re.search(r"\+max_episodes=(\d+)", command)
        if match:
            return int(match.group(1))
    return None


def eb_job_run_dir(job: dict[str, Any], run_id: str) -> Path | None:
    configured = job.get("run_dir")
    if configured:
        return Path(str(configured))
    kind = job.get("kind")
    name = str(job.get("name", ""))
    eval_set = eb_job_eval_set(job)
    if kind == "alfred":
        return (
            job_output_dir(run_id)
            / "runtime"
            / "embodiedbench"
            / "running"
            / "eb_alfred"
            / f"qwen35-vl_{run_id}_full_alfred_{name}"
            / eval_set
        )
    if kind == "habitat":
        return (
            job_output_dir(run_id)
            / "runtime"
            / "embodiedbench"
            / "running"
            / "eb_habitat"
            / f"qwen35-vl_{run_id}_full_habitat_{name}"
            / eval_set
        )
    return None


def eb_episode_progress(root: Path, run_id: str, manifest: list[dict[str, Any]], progress: dict[str, Any]) -> dict[str, Any]:
    total_expected = 0
    known_expected_jobs = 0
    result_count = 0
    model_output_count = 0
    by_kind: dict[str, dict[str, int]] = {}
    per_running_job: list[dict[str, Any]] = []

    for job in manifest:
        kind = str(job.get("kind", "unknown"))
        if kind not in {"alfred", "habitat"}:
            continue
        expected = eb_expected_episodes(job)
        if expected is not None:
            total_expected += expected
            known_expected_jobs += 1
        run_dir = eb_job_run_dir(job, run_id)
        results = 0
        model_outputs = 0
        if run_dir and run_dir.exists():
            results = len(list((run_dir / "results").glob("episode_*_final_res.json")))
            model_outputs = len(list((run_dir / "model_outputs").glob("episode_*_model_outputs.jsonl")))
        result_count += results
        model_output_count += model_outputs
        item = by_kind.setdefault(kind, {"expected_episodes": 0, "episode_results": 0, "model_output_files": 0})
        if expected is not None:
            item["expected_episodes"] += expected
        item["episode_results"] += results
        item["model_output_files"] += model_outputs

        status_path = Path(str(job["status"])).name
        status = load_json(root / "status" / status_path) if (root / "status" / status_path).exists() else {}
        if status.get("status") == "running":
            per_running_job.append(
                {
                    "status_file": status_path,
                    "kind": kind,
                    "eval_set": eb_job_eval_set(job),
                    "episode_results": results,
                    "model_output_files": model_outputs,
                    "expected_episodes": expected,
                }
            )

    wall_elapsed = progress.get("wall_elapsed_sec")
    throughput = (result_count / wall_elapsed * 3600) if wall_elapsed and wall_elapsed > 0 else None
    remaining = max(total_expected - result_count, 0) if total_expected else None
    eta = (remaining / throughput * 3600) if throughput and remaining is not None else None
    progress_ratio = result_count / total_expected if total_expected else 0.0

    return {
        "expected_episodes": total_expected,
        "known_expected_jobs": known_expected_jobs,
        "episode_results": result_count,
        "model_output_files": model_output_count,
        "progress_ratio": round(progress_ratio, 6),
        "progress_percent": round(progress_ratio * 100, 2),
        "throughput_episodes_per_hour": round(throughput, 3) if throughput else None,
        "rough_eta_sec": round(eta, 3) if eta else None,
        "by_kind": by_kind,
        "running_jobs": sorted(per_running_job, key=lambda item: item["status_file"]),
    }


def task_error_resolved(error_path: Path) -> bool:
    match = re.search(r"task_(\d+)(?:_score)?_error\.json$", error_path.name)
    if not match:
        return False
    task_id = int(match.group(1))
    summary_path = error_path.parent / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = load_json(summary_path)
    except Exception:
        return False
    for row in summary.get("results", []):
        if row.get("task_id") == task_id:
            score_path = row.get("score_summary_file")
            return not row.get("error") and bool(score_path) and Path(score_path).exists()
    return False


def probe_vllm(api_base: str) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/models"
    try:
        subprocess.run(["curl", "-s", "-m", "5", url], check=True, stdout=subprocess.DEVNULL)
        ok = True
    except subprocess.CalledProcessError:
        ok = False
    gpu: dict[str, Any] = {}
    try:
        out = subprocess.check_output(["rocm-smi", "--showuse", "--showmemuse", "--json"], text=True, errors="ignore")
        gpu = json.loads(out)
    except Exception:
        gpu = {}
    return {"api_ok": ok, "api_url": url, "rocm_smi": gpu}


def build_retry_plan(
    root: Path,
    manifest: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    *,
    include_et_task_errors: bool = True,
) -> dict[str, Any]:
    failed_jobs: list[dict[str, Any]] = []
    for job in manifest:
        status = statuses.get(status_name_for_job(job))
        if status is None:
            retry_reason = "pending"
        elif status.get("status") == "failed":
            retry_reason = "failed"
        elif status.get("status") == "running" and not status_process_is_active(status):
            retry_reason = "stale_running"
        else:
            retry_reason = None
        if retry_reason:
            failed_jobs.append(
                {
                    "kind": job.get("kind"),
                    "name": job.get("name"),
                    "status_file": status_name_for_job(job),
                    "command": job.get("command"),
                    "log": job.get("log"),
                    "retry_reason": retry_reason,
                }
            )

    et_task_errors = []
    if include_et_task_errors:
        strict_error_root = root / "strict_rescore" / "ET-Plan_Bench"
        error_root = strict_error_root if strict_error_root.exists() else root / "full" / "ET-Plan_Bench"
        for path in sorted(error_root.glob("*/*/task_*_error.json")):
            if task_error_resolved(path):
                continue
            try:
                payload = load_json(path)
            except Exception:
                payload = {}
            et_task_errors.append(
                {
                    "error_file": str(path.relative_to(root)),
                    "task_id": payload.get("task_id"),
                    "error_type": payload.get("error_type"),
                    "returncode": payload.get("returncode"),
                }
            )

    return {"failed_jobs": failed_jobs, "et_task_errors": et_task_errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", dest="run_id", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:29199/v1")
    parser.add_argument("--probe-vllm", action="store_true")
    parser.add_argument("--skip-et-task-errors", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_id = load_run_id(args)
    root = job_output_dir(run_id)
    manifest = load_manifest(root)
    statuses = load_statuses(root)
    progress = classify_statuses(manifest, statuses)
    report = {
        "job_name": run_id,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": str(root),
        "progress": progress,
        "eb_episode_progress": eb_episode_progress(root, run_id, manifest, progress),
        "et": {} if args.skip_et_task_errors else et_task_details(root),
        "eb_artifacts": eb_artifact_details(root, run_id),
        "root_artifacts": root_artifact_counts(root),
        "retry_plan": build_retry_plan(
            root,
            manifest,
            statuses,
            include_et_task_errors=not args.skip_et_task_errors,
        ),
    }
    if args.probe_vllm:
        report["vllm"] = probe_vllm(args.api_base)

    if args.write:
        progress_dir = root / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        (progress_dir / "latest_status.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (progress_dir / "final_status.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (progress_dir / "retry_plan.json").write_text(
            json.dumps(report["retry_plan"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    progress = report["progress"]
    print(f"run_id: {run_id}")
    print(f"jobs: {progress['by_state']} / total={progress['total_jobs']}")
    print(f"running: {', '.join(progress['running'][:8])}")
    print(f"failed: {len(progress['failed'])}, pending: {len(progress['pending'])}")
    print(
        "wall_elapsed_sec: "
        f"{progress['wall_elapsed_sec']}, progress={progress['progress_percent']}%, "
        f"throughput_jobs_per_hour={progress['throughput_jobs_per_hour']}, "
        f"rough_eta_sec={progress['rough_eta_sec']}"
    )
    print(f"running_details: {progress['running_details'][:8]}")
    print(f"eb_episode_progress: {report['eb_episode_progress']}")
    print(f"eb_artifacts: {report['eb_artifacts']}")
    print(f"root_artifacts: {report['root_artifacts']}")
    print(f"retry_failed_jobs: {len(report['retry_plan']['failed_jobs'])}")
    print(f"retry_et_task_errors: {len(report['retry_plan']['et_task_errors'])}")


if __name__ == "__main__":
    main()
