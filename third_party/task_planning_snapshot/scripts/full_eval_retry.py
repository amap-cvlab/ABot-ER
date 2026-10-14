#!/usr/bin/env python
"""Retry failed or incomplete full-evaluation shards with existing outputs reused."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import shlex
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import full_eval_status
from task_planning_config import (
    DEFAULT_ALFRED_SHARD_TIMEOUT,
    DEFAULT_HABITAT_SHARD_TIMEOUT,
    job_output_dir,
    job_scratch_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def purge_exception_files(root: Path) -> int:
    """Delete Unity exception sidecars instead of retaining retry artifacts."""
    deleted = 0
    if not root.exists():
        return deleted
    for path in root.rglob("*.excp"):
        if not path.is_file():
            continue
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted


def eb_job_outputs_complete(job: dict[str, Any]) -> bool:
    """Return whether every selected EB episode already has an official result."""
    if job.get("kind") not in {"alfred", "habitat"}:
        return False
    match = re.search(r"\+selected_indexes=['\"]\[([0-9,]+)\]['\"]", str(job.get("command", "")))
    run_dir = Path(str(job.get("run_dir", "")))
    if not match or not run_dir.is_dir():
        return False
    episode_ids = [int(value) for value in match.group(1).split(",")]
    results_dir = run_dir / "results"
    return bool(episode_ids) and all(
        (results_dir / f"episode_{episode_id + 1}_final_res.json").is_file() for episode_id in episode_ids
    )


def run_shell(command: str, cwd: Path, status_path: Path) -> dict[str, Any]:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status: dict[str, Any] = {
        "command": command,
        "cwd": str(cwd),
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "running",
        "retry": True,
        "hostname": socket.gethostname(),
    }
    print(json.dumps({"message": "retry_shard_start", **status}, ensure_ascii=False), flush=True)
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
            "ended_at": datetime.fromtimestamp(ended, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_sec": round(ended - started, 3),
            "discarded_excp": discarded_excp,
        }
    )
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"message": "retry_shard_done", **status}, ensure_ascii=False), flush=True)
    return status


def build_jobs(run_id: str, kinds: set[str] | None = None) -> list[dict[str, Any]]:
    root = job_output_dir(run_id)
    manifest = full_eval_status.load_manifest(root)
    statuses = full_eval_status.load_statuses(root)
    include_et_task_errors = kinds is None or "et" in kinds
    retry_plan = full_eval_status.build_retry_plan(
        root,
        manifest,
        statuses,
        include_et_task_errors=include_et_task_errors,
    )

    jobs_by_status: dict[str, dict[str, Any]] = {}
    for job in manifest:
        status_name = full_eval_status.status_name_for_job(job)
        status = statuses.get(status_name)
        if status and full_eval_status.status_process_is_active(status):
            continue
        jobs_by_status[status_name] = job

    selected: dict[str, dict[str, Any]] = {}
    for failed in retry_plan["failed_jobs"]:
        status_name = failed["status_file"]
        job = jobs_by_status.get(status_name)
        if job and not eb_job_outputs_complete(job) and (kinds is None or job.get("kind") in kinds):
            selected[status_name] = job

    if kinds is None or "et" in kinds:
        for task_error in retry_plan["et_task_errors"]:
            error_path = root / task_error["error_file"]
            shard_dir = error_path.parent
            for status_name, job in jobs_by_status.items():
                if kinds is not None and job.get("kind") not in kinds:
                    continue
                if str(shard_dir) in str(job.get("command", "")):
                    selected[status_name] = job
                    break

    return [selected[key] for key in sorted(selected)]


def retry_command(
    job: dict[str, Any],
    run_id: str,
    alfred_timeout: int | None = None,
    habitat_timeout: int | None = None,
) -> str:
    command = str(job["command"])
    if job.get("kind") == "et" and "NOTEBOOK_CONTAINER=" not in command:
        command = f"NOTEBOOK_CONTAINER=1 {command}"
    if job.get("kind") == "et" and "ET_PLAN_RUNTIME_DIR=" not in command:
        runtime_dir = job_scratch_dir(run_id) / "et_plan_retry"
        command = f"ET_PLAN_RUNTIME_DIR={shlex.quote(str(runtime_dir))} {command}"
    if job.get("kind") == "et" and "ET_PLAN_WORK_DIR=" not in command:
        work_dir = job_scratch_dir(run_id) / "et_workdir"
        command = f"ET_PLAN_WORK_DIR={shlex.quote(str(work_dir))} {command}"
    if job.get("kind") in {"alfred", "habitat"} and "PYTHONPATH=" not in command:
        embodied_root = PROJECT_ROOT / "EmbodiedBench"
        command = f"export PYTHONPATH={shlex.quote(str(embodied_root))}:\"${{PYTHONPATH:-}}\"; {command}"
    timeout = {"alfred": alfred_timeout, "habitat": habitat_timeout}.get(job.get("kind"))
    if timeout:
        command = re.sub(
            r"\btimeout\s+\d+s\s+((?:/[^ ]+/)?python(?:\d+(?:\.\d+)?)?)\s+-u\s+-m\s+embodiedbench\.main",
            rf"timeout {timeout}s \1 -u -m embodiedbench.main",
            command,
            count=1,
        )
    return command


def retry_cwd(root: Path, job: dict[str, Any]) -> Path:
    if job.get("kind") == "et":
        work_dir = job_scratch_dir(root.name) / "et_workdir"
    else:
        work_dir = job_scratch_dir(root.name) / "embodiedbench"
        package_link = work_dir / "embodiedbench"
        work_dir.mkdir(parents=True, exist_ok=True)
        if not package_link.exists():
            package_link.symlink_to(PROJECT_ROOT / "EmbodiedBench" / "embodiedbench", target_is_directory=True)
        running_dir = work_dir / "running"
        running_dir.mkdir(parents=True, exist_ok=True)
        for link_name, dataset_name in (("eb_alfred", "EB-ALFRED"), ("eb_habitat", "EB-Habitat")):
            target = root / "full" / dataset_name / "runs"
            target.mkdir(parents=True, exist_ok=True)
            link = running_dir / link_name
            if link.is_symlink() and link.resolve() != target.resolve():
                link.unlink()
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
            elif link.resolve() != target.resolve():
                raise RuntimeError(f"EmbodiedBench running directory is not the NAS output target: {link}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def resolve_task_errors(root: Path, job: dict[str, Any]) -> int:
    if job.get("kind") != "et":
        return 0
    command = str(job.get("command", ""))
    marker = "--output-dir "
    if marker not in command:
        return 0
    try:
        parts = shlex.split(command)
    except ValueError:
        return 0
    if "--output-dir" not in parts:
        return 0
    shard_dir = Path(parts[parts.index("--output-dir") + 1])
    moved = 0
    for error_path in shard_dir.glob("task_*_error.json"):
        if full_eval_status.task_error_resolved(error_path):
            error_path.unlink()
            moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", dest="run_id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--alfred-timeout", type=int, default=DEFAULT_ALFRED_SHARD_TIMEOUT)
    parser.add_argument("--habitat-timeout", type=int, default=DEFAULT_HABITAT_SHARD_TIMEOUT)
    parser.add_argument(
        "--kind",
        action="append",
        choices=["et", "alfred", "habitat"],
        help="Restrict retry selection to one shard kind. Can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id
    root = job_output_dir(run_id)
    kinds = set(args.kind) if args.kind else None
    jobs = build_jobs(run_id, kinds)
    plan_payload = [
        {
            "kind": job.get("kind"),
            "name": job.get("name"),
            "status": str(job.get("status")),
            "command": retry_command(job, run_id, args.alfred_timeout, args.habitat_timeout),
        }
        for job in jobs
    ]
    print(
        json.dumps(
            {
                "job_name": run_id,
                "retry_jobs": len(jobs),
                "jobs": plan_payload if args.dry_run else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run or not jobs:
        return

    failed_count = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_to_job = {}
        for job in jobs:
            status_path = Path(str(job["status"]))
            fut = pool.submit(
                run_shell,
                retry_command(job, run_id, args.alfred_timeout, args.habitat_timeout),
                retry_cwd(root, job),
                status_path,
            )
            fut_to_job[fut] = job
        for fut in futures.as_completed(fut_to_job):
            job = fut_to_job[fut]
            result = fut.result()
            if result.get("status") != "success":
                failed_count += 1
            resolved = resolve_task_errors(root, job) if result.get("status") == "success" else 0
            print(json.dumps({"job": job.get("name"), "resolved_task_error_files": resolved, **result}, ensure_ascii=False), flush=True)
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
