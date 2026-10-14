#!/usr/bin/env python
"""One-command local TP full-evaluation pipeline around an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from task_planning_config import (
        DEFAULT_ALFRED_SHARD_TIMEOUT,
        DEFAULT_ET_COMMAND_TIMEOUT,
        DEFAULT_HABITAT_SHARD_TIMEOUT,
        job_output_dir,
        job_scratch_dir,
    )
except ModuleNotFoundError:  # Imported as scripts.run_tp_full_eval_pipeline in tests.
    from scripts.task_planning_config import (
        DEFAULT_ALFRED_SHARD_TIMEOUT,
        DEFAULT_ET_COMMAND_TIMEOUT,
        DEFAULT_HABITAT_SHARD_TIMEOUT,
        job_output_dir,
        job_scratch_dir,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], log_path: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path else None
    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    line = "+ " + " ".join(cmd) + "\n"
    if log_path:
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n# started_at={started}\n{line}")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT)
            return proc.wait()
    print(line, end="", flush=True)
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def retry_kinds_for_suite(suite: str) -> list[str]:
    if suite == "et":
        return ["et"]
    if suite == "alfred":
        return ["alfred"]
    if suite == "habitat":
        return ["habitat"]
    if suite == "eb":
        return ["alfred", "habitat"]
    return ["et", "alfred", "habitat"]


def retry_kind_args(suite: str) -> list[str]:
    args: list[str] = []
    for kind in retry_kinds_for_suite(suite):
        args.extend(["--kind", kind])
    return args


def retry_jobs(run_id: str, suite: str) -> int:
    out = subprocess.check_output(
        [sys.executable, "scripts/full_eval_retry.py", "--job-name", run_id, "--dry-run", *retry_kind_args(suite)],
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    payload = json.loads(out)
    return int(payload.get("retry_jobs", 0))


def status_counts(run_id: str, suite: str) -> dict:
    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/full_eval_status.py",
            "--job-name",
            run_id,
            "--json",
            *(["--skip-et-task-errors"] if "et" not in retry_kinds_for_suite(suite) else []),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    payload = json.loads(out)
    return {
        "progress": payload.get("progress", {}),
        "failed_jobs": len(payload.get("retry_plan", {}).get("failed_jobs", [])),
        "et_task_errors": len(payload.get("retry_plan", {}).get("et_task_errors", [])),
    }


def prune_completed_job(root: Path) -> dict[str, object]:
    """Remove disposable run state after final artifacts have been generated."""
    removed: dict[str, object] = {"directories": 0, "files": 0, "skipped": 0, "errors": []}

    def record_error(path: Path, exc: OSError) -> None:
        removed["skipped"] = int(removed["skipped"]) + 1
        errors = removed["errors"]
        assert isinstance(errors, list)
        errors.append({"path": str(path.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"})
        print(f"WARNING: could not prune {path}: {exc}", file=sys.stderr, flush=True)

    for name in ("runtime", "runtime_artifacts", "retry", "recovery", "logs", "progress", "status"):
        path = root / name
        if path.exists():
            try:
                shutil.rmtree(path)
            except OSError as exc:
                record_error(path, exc)
            else:
                removed["directories"] = int(removed["directories"]) + 1
    for pattern in ("manifest_*.json", "eval_config_*.json"):
        for path in (root / "full").glob(pattern):
            try:
                path.unlink()
            except OSError as exc:
                record_error(path, exc)
            else:
                removed["files"] = int(removed["files"]) + 1
    manifest_path = root / "full" / "artifacts_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_contract"] = {
            "storage": "one outputs/<job_name> directory",
            "images": "not persisted",
            "logs": "Worker stdout; retain logs through your process launcher",
            "runtime": "node-local scratch removed after completion",
        }
        manifest.pop("configs", None)
        manifest.pop("manifests", None)
        manifest.pop("logs", None)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = root / "full" / "final_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        artifacts = summary.get("artifacts")
        if isinstance(artifacts, dict):
            artifacts.pop("readme", None)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--suite", choices=["all", "et", "eb", "alfred", "habitat"], default="all")
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--api-base", default="http://127.0.0.1:29199/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-workers", type=int, default=3)
    parser.add_argument("--max-retry-rounds", type=int, default=3)
    parser.add_argument("--et-shard-size", type=int, default=30)
    parser.add_argument("--et-task-retries", type=int, default=2)
    parser.add_argument("--et-port-base", type=int, default=31000)
    parser.add_argument("--et-command-timeout", type=int, default=DEFAULT_ET_COMMAND_TIMEOUT)
    parser.add_argument("--eb-shard-size", type=int, default=20)
    parser.add_argument("--eb-shards-per-dataset", type=int, default=80)
    parser.add_argument("--display-base", type=int, default=130)
    parser.add_argument("--alfred-timeout", type=int, default=DEFAULT_ALFRED_SHARD_TIMEOUT)
    parser.add_argument("--habitat-timeout", type=int, default=DEFAULT_HABITAT_SHARD_TIMEOUT)
    parser.add_argument("--eb-max-completion-tokens", type=int, default=512)
    parser.add_argument("--alfred-resolution", type=int, default=300)
    parser.add_argument("--habitat-resolution", type=int, default=256)
    parser.add_argument("--run-et-strict-rescore", action="store_true")
    parser.add_argument("--et-rescore-port-base", type=int, default=43000)
    parser.add_argument("--et-rescore-timeout", type=int, default=900)
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--probe-vllm", action="store_true")
    parser.add_argument("--prune-artifacts", action="store_true",
                        help="Opt-in removal of this run's temporary artifacts; disabled for reproducibility")
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Launch only the first N shard jobs. Intended for smoke/probe runs; unset for official full runs.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip checkpoint error analysis for intermediate phase runs; final phase should keep it enabled.",
    )
    args = parser.parse_args()

    run_id = args.job_name
    root = job_output_dir(run_id)
    if not args.skip_launch:
        rc = run(
            [
                sys.executable,
                "scripts/run_full_eval_local.py",
                "launch",
                "--job-name",
                run_id,
                "--suite",
                args.suite,
                "--model",
                args.model,
                "--api-base",
                args.api_base,
                "--api-key",
                args.api_key,
                "--workers",
                str(args.workers),
                "--et-shard-size",
                str(args.et_shard_size),
                "--et-task-retries",
                str(args.et_task_retries),
                "--et-port-base",
                str(args.et_port_base),
                "--et-command-timeout",
                str(args.et_command_timeout),
                "--eb-shard-size",
                str(args.eb_shard_size),
                "--eb-shards-per-dataset",
                str(args.eb_shards_per_dataset),
                "--display-base",
                str(args.display_base),
                "--alfred-timeout",
                str(args.alfred_timeout),
                "--habitat-timeout",
                str(args.habitat_timeout),
                "--eb-max-completion-tokens",
                str(args.eb_max_completion_tokens),
                "--alfred-resolution",
                str(args.alfred_resolution),
                "--habitat-resolution",
                str(args.habitat_resolution),
                "--no-aggregate",
                *(["--max-jobs", str(args.max_jobs)] if args.max_jobs is not None else []),
            ],
        )
        if rc != 0:
            raise SystemExit(rc)

    for round_idx in range(1, args.max_retry_rounds + 1):
        jobs = retry_jobs(run_id, args.suite)
        print(json.dumps({"round": round_idx, "retry_jobs": jobs, "status": status_counts(run_id, args.suite)}, ensure_ascii=False))
        if jobs <= 0:
            break
        rc = run(
            [
                sys.executable,
                "scripts/full_eval_retry.py",
                "--job-name",
                run_id,
                "--workers",
                str(args.retry_workers),
                "--alfred-timeout",
                str(args.alfred_timeout),
                "--habitat-timeout",
                str(args.habitat_timeout),
                *retry_kind_args(args.suite),
            ],
        )
        if rc != 0:
            raise SystemExit(rc)

    remaining_retry_jobs = retry_jobs(run_id, args.suite)
    if remaining_retry_jobs:
        raise RuntimeError(
            f"evaluation still has {remaining_retry_jobs} retryable shard(s) after "
            f"{args.max_retry_rounds} retry round(s); run_id={run_id}"
        )

    if args.suite in {"all", "et"} and args.run_et_strict_rescore:
        rc = run(
            [
                sys.executable,
                "scripts/experimental/rescore_et_predictions.py",
                "--source-run",
                str(root),
                "--port-base",
                str(args.et_rescore_port_base),
                "--timeout",
                str(args.et_rescore_timeout),
            ],
        )
        if rc != 0:
            raise SystemExit(rc)

    rc = run(
        [
            sys.executable,
            "scripts/full_eval_status.py",
            "--job-name",
            run_id,
            "--write",
            "--json",
            *(["--skip-et-task-errors"] if "et" not in retry_kinds_for_suite(args.suite) else []),
            *(["--probe-vllm"] if args.probe_vllm else []),
        ],
    )
    if rc != 0:
        raise SystemExit(rc)
    finalize_cmd = [
        sys.executable,
        "scripts/finalize_full_eval.py",
        "--job-name",
        run_id,
        "--model",
        args.model,
        "--skip-full-eval-status",
        "--skip-runtime-mirror",
    ]
    if args.skip_analysis:
        finalize_cmd.append("--skip-model-output-mirror")
    rc = run(finalize_cmd)
    if rc != 0:
        raise SystemExit(rc)
    if not args.skip_analysis:
        rc = run(
            [sys.executable, "scripts/analyze_checkpoint_errors.py", "--job-name", run_id, "--model", args.model],
        )
        if rc != 0:
            raise SystemExit(rc)
    if args.prune_artifacts and args.max_jobs is None and not args.skip_analysis:
        pruned = prune_completed_job(root)
    else:
        pruned = {"directories": 0}
    if args.prune_artifacts and not args.skip_analysis:
        shutil.rmtree(job_scratch_dir(run_id), ignore_errors=True)
    print(
        json.dumps(
            {"job_name": run_id, "final": str(root / "full" / "final_summary.json"), "pruned": pruned},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
