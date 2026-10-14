#!/usr/bin/env python
"""Run small ET-Plan-Bench samples with qwen and VirtualHome scoring."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

try:
    from task_planning_io import atomic_write_json
except ModuleNotFoundError:  # Imported as scripts.et_plan_navigation_sample_runner in tests.
    from scripts.task_planning_io import atomic_write_json


os.environ["NOTEBOOK_CONTAINER"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(os.environ.get("ET_PLAN_WORK_DIR", str(PROJECT_ROOT))).resolve()
DIRECT_PLAN_PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "experimental" / "et_plan_qwen_predict.py"
OFFICIAL_AGENT_PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "et_plan_official_agent_predict.py"
SCORE_SCRIPT = PROJECT_ROOT / "scripts" / "experimental" / "et_plan_strict_score.py"
OFFICIAL_SCORE_SCRIPT = PROJECT_ROOT / "scripts" / "et_plan_official_score.py"


def run_command(cmd: list[str], cwd: Path, timeout: int, extra_env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env["NOTEBOOK_CONTAINER"] = "1"
    subprocess.run(cmd, cwd=str(cwd), check=True, timeout=timeout, env=env)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def is_valid_json_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def zero_score_result(
    task_id: int,
    prediction_file: Path | None,
    batch_score_file: Path | None,
    error_payload: dict,
) -> dict:
    task = None
    if prediction_file is not None and prediction_file.exists():
        try:
            task = load_json(prediction_file).get("task")
        except Exception:
            task = None
    return {
        "task_id": task_id,
        "task": task,
        "prediction_file": str(prediction_file) if prediction_file is not None and prediction_file.exists() else None,
        "score_summary_file": str(batch_score_file) if batch_score_file is not None and batch_score_file.exists() else None,
        "score_episode_file": None,
        "success_rate": 0,
        "task_success_rate": 0,
        "format_success_rate": 0,
        "action_execution_success_rate": 0,
        "average_steps": 0,
        "average_lcs": 0,
        "average_lcs_ratio": 0,
        "error": str(error_payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--task-type",
        default="navigation",
        choices=["navigation", "nav_manip_single", "nav_manip_multi"],
        help="Supported ET-Plan task family for the VirtualHome scorer.",
    )
    parser.add_argument(
        "--task-ids",
        default="0",
        help="Comma-separated task ids, for example: 0,4,7.",
    )
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument(
        "--planner-mode",
        choices=["official-agent", "direct-plan"],
        default=os.environ.get("ET_PLAN_PLANNER_MODE", "official-agent"),
        help=(
            "official-agent calls the upstream ET-Plan-Bench exploration logic; "
            "direct-plan is the project-local experimental one-shot DSL generator."
        ),
    )
    parser.add_argument("--port-base", type=int, default=2600)
    parser.add_argument("--command-timeout", type=int, default=900)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Job-scoped output directory under outputs/<job_name>.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing prediction and score files when present.",
    )
    parser.add_argument(
        "--task-retries",
        type=int,
        default=int(os.environ.get("ET_PLAN_TASK_RETRIES", "2")),
        help="Retry each failed task this many times before recording an error.",
    )
    parser.add_argument(
        "--retry-port-stride",
        type=int,
        default=int(os.environ.get("ET_PLAN_RETRY_PORT_STRIDE", "1000")),
        help="Port offset added on each task retry to avoid recently occupied Unity ports.",
    )
    parser.add_argument(
        "--retry-sleep-sec",
        type=float,
        default=float(os.environ.get("ET_PLAN_RETRY_SLEEP_SEC", "3")),
        help="Seconds to wait before retrying a failed task.",
    )
    parser.add_argument(
        "--context-cache-dir",
        help="Directory for cached VirtualHome scene graphs. Defaults to OUTPUT_DIR/context_cache.",
    )
    parser.add_argument(
        "--context-mode",
        choices=["none", "env"],
        default=os.environ.get("ET_PLAN_CONTEXT_MODE", "none"),
        help="Only used by --planner-mode direct-plan; official-agent gets observations through the official exploration loop.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["fast-no-image", "official-step-ca"],
        default=os.environ.get("ET_PLAN_SCORE_EXECUTION_MODE", "fast-no-image"),
        help="Execution mode passed to experimental/et_plan_strict_score.py for --planner-mode direct-plan.",
    )
    parser.add_argument(
        "--disable-put-fallback",
        action="store_true",
        help="Pass through to experimental/et_plan_strict_score.py.",
    )
    args = parser.parse_args()

    benchmark = Path(args.benchmark).resolve()
    output_dir = Path(args.output_dir).resolve()
    predictions_dir = output_dir / "predictions"
    scores_dir = output_dir / "scores"
    context_cache_dir = Path(args.context_cache_dir).resolve() if args.context_cache_dir else output_dir / "context_cache"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    context_cache_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [int(item.strip()) for item in args.task_ids.split(",") if item.strip()]
    result_by_task: dict[int, dict] = {}
    prediction_by_task: dict[int, Path] = {}

    for index, task_id in enumerate(task_ids):
        prediction_dir = predictions_dir / f"task_{task_id}"
        prediction_file = prediction_dir / f"prediction_task{task_id}.json"
        error_file = output_dir / f"task_{task_id}_error.json"

        last_exc = None
        for attempt in range(args.task_retries + 1):
            context_port = args.port_base + index * 2 + attempt * args.retry_port_stride
            if attempt > 0:
                time.sleep(args.retry_sleep_sec)
                if prediction_file.exists():
                    prediction_file.unlink()
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "retry_attempt": attempt,
                            "context_port": context_port,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            try:
                if not (args.reuse_existing and is_valid_json_file(prediction_file)):
                    if args.planner_mode == "official-agent":
                        predict_cmd = [
                            sys.executable,
                            str(OFFICIAL_AGENT_PREDICT_SCRIPT),
                            "--benchmark",
                            str(benchmark),
                            "--task-id",
                            str(task_id),
                            "--task-type",
                            args.task_type,
                            "--model",
                            args.model,
                            "--port",
                            str(context_port),
                            "--output-dir",
                            str(prediction_dir),
                        ]
                    else:
                        predict_cmd = [
                            sys.executable,
                            str(DIRECT_PLAN_PREDICT_SCRIPT),
                            "--benchmark",
                            str(benchmark),
                            "--task-id",
                            str(task_id),
                            "--model",
                            args.model,
                            "--output-dir",
                            str(prediction_dir),
                        ]
                    if args.planner_mode == "direct-plan" and args.context_mode == "env":
                        predict_cmd.extend(
                            [
                                "--context-from-env",
                                "--context-port",
                                str(context_port),
                                "--context-cache-dir",
                                str(context_cache_dir),
                            ]
                        )
                    run_command(
                        predict_cmd,
                        WORK_DIR,
                        args.command_timeout,
                    )
                if not prediction_file.exists():
                    raise FileNotFoundError(prediction_file)
                last_exc = None
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
                last_exc = exc
                if attempt >= args.task_retries:
                    break

        try:
            if last_exc is not None:
                raise last_exc
            if not prediction_file.exists():
                raise FileNotFoundError(prediction_file)
            prediction_by_task[task_id] = prediction_file
            error_file.unlink(missing_ok=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            error_payload = {
                "task_id": task_id,
                "error_type": type(exc).__name__,
                "returncode": getattr(exc, "returncode", None),
                "command": getattr(exc, "cmd", None),
            }
            save_json(error_file, error_payload)
            result_by_task[task_id] = {
                "task_id": task_id,
                "task": None,
                "prediction_file": str(prediction_file) if prediction_file.exists() else None,
                "score_summary_file": None,
                "score_episode_file": None,
                "success_rate": 0,
                "task_success_rate": 0,
                "format_success_rate": 0,
                "action_execution_success_rate": 0,
                "average_steps": 0,
                "average_lcs": 0,
                "average_lcs_ratio": 0,
                "error": str(error_payload),
            }

    score_task_ids = [task_id for task_id in task_ids if task_id in prediction_by_task]
    batch_score_dir = scores_dir / "batch"
    batch_score_file = batch_score_dir / "summary.json"
    if score_task_ids:
        reusable_batch_score = is_valid_json_file(batch_score_file) and all(
            is_valid_json_file(batch_score_dir / f"episode_{task_id}.json") for task_id in score_task_ids
        )
        last_exc = None
        for attempt in range(args.task_retries + 1):
            score_port = args.port_base + len(task_ids) * 2 + attempt * args.retry_port_stride
            if attempt > 0:
                time.sleep(args.retry_sleep_sec)
                print(
                    json.dumps(
                        {
                            "score_retry_attempt": attempt,
                            "score_port": score_port,
                            "task_ids": score_task_ids,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            try:
                if not (args.reuse_existing and reusable_batch_score):
                    batch_score_file.unlink(missing_ok=True)
                    if args.planner_mode == "official-agent":
                        score_cmd = [
                            sys.executable,
                            str(OFFICIAL_SCORE_SCRIPT),
                            "--benchmark",
                            str(benchmark),
                            "--task-type",
                            args.task_type,
                            "--task-ids",
                            ",".join(str(task_id) for task_id in score_task_ids),
                            "--prediction-root",
                            str(predictions_dir),
                            "--output-dir",
                            str(batch_score_dir),
                        ]
                    else:
                        score_cmd = [
                            sys.executable,
                            str(SCORE_SCRIPT),
                            "--benchmark",
                            str(benchmark),
                            "--task-type",
                            args.task_type,
                            "--task-ids",
                            ",".join(str(task_id) for task_id in score_task_ids),
                            "--prediction-root",
                            str(predictions_dir),
                            "--port",
                            str(score_port),
                            "--output-dir",
                            str(batch_score_dir),
                            "--execution-mode",
                            args.execution_mode,
                        ]
                        if args.disable_put_fallback:
                            score_cmd.append("--disable-put-fallback")
                    run_command(score_cmd, WORK_DIR, args.command_timeout)
                last_exc = None
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_exc = exc
                if attempt >= args.task_retries:
                    break

        if last_exc is None:
            for task_id in score_task_ids:
                prediction_file = prediction_by_task[task_id]
                episode_file = batch_score_dir / f"episode_{task_id}.json"
                if not episode_file.exists():
                    error_payload = {
                        "task_id": task_id,
                        "error_type": "MissingScoreEpisode",
                        "score_episode_file": str(episode_file),
                        "score_summary_file": str(batch_score_file) if batch_score_file.exists() else None,
                    }
                    save_json(output_dir / f"task_{task_id}_score_error.json", error_payload)
                    result_by_task[task_id] = zero_score_result(task_id, prediction_file, batch_score_file, error_payload)
                    continue
                try:
                    prediction = load_json(prediction_file)
                    episode = load_json(episode_file)
                except Exception as exc:
                    error_payload = {
                        "task_id": task_id,
                        "error_type": type(exc).__name__,
                        "score_episode_file": str(episode_file),
                        "score_summary_file": str(batch_score_file) if batch_score_file.exists() else None,
                    }
                    save_json(output_dir / f"task_{task_id}_score_error.json", error_payload)
                    result_by_task[task_id] = zero_score_result(task_id, prediction_file, batch_score_file, error_payload)
                    continue
                result_by_task[task_id] = {
                    "task_id": task_id,
                    "task": prediction.get("task"),
                    "prediction_file": str(prediction_file),
                    "score_summary_file": str(batch_score_file),
                    "score_episode_file": str(episode_file),
                    "success_rate": 1.0 if episode["task_success"] else 0.0,
                    "task_success_rate": 1.0 if episode["task_success"] else 0.0,
                    "format_success_rate": 1.0 if episode["format_success"] else 0.0,
                    "action_execution_success_rate": None
                    if episode.get("action_execution_success") is None
                    else 1.0
                    if episode["action_execution_success"]
                    else 0.0,
                    "average_steps": episode["num_steps"],
                    "average_lcs": episode["lcs"],
                    "average_lcs_ratio": episode["lcs_ratio"],
                    "error": None,
                }
                (output_dir / f"task_{task_id}_error.json").unlink(missing_ok=True)
                (output_dir / f"task_{task_id}_score_error.json").unlink(missing_ok=True)
        else:
            for task_id in score_task_ids:
                error_payload = {
                    "task_id": task_id,
                    "error_type": type(last_exc).__name__,
                    "returncode": getattr(last_exc, "returncode", None),
                    "command": getattr(last_exc, "cmd", None),
                }
                save_json(output_dir / f"task_{task_id}_score_error.json", error_payload)
                prediction_file = prediction_by_task[task_id]
                result_by_task[task_id] = zero_score_result(task_id, prediction_file, batch_score_file, error_payload)

    results = [result_by_task[task_id] for task_id in task_ids]

    aggregate = {
        "benchmark": str(benchmark),
        "task_type": args.task_type,
        "model": args.model,
        "planner_mode": args.planner_mode,
        "task_count": len(results),
        "task_ids": task_ids,
        "context_mode": args.context_mode,
        "score_execution_mode": "official-output" if args.planner_mode == "official-agent" else args.execution_mode,
        "score_summary_file": str(batch_score_file) if batch_score_file.exists() else None,
        "success_rate": mean(item["success_rate"] for item in results) if results else 0,
        "task_success_rate": mean(item.get("task_success_rate", item["success_rate"]) for item in results) if results else 0,
        "format_success_rate": mean(item.get("format_success_rate") or 0 for item in results) if results else 0,
        "action_execution_success_rate": (
            mean(
                item["action_execution_success_rate"]
                for item in results
                if item.get("action_execution_success_rate") is not None
            )
            if any(item.get("action_execution_success_rate") is not None for item in results)
            else None
        ),
        "average_steps": mean(item["average_steps"] for item in results) if results else 0,
        "average_lcs": mean(item["average_lcs"] for item in results) if results else 0,
        "average_lcs_ratio": mean(item["average_lcs_ratio"] for item in results) if results else 0,
        "results": results,
    }

    aggregate_path = output_dir / "summary.json"
    atomic_write_json(aggregate_path, aggregate)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
