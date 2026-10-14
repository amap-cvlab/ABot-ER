#!/usr/bin/env python
"""Summarize ET-Plan-Bench official-agent outputs without re-executing plans."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from task_planning_io import atomic_write_json
except ModuleNotFoundError:  # Imported as scripts.et_plan_official_score in tests.
    from scripts.task_planning_io import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ET_ROOT = PROJECT_ROOT / "ET-Plan-Bench"
OFFICIAL_EVAL_UTILS = (
    ET_ROOT
    / "evaluation_generation"
    / "navigation_manipulation_tasks"
    / "EmbodiedAI"
    / "utils"
    / "eval.py"
)
VH_ACTION_RE = re.compile(
    r"^<char\d+>\s+\[([A-Za-z_][A-Za-z0-9_]*)\](?:\s+<[^<>]+>\s+\(\d+\))*$"
)


def load_official_eval_helpers():
    spec = importlib.util.spec_from_file_location("_et_plan_official_eval_utils", OFFICIAL_EVAL_UTILS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load official eval helper: {OFFICIAL_EVAL_UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_action_object_list, module.lcs


OFFICIAL_GET_ACTION_OBJECT_LIST, OFFICIAL_LCS = load_official_eval_helpers()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_task_ids(task_id: int, task_ids: str | None) -> list[int]:
    if not task_ids:
        return [task_id]
    parsed = [int(item.strip()) for item in task_ids.split(",") if item.strip()]
    return parsed or [task_id]


def prediction_path_for_task(prediction_root: Path, task_id: int) -> Path | None:
    candidates = [
        prediction_root / f"prediction_task{task_id}.json",
        prediction_root / f"task_{task_id}" / f"prediction_task{task_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    recursive = sorted(prediction_root.glob(f"**/prediction_task{task_id}.json"))
    return recursive[0] if recursive else None


def action_signature(action: str) -> str:
    return str(OFFICIAL_GET_ACTION_OBJECT_LIST(action)).strip().lower()


def lcs_len(left: list[str], right: list[str]) -> int:
    return max(len(OFFICIAL_LCS(left, right)) - 1, 0)


def validate_action_history(action_history: list[str]) -> tuple[bool, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    ok = bool(action_history)
    for step_idx, action in enumerate(action_history, start=1):
        action_ok = bool(VH_ACTION_RE.fullmatch(action.strip()))
        if not action_ok:
            ok = False
        results.append(
            {
                "step": step_idx,
                "action": action,
                "valid_format": action_ok,
                "message": None if action_ok else "action is not a VirtualHome script action",
            }
        )
    if not action_history:
        results.append(
            {
                "step": 0,
                "action": None,
                "valid_format": False,
                "message": "official action_history is empty",
            }
        )
    return ok, results


def official_success(prediction: dict[str, Any]) -> bool | None:
    for key in ("official_agent_success", "success", "task_success"):
        if key in prediction and prediction[key] is not None:
            return bool(prediction[key])
    return None


def build_episode(
    *,
    task: dict[str, Any],
    task_id: int,
    prediction_file: Path | None,
    prediction: dict[str, Any] | None,
    invalid_message: str | None,
) -> dict[str, Any]:
    if invalid_message or prediction is None:
        action_history: list[str] = []
        success = False
        format_success = False
        format_results = [
            {
                "step": 0,
                "action": None,
                "valid_format": False,
                "message": invalid_message or "prediction was not loaded",
            }
        ]
        official_raw_output = None
    else:
        action_history = prediction.get("action_history") or prediction.get("predicted_plan") or []
        success_value = official_success(prediction)
        success = bool(success_value)
        format_success, format_results = validate_action_history(action_history)
        official_raw_output = prediction.get("official_raw_output")

    gt_plan = task["ground_truth_plan"]
    gt_sig = [action_signature(action) for action in gt_plan]
    pred_sig = [action_signature(action) for action in action_history]
    lcs_value = lcs_len(gt_sig, pred_sig)

    return {
        "task_id": task_id,
        "env_id": task["env_id"],
        "task": task["task"],
        "task_completion_criterion": task.get("task_completion_criterion"),
        "success": success,
        "task_success": success,
        "format_success": format_success,
        "action_execution_success": None,
        "action_execution_success_note": "not recomputed; official-agent success comes from the upstream exploration run",
        "failed_action": None,
        "num_steps": len(action_history),
        "ground_truth_steps": len(gt_plan),
        "lcs": lcs_value,
        "lcs_ratio": lcs_value / len(gt_plan) if gt_plan else 0,
        "predicted_plan": action_history,
        "ground_truth_plan": gt_plan,
        "action_results": format_results,
        "prediction_file": str(prediction_file) if prediction_file else None,
        "official_raw_output": official_raw_output,
        "scoring_mode": "official-output",
    }


def optional_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def summarize_results(
    *,
    benchmark: Path,
    task_type: str,
    task_ids: list[int],
    results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "benchmark": str(benchmark),
        "task_type": task_type,
        "task_count": len(results),
        "task_ids": task_ids,
        "success_rate": mean(1.0 if row["task_success"] else 0.0 for row in results) if results else 0.0,
        "task_success_rate": mean(1.0 if row["task_success"] else 0.0 for row in results) if results else 0.0,
        "format_success_rate": mean(1.0 if row["format_success"] else 0.0 for row in results) if results else 0.0,
        "action_execution_success_rate": optional_mean(
            [
                None
                if row["action_execution_success"] is None
                else 1.0
                if row["action_execution_success"]
                else 0.0
                for row in results
            ]
        ),
        "average_steps": mean(row["num_steps"] for row in results) if results else 0.0,
        "average_lcs": mean(row["lcs"] for row in results) if results else 0.0,
        "average_lcs_ratio": mean(row["lcs_ratio"] for row in results) if results else 0.0,
        "episode_file": str(output_dir / f"episode_{task_ids[0]}.json") if len(task_ids) == 1 else None,
        "episode_files": [str(output_dir / f"episode_{task_id}.json") for task_id in task_ids],
        "scoring_mode": "official-output",
        "official_metric_note": (
            "success_rate uses upstream official-agent success from VirtualHome exploration; "
            "LCS uses ET-Plan-Bench's official eval.py helper; plans are not re-executed here."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--task-type",
        choices=["navigation", "nav_manip_single", "nav_manip_multi"],
        required=True,
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-ids")
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    benchmark = Path(args.benchmark).resolve()
    prediction_root = Path(args.prediction_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(benchmark)
    task_ids = parse_task_ids(args.task_id, args.task_ids)
    results: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = tasks[task_id]
        prediction_file = prediction_path_for_task(prediction_root, task_id)
        prediction = None
        invalid_message = None
        if prediction_file is None:
            invalid_message = "prediction file was not found"
        else:
            try:
                prediction = load_json(prediction_file)
            except Exception as exc:  # pragma: no cover - defensive around local file IO
                invalid_message = f"failed to read prediction file: {exc}"
        episode = build_episode(
            task=task,
            task_id=task_id,
            prediction_file=prediction_file,
            prediction=prediction,
            invalid_message=invalid_message,
        )
        atomic_write_json(output_dir / f"episode_{task_id}.json", episode)
        results.append(episode)

    summary = summarize_results(
        benchmark=benchmark,
        task_type=args.task_type,
        task_ids=task_ids,
        results=results,
        output_dir=output_dir,
    )
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
