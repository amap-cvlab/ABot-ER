from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LLM_JUDGE_TASKS = {"unieqa", "openeqa-emeqa", "robovqa"}


@dataclass(frozen=True)
class TaskArtifactStatus:
    complete: bool
    reason: str
    expected_samples: int
    actual_samples: int | None = None
    saw_any_artifact: bool = False


def fast_task_dir_name(task_name: str) -> str:
    return task_name.removesuffix("_aboter")


def expected_sample_count(task: Any, limit_value: Any) -> int:
    total_docs = len(task.eval_docs)
    limit = parse_limit(limit_value, total_docs)
    if limit is None:
        return total_docs
    return min(max(limit, 0), total_docs)


def parse_limit(limit_value: Any, total_docs: int) -> int | None:
    if limit_value is None:
        return None
    try:
        limit = float(limit_value)
    except (TypeError, ValueError):
        return None

    if limit < 1:
        return int(total_docs * limit)
    return int(limit)


def task_artifact_status(
    output_path: Path,
    task_name: str,
    llm_judge: str,
    expected_samples: int,
) -> TaskArtifactStatus:
    task_dir = output_path / fast_task_dir_name(task_name)
    samples_path = task_dir / f"samples_{task_name}.json"
    results_path = task_dir / f"results_{task_name}.json"
    configs_path = task_dir / f"configs_{task_name}.json"
    saw_any = any(path.exists() for path in (samples_path, results_path, configs_path))

    samples_ok, actual_samples, samples_reason = validate_samples_file(samples_path, expected_samples)
    if not samples_ok:
        return TaskArtifactStatus(
            complete=False,
            reason=samples_reason,
            expected_samples=expected_samples,
            actual_samples=actual_samples,
            saw_any_artifact=saw_any,
        )

    if llm_judge == "no" and task_name in LLM_JUDGE_TASKS:
        return TaskArtifactStatus(
            complete=True,
            reason=f"samples complete ({actual_samples}/{expected_samples}); llm_judge=no",
            expected_samples=expected_samples,
            actual_samples=actual_samples,
            saw_any_artifact=True,
        )

    for path, label in ((results_path, "results"), (configs_path, "configs")):
        ok, reason = validate_json_object_file(path, label)
        if not ok:
            return TaskArtifactStatus(
                complete=False,
                reason=reason,
                expected_samples=expected_samples,
                actual_samples=actual_samples,
                saw_any_artifact=saw_any,
            )

    return TaskArtifactStatus(
        complete=True,
        reason=f"samples/results/configs complete ({actual_samples}/{expected_samples})",
        expected_samples=expected_samples,
        actual_samples=actual_samples,
        saw_any_artifact=True,
    )


def validate_samples_file(path: Path, expected_samples: int) -> tuple[bool, int | None, str]:
    if not path.is_file():
        return False, None, f"missing samples file: {path}"

    actual_samples = 0
    doc_ids = set()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    return False, actual_samples, f"blank line in samples file: {path}:{line_no}"
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    return False, actual_samples, f"invalid JSONL samples file: {path}:{line_no} ({exc})"
                if not isinstance(sample, dict):
                    return False, actual_samples, f"non-object sample in samples file: {path}:{line_no}"
                if "doc_id" not in sample:
                    return False, actual_samples, f"sample missing doc_id: {path}:{line_no}"
                doc_id_key = json.dumps(sample["doc_id"], sort_keys=True, default=str, ensure_ascii=False)
                if doc_id_key in doc_ids:
                    return False, actual_samples, f"duplicate doc_id in samples file: {path}:{line_no}"
                doc_ids.add(doc_id_key)
                actual_samples += 1
    except OSError as exc:
        return False, None, f"cannot read samples file: {path} ({exc})"

    if actual_samples != expected_samples:
        return (
            False,
            actual_samples,
            f"incomplete samples file: {path} ({actual_samples}/{expected_samples})",
        )
    return True, actual_samples, "samples complete"


def validate_json_object_file(path: Path, label: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {label} file: {path}"
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid {label} file: {path} ({exc})"
    if not isinstance(value, dict):
        return False, f"{label} file is not a JSON object: {path}"
    if not value:
        return False, f"{label} file is empty: {path}"
    return True, f"{label} complete"
