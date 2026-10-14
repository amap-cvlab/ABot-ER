#!/usr/bin/env python
"""Merge split Task Planning suite final_summary.json files.

This is a bookkeeping helper for runs where ET-Plan_Bench, EB-ALFRED, and
EB-Habitat were launched as separate jobs. It does not rescore or reinterpret
any benchmark output; it only copies the already-finalized suite score blocks
into one final_summary.json so the existing CSV updater can consume it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task_planning_config import job_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SUITE_KEYS = {
    "et": "ET-Plan_Bench",
    "alfred": "EB-ALFRED",
    "habitat": "EB-Habitat",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_score(summary: dict[str, Any], suite: str, path: Path) -> dict[str, Any]:
    score_key = SUITE_KEYS[suite]
    score = (summary.get("scores") or {}).get(score_key)
    if not isinstance(score, dict):
        raise SystemExit(f"{path} does not contain scores.{score_key}")
    if suite == "et":
        if int(score.get("task_count") or 0) <= 0:
            raise SystemExit(f"{path} has no finalized ET task_count")
    else:
        if int(score.get("episode_count") or 0) <= 0:
            raise SystemExit(f"{path} has no finalized {score_key} episode_count")
    return score


def require_complete(summary: dict[str, Any], path: Path) -> None:
    completeness = ((summary.get("artifacts") or {}).get("completeness") or {})
    if completeness.get("status") != "complete":
        raise SystemExit(
            f"{path} is not complete: status={completeness.get('status')!r}, "
            f"issues={completeness.get('issues', [])!r}"
        )
    residual = summary.get("residual_errors") or {}
    if residual.get("failed_jobs") or residual.get("et_task_errors"):
        raise SystemExit(f"{path} still has residual errors: {residual!r}")


def runtime_for(summary: dict[str, Any], suite: str) -> dict[str, Any]:
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        return {}
    by_kind = runtime.get("by_kind")
    kind_key = "et" if suite == "et" else suite
    if isinstance(by_kind, dict) and isinstance(by_kind.get(kind_key), dict):
        return dict(by_kind[kind_key])
    return dict(runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", dest="run_id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--et-summary", required=True)
    parser.add_argument("--alfred-summary", required=True)
    parser.add_argument("--habitat-summary", required=True)
    args = parser.parse_args()

    inputs = {
        "et": Path(args.et_summary),
        "alfred": Path(args.alfred_summary),
        "habitat": Path(args.habitat_summary),
    }
    summaries = {suite: load_json(path) for suite, path in inputs.items()}
    for suite, summary in summaries.items():
        require_complete(summary, inputs[suite])
    scores = {
        SUITE_KEYS[suite]: require_score(summary, suite, inputs[suite])
        for suite, summary in summaries.items()
    }
    runtime_by_kind = {
        ("et" if suite == "et" else suite): runtime_for(summary, suite)
        for suite, summary in summaries.items()
    }
    shard_success = sum(int(item.get("success") or item.get("shards_success") or 0) for item in runtime_by_kind.values())
    shard_total = sum(int(item.get("shards") or item.get("shards_total") or 0) for item in runtime_by_kind.values())

    payload: dict[str, Any] = {
        "job_name": args.run_id,
        "run_id": args.run_id,
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scores": scores,
        "runtime": {
            "shards_total": shard_total,
            "shards_success": shard_success,
            "shards_failed": max(shard_total - shard_success, 0),
            "by_kind": runtime_by_kind,
        },
        "source_summaries": {suite: rel(path) for suite, path in inputs.items()},
        "artifacts": {
            "completeness": {"status": "complete", "issues": []},
            "source_manifests": {
                suite: rel(path.parent / "artifacts_manifest.json") for suite, path in inputs.items()
            },
        },
        "residual_errors": {"failed_jobs": [], "et_task_errors": []},
        "notes": [
            "Merged from split suite final_summary.json files.",
            "Scores are copied from each suite summary without rescoring.",
        ],
    }

    output = job_output_dir(args.run_id) / "full" / "final_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "sources": payload["source_summaries"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
