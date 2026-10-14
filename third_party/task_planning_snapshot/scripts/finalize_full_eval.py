#!/usr/bin/env python
"""Write final score, runtime, and residual-error summaries for a full-eval run."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task_planning_config import job_output_dir, job_scratch_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBODIED_ROOT = PROJECT_ROOT / "EmbodiedBench"
EXPECTED_ET_TASKS = 2023
EXPECTED_ET_SHARDS = 205
EXPECTED_EB_EPISODES = 300


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    stat = path.stat()
    return {
        "path": rel(path),
        "exists": True,
        "bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def parse_utc(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def model_basename(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def eb_direct_run_root(root: Path, dataset: str) -> Path:
    if dataset == "alfred":
        return root / "full" / "EB-ALFRED" / "runs"
    if dataset == "habitat":
        return root / "full" / "EB-Habitat" / "runs"
    if dataset == "EB-ALFRED":
        return root / "full" / "EB-ALFRED" / "runs"
    if dataset == "EB-Habitat":
        return root / "full" / "EB-Habitat" / "runs"
    raise ValueError(f"Unknown EB dataset: {dataset}")


def official_average_json_values():
    if str(EMBODIED_ROOT) not in sys.path:
        sys.path.insert(0, str(EMBODIED_ROOT))
    from embodiedbench.evaluator.summarize_result import average_json_values

    return average_json_values


def et_score(root: Path) -> dict[str, Any]:
    strict_path = root / "strict_rescore" / "et_strict_summary.json"
    if strict_path.exists():
        strict = load_json(strict_path)
        return {
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

    summaries = sorted((root / "full" / "ET-Plan_Bench").glob("*/*/summary.json"))
    rows = [load_json(path) for path in summaries]
    total = sum(row.get("task_count", 0) for row in rows)
    if not total:
        return {
            "task_count": 0,
            "summary_files": len(summaries),
            "note": "ET score omitted until official full/ET-Plan_Bench summaries exist.",
        }
    return {
        "task_count": total,
        "summary_files": len(summaries),
        "scoring_mode": "official-output",
        "success_rate": sum(row.get("success_rate", 0) * row.get("task_count", 0) for row in rows) / total,
        "task_success_rate": sum(row.get("task_success_rate", 0) * row.get("task_count", 0) for row in rows) / total,
        "format_success_rate": sum(row.get("format_success_rate", 0) * row.get("task_count", 0) for row in rows) / total,
        "action_execution_success_rate": None,
        "average_steps": sum(row.get("average_steps", 0) * row.get("task_count", 0) for row in rows) / total,
        "average_lcs": sum(row.get("average_lcs", 0) * row.get("task_count", 0) for row in rows) / total,
        "average_lcs_ratio": sum(row.get("average_lcs_ratio", 0) * row.get("task_count", 0) for row in rows) / total,
    }


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
    direct = sorted(eb_direct_run_root(root, dataset_name).glob("*/*/results/episode_*_final_res.json"))
    if direct:
        return direct
    return sorted((root / "full" / dataset_name / "episode_results").glob("*/episode_*_final_res.json"))


def archived_eb_model_output_paths(root: Path, dataset_name: str) -> list[Path]:
    direct = sorted(eb_direct_run_root(root, dataset_name).glob("*/*/model_outputs/episode_*_model_outputs.jsonl"))
    if direct:
        return direct
    return sorted((root / "full" / dataset_name / "model_outputs").glob("*/episode_*_model_outputs.jsonl"))


def eb_exp_dir_from_result_path(path: Path) -> str:
    if path.parent.name in {"results", "model_outputs"} and len(path.parents) >= 3:
        return path.parents[2].name
    return path.parent.name


def call_eb_episode_paths(run_id: str, model: str, dataset: str, root: Path) -> list[Path]:
    try:
        return eb_episode_paths(run_id, model, dataset, root)
    except TypeError:
        return eb_episode_paths(run_id, model, dataset)


def call_eb_model_output_paths(run_id: str, model: str, dataset: str, root: Path) -> list[Path]:
    try:
        return eb_model_output_paths(run_id, model, dataset, root)
    except TypeError:
        return eb_model_output_paths(run_id, model, dataset)


def eb_model_output_for_result(root: Path, dataset: str, result_path: Path) -> Path | None:
    ep_num = episode_number(result_path)
    if ep_num is None:
        return None
    if result_path.parent.name == "results" and len(result_path.parents) >= 2:
        return result_path.parent.parent / "model_outputs" / f"episode_{ep_num}_model_outputs.jsonl"
    exp_dir = result_path.parent.name
    return root / "full" / dataset / "model_outputs" / exp_dir / f"episode_{ep_num}_model_outputs.jsonl"


def mirror_episode_results(paths: list[Path], dest_root: Path) -> int:
    copied = 0
    for path in paths:
        exp_dir = eb_exp_dir_from_result_path(path)
        dest = dest_root / exp_dir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == path.stat().st_size:
            copied += 1
            continue
        shutil.copyfile(path, dest)
        copied += 1
    return copied


def mirror_model_outputs(paths: list[Path], dest_root: Path) -> int:
    copied = 0
    for path in paths:
        exp_dir = eb_exp_dir_from_result_path(path)
        dest = dest_root / exp_dir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == path.stat().st_size:
            copied += 1
            continue
        shutil.copy2(path, dest)
        copied += 1
    return copied


def jsonl_light_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "records": None,
        "parse_errors_on_first_record": 0,
        "first_record": {},
        "contains_prompt_text": None,
        "contains_raw_output_text": None,
        "contains_parsed_action": None,
        "max_image_count": 0,
    }
    with path.open("rb") as f:
        first_line = f.readline(2 * 1024 * 1024).strip()
    if not first_line:
        stats["records"] = 0
        return stats
    text = first_line.decode("utf-8", errors="replace")
    stats["contains_prompt_text"] = '"prompt_text"' in text
    stats["contains_raw_output_text"] = '"raw_output_text"' in text
    stats["contains_parsed_action"] = '"parsed_action"' in text
    try:
        item = json.loads(text)
    except json.JSONDecodeError:
        stats["parse_errors_on_first_record"] = 1
        return stats
    if isinstance(item, dict):
        stats["first_record"] = item
        try:
            stats["max_image_count"] = max(stats["max_image_count"], int(item.get("image_count", 0) or 0))
        except (TypeError, ValueError):
            pass
    return stats


def episode_number(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def parse_eb_exp_dir(exp_dir: str, dataset: str) -> dict[str, Any]:
    if dataset == "EB-ALFRED":
        match = re.search(r"_full_alfred_(?P<eval_set>.+)_shard(?P<shard>\d+)$", exp_dir)
    else:
        match = re.search(r"_full_habitat_(?P<eval_set>.+)_shard(?P<shard>\d+)$", exp_dir)
    if not match:
        return {"eval_set": None, "shard": None}
    return {"eval_set": match.group("eval_set"), "shard": int(match.group("shard"))}


def write_eb_episode_index(root: Path, dataset: str, *, deep_scan: bool = False) -> dict[str, Any]:
    dataset_dir = root / "full" / dataset
    result_paths = archived_eb_episode_paths(root, dataset)
    available_model_outputs = archived_eb_model_output_paths(root, dataset)
    rows: list[dict[str, Any]] = []
    missing_model_outputs = 0
    result_load_errors = 0
    model_output_first_record_json_errors = 0
    model_outputs_with_prompt_text = 0
    model_outputs_with_raw_output_text = 0
    model_outputs_with_parsed_action = 0
    total_model_output_bytes = 0

    for result_path in result_paths:
        exp_dir = eb_exp_dir_from_result_path(result_path)
        ep_num = episode_number(result_path)
        model_output_path = eb_model_output_for_result(root, dataset, result_path)
        model_output_exists = bool(model_output_path and model_output_path.exists())
        result: dict[str, Any] = {}
        result_load_error = None
        if deep_scan:
            try:
                loaded = load_json(result_path)
                if isinstance(loaded, dict):
                    result = loaded
                else:
                    result_load_error = f"unexpected_json_type:{type(loaded).__name__}"
                    result_load_errors += 1
            except Exception as exc:
                result_load_error = repr(exc)
                result_load_errors += 1
        if not model_output_exists:
            missing_model_outputs += 1
        model_output_bytes = None
        model_output_records = None
        model_output_first_record_json_error = None
        contains_prompt_text = None
        contains_raw_output_text = None
        contains_parsed_action = None
        max_image_count = None
        if deep_scan and model_output_exists and model_output_path is not None:
            model_output_bytes = model_output_path.stat().st_size
            total_model_output_bytes += model_output_bytes
            stats = jsonl_light_stats(model_output_path)
            model_output_records = stats["records"]
            model_output_first_record_json_error = stats["parse_errors_on_first_record"]
            contains_prompt_text = stats["contains_prompt_text"]
            contains_raw_output_text = stats["contains_raw_output_text"]
            contains_parsed_action = stats["contains_parsed_action"]
            max_image_count = stats["max_image_count"]
            model_output_first_record_json_errors += int(stats["parse_errors_on_first_record"] or 0)
            model_outputs_with_prompt_text += int(bool(contains_prompt_text))
            model_outputs_with_raw_output_text += int(bool(contains_raw_output_text))
            model_outputs_with_parsed_action += int(bool(contains_parsed_action))
        exp_info = parse_eb_exp_dir(exp_dir, dataset)
        rows.append(
            {
                "dataset": dataset,
                "eval_set": exp_info.get("eval_set"),
                "shard": exp_info.get("shard"),
                "exp_dir": exp_dir,
                "episode_index": ep_num,
                "instruction": result.get("instruction"),
                "task_success": result.get("task_success"),
                "task_progress": result.get("task_progress"),
                "subgoal_reward": result.get("subgoal_reward"),
                "num_steps": result.get("num_steps"),
                "planner_steps": result.get("planner_steps"),
                "planner_output_error": result.get("planner_output_error"),
                "num_invalid_actions": result.get("num_invalid_actions"),
                "num_invalid_action_ratio": result.get("num_invalid_action_ratio"),
                "episode_elapsed_seconds": result.get("episode_elapsed_seconds"),
                "episode_result": rel(result_path),
                "episode_result_load_error": result_load_error,
                "model_outputs": rel(model_output_path) if model_output_path else None,
                "model_output_exists": model_output_exists,
                "model_output_bytes": model_output_bytes,
                "model_output_records": model_output_records,
                "model_output_records_expected_from_episode_result": result.get("planner_steps"),
                "model_output_first_record_json_error": model_output_first_record_json_error,
                "contains_prompt_text": contains_prompt_text,
                "contains_raw_output_text": contains_raw_output_text,
                "contains_parsed_action": contains_parsed_action,
                "max_image_count": max_image_count,
            }
        )

    index_path = dataset_dir / "episode_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    summary = {
        "dataset": dataset,
        "episode_results": len(result_paths),
        "indexed_episodes": len(rows),
        "result_load_errors": result_load_errors,
        "missing_model_outputs": missing_model_outputs,
        "model_output_files": len(available_model_outputs),
        "model_output_bytes": total_model_output_bytes,
        "model_output_first_record_json_errors": model_output_first_record_json_errors,
        "model_outputs_with_prompt_text": model_outputs_with_prompt_text,
        "model_outputs_with_raw_output_text": model_outputs_with_raw_output_text,
        "model_outputs_with_parsed_action": model_outputs_with_parsed_action,
        "model_output_scan_mode": "first_record_schema_and_size" if deep_scan else "metadata_only",
        "model_output_schema_source": "EmbodiedBench evaluator save_model_outputs writes prompt_text, raw_output_text, parsed_action, and image_count",
        "index_file": rel(index_path),
    }
    write_json(dataset_dir / "episode_index_summary.json", summary)
    return summary


def write_et_shard_index(root: Path, *, deep_scan: bool = False) -> dict[str, Any]:
    et_dir = root / "full" / "ET-Plan_Bench"
    rows: list[dict[str, Any]] = []
    if et_dir.exists():
        for summary_path in sorted(et_dir.glob("*/*/summary.json")):
            shard_dir = summary_path.parent
            try:
                summary = load_json(summary_path)
            except Exception as exc:
                summary = {"_load_error": repr(exc)}
            rows.append(
                {
                    "benchmark": shard_dir.parent.name,
                    "shard": shard_dir.name,
                    "summary": rel(summary_path),
                    "predictions_dir": rel(shard_dir / "predictions"),
                    "scores_dir": rel(shard_dir / "scores"),
                    "prediction_files": len(list((shard_dir / "predictions").glob("task_*/prediction_task*.json")))
                    if deep_scan
                    else None,
                    # Official ET scores are written as scores/<batch>/*.json.
                    # Avoid 205 repeated NAS metadata walks during normal finalize;
                    # task_count from official shard summaries is the default
                    # completeness source, and --deep-artifact-index restores
                    # per-file counting when an artifact audit is needed.
                    "score_files": len(list((shard_dir / "scores").glob("*/*.json"))) if deep_scan else None,
                    "task_count": summary.get("task_count"),
                    "task_success_rate": summary.get("task_success_rate", summary.get("success_rate")),
                    "format_success_rate": summary.get("format_success_rate"),
                    "average_steps": summary.get("average_steps"),
                }
            )

    index_path = et_dir / "shard_index.jsonl"
    if rows:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
    summary = {
        "dataset": "ET-Plan_Bench",
        "shards": len(rows),
        "tasks_from_summaries": sum(int(row.get("task_count") or 0) for row in rows),
        "prediction_files": sum(int(row.get("prediction_files") or 0) for row in rows) if deep_scan else None,
        "score_files": sum(int(row.get("score_files") or 0) for row in rows) if deep_scan else None,
        "file_count_scan_mode": "per_file" if deep_scan else "summary_only",
        "index_file": rel(index_path) if rows else None,
    }
    if et_dir.exists():
        write_json(et_dir / "shard_index_summary.json", summary)
    return summary


def previous_manifest_datasets(root: Path) -> dict[str, Any]:
    manifest_path = root / "full" / "artifacts_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return {}
    datasets = manifest.get("datasets")
    return datasets if isinstance(datasets, dict) else {}


def previous_final_scores(root: Path) -> dict[str, Any]:
    summary_path = root / "full" / "final_summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = load_json(summary_path)
    except Exception:
        return {}
    scores = summary.get("scores")
    return scores if isinstance(scores, dict) else {}


def reusable_et_score(root: Path) -> dict[str, Any] | None:
    score = previous_final_scores(root).get("ET-Plan_Bench")
    if not isinstance(score, dict):
        return None
    if int(score.get("task_count", 0) or 0) != EXPECTED_ET_TASKS:
        return None
    if int(score.get("summary_files", 0) or 0) != EXPECTED_ET_SHARDS:
        return None
    return score


def reusable_dataset_index(root: Path, dataset: str, previous: dict[str, Any]) -> dict[str, Any] | None:
    info = previous.get(dataset)
    if not isinstance(info, dict):
        return None
    index_file = info.get("index_file")
    if index_file and not (PROJECT_ROOT / str(index_file)).exists():
        return None
    if dataset == "ET-Plan_Bench":
        if int(info.get("tasks_from_summaries", 0) or 0) != EXPECTED_ET_TASKS:
            return None
        if int(info.get("shards", 0) or 0) != EXPECTED_ET_SHARDS:
            return None
        return info
    if dataset in {"EB-ALFRED", "EB-Habitat"}:
        if int(info.get("episode_results", 0) or 0) != EXPECTED_EB_EPISODES:
            return None
        if int(info.get("missing_model_outputs", 0) or 0) != 0:
            return None
        return info
    return None


def official_eb_summary(paths: list[Path], root: Path, dataset_name: str) -> dict[str, Any]:
    out_summary = root / "full" / dataset_name / "summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"tp_{dataset_name}_") as staging_dir:
        staging_results = Path(staging_dir)
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
            average_json_values(
                str(staging_results),
                target_file="episode_*_final_res.json",
                output_file="summary.json",
            )
        summary = load_json(staging_results / "summary.json")
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "episode_count": len(paths),
        **summary,
        "summary_source": "EmbodiedBench.average_json_values",
        "official_summary_path": str(out_summary),
    }


def existing_eb_index_summary(
    root: Path,
    dataset_name: str,
    previous_datasets: dict[str, Any],
    *,
    create_metadata_index: bool = False,
) -> dict[str, Any]:
    previous = previous_datasets.get(dataset_name)
    if isinstance(previous, dict):
        return previous
    summary_path = root / "full" / dataset_name / "episode_index_summary.json"
    if summary_path.exists():
        try:
            loaded = load_json(summary_path)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    if create_metadata_index and (root / "full" / dataset_name).exists():
        return write_eb_episode_index(root, dataset_name, deep_scan=False)
    return {}


def existing_eb_score(root: Path, dataset_name: str, previous_datasets: dict[str, Any]) -> dict[str, Any]:
    """Score the current archived episode set, never a cached suite summary.

    A previous finalization can have written ``summary.json`` while only a
    subset of asynchronous EB shards had reached NAS.  Later shards then add
    result files without updating that summary.  Therefore an archived summary
    is not evidence for the score of the archived result set: rebuild it with
    the official EmbodiedBench aggregator whenever finalizing.
    """
    index = existing_eb_index_summary(root, dataset_name, previous_datasets, create_metadata_index=True)
    paths = archived_eb_episode_paths(root, dataset_name)
    episode_count = int(index.get("episode_results", 0) or 0)
    if episode_count <= 0 and paths:
        index = write_eb_episode_index(root, dataset_name, deep_scan=False)
        episode_count = int(index.get("episode_results", 0) or 0) or len(paths)
    if episode_count <= 0:
        return {"episode_count": 0}
    score = official_eb_summary(paths, root, dataset_name) if paths else {"episode_count": episode_count}
    score["mirrored_episode_results"] = episode_count
    score["mirrored_model_output_files"] = int(index.get("model_output_files", 0) or 0)
    return score


def eb_scores(
    run_id: str,
    model: str,
    root: Path,
    *,
    skip_runtime_mirror: bool = False,
    skip_model_output_mirror: bool = False,
) -> dict[str, dict[str, Any]]:
    previous_datasets = previous_manifest_datasets(root)
    if skip_runtime_mirror:
        return {
            "EB-ALFRED": existing_eb_score(root, "EB-ALFRED", previous_datasets),
            "EB-Habitat": existing_eb_score(root, "EB-Habitat", previous_datasets),
        }

    alfred_raw_paths = call_eb_episode_paths(run_id, model, "alfred", root)
    habitat_raw_paths = call_eb_episode_paths(run_id, model, "habitat", root)
    if alfred_raw_paths and not all(path.is_relative_to(root) for path in alfred_raw_paths):
        mirror_episode_results(alfred_raw_paths, root / "full" / "EB-ALFRED" / "episode_results")
        if not skip_model_output_mirror:
            mirror_model_outputs(
                call_eb_model_output_paths(run_id, model, "alfred", root),
                root / "full" / "EB-ALFRED" / "model_outputs",
            )
    if habitat_raw_paths and not all(path.is_relative_to(root) for path in habitat_raw_paths):
        mirror_episode_results(habitat_raw_paths, root / "full" / "EB-Habitat" / "episode_results")
        if not skip_model_output_mirror:
            mirror_model_outputs(
                call_eb_model_output_paths(run_id, model, "habitat", root),
                root / "full" / "EB-Habitat" / "model_outputs",
            )

    # A resumed container may expose only newly generated episodes. Score the
    # durable union after mirroring instead of replacing/scoring that subset.
    alfred_paths = archived_eb_episode_paths(root, "EB-ALFRED")
    habitat_paths = archived_eb_episode_paths(root, "EB-Habitat")
    if alfred_raw_paths:
        alfred_score = official_eb_summary(alfred_paths, root, "EB-ALFRED")
    elif alfred_paths:
        # Do not reuse full/EB-ALFRED/summary.json: it may predate late shards.
        alfred_score = official_eb_summary(alfred_paths, root, "EB-ALFRED")
    else:
        alfred_score = {"episode_count": 0}

    if habitat_raw_paths:
        habitat_score = official_eb_summary(habitat_paths, root, "EB-Habitat")
    elif habitat_paths:
        # Do not reuse full/EB-Habitat/summary.json: it may predate late shards.
        habitat_score = official_eb_summary(habitat_paths, root, "EB-Habitat")
    else:
        habitat_score = {"episode_count": 0}

    scores = {"EB-ALFRED": alfred_score, "EB-Habitat": habitat_score}
    if alfred_paths:
        scores["EB-ALFRED"]["mirrored_episode_results"] = len(alfred_paths)
        scores["EB-ALFRED"]["mirrored_model_output_files"] = len(
            archived_eb_model_output_paths(root, "EB-ALFRED")
        )
    if habitat_paths:
        scores["EB-Habitat"]["mirrored_episode_results"] = len(habitat_paths)
        scores["EB-Habitat"]["mirrored_model_output_files"] = len(
            archived_eb_model_output_paths(root, "EB-Habitat")
        )
    return scores


def write_output_readme(root: Path, manifest: dict[str, Any]) -> Path:
    lines = [
        "# TP 评测输出目录说明",
        "",
        "本目录是评测结束后自动生成的规范输出包。`scripts/` 只做归档、索引和完整性检查；三套测试集的执行、后处理和算分仍来自对应官方评测框架。",
        "",
        "## 关键文件",
        "",
        "- `final_summary.json`：最终分数、运行耗时、残余错误和输出完整性摘要。",
        "- `final_report.md`：便于人工阅读的最终报告。",
        "- `artifacts_manifest.json`：机器可读的输出清单，记录配置、分数文件、日志、索引文件和完整性检查结果。",
        "- `ET-Plan_Bench/shard_index.jsonl`：ET 每个 benchmark shard 的 summary、prediction、score 位置索引。",
        "- `EB-ALFRED/episode_index.jsonl`：EB-ALFRED 每个 episode 的官方结果与模型逐步输出对应关系。",
        "- `EB-Habitat/episode_index.jsonl`：EB-Habitat 每个 episode 的官方结果与模型逐步输出对应关系。",
        "",
        "## EB episode 级文件",
        "",
        "- `episode_results/**/episode_*_final_res.json`：官方 evaluator 保存的 episode 指标，包含 `task_success`、`task_progress`、`num_steps`、`planner_output_error`、`num_invalid_actions` 等。",
        "- `model_outputs/**/episode_*_model_outputs.jsonl`：逐轮模型调用记录，包含输入模型前的文本 prompt、图像数量、模型原始输出、解析后的动作等。为避免输出体积爆炸，这里不内嵌 base64 图像。",
        "",
        "## 完整性",
        "",
        f"- 输出包生成时间：`{manifest['generated_at']}`",
        f"- 完整性状态：`{manifest['completeness']['status']}`",
    ]
    for issue in manifest["completeness"].get("issues", []):
        lines.append(f"- `{issue}`")
    path = root / "full" / "OUTPUT_README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_artifacts_manifest(root: Path, summary: dict[str, Any], *, deep_artifact_index: bool = False) -> dict[str, Any]:
    full_dir = root / "full"
    previous_datasets = previous_manifest_datasets(root)
    et_index = reusable_dataset_index(root, "ET-Plan_Bench", previous_datasets) or write_et_shard_index(
        root, deep_scan=deep_artifact_index
    )
    eb_indices = {}
    for dataset in ["EB-ALFRED", "EB-Habitat"]:
        if not (full_dir / dataset).exists():
            continue
        eb_indices[dataset] = reusable_dataset_index(root, dataset, previous_datasets) or write_eb_episode_index(
            root, dataset, deep_scan=deep_artifact_index
        )

    manifest_jobs: list[dict[str, Any]] = []
    for path in sorted(full_dir.glob("manifest_*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, list):
            manifest_jobs.extend(item for item in payload if isinstance(item, dict))
    jobs_by_kind = {
        kind: {Path(str(job.get("status", ""))).name for job in manifest_jobs if job.get("kind") == kind}
        for kind in ("et", "alfred", "habitat")
    }

    issues: list[str] = []
    et_score_count = int(summary["scores"].get("ET-Plan_Bench", {}).get("task_count", 0) or 0)
    et_shards = int(et_index.get("shards", 0) or 0)
    if et_score_count != EXPECTED_ET_TASKS:
        issues.append(f"ET-Plan_Bench: expected {EXPECTED_ET_TASKS} tasks for a full run, got {et_score_count}")
    if et_shards != EXPECTED_ET_SHARDS:
        issues.append(f"ET-Plan_Bench: expected {EXPECTED_ET_SHARDS} shards for a full run, got {et_shards}")
    if jobs_by_kind["et"]:
        if et_score_count != int(et_index.get("tasks_from_summaries", 0) or 0):
            issues.append(
                f"ET-Plan_Bench: score task_count={et_score_count} but indexed tasks={et_index.get('tasks_from_summaries')}"
            )
        if deep_artifact_index:
            if int(et_index.get("prediction_files", 0) or 0) < et_score_count:
                issues.append(
                    f"ET-Plan_Bench: prediction files={et_index.get('prediction_files')} below task_count={et_score_count}"
                )
            if int(et_index.get("score_files", 0) or 0) < et_score_count:
                issues.append(
                    f"ET-Plan_Bench: score files={et_index.get('score_files')} below task_count={et_score_count}"
                )
    for dataset in ["EB-ALFRED", "EB-Habitat"]:
        if dataset not in eb_indices:
            issues.append(f"{dataset}: missing dataset artifacts for a full run")
            continue
        index = eb_indices[dataset]
        score_count = int(summary["scores"].get(dataset, {}).get("episode_count", 0) or 0)
        if score_count != int(index.get("episode_results", 0) or 0):
            issues.append(f"{dataset}: summary episode_count={score_count} but indexed episode_results={index.get('episode_results')}")
        if index.get("result_load_errors"):
            issues.append(f"{dataset}: episode result JSON load errors={index['result_load_errors']}")
        if index.get("missing_model_outputs"):
            issues.append(f"{dataset}: missing model_outputs for {index['missing_model_outputs']} episodes")
        if index.get("model_output_first_record_json_errors"):
            issues.append(
                f"{dataset}: model_outputs first-record JSON parse errors={index['model_output_first_record_json_errors']}"
            )
        kind = "alfred" if dataset == "EB-ALFRED" else "habitat"
        if score_count != EXPECTED_EB_EPISODES:
            issues.append(f"{dataset}: expected {EXPECTED_EB_EPISODES} episodes for a full run, got {score_count}")
    status_files = {path.name for path in (root / "status").glob("full_*.json")}
    expected_status_files = set().union(*jobs_by_kind.values())
    missing_status_files = sorted(expected_status_files - status_files)
    if missing_status_files:
        issues.append(f"missing shard status files={len(missing_status_files)}")
    failed_jobs = summary.get("residual_errors", {}).get("failed_jobs", [])
    et_task_errors = summary.get("residual_errors", {}).get("et_task_errors", [])
    if failed_jobs:
        issues.append(f"retryable shard jobs remain={len(failed_jobs)}")
    if et_task_errors:
        issues.append(f"unresolved ET task errors remain={len(et_task_errors)}")
    if not (full_dir / "final_summary.json").exists():
        issues.append("final_summary.json missing")
    if not (full_dir / "final_report.md").exists():
        issues.append("final_report.md missing")

    manifest = {
        "schema_version": "tp-full-eval-artifacts-v1",
        "job_name": summary["run_id"],
        "run_id": summary["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root": rel(root),
        "output_contract": {
            "scoring_logic": "official evaluator/scorer outputs only; wrapper mirrors, indexes, validates, and summarizes artifacts",
            "eb_model_outputs": "per-episode JSONL with text prompt, image_count, raw_output_text, parsed_action, and parse counters",
            "image_policy": "image binaries stay in simulator/runtime outputs when produced; JSONL records keep image_count instead of embedded base64",
        },
        "primary_files": {
            "final_summary": file_info(full_dir / "final_summary.json"),
            "final_report": file_info(full_dir / "final_report.md"),
            "aggregate_summary": file_info(full_dir / "aggregate_summary.json"),
        },
        "configs": [file_info(path) for path in sorted(full_dir.glob("eval_config_*.json"))],
        "manifests": [file_info(path) for path in sorted(full_dir.glob("manifest_*.json"))],
        "datasets": {
            "ET-Plan_Bench": et_index,
            **eb_indices,
        },
        "logs": {
            "pipeline_logs": [file_info(path) for path in sorted((root / "logs").glob("pipeline_*.log"))],
            "shard_logs_count": len(list((root / "logs").glob("full_*.log"))),
            "status_files_count": len(list((root / "status").glob("full_*.json"))),
        },
        "completeness": {
            "status": "complete" if not issues else "incomplete",
            "issues": issues,
        },
    }
    manifest_path = full_dir / "artifacts_manifest.json"
    write_json(manifest_path, manifest)
    readme_path = write_output_readme(root, manifest)
    manifest["primary_files"]["artifacts_manifest"] = file_info(manifest_path)
    manifest["primary_files"]["output_readme"] = file_info(readme_path)
    write_json(manifest_path, manifest)
    return manifest


def refresh_manifest_primary_files(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Refresh manifest file metadata without rebuilding expensive dataset indices."""
    full_dir = root / "full"
    manifest_path = full_dir / "artifacts_manifest.json"
    manifest["primary_files"].update(
        {
            "final_summary": file_info(full_dir / "final_summary.json"),
            "final_report": file_info(full_dir / "final_report.md"),
            "aggregate_summary": file_info(full_dir / "aggregate_summary.json"),
            "output_readme": file_info(full_dir / "OUTPUT_README.md"),
            "artifacts_manifest": file_info(manifest_path),
        }
    )
    write_json(manifest_path, manifest)
    manifest["primary_files"]["artifacts_manifest"] = file_info(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def reusable_complete_manifest(root: Path) -> dict[str, Any] | None:
    """Return an existing complete manifest when it already covers the full run."""
    manifest_path = root / "full" / "artifacts_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return None
    if manifest.get("completeness", {}).get("status") != "complete":
        return None
    datasets = manifest.get("datasets", {})
    required = {"ET-Plan_Bench", "EB-ALFRED", "EB-Habitat"}
    if not required.issubset(datasets):
        return None
    et = datasets.get("ET-Plan_Bench") or {}
    alfred = datasets.get("EB-ALFRED") or {}
    habitat = datasets.get("EB-Habitat") or {}
    if int(et.get("tasks_from_summaries", 0) or 0) != EXPECTED_ET_TASKS:
        return None
    if int(et.get("shards", 0) or 0) != EXPECTED_ET_SHARDS:
        return None
    if int(alfred.get("episode_results", 0) or 0) != EXPECTED_EB_EPISODES:
        return None
    if int(habitat.get("episode_results", 0) or 0) != EXPECTED_EB_EPISODES:
        return None
    return manifest


def runtime_summary(root: Path) -> dict[str, Any]:
    statuses: list[dict[str, Any]] = []
    for path in sorted((root / "status").glob("full_*.json")):
        try:
            item = load_json(path)
        except Exception:
            continue
        item["_file"] = path.name
        statuses.append(item)

    by_kind: dict[str, dict[str, Any]] = {}
    starts: list[float] = []
    ends: list[float] = []
    for item in statuses:
        name = item["_file"]
        if name.startswith("full_et_"):
            kind = "et"
        elif name.startswith("full_eb_alfred_"):
            kind = "alfred"
        elif name.startswith("full_eb_habitat_"):
            kind = "habitat"
        else:
            kind = "unknown"
        group = by_kind.setdefault(
            kind,
            {"shards": 0, "success": 0, "failed": 0, "elapsed_sec_sum": 0.0, "elapsed_sec_avg": None, "_values": []},
        )
        group["shards"] += 1
        group["success"] += int(item.get("status") == "success")
        group["failed"] += int(item.get("status") == "failed")
        if item.get("elapsed_sec") is not None:
            value = float(item["elapsed_sec"])
            group["elapsed_sec_sum"] += value
            group["_values"].append(value)
        started = parse_utc(item.get("started_at"))
        ended = parse_utc(item.get("ended_at"))
        if started:
            starts.append(started)
        if ended:
            ends.append(ended)

    for group in by_kind.values():
        values = group.pop("_values")
        group["elapsed_sec_avg"] = sum(values) / len(values) if values else None

    return {
        "shards_total": len(statuses),
        "shards_success": sum(1 for item in statuses if item.get("status") == "success"),
        "shards_failed": sum(1 for item in statuses if item.get("status") == "failed"),
        "first_started_at": datetime.fromtimestamp(min(starts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        if starts
        else None,
        "last_ended_at": datetime.fromtimestamp(max(ends), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        if ends
        else None,
        "wall_elapsed_sec_first_start_to_last_end": max(ends) - min(starts) if starts and ends else None,
        "elapsed_sec_sum_all_shards": sum(float(item.get("elapsed_sec", 0) or 0) for item in statuses),
        "by_kind": by_kind,
    }


def full_eval_status(run_id: str, probe_vllm: bool) -> dict[str, Any]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "full_eval_status.py"), "--job-name", run_id, "--json"]
    if probe_vllm:
        cmd.append("--probe-vllm")
    return json.loads(subprocess.check_output(cmd, cwd=str(PROJECT_ROOT), text=True))


def render_markdown(summary: dict[str, Any]) -> str:
    scores = summary["scores"]
    runtime = summary["runtime"]
    et = scores["ET-Plan_Bench"]
    alfred = scores["EB-ALFRED"]
    habitat = scores["EB-Habitat"]
    residual = summary["residual_errors"].get("et_task_errors", [])
    wall_elapsed = runtime.get("wall_elapsed_sec_first_start_to_last_end")
    wall_elapsed_text = f"{wall_elapsed:.1f}" if wall_elapsed is not None else "N/A"

    lines = [
        "# TP 三测试集全量评测最终结果",
        "",
        f"- Job Name: `{summary['job_name']}`",
        f"- 生成时间: `{summary['generated_at']}`",
        f"- Shard 状态: `{runtime['shards_success']}/{runtime['shards_total']}` success, failed `{runtime['shards_failed']}`",
        f"- ET task-level 残余错误: `{len(residual)}`",
        f"- 根目录临时文件: `{summary['root_artifacts']}`",
        "",
        "## 分数",
        "",
        "| 测试集 | 样本数 | 主指标 | 任务进度/补充指标 |",
        "|---|---:|---:|---:|",
        f"| ET-Plan_Bench | {et.get('task_count', 0)} tasks | task_success_rate {et.get('task_success_rate', et.get('success_rate', 0)):.6f} | format_success_rate {et.get('format_success_rate', 0):.6f}, action_execution_success_rate {et.get('action_execution_success_rate') if et.get('action_execution_success_rate') is not None else 'N/A'} |",
        f"| EB-ALFRED | {alfred.get('episode_count', 0)} episodes | task_success {alfred.get('task_success', 0):.6f} | task_progress {alfred.get('task_progress', 0):.6f} |",
        f"| EB-Habitat | {habitat.get('episode_count', 0)} episodes | task_success {habitat.get('task_success', 0):.6f} | task_progress {habitat.get('task_progress', 0):.6f} |",
        "",
        "## EB 输出归档",
        "",
        "| 测试集 | episode_results | model_outputs |",
        "|---|---:|---:|",
        f"| EB-ALFRED | {alfred.get('mirrored_episode_results', 0)} | {alfred.get('mirrored_model_output_files', 0)} |",
        f"| EB-Habitat | {habitat.get('mirrored_episode_results', 0)} | {habitat.get('mirrored_model_output_files', 0)} |",
        "",
        "## 耗时",
        "",
        f"- 首个 shard 开始: `{runtime['first_started_at']}`",
        f"- 最后 shard 结束: `{runtime['last_ended_at']}`",
        f"- wall time: `{wall_elapsed_text}` 秒",
        f"- shard elapsed 累加: `{runtime['elapsed_sec_sum_all_shards']:.1f}` 秒",
        "",
        "| 类型 | shards | avg elapsed sec | sum elapsed sec |",
        "|---|---:|---:|---:|",
    ]
    for kind in ["et", "alfred", "habitat"]:
        item = runtime["by_kind"].get(kind, {})
        lines.append(
            f"| {kind} | {item.get('shards', 0)} | {(item.get('elapsed_sec_avg') or 0):.1f} | {(item.get('elapsed_sec_sum') or 0):.1f} |"
        )

    lines.extend(["", "## 残余错误", ""])
    if residual:
        for error in residual:
            lines.append(
                f"- `{error['error_file']}`: `{error.get('error_type')}`, task_id `{error.get('task_id')}`"
            )
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 调度结论",
            "",
            "调度结论需结合本次 vLLM 日志中的请求队列、GPU 利用率和各类 shard 耗时判断；报告不预填固定瓶颈结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", dest="run_id", required=True)
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--probe-vllm", action="store_true")
    parser.add_argument(
        "--skip-full-eval-status",
        action="store_true",
        help="Skip the progress/status helper when finalizing already mirrored artifacts from a slow output tree.",
    )
    parser.add_argument(
        "--deep-artifact-index",
        action="store_true",
        help="Read every EB result/model-output record while indexing artifacts. Slower on NAS; default is metadata-only.",
    )
    parser.add_argument(
        "--skip-runtime-mirror",
        action="store_true",
        help="Use already mirrored EB summaries/artifact counts instead of scanning runtime scratch and model_outputs.",
    )
    parser.add_argument(
        "--skip-model-output-mirror",
        action="store_true",
        help="Mirror EB episode result JSON only. Useful for fast partial score publication before final archival.",
    )
    args = parser.parse_args()

    run_id = args.run_id
    root = job_output_dir(run_id)
    previous_payload: dict[str, Any] = {}
    previous_summary_path = root / "full" / "final_summary.json"
    if previous_summary_path.exists():
        try:
            loaded = load_json(previous_summary_path)
            if isinstance(loaded, dict):
                previous_payload = loaded
        except Exception:
            pass
    if args.skip_full_eval_status:
        status = {
            "retry_plan": previous_payload.get("residual_errors", {"failed_jobs": [], "et_task_errors": []}),
            "root_artifacts": previous_payload.get("root_artifacts", {}),
        }
    else:
        status = full_eval_status(run_id, args.probe_vllm)
    runtime = runtime_summary(root)
    if args.skip_full_eval_status and not runtime["shards_total"]:
        saved_runtime = previous_payload.get("runtime")
        if isinstance(saved_runtime, dict):
            runtime = saved_runtime
    et_scores = reusable_et_score(root) if args.skip_runtime_mirror else None
    summary = {
        "job_name": run_id,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scores": {
            "ET-Plan_Bench": et_scores or et_score(root),
            **eb_scores(
                run_id,
                args.model,
                root,
                skip_runtime_mirror=args.skip_runtime_mirror,
                skip_model_output_mirror=args.skip_model_output_mirror,
            ),
        },
        "runtime": runtime,
        "residual_errors": status.get("retry_plan", {}),
        "root_artifacts": status.get("root_artifacts", {}),
        "notes": [
            "All shard-level jobs completed successfully when shards_success equals shards_total.",
            "Remaining ET task-level errors should be reviewed as residual engineering/model failures.",
            "Use scripts/full_eval_retry.py for repeatable failed-shard and ET task-error retry.",
        ],
    }
    out_dir = root / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "final_summary.json"
    report_path = out_dir / "final_report.md"
    summary["artifacts"] = {
        "manifest": rel(out_dir / "artifacts_manifest.json"),
        "readme": rel(out_dir / "OUTPUT_README.md"),
        "completeness": {"status": "pending", "issues": ["artifact manifest has not been generated yet"]},
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    # The run directory should be complete immediately after finalization:
    # manifest/readme/index files are part of the default artifact contract.
    artifacts = reusable_complete_manifest(root)
    if artifacts is not None:
        artifacts = refresh_manifest_primary_files(root, artifacts)
    else:
        artifacts = write_artifacts_manifest(root, summary, deep_artifact_index=args.deep_artifact_index)
    summary["artifacts"]["completeness"] = artifacts["completeness"]
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    refresh_manifest_primary_files(root, artifacts)
    print(json.dumps({"json": str(json_path), "markdown": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
