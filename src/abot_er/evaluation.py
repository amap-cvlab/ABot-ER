"""Portable orchestration of the frozen upstream EQA evaluator (no new scorer)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

from .reproduction import ROOT, read_json, sha256, write_json

SNAPSHOT = ROOT / "third_party/embodied_eval_snapshot"
TASK_ROOT = SNAPSHOT / "embodied_eval/tasks"
DEFAULT_TASKS = (
    "vsibench", "erqa", "where2place-point_aboter", "emspatial-bench",
    "cosmos-agibot", "cosmos-bridgev2", "cosmos-holoassist", "cosmos-robofail",
    "cosmos-robovqa", "vabench_aboter", "robovqa", "openeqa-emeqa", "unieqa",
)
JUDGED = {"robovqa", "openeqa-emeqa", "unieqa"}


def has_response_text(value):
    """Inspect nested upstream response lists without treating [] or None as text."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(has_response_text(item) for item in value)
    return False


def audit_response_file(path):
    """Record missing/empty generations separately from the unchanged scorer."""
    counts = {"records": 0, "empty_responses": 0, "missing_response_field": 0}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            counts["records"] += 1
            key = "resps" if "resps" in sample else "filtered_resps"
            if key not in sample:
                counts["missing_response_field"] += 1
            if not has_response_text(sample.get(key)):
                counts["empty_responses"] += 1
    return counts


class ContractLoader(yaml.SafeLoader):
    pass


ContractLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))


def task_index():
    result = {}
    for path in sorted(TASK_ROOT.glob("*/*.yaml")):
        if path.name.startswith("."):
            continue
        config = yaml.load(path.read_text(), Loader=ContractLoader)
        if config["task"] in result:
            raise ValueError(f"Duplicate frozen task: {config['task']}")
        result[config["task"]] = (path, config)
    return result


def resolve_contracts(dataset_manifest, selected=DEFAULT_TASKS):
    """Only relocate dataset/media paths; reject prompt/scorer overrides."""
    manifest = read_json(dataset_manifest)
    mappings = manifest["tasks"]
    index = task_index()
    configs = {}
    for name in selected:
        if name not in index or name not in mappings:
            raise ValueError(f"Missing frozen task or dataset mapping: {name}")
        path, config = index[name]
        mapping = mappings[name]
        allowed = {"dataset_path", "video_dir", "image_root", "revision", "expected_count"}
        if set(mapping) - allowed:
            raise ValueError(f"Unsupported overrides for {name}: {set(mapping) - allowed}")
        if not mapping.get("revision"):
            raise ValueError(f"Dataset revision is required for {name}")
        if not isinstance(mapping.get("expected_count"), int) or mapping["expected_count"] <= 0:
            raise ValueError(f"Positive frozen expected_count required for {name}")
        config["dataset_kwargs"] = dict(config.get("dataset_kwargs") or {})
        for key in ("dataset_path", "video_dir", "image_root"):
            old = config.get(key) if key == "dataset_path" else config["dataset_kwargs"].get(key)
            if key not in mapping:
                if old:
                    raise ValueError(f"Explicit portable {key} required for {name}")
                continue
            new = Path(mapping[key]).expanduser()
            if not new.is_absolute() or not new.exists():
                raise ValueError(f"Existing absolute {key} required for {name}: {new}")
            if key == "dataset_path":
                config[key] = str(new.resolve())
            else:
                config["dataset_kwargs"][key] = str(new.resolve())
        if name == "vabench_aboter":
            contract = read_json(ROOT / "configs/eval/vabench_runtime_contract.json")
            config["dataset_kwargs"].update(contract["dataset_kwargs"])
        configs[name] = config
    return configs


def source_hashes():
    return {str(p.relative_to(SNAPSHOT)): sha256(p) for p in sorted(SNAPSHOT.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".yaml", ".jinja"}}


def model_identity(model, *, is_judge=False):
    model = Path(model).resolve()
    if not (model / "config.json").is_file():
        raise ValueError("Use a downloaded local HF checkpoint containing config.json")
    config = read_json(model / "config.json")
    from .checkpoint_contract import (normalize_checkpoint_config,
                                      require_task_conditioned_checkpoint, validate_local_index)
    namespace = (normalize_checkpoint_config(config) if is_judge
                 else require_task_conditioned_checkpoint(config))
    validate_local_index(config, model)
    hashes = {p.name: sha256(p) for p in sorted(model.iterdir())
              if p.is_file() and (p.suffix in {".json", ".jinja"})}
    return {"path": str(model), "metadata_sha256": hashes,
            "dhta_enabled": bool(config.get("dhta_enabled")),
            "residual_namespace": namespace,
            "weight_hash_scope": "metadata/index only; use distribution manifest for weight hashes"}


def configured_task_manager(configs):
    """Preserve the original task loading/rendering/scoring, replacing only paths."""
    from embodied_eval.tasks import TaskManager
    from embodied_eval.utils import load_yaml_config

    class PortableTaskManager(TaskManager):
        def initialize_tasks(self, include_defaults=True):
            index = task_index()
            return {name: {"type": "task", "yaml_path": str(index[name][0])} for name in configs}

        def _get_config(self, name):
            if name not in configs:
                raise ValueError(f"Task not in this frozen run: {name}")
            original = load_yaml_config(self.task_index[name]["yaml_path"])
            original["dataset_path"] = configs[name]["dataset_path"]
            original["dataset_kwargs"] = configs[name]["dataset_kwargs"]
            return original

    return PortableTaskManager


def run(args):
    if args.batch_size < 1 or args.smoke_max_tokens < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size, smoke-max-tokens and limit must be positive")
    selected = tuple(args.tasks.split(",")) if args.tasks else DEFAULT_TASKS
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate task requested")
    configs = resolve_contracts(args.datasets, selected)
    model = model_identity(args.model)
    if any(c in model["path"] for c in (",", "=")):
        raise ValueError("Model path must not contain ',' or '=' (upstream argument parser)")
    needs_judge = bool(set(selected) & JUDGED) and not args.no_judge
    judge = model_identity(args.judge_model, is_judge=True) if needs_judge and args.judge_model else None
    if needs_judge and judge is None:
        raise ValueError("OpenEQA/RoboVQA/UniEQA require --judge-model, or explicitly --no-judge")
    record = {
        "mode": "smoke" if args.limit else "full", "model": model, "judge": judge,
        "dataset_manifest_sha256": sha256(Path(args.datasets)),
        "datasets": read_json(args.datasets), "contracts": configs,
        "source_sha256": source_hashes(), "route": "[2D_QA]",
        "max_frames": 32, "frame_sampling": "numpy.linspace, total video frames, dtype=int",
        "enable_thinking": False, "seed": args.seed, "batch_size": args.batch_size, "device": args.device,
        "limit": args.limit, "judge_enabled": needs_judge,
        "generation": {"temperature": 0, "top_p": 1., "do_sample": False, "num_beams": 1},
        "note": "Per-task generation kwargs and historical checkpoint profile are preserved; see saved configs and logs.",
    }
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    os.environ["NOTEBOOK_CONTAINER"] = "1"
    # Some original task modules construct an API client even for local judging.
    # Never use a caller's real API credential or endpoint in this local-only runner.
    os.environ["OPENAI_API_KEY"] = "local-judge-unused"
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:9/v1"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
    sys.path.insert(0, str(SNAPSHOT))
    tm_class = configured_task_manager(configs)
    if args.check_data:
        manager = tm_class()
        counts = {}
        for name in selected:
            task = manager._load_individual_task(name)[name]
            count = len(task.eval_docs)
            expected = record["datasets"]["tasks"][name]["expected_count"]
            if count != expected:
                raise ValueError(f"{name}: loaded {count}, frozen expected {expected}")
            # Render the first sample; media errors must surface before GPU loading.
            doc = task.eval_docs[0]
            task.config.doc_to_visual(doc, task.dataset_kwargs)
            task.config.doc_to_text(doc, task.dataset_kwargs)
            counts[name] = count
        print(json.dumps({"status": "pass", "counts": counts}, indent=2))
        return
    from accelerate import Accelerator
    from accelerate.utils import broadcast_object_list
    from embodied_eval.evaluators import eqa_evaluator
    eqa_evaluator.TaskManager = tm_class
    output = Path(args.output).resolve()
    # For distributed runs, require launcher-generated unique output on all ranks.
    accelerator = Accelerator(cpu=args.device == "cpu")
    rank = accelerator.process_index
    failure = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "reproduction_manifest.json", record)
        except Exception as exc:
            failure[0] = f"{type(exc).__name__}: {exc}"
    broadcast_object_list(failure)
    if failure[0]:
        raise RuntimeError(failure[0])
    kwargs = ",".join((f"model_name_or_path={model['path']}", "max_frames=32", "temperature=0",
                       "top_p=1", "num_beams=1", "do_sample=False", "use_cache=True",
                       "enable_thinking=False", "max_new_tokens=1024", f"batch_size={args.batch_size}",
                       f"device={args.device}", f"device_map={args.device}"))
    config = SimpleNamespace(model="aboter", model_args=kwargs, evaluator="eqa", tasks=",".join(selected),
                             env=None, limit=args.limit, sample="no", sample_doc_ids_by_task=None,
                             batch_size=args.batch_size, save_results=True, output_path=str(output),
                             seed=args.seed, verbosity="INFO", debug=False, gpu_memory_utilization=.8,
                             job_name=output.name, preserve_task_order=True, flat_output=True, timezone="UTC")
    evaluator = eqa_evaluator.EQAEvaluator(config)
    for name, task in evaluator.task_dict.items():
        expected = record["datasets"]["tasks"][name]["expected_count"]
        if len(task.eval_docs) != expected:
            raise ValueError(f"Coverage mismatch for {name}")
        if args.limit:
            task.config.generation_kwargs = dict(task.config.generation_kwargs or {})
            task.config.generation_kwargs["max_new_tokens"] = min(
                int(task.config.generation_kwargs.get("max_new_tokens", 1024)), args.smoke_max_tokens)
    evaluator.run_sequential_tasks(llm_judge="no" if args.no_judge else "yes",
                                   llm_judge_model="local",
                                   llm_judge_model_path=str(Path(args.judge_model).resolve()) if judge else None)
    if getattr(evaluator.model, "accelerator", None):
        evaluator.model.accelerator.wait_for_everyone()
    # The upstream generator can catch an exception and emit an empty response.
    # Preserve its score, but never count that as a successful smoke generation.
    failure = [None]
    if rank == 0:
        from embodied_eval.evaluators.fast_artifacts import task_artifact_status, fast_task_dir_name
        audit = {"mode": record["mode"], "tasks": {}, "scoring_changed": False}
        try:
            for name in selected:
                expected = record["datasets"]["tasks"][name]["expected_count"]
                expected = min(expected, args.limit) if args.limit else expected
                status = task_artifact_status(output, name, "yes" if needs_judge else "no", expected)
                if not status.complete:
                    raise RuntimeError(status.reason)
                path = output / fast_task_dir_name(name) / f"samples_{name}.json"
                audit["tasks"][name] = audit_response_file(path)
            if args.limit and any(v["empty_responses"] for v in audit["tasks"].values()):
                raise RuntimeError("Empty generation in smoke test; inspect raw logs. Original scores preserved.")
            audit["status"] = "artifacts_verified"
        except Exception as exc:
            audit["status"] = "failed"
            failure[0] = f"{type(exc).__name__}: {exc}"
            audit["error"] = failure[0]
        write_json(output / "artifact_audit.json", audit)
    broadcast_object_list(failure)
    if failure[0]:
        raise RuntimeError(failure[0])


def add_arguments(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", required=True, help="Explicit path/revision/coverage manifest")
    parser.add_argument("--output", required=True, help="New output directory; no leaderboard writes")
    parser.add_argument("--tasks", help="Comma-separated frozen task names; default all 13 tasks / 9 benchmarks")
    parser.add_argument("--judge-model")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-max-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-data", action="store_true")
