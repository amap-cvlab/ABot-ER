"""Stage the frozen interactive TP evaluators without modifying source assets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .reproduction import ROOT, read_json, sha256, write_json

SNAPSHOT = ROOT / "third_party/task_planning_snapshot"
LINKS = {
    "et_benchmarks": "ET-Plan-Bench/benchmark",
    "et_simulator": "ET-Plan-Bench/simulator",
    "alfred_data": "EmbodiedBench/embodiedbench/envs/eb_alfred/data",
    "habitat_data": "EmbodiedBench/embodiedbench/envs/eb_habitat/data",
    "habitat_datasets": "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets",
    "navigation_datasets": "EmbodiedBench/embodiedbench/envs/eb_navigation/datasets",
}
ENV_KEYS = {
    "et_simulator": "TP_ET_SIMULATOR_ROOT", "ai2thor_releases": "TP_AI2THOR_RELEASES_ROOT",
    "alfred_data": "TP_EB_ALFRED_DATA_ROOT", "habitat_data": "TP_EB_HABITAT_DATA_ROOT",
    "habitat_datasets": "TP_EB_HABITAT_DATASETS_ROOT",
    "navigation_datasets": "TP_EB_NAVIGATION_DATASETS_ROOT",
}


def validate_assets(manifest):
    if not manifest.get("revision"):
        raise ValueError("An immutable benchmark asset revision is required")
    paths = manifest.get("paths", {})
    if set(paths) != set(LINKS) | {"ai2thor_releases"}:
        raise ValueError(f"Exactly these asset paths are required: {sorted(set(LINKS) | {'ai2thor_releases'})}")
    result = {}
    for key, value in paths.items():
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ValueError(f"Existing absolute directory required for {key}")
        result[key] = str(path.resolve())
    return result


def prepare(args):
    paths = validate_assets(read_json(args.assets))
    work = Path(args.output).resolve()
    if work.exists():
        raise FileExistsError("Use a new private TP runtime directory")
    for value in paths.values():
        if work.is_relative_to(Path(value)) or Path(value).is_relative_to(work):
            raise ValueError("Runtime and immutable assets must not contain one another")
    source = work / "source"
    shutil.copytree(SNAPSHOT, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"))
    for key, relative in LINKS.items():
        destination = source / relative
        if destination.exists():
            raise FileExistsError(f"Unexpected data embedded in code snapshot: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(paths[key], target_is_directory=True)
    env = {name: paths[key] for key, name in ENV_KEYS.items()}
    env.update(TP_NAS_PROJECT_ROOT=str(source), TP_OUTPUT_ROOT=str(work / "results"),
               TP_LOCAL_SCRATCH_ROOT=str(work / "scratch"), NOTEBOOK_CONTAINER="1")
    record = {"schema_version": 1, "source": str(source), "environment": env,
              "assets": read_json(args.assets), "asset_manifest_sha256": sha256(Path(args.assets)),
              "code_sha256": {str(p.relative_to(SNAPSHOT)): sha256(p)
                              for p in sorted(SNAPSHOT.rglob("*")) if p.is_file()
                              and p.suffix in {".py", ".yaml", ".jinja"}},
              "validation": "staged only; simulator and serving smoke tests required",
              "coverage_expected": {"ET-Plan": 2023, "EB-ALFRED": 300, "EB-Habitat": 300}}
    write_json(work / "runtime.json", record)
    print(json.dumps({"runtime": str(work / "runtime.json"), "status": "staged"}, indent=2))


def prepare_model(args):
    source, output = Path(args.model).resolve(), Path(args.output).resolve()
    if not (source / "config.json").is_file():
        raise ValueError("A complete local HF checkpoint is required")
    from .checkpoint_contract import require_task_conditioned_checkpoint
    require_task_conditioned_checkpoint(read_json(source / "config.json"))
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a new staging directory disjoint from original weights")
    files = [p for p in sorted(source.iterdir()) if p.is_file()
             and (p.suffix in {".safetensors", ".json", ".jinja", ".model", ".txt"})
             and not p.name.startswith(".")]
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Safetensors weights are required")
    output.mkdir(parents=True)
    before = {}
    for path in files:
        before[path.name] = sha256(path)
        # Real copies, never hardlinks: upstream route merging modifies files.
        shutil.copy2(path, output / path.name)
        if sha256(output / path.name) != before[path.name]:
            raise ValueError(f"Checkpoint copy failed verification: {path.name}")
    sys.path.insert(0, str(SNAPSHOT / "scripts"))
    from dhta_runtime import prepare_dhta_task_plan_model
    from qwen_dhta_eval_profile import historical_dhta_profile
    profile = historical_dhta_profile(source)
    if profile:
        shutil.copy2(profile["chat_template"], output / "abot_historical_chat_template.jinja")
        profile = {**profile, "chat_template": str(output / "abot_historical_chat_template.jinja")}
    result = prepare_dhta_task_plan_model(output)
    for path in files:
        if sha256(path) != before[path.name]:
            raise RuntimeError("Original checkpoint changed during staging")
    record = {"source": str(source), "model_path": str(output), "source_sha256": before,
              "prepared_sha256": {p.name: sha256(p) for p in sorted(output.iterdir())
                                   if p.is_file() and p.suffix in {".safetensors", ".json", ".jinja", ".model", ".txt"}},
              "dhta_enabled": True,
              "route": "[TASK_PLAN]", "preparation": result, "historical_profile": profile,
              "warning": "TP-only merged copy; never use for EQA or training"}
    write_json(output / "abot_tp_server_manifest.json", record)
    print(json.dumps({"model_path": str(output), "manifest": str(output / "abot_tp_server_manifest.json")}, indent=2))


def serve(args):
    """Preserve the official server's TP route/template policy on local weights."""
    server = read_json(args.server_manifest)
    model = Path(server["model_path"]).resolve()
    if (server.get("dhta_enabled") is not True or server.get("route") != "[TASK_PLAN]"
            or not server.get("prepared_sha256")):
        raise ValueError("Verified TP-only staging manifest required; prepare the model first")
    if args.tensor_parallel_size < 1 or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("Invalid serving GPU configuration")
    for name, expected in server["prepared_sha256"].items():
        if Path(name).name != name or sha256(model / name) != expected:
            raise ValueError(f"Staged TP checkpoint changed: {name}")
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "--served-model-name", args.served_model_name, "--dtype", "bfloat16",
               "--max-model-len", "8192", "--tensor-parallel-size", str(args.tensor_parallel_size),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    command += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": True})]
    profile = server.get("historical_profile")
    if profile:
        if sha256(Path(profile["chat_template"])) != profile["chat_template_sha256"]:
            raise ValueError("Historical TP template changed")
        command += ["--chat-template", profile["chat_template"], "--generation-config", str(model),
                    "--override-generation-config", json.dumps(profile["generation_defaults"])]
    record = {"command": command, "server_manifest_sha256": sha256(Path(args.server_manifest)),
              "route": server["route"], "dhta_enabled": server["dhta_enabled"],
              "note": "TP thinking-on matches the original launcher; EQA thinking-off is independent"}
    print(json.dumps(record, indent=2), flush=True)
    if args.dry_run:
        return record
    env = os.environ.copy()
    env.update(NOTEBOOK_CONTAINER="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    # Respect the caller's assigned devices; never expand a scheduler GPU mask.
    subprocess.run(command, env=env, check=True)


def run(args):
    if args.workers < 1:
        raise ValueError("workers must be positive")
    runtime = read_json(args.runtime)
    source = Path(runtime["source"])
    for relative, expected in runtime["code_sha256"].items():
        if sha256(source / relative) != expected:
            raise ValueError(f"Staged evaluation code changed: {relative}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.job_name):
        raise ValueError("Unsafe job name")
    server = read_json(args.server_manifest)
    if server.get("dhta_enabled") is not True or server.get("route") != "[TASK_PLAN]":
        raise ValueError("TP-specific model staging manifest required")
    env = os.environ.copy()
    env.update(runtime["environment"])
    env["TP_DHTA_TASK_PLAN"] = "1"
    env["ET_PLAN_PLANNER_MODE"] = "official-agent"
    output = Path(env["TP_OUTPUT_ROOT"]) / args.job_name
    if output.exists():
        raise FileExistsError("Use a new TP job name; implicit reuse is disabled")
    command = [sys.executable, "scripts/run_tp_full_eval_pipeline.py", "--job-name", args.job_name,
               "--model", args.served_model_name, "--api-base", args.api_base, "--api-key", "EMPTY",
               "--suite", args.suite, "--workers", str(args.workers), "--retry-workers", str(args.workers),
               "--alfred-resolution", "300", "--habitat-resolution", "256",
               "--eb-max-completion-tokens", "512"]
    if args.max_jobs is not None:
        if args.max_jobs <= 0: raise ValueError("max-jobs must be positive")
        command += ["--max-jobs", str(args.max_jobs), "--skip-analysis"]
    record = {"command": command, "cwd": str(source), "environment": runtime["environment"],
              "route_enabled": server["dhta_enabled"], "server_manifest_sha256": sha256(Path(args.server_manifest)),
              "mode": "smoke" if args.max_jobs else "full", "no_automatic_artifact_deletion": True}
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    output.mkdir(parents=True)
    write_json(output / "reproduction_manifest.json", record)
    subprocess.run(command, cwd=source, env=env, check=True)


def add_commands(sub):
    p = sub.add_parser("prepare-tp")
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(handler=prepare)
    p = sub.add_parser("prepare-tp-model")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(handler=prepare_model)
    p = sub.add_parser("serve-tp")
    p.add_argument("--server-manifest", required=True)
    p.add_argument("--served-model-name", default="abot-er-tp")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=.9)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(handler=serve)
    p = sub.add_parser("evaluate-tp")
    p.add_argument("--runtime", required=True)
    p.add_argument("--server-manifest", required=True)
    p.add_argument("--job-name", required=True)
    p.add_argument("--served-model-name", default="abot-er-tp")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--suite", choices=("all", "et", "alfred", "habitat"), default="all")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(handler=run)
