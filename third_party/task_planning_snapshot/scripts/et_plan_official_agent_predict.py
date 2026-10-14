#!/usr/bin/env python
"""Run an ET-Plan-Bench official-style interactive agent for one task.

This is an adapter around the upstream ET-Plan-Bench exploration code. The
adapter handles only project-local concerns: OpenAI-compatible client injection,
single-task benchmark files, and prediction JSON normalization. It does not ask
the model to generate a full VirtualHome script directly.
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from openai import OpenAI

try:
    from task_planning_io import atomic_write_json
except ModuleNotFoundError:  # Imported as scripts.et_plan_official_agent_predict in tests.
    from scripts.task_planning_io import atomic_write_json


os.environ["NOTEBOOK_CONTAINER"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ET_ROOT = PROJECT_ROOT / "ET-Plan-Bench"
NAV_TASKS_ROOT = ET_ROOT / "evaluation_generation" / "navigation_tasks"
EMBODIED_AI_ROOT = (
    ET_ROOT
    / "evaluation_generation"
    / "navigation_manipulation_tasks"
    / "EmbodiedAI"
)
MULTI_OBJECT_ROOT = ET_ROOT / "evaluation_generation" / "navigation_manipulation_multiple_objects"
VH_SIMULATION_ROOT = ET_ROOT / "simulator" / "virtualhome" / "virtualhome" / "simulation"
ROOM_NAMES = {"bathroom", "bedroom", "kitchen", "livingroom"}
INIT_ROOM_POOL = ["bathroom", "bedroom", "kitchen", "livingroom"]
OFFICIAL_TOP_LEVEL_MODULES = {
    "constant",
    "envs",
    "tasks",
    "task_generation_codes",
    "utils",
}


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def purge_official_namespace_modules() -> None:
    """Avoid cross-family imports through ET-Plan-Bench's repeated package names."""
    for name, module in list(sys.modules.items()):
        top_name = name.split(".", 1)[0]
        if top_name not in OFFICIAL_TOP_LEVEL_MODULES:
            continue
        module_file = getattr(module, "__file__", None)
        module_paths = [Path(item) for item in getattr(module, "__path__", [])]
        if (module_file is not None and is_under(Path(module_file), ET_ROOT)) or any(
            is_under(item, ET_ROOT) for item in module_paths
        ):
            sys.modules.pop(name, None)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_conditions(condition_text: str) -> list[tuple[str, str, str]]:
    conditions: list[tuple[str, str, str]] = []
    for item in re.findall(r"\(([^)]*)\)", condition_text or ""):
        parts = [part.strip().lower() for part in item.split(",")]
        if len(parts) == 3:
            conditions.append((parts[0].upper(), parts[1], parts[2]))
    return conditions


def primary_navigation_object(task: dict[str, Any]) -> str:
    if isinstance(task.get("object"), str):
        return task["object"].strip().lower()
    for relation, subject, obj in parse_conditions(task.get("task_completion_criterion", "")):
        if relation == "CLOSE" and subject == "robot":
            return obj
    raise ValueError("could not infer navigation target object from task criterion")


def spatial_navigation_objects(task: dict[str, Any]) -> tuple[str, str, str]:
    conditions = parse_conditions(task.get("task_completion_criterion", ""))
    object_1 = primary_navigation_object(task)
    for relation, subject, obj in conditions:
        if relation != "CLOSE" and subject == object_1:
            return object_1, obj, relation
    raise ValueError("could not infer spatial navigation relation from task criterion")


def single_manipulation_spec(task: dict[str, Any]) -> dict[str, str]:
    close_objects = [
        obj
        for relation, subject, obj in parse_conditions(task.get("task_completion_criterion", ""))
        if relation == "CLOSE" and subject == "robot"
    ]
    if len(close_objects) < 2:
        raise ValueError("could not infer navigation-manipulation objects from CLOSE criteria")
    object_1 = close_objects[0]
    object_2 = close_objects[1]
    placement_relation = None
    object_constraint = None
    container_constraint = None
    for relation, subject, obj in parse_conditions(task.get("task_completion_criterion", "")):
        if relation == "CLOSE":
            continue
        if subject == object_1 and obj == object_2:
            placement_relation = relation
        elif subject == object_1:
            object_constraint = (relation, obj)
        elif subject == object_2:
            container_constraint = (relation, obj)

    if placement_relation not in {"ON", "INSIDE"}:
        raise ValueError("could not infer ON/INSIDE placement relation")

    task_name = "nav_on" if placement_relation == "ON" else "nav_inside"
    spec = {
        "object1": object_1,
        "object2": object_2,
        "task_name": task_name,
        "task_type": "on" if placement_relation == "ON" else "in",
    }
    if object_constraint and container_constraint:
        spec.update(
            {
                "object3": object_constraint[1],
                "rel1": object_constraint[0],
                "object4": container_constraint[1],
                "rel2": container_constraint[0],
                "task_name": task_name + "_cons",
            }
        )
    return spec


def multi_object_spec(task: dict[str, Any], *, temporal: bool = False) -> dict[str, str]:
    relations = [
        (relation, subject, obj)
        for relation, subject, obj in parse_conditions(task.get("task_completion_criterion", ""))
        if relation != "CLOSE"
    ]
    if len(relations) < 2:
        raise ValueError("could not infer multi-object placement relations")

    if temporal:
        relation_1, object_1, object_2 = relations[0]
        relation_2, subject_2, object_3 = relations[1]
        if subject_2 != object_2:
            raise ValueError("temporal task does not form object1 -> object2 -> object3 chain")
        return {
            "object1": object_1,
            "object2": object_2,
            "object3": object_3,
            "relation1": relation_1,
            "relation2": relation_2,
        }

    destination_counts: dict[str, int] = {}
    for _, _, obj in relations:
        destination_counts[obj] = destination_counts.get(obj, 0) + 1
    object_3 = max(destination_counts, key=destination_counts.get)
    moved = [subject for _, subject, obj in relations if obj == object_3]
    if len(moved) < 2:
        raise ValueError("could not infer two moved objects and common destination")
    placement_relation = relations[0][0]
    return {
        "object1": moved[0],
        "object2": moved[1],
        "object3": object_3,
        "relation": placement_relation,
    }


def load_official_module(path: Path, module_name: str, api_key: str, temp_root: Path) -> ModuleType:
    purge_official_namespace_modules()
    sys.path.insert(0, str(ET_ROOT))
    sys.path.insert(0, str(VH_SIMULATION_ROOT))
    sys.path.insert(0, str(NAV_TASKS_ROOT))
    sys.path.insert(0, str(NAV_TASKS_ROOT.parent))
    old_cwd = Path.cwd()
    # The upstream module can start Unity while it is imported. Unity may keep
    # touching its working directory briefly after the import returns, which
    # makes TemporaryDirectory cleanup race with the simulator on NAS. Keep the
    # tiny import directory on container-local storage for the process lifetime.
    local_temp_root = Path(os.environ.get("ET_PLAN_MODULE_TMPDIR", tempfile.gettempdir()))
    local_temp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix="et_plan_official_module_", dir=local_temp_root))
    atexit.register(shutil.rmtree, tmp_path, ignore_errors=True)
    (tmp_path / "GPT_API_KEY.txt").write_text(api_key, encoding="utf-8")
    os.chdir(tmp_path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load official module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def load_official_module_with_argv(
    path: Path,
    module_name: str,
    argv: list[str],
    search_paths: list[Path],
) -> ModuleType:
    purge_official_namespace_modules()
    sys.path.insert(0, str(VH_SIMULATION_ROOT))
    sys.path.insert(0, str(ET_ROOT))
    for search_path in reversed(search_paths):
        sys.path.insert(0, str(search_path))
    old_argv = sys.argv[:]
    sys.argv = [str(path), *argv]
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load official module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


def make_official_chat_func(client: OpenAI, model: str):
    def gpt4v_func(prompt, model=model, max_tokens=2048, user_role=True):
        token_cap = os.environ.get("TP_GENERATION_TOKEN_CAP")
        if token_cap:
            max_tokens = min(max_tokens, int(token_cap))
        messages = [{"role": "user", "content": prompt}] if user_role else prompt
        try:
            from dhta_runtime import inject_dhta_task_plan_route
        except ModuleNotFoundError:
            from scripts.dhta_runtime import inject_dhta_task_plan_route
        messages = inject_dhta_task_plan_route(messages)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )

    return gpt4v_func


def make_official_text_func(client: OpenAI, model: str):
    def query_gpt_single(prompt, max_tokens=2048, temperature=None, user_role=True, full_response=False):
        token_cap = os.environ.get("TP_GENERATION_TOKEN_CAP")
        if token_cap:
            max_tokens = min(max_tokens, int(token_cap))
        messages = [{"role": "user", "content": prompt}] if user_role else prompt
        try:
            from dhta_runtime import inject_dhta_task_plan_route
        except ModuleNotFoundError:
            from scripts.dhta_runtime import inject_dhta_task_plan_route
        messages = inject_dhta_task_plan_route(messages)
        request = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0 if temperature is None else temperature,
        }
        response = client.chat.completions.create(
            **request,
        )
        if full_response:
            return response
        return response.choices[0].message.content

    return query_gpt_single


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def patch_text_llm_modules(client: OpenAI, model: str) -> None:
    query_fn = make_official_text_func(client, model)
    for name in ("utils.gpt_call", "utils.robot"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "query_gpt_single"):
            module.query_gpt_single = query_fn


def navigation_context_benchmark(
    task: dict[str, Any],
    output_dir: Path,
    *,
    spatial: bool,
    port: int,
) -> Path:
    item = dict(task)
    if spatial:
        object_1, object_2, relation = spatial_navigation_objects(item)
        item["object_1"] = object_1
        item["object_2"] = object_2
        if object_2 not in ROOM_NAMES:
            id_1, id_2 = infer_spatial_instance_ids(
                item,
                object_1,
                object_2,
                relation,
                output_dir=output_dir,
                port=port,
            )
            item["instant_id_1"] = id_1
            item["instant_id_2"] = id_2
    else:
        target_object = primary_navigation_object(item)
        item["object"] = target_object
        item["object_1"] = target_object
        item["object_2"] = ""

    path = output_dir / "official_agent_single_task.jsonl"
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def single_manipulation_task(task: dict[str, Any], task_id: int) -> tuple[dict[str, Any], str]:
    spec = single_manipulation_spec(task)
    official_task = {
        **task,
        "index": task_id,
        "object1": spec["object1"],
        "object2": spec["object2"],
    }
    if spec["task_name"].endswith("_cons"):
        official_task.update(
            {
                "object3": spec["object3"],
                "rel1": spec["rel1"],
                "object4": spec["object4"],
                "rel2": spec["rel2"],
            }
        )
    return official_task, spec["task_name"]


def multi_object_task(task: dict[str, Any], task_id: int, *, temporal: bool = False) -> tuple[dict[str, Any], str]:
    spec = multi_object_spec(task, temporal=temporal)
    official_task = {
        **task,
        "index": task_id,
        "object1": spec["object1"],
        "object2": spec["object2"],
        "object3": spec["object3"],
    }
    if temporal:
        return official_task, "temporal"
    relation = spec["relation"]
    if relation == "INSIDE":
        return official_task, "inside"
    if relation == "ON":
        return official_task, "on"
    raise ValueError(f"unsupported multi-object relation: {relation}")


def write_env_task_file(task: dict[str, Any], save_task_folder: Path) -> Path:
    env_id = int(task["env_id"])
    save_task_folder.mkdir(parents=True, exist_ok=True)
    path = save_task_folder / f"env_{env_id}.json"
    path.write_text(json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def collect_navigation_graph(
    benchmark: Path,
    task: dict[str, Any],
    task_id: int,
    port: int,
) -> dict[str, Any]:
    """Collect the full VirtualHome graph required by the official spatial adapter."""
    sys.path.insert(0, str(VH_SIMULATION_ROOT))
    sys.path.insert(0, str(NAV_TASKS_ROOT))
    from envs.env import VirtualHomeNavigationEnv

    os.environ.setdefault("ET_PLAN_SKIP_START_UNITY", "1")
    os.environ.setdefault("ET_PLAN_NO_GRAPHICS", "1")
    os.environ.setdefault("ET_PLAN_TIMEOUT_WAIT", "60")
    runtime_dir = Path(os.environ["ET_PLAN_RUNTIME_DIR"])
    runtime_dir.mkdir(parents=True, exist_ok=True)

    env = VirtualHomeNavigationEnv(port=str(port), input_data_file=str(benchmark))
    try:
        obs = env.reset_CA(env_id=int(task["env_id"]), task_id=task_id)
    finally:
        env.close()
    return obs["full_graph"]


def infer_spatial_instance_ids(
    task: dict[str, Any],
    object_1: str,
    object_2: str,
    relation: str,
    *,
    output_dir: Path,
    port: int,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_benchmark = output_dir / f"spatial_id_probe_{os.getpid()}.jsonl"
    tmp_item = dict(task)
    tmp_item.setdefault("object_1", object_1)
    tmp_item.setdefault("object_2", object_2)
    tmp_benchmark.write_text(json.dumps(tmp_item, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        graph = collect_navigation_graph(tmp_benchmark, tmp_item, 0, port)
    finally:
        tmp_benchmark.unlink(missing_ok=True)

    nodes = {int(node["id"]): node for node in graph["nodes"]}
    for edge in graph["edges"]:
        from_node = nodes.get(int(edge["from_id"]))
        to_node = nodes.get(int(edge["to_id"]))
        if not from_node or not to_node:
            continue
        if (
            str(edge.get("relation_type", "")).upper() == relation
            and str(from_node.get("class_name", "")).lower() == object_1
            and str(to_node.get("class_name", "")).lower() == object_2
        ):
            return int(edge["from_id"]), int(edge["to_id"])
    raise ValueError(f"could not infer instance ids for {relation}({object_1}, {object_2})")


def normalize_prediction(
    *,
    benchmark: Path,
    task: dict[str, Any],
    task_id: int,
    model: str,
    output_dir: Path,
    official_raw_path: Path,
    action_history: list[str],
    success: bool | None,
    official_agent_error: str | None = None,
) -> Path:
    output_path = output_dir / f"prediction_task{task_id}.json"
    payload = {
        "benchmark": str(benchmark),
        "task_id": task_id,
        "model": model,
        "planner_mode": "official-agent",
        "official_raw_output": str(official_raw_path),
        "task": task["task"],
        "task_completion_criterion": task.get("task_completion_criterion"),
        "official_agent_success": success,
        "official_agent_error": official_agent_error,
        "action_history": action_history,
    }
    atomic_write_json(output_path, payload)
    return output_path


def run_navigation_official_agent(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark).resolve()
    tasks = load_jsonl(benchmark)
    task = tasks[args.task_id]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    official_raw_dir = output_dir / "official_raw"
    official_raw_dir.mkdir(parents=True, exist_ok=True)

    spatial = benchmark.name == "navigation_tasks_with_spatial_constraint.json"
    temp_benchmark = navigation_context_benchmark(task, output_dir, spatial=spatial, port=args.port)
    api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = make_client()
    official_chat = make_official_chat_func(client, args.model)

    if spatial:
        module = load_official_module(
            NAV_TASKS_ROOT / "gpt4_navigation_with_spatial_constraint.py",
            "et_plan_official_navigation_spatial",
            api_key,
            output_dir,
        )
        module.gpt4v_func = official_chat
        imported_nav = sys.modules.get("evaluation_generation.navigation_tasks.gpt4_navigation")
        if imported_nav is not None:
            imported_nav.gpt4v_func = official_chat
    else:
        module = load_official_module(
            NAV_TASKS_ROOT / "gpt4_navigation.py",
            "et_plan_official_navigation",
            api_key,
            output_dir,
        )
        module.gpt4v_func = official_chat

    env = module.VirtualHomeNavigationEnv(port=str(args.port), input_data_file=str(temp_benchmark))
    action_history = []
    success = False
    official_agent_error = None
    try:
        try:
            if spatial:
                object_1, object_2, relation = spatial_navigation_objects(task)
                if object_2 in ROOM_NAMES:
                    action_history, success, _, _ = module.robot_exploration_without_constraint(
                        task["task"],
                        object_1,
                        args.port,
                        env,
                        str(official_raw_dir),
                        0,
                        int(task["env_id"]),
                        str(temp_benchmark),
                        only_explore_one_room=[object_2],
                    )
                else:
                    temp_task = load_jsonl(temp_benchmark)[0]
                    action_history, success, _, _ = module.robot_exploration(
                        task["task"],
                        object_1,
                        object_2,
                        args.port,
                        env,
                        str(official_raw_dir),
                        0,
                        int(temp_task["instant_id_1"]),
                        int(temp_task["instant_id_2"]),
                        relation,
                        int(task["env_id"]),
                        str(temp_benchmark),
                    )
            else:
                action_history, success, _, _ = module.robot_exploration_without_constraint(
                    task["task"],
                    primary_navigation_object(task),
                    args.port,
                    env,
                    str(official_raw_dir),
                    0,
                    int(task["env_id"]),
                    str(temp_benchmark),
                )
        except Exception as exc:
            official_agent_error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    return normalize_prediction(
        benchmark=benchmark,
        task=task,
        task_id=args.task_id,
        model=args.model,
        output_dir=output_dir,
        official_raw_path=official_raw_dir / "task_0.json",
        action_history=action_history or [],
        success=success,
        official_agent_error=official_agent_error,
    )


def run_single_manipulation_official_agent(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark).resolve()
    tasks = load_jsonl(benchmark)
    task = tasks[args.task_id]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    official_raw_dir = output_dir / "official_raw"
    official_raw_dir.mkdir(parents=True, exist_ok=True)

    official_task, task_name = single_manipulation_task(task, args.task_id)
    client = make_client()
    module = load_official_module_with_argv(
        EMBODIED_AI_ROOT / "data_gen" / "visible_single_task_type.py",
        "et_plan_official_visible_single_task_type",
        ["--task_name_id", "0"],
        [EMBODIED_AI_ROOT],
    )
    patch_text_llm_modules(client, args.model)
    module.PORT_NUM = args.port

    env = module.VirtualHomeEnv(port=str(args.port))
    try:
        success = module.nav_manipulation_single_from_task(
            task_name,
            env,
            str(official_raw_dir),
            official_task,
            model="gpt",
        )
    finally:
        env.close()

    raw_path = official_raw_dir / f"task_{args.task_id}.json"
    action_history = []
    saved_success = success
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        action_history = raw.get("action_history", [])
        saved_success = raw.get("success", success)

    return normalize_prediction(
        benchmark=benchmark,
        task=task,
        task_id=args.task_id,
        model=args.model,
        output_dir=output_dir,
        official_raw_path=raw_path,
        action_history=action_history,
        success=bool(saved_success),
    )


def run_multi_object_official_agent(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark).resolve()
    tasks = load_jsonl(benchmark)
    task = tasks[args.task_id]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    official_raw_dir = output_dir / "official_raw"
    official_raw_dir.mkdir(parents=True, exist_ok=True)
    save_task_folder = output_dir / "official_tasks"

    temporal = benchmark.name == "navigation_and_manipulation_tasks_with_temporal_constraint.json"
    official_task, relation_mode = multi_object_task(task, args.task_id, temporal=temporal)
    task_file = write_env_task_file(official_task, save_task_folder)
    env_id = int(task["env_id"])
    raw_env_dir = official_raw_dir / f"env_{env_id}"
    raw_env_dir.mkdir(parents=True, exist_ok=True)

    if temporal:
        module_file = "inference_exploration_temp.py"
    elif benchmark.name == "navi_and_mani_multi_two_arm.json":
        module_file = "inference_exploration_inside_both.py" if relation_mode == "inside" else "inference_exploration_on_both.py"
    else:
        module_file = "inference_exploration_inside_step.py" if relation_mode == "inside" else "inference_exploration_on_step.py"

    client = make_client()
    module = load_official_module_with_argv(
        MULTI_OBJECT_ROOT / module_file,
        f"et_plan_official_{module_file.replace('.', '_')}",
        [
            "--mode",
            "test",
            "--skip",
            "[]",
            "-path",
            str(official_raw_dir.parent),
            "--save_task_folder",
            str(save_task_folder),
            "--start_from",
            "0",
            "--port_num",
            str(args.port),
            "--api_key",
            os.environ.get("OPENAI_API_KEY") or "EMPTY",
        ],
        [MULTI_OBJECT_ROOT, ET_ROOT],
    )
    module.query_gpt_single = make_official_chat_func(client, args.model)

    env = module.VirtualHomeNavigationEnv(port=str(args.port), input_data_file=str(task_file))
    task_env = module.TaskEnv(env, official_task, args.task_id, str(raw_env_dir), use_partial_graph=False)
    try:
        result = module.robot_exploration_multi(
            official_task,
            args.port,
            env,
            task_env,
            str(raw_env_dir),
            args.task_id,
            env_id,
            only_explore_one_room=None,
        )
    finally:
        env.close()

    raw_path = raw_env_dir / f"task_{args.task_id}.json"
    action_history = []
    saved_success = None
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        steps = raw.get("steps", {})
        if isinstance(steps, dict):
            final = steps.get(f"STEP_{len(steps) - 1}") or next(reversed(steps.values()), {})
            if isinstance(final, dict):
                action_history = final.get("action_history", []) or raw.get("action_history", [])
                saved_success = final.get("msg")
        if not action_history:
            action_history = raw.get("action_history", [])
    if not action_history and isinstance(result, tuple) and result:
        maybe_actions = result[0]
        if isinstance(maybe_actions, list):
            action_history = maybe_actions
        if len(result) > 1:
            saved_success = result[1]

    return normalize_prediction(
        benchmark=benchmark,
        task=task,
        task_id=args.task_id,
        model=args.model,
        output_dir=output_dir,
        official_raw_path=raw_path,
        action_history=action_history,
        success=bool(saved_success),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-type", choices=["navigation", "nav_manip_single", "nav_manip_multi"], required=True)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--port", type=int, default=2522)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.task_type == "navigation":
        output_path = run_navigation_official_agent(args)
    elif args.task_type == "nav_manip_single":
        output_path = run_single_manipulation_official_agent(args)
    else:
        output_path = run_multi_object_official_agent(args)
    print(json.dumps({"prediction_file": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
