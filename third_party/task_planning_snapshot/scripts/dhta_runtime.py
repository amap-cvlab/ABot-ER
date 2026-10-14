#!/usr/bin/env python
"""Inference-only DHTA preparation for the fixed Task-Planning route.

DHTA adds two low-rank residual families to selected Qwen3.5 attention Q/V
projections.  Every request in this repository belongs to the Task-Planning
family, so the selected residual is linear and can be folded exactly into the
base projection before vLLM tensor-parallel loading::

    W_task_plan = W + (alpha / rank) * B_task_plan @ A_task_plan

The source must have an enabled public or legacy task-conditioned schema.
Unconditioned checkpoints are rejected rather than evaluated as released models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DHTA_QA_TOKEN = "[2D_QA]"
DHTA_TASK_PLAN_TOKEN = "[TASK_PLAN]"
DHTA_ENV = "TP_DHTA_TASK_PLAN"
DHTA_MERGE_MARKER = ".dhta_vllm_task_plan_merged.json"
_ROUTE_LOGGED = False


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _parse_layers(value: str | list[int]) -> list[int]:
    if isinstance(value, str):
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        layers = [int(item) for item in value]
    if not layers or len(layers) != len(set(layers)) or min(layers) < 0:
        raise ValueError(f"Invalid DHTA layer selection: {value!r}")
    return sorted(layers)


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    families = {
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
        "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
    }
    if config.get("model_type") not in families:
        raise ValueError(
            f"Unsupported DHTA model_type: {config.get('model_type')!r}"
        )
    architectures = config.get("architectures") or []
    if families[config["model_type"]] not in architectures and not (
        config.get("abot_residual_namespace") == "task_conditioned"
        and architectures == ["ABotERForConditionalGeneration"]
    ):
        raise ValueError(f"Unsupported DHTA architecture list: {architectures!r}")
    layers = _parse_layers(config["dhta_layers"])
    rank = int(config["dhta_rank"])
    alpha = float(config["dhta_alpha"])
    if rank <= 0:
        raise ValueError(f"DHTA rank must be positive, got {rank}")
    if config.get("dhta_qa_token") != DHTA_QA_TOKEN:
        raise ValueError(f"Unsupported DHTA QA token: {config.get('dhta_qa_token')!r}")
    if config.get("dhta_task_plan_token") != DHTA_TASK_PLAN_TOKEN:
        raise ValueError(
            f"Unsupported DHTA Task-Planning token: {config.get('dhta_task_plan_token')!r}"
        )
    if not bool(config.get("dhta_require_route_token", True)):
        raise ValueError("DHTA checkpoint must require an explicit route token")
    qa_id = int(config["dhta_qa_token_id"])
    plan_id = int(config["dhta_task_plan_token_id"])
    if qa_id == plan_id:
        raise ValueError("DHTA route tokens resolve to the same token id")
    return {
        "layers": layers,
        "rank": rank,
        "alpha": alpha,
        "qa_token_id": qa_id,
        "task_plan_token_id": plan_id,
        "namespace": config.get("abot_residual_namespace", "dhta"),
    }


def _validate_tokenizer(model_path: Path, config_info: dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    expected = (
        (DHTA_QA_TOKEN, config_info["qa_token_id"]),
        (DHTA_TASK_PLAN_TOKEN, config_info["task_plan_token_id"]),
    )
    for token, expected_id in expected:
        actual = tokenizer.encode(token, add_special_tokens=False)
        if actual != [expected_id]:
            raise ValueError(
                f"DHTA token must be atomic and match config: token={token!r}, "
                f"expected={[expected_id]}, actual={actual}"
            )


def _load_weight_map(model_path: Path) -> tuple[Path | None, dict[str, str]]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid safetensors index: {index_path}")
        if any(not isinstance(value, str) or Path(value).name != value or not value.endswith(".safetensors")
               for value in weight_map.values()):
            raise ValueError("Safetensors index contains unsafe filenames")
        return index_path, dict(weight_map)
    single = model_path / "model.safetensors"
    if not single.is_file():
        raise RuntimeError("DHTA checkpoint must use model.safetensors or a sharded safetensors index")
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        return None, {key: single.name for key in handle.keys()}


def _expected_dhta_keys(layers: list[int], namespace: str = "dhta") -> set[str]:
    keys: set[str] = set()
    for layer in layers:
        for projection in ("q_proj", "v_proj"):
            prefix = f"model.language_model.layers.{layer}.self_attn.{projection}"
            for route in ("qa", "task_plan"):
                for side in ("a", "b"):
                    keys.add(f"{prefix}.{namespace}.{route}_{side}.weight")
    return keys


def merge_dhta_task_plan_weights(model_path: Path, config_info: dict[str, Any]) -> dict[str, Any]:
    """Fold the Task-Planning residual into Q/V weights and remove DHTA tensors."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    marker = model_path / DHTA_MERGE_MARKER
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (payload.get("route") != "task_plan" or any(payload.get(key) != config_info[key]
                for key in ("layers", "rank", "alpha", "qa_token_id", "task_plan_token_id", "namespace"))):
            raise RuntimeError(f"DHTA merge marker does not match checkpoint config: {marker}")
        return payload

    index_path, weight_map = _load_weight_map(model_path)
    from abot_er.checkpoint_contract import residual_weight_key
    namespace = config_info.get("namespace", "dhta")
    expected_dhta = _expected_dhta_keys(config_info["layers"], namespace)
    actual_dhta = {key for key in weight_map if residual_weight_key(key)}
    if actual_dhta != expected_dhta:
        missing = sorted(expected_dhta - actual_dhta)
        unexpected = sorted(actual_dhta - expected_dhta)
        raise RuntimeError(
            f"DHTA checkpoint tensor mismatch: missing={missing}, unexpected={unexpected}"
        )

    base_keys: list[str] = []
    for layer in config_info["layers"]:
        for projection in ("q_proj", "v_proj"):
            key = f"model.language_model.layers.{layer}.self_attn.{projection}.weight"
            if key not in weight_map:
                raise RuntimeError(f"DHTA base projection is missing: {key}")
            base_keys.append(key)

    dhta_tensors: dict[str, torch.Tensor] = {}
    for key in sorted(expected_dhta):
        path = model_path / weight_map[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            dhta_tensors[key] = handle.get_tensor(key)

    affected_files = sorted({weight_map[key] for key in expected_dhta.union(base_keys)})
    removed_bytes = sum(tensor.numel() * tensor.element_size() for tensor in dhta_tensors.values())
    scale = config_info["alpha"] / config_info["rank"]
    merged_shapes: dict[str, list[int]] = {}
    for file_name in affected_files:
        path = model_path / file_name
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for base_key in base_keys:
            if weight_map[base_key] != file_name:
                continue
            prefix = base_key.removesuffix(".weight")
            a = dhta_tensors[f"{prefix}.{namespace}.task_plan_a.weight"]
            b = dhta_tensors[f"{prefix}.{namespace}.task_plan_b.weight"]
            base = tensors[base_key]
            if a.shape != (config_info["rank"], base.shape[1]):
                raise RuntimeError(f"DHTA A shape mismatch for {base_key}: A={tuple(a.shape)} base={tuple(base.shape)}")
            if b.shape != (base.shape[0], config_info["rank"]):
                raise RuntimeError(f"DHTA B shape mismatch for {base_key}: B={tuple(b.shape)} base={tuple(base.shape)}")
            delta = torch.matmul(b.float(), a.float()).mul_(scale)
            tensors[base_key] = base.add(delta.to(dtype=base.dtype)).contiguous()
            merged_shapes[base_key] = list(base.shape)
        for key in expected_dhta:
            if weight_map[key] == file_name:
                tensors.pop(key, None)
        tmp = path.with_name(path.name + ".dhta-tmp")
        save_file(tensors, tmp, metadata=metadata)
        tmp.replace(path)

    if index_path is not None:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        for key in expected_dhta:
            index_payload["weight_map"].pop(key, None)
        metadata = index_payload.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
            metadata["total_size"] = max(0, metadata["total_size"] - removed_bytes)
        _atomic_json(index_path, index_payload)

    payload = {
        "enabled": True,
        "route": "task_plan",
        "route_token": DHTA_TASK_PLAN_TOKEN,
        "qa_token_id": config_info["qa_token_id"],
        "task_plan_token_id": config_info["task_plan_token_id"],
        "layers": config_info["layers"],
        "rank": config_info["rank"],
        "alpha": config_info["alpha"],
        "namespace": namespace,
        "merged_projection_count": len(base_keys),
        "removed_dhta_tensor_count": len(expected_dhta),
        "removed_dhta_bytes": removed_bytes,
        "affected_files": affected_files,
        "merged_shapes": merged_shapes,
    }
    _atomic_json(marker, payload)
    return payload


def prepare_dhta_task_plan_model(model_path: Path) -> dict[str, Any]:
    """Validate and convert a staged checkpoint when DHTA is enabled."""
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from abot_er.checkpoint_contract import normalize_checkpoint_config
    if normalize_checkpoint_config(config) is None:
        if (model_path / DHTA_MERGE_MARKER).exists():
            raise ValueError("Already merged TP copy; use its server manifest, or stage fresh source weights")
        raise ValueError("Evaluation requires a task-conditioned ABot-ER checkpoint")
    config_info = _validate_config(config)
    _validate_tokenizer(model_path, config_info)
    payload = merge_dhta_task_plan_weights(model_path, config_info)
    # This is a private TP-only copy. Its residual has been merged into W;
    # presenting it as an adaptive model would double-apply or reinitialize it.
    for target in (config, config.get("text_config", {})):
        for key in list(target):
            if key.startswith(("dhta_", "task_conditioned_", "abot_residual_")):
                del target[key]
    config.pop("auto_map", None)
    config["architectures"] = [{
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
        "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
    }[config["model_type"]]]
    _atomic_json(config_path, config)
    os.environ[DHTA_ENV] = "1"
    return payload


def inject_dhta_task_plan_route(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inject exactly one Task-Planning token before text/image placeholders."""
    global _ROUTE_LOGGED
    if os.environ.get(DHTA_ENV) != "1":
        return messages
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            segments = [(message, "content")]
        elif isinstance(content, list):
            segments = [
                (item, "text")
                for item in content
                if isinstance(item, dict)
                and item.get("type") in {"text", "input_text"}
                and isinstance(item.get("text"), str)
            ]
        else:
            raise TypeError("First DHTA user content must be a string or multimodal content list")
        qa_count = sum(holder[key].count(DHTA_QA_TOKEN) for holder, key in segments)
        plan_count = sum(holder[key].count(DHTA_TASK_PLAN_TOKEN) for holder, key in segments)
        if qa_count + plan_count == 0:
            if isinstance(content, list):
                content.insert(0, {"type": "text", "text": DHTA_TASK_PLAN_TOKEN + "\n"})
            else:
                message["content"] = DHTA_TASK_PLAN_TOKEN + "\n" + content.lstrip()
        elif qa_count == 0 and plan_count == 1:
            pass
        else:
            raise ValueError(
                f"Invalid DHTA Task-Planning route: qa_count={qa_count}, plan_count={plan_count}"
            )
        if not _ROUTE_LOGGED:
            print(
                json.dumps(
                    {
                        "message": "dhta_task_plan_route_verified",
                        "route_token": DHTA_TASK_PLAN_TOKEN,
                        "multimodal": isinstance(content, list),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _ROUTE_LOGGED = True
        return messages
    raise ValueError("No user message found for DHTA Task-Planning route injection")
