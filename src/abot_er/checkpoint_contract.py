"""Strict, dependency-free compatibility for training and published checkpoints.

Aliases are added in memory only. The tensor namespace is NOT renamed: published
weights keep ``task_conditioned``; historical weights keep ``dhta``.
"""
from __future__ import annotations

import math

FIELDS = ("enabled", "alpha", "layers", "rank", "qa_token", "qa_token_id",
          "task_plan_token", "task_plan_token_id", "require_route_token",
          "inject_during_init", "schema_version")


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _set(config, key, value):
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def normalize_checkpoint_config(config):
    """Validate both schemas and return the residual tensor namespace, or None."""
    public = _get(config, "task_conditioned_enabled")
    legacy = _get(config, "dhta_enabled")
    for value in (public, legacy):
        if value is not None and type(value) is not bool:
            raise ValueError("Residual enabled flags must be booleans")
    if public is not None and legacy is not None and public != legacy:
        raise ValueError("Conflicting public/legacy residual enabled flags")
    prefix = "task_conditioned" if public is True else "dhta" if legacy is True else None
    if prefix is None:
        if "ABotERForConditionalGeneration" in (_get(config, "architectures", []) or []):
            raise ValueError("ABotER architecture without an enabled residual schema")
        return None
    expected_schema = prefix + "-qv-v1"
    if _get(config, prefix + "_schema_version") != expected_schema:
        raise ValueError("Unsupported task-conditioned checkpoint schema")
    values = {field: _get(config, prefix + "_" + field) for field in FIELDS}
    layers = values["layers"]
    if not isinstance(layers, list) or not layers or any(type(x) is not int or x < 0 for x in layers) or len(set(layers)) != len(layers):
        raise ValueError("Invalid residual layers")
    rank, alpha = values["rank"], values["alpha"]
    if type(rank) is not int or rank <= 0 or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid residual rank/alpha")
    if values["require_route_token"] is not True:
        raise ValueError("Residual checkpoints must require explicit routing")
    for task, token in (("qa", "[2D_QA]"), ("task_plan", "[TASK_PLAN]")):
        if values[task + "_token"] != token or type(values[task + "_token_id"]) is not int or values[task + "_token_id"] < 0:
            raise ValueError("Invalid route token or token ID")
    if values["qa_token_id"] == values["task_plan_token_id"]:
        raise ValueError("Route token IDs must be distinct")
    text = _get(config, "text_config", config)
    depth = _get(text, "num_hidden_layers")
    if type(depth) is not int or max(layers) >= depth:
        raise ValueError("Residual layers exceed decoder depth")
    layer_types = _get(text, "layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or len(layer_types) != depth:
            raise ValueError("Invalid decoder layer_types length")
        if any(layer_types[layer] != "full_attention" for layer in layers):
            raise ValueError("Residuals must target full-attention layers")
    if values["inject_during_init"] is not True:
        raise ValueError("Saved residual checkpoints must enable construction-time injection")
    for target in (config, text):
        for field, value in values.items():
            # Legacy aliases are internal compatibility only, not publication edits.
            alias = "dhta-qv-v1" if field == "schema_version" else value
            old = _get(target, "dhta_" + field)
            native = _get(target, prefix + "_" + field)
            if old is not None and old != alias:
                raise ValueError(f"Conflicting residual field: dhta_{field}")
            if native is not None and native != value:
                raise ValueError(f"Conflicting top-level/text residual field: {field}")
            _set(target, "dhta_" + field, alias)
        _set(target, "abot_residual_namespace", prefix)
    return prefix


def require_task_conditioned_checkpoint(config):
    """Public evaluation accepts released task-conditioned checkpoints only.

    Normalization itself still accepts original base configurations for Stage A
    initialization and the independent judge; this guard is for the evaluated model.
    """
    namespace = normalize_checkpoint_config(config)
    if namespace is None:
        raise ValueError("Evaluation requires a task-conditioned ABot-ER checkpoint")
    return namespace


def residual_weight_key(key):
    return ".dhta." in str(key) or ".task_conditioned." in str(key)


def validate_tensor_index(config, weight_map):
    """Reject silent residual omission or mixed/incomplete tensor namespaces."""
    namespace = normalize_checkpoint_config(config)
    actual = {key for key in weight_map if residual_weight_key(key)}
    expected = set()
    if namespace:
        for layer in _get(config, "dhta_layers"):
            for projection in ("q_proj", "v_proj"):
                prefix = f"model.language_model.layers.{layer}.self_attn.{projection}"
                if prefix + ".weight" not in weight_map:
                    raise ValueError("Missing base projection in checkpoint index")
                for route in ("qa", "task_plan"):
                    for side in ("a", "b"):
                        expected.add(f"{prefix}.{namespace}.{route}_{side}.weight")
    if actual != expected:
        raise ValueError("Checkpoint index does not match the residual contract")


def validate_local_index(config, model_path):
    import json
    from pathlib import Path
    path = Path(model_path) / "model.safetensors.index.json"
    if path.is_file():
        index = json.loads(path.read_text())
        if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
            raise ValueError("Invalid checkpoint weight index")
        validate_tensor_index(config, index["weight_map"])
        if any(not isinstance(name, str) or Path(name).name != name
               or not name.endswith(".safetensors") for name in index["weight_map"].values()):
            raise ValueError("Unsafe checkpoint shard filename")
    elif (Path(model_path) / "model.safetensors").is_file():
        from safetensors import safe_open
        with safe_open(Path(model_path) / "model.safetensors", framework="pt", device="cpu") as handle:
            validate_tensor_index(config, {key: "model.safetensors" for key in handle.keys()})


def validate_loading_info(info):
    # An adapted model must never silently fall back to randomly initialized
    # residuals, or ignore an alternate residual namespace.
    bad = {name: [item for item in info.get(name, ()) if residual_weight_key(item)]
           for name in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    if any(bad.values()) or info.get("error_msgs"):
        raise RuntimeError(f"Task-conditioned weights did not load exactly: {bad}")
