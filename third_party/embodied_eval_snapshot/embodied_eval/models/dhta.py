"""Inference-only DHTA integration for Qwen3.5 checkpoints.

The implementation mirrors the training-side DHTA construction and routing
semantics.  It is deliberately activated from checkpoint config only, so
ordinary Qwen3.5 and ABotER checkpoints keep their existing loading path.
"""

from __future__ import annotations

import math
import os
import weakref
from functools import wraps
from importlib import import_module
from typing import Iterable, Sequence

import torch
from loguru import logger as eval_logger
from torch import nn


DHTA_QA_TOKEN = "[2D_QA]"
DHTA_TASK_PLAN_TOKEN = "[TASK_PLAN]"
DHTA_QA_ROUTE = 0
DHTA_TASK_PLAN_ROUTE = 1


def _parse_layers(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        layers = [int(item) for item in value]
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError(f"Invalid DHTA layer selection: {value!r}")
    return sorted(layers)


class DHTALowRankResidual(nn.Module):
    """Two task-specific low-rank residual branches."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"DHTA rank must be positive, got {rank}.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        factory_kwargs = {"device": device, "dtype": dtype}
        self.qa_a = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.qa_b = nn.Linear(self.rank, self.out_features, bias=False, **factory_kwargs)
        self.task_plan_a = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.task_plan_b = nn.Linear(self.rank, self.out_features, bias=False, **factory_kwargs)
        self._z3_leaf = True
        self.reset_parameters()
        for projection in (self.qa_a, self.qa_b, self.task_plan_a, self.task_plan_b):
            projection._is_hf_initialized = True
            projection.weight._is_hf_initialized = True
        self._is_hf_initialized = True

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.qa_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.task_plan_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.qa_b.weight)
        nn.init.zeros_(self.task_plan_b.weight)

    def _qa(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.qa_b(self.qa_a(hidden_states)) * self.scaling

    def _task_plan(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.task_plan_b(self.task_plan_a(hidden_states)) * self.scaling

    def forward(self, hidden_states: torch.Tensor, route: int | torch.Tensor) -> torch.Tensor:
        if isinstance(route, int):
            if route not in (DHTA_QA_ROUTE, DHTA_TASK_PLAN_ROUTE):
                raise ValueError(f"Unknown DHTA route: {route}.")
            qa_gate = hidden_states.new_tensor(float(route == DHTA_QA_ROUTE))
            plan_gate = hidden_states.new_tensor(float(route == DHTA_TASK_PLAN_ROUTE))
            return self._qa(hidden_states) * qa_gate + self._task_plan(hidden_states) * plan_gate
        if route.ndim != 1:
            raise ValueError(f"DHTA route must be rank-1, got shape={tuple(route.shape)}.")
        if hidden_states.shape[0] != route.shape[0]:
            if route.numel() == 1:
                return self.forward(hidden_states, int(route.item()))
            raise ValueError(
                "DHTA route batch does not match projection input: "
                f"route={tuple(route.shape)}, hidden={tuple(hidden_states.shape)}."
            )
        gate_shape = (route.shape[0],) + (1,) * (hidden_states.ndim - 1)
        qa_gate = (route == DHTA_QA_ROUTE).view(gate_shape).to(hidden_states.dtype)
        plan_gate = (route == DHTA_TASK_PLAN_ROUTE).view(gate_shape).to(hidden_states.dtype)
        if not torch.all((qa_gate + plan_gate) == 1):
            raise ValueError("DHTA route contains an unknown route value.")
        return self._qa(hidden_states) * qa_gate + self._task_plan(hidden_states) * plan_gate


class DHTARoutedLinear(nn.Linear):
    """State-dict-compatible linear projection with a routed residual."""

    def __init__(self, base_projection: nn.Linear, rank: int, alpha: float, namespace: str = "dhta") -> None:
        if not isinstance(base_projection, nn.Linear):
            raise TypeError(f"DHTA requires nn.Linear projections, got {type(base_projection)!r}.")
        nn.Module.__init__(self)
        self.in_features = int(base_projection.in_features)
        self.out_features = int(base_projection.out_features)
        self.weight = base_projection.weight
        if base_projection.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = base_projection.bias
        if namespace not in ("dhta", "task_conditioned"):
            raise ValueError("Unknown residual namespace")
        self._residual_namespace = namespace
        self.add_module(namespace, DHTALowRankResidual(
            self.in_features,
            self.out_features,
            rank,
            alpha,
            device=base_projection.weight.device,
            dtype=base_projection.weight.dtype,
        ))
        self._dhta_route = None

    def residual(self):
        return self._modules[self._residual_namespace]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._dhta_route is None:
            raise RuntimeError("DHTA route was not configured before an adapted projection ran.")
        shared = nn.functional.linear(hidden_states, self.weight, self.bias)
        residual = self.residual()(hidden_states, self._dhta_route)
        return shared + residual.to(dtype=shared.dtype)


def _iter_dhta_projections(model: nn.Module) -> Iterable[DHTARoutedLinear]:
    for module in model.modules():
        if isinstance(module, DHTARoutedLinear):
            yield module


def _set_model_route(model: nn.Module, input_ids: torch.Tensor | None, past_key_values) -> None:
    projections = [ref() for ref in getattr(model, "_dhta_projection_refs", ()) if ref() is not None]
    if not projections:
        return
    previous_route = getattr(model, "_dhta_current_route", None)
    if input_ids is None:
        if previous_route is not None and past_key_values is not None:
            return
        if bool(getattr(model.config, "dhta_require_route_token", True)):
            raise ValueError("DHTA requires input_ids containing one route token per sample.")
        return
    qa_id = int(getattr(model.config, "dhta_qa_token_id"))
    plan_id = int(getattr(model.config, "dhta_task_plan_token_id"))
    qa_hits = input_ids.eq(qa_id).sum(dim=-1)
    plan_hits = input_ids.eq(plan_id).sum(dim=-1)
    valid = (qa_hits + plan_hits).eq(1)
    if not bool(valid.all()):
        if previous_route is not None and past_key_values is not None and input_ids.shape[-1] <= 1:
            return
        if bool(getattr(model.config, "dhta_require_route_token", True)):
            bad_rows = (~valid).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Each DHTA sample must contain exactly one route token; "
                f"invalid batch rows={bad_rows}."
            )
        return
    route_tensor = plan_hits.to(dtype=torch.long)
    route: int | torch.Tensor = int(route_tensor.item()) if route_tensor.numel() == 1 else route_tensor
    model._dhta_current_route = route
    for projection in projections:
        projection._dhta_route = route


def _configure_attention(attention: nn.Module, config, layer_idx: int) -> None:
    if layer_idx not in set(_parse_layers(getattr(config, "dhta_layers"))):
        return
    if not hasattr(attention, "q_proj") or not hasattr(attention, "v_proj"):
        raise TypeError(f"DHTA layer {layer_idx} does not expose q_proj/v_proj.")
    wrapped = isinstance(attention.q_proj, DHTARoutedLinear), isinstance(attention.v_proj, DHTARoutedLinear)
    if all(wrapped):
        return
    if any(wrapped):
        raise RuntimeError(f"DHTA layer {layer_idx} is only partially wrapped: {wrapped}.")
    rank = int(getattr(config, "dhta_rank"))
    alpha = float(getattr(config, "dhta_alpha"))
    namespace = getattr(config, "abot_residual_namespace", "dhta")
    attention.q_proj = DHTARoutedLinear(attention.q_proj, rank, alpha, namespace)
    attention.v_proj = DHTARoutedLinear(attention.v_proj, rank, alpha, namespace)
    for name in ("q_proj", "v_proj"):
        projection = getattr(attention, name)
        projection.dhta_layer_idx = layer_idx
        projection.dhta_projection_name = name


def prepare_dhta_model_class(config) -> None:
    """Patch Qwen3.5 constructors before ``from_pretrained``."""
    if not bool(getattr(config, "dhta_enabled", False)):
        return
    families = {
        "qwen3_5": ("Qwen3_5Attention", "Qwen3_5ForConditionalGeneration"),
        "qwen3_5_moe": ("Qwen3_5MoeAttention", "Qwen3_5MoeForConditionalGeneration"),
    }
    family = getattr(config, "model_type", None)
    if family not in families:
        raise ValueError(f"Unsupported DHTA model_type: {family!r}.")
    modeling = import_module(f"transformers.models.{family}.modeling_{family}")
    attention_name, top_name = families[family]
    attention_cls = getattr(modeling, attention_name)
    top_cls = getattr(modeling, top_name)
    if not getattr(attention_cls, "_embodied_eval_dhta_patched", False):
        original_attention_init = attention_cls.__init__

        @wraps(original_attention_init)
        def attention_init(self, attention_config, layer_idx):
            original_attention_init(self, attention_config, layer_idx)
            if bool(getattr(attention_config, "dhta_enabled", False)):
                _configure_attention(self, attention_config, layer_idx)

        attention_cls.__init__ = attention_init
        attention_cls._embodied_eval_dhta_patched = True
    if not getattr(top_cls, "_embodied_eval_dhta_patched", False):
        original_top_init = top_cls.__init__
        original_top_forward = top_cls.forward

        @wraps(original_top_init)
        def top_init(self, top_config):
            original_top_init(self, top_config)
            projections = list(_iter_dhta_projections(self))
            if projections:
                self._dhta_projection_refs = tuple(weakref.ref(item) for item in projections)
                self._dhta_current_route = None

        @wraps(original_top_forward)
        def top_forward(self, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            _set_model_route(self, input_ids, kwargs.get("past_key_values"))
            return original_top_forward(self, *args, **kwargs)

        top_cls.__init__ = top_init
        top_cls.forward = top_forward
        top_cls._embodied_eval_dhta_patched = True


def validate_dhta_loading_info(loading_info: dict) -> None:
    """Fail if any DHTA checkpoint tensors were not restored exactly."""
    from abot_er.checkpoint_contract import validate_loading_info
    validate_loading_info(loading_info)
    missing = [key for key in loading_info.get("missing_keys", ()) if ".dhta." in key]
    unexpected = [key for key in loading_info.get("unexpected_keys", ()) if ".dhta." in key]
    mismatched = [item for item in loading_info.get("mismatched_keys", ()) if ".dhta." in str(item)]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "DHTA checkpoint state did not load exactly: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}."
        )


def validate_dhta_model(model: nn.Module) -> None:
    if not bool(getattr(model.config, "dhta_enabled", False)):
        return
    layers = _parse_layers(getattr(model.config, "dhta_layers"))
    expected = sorted((layer, name) for layer in layers for name in ("q_proj", "v_proj"))
    projections = list(_iter_dhta_projections(model))
    actual = sorted(
        (int(projection.dhta_layer_idx), str(projection.dhta_projection_name))
        for projection in projections
    )
    if actual != expected:
        raise RuntimeError(f"DHTA projection mismatch: expected={expected}, actual={actual}.")
    rank = int(getattr(model.config, "dhta_rank"))
    if any(projection.residual().rank != rank for projection in projections):
        raise RuntimeError(f"DHTA rank mismatch; expected rank={rank}.")
    eval_logger.info(
        "DHTA model validated: layers={} projections={} rank={} schema={}",
        layers,
        len(projections),
        rank,
        getattr(model.config, "dhta_schema_version", "unknown"),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        eval_logger.info("DHTA route enforcement is enabled for every prefill batch")


def validate_dhta_tokenizer(config, tokenizer) -> None:
    if not bool(getattr(config, "dhta_enabled", False)):
        return
    expected = (
        (str(getattr(config, "dhta_qa_token")), int(getattr(config, "dhta_qa_token_id"))),
        (str(getattr(config, "dhta_task_plan_token")), int(getattr(config, "dhta_task_plan_token_id"))),
    )
    if expected[0][0] != DHTA_QA_TOKEN or expected[1][0] != DHTA_TASK_PLAN_TOKEN:
        raise ValueError(f"Unsupported DHTA route token configuration: {expected!r}.")
    resolved = []
    for token, expected_id in expected:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids != [expected_id]:
            raise ValueError(
                f"DHTA token must resolve atomically to checkpoint id: token={token!r}, "
                f"expected={[expected_id]}, actual={ids}."
            )
        resolved.append(expected_id)
    if resolved[0] == resolved[1]:
        raise ValueError("DHTA route tokens resolve to the same token id.")
    eval_logger.info("DHTA tokenizer validated: qa_token_id={} task_plan_token_id={}", *resolved)


def inject_dhta_route(messages: list[dict], task_family: str) -> list[dict]:
    """Inject exactly one route token into the first user text segment."""
    routes = {"2d_qa": DHTA_QA_TOKEN, "task_plan": DHTA_TASK_PLAN_TOKEN}
    if task_family not in routes:
        raise ValueError(f"Unknown DHTA task_family={task_family!r}.")
    expected = routes[task_family]
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text_segments = [(message, "content")]
        elif isinstance(content, list):
            text_segments = [
                (item, "text")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
        else:
            raise TypeError("First DHTA user content must be a string or multimodal content list.")
        if not text_segments:
            raise ValueError("First DHTA user message has no text segment.")
        qa_count = sum(holder[key].count(DHTA_QA_TOKEN) for holder, key in text_segments)
        plan_count = sum(holder[key].count(DHTA_TASK_PLAN_TOKEN) for holder, key in text_segments)
        if qa_count + plan_count == 0:
            holder, key = text_segments[0]
            if isinstance(content, list):
                holder[key] = holder[key].lstrip()
                content.insert(0, {"type": "text", "text": f"{expected}\n"})
            else:
                holder[key] = f"{expected}\n{holder[key].lstrip()}"
        elif qa_count + plan_count == 1 and sum(holder[key].count(expected) for holder, key in text_segments) == 1:
            pass
        else:
            raise ValueError(
                f"Invalid DHTA route: expected={expected}, qa_count={qa_count}, plan_count={plan_count}."
            )
        return messages
    raise ValueError("No user message found for DHTA route injection.")
