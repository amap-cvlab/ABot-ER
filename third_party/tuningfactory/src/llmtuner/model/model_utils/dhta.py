"""Deterministic hierarchical task adaptation (DHTA) for Qwen3.5.

DHTA keeps the pretrained projection weights shared and adds one low-rank
residual family for 2D-QA and one for Task Plan.  A task token in ``input_ids``
selects exactly one family for the whole sample.  Residual modules are injected
during model construction when restoring a DHTA checkpoint.  When starting
from an ordinary backbone checkpoint they are injected only after
``from_pretrained`` has finished, avoiding Transformers 5.x ZeRO-3 missing-key
initialization corrupting their partition metadata.
"""

from __future__ import annotations

import math
import os
import weakref
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Iterable, List, Sequence, Union

import torch
from torch import nn

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

    from ...hparams import ModelArguments


logger = get_logger(__name__)

DHTA_QA_ROUTE = 0
DHTA_TASK_PLAN_ROUTE = 1


def _parse_layers(value: Union[str, Sequence[int]]) -> List[int]:
    if isinstance(value, str):
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        layers = [int(item) for item in value]
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError(f"Invalid DHTA layer selection: {value!r}")
    return sorted(layers)


class DHTALowRankResidual(nn.Module):
    """Two task-specific low-rank residuals sharing no branch parameters."""

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
        # DeepSpeed discovers ZeRO-3 leaf modules from this attribute while it
        # installs parameter-coordinator hooks.  Mark the complete residual at
        # construction time so all four low-rank projections are gathered and
        # released as one unit.  Runtime ``is_deepspeed_zero3_enabled`` checks
        # are too late/fragile here: model construction can happen before that
        # Transformers global flag is visible, leaving child Linear hooks to
        # gather the wrong parameter while Qwen's parent projection is active.
        # The attribute is inert when DeepSpeed/ZeRO-3 is not in use.
        self._z3_leaf = True
        self.reset_parameters()
        # Qwen/Transformers calls ``post_init().apply(_initialize_weights)``
        # after attention construction.  These projections already have the
        # required LoRA-style initialization (A=random, B=zero), so mark every
        # child as initialized before that traversal.  This is also critical
        # under ZeRO-3: resetting these parameters after ``zero.Init`` has
        # partitioned them corrupts DeepSpeed's parameter-shape metadata.
        for projection in (self.qa_a, self.qa_b, self.task_plan_a, self.task_plan_b):
            projection._is_hf_initialized = True
            # Transformers 5.x selects ZeRO-3 missing-key initialization from
            # flags on the parameters themselves (state_dict(keep_vars=True)).
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

    def _parameter_snapshot(self) -> dict:
        snapshot = {"z3_leaf": getattr(self, "_z3_leaf", None)}
        for name, projection in (
            ("qa_a", self.qa_a),
            ("qa_b", self.qa_b),
            ("task_plan_a", self.task_plan_a),
            ("task_plan_b", self.task_plan_b),
        ):
            parameter = projection.weight
            ds_tensor = getattr(parameter, "ds_tensor", None)
            snapshot[name] = {
                "shape": tuple(parameter.shape),
                "ds_shape": tuple(getattr(parameter, "ds_shape", ())),
                "ds_tensor_shape": tuple(ds_tensor.shape) if ds_tensor is not None else None,
                "ds_numel": getattr(parameter, "ds_numel", None),
                "ds_id": getattr(parameter, "ds_id", None),
                "ds_status": str(getattr(parameter, "ds_status", None)),
                "leaf_owner": type(getattr(parameter, "ds_z3_leaf_module", None)).__name__
                if getattr(parameter, "ds_z3_leaf_module", None) is not None
                else None,
            }
        return snapshot

    def forward(self, hidden_states: torch.Tensor, route: Union[int, torch.Tensor]) -> torch.Tensor:
        bad_parameters = [
            name
            for name, projection in (
                ("qa_a", self.qa_a),
                ("qa_b", self.qa_b),
                ("task_plan_a", self.task_plan_a),
                ("task_plan_b", self.task_plan_b),
            )
            if projection.weight.ndim != 2
        ]
        if bad_parameters:
            raise RuntimeError(
                "DHTA ZeRO parameter shape corruption before routed linear: "
                f"bad={bad_parameters}, snapshot={self._parameter_snapshot()}"
            )

        # ZeRO-3 requires every data-parallel rank to traverse parameterized
        # child modules in the same order. With per-device batch size 1 the
        # route is an ``int``; conditionally executing only one branch then
        # makes QA ranks gather qa_* while Task Plan ranks gather task_plan_*,
        # corrupting the parameter coordinator's gather sequence. Always run
        # both tiny rank-r residuals and select with differentiable gates.
        # This keeps the forward/backward graph identical across ranks; the
        # inactive branch receives an exact zero gradient.
        if isinstance(route, int):
            if route not in (DHTA_QA_ROUTE, DHTA_TASK_PLAN_ROUTE):
                raise ValueError(f"Unknown DHTA route: {route}.")
            qa_gate = hidden_states.new_tensor(float(route == DHTA_QA_ROUTE))
            task_plan_gate = hidden_states.new_tensor(float(route == DHTA_TASK_PLAN_ROUTE))
            return self._qa(hidden_states) * qa_gate + self._task_plan(hidden_states) * task_plan_gate

        if route.ndim != 1:
            raise ValueError(f"DHTA route must be rank-1, got shape={tuple(route.shape)}.")
        if hidden_states.shape[0] != route.shape[0]:
            if route.numel() == 1:
                scalar_route = int(route.item())
                return self.forward(hidden_states, scalar_route)
            raise ValueError(
                "DHTA route batch does not match projection input: "
                f"route={tuple(route.shape)}, hidden={tuple(hidden_states.shape)}."
            )

        gate_shape = (route.shape[0],) + (1,) * (hidden_states.ndim - 1)
        qa_gate = (route == DHTA_QA_ROUTE).view(gate_shape).to(dtype=hidden_states.dtype)
        task_plan_gate = (route == DHTA_TASK_PLAN_ROUTE).view(gate_shape).to(dtype=hidden_states.dtype)
        if not torch.all((qa_gate + task_plan_gate) == 1):
            raise ValueError("DHTA route contains values other than 0 (2D-QA) and 1 (Task Plan).")
        return self._qa(hidden_states) * qa_gate + self._task_plan(hidden_states) * task_plan_gate


class DHTARoutedLinear(nn.Linear):
    """State-dict-compatible linear projection with an explicit DHTA child.

    The shared ``weight`` and ``bias`` stay directly registered on the
    projection, preserving keys such as ``q_proj.weight`` when loading a base
    checkpoint.  The residual is executed from the module's normal forward
    graph instead of a forward hook.  This ordering is required by ZeRO-3's
    parameter coordinator, which cannot safely discover a new sibling module
    while another module's post-forward hooks are running.
    """

    def __init__(self, base_projection: nn.Linear, rank: int, alpha: float, namespace: str = "dhta") -> None:
        if not isinstance(base_projection, nn.Linear):
            raise TypeError(f"DHTA requires nn.Linear projections, got {type(base_projection)!r}.")

        # Do not call nn.Linear.__init__: reuse the already-created base
        # parameters so their checkpoint keys and ZeRO metadata stay intact.
        nn.Module.__init__(self)
        self.in_features = int(base_projection.in_features)
        self.out_features = int(base_projection.out_features)
        self.weight = base_projection.weight
        if base_projection.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = base_projection.bias
        if namespace not in ("dhta", "task_conditioned"):
            raise ValueError("Unknown residual tensor namespace")
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
        route = self._dhta_route
        if route is None:
            raise RuntimeError("DHTA route was not configured before an adapted projection ran.")
        shared = nn.functional.linear(hidden_states, self.weight, self.bias)
        residual = self.residual()(hidden_states, route)
        return shared + residual.to(dtype=shared.dtype)


def _iter_dhta_projections(model: nn.Module) -> Iterable[DHTARoutedLinear]:
    for module in model.modules():
        if isinstance(module, DHTARoutedLinear):
            yield module


def _set_model_route(model: nn.Module, input_ids: torch.Tensor | None, past_key_values) -> None:
    refs = getattr(model, "_dhta_projection_refs", ())
    projections = [ref() for ref in refs if ref() is not None]
    if not projections:
        return

    config = model.config
    require_route = bool(getattr(config, "dhta_require_route_token", True))
    previous_route = getattr(model, "_dhta_current_route", None)
    if input_ids is None:
        if previous_route is not None and past_key_values is not None:
            return
        if require_route:
            raise ValueError("DHTA requires input_ids containing exactly one task route token per sample.")
        return

    qa_token_id = int(getattr(config, "dhta_qa_token_id"))
    task_plan_token_id = int(getattr(config, "dhta_task_plan_token_id"))
    qa_hits = input_ids.eq(qa_token_id).sum(dim=-1)
    task_plan_hits = input_ids.eq(task_plan_token_id).sum(dim=-1)
    valid = (qa_hits + task_plan_hits).eq(1)

    if not bool(valid.all()):
        # Generation decode steps normally contain only the newest token.  Keep
        # the route established by the prefill call when a cache is present.
        if previous_route is not None and past_key_values is not None and input_ids.shape[-1] <= 1:
            return
        if require_route:
            bad_rows = (~valid).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Each DHTA sample must contain exactly one of the two route tokens; "
                f"invalid batch rows={bad_rows}."
            )
        return

    route_tensor = task_plan_hits.to(dtype=torch.long)
    route: Union[int, torch.Tensor]
    if route_tensor.numel() == 1:
        route = int(route_tensor.item())
    else:
        route = route_tensor
    model._dhta_current_route = route
    for projection in projections:
        projection._dhta_route = route


def _configure_attention_instance(attention: nn.Module, config, layer_idx: int) -> None:
    selected_layers = set(_parse_layers(getattr(config, "dhta_layers")))
    if layer_idx not in selected_layers:
        return
    if not hasattr(attention, "q_proj") or not hasattr(attention, "v_proj"):
        raise TypeError(f"DHTA layer {layer_idx} does not expose q_proj/v_proj.")

    already_wrapped = (
        isinstance(attention.q_proj, DHTARoutedLinear),
        isinstance(attention.v_proj, DHTARoutedLinear),
    )
    if all(already_wrapped):
        return
    if any(already_wrapped):
        raise RuntimeError(f"DHTA layer {layer_idx} is only partially wrapped: {already_wrapped}.")

    rank = int(getattr(config, "dhta_rank"))
    alpha = float(getattr(config, "dhta_alpha"))
    namespace = getattr(config, "abot_residual_namespace", "dhta")
    attention.q_proj = DHTARoutedLinear(attention.q_proj, rank, alpha, namespace)
    attention.v_proj = DHTARoutedLinear(attention.v_proj, rank, alpha, namespace)
    attention.q_proj.dhta_layer_idx = layer_idx
    attention.q_proj.dhta_projection_name = "q_proj"
    attention.v_proj.dhta_layer_idx = layer_idx
    attention.v_proj.dhta_projection_name = "v_proj"


def prepare_dhta_model_class(config: "PretrainedConfig") -> None:
    """Patch Qwen3.5 constructors before ``from_pretrained`` instantiates them."""

    if not bool(getattr(config, "dhta_enabled", False)):
        return
    if getattr(config, "model_type", None) != "qwen3_5":
        raise ValueError(f"DHTA currently supports only qwen3_5, got {config.model_type!r}.")

    modeling = import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    attention_cls = modeling.Qwen3_5Attention
    top_cls = modeling.Qwen3_5ForConditionalGeneration

    if not getattr(attention_cls, "_llmtuner_dhta_patched", False):
        original_attention_init = attention_cls.__init__

        @wraps(original_attention_init)
        def attention_init(self, attention_config, layer_idx):
            original_attention_init(self, attention_config, layer_idx)
            if bool(getattr(attention_config, "dhta_enabled", False)) and bool(
                getattr(attention_config, "dhta_inject_during_init", True)
            ):
                _configure_attention_instance(self, attention_config, layer_idx)

        attention_cls.__init__ = attention_init
        attention_cls._llmtuner_dhta_patched = True

    if not getattr(top_cls, "_llmtuner_dhta_patched", False):
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
        top_cls._llmtuner_dhta_patched = True


def set_dhta_injection_mode(config: "PretrainedConfig", inject_during_init: bool) -> None:
    """Select the safe DHTA construction lifecycle for the next model load."""

    if not bool(getattr(config, "dhta_enabled", False)):
        return
    for target in (config, getattr(config, "text_config", None)):
        if target is not None:
            setattr(target, "dhta_inject_during_init", bool(inject_during_init))


def inject_dhta_after_pretrained_load(model: "PreTrainedModel") -> None:
    """Inject fresh DHTA tensors after loading a non-DHTA backbone checkpoint.

    The Transformers missing-key initialization path has already completed at
    this point.  DeepSpeed converts the newly-created ordinary parameters to
    ZeRO tensors when the training engine is initialized.
    """

    if not bool(getattr(model.config, "dhta_enabled", False)):
        return
    if getattr(model.config, "model_type", None) != "qwen3_5":
        raise ValueError(f"DHTA currently supports only qwen3_5, got {model.config.model_type!r}.")

    modeling = import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    attention_cls = modeling.Qwen3_5Attention
    for module in list(model.modules()):
        if isinstance(module, attention_cls):
            _configure_attention_instance(module, model.config, int(getattr(module, "layer_idx")))

    projections = list(_iter_dhta_projections(model))
    model._dhta_projection_refs = tuple(weakref.ref(item) for item in projections)
    model._dhta_current_route = None
    # Checkpoints produced from this model contain DHTA state and must inject
    # the modules during their next from_pretrained call.
    set_dhta_injection_mode(model.config, True)
    logger.info("[dhta] injected %d routed projections after backbone load.", len(projections))


def configure_dhta_config(
    config: "PretrainedConfig", tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments"
) -> None:
    """Validate task tokens and persist all DHTA architecture fields in config."""

    from abot_er.checkpoint_contract import normalize_checkpoint_config
    namespace = normalize_checkpoint_config(config)
    enabled_from_checkpoint = namespace is not None
    enabled = bool(model_args.enable_dhta or enabled_from_checkpoint)
    if not enabled:
        return

    if model_args.enable_dhta:
        qa_token = model_args.dhta_qa_token
        task_plan_token = model_args.dhta_task_plan_token
        layers = _parse_layers(model_args.dhta_layers)
        rank = int(model_args.dhta_rank)
        alpha = float(model_args.dhta_alpha)
        require_route = bool(model_args.dhta_require_route_token)
    else:
        qa_token = str(getattr(config, "dhta_qa_token"))
        task_plan_token = str(getattr(config, "dhta_task_plan_token"))
        layers = _parse_layers(getattr(config, "dhta_layers"))
        rank = int(getattr(config, "dhta_rank"))
        alpha = float(getattr(config, "dhta_alpha"))
        require_route = bool(getattr(config, "dhta_require_route_token", True))

    if qa_token == task_plan_token:
        raise ValueError("DHTA route tokens must be distinct.")
    if rank <= 0 or alpha <= 0:
        raise ValueError(f"DHTA rank and alpha must be positive, got rank={rank}, alpha={alpha}.")
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        invalid_layers = [
            layer_idx
            for layer_idx in layers
            if layer_idx >= len(layer_types) or layer_types[layer_idx] != "full_attention"
        ]
        if invalid_layers:
            raise ValueError(
                "DHTA q/v residuals require Qwen3.5 full-attention layers; "
                f"invalid selection={invalid_layers}."
            )

    token_ids = {}
    for name, token in (("qa", qa_token), ("task_plan", task_plan_token)):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"DHTA {name} token must encode atomically, got token={token!r}, ids={encoded}.")
        token_ids[name] = int(encoded[0])
    if token_ids["qa"] == token_ids["task_plan"]:
        raise ValueError("DHTA route tokens resolved to the same token id.")

    values = {
        "dhta_enabled": True,
        "dhta_qa_token": qa_token,
        "dhta_task_plan_token": task_plan_token,
        "dhta_qa_token_id": token_ids["qa"],
        "dhta_task_plan_token_id": token_ids["task_plan"],
        "dhta_layers": layers,
        "dhta_rank": rank,
        "dhta_alpha": alpha,
        "dhta_require_route_token": require_route,
        "dhta_schema_version": "dhta-qv-v1",
    }
    if enabled_from_checkpoint:
        for key, value in values.items():
            if getattr(config, key, None) != value:
                raise ValueError(f"Cannot override restored residual contract: {key}")
    for target in (config, getattr(config, "text_config", None)):
        if target is not None:
            for key, value in values.items():
                setattr(target, key, value)

    model_args.resize_vocab = True
    logger.info(
        "[dhta] enabled schema=%s layers=%s rank=%d alpha=%s qa_token_id=%d task_plan_token_id=%d",
        values["dhta_schema_version"],
        layers,
        rank,
        alpha,
        token_ids["qa"],
        token_ids["task_plan"],
    )


def validate_dhta_model(model: "PreTrainedModel") -> None:
    if not bool(getattr(model.config, "dhta_enabled", False)):
        return
    expected_layers = _parse_layers(getattr(model.config, "dhta_layers"))
    projections = list(_iter_dhta_projections(model))
    expected_projections = sorted(
        (layer_idx, projection_name)
        for layer_idx in expected_layers
        for projection_name in ("q_proj", "v_proj")
    )
    actual_projections = sorted(
        (int(projection.dhta_layer_idx), str(projection.dhta_projection_name))
        for projection in projections
    )
    if actual_projections != expected_projections:
        raise RuntimeError(
            "DHTA projection mismatch: "
            f"expected={expected_projections}, actual={actual_projections}."
        )

    rank = int(getattr(model.config, "dhta_rank"))
    for projection in projections:
        if projection.residual().rank != rank:
            raise RuntimeError(
                "DHTA rank mismatch in "
                f"layer={projection.dhta_layer_idx} {projection.dhta_projection_name}."
            )

    if projections and int(os.environ.get("RANK", "0")) == 0:
        first = projections[0]
        print(
            "[dhta-param-audit][pre-engine] "
            f"layer={first.dhta_layer_idx} projection={first.dhta_projection_name} "
            f"snapshot={first.residual()._parameter_snapshot()}",
            flush=True,
        )

    logger.info(
        "[dhta] validated %d adapted attention layers and %d task-specific residual projections.",
        len(expected_layers),
        len(projections) * 2,
    )
