import os
from importlib import import_module
from typing import TYPE_CHECKING

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = get_logger(__name__)


def maybe_disable_qwen3_5_fast_linear_attention(model: "PreTrainedModel") -> None:
    if os.getenv("DISABLE_QWEN3_5_FAST_LINEAR_ATTN", "0") != "1":
        return

    patched_count = 0
    candidate_count = 0
    for module in model.modules():
        module_name = module.__class__.__name__
        if "Qwen3_5" not in module_name or not module_name.endswith("GatedDeltaNet"):
            continue

        candidate_count += 1
        modeling_module = import_module(module.__class__.__module__)
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = modeling_module.torch_causal_conv1d_update
        module.chunk_gated_delta_rule = modeling_module.torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = modeling_module.torch_recurrent_gated_delta_rule
        patched_count += 1

    if patched_count == 0:
        raise RuntimeError(
            "DISABLE_QWEN3_5_FAST_LINEAR_ATTN=1 was set, but no Qwen3.5 GatedDeltaNet modules were found "
            f"(candidate_count={candidate_count}, model_class={model.__class__.__name__})."
        )

    logger.warning(
        "DISABLE_QWEN3_5_FAST_LINEAR_ATTN=1: patched %d Qwen3.5 linear attention modules "
        "to use torch fallback kernels.",
        patched_count,
    )
