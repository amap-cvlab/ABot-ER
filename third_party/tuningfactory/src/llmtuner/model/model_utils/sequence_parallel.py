from functools import partial

import torch
import torch.distributed as dist
import transformers
from transformers.utils import is_flash_attn_greater_or_equal_2_10

from ...extras.misc import get_logger
from ...extras.packages import _get_package_version
from .ulysses import UlyssesAttention


logger = get_logger(__name__)

_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


def _flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    sequence_parallel_size=1,
    dropout=0.0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    group=None,
    mode="zigzag-ring",
    attn_fn=None,
    **kwargs,
):
    if mode == "zigzag-ring":
        from ring_flash_attn import zigzag_ring_flash_attn_func

        attn_output = zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, deterministic=deterministic, causal=is_causal, group=group
        )
    elif mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_length * sequence_parallel_size,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
        )  # reset query_length to the real q_len before sp, Special settings for ulysses
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


def _new_flash_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    softcap=None,
    **kwargs,
):
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = _flash_attn_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]


def apply_sequence_parallel(model_args, full_determinism=False):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)
    original_attn = transformers.modeling_flash_attention_utils._flash_attention_forward
    try:
        if model_args.sequence_parallel_mode == "zigzag-ring":
            new_flash_attention_forward = partial(
                _new_flash_attention_forward,
                group=group_this,
                mode=model_args.sequence_parallel_mode,
                deterministic=full_determinism,
            )
        elif model_args.sequence_parallel_mode == "ulysses":
            new_flash_attention_forward = partial(
                _new_flash_attention_forward,
                group=group_this,
                mode=model_args.sequence_parallel_mode,
                deterministic=full_determinism,
                attn_fn=original_attn,
                sequence_parallel_size=model_args.sequence_parallel_size,
            )
        else:
            raise NotImplementedError("Other sequence parallel modes are to be implemented.")

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = new_flash_attention_forward
    except Exception:
        version = _get_package_version("transformers")
        logger.warning(f"Sequence parallelism is not supported in transformers {version}.")

    return group_this
