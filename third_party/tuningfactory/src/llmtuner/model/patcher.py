import json
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from peft import PeftModel
from transformers import AddedToken, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras.logging import get_logger
from ..extras.misc import infer_optim_dtype
from ..extras.patches.internlm_xcomposer_patch import patch_internlm_xcomposer_forward_impl
from ..extras.patches.internvl_patch import patch_internvl_chat_model
from ..extras.patches.minicpmo_patch import patch_minicpmo_generate_impl
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import configure_embedding_model, resize_embedding_layer
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.multimodal import autocast_projector_dtype, configure_visual_model
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def _add_tokens(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if not model_args.add_tokens_file:
        return
    with open(model_args.add_tokens_file, "r") as f:
        add_tokens = json.load(f)
    add_tokens = [AddedToken(**token) if isinstance(token, dict) else token for token in add_tokens]
    old_token_num = len(tokenizer)
    tokenizer.add_tokens(add_tokens)
    new_token_num = len(tokenizer)
    logger.info(
        f"Added {new_token_num - old_token_num} tokens to tokenizer. Now tokenizer has {new_token_num} tokens."
    )


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments"):
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
    _add_tokens(tokenizer, model_args)


def configure_rm_model(config: "PretrainedConfig", tokenizer: "PreTrainedTokenizer") -> None:
    """
    TODO: 目前没找到RM模型明确的判断方式，先采用模型逐个注册的方式，后续考虑通用方式，代码先暂时放在loader.py中
    """

    # 采用手动注册的方式
    manual_register =  (architecture := getattr(config, "architectures", None)) and (architecture[0] == "Qwen2ForRewardModel")
    # 采用自动检测的方式，使用TuningFactory训练的RM模型具备此标识
    training_from_tuning_factory = getattr(config, "is_trl_causal_lm_with_value_head", None)

    if manual_register or training_from_tuning_factory:
        setattr(config, "is_rm_model", True)
        # 当前版本Qwen/Qwen2.5-Math-RM-72B加载后, model_config不包含pad_token_id, 需要tokenizer注入
        if not getattr(config, "pad_token_id", None):
            setattr(config, "pad_token_id", tokenizer.pad_token_id)


def configure_sequence_parallel(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    if (
        model_args.sequence_parallel_size > 1
        and hasattr(config, "attention_dropout")
        and config.attention_dropout != 0.0
    ):
        logger.warning("Sequence Parallel doesn't support attention_dropout yet. Forcing it to zero.")
        config.attention_dropout = 0.0


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: Dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args, is_trainable)
    configure_rope(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_embedding_model(config, model_args)
    configure_rm_model(config, tokenizer)
    configure_sequence_parallel(config, model_args)

    if not is_trainable:
        setattr(config, "use_cache", True)
        logger.info("Using KV cache for faster generation.")

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "qwen2" and is_trainable and model_args.flash_attn == "fa2":
        setattr(config, "use_cache", False)  # qwen2 does not support use_cache when using flash attn

    if getattr(config, "model_type", None) == "deepseek_vl_v2" and is_trainable:
        setattr(config, "use_cache", True)

    if getattr(config, "model_type", None) == "internvl_chat":
        setattr(config, "vocab_size", config.llm_config.vocab_size)

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = not is_deepspeed_zero3_enabled()

    # cast data type of the model if:
    # 1. not deepspeed zero3 and not fsdp (keep zero3 or fsdp in float32)
    # 2. quantization_bit is not None (qlora)
    if (not is_deepspeed_zero3_enabled() and not is_fsdp_enabled()) or model_args.quantization_bit is not None:
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"]:  # device map requires low_cpu_mem_usage=True
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map


def patch_model(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if gen_config and not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    # if "GenerationMixin" not in str(model.generate.__func__):
    #     model.generate = MethodType(PreTrainedModel.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if getattr(model.config, "model_type", None) == "internvl_chat":
        model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        use_submodel_func(model, 'language_model', ['get_input_embeddings', 'gradient_checkpointing_enable'])
        patch_internvl_chat_model(model)

    if (architecture := getattr(model.config, "architectures", None)) and (architecture[0]  == "InternLMXComposer2ForCausalLM"):
        patch_internlm_xcomposer_forward_impl(model)
        model.image_token_id = tokenizer.convert_tokens_to_ids('<image>')
        model.tokenizer = tokenizer

    if getattr(model.config, "model_type", None) == "deepseek_vl_v2":
        model.image_token_id = tokenizer.convert_tokens_to_ids('<image>')

    if getattr(model.config, "model_type", None) == "minicpmo":
        patch_minicpmo_generate_impl(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    autocast_projector_dtype(model, model_args)

    if is_trainable:
        prepare_model_for_training(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead", stage="rm") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
    if stage == "rm":
        # Compatibility flag for the optional distributed backend.
        setattr(model.config, "is_trl_causal_lm_with_value_head", True)

def use_submodel_func(model, submodel_name: str, func_list: List[str]) -> None:
    submodel = getattr(model, submodel_name)

    def _get_new_func(func_name: str):
        _old_func = getattr(submodel, func_name)

        @wraps(_old_func)
        def _new_func(*args, **kwargs):
            return _old_func(*args, **kwargs)

        return _new_func

    for key in func_list:
        setattr(model, key, _get_new_func(key))
