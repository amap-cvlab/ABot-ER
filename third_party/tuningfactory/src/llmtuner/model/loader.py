from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import AutoModelForCausalLMWithValueHead

from ..data.template import get_template_and_fix_tokenizer
from ..extras.logging import get_logger
from ..extras.misc import (
    count_parameters,
    try_download_model_from_ms,
)
from ..extras.packages import is_transformers_version_greater_than
from ..extras.patches.internvl_patch import dispatch_model
from .adapter import init_adapter
from .custom_model.deepseek_vl2 import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from .model_utils.dhta import (
    configure_dhta_config,
    inject_dhta_after_pretrained_load,
    prepare_dhta_model_class,
    set_dhta_injection_mode,
    validate_dhta_model,
)
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.qwen3_5_backend import maybe_disable_qwen3_5_fast_linear_attention
from .model_utils.sequence_parallel import apply_sequence_parallel
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model


if is_transformers_version_greater_than("4.48.0"):
    from transformers import AutoModelForImageTextToText
else:
    from transformers import AutoModelForVision2Seq
    AutoModelForImageTextToText = AutoModelForVision2Seq


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..data.template import Template
    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    return {
        "trust_remote_code": False,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(
    model_args: "ModelArguments",
) -> Tuple["PreTrainedTokenizer", Optional["ProcessorMixin"]]:
    r"""
    Loads pretrained tokenizer. Must before load_model.

    Note: including inplace operation of model_args.
    """
    try_download_model_from_ms(model_args)
    init_kwargs = _get_init_kwargs(model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **init_kwargs,
    )
    patch_tokenizer(tokenizer, model_args)

    processor = None

    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    except Exception as e:
        logger.info(f"Processor was not found: {e}.")
        processor = None

    try:
        model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    except Exception as e:  # turbo ckpt
        model_config = None

    if getattr(model_config, "visual", None) and getattr(model_config, "model_type", None) == "qwen":
        from .model_utils.multimodal import CommonVisualProcessor
        processor = CommonVisualProcessor(model_config.visual)
    elif getattr(model_config, "model_type", None) == "chatglm" and getattr(model_config, "vision_config", None):
        from .model_utils.multimodal import CommonVisualProcessor
        processor = CommonVisualProcessor(model_config.vision_config)
    elif getattr(model_config, "model_type", None) == "internvl_chat":
        from .model_utils.multimodal import InternVL2VisualProcessor
        processor = InternVL2VisualProcessor(model_config.vision_config)
    elif (architecture := getattr(model_config, "architectures", None)) and (architecture[0] == "InternLMXComposer2ForCausalLM"):
        from .model_utils.multimodal import InternLMXcomposerVisualProcessor
        processor = InternLMXcomposerVisualProcessor()
    elif getattr(model_config, "model_type", None) == "deepseek_vl_v2":
        from .model_utils.multimodal import DeepseekVL2Processor
        deepseek_vl2_processor = DeepseekVLV2Processor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        processor = DeepseekVL2Processor(deepseek_vl2_processor)
        tokenizer = deepseek_vl2_processor.tokenizer
    else:
        # Avoid load tokenizer, see:
        # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
        if processor is not None and "Processor" not in processor.__class__.__name__:
            processor = None

    if processor is not None:
        setattr(processor, "tokenizer", tokenizer)

    return tokenizer, processor


def load_model_config(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: Optional[bool] = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model. Must after load_tokenizer.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    configure_dhta_config(config, tokenizer, model_args)
    return config


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False,
    full_determinism: Optional[bool] = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model. Must after load_tokenizer.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    from abot_er.checkpoint_contract import normalize_checkpoint_config, validate_loading_info, validate_local_index
    dhta_from_checkpoint = normalize_checkpoint_config(config) is not None
    validate_local_index(config, model_args.model_name_or_path)
    configure_dhta_config(config, tokenizer, model_args)
    # An unadapted backbone contains no residual keys. Constructing fresh residual tensors
    # inside from_pretrained makes Transformers 5.x run them through its
    # ZeRO-3 missing-key gather, which can attach unrelated partition metadata.
    # Existing DHTA checkpoints still need construction-time injection so their
    # state dict keys have destinations.
    set_dhta_injection_mode(config, dhta_from_checkpoint)
    prepare_dhta_model_class(config)

    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))
    sequence_parallel_group = apply_sequence_parallel(model_args, full_determinism)

    model = None
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            raise ValueError("adapter_name_or_path is not supported when use_unsloth")
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # assume built-in models
            model_class = AutoModelForImageTextToText        # image and video
        # elif type(config) in AutoModelForImageTextToText._model_mapping.keys():
        #     model_class = AutoModelForImageTextToText   # image
        elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():
            model_class = AutoModelForSeq2SeqLM         # audio
        else:
            model_class = AutoModelForCausalLM          # text

        if getattr(config, "model_type", None) == "internvl_chat":
            model_class = AutoModel
            if not is_trainable and not is_deepspeed_zero3_enabled():
                init_kwargs.update({'device_map': dispatch_model(config.llm_config)})

        if getattr(config, "model_type", None) == "minicpmv":
            model_class = AutoModel
            init_kwargs.pop('device_map')  # not support device_map
        if getattr(config, "model_type", None) == "deepseek_vl_v2":
            model_class = DeepseekVLV2ForCausalLM
        if getattr(config, "model_type", None) == "qwen2_5_omni":
            from transformers import Qwen2_5OmniForConditionalGeneration
            model_class = Qwen2_5OmniForConditionalGeneration
            config.enable_audio_output = False
            # init_kwargs['attn_implementation'] = "flash_attention_2"
        if getattr(config, "model_type", None) == "qwen3_omni_moe":
            from transformers import Qwen3OmniMoeForConditionalGeneration
            model_class = Qwen3OmniMoeForConditionalGeneration
            model_args.resize_vocab = False
            # init_kwargs['attn_implementation'] = "flash_attention_2"
        if getattr(config, "model_type", None) == "minicpmo":
            if is_trainable:
                init_kwargs['config'].init_audio = False
                init_kwargs['config'].init_tts = False
        if hasattr(config, "is_rm_model"):
            # （TuningFactory自行训练的模型）如果包含value_head的话，那就将add_valuehead设置为true
            if hasattr(config, "is_trl_causal_lm_with_value_head"):
                if add_valuehead is None:
                    add_valuehead = True
            else:
                # （一些官方直出模型：Qwen/Qwen2.5-Math-RM-72B）使用AutoModel加载
                model_class = AutoModel
        if hasattr(config, "is_embedding_model"):
            model_class = AutoModel

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if dhta_from_checkpoint:
                if getattr(config, "abot_residual_namespace", None) == "task_conditioned":
                    # Use the reviewed local implementation, never execute hub code.
                    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration
                    model_class = (Qwen3_5MoeForConditionalGeneration if config.model_type == "qwen3_5_moe"
                                   else Qwen3_5ForConditionalGeneration)
                    init_kwargs["trust_remote_code"] = False
                    config.auto_map = {}
                    config.architectures = [model_class.__name__]
                model, loading_info = model_class.from_pretrained(output_loading_info=True, **init_kwargs)
                validate_loading_info(loading_info)
            else:
                model = model_class.from_pretrained(**init_kwargs)
            if model_class.__name__ == 'Qwen2_5OmniForConditionalGeneration':
                model = model.thinker

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if bool(getattr(config, "dhta_enabled", False)) and not dhta_from_checkpoint:
        inject_dhta_after_pretrained_load(model)

    maybe_disable_qwen3_5_fast_linear_attention(model)
    validate_dhta_model(model)
    patch_model(model, finetuning_args, tokenizer, model_args, is_trainable, add_valuehead)
    register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model, stage=finetuning_args.stage)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            if is_deepspeed_zero3_enabled():
                import deepspeed  # type: ignore

                params = [param for _, param in model.v_head.named_parameters(recurse=False)]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        model.load_state_dict(vhead_params, strict=False)
            else:
                model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        if finetuning_args.finetuning_type != "reft":
            for param in model.parameters():
                if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                    param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:d}".format(all_param)
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    model.sequence_parallel_group = sequence_parallel_group
    return model


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.
    """
    tokenizer, processor = load_tokenizer(model_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)
    return model, tokenizer, processor


def load_template(model_args, template: Optional[str] = None, enable_thinking: Optional[bool] = True) -> "Template":
    tokenizer, processor = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(
        tokenizer, name=template, processor=processor, processor_args=model_args, enable_thinking=enable_thinking
    )
    return template


def load_model_and_template(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    template: Optional[str] = None,
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False,
    enable_thinking: Optional[bool] = True,
) -> Tuple["PreTrainedModel", "Template"]:
    tokenizer, processor = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(
        tokenizer, name=template, processor=processor, processor_args=model_args, enable_thinking=enable_thinking
    )
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)
    return model, template
