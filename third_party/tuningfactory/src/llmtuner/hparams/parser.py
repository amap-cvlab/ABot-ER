import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available
from transformers.utils.versions import require_version

from ..extras.logging import get_logger
from ..extras.misc import down_model, get_current_device, patch_training_args, process_mos_info_for_training_args
from ..extras.packages import is_turbo_available
from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments


logger = get_logger(__name__)


_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_INFER_CLS = Tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_EVAL_ARGS = [ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
_EVAL_CLS = Tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]

if is_turbo_available():
    from transformers_turbo import Seq2SeqTrainingArguments

    _TRAIN_TURBO_ARGS = [
        ModelArguments,
        DataArguments,
        Seq2SeqTrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
    ]


def _check_dependencies(disabled: bool) -> None:
    if disabled:
        logger.warning("Version checking has been disabled, may lead to unexpected behaviors.")
    else:
        require_version("transformers>=4.37.2", "To fix: pip install transformers>=4.37.2")
        require_version("datasets>=2.14.3", "To fix: pip install datasets>=2.14.3")
        require_version("accelerate>=0.27.2", "To fix: pip install accelerate>=0.27.2")
        require_version("peft>=0.9.0", "To fix: pip install peft>=0.9.0")
        require_version("trl>=0.7.11", "To fix: pip install trl>=0.7.11")


def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        try:
            print(parser.format_help())
        except TypeError as e:
           logger.warning(f"Failed to format help message. Error msg: `{str(e)}`. This is an known issue in transformers==5.2.0 and will be fixed in the future version.")
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> None:
    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantization is only compatible with the LoRA method.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")

    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")


def _is_turbo_training():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_turbo",
        type=lambda x: str(x).lower() in ("yes", "y", "true", "t", "1"),
        nargs="?",
        const=True,
        default=False,
    )
    args, _ = parser.parse_known_args()
    if args.use_turbo and not is_turbo_available():
        raise ImportError("Please install transformers-turbo to enable turbo training.")
    return args.use_turbo


def _check_extra_dependencies(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    training_args: Optional["Seq2SeqTrainingArguments"] = None,
) -> None:
    if model_args.use_unsloth:
        require_version("unsloth", "Please install unsloth: https://github.com/unslothai/unsloth")

    if model_args.mixture_of_depths is not None:
        require_version("mixture-of-depth>=1.1.6", "To fix: pip install mixture-of-depth>=1.1.6")

    if finetuning_args.plot_loss:
        require_version("matplotlib", "To fix: pip install matplotlib")

    if training_args is not None and training_args.predict_with_generate:
        require_version("jieba", "To fix: pip install jieba")
        require_version("nltk", "To fix: pip install nltk")
        require_version("rouge_chinese", "To fix: pip install rouge-chinese")

    if finetuning_args.pref_loss == "ift":
        require_version("torch_discounted_cumsum", "To fix: find how to install torch_discounted_cumsum in the document.")


def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_TURBO_ARGS if _is_turbo_training() else _TRAIN_ARGS)
    return _parse_args(parser, args)


def _parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return _parse_args(parser, args)


def _parse_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    parser = HfArgumentParser(_EVAL_ARGS)
    return _parse_args(parser, args)


def refine_args(*args):
    """
    replace mos_uri in begining to avoid duplicate downloads;
    nebula localization
    """
    # download models
    model_arg_names = [
        "model_name_or_path",
        "adapter_name_or_path",
        "teacher_model_name_or_path",
        "teacher_adapter_name_or_path",
        "ref_model",
        "ref_model_checkpoint",
        "reward_model",
        "reward_model_checkpoint",
        "ref_model_adapters",
    ]

    def download_model_arg(model_arg, used_for):
        if isinstance(model_arg, str):
            return down_model(model_arg, used_for)
        if isinstance(model_arg, list):
            return [down_model(item, used_for) for item in model_arg]
        return model_arg

    def get_arg_value(name, default=None):
        for arg in args:
            if hasattr(arg, name):
                return getattr(arg, name, default)
        return default

    def set_arg_value(name, value):
        for arg in args:
            if hasattr(arg, name):
                setattr(arg, name, value)

    for name in model_arg_names:
        value = get_arg_value(name, None)
        if value is not None:
            set_arg_value(name, download_model_arg(value, name))

    if (
        get_arg_value("streaming")
        and get_arg_value("accelerator_config")
        and get_arg_value("accelerator_config").dispatch_batches is not False
    ):
        logger.warning(
            "dispatch_batches in accelerator_config needs to be set as False for streaming dataset."
            "accelerator_config[`dispatch_batches`] is modified as False."
        )
        get_arg_value("accelerator_config").dispatch_batches = False


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
    assert finetuning_args.finetuning_type is not None and finetuning_args.finetuning_type in ["lora", "freeze", "full", "reft"], f"Invalid fine-tuning method: {finetuning_args.finetuning_type}."

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    refine_args(model_args, data_args, training_args, finetuning_args, generating_args)
    patch_training_args(training_args)

    # Check arguments
    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt` or `mask_history` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo" and not training_args.do_train:
        raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

    if finetuning_args.stage == "ppo" and finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
        raise ValueError("Unsloth does not support lora reward model.")

    if finetuning_args.stage == "distill" and  model_args.teacher_model_name_or_path is None:
        raise ValueError("teacher_model_name_or_path is necessary for distill")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")

    if finetuning_args.pure_bf16:
        if not is_torch_bf16_gpu_available():
            raise ValueError("This device does not support `pure_bf16`.")

        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")

    if model_args.use_unsloth and is_deepspeed_zero3_enabled():
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")

    if data_args.neat_packing and not data_args.packing:
        logger.warning("`neat_packing` requires `packing` is True. Change it to True.")
        data_args.packing = True

    _verify_model_args(model_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)

    if (
        training_args.do_train
        and finetuning_args.finetuning_type == "lora"
        and model_args.quantization_bit is None
        and model_args.resize_vocab
        and finetuning_args.additional_target is None
    ):
        logger.warning("Remember to add embedding layers to `additional_target` to make the added tokens trainable.")

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ["ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning("Cannot resume from checkpoint in current stage.")
            # training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if training_args.do_train:
        process_mos_info_for_training_args(training_args)

    if finetuning_args.lr_scheduler_kwargs_json is not None:
        training_args.lr_scheduler_kwargs = finetuning_args.lr_scheduler_kwargs_json

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning(
            "Add {} to `adapter_name_or_path` to resume training from checkpoint.".format(
                training_args.resume_from_checkpoint
            )
        )

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    if data_args.packing is None:
        data_args.packing = finetuning_args.stage == "pt"

    # Log on each process the small summary
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode == ParallelMode.DISTRIBUTED,
            str(model_args.compute_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(args)
    refine_args(model_args, data_args, finetuning_args, generating_args)

    _set_transformers_logging()
    _verify_model_args(model_args, finetuning_args)
    _check_dependencies(disabled=finetuning_args.disable_version_checking)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = _parse_eval_args(args)

    _set_transformers_logging()
    _verify_model_args(model_args, finetuning_args)
    _check_dependencies(disabled=finetuning_args.disable_version_checking)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args
