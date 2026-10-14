import inspect
from typing import TYPE_CHECKING, Optional

import torch
from peft import LoraConfig, LoraModel, PeftModel, get_peft_model

from ..extras.logging import get_logger
from ..extras.misc import count_parameters
from ..extras.packages import is_turbo_available


if is_turbo_available():
    try:
        from transformers_turbo.adapters import (
            apply_megatron_lora,
            find_all_embedding_modules,
            find_all_linear_modules,
            find_all_router_modules,
            set_linear_is_expert,
        )
    except ImportError:
        pass

    from transformers_turbo.models import AutoModel, TurboGPTModel


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def get_target_modules(model: "PreTrainedModel", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"):
    target_modules = finetuning_args.lora_target
    if "all-linear" in finetuning_args.lora_target:
        target_modules.remove("all-linear")
        target_modules += find_all_linear_modules(model)
    if "all-embedding" in finetuning_args.lora_target:
        target_modules.remove("all-embedding")
        target_modules += find_all_embedding_modules(model)
    if "all-router" in finetuning_args.lora_target:
        target_modules.remove("all-router")
        target_modules += find_all_router_modules(model)

    return target_modules


def setup_lora_training(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
):
    model.enable_input_require_grads()
    if model_args.adapter_name_or_path is not None:
        adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        for adapter in adapter_to_merge:
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

    if is_trainable:
        target_modules = get_target_modules(model, model_args, finetuning_args)

        if finetuning_args.additional_target:
            if "all-embedding" in finetuning_args.additional_target:
                finetuning_args.additional_target.remove("all-embedding")
                finetuning_args.additional_target += find_all_embedding_modules(model)

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "use_rslora": finetuning_args.use_rslora,
            "modules_to_save": finetuning_args.additional_target,
        }
        if model.config.num_moe_experts is not None:
            peft_kwargs["rank_pattern"] = {
                p: max(finetuning_args.lora_rank // model.config.moe_router_topk, 1)
                for p in ["linear_fc1", "linear_fc2"]
            }

        lora_config = LoraConfig(
            **peft_kwargs,
        )
        model = get_peft_model(model, lora_config)

        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def load_turbo_model(
    tokenizer: "PreTrainedTokenizer",
    training_args: "TrainingArguments",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,  # TODO: support ref model
    add_valuehead: Optional[bool] = False,  # TODO: support reward model
):
    kwargs = {}
    # avoid using low version turbo
    if "tokenizer" in inspect.signature(TurboGPTModel.from_pretrained).parameters:
        kwargs["tokenizer"] = tokenizer
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args, **kwargs)
    if finetuning_args.finetuning_type == "lora":
        apply_megatron_lora()
        set_linear_is_expert(model[0])
        model.models[0] = setup_lora_training(model[0], model_args, finetuning_args, is_trainable)
    elif finetuning_args.finetuning_type == "freeze" and finetuning_args.freeze_module_prefix is not None:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in finetuning_args.freeze_module_prefix):
                param.requires_grad_(False)
    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:d}".format(all_param)
    logger.info(param_stats)
    return model
