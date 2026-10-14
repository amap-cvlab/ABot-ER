import re
from typing import TYPE_CHECKING, List
from types import MethodType

import torch
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras.logging import get_logger
from ..extras.misc import count_parameters
from ..extras.packages import is_pyreft_available
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model


if is_pyreft_available():
    import pyreft
    from pyreft import (
        ConsreftIntervention,
        DireftIntervention,
        LobireftIntervention,
        LoreftIntervention,
        NodireftIntervention,
        NoreftIntervention,
    )

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def _get_freeze_module_prefixes(finetuning_args: "FinetuningArguments") -> List[str]:
    if not finetuning_args.freeze_module_prefix:
        return []

    if isinstance(finetuning_args.freeze_module_prefix, str):
        prefixes = finetuning_args.freeze_module_prefix.split(",")
    else:
        prefixes = finetuning_args.freeze_module_prefix

    return [prefix.strip() for prefix in prefixes if prefix.strip()]


def _setup_full_tuning(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info("Fine-tuning method: Full")

    freeze_module_prefixes = _get_freeze_module_prefixes(finetuning_args)
    frozen_param_matched = 0
    frozen_param_count = 0
    for name, param in model.named_parameters():
        if freeze_module_prefixes and any(name.startswith(prefix) for prefix in freeze_module_prefixes):
            param.requires_grad_(False)
            frozen_param_matched += 1
            frozen_param_count += getattr(param, "ds_numel", param.numel())
        else:
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
    if freeze_module_prefixes:
        if frozen_param_matched == 0:
            raise ValueError(
                "freeze_module_prefix does not match any parameter in full tuning: "
                f"{freeze_module_prefixes}"
            )
        logger.info(
            "Frozen params by prefix: {:d} tensors, {:d} params ({})".format(
                frozen_param_matched, frozen_param_count, ",".join(freeze_module_prefixes)
            )
        )


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    freeze_module_prefixes = _get_freeze_module_prefixes(finetuning_args)
    if freeze_module_prefixes:
        logger.info("Fine-tuning method: Freeze with prefix")
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_module_prefixes):
                param.requires_grad_(False)
            else:
                if cast_trainable_params_to_fp32:
                    param.data = param.data.to(torch.float32)
        return

    logger.info("Fine-tuning method: Freeze")
    if hasattr(model.config, "text_config") or hasattr(model.config, "language_config"):  # multi-modal
        config = getattr(model.config, "text_config", None) or getattr(model.config, "language_config", None)
    else:
        config = model.config
    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                "`num_layers` {} should be divisible by `num_layer_trainable` {}.".format(
                    num_layers, finetuning_args.freeze_trainable_layers
                )
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)

    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info("Set trainable layers: {}".format(",".join(trainable_layers)))


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        logger.info("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
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

        if len(adapter_to_merge) > 0:
            logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

        if adapter_to_resume is not None:  # resume lora training
            if model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, is_trainable=is_trainable)
            else:
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        if (
            isinstance(finetuning_args.lora_target, list)
            and len(finetuning_args.lora_target) == 1
            and finetuning_args.lora_target[0] == "all"
        ):
            target_modules = find_all_linear_modules(model)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BITS_AND_BYTES
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "use_rslora": finetuning_args.use_rslora,
            "use_dora": finetuning_args.use_dora,
            "modules_to_save": finetuning_args.additional_target,
        }

        if model_args.use_unsloth:
            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **peft_kwargs,
            )
            model = get_peft_model(model, lora_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def _setup_reft_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PeftModel":
    if is_trainable:
        logger.info("Fine-tuning method: ReFT")
    intervention_mapping = {
        'NoreftIntervention': NoreftIntervention,
        'LoreftIntervention': LoreftIntervention,
        'ConsreftIntervention': ConsreftIntervention,
        'LobireftIntervention': LobireftIntervention,
        'DireftIntervention': DireftIntervention,
        'NodireftIntervention': NodireftIntervention,
    }

    if model_args.adapter_name_or_path is not None:
        reft_model = pyreft.ReftModel.load(model_args.adapter_name_or_path[0], model)
        reft_model.set_device(model.device)
        logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))
    else:
        reft_layers = finetuning_args.reft_layers
        assert reft_layers, "reft_layers parameter missing"

        if reft_layers != "all":
            layers = [int(l) for l in reft_layers.split(",")]
        else:
            layers = list(range(config.num_hidden_layers))
        logger.info(f'Applying Reft to module: {layers}')
        representations = []
        for layer_index in layers:
            intervention_config = {
                'layer': layer_index,
                'component': "block_output",
                'low_rank_dimension': finetuning_args.reft_rank,
                'intervention':  intervention_mapping[finetuning_args.reft_intervention_type](
                    embed_dim=config.hidden_size, low_rank_dimension=finetuning_args.reft_rank)
            }
            representations.append(intervention_config)

        reft_config = pyreft.ReftConfig(representations=representations)
        reft_model = pyreft.get_reft_model(model, reft_config)

    reft_model.reft_config = reft_model.config
    reft_model.config = reft_model.model.config
    reft_model.dtype = reft_model.model.dtype
    reft_model.device = reft_model.model.device

    if is_trainable:
        logger.info("########## reft model trainable parameters ##########")
        reft_model.print_trainable_parameters()

    def _pre_forward_hook(module, args, kwargs):
        if 'base' in kwargs:
            return args, kwargs

        if 'input_ids' not in kwargs:
            raise ValueError('Input does not contain `input_ids`, maybe the model does not support ReFT.')
        # run intervened forward pass
        unit_locations = None
        if 'intervention_locations' in kwargs:
            if kwargs['intervention_locations'].dim() == 3:
                unit_locations = {
                    'sources->base': (None, kwargs['intervention_locations'].permute(1, 0, 2).tolist())
                }
            else:
                # this is dummy for lora only baseline
                unit_locations = {'sources->base': (None, 0)}
        kwargs = {
            'base': {
                'input_ids': kwargs['input_ids'],
                'attention_mask': kwargs['attention_mask']
            },
            'unit_locations': unit_locations,
            'labels': kwargs['labels'],
            'subspaces': kwargs['subspaces'].permute(1, 0, 2).tolist() if 'subspaces' in kwargs else None
        }
        return args, kwargs

    def _post_forward_hook(module, args, kwargs, outputs):
        return outputs[1]

    def _generate(self, **kwargs):
        # run intervened forward pass
        unit_locations = None
        if 'intervention_locations' in kwargs:
            if kwargs['intervention_locations'].dim() == 3:
                unit_locations = {
                    'sources->base': (None, kwargs['intervention_locations'].permute(1, 0, 2).tolist())
                }
            else:
                # this is dummy for lora only baseline
                unit_locations = {'sources->base': (None, 0)}

        _kwargs = {
            'base': {
                'input_ids': kwargs.pop('input_ids'),
                'attention_mask': kwargs.pop('attention_mask')
            },
            'unit_locations': unit_locations,
            'subspaces': kwargs.pop('subspaces').permute(1, 0, 2).tolist() if 'subspaces' in kwargs else None
        }
        _kwargs = {**_kwargs, **kwargs}
        return self.generate_origin(**_kwargs)[1]

    reft_model.generate_origin = reft_model.generate
    reft_model.generate = MethodType(_generate, reft_model)
    reft_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)
    reft_model.register_forward_hook(_post_forward_hook, with_kwargs=True)

    return reft_model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantized models can only be used for the LoRA tuning.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 and not fsdp (zero3 or fsdp already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16:
        logger.info("Pure bf16 detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and (is_deepspeed_zero3_enabled() or is_fsdp_enabled()):
        logger.info("ZeRO3 / FSDP detected, remaining trainable params in float32.")
    else:
        logger.info("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
        trainable_params, all_param = count_parameters(model)
        if trainable_params == 0 or trainable_params >= all_param:
            raise ValueError(
                f"freeze tuning does not work. trainable params: {trainable_params}, all params: {all_param}",
            )
    elif finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    elif finetuning_args.finetuning_type == "reft":
        model = _setup_reft_tuning(
            config, model, model_args, finetuning_args, is_trainable
        )
    else:
        if is_trainable:
            raise NotImplementedError("Unknown finetuning type: {}.".format(finetuning_args.finetuning_type))

    return model
