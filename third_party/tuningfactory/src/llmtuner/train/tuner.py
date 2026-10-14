import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..extras.interactive_ckpt import interactive_ckpt_enable
from ..extras.callbacks import (
    CustomEarlyStoppingCallback,
    InteractiveCkptCallback,
    LogCallback,
    PerfTrackerCallback,
    SaveTokenizerCallback,
    WorldWritableOutputCallback,
    make_world_writable,
)
from ..extras.logging import get_logger
from ..extras.misc import log_meta_to_openlm, patch_model_save_pretrained
from ..extras.timing import dump as dump_phase_timings, mark_run_start, print_summary as print_phase_summary
from ..hparams import get_infer_args, get_train_args
from ..model import load_model_and_template
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)

# avoid multi-rank race condition on automap files copy
import transformers.dynamic_module_utils as _dmu
_dmu.HF_MODULES_CACHE = os.path.join(_dmu.HF_MODULES_CACHE, f"rank_{os.getenv('RANK', '0')}")


# ml_tracker 0.1.37 的 integration 子模块仍 from transformers import TFPreTrainedModel, is_tf_available，
# transformers 5.x 已删除这两个 TF 相关符号 → 直接 import 会 ImportError，导致 report_to=ml_tracker
# 在 HF Trainer 端找不到回调注册（"找不到 MLTracker 相关类"）。
# 这里在 import 前补两个轻量 stub，让 ml_tracker 子模块能装载，TF 路径反正本项目永远走不到。
import transformers as _tf_compat
if not hasattr(_tf_compat, 'TFPreTrainedModel'):
    _tf_compat.TFPreTrainedModel = type('TFPreTrainedModel', (), {})
if not hasattr(_tf_compat, 'is_tf_available'):
    _tf_compat.is_tf_available = lambda: False
if os.environ.get("ABOT_ENABLE_MOS") == "1":
    import ml_tracker.integration.transformers  # noqa: F401
    os.environ['ML_TRACKER_STEP'] = 'global_step'


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    mark_run_start()
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    if finetuning_args.stage != "sft" or finetuning_args.use_turbo:
        raise ValueError("This reproduction entry point supports released standard SFT recipes only.")
    training_args.remove_unused_columns = False  # for mllm and rm/kto dataset

    # log train meta to openlm
    log_meta_to_openlm(
        data_args = data_args,
        finetuning_args= finetuning_args
    )

    callbacks = [LogCallback()] if callbacks is None else callbacks

    # # 开启交互式ckpt
    # enable_success = interactive_ckpt_enable()
    # if enable_success:
    #     # 交互式ckpt callbacks
    #     callbacks.append(InteractiveCkptCallback())

    # 启动c4d perf tracker
    if os.environ.get("ENABLE_C4D_PERF_TRACKER", "0") == "1":
        try:
            callbacks.append(PerfTrackerCallback())
        except Exception:
            logger.exception("Failed to enable PerfTrackerCallback, skip.")

    if finetuning_args.stage != "ppo":
        # ppo trainer has no callback handler
        callbacks.append(SaveTokenizerCallback())
        callbacks.append(WorldWritableOutputCallback())

    if (
        finetuning_args.early_stopping_patience is not None
        and finetuning_args.stage != "ppo"
    ):
        callbacks.append(
            CustomEarlyStoppingCallback(
                finetuning_args.early_stopping_patience,
                finetuning_args.early_stopping_threshold,
            )
        )

    # Apply global patches for model saving before running pipelines
    patch_model_save_pretrained()

    try:
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    finally:
        # 无论成功或异常，落 phase 计时（仅 rank 0 内部已守护）；保证瓶颈分析数据不丢
        dump_phase_timings(training_args.output_dir)
        print_phase_summary()
        if training_args.should_save:
            make_world_writable(training_args.output_dir)


def export_model(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir`.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    model, template = load_model_and_template(
        model_args, finetuning_args, template=data_args.template, add_valuehead=finetuning_args.stage == "rm"
    )
    tokenizer, processor = template.tokenizer, template.processor

    if getattr(model, "quantization_method", None) and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None):
        model = model.to("cpu")
    elif hasattr(model.config, "torch_dtype"):
        model = model.to(getattr(model.config, "torch_dtype")).to("cpu")
    else:
        model = model.to(torch.float16).to("cpu")
        setattr(model.config, "torch_dtype", torch.float16)

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)
    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")

    if os.environ.get("ABOT_ENABLE_MOS") != "1":
        return
    from openlm_hub import get_mos_version_uri_in_mdl_task, push_to_mos
    try:
        mos_uri = get_mos_version_uri_in_mdl_task()
        push_to_mos(model_args.export_dir, checkpoint_id="merged", mos_version_uri=mos_uri)
    except Exception as e:
        logger.warning(f"Cannot push to mos due to {e}")


if __name__ == "__main__":
    run_exp()
