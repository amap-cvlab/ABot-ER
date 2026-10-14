from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset
from ...extras.callbacks import SaveProcessorCallback
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_turbo_available
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_template, load_turbo_model


if is_turbo_available():
    from transformers_turbo.models import AutoConfig, AutoModel
    from transformers_turbo.trainer import DPOTrainer
    from transformers_turbo.trainer.dpo_config import DPOConfig

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_dpo_turbo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    assert finetuning_args.pref_ftx == 0, "pref_ftx not supported"
    assert finetuning_args.pref_loss in ["sigmoid", "orpo"], f"pref_loss {finetuning_args.pref_loss} not supported"

    template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)
    tokenizer, processor = template.tokenizer, template.processor
    data_args.cutoff_len += 1  # turbo will shift the input_ids and labels
    dataset_module = get_dataset(
        tokenizer, model_args, data_args, training_args, stage="rm", template=template
    )
    model = load_turbo_model(tokenizer, training_args, model_args, finetuning_args, is_trainable=training_args.do_train)
    pad_of = 128 * (training_args.tensor_model_parallel_size or 1) * (training_args.context_parallel_size or 1)
    if (training_args.expert_model_parallel_size or 1) > 1:
        max_len = data_args.cutoff_len - 1  # all pad to max when expert parallel
        pad_of = max_len + (-max_len) % pad_of
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        tokenizer=tokenizer,
        is_turbo=True,
        pad_to_multiple_of=pad_of,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    callbacks = [SaveProcessorCallback(processor)] + callbacks

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None:
            ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path, training_args)
            ref_model = AutoModel.from_config(ref_config)
            ref_model.load_state_dict(model.state_dict())
        else:
            ref_model = AutoModel.from_pretrained(finetuning_args.ref_model, training_args)
    else:
        ref_model = None

    dpo_config = DPOConfig(
        beta=finetuning_args.pref_beta,
        label_smoothing=finetuning_args.dpo_label_smoothing,
        pref_loss=finetuning_args.pref_loss,
    )

    # Initialize our Trainer
    trainer = DPOTrainer(
        model=model,
        train_config=dpo_config,
        ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards without a reference model
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
