# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset
from ...data.collator import MultiModalDataCollatorForSeq2Seq
from ...extras.callbacks import SaveProcessorCallback
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.packages import is_turbo_available
from ...extras.ploting import plot_loss
from ...model import load_template, load_turbo_model


if is_turbo_available():
    from transformers_turbo import TurboTrainer

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_pt_turbo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)
    tokenizer, processor = template.tokenizer, template.processor
    data_args.packing = False if processor is not None else data_args.packing
    data_args.cutoff_len += 1
    dataset_module = get_dataset(
        tokenizer, model_args, data_args, training_args, stage="pt", template=template
    )
    data_args.cutoff_len -= 1
    model = load_turbo_model(
        tokenizer, training_args, model_args, finetuning_args, is_trainable=training_args.do_train
    )

    data_collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        is_turbo=True,
        pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    data_collator.template = template
    callbacks = [SaveProcessorCallback(processor)] + callbacks

    # Initialize our Trainer
    trainer = TurboTrainer(
        model=model,
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
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
