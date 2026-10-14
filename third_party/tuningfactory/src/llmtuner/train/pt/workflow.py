# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling, Trainer

from ...data import get_dataset
from ...data.collator import MultiModalDataCollatorForSeq2Seq
from ...extras.callbacks import SaveProcessorCallback
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_template
from ...train.utils import create_modelcard_and_push
from .trainer import CustomTrainer


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)
    tokenizer, processor = template.tokenizer, template.processor
    data_args.packing = False if processor is not None else data_args.packing
    dataset_module = get_dataset(
        tokenizer, model_args, data_args, training_args, stage="pt", template=template
    )
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if processor is not None:
        # mllm pre constructed labels
        data_collator = MultiModalDataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=8, template=template
        )

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_args=data_args,
        processor=processor,
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
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")

        if isinstance(dataset_module.get("eval_dataset"), dict):
            for key in dataset_module["eval_dataset"].keys():
                try:
                    perplexity = math.exp(metrics[f"eval_{key}_loss"])
                except OverflowError:
                    perplexity = float("inf")

                metrics[f"eval_{key}_perplexity"] = perplexity
        else:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["eval_perplexity"] = perplexity

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
