# Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_template
from ...train.rm.metric import compute_accuracy
from ...train.rm.trainer import PairwiseTrainer
from ...train.utils import create_modelcard_and_push


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)
    tokenizer, processor = template.tokenizer, template.processor
    dataset_module = get_dataset(
        tokenizer, model_args, data_args, training_args, stage="rm", template=template
    )
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=training_args.do_train, add_valuehead=True)

    data_collator = PairwiseDataCollatorWithPadding(tokenizer, template=template, pad_to_multiple_of=8)

    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=model,
        data_args=data_args,
        finetuning_args=finetuning_args,
        processor=processor,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=compute_accuracy,
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
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
