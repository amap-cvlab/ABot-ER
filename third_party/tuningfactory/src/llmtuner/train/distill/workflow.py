# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

import copy
from typing import TYPE_CHECKING, List, Optional

from datasets import concatenate_datasets

from ...data import (
    DistillDataCollatorWith4DAttentionMask,
    get_dataset,
)
from ...data.utils import get_sample_data
from ...extras.callbacks import (
    DynamicInterleaveCallBack,
    PredictInTrainingCallback,
)
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_template
from ...train.distill.trainer import DistillationTrainer
from ...train.sft.metric import ComputeMetrics, compute_accuracy, eval_logit_processor


logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def load_teacher_or_student_module(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        training_args: "Seq2SeqTrainingArguments",
        predict_with_generate: bool,
        target: str = "student",
):
    model_args = copy.deepcopy(model_args)
    data_args = copy.deepcopy(data_args)
    finetuning_args = copy.deepcopy(finetuning_args)

    if target == "teacher":
        model_args.model_name_or_path = model_args.teacher_model_name_or_path
        data_args.template = data_args.teacher_template
        if model_args.teacher_adapter_name_or_path:
            finetuning_args.finetuning_type = "lora"
            model_args.adapter_name_or_path = [path.strip() for path in model_args.teacher_adapter_name_or_path.split(",")]
        else:
            finetuning_args.finetuning_type = "full"

    template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)

    if target == "student":
        model = load_model(template.tokenizer, model_args, finetuning_args, is_trainable=True)
    elif target == "teacher":
        model = load_model(template.tokenizer, model_args, finetuning_args, is_trainable=False)
        model.eval()

    dataset_module = get_dataset(
        template.tokenizer,
        model_args,
        data_args,
        training_args,
        stage="sft",
        template=template,
        predict_with_generate=predict_with_generate,
    )
    sample_data = get_sample_data(dataset_module["train_dataset"], use_subprocess=True)
    for name in sample_data.keys():
        dataset_module["train_dataset"] = dataset_module["train_dataset"].rename_column(name, f"{target}_{name}")
    return model, template, dataset_module


def run_distill(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    predict_mode = ''
    if training_args.do_train and training_args.predict_with_generate:
        predict_mode = 'train_and_predict'
    if training_args.predict_with_generate and training_args.do_predict:
        predict_mode = 'predict_only'

    # load student related module
    student_model, student_template, student_dataset_module = load_teacher_or_student_module(
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
        training_args=training_args,
        predict_with_generate=predict_mode == "train_and_predict",
        target="student"
    )
    tokenizer, processor = student_template.tokenizer, student_template.processor

    # load teacher related module
    teacher_model, teacher_template, teacher_dataset_module = load_teacher_or_student_module(
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
        training_args=training_args,
        predict_with_generate=predict_mode == "train_and_predict",
        target="teacher"
    )
    student_dataset_module["train_dataset"] = concatenate_datasets([student_dataset_module["train_dataset"], teacher_dataset_module["train_dataset"]],axis=1)

    predict_dataset = student_dataset_module.pop("predict_dataset", None)

    if predict_mode == 'predict_only':
        tokenizer.padding_side = 'left'

    if getattr(student_model, "is_quantized", False) and not training_args.do_train:
        setattr(student_model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DistillDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer,
        template=student_template,
        teacher_template=teacher_template,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    if data_args.dynamic_probs:
        callbacks = [DynamicInterleaveCallBack()] + callbacks

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    compute_metrics = ComputeMetrics(tokenizer)
    # Initialize our Trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        processor=processor,
        args=training_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics if predict_mode == 'predict_only' else compute_accuracy,
        preprocess_logits_for_metrics=None if predict_mode else eval_logit_processor,
        **student_dataset_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    trainer.set_gen_kwargs(gen_kwargs)

    if predict_mode == 'train_and_predict':
        predict_data_collator = copy.copy(data_collator)
        predict_data_collator.predict_mode = True
        pit_callback = PredictInTrainingCallback(trainer, predict_dataset, compute_metrics, predict_data_collator, gen_kwargs)
        trainer.add_callback(pit_callback)

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
    if training_args.do_eval and not data_args.dynamic_probs:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(student_dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(student_dataset_module["eval_dataset"], predict_results)