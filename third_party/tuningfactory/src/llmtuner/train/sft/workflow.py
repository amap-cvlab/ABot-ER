# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

import copy
import math
from typing import TYPE_CHECKING, List, Optional

from ...data import (
    SFTDataCollatorWith4DAttentionMask,
    SFTDataCollatorWithSequenceParallel,
    get_dataset,
)
from ...extras.callbacks import (
    DynamicInterleaveCallBack,
    PredictInTrainingCallback,
    SaveProcessorCallback,
    SaveReftCallback,
)
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import get_logits_processor
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...extras.timing import phase
from ...model import load_model, load_template
from ...train.sft.trainer import CustomSeq2SeqTrainer
from ...train.utils import create_modelcard_and_push
from .metric import ComputeMetrics, compute_accuracy, eval_logit_processor


logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_sft(
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

    with phase("load_template"):
        template = load_template(model_args, template=data_args.template, enable_thinking=data_args.enable_thinking)

    tokenizer, processor = template.tokenizer, template.processor
    with phase("get_dataset"):
        dataset_module = get_dataset(
            tokenizer,
            model_args,
            data_args,
            training_args,
            stage="sft",
            template=template,
            predict_with_generate=predict_mode == "train_and_predict",
        )
    with phase("load_model"):
        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=training_args.do_train, full_determinism=training_args.full_determinism)

    predict_dataset = dataset_module.pop("predict_dataset", None)

    if predict_mode == 'predict_only':
        tokenizer.padding_side = 'left'

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    if model_args.sequence_parallel_size == 1:
        data_collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=tokenizer,
            model=model if not training_args.predict_with_generate else None,
            template=template,
            pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            block_diag_attn=model_args.block_diag_attn,
            attn_implementation=getattr(model.config, "_attn_implementation", None),
            compute_dtype=model_args.compute_dtype,
        )
    else:
        data_collator = SFTDataCollatorWithSequenceParallel(
            tokenizer=tokenizer,
            model=model if not training_args.predict_with_generate else None,
            template=template,
            pad_to_multiple_of=math.lcm(8, 2 * model_args.sequence_parallel_size),  # for shift short attention
            label_pad_token_id=IGNORE_INDEX,
            sequence_parallel_size=model_args.sequence_parallel_size,
            sequence_parallel_mode=model_args.sequence_parallel_mode,
            rank=training_args.process_index % model_args.sequence_parallel_size,
        )

    if processor is not None and finetuning_args.finetuning_type == "full" and not finetuning_args.freeze_module_prefix:
        logger.warning("Training MLLM model without freezing vision tower may lead to worse performance.")

    if data_args.dynamic_probs:
        callbacks = [DynamicInterleaveCallBack()] + callbacks

    if finetuning_args.finetuning_type == "reft":
        callbacks = [SaveReftCallback()] + callbacks

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    compute_metrics = ComputeMetrics(tokenizer)
    # Initialize our Trainer
    with phase("trainer_init"):
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            data_args=data_args,
            finetuning_args=finetuning_args,
            processor=processor,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics if predict_mode == 'predict_only' else compute_accuracy,
            preprocess_logits_for_metrics=None if predict_mode else eval_logit_processor,
            sequence_parallel_size=model_args.sequence_parallel_size,
            **dataset_module,
        )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + \
        (tokenizer.extra_special_tokens_ids if is_transformers_version_greater_than("5.0.0") else tokenizer.additional_special_tokens_ids)
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
        with phase("train"):
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        with phase("save_model"):
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
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
