import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import is_deepspeed_available
from transformers import Seq2SeqTrainer
from transformers.training_args import OptimizerNames

from ...extras.callbacks import PushToMosCallback, SaveProcessorCallback
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.packages import is_transformers_version_greater_than
from ..mixin import OpenlmHubUtils
from .various_divergence import VariousDivergence


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams.finetuning_args import FinetuningArguments

if is_deepspeed_available():
    import deepspeed

logger = get_logger(__name__)


class DistillationTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(self, data_args, finetuning_args, processor: Optional["ProcessorMixin"], *args, **kwargs):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        self.finetuning_args: "FinetuningArguments" = finetuning_args
        self.teacher_model = kwargs.pop("teacher_model")
        self.padding_id = IGNORE_INDEX

        super().__init__(**kwargs)
        self.openlm_hub_utils = OpenlmHubUtils()

        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.kl_loss = VariousDivergence(self.finetuning_args, self.padding_id)

        if self.is_deepspeed_enabled:
            if not (
                getattr(self.teacher_model, "is_loaded_in_8bit", False)
                or getattr(self.teacher_model, "is_loaded_in_4bit", False)
            ):  # quantized models are already set on the correct device
                self.teacher_model = self._prepare_deepspeed(self.teacher_model)
        else:
            # https://github.com/huggingface/accelerate/blob/31fd2b1ad6b9c1cd1480568399a311b3caaf62dc/src/accelerate/accelerator.py#L1341
            # when evaluation_mode=True, acceleare  will not set the FSDP on the model
            self.teacher_model = self.accelerator.prepare_model(
                self.teacher_model, evaluation_mode=False if self.is_fsdp_enabled else True
            )
            self.teacher_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if self.openlm_hub_utils.mos_uri is not None:
            self.add_callback(PushToMosCallback(self, self.args, self.model.config, data_args.cutoff_len))

    # copy from DPOTrainer
    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, *args, **kwargs)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns += [
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_labels",
            "student_input_ids",
            "student_attention_mask",
            "student_labels",
        ]

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        teacher_inputs = inputs["teacher_features"]
        teacher_inputs = self._prepare_inputs(teacher_inputs)
        self.teacher_model.eval()
        with torch.no_grad():
            _ = teacher_inputs.pop("labels")
            teacher_logits = self.teacher_model(**teacher_inputs).logits

        student_inputs = inputs["student_features"]
        student_inputs = self._prepare_inputs(student_inputs)

        with self.compute_loss_context_manager():
            return_outputs = True
            student_loss = self.compute_loss(
                model, student_inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
            # calculate distill loss
            student_logits = student_loss[1].logits
            if teacher_logits.shape[-1] != student_logits.shape[-1]:
                teacher_logits = teacher_logits[:, :, : min(student_logits.shape[-1], teacher_logits.shape[-1])]
            attention_mask = None
            if self.finetuning_args.distill_on_prompt:
                attention_mask = student_inputs["attention_mask"]
            distill_loss = self.kl_loss(
                student_logits, teacher_logits, student_inputs["labels"], attention_mask=attention_mask
            )
            loss = (1 - self.finetuning_args.distill_loss_weight) * student_loss[
                0
            ] + self.finetuning_args.distill_loss_weight * distill_loss

        del student_inputs
        del teacher_inputs
        del inputs

        kwargs = {}
        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)

        train_eval = "train"
        metrics = {}
        metrics[f"{train_eval}_student_loss"] = student_loss[0].detach().mean().cpu()
        metrics[f"{train_eval}_distill_loss"] = distill_loss.detach().mean().cpu()
        self.store_metrics(metrics, train_eval)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _move_right_padding_token_to_left(self, input_tensor, token):
        new_inputs = []
        for _ids in input_tensor:
            _ids_list = _ids.tolist()
            remove_pad_id_cnt = 0
            while _ids_list and _ids_list[-1] == token:
                _ids_list.pop()
                remove_pad_id_cnt += 1
            new_inputs.append([token] * remove_pad_id_cnt + _ids_list)
        new_inputs = torch.tensor(new_inputs, dtype=torch.long).cuda()
        return new_inputs

    def tokenizer_transform_right_to_left(self, inputs):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_masks = inputs["attention_mask"]

        inputs["input_ids"] = self._move_right_padding_token_to_left(input_ids, self.processing_class.pad_token_id)
        inputs["labels"] = self._move_right_padding_token_to_left(labels, IGNORE_INDEX)
        inputs["attention_mask"] = self._move_right_padding_token_to_left(attention_masks, 0)

    def open_left_padding_mode(self):
        logger.info("[CustomSeq2SeqTrainer] left padding mode is open")
        self.left_padding = True

    def close_left_padding_mode(self):
        logger.info("[CustomSeq2SeqTrainer] left padding mode is close")

        self.left_padding = False

    def get_left_padding_mode(self):
        if not hasattr(self, "left_padding"):
            self.left_padding = False
        return self.left_padding

    def set_gen_kwargs(self, gen_kwargs):
        self.gen_kwargs = gen_kwargs

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if (
            self.processing_class.padding_side == "right"
            and self.args.predict_with_generate
            and self.get_left_padding_mode()
        ):
            self.tokenizer_transform_right_to_left(inputs)
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **self.gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.processing_class.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.processing_class.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
            return res

    def process_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        predict_results.metrics.pop("predict_loss", None)
        self.log_metrics("predict", predict_results.metrics)
        self.save_metrics("predict", predict_results.metrics)
        generate_results = self.save_predictions(dataset, predict_results)
        logger.info(generate_results)
