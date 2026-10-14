import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from ...extras.callbacks import PushToMosCallback, SaveProcessorCallback
from ...extras.logging import get_logger
from ...extras.packages import is_safetensors_available, is_transformers_version_greater_than
from ..mixin import OpenlmHubUtils
from ..utils import save_trl_vhead_model


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class PairwiseTrainer(Trainer):
    r"""Inherits Trainer to compute pairwise loss."""

    def __init__(
        self, data_args, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.openlm_hub_utils = OpenlmHubUtils()

        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if self.openlm_hub_utils.mos_uri is not None:
            self.add_callback(PushToMosCallback(self, self.args, self.model.config, data_args.cutoff_len))

    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, torch.Tensor], return_outputs: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple.
        See: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L3842
        """
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        batch_size = inputs["input_ids"].size(0) // 2
        input_embeds_len = values.size(1)
        input_ids_len = inputs["input_ids"].size(1)
        # for llava-like models where input_embeds_len is longer than input_ids_len
        len_diff = input_embeds_len - input_ids_len

        chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)
        chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1 + len_diff))
        rejected_scores = rejected_rewards.gather(
            dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1 + len_diff)
        )
        chosen_scores, rejected_scores = chosen_scores.squeeze(), rejected_scores.squeeze()

        loss = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()
        if return_outputs:
            return loss, (loss, chosen_scores, rejected_scores)
        else:
            return loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))

            writer.write("\n".join(res))

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        save_trl_vhead_model(
            self.model,
            output_dir,
            state_dict=state_dict,
            dup_head_weight=True,
            save_safetensors=is_safetensors_available(),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
