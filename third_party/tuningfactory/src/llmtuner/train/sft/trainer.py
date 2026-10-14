import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import torch.distributed as dist
from accelerate.data_loader import prepare_data_loader
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Seq2SeqTrainer
from transformers.trainer import _is_peft_model
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from ...extras.callbacks import PushToMosCallback, SaveProcessorCallback
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.packages import is_transformers_version_greater_than
from ...extras.sample_weight import compute_sample_weighted_sft_loss
from ..mixin import OpenlmHubUtils
from ..utils import get_batch_logps


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import DataArguments, FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        sequence_parallel_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)

        self.openlm_hub_utils = OpenlmHubUtils()

        if processor is not None:
            self.model_accepts_loss_kwargs = False
        # self.model_accepts_loss_kwargs = False

        # for sequence parallel
        self.sequence_parallel_size = sequence_parallel_size
        self._has_dummy_forwarded = False
        self.loss_fct = CrossEntropyLoss(reduction="sum")

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self._stored_log_loss = defaultdict(list)
        self._nonfinite_debug_dumped = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if self.openlm_hub_utils.mos_uri is not None:
            self.add_callback(PushToMosCallback(self, self.args, self.model.config, data_args.cutoff_len))

        if finetuning_args.use_dft_loss:
            self.dft_weight_min = finetuning_args.dft_weight_min
            self.dft_start_epoch = finetuning_args.dft_start_epoch
            self.compute_loss_func = self._compute_dft_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop("sample_weight", None)
        if not self.finetuning_args.use_sample_weighted_loss:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,
            )
        if sample_weight is None:
            raise ValueError("`use_sample_weighted_loss` requires the dataset sample_weight column")
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("sample-weighted SFT requires labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("sample-weighted SFT requires model logits")
        loss, per_example, normalized = compute_sample_weighted_sft_loss(
            logits, labels, sample_weight, self.finetuning_args.sample_weight_mean,
        )
        train_eval = "train" if torch.is_grad_enabled() else "eval"
        self.store_metrics({
            "sample_weight_mean_batch": normalized.detach().mean().item(),
            "sample_ce_mean_batch": per_example.detach().mean().item(),
        }, train_eval)
        return (loss, outputs) if return_outputs else loss

    def _compute_dft_loss(self, outputs, labels, num_items_in_batch=None):
        logits = outputs.get("logits")
        if logits is None:
            return outputs.get("loss", torch.tensor(0.0))

        logits = logits.float()
        vocab_size = logits.size(-1)
        labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(logits.device)

        per_token_loss = torch.nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="none")
        loss_mask = (shift_labels != -100).float()
        num_valid = loss_mask.sum().clamp(min=1)  # avoid divide by zero when all labels are -100

        use_dft = self.state.epoch >= self.dft_start_epoch

        with torch.no_grad():
            if use_dft:
                dft_weights = torch.exp(-per_token_loss)
                if self.dft_weight_min > 0:
                    clipped_count = ((dft_weights < self.dft_weight_min).float() * loss_mask).sum()
                    clipped_ratio = clipped_count / num_valid
                    dft_weights = torch.clamp(dft_weights, min=self.dft_weight_min)
                else:
                    clipped_ratio = torch.tensor(0.0)
            else:
                dft_weights = torch.ones_like(per_token_loss)
                clipped_ratio = torch.tensor(0.0)

        weighted_losses = per_token_loss * dft_weights * loss_mask

        if num_items_in_batch is not None:
            total_loss = weighted_losses.sum()
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(total_loss.device)
            loss = total_loss / num_items_in_batch
        else:
            loss = weighted_losses.sum() / num_valid

        train_eval = "train" if torch.is_grad_enabled() else "eval"
        self.store_metrics({"dft_clipped_ratio": clipped_ratio.item()}, train_eval)

        return loss

    def get_ift_lambda(self):
        # TODO: lambda scheduler
        return 0.2

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

        for key, value in self._stored_log_loss.items():
            sum_value = torch.tensor(value, device=value[0].device).sum()
            mean_value = self._nested_gather(sum_value).mean().item()
            # 使用self.state.logging_steps会导致打印的个别step的loss值不准确
            logs[key] = round(mean_value / self.state.logging_steps, 4)
        self._stored_log_loss = defaultdict(list)

        return super().log(logs, *args, **kwargs)

    def get_cumsum_weight(
        self, logps, loss_mask, gamma=1, propagation_type="loss", propagation_norm="L1", propagation_side="left"
    ) -> torch.FloatTensor:
        from torch_discounted_cumsum import discounted_cumsum_left, discounted_cumsum_right

        if gamma == 0:
            return torch.ones_like(logps)
        if propagation_type == "mask":
            cumsum_item = loss_mask
        elif propagation_type == "loss":
            cumsum_item = -logps / loss_mask.sum(-1).unsqueeze(-1)
        elif propagation_type == "logps":
            cumsum_item = -logps
        else:
            raise ValueError(f"unknown propagation_type {propagation_type}")

        cumsum_item[loss_mask == 0] = 0

        if propagation_side == "right":
            if gamma == 1:
                cumsum_weight = torch.cumsum(cumsum_item, dim=-1)
            else:
                cumsum_weight = discounted_cumsum_left(cumsum_item, gamma=gamma)

            cumsum_weight[loss_mask == 0] = 1e6
            cumsum_weight += cumsum_weight[cumsum_weight.nonzero(as_tuple=True)].min()

            if propagation_norm == "L1":  # sharp level 1
                cumsum_weight = 1 / (cumsum_weight)
            elif propagation_norm == "L2":  # sharp level 2
                cumsum_weight = 1 / (cumsum_weight) ** 2
            elif propagation_norm == "softmax":  # sharp level 3
                cumsum_weight = torch.softmax(1 / cumsum_weight, dim=-1)
            elif propagation_norm == "log":  # sharp level 0
                cumsum_weight = 1 / torch.log(cumsum_weight + 1)
            else:
                raise ValueError(f"unknown propagation_norm {propagation_norm}")

        elif propagation_side == "left":
            if gamma == 1:
                cumsum_weight = torch.flip(torch.cumsum(torch.flip(cumsum_item, [-1]), dim=-1), [-1])
            else:
                cumsum_weight = discounted_cumsum_right(cumsum_item, gamma=gamma)
            cumsum_weight[loss_mask == 0] = 0

            if propagation_norm == "L1":  # sharp level 2
                cumsum_weight = cumsum_weight
            elif propagation_norm == "L2":  # sharp level 3
                cumsum_weight = cumsum_weight**2
            elif propagation_norm == "softmax":  # sharp level 1
                cumsum_weight = torch.softmax(cumsum_weight, dim=-1)
            elif propagation_norm == "log":  # sharp level 0
                cumsum_weight = torch.log(cumsum_weight + 1)
            else:
                raise ValueError(f"unknown propagation_norm {propagation_norm}")

        else:
            raise ValueError(f"unknown propagation_side {propagation_side}")

        cumsum_weight[loss_mask == 0] = 0
        cumsum_weight *= loss_mask.sum(-1, keepdim=True) / cumsum_weight.sum(-1, keepdim=True)

        return cumsum_weight

    def compute_ift_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        metrics = {}
        train_eval = "train" if torch.is_grad_enabled() else "eval"

        with torch.no_grad():
            logits = model(inputs_embeds=inputs_embeds, **inputs).logits
            logps, loss_mask = get_batch_logps(logits, labels, per_token=True)
            losses_sft = -logps.sum(-1) / loss_mask.sum(-1)
            greedy_preds = torch.argmax(logits, dim=-1)[:, :-1]
            tokens_further = torch.cat((input_ids[:, 0].unsqueeze(-1), greedy_preds), dim=-1)
            input_ids_further = torch.where(labels==IGNORE_INDEX, input_ids, tokens_further)
        inputs_embeds_further = self.model.get_input_embeddings()(input_ids_further)
        ift_lambda = self.get_ift_lambda()
        inputs_embeds_further = (1 - ift_lambda) * inputs_embeds + ift_lambda * inputs_embeds_further
        logits_further = model(inputs_embeds=inputs_embeds_further, **inputs).logits
        logps_further, loss_mask = get_batch_logps(logits_further, labels, per_token=True)
        cumsum_weight = self.get_cumsum_weight(
            logps=logps_further,
            loss_mask=loss_mask,
            gamma=self.finetuning_args.ift_gamma,
        )
        losses = ((-logps_further * cumsum_weight).sum(-1) / loss_mask.sum(-1)).mean()
        # TODO: average losses in ranks
        metrics[f"{train_eval}_sft_loss"] = losses_sft.detach().mean().cpu()
        metrics[f"{train_eval}_lambda"] = ift_lambda
        metrics[f"{train_eval}_loss_further"] = (-logps_further.sum(-1) / loss_mask.sum(-1)).detach().mean().cpu()
        self.store_metrics(metrics, train_eval)
        return losses

    def training_step(self, model, inputs, *args, **kwargs):
        # TODO: sequence_parallel modes other than 'zigzag-ring' may not need dummy forward
        if not self._has_dummy_forwarded and getattr(model, "sequence_parallel_group", None) is not None:
            model.eval()
            with torch.no_grad():
                _ = model(**inputs)
            model.train()
            self._has_dummy_forwarded = True
        return super().training_step(model, inputs, *args, **kwargs)

    def get_train_dataloader(self):
        if self.model.sequence_parallel_group is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return prepare_data_loader(
            DataLoader(train_dataset, **dataloader_params),
            device=self.args.device,
            num_processes=self.args.world_size // self.sequence_parallel_size,
            process_index=self.accelerator.process_index // self.sequence_parallel_size,
            split_batches=self.accelerator.split_batches,
            put_on_device=True,
            rng_types=self.accelerator.rng_types.copy(),
            dispatch_batches=False
        )

    def _get_train_sampler(self, *args, **kwargs):
        if self.model.sequence_parallel_group is None:
            return super()._get_train_sampler(*args, **kwargs)
        logger.warning("[CustomSeq2SeqTrainer] using SequentialSampler for sequence parallel training")
        return SequentialSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        if getattr(model, "sequence_parallel_group", None) is None:
            if self.finetuning_args.pref_loss == "ift":
                return self.compute_ift_loss(model, inputs, return_outputs)
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs)
            if torch.is_tensor(loss) and not torch.isfinite(loss).all():
                labels = inputs.get("labels")
                if torch.is_tensor(labels):
                    valid_labels = labels.ne(IGNORE_INDEX).sum().item()
                    total_labels = labels.numel()
                    label_shape = tuple(labels.shape)
                else:
                    valid_labels = "unknown"
                    total_labels = "unknown"
                    label_shape = "unknown"
                rank = int(os.getenv("RANK", "0"))
                if torch.is_tensor(labels) and valid_labels == 0:
                    # all labels are -100 (no learnable tokens), replace NaN with
                    # a zero loss connected to the computation graph so DeepSpeed
                    # gradient reduction doesn't hit empty buckets
                    logits = outputs.get("logits") if isinstance(outputs, dict) else None
                    if logits is not None:
                        loss = logits.flatten()[0] * 0
                    else:
                        loss = loss.new_zeros((), requires_grad=True)
                    if isinstance(outputs, dict):
                        outputs["loss"] = loss
                else:
                    # Skip non-finite loss instead of crashing — treat as gradient explosion
                    # and replace with zero loss to keep training alive
                    debug_path = self._dump_nonfinite_debug(inputs, outputs, loss, rank)
                    logger.warning(
                        "Non-finite SFT loss detected — SKIPPING step "
                        f"(rank={rank}, loss={loss.detach()}, valid_labels={valid_labels}, "
                        f"total_labels={total_labels}, label_shape={label_shape}, debug_path={debug_path})."
                    )
                    logits = outputs.get("logits") if isinstance(outputs, dict) else None
                    if logits is not None:
                        loss = logits.flatten()[0].detach() * 0
                        loss.requires_grad_(True)
                    else:
                        loss = loss.new_zeros((), requires_grad=True)
                    if isinstance(outputs, dict):
                        outputs["loss"] = loss

            log_loss = {}
            if isinstance(outputs, dict):
                if "lm_loss" in outputs:
                    log_loss["lm_loss"] = outputs["lm_loss"]
                if "mtp_loss" in outputs:
                    log_loss["mtp_loss"] = outputs["mtp_loss"]
                if "aux_loss" in outputs:
                    log_loss["aux_loss"] = outputs["aux_loss"]

                if (
                    self.args.average_tokens_across_devices
                    and (self.model_accepts_loss_kwargs or self.compute_loss_func)
                    and num_items_in_batch is not None
                ):
                    for k, v in log_loss.items():
                        v *= self.accelerator.num_processes

                if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                    for k, v in log_loss.items():
                        v /= self.args.gradient_accumulation_steps

                for k, v in log_loss.items():
                    self._stored_log_loss[k].append(v.detach())

            return (loss, outputs) if return_outputs else loss
        else:
            # compute loss without shift labels, as we have already shifted labels in data processing when using sequence parallel
            _, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs)
            # Flatten the tokens
            logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]
            # Get vocab_size
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                vocab_size = unwrapped_model.base_model.model.config.vocab_size
            else:
                vocab_size = unwrapped_model.config.vocab_size
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits, labels)

            # weighted reduce within sequence_parallel_group
            sp_group = model.sequence_parallel_group
            loss = dist.nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)
            label_num = num_items_in_batch.clone()
            label_num = dist.nn.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)
            loss /= label_num

        return loss

    def _summarize_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": tensor.numel(),
        }
        detached = tensor.detach()
        if detached.numel() == 0:
            return summary

        if torch.is_floating_point(detached) or torch.is_complex(detached):
            finite_mask = torch.isfinite(detached)
            summary["finite"] = bool(finite_mask.all().item())
            summary["finite_ratio"] = float(finite_mask.float().mean().item())
            if finite_mask.any().item():
                finite_values = detached[finite_mask].float()
                summary["min"] = float(finite_values.min().item())
                summary["max"] = float(finite_values.max().item())
                summary["mean"] = float(finite_values.mean().item())
            if not summary["finite"] and detached.ndim >= 2 and detached.shape[-1] >= 1024:
                reduce_dims = tuple(range(detached.ndim - 1))
                bad_last_dim = (~finite_mask).any(dim=reduce_dims).nonzero(as_tuple=False).flatten()
                summary["nonfinite_last_dim_count"] = int(bad_last_dim.numel())
                summary["nonfinite_last_dim_first"] = bad_last_dim[:32].detach().cpu().tolist()
                summary["nonfinite_last_dim_last"] = bad_last_dim[-32:].detach().cpu().tolist()
        elif detached.dtype == torch.bool:
            summary["true_count"] = int(detached.sum().item())
        else:
            summary["min"] = int(detached.min().item())
            summary["max"] = int(detached.max().item())

        return summary

    def _dump_nonfinite_debug(self, inputs: Dict[str, Any], outputs: Any, loss: torch.Tensor, rank: int) -> str:
        if self._nonfinite_debug_dumped:
            return "already_dumped"

        self._nonfinite_debug_dumped = True
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            debug_path = os.path.join(
                self.args.output_dir,
                f"nonfinite_debug_rank{rank}_step{self.state.global_step}.json",
            )
            payload: Dict[str, Any] = {
                "rank": rank,
                "global_step": self.state.global_step,
                "loss": str(loss.detach()),
                "inputs": {},
                "outputs": {},
            }
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    payload["inputs"][key] = self._summarize_tensor(value)

            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if torch.is_tensor(value):
                        payload["outputs"][key] = self._summarize_tensor(value)

            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return debug_path
        except Exception as exc:
            logger.warning("Failed to dump non-finite loss debug payload: %s", exc)
            return f"dump_failed:{exc}"

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
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        attention_masks = inputs['attention_mask']

        inputs['input_ids'] = self._move_right_padding_token_to_left(input_ids, self.processing_class.pad_token_id)
        inputs['labels'] = self._move_right_padding_token_to_left(labels, IGNORE_INDEX)
        inputs['attention_mask'] = self._move_right_padding_token_to_left(attention_masks, 0)

    def open_left_padding_mode(self):
        logger.info('[CustomSeq2SeqTrainer] left padding mode is open')
        self.left_padding = True

    def close_left_padding_mode(self):
        logger.info('[CustomSeq2SeqTrainer] left padding mode is close')

        self.left_padding = False

    def get_left_padding_mode(self):
        if not hasattr(self, 'left_padding'):
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
        if self.processing_class.padding_side == "right" and self.args.predict_with_generate and self.get_left_padding_mode():
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
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.processing_class.pad_token_id
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
