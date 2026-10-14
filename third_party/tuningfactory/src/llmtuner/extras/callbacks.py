import json
import math
import os
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    IntervalStrategy,
    PredictionOutput,
    has_length,
    speed_metrics,
)

from .constants import LOG_FILE_NAME
from .interactive_ckpt import interactive_ckpt_finish_save, interactive_ckpt_should_save
from .logging import get_logger
from .packages import is_transformers_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig, Trainer, TrainerControl, TrainerState, TrainingArguments


logger = get_logger(__name__)


def make_world_writable(path: str) -> None:
    # Preserve the caller's umask in portable runs.
    if os.environ.get("ABOT_ALLOW_SHARED_OUTPUT_CHMOD") != "1":
        return
    if not path or not os.path.exists(path):
        return

    def add_write_bits(item_path: str) -> None:
        try:
            mode = os.stat(item_path).st_mode
            permission_bits = 0o777 if os.path.isdir(item_path) else 0o666
            os.chmod(item_path, mode | permission_bits)
        except OSError as err:
            logger.warning(f"Failed to chmod {item_path}: {err}")

    add_write_bits(path)
    if not os.path.isdir(path):
        return

    for root, dirs, files in os.walk(path):
        for name in dirs:
            add_write_bits(os.path.join(root, name))
        for name in files:
            add_write_bits(os.path.join(root, name))


class LogCallback(TrainerCallback):
    def __init__(self, runner=None):
        self.runner = runner
        self.in_training = False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.last_tput_time = self.start_time
        self.last_tput_step = 0

    def timing(self):
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
        remaining_time = (self.max_steps - self.cur_steps) * avg_time_per_step
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.last_tput_time = self.start_time
            self.last_tput_step = 0
            self.max_steps = state.max_steps
            if (
                state.is_world_process_zero
                and os.path.exists(os.path.join(args.output_dir, LOG_FILE_NAME))
                and getattr(args, "overwrite_output_dir", False)
            ):
                logger.warning("Previous log file in this folder will be deleted.")
                os.remove(os.path.join(args.output_dir, LOG_FILE_NAME))

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_predict(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", *other, **kwargs
    ):
        r"""
        Event called after a successful prediction.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if not state.is_local_process_zero:
            return

        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            reward=state.log_history[-1].get("reward", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if self.runner is not None:
            logger.info(
                "{{'loss': {:.4f}, 'learning_rate': {:2.4e}, 'epoch': {:.2f}}}".format(
                    logs["loss"] or 0, logs["learning_rate"] or 0, logs["epoch"] or 0
                )
            )

        if not state.is_world_process_zero:
            return

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, LOG_FILE_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

        cur_time = time.time()
        interval_steps = self.cur_steps - self.last_tput_step
        interval_secs = cur_time - self.last_tput_time
        total_secs = cur_time - self.start_time
        world_size = max(int(getattr(args, "world_size", 1) or 1), 1)
        per_device_bs = int(getattr(args, "per_device_train_batch_size", 1) or 1)
        grad_accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        effective_bs = per_device_bs * world_size * grad_accum
        if interval_steps > 0 and interval_secs > 0:
            throughput = {
                "step": self.cur_steps,
                "max_steps": self.max_steps,
                "interval_steps": interval_steps,
                "interval_sec": round(interval_secs, 4),
                "interval_sec_per_step": round(interval_secs / interval_steps, 4),
                "avg_sec_per_step": round(total_secs / self.cur_steps, 4) if self.cur_steps else None,
                "world_size": world_size,
                "per_device_train_batch_size": per_device_bs,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_size": effective_bs,
                "interval_samples_per_sec": round(interval_steps * effective_bs / interval_secs, 4),
                "avg_samples_per_sec": round(self.cur_steps * effective_bs / total_secs, 4)
                if total_secs > 0
                else None,
                "seen_samples": self.cur_steps * effective_bs,
                "progress_pct": round(self.cur_steps / self.max_steps * 100, 4) if self.max_steps else None,
                "eta_hours": round((self.max_steps - self.cur_steps) * (total_secs / self.cur_steps) / 3600, 4)
                if self.cur_steps and self.max_steps
                else None,
            }
            print("[taskplan-throughput] " + json.dumps(throughput, sort_keys=True), flush=True)
            self.last_tput_time = cur_time
            self.last_tput_step = self.cur_steps

    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        r"""
        Event called after a prediction step.
        """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()
     
class PerfTrackerCallback(TrainerCallback):
    def __init__(self):
        try:
            from c4d_perftracker_collector.PerfTracker import PerfTracker
            self.tracer = PerfTracker()
        except:
            self.tracer = None
        

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.tracer is not None:
            self.tracer.step() 


class SaveTokenizerCallback(TrainerCallback):
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a checkpoint save.
        """
        # online predict will panic when they get a tokenizer with padding_side = "right"
        tokenizer = kwargs.pop("tokenizer", None)
        if args.should_save and tokenizer is not None:
            output_dir = os.path.join(args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step))
            pad_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Resaved tokenizer to {output_dir} with padding_side = left")
            tokenizer.padding_side = pad_side


class WorldWritableOutputCallback(TrainerCallback):
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint_dir = os.path.join(args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step))
        make_world_writable(checkpoint_dir)

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if state.is_world_process_zero:
            make_world_writable(args.output_dir)


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def on_train_begin(self, args: "TrainingArguments", state, control, **kwargs):
        # EarlyStoppingCallback also assert load_best_model_at_end=True which is not supported for MOS
        assert args.metric_for_best_model is not None, (
            "EarlyStoppingCallback requires metric_for_best_model is defined"
        )
        assert args.eval_strategy != IntervalStrategy.NO, (
            "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"
        )


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor
        super().__init__()

    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a checkpoint save.
        """
        # online predict will panic when they get a tokenizer with padding_side = "right"
        if args.should_save and self.processor is not None:
            output_dir = os.path.join(args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step))
            self.processor.save_pretrained(output_dir)
            logger.info(f"Saved processor to {output_dir}")


class DynamicInterleaveCallBack(TrainerCallback):
    "A callback that support Dynamic probabilities of Interleave dataset"

    # input trainer as private member
    def __init__(self):
        self.mdl_eval_metric_recoder = {}
        self.epoch_eval_loss_recoder = []

    def on_evaluate(self, args, state, control, **kwargs):
        self.mdl_eval_metric_recoder.update(kwargs["metrics"])

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.mdl_eval_metric_recoder:
            from decimal import ROUND_HALF_UP, Decimal

            import torch
            import torch.nn.functional as F

            loss_dic = {}
            for k, v in self.mdl_eval_metric_recoder.items():
                if "loss" in k:
                    loss_dic[k] = v

            loss_list = []
            for i in range(len(loss_dic)):
                loss_list.append(loss_dic[f"eval_dataset_{i}_loss"])

            cur_logits = torch.tensor(loss_list)
            self.epoch_eval_loss_recoder.append(cur_logits)

            if len(self.epoch_eval_loss_recoder) == 1:
                logits = cur_logits
            else:
                logits = cur_logits / self.epoch_eval_loss_recoder[-2]

            probabilities = F.softmax(logits).tolist()
            probabilities = [
                float(Decimal.from_float(num).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
                for num in probabilities
            ]

            self.mdl_eval_metric_recoder = {}
            print(f"[DynamicInterleaveCallBack] after eval, new dataset probabilities is {probabilities}")

            # TODO:
            # Currently, we modifying RandomlyCyclingMultiSourcesExamplesIterable probabilities parameters directly
            # and need a more elegant way in the future
            dataset = kwargs["train_dataloader"].dataset
            if dataset.__class__.__name__ == "IterableDatasetShard":
                dataset = dataset.dataset

            data_iter = dataset._ex_iterable
            while hasattr(data_iter, "ex_iterable"):
                data_iter = data_iter.ex_iterable

            if data_iter.__class__.__name__ == "RandomlyCyclingMultiSourcesExamplesIterable":
                data_iter.probabilities = probabilities
            else:
                raise Exception(f"[DynamicInterleaveCallBack] Unknown data iter:{data_iter}")


class PredictInTrainingCallback(TrainerCallback):
    def __init__(self, trainer, predict_dataset, compute_metrics, data_collator, gen_kwargs):
        logger.info("[PredictInTrainingCallback] setup ...")
        self.trainer = trainer
        self.predict_dataset = predict_dataset
        self.compute_metrics = compute_metrics
        self.gen_kwargs = gen_kwargs
        self.data_collator = data_collator

    def get_generate_dataloader(self, test_dataset):
        data_collator = self.data_collator
        test_dataset = self.trainer._remove_unused_columns(test_dataset, description="test")

        dataloader_params = {
            "batch_size": self.trainer.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.trainer.args.dataloader_num_workers,
            "pin_memory": self.trainer.args.dataloader_pin_memory,
            "persistent_workers": self.trainer.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self.trainer._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.trainer.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.trainer.args.dataloader_prefetch_factor

        return self.trainer.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix: str = "test") -> PredictionOutput:
        logger.info("##################custom predict func#######################")
        # memory metrics - must set up as early as possible
        self.trainer._memory_tracker.start()

        test_dataloader = self.get_generate_dataloader(test_dataset)
        start_time = time.time()

        if is_transformers_version_greater_than("5.0.0"):
            eval_loop = self.trainer.evaluation_loop
        else:
            eval_loop = (
                self.trainer.prediction_loop
                if self.trainer.args.use_legacy_prediction_loop
                else self.trainer.evaluation_loop
            )

        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.trainer.args.eval_batch_size * self.trainer.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.trainer.callback_handler.on_predict(
            self.trainer.args, self.trainer.state, self.trainer.control, output.metrics
        )
        self.trainer._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        logger.info("[PredictInTrainingCallback] evaluating ...")
        current_dataset = None
        current_dataset_name = None
        if isinstance(self.predict_dataset, dict):
            # kwargs['metrics']中包含eval_dataset_0_loss这个指标，其中dataset_0是多eval dataset的key
            for _name in self.predict_dataset.keys():
                loss_key_name = "eval_" + _name + "_loss"
                if loss_key_name in kwargs["metrics"]:
                    current_dataset = self.predict_dataset[_name]
                    current_dataset_name = _name
                    break
        else:
            current_dataset = self.predict_dataset

        assert current_dataset is not None, f"current_dataset is None, self.predict_dataset is {self.predict_dataset}"

        self.trainer.open_left_padding_mode()
        compute_accuracy = self.trainer.compute_metrics
        self.trainer.compute_metrics = self.compute_metrics
        predict_results = self.predict(
            current_dataset,
            metric_key_prefix=f"predict_{current_dataset_name}" if current_dataset_name is not None else "predict",
        )
        self.trainer.process_predictions(current_dataset, predict_results)
        self.trainer.compute_metrics = compute_accuracy
        self.trainer.close_left_padding_mode()


class SaveReftCallback(TrainerCallback):
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a checkpoint save.
        """
        if args.should_save:
            output_dir = os.path.join(args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step))
            model = kwargs.pop("model", None)
            hf_model_config = model.config
            model.config = model.reft_config
            model.save(save_directory=output_dir, include_model=False)
            model.config = hf_model_config
            logger.info(f"Saved reft intervention to {output_dir}")


class PushToMosCallback(TrainerCallback):
    def __init__(
        self, trainer: "Trainer", training_args: "TrainingArguments", config: "PretrainedConfig", seq_len: int
    ):
        self.trainer = trainer
        self.start_time = time.time()
        self.elapsed_time = 0
        self.total_flops = 0
        self.tflops = 0

        self.batch_size = training_args.per_device_train_batch_size
        self.seq_len = seq_len
        self.hidden_size = getattr(config, "hidden_size", 0)
        self.num_layers = getattr(config, "num_hidden_layers", 0)
        self.vocab_size = getattr(config, "vocab_size", 0)
        self.num_attention_heads = getattr(config, "num_attention_heads", 1)
        self.num_key_value_heads = getattr(config, "num_key_value_heads", 0)

    # TODO: 1. 使用每个 batch 的 seq_len 而非 cutoff_len; 2. 增加 vit 部分计算
    def cal_flops(self):
        r"""
        Adapted training callback logic.
        """
        fwd_flops_attn = self.num_layers * 4 * self.batch_size * self.hidden_size * self.seq_len ** 2
        fwd_flops_logits = 2 * self.batch_size * self.seq_len * self.hidden_size * self.vocab_size

        fwd_flops_ffn = self.num_layers * self.batch_size * 16 * self.seq_len * self.hidden_size ** 2

        qo = 4 * self.seq_len * self.hidden_size ** 2
        kv = 4 * self.seq_len * self.num_key_value_heads * self.hidden_size ** 2 / self.num_attention_heads
        fwd_flops_attn_proj = self.num_layers * self.batch_size * (qo + kv)

        fwd_flops_linear = fwd_flops_ffn + fwd_flops_attn_proj

        flops_per_iter = 3 * (fwd_flops_linear + fwd_flops_attn + fwd_flops_logits)
        return flops_per_iter

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_world_process_zero:
            self.start_time = time.time()

    def on_step_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of a training step.
        """
        if state.is_world_process_zero:
            self.start_time = time.time()

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_world_process_zero:
            current_time =  time.time()
            self.elapsed_time += current_time - self.start_time
            self.total_flops += self.cal_flops()
            self.tflops = self.total_flops / (self.elapsed_time * (10**12))

    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        ckpt_id = "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step)
        output_dir = os.path.join(args.output_dir, ckpt_id)
        metrics = {
            "tflops": self.tflops,
        }
        self.trainer.openlm_hub_utils.save_local_model_to_mos(ckpt_id, output_dir, metrics=metrics)


class InteractiveCkptCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        start_time = time.time()
        rank = int(os.getenv('RANK', '0'))
        should_save = interactive_ckpt_should_save(rank, state.global_step)
        if should_save:
            control.should_save = True
        if time.time() - start_time > 1:
            logger.warning(f"InteractiveCkptCallback on_step_end, cost time is {time.time() - start_time:.2f} seconds")

    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a checkpoint save.
        """
        rank = int(os.getenv('RANK', '0'))
        ckpt_id = "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step)
        interactive_ckpt_finish_save(rank, state.global_step, ckpt_id)
