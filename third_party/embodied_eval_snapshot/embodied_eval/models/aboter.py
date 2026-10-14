from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import decord
import numpy as np
import torch
from accelerate import DistributedType
from loguru import logger as eval_logger
from PIL import Image
from transformers import AutoProcessor

from embodied_eval.common.registry import register_model
from embodied_eval.models.qwen3_5 import Qwen3_5


def _maybe_lora_adapter_path(model_name_or_path: str) -> Path | None:
    model_path = Path(model_name_or_path)
    adapter_config_path = model_path / "adapter_config.json"
    adapter_weight_paths = (
        model_path / "adapter_model.safetensors",
        model_path / "adapter_model.bin",
    )
    if adapter_config_path.exists() and any(path.exists() for path in adapter_weight_paths):
        return model_path
    return None


@register_model("aboter")
class ABotER(Qwen3_5):
    """
    ABotER adapter based on Qwen3.5.

    This keeps the official evaluator and task scoring untouched while aligning
    ABotER-specific input construction with the offline vLLM inference path:
    no system message, videos are uniformly sampled into image frames, and
    thinking is disabled by default.
    """

    def __init__(
        self,
        model_name_or_path: str = "./path/to/aboter",
        system_prompt: Optional[str] = None,
        max_frames: Optional[int] = 32,
        fps: Optional[float] = None,
        min_pixels: int = 3126,
        max_pixels: int = 524288,
        max_new_tokens: int = 1024,
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        num_beams: int = 1,
        do_sample: bool = False,
        use_cache: bool = True,
        enable_thinking: Optional[bool] = False,
        **kwargs,
    ) -> None:
        adapter_path = _maybe_lora_adapter_path(model_name_or_path)
        base_model_name_or_path = model_name_or_path
        if adapter_path is not None:
            adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
            base_model_name_or_path = adapter_config["base_model_name_or_path"]

        super().__init__(
            model_name_or_path=base_model_name_or_path,
            system_prompt=system_prompt,
            max_frames=max_frames,
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            use_cache=use_cache,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        if adapter_path is not None:
            self._load_lora_adapter(adapter_path, base_model_name_or_path)

        self._active_task: str | None = None
        disable_lora_tasks = (
            os.environ.get("ABOTER_DISABLE_LORA_TASKS", "")
            .replace(";", ",")
            .replace(":", ",")
        )
        self._disable_lora_tasks = {
            item.strip()
            for item in disable_lora_tasks.split(",")
            if item.strip()
        }
        self._disable_lora_warning_emitted = False
        if adapter_path is not None and self._disable_lora_tasks:
            eval_logger.info(
                "ABotER will disable LoRA adapter for tasks: "
                + ",".join(sorted(self._disable_lora_tasks))
            )

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            self._rank = accelerator.process_index
            self._world_size = accelerator.num_processes

        # 与 offline vLLM 路径保持一致：processor 只使用模型目录默认配置，
        # 不额外注入 max_pixels/min_pixels 或 trust_remote_code=True。
        processor_path = str(adapter_path) if adapter_path is not None else model_name_or_path
        try:
            self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=False)
        except Exception:
            if adapter_path is None:
                raise
            eval_logger.warning(
                f"ABotER failed to load processor from LoRA adapter {adapter_path}; "
                f"falling back to base model {base_model_name_or_path}"
            )
            self.processor = AutoProcessor.from_pretrained(base_model_name_or_path, trust_remote_code=False)
        self._tokenizer = self.processor.tokenizer
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        self._dhta_enabled = bool(getattr(self.config, "dhta_enabled", False))
        if self._dhta_enabled:
            from embodied_eval.models.dhta import validate_dhta_tokenizer

            validate_dhta_tokenizer(self.config, self._tokenizer)

            from embodied_eval.models.qwen_dhta_eval_profile import historical_dhta_profile

            profile = historical_dhta_profile(processor_path)
            if profile is not None:
                self.processor.chat_template = Path(profile["chat_template"]).read_text()
                eval_logger.info(
                    "ABotER historical DHTA profile: "
                    + json.dumps({**profile, "enable_thinking": self.enable_thinking},
                                 ensure_ascii=False)
                )
            eval_logger.info("ABotER DHTA mode enabled with 2D-QA route injection")

    def _load_lora_adapter(self, adapter_path: Path, base_model_name_or_path: str) -> None:
        from peft import PeftModel

        eval_logger.info(
            f"Loading ABotER LoRA adapter from {adapter_path} with base {base_model_name_or_path}"
        )
        base_model = self.model
        lora_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None:
            self._model = lora_model.eval()
        elif accelerator.distributed_type == DistributedType.FSDP:
            self._model = accelerator.prepare(lora_model.eval())
        else:
            self._model = accelerator.prepare_model(lora_model.eval(), evaluation_mode=True)

    def set_active_task(self, task_name: str | None) -> None:
        self._active_task = task_name

    def _should_disable_lora_for_active_task(self) -> bool:
        if not self._disable_lora_tasks or not self._active_task:
            return False
        task = self._active_task
        aliases = {task, task.removesuffix("_aboter")}
        return bool(aliases.intersection(self._disable_lora_tasks))

    def _lora_adapter_context(self):
        if not self._should_disable_lora_for_active_task():
            return nullcontext(self.model)

        model = self.model
        target = getattr(model, "module", model)
        get_base_model = getattr(target, "get_base_model", None)
        base_model = getattr(target, "base_model", None)
        disable_adapter_layers = getattr(base_model, "disable_adapter_layers", None)
        enable_adapter_layers = getattr(base_model, "enable_adapter_layers", None)
        if callable(get_base_model) and callable(disable_adapter_layers) and callable(enable_adapter_layers):
            return self._base_generation_context(
                target,
                get_base_model,
                disable_adapter_layers,
                enable_adapter_layers,
            )

        disable_adapter = getattr(target, "disable_adapter", None)
        if callable(disable_adapter):
            return self._disable_adapter_generation_context(target, disable_adapter)

        if not self._disable_lora_warning_emitted:
            eval_logger.warning(
                "ABOTER_DISABLE_LORA_TASKS was set, but current model does not expose "
                "disable_adapter(); LoRA remains enabled."
            )
            self._disable_lora_warning_emitted = True
        return nullcontext(self.model)

    @contextmanager
    def _base_generation_context(
        self,
        peft_model,
        get_base_model,
        disable_adapter_layers,
        enable_adapter_layers,
    ):
        model_status = None
        get_model_status = getattr(peft_model, "get_model_status", None)
        if callable(get_model_status):
            try:
                model_status = get_model_status()
            except Exception:
                model_status = None
        eval_logger.info(f"Disabling ABotER LoRA layers and using base generate for task={self._active_task}")
        disable_adapter_layers()
        try:
            yield get_base_model()
        finally:
            enabled = getattr(model_status, "enabled", True)
            if enabled is not False:
                enable_adapter_layers()

    @contextmanager
    def _disable_adapter_generation_context(self, generation_model, disable_adapter):
        eval_logger.info(f"Disabling ABotER LoRA adapter for task={self._active_task}")
        with disable_adapter():
            yield generation_model

    def process_visuals(self, visual):
        if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov")):
            return self._process_video_as_images(visual)

        if isinstance(visual, str) and visual.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            try:
                visual = Image.open(visual).convert("RGB")
            except Exception as e:
                eval_logger.warning(f"ABotER failed to load image {visual}: {e}")
                return []

        if isinstance(visual, np.ndarray):
            visual = Image.fromarray(visual)

        if isinstance(visual, Image.Image):
            return [visual.convert("RGB")]

        eval_logger.warning(f"ABotER unsupported visual type: {type(visual)}")
        return []

    def _process_video_as_images(self, video_path: str) -> list[Image.Image]:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            if total_frames <= 0:
                return []

            indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
            frames = vr.get_batch(indices.tolist()).asnumpy()
        except Exception as e:
            eval_logger.warning(f"ABotER failed to sample video {video_path}: {e}")
            return []

        return [Image.fromarray(frame).convert("RGB") for frame in frames]

    def _prepare_text_and_images(self, context, visuals):
        if "<image>" in context:
            context = context.replace("<image>", "")

        # ERQA 会返回 (images, visual_indices)。offline vLLM 推理只取 images；
        # 这里保持同样处理，避免把 visual_indices 当成视觉输入。
        if (
            isinstance(visuals, tuple)
            and len(visuals) == 2
            and isinstance(visuals[0], (list, tuple))
        ):
            visuals = visuals[0]

        image_inputs = []
        for visual in visuals:
            image_inputs.extend(self.process_visuals(visual))

        content = [{"type": "image"} for _ in image_inputs]
        content.append({"type": "text", "text": context})
        message = [{"role": "user", "content": content}]
        if self.system_prompt:
            message.insert(0, {"role": "system", "content": self.system_prompt})
        if self._dhta_enabled:
            from embodied_eval.models.dhta import inject_dhta_route

            message = inject_dhta_route(message, "2d_qa")
        text = self._apply_chat_template(message)
        return text, image_inputs

    def _generation_kwargs(self, gen_kwargs):
        generation_kwargs = dict(gen_kwargs)
        for key, default in (
            ("max_new_tokens", self.max_new_tokens),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("num_beams", self.num_beams),
            ("do_sample", self.do_sample),
            ("use_cache", self.use_cache),
        ):
            generation_kwargs.setdefault(key, default)
        if self.top_k is not None:
            generation_kwargs.setdefault("top_k", self.top_k)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        return generation_kwargs

    def respond(self, context, visuals, **gen_kwargs):
        return self.respond_batch([context], [visuals], [gen_kwargs])[0]

    def respond_batch(self, contexts, visuals_batch, gen_kwargs_batch):
        if not contexts:
            return []

        texts = []
        image_inputs = []
        for context, visuals in zip(contexts, visuals_batch):
            text, sample_images = self._prepare_text_and_images(context, visuals)
            texts.append(text)
            image_inputs.extend(sample_images)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        gen_kwargs = self._generation_kwargs(gen_kwargs_batch[0] if gen_kwargs_batch else {})

        with torch.no_grad():
            with self._lora_adapter_context() as generation_model:
                cont = generation_model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)
        ]
        text_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [self._strip_thinking(output.strip()) for output in text_output]
