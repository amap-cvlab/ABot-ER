import base64
import json
import numpy as np
import torch

import decord
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from qwen_vl_utils import process_vision_info


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


def _load_qwen3_5_model(model_name_or_path: str, device_map: str, attn_implementation: Optional[str]):
    """Load dense or MoE Qwen3.5 checkpoint."""
    adapter_path = _maybe_lora_adapter_path(model_name_or_path)
    base_model_name_or_path = model_name_or_path
    if adapter_path is not None:
        adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
        base_model_name_or_path = adapter_config["base_model_name_or_path"]

    from abot_er.checkpoint_contract import require_task_conditioned_checkpoint, validate_local_index
    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=False)
    require_task_conditioned_checkpoint(config)
    validate_local_index(config, base_model_name_or_path)
    from embodied_eval.models.dhta import prepare_dhta_model_class

    prepare_dhta_model_class(config)
    architectures = getattr(config, "architectures", None) or []
    model_type = getattr(config, "model_type", "") or ""
    text_config = getattr(config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", "") if text_config is not None else ""
    is_moe = any(
        "moe" in value.lower()
        for value in [model_type, text_model_type, *architectures]
        if isinstance(value, str)
    )
    if is_moe:
        from transformers import Qwen3_5MoeForConditionalGeneration

        model_cls = Qwen3_5MoeForConditionalGeneration
    else:
        from transformers import Qwen3_5ForConditionalGeneration

        model_cls = Qwen3_5ForConditionalGeneration
    eval_logger.info(
        f"Using {model_cls.__name__} for {base_model_name_or_path} "
        f"(model_type={model_type}, text_model_type={text_model_type}, architectures={architectures})"
    )

    model_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": False,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    from embodied_eval.models.dhta import validate_dhta_loading_info, validate_dhta_model

    model, loading_info = model_cls.from_pretrained(
        base_model_name_or_path,
        output_loading_info=True,
        **model_kwargs,
    )
    validate_dhta_loading_info(loading_info)
    validate_dhta_model(model)
    if adapter_path is not None:
        from peft import PeftModel

        eval_logger.info(
            f"Loading LoRA adapter from {adapter_path} with base {base_model_name_or_path}"
        )
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    return model.eval()


@register_model("qwen3_5")
class Qwen3_5(BaseAPIModel):
    """
    Qwen3.5 model for embodied-arena evaluation.
    https://huggingface.co/Qwen/Qwen3.5-4B
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        max_length: Optional[int] = 2048,
        batch_size: Optional[Union[int, str]] = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        num_beams: int = 1,
        do_sample: bool = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        min_pixels: int = 3126,
        max_pixels: int = 200704,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 768,
        max_frames: Optional[int] = 32,
        fps: Optional[float] = 2,
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if max_frames is not None:
            max_num_frames = max_frames

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        eval_logger.info(f"Loading Qwen3.5 model from {model_name_or_path}")
        self._model = _load_qwen3_5_model(
            model_name_or_path,
            self.device_map,
            attn_implementation,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            trust_remote_code=False,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

        self._config = self._model.config
        self._max_length = max_length if getattr(self._config, "max_length", None) else getattr(
            self._config, "max_position_embeddings", 2048
        )
        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.system_prompt = system_prompt

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.interleave_visuals = interleave_visuals
        self.enable_thinking = enable_thinking

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def max_length(self):
        return self._max_length

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _build_video_kwargs(self):
        """Build video processing kwargs for Qwen3.5."""
        video_kwargs = {"min_pixels": self.min_pixels}

        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        elif self.total_pixels is not None:
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames

        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        else:
            video_kwargs["max_pixels"] = self.max_pixels

        return video_kwargs

    def _apply_chat_template(self, message, **kwargs):
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        template_kwargs.update(kwargs)
        return self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    def _strip_thinking(self, answer: str) -> str:
        if self.enable_thinking:
            _, _, remaining = answer.partition("</think>")
            return remaining.strip()
        return answer

    def process_visuals(self, visual):
        """Process visuals for the model."""
        video_kwargs = self._build_video_kwargs()

        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
            try:
                vr = decord.VideoReader(visual)
                first_frame = vr[0].asnumpy()
                height, width = first_frame.shape[:2]
                total_frames = len(vr)
                if self.fps is not None:
                    video_fps = int(self.fps)
                else:
                    video_fps = int(vr.get_avg_fps())

                processed_visual = {
                    "type": "video",
                    "video": visual,
                    **video_kwargs,
                    "video_metadata": {
                        "fps": video_fps,
                        "total_frames": total_frames,
                        "height": height,
                        "width": width,
                    },
                }
            except Exception as e:
                eval_logger.warning(f"Failed to read video metadata {visual}: {e}")
                fallback_fps = int(self.fps) if self.fps is not None else 2
                processed_visual = {
                    "type": "video",
                    "video": visual,
                    **video_kwargs,
                    "fps": fallback_fps,
                }
        elif isinstance(visual, str) and visual.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            try:
                image = Image.open(visual).convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                processed_visual = {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{base64_string}",
                    "max_pixels": self.max_pixels,
                    "min_pixels": self.min_pixels,
                }
            except Exception as e:
                eval_logger.warning(f"Failed to load image {visual}: {e}")
                processed_visual = None
        elif isinstance(visual, Image.Image) or isinstance(visual, np.ndarray):
            if isinstance(visual, np.ndarray):
                visual = Image.fromarray(visual)
            processed_visual = {
                "type": "image",
                "image": visual,
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
        else:
            eval_logger.warning(f"Unsupported visual type: {type(visual)}")
            processed_visual = None

        return processed_visual

    def build_messages(self, context: str, visuals: List[dict]) -> List[dict]:
        """Build messages for the model."""
        message = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

        if self.reasoning_prompt:
            context = context.strip() + self.reasoning_prompt

        if self.interleave_visuals is False:
            message.append(
                {
                    "role": "user",
                    "content": visuals + [{"type": "text", "text": context}],
                }
            )
        else:
            image_placeholders = re.findall(r"<image \d+>", context)
            content_parts = []
            text_parts = re.split(r"<image \d+>", context)
            if text_parts[0]:
                content_parts.append({"type": "text", "text": text_parts[0]})

            for i, placeholder in enumerate(image_placeholders):
                img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                image_idx = min(img_idx, len(visuals) - 1) if visuals else 0
                if visuals and image_idx < len(visuals):
                    content_parts.append(visuals[image_idx])
                if i + 1 < len(text_parts) and text_parts[i + 1]:
                    content_parts.append({"type": "text", "text": text_parts[i + 1]})

            message.append(
                {
                    "role": "user",
                    "content": content_parts,
                }
            )
        return message

    def respond(self, context, visuals, **gen_kwargs):
        """
        Generate a text response based on the given context and visual inputs.
        """
        if "<image>" in context:
            context = context.replace("<image>", "")

        processed_visuals = [self.process_visuals(visual) for visual in visuals]
        processed_visuals = [v for v in processed_visuals if v is not None]

        message = self.build_messages(context, processed_visuals)
        text = self._apply_chat_template(message)

        image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
            [message],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = map(list, zip(*video_inputs))

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata_list,
            **processed_video_kwargs,
            do_resize=False,
            return_tensors="pt",
        ).to(self.device)

        for key, default in (
            ("max_new_tokens", self.max_new_tokens),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("num_beams", self.num_beams),
            ("do_sample", self.do_sample),
            ("use_cache", self.use_cache),
        ):
            gen_kwargs.setdefault(key, default)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        cont = self.model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)
        ]
        text_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._strip_thinking(text_output.strip())
