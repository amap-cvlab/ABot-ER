import math
from dataclasses import dataclass, fields
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import traceback
import os

import av
import numpy as np
import requests
import torch
from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.image_utils import OPENAI_CLIP_MEAN, ChannelDimension, get_image_size, load_image, to_numpy_array

from ..extras.constants import IGNORE_INDEX, MediaType
from ..extras.logging import get_logger
from ..extras.packages import is_librosa_available, is_vllm_version_greater_than
from ..hparams.model_args import ProcessorArguments
from .oss_utils import create_oss2_downloader


if is_librosa_available():
    import librosa


if TYPE_CHECKING:
    from av.container import InputContainer
    from av.stream import Stream
    from PIL.Image import Image as ImageObject

    from ..data.template import Template

    ImageInput = Union[str, bytes, ImageObject]
    VideoInput = Union[str, List[str]]
    AudioInput = str

logger = get_logger(__name__)


class BasePlugin:
    def __init__(
        self, tokenizer: "PreTrainedTokenizer", processor: "ProcessorMixin", processor_args: "ProcessorArguments"
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer

        processor_args_fileds = [f.name for f in fields(ProcessorArguments)]
        processor_args_dict = {name: getattr(processor_args, name) for name in processor_args_fileds}
        self.processor_args = ProcessorArguments(**processor_args_dict)

        self.image_token_id = None
        self.video_token_id = None
        self.audio_token_id = None

        self.oss2_downloader = create_oss2_downloader()

    def validate_input(self, input_ids: List[int], examples: Dict[str, List[Any]], index: int) -> bool:
        r"""
        Validate if the number of images, videos and audios match the number of placeholders in messages.
        """
        num_image_tokens = input_ids.count(self.image_token_id) if self.image_token_id is not None else 0
        num_video_tokens = input_ids.count(self.video_token_id) if self.video_token_id is not None else 0
        num_audio_tokens = input_ids.count(self.audio_token_id) if self.audio_token_id is not None else 0

        images = examples["image"][index]
        # TODO: merge video and video_frames
        video_key = "video_frames" if len(examples["video_frames"][index]) > 0 else "video"
        videos = examples[video_key][index]
        audios = examples["audio"][index]

        if len(images) > num_image_tokens:
            examples["image"][index] = images[:num_image_tokens]
        elif len(images) < num_image_tokens:
            logger.warning(f"Number of images {len(images)} does not match number of image tokens {num_image_tokens}")
            return False

        if len(videos) > num_video_tokens and video_key == "video":
            examples[video_key][index] = videos[:num_video_tokens]
        elif len(videos) < num_video_tokens:
            logger.warning(f"Number of videos {len(videos)} does not match number of video tokens {num_video_tokens}")
            return False

        if len(audios) > num_audio_tokens:
            examples["audio"][index] = audios[:num_audio_tokens]
        elif len(audios) < num_audio_tokens:
            logger.warning(f"Number of audios {len(audios)} does not match number of audio tokens {num_audio_tokens}")
            return False

        return True

    def validate_input_for_vllm(
        self, input_ids: List[int], template: "Template", examples: Dict[str, List[Any]]
    ) -> bool:
        r"""
        Validate if the number of images, videos and audios match the number of placeholders in messages.
        """
        message = self.tokenizer.decode(input_ids)
        num_image_tokens = message.count(template.image_token)
        num_video_tokens = message.count(template.video_token)
        num_audio_tokens = message.count(template.audio_token)

        images = examples["image"] if "image" in examples else []
        # TODO: merge video and video_frames
        video_key = "video_frames" if "video_frames" in examples and len(examples["video_frames"]) > 0 else "video"
        videos = examples[video_key] if video_key in examples else []
        audios = examples["audio"] if "audio" in examples else []

        assert len(images) == num_image_tokens, (
            f"Number of images {len(images)} does not match number of image tokens {num_image_tokens}"
        )
        assert len(videos) == num_video_tokens, (
            f"Number of videos {len(videos)} does not match number of video tokens {num_video_tokens}"
        )
        assert len(audios) == num_audio_tokens, (
            f"Number of audios {len(audios)} does not match number of audio tokens {num_audio_tokens}"
        )

    def _preprocess_image(self, image: "ImageObject", from_video=False, **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        max_pixels = self.processor_args.video_resolution if from_video else self.processor_args.image_resolution
        min_pixels = self.processor_args.min_resolution
        if max_pixels > 0 and (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)
            logger.warning_once(f"Image size is too large, resized to {max_pixels}")

        if min_pixels > 0 and (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _regularize_image(self, image: "ImageInput", **kwargs) -> "ImageObject":
        r"""
        Regularizes image to avoid error. Including reading and pre-processing.
        """
        if isinstance(image, bytes):  # ai-lake data type
            image = Image.open(BytesIO(image))
        elif isinstance(image, dict):
            if image["bytes"] is not None:
                image = Image.open(BytesIO(image["bytes"]))
            else:
                image = Image.open(image["path"])
        else:
            image = load_image(image, timeout=30)
        return self._preprocess_image(image, **kwargs)

    def _regularize_video(self, video: "VideoInput", **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes video to avoid error. Including reading, resizing and converting.
        """
        if not isinstance(video, list):
            frames = self.load_video_as_frames(video)
        else:
            frames = video
        return [self._regularize_image(frame, from_video=True) for frame in frames]

    def _regularize_audio(self, audio: "AudioInput", **kwargs) -> "np.ndarray":
        r"""
        Regularizes audio to avoid error. Including reading and converting.
        """
        if isinstance(audio, np.ndarray):  # mock
            return audio
        if audio.startswith(("http://", "https://")):
            audio = BytesIO(requests.get(audio, timeout=None).content)
        audio, _ = librosa.load(audio, sr=self.sampling_rate)
        return audio

    def _get_video_sample_indices(self, video_stream: "Stream", video_container: "InputContainer"):
        total_frames = video_stream.frames
        # +1 for frame 0, a 9s video should get 10 frames when fps=1
        if video_stream.duration is None or total_frames is None or total_frames == 0:  # this fork cost more
            total_frames = sum([1 for _ in video_container.decode(video_stream)])
            sample_frames = (total_frames / float(video_stream.average_rate)) * self.processor_args.video_fps + 1
        else:
            sample_frames = float(video_stream.duration * video_stream.time_base) * self.processor_args.video_fps + 1
        sample_frames = min(total_frames, self.processor_args.video_maxlen, sample_frames)
        sample_frames = math.floor(sample_frames)
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        return sample_indices

    def load_video_as_frames(self, video_path: str):
        with av.open(video_path, "r", metadata_errors="ignore") as container:
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            sample_indices = self._get_video_sample_indices(video_stream, container)
            frames = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())
        return frames

    def _flatten(self, array: List[Union[int, List[int]]]):
        flatten_array = []
        for item in array:
            if isinstance(item, list):
                flatten_array.extend(item)
            else:
                flatten_array.append(item)
        return flatten_array

    def _images_to_features(self, images: List["ImageObject"]):
        return self.processor.image_processor(images, return_tensors="pt")

    def _videos_to_features(self, videos: List[List["ImageObject"]]):
        if hasattr(self.processor, "video_processor"):
            return self.processor.video_processor(videos, return_tensors="pt")
        else:
            return self.processor(videos, media_type=MediaType.VIDEO, return_tensors="pt")

    def _audios_to_features(self, audios: List[Any]):
        return self.processor(audios, media_type=MediaType.AUDIO, return_tensors="pt")

    
    def download_by_oss2(
        self,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        if images is not None and len(images) > 0:
            images = [self.oss2_downloader.download(im) for im in images]
        if videos is not None and len(videos) > 0:
            videos = [self.oss2_downloader.download(vid) for vid in videos]
        if audios is not None and len(audios) > 0:
            audios = [self.oss2_downloader.download(audio) for audio in audios]
        return images, videos, audios

    def get_mm_features(
        self,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        if self.oss2_downloader:
            images, videos, audios = self.download_by_oss2(images=images, videos=videos, audios=audios)
        
        mm_features = {}
        if images is not None and len(images) > 0:
            images = [self._regularize_image(im) for im in images]
            mm_features.update(self._images_to_features(images))
        if videos is not None and len(videos) > 0:
            videos = [self._regularize_video(vid) for vid in videos]
            mm_features.update(self._videos_to_features(videos))
        if audios is not None and len(audios) > 0:
            audios = [self._regularize_audio(audio) for audio in audios]
            mm_features.update(self._audios_to_features(audios))
        return mm_features

    def get_mm_source_features(
        self,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        if self.oss2_downloader:
            images, videos, audios = self.download_by_oss2(images=images, videos=videos, audios=audios)
        mm_features = {}
        if images is not None and len(images) > 0:
            mm_features[MediaType.IMAGE] = [self._regularize_image(im) for im in images]
        if videos is not None and len(videos) > 0:
            mm_features[MediaType.VIDEO] = [self._regularize_video(vid) for vid in videos]
        if audios is not None and len(audios) > 0:
            mm_features[MediaType.AUDIO] = [self._regularize_audio(vid) for vid in videos]
        return mm_features

    def _process_single_mm_input(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        mm_features = self.get_mm_features(images=images, videos=videos, audios=audios)
        processed_token_ids = self.process_single_token_ids(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, mm_features=mm_features
        )
        return processed_token_ids, mm_features

    def _get_mock_single_mm_input(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        input_ids = [self.audio_token_id] if audios is not None and len(audios) > 0 else [self.image_token_id]
        if labels is not None:
            labels = [IGNORE_INDEX]
        if attention_mask is not None:
            attention_mask = [1]
        images = [Image.new("RGB", (224, 224), (255, 255, 255))]
        if audios is not None and len(audios) > 0:
            audios = [np.zeros(1600)]
            return self._process_single_mm_input(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask, audios=audios
            )
        return self._process_single_mm_input(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, images=images
        )

    def append_mock_media(self, feature: Dict[str, Any]):
        feature["image"] = [Image.new("RGB", (224, 224), (255, 255, 255))]
        feature["input_ids"] = feature["input_ids"] + [self.image_token_id]
        if "labels" in feature:
            feature["labels"] = feature["labels"] + [IGNORE_INDEX]
        if "attention_mask" in feature:
            feature["attention_mask"] = feature["attention_mask"] + [1]
        return feature

    def process_single_mm_input(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
        failover: bool = True,
    ):
        try:
            processed_token_ids, mm_features = self._process_single_mm_input(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                images=images,
                videos=videos,
                audios=audios,
            )
        except Exception as e:  # TODO: specify load and process error
            mm_input = (images or []) + (videos or []) + (audios or [])
            mm_input = [i if isinstance(i, str) or isinstance(i, list) else "BYTES_DATA" for i in mm_input]
            logger.error(f"Failed to process mm input: {e}, mm_input: {mm_input}, input_ids: {input_ids}\n traceback is\n{traceback.format_exc()}")
            if not failover:
                return None, None
            return self._get_mock_single_mm_input(input_ids, labels, attention_mask, images, videos, audios)
        return processed_token_ids, mm_features

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        processed_token_ids = {"input_ids": input_ids}
        if labels:
            processed_token_ids["labels"] = labels
        if attention_mask:
            processed_token_ids["attention_mask"] = attention_mask
        return processed_token_ids

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        collated_features = {}
        for mm_feature in mm_features:
            for k, v in mm_feature.items():
                if k not in collated_features:
                    collated_features[k] = []
                collated_features[k].append(v)
        collated_features = {k: torch.concat(v) for k, v in collated_features.items()}
        return collated_features



class CommonPlugin(BasePlugin):
    pass



class Qwen2VLPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.image_placeholder_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_placeholder_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.merge_length = self.processor.image_processor.merge_size**2

    def _process_single_token_ids(
        self,
        input_ids: List[int],
        media_type: MediaType,
        grid_thw: torch.Tensor,
        labels: List[int] = None,
        attention_mask: List[int] = None,
        **kwargs,
    ):
        placeholder_token_id = (
            self.image_placeholder_token_id if media_type == MediaType.IMAGE else self.video_placeholder_token_id
        )
        image_index = 0
        media_token_id = self.image_token_id if media_type == MediaType.IMAGE else self.video_token_id
        for idx, token_id in enumerate(input_ids):
            if token_id == media_token_id:
                placeholder_num = grid_thw[image_index].prod() // self.merge_length
                placeholder_token_ids = [placeholder_token_id] * placeholder_num
                placeholder_sequence = (
                    [self.vision_start_token_id] + placeholder_token_ids + [self.vision_end_token_id]
                )
                input_ids[idx] = placeholder_sequence
                if labels:
                    labels[idx] = [IGNORE_INDEX] * len(placeholder_sequence)
                if attention_mask:
                    attention_mask[idx] = [attention_mask[idx]] * len(placeholder_sequence)
                image_index += 1

        return (
            self._flatten(input_ids),
            self._flatten(labels) if labels is not None else labels,
            self._flatten(attention_mask) if attention_mask is not None else attention_mask,
        )

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        if "image_grid_thw" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                media_type=MediaType.IMAGE,
                grid_thw=mm_features.get("image_grid_thw"),
            )
        if "video_grid_thw" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                media_type=MediaType.VIDEO,
                grid_thw=mm_features.get("video_grid_thw"),
            )
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        return super().process_single_token_ids(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, mm_features=mm_features
        )

    def _preprocess_image(self, image: "ImageObject", from_video=False, **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        # patch_size: int = 14
        # merge_size: int = 2
        # factor = patch_size * merge_size
        MIN_FACTOR = 28
        image = super()._preprocess_image(image=image, from_video=from_video, **kwargs)

        assert image.width > MIN_FACTOR and image.height > MIN_FACTOR, (
            f"Image single size [{image.width}, {image.height}]is smaller than 28, skip!"
        )
        return image

    def _videos_to_features(self, videos):
        return self.processor.image_processor(None, videos=videos, return_tensors="pt")

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        mm_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)
        image_processor = self.processor.image_processor
        if (
            "second_per_grid_ts" in getattr(image_processor, "model_input_names", [])
            and "video_grid_thw" in mm_features
        ):
            # needed for qwen2.5 vl
            mm_features["second_per_grid_ts"] = [
                image_processor.temporal_patch_size / self.processor_args.video_fps
            ] * len(mm_features["video_grid_thw"])

        # modified from llama_factory
        if model is not None and hasattr(model, "get_rope_index"):  # for qwen2vl mrope
            rope_index_kwargs = {
                "input_ids": collated_text_features["input_ids"],
                "image_grid_thw": mm_features.get("image_grid_thw"),
                "video_grid_thw": mm_features.get("video_grid_thw"),
                "attention_mask": (collated_text_features["attention_mask"] >= 1).float(),
            }
            if "second_per_grid_ts" in mm_features:  # for qwen2vl
                rope_index_kwargs["second_per_grid_ts"] = mm_features.get("second_per_grid_ts", None)
            if "video_second_per_grid" in mm_features:  # for qwen2omni
                rope_index_kwargs["second_per_grids"] = mm_features.get("video_second_per_grid", None)

            if getattr(model.config, "model_type", None) == "qwen2_5_omni_thinker":  # for qwen2omni
                rope_index_kwargs["use_audio_in_video"] = getattr(self.template.processor, "use_audio_in_video", False)
                feature_attention_mask = mm_features.get("feature_attention_mask", None)
                if feature_attention_mask is not None:
                    audio_feature_lengths = torch.sum(
                        feature_attention_mask, dim=1
                    )  # FIXME need to get video image lengths
                    rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

                delta0 = (1 - rope_index_kwargs["attention_mask"]).sum(dim=-1).unsqueeze(1)
                # avoid conflict
                new_position_ids, rope_deltas = model.get_rope_index(**rope_index_kwargs)
                collated_text_features["position_ids"], collated_text_features["rope_deltas"] = (
                    new_position_ids.clone(),
                    rope_deltas - delta0,
                )  # avoid inplace operation FIXME
            else:  # for qwen2vl
                collated_text_features["position_ids"], collated_text_features["rope_deltas"] = model.get_rope_index(
                    **rope_index_kwargs
                )

        return mm_features


class Qwen3VLPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.image_placeholder_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_placeholder_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        self.processor.video_processor.size = {"longest_edge": self.processor_args.video_longest_edge, "shortest_edge": self.processor_args.video_shortest_edge}
        self.processor.video_processor.max_frames = self.processor_args.video_maxlen

    def _process_single_token_ids(
        self,
        input_ids: List[int],
        media_type: MediaType,
        grid_thw: torch.Tensor,
        labels: List[int] = None,
        attention_mask: List[int] = None,
        video_meta=None,
        **kwargs,
    ):
        if media_type == MediaType.IMAGE:
            image_index = 0
            placeholder_token_id = self.image_placeholder_token_id
            media_token_id = self.image_token_id
            merge_length = self.processor.image_processor.merge_size**2
            for idx, token_id in enumerate(input_ids):
                if token_id == media_token_id:
                    placeholder_num = grid_thw[image_index].prod() // merge_length
                    placeholder_token_ids = [placeholder_token_id] * placeholder_num
                    placeholder_sequence = (
                        [self.vision_start_token_id] + placeholder_token_ids + [self.vision_end_token_id]
                    )
                    input_ids[idx] = placeholder_sequence
                    if labels:
                        labels[idx] = [IGNORE_INDEX] * len(placeholder_sequence)
                    if attention_mask:
                        attention_mask[idx] = [attention_mask[idx]] * len(placeholder_sequence)
                    image_index += 1
        else:
            video_index = 0
            placeholder_token_id = self.video_placeholder_token_id
            media_token_id = self.video_token_id
            # if timestamps are not provided, calculate them
            curr_timestamp = self.processor._calculate_timestamps(
                video_meta[video_index].frames_indices,
                video_meta[video_index].fps,
                self.processor.video_processor.merge_size,
            )
            merge_length = self.processor.video_processor.merge_size**2

            for idx, token_id in enumerate(input_ids):
                if token_id == media_token_id:
                    placeholder_num = grid_thw[video_index][1:].prod() // merge_length
                    placeholder_token_ids = [placeholder_token_id] * placeholder_num
                    video_placeholder = []
                    for frame_idx in range(grid_thw[video_index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        time_stamp_tokens = self.tokenizer.encode(f"<{curr_time:.1f} seconds>")
                        placeholder_sequence = (
                            time_stamp_tokens + [self.vision_start_token_id] + placeholder_token_ids + [self.vision_end_token_id]
                        )
                        video_placeholder += placeholder_sequence
                    input_ids[idx] = video_placeholder
                    if labels:
                        labels[idx] = [IGNORE_INDEX] * len(video_placeholder)
                    if attention_mask:
                        attention_mask[idx] = [attention_mask[idx]] * len(video_placeholder)
                    video_index += 1
        return (
            self._flatten(input_ids),
            self._flatten(labels) if labels is not None else labels,
            self._flatten(attention_mask) if attention_mask is not None else attention_mask,
        )

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        if "image_grid_thw" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                media_type=MediaType.IMAGE,
                grid_thw=mm_features.get("image_grid_thw"),
            )
        if "video_grid_thw" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                media_type=MediaType.VIDEO,
                grid_thw=mm_features.get("video_grid_thw"),
                video_meta=mm_features.pop("video_metadata"),
            )
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        return super().process_single_token_ids(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, mm_features=mm_features
        )

    def _regularize_video(self, video: "VideoInput", **kwargs) -> List["ImageObject"]:
        # Avoid qwen-vl-utils falling back to torchvision video decoding on ROCm.
        # Returning frame lists makes qwen-vl-utils take its image-list path.
        if isinstance(video, list):
            return video
        return self.load_video_as_frames(video)

    def _process_video_in_qwen_vl_utils(self, videos):
        from .third_party_utils.qwen_vl_utils import process_vision_info
        conversation = [
            {"content": 
                [{"type": "video", 
                "video": video, 
                "min_pixels": self.processor_args.min_resolution, 
                "max_pixels": self.processor_args.video_resolution, 
                "fps":self.processor_args.video_fps, 
                "max_frames":self.processor_args.video_maxlen}]
                } for video in videos]
        _, video_inputs, video_kwargs = process_vision_info(
            conversation,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
            video_resize_chunk_frames=self.processor_args.video_resize_chunk_frames,
        )
        return video_inputs, video_kwargs

    def _videos_to_features(self, videos):
        video_inputs, video_kwargs = self._process_video_in_qwen_vl_utils(videos)

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        # since qwen-vl-utils has resize the images/videos, \
        # we should pass do_resize=False to avoid duplicate operation in processor!
        return self.processor.video_processor(videos=video_inputs, video_metadata=video_metadatas, return_tensors="pt", **video_kwargs, return_metadata=True)


    def _images_to_features(self, images: List["ImageObject"]):
        # since qwen-vl-utils has resize the images/videos, \
        # we should pass do_resize=False to avoid duplicate operation in processor!
        return self.processor.image_processor(images, return_tensors="pt", do_resize=False)

    def _preprocess_image(self, image: "ImageObject", from_video=False, **kwargs) -> "ImageObject":
        from .third_party_utils.qwen_vl_utils import process_vision_info
        conversation = [
            {"content": 
                [{"type": "image", 
                "image": image, 
                "min_pixels": self.processor_args.min_resolution, 
                "max_pixels": self.processor_args.image_resolution}]
                }]
        image_inputs, _ = process_vision_info(
            conversation,
            image_patch_size=self.processor.image_processor.patch_size
        )
        return image_inputs[0]
    
    def process_single_mm_input_for_vllm(
        self,
        input_ids: List[int],
        mm_features: List[Dict[str, Any]],
    ):
        if MediaType.VIDEO in mm_features:
            # from qwen_vl_utils import process_vision_info
            # 原版qwen_vl_utils在video backend的异常处理上会走向torchvision，而torchvision处理视频易触发cpu OOM，本地qwen_vl_utils修改了异常处理逻辑，具体差异可对比两侧代码
            # https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
            video_inputs, video_kwargs = self._process_video_in_qwen_vl_utils(mm_features[MediaType.VIDEO])
            mm_features[MediaType.VIDEO] = video_inputs
            video_kwargs['size'] = self.processor.video_processor.size
            mm_features['mm_processor_kwargs'] = video_kwargs
        return input_ids, mm_features


class Qwen2OMNIPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|IMAGE|>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|VIDEO|>")
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.use_audio_in_video = True
        os.environ["QWEN_VL_RESIZE_CPU"] = "1"

    def _process_single_mm_input(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        from qwen_omni_utils import process_mm_info

        conversation = []
        if images:
            conversation += [
                {"content": [{"type": "image", "image": self._regularize_image(item)}]} for item in images
            ]
        else:
            images = None

        if videos:
            conversation += [{"content": [{"type": "video", "video": item}]} for item in videos]
        else:
            videos = None

        if audios:
            conversation += [{"content": [{"type": "audio", "audio": item}]} for item in audios]
        else:
            audios = None

        if conversation:
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)

        text = self.tokenizer.decode(input_ids)

        text = text.replace("<|IMAGE|>", "<|vision_bos|><|IMAGE|><|vision_eos|>")
        text = text.replace("<|VIDEO|>", "<|vision_bos|><|VIDEO|><|vision_eos|>")
        text = text.replace("<|AUDIO|>", "<|audio_bos|><|AUDIO|><|audio_eos|>")

        inputs = self.processor(
            text=[text],
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        # 默认返回一个batch的数据，所以需要取0

        processed_token_ids = {
            "input_ids": inputs.pop("input_ids").tolist()[0],
            "attention_mask": inputs.pop("attention_mask").tolist()[0],
        }
        if labels:
            extended_cnt = len(processed_token_ids["input_ids"]) - len(labels)
            labels = [IGNORE_INDEX] * extended_cnt + labels
            processed_token_ids["labels"] = labels
        return processed_token_ids, inputs

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        mm_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)
        mm_features["use_audio_in_video"] = self.use_audio_in_video
        return mm_features


class Qwen3OMNIPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.use_audio_in_video = True
        self.merge_length_image = self.processor.image_processor.merge_size**2
        self.merge_length_video = self.processor.video_processor.merge_size**2
        os.environ["QWEN_VL_RESIZE_CPU"] = "1"


    def _regularize_image(self, image: "ImageInput", **kwargs) -> "ImageObject":
        r"""
        Regularizes image to avoid error. Including reading and pre-processing.
        """
        if isinstance(image, bytes):  # ai-lake data type
            image = Image.open(BytesIO(image))
        elif isinstance(image, dict):
            if image["bytes"] is not None:
                image = Image.open(BytesIO(image["bytes"]))
            else:
                image = image["path"]
        return image

    def _process_mm_in_omni_utils(self, input_ids, images, videos, audios):
        from qwen_omni_utils import process_mm_info
        conversation = []
        if images:
            conversation += [
                {
                    "content": 
                    [
                        {
                            "type": "image", 
                            "image": self._regularize_image(item),
                            "min_pixels": self.processor_args.min_resolution,
                            "max_pixels": self.processor_args.image_resolution
                        }
                    ]
                } for item in images]
        else:
            images = None

        if videos:
            conversation += [
                {
                    "content": 
                    [
                        {
                            "type": "video", 
                            "video": item,
                            "min_pixels": self.processor_args.min_resolution, 
                            "max_pixels": self.processor_args.video_resolution, 
                            "fps": self.processor_args.video_fps, 
                            "max_frames":self.processor_args.video_maxlen
                        }
                    ]
                } for item in videos]
        else:
            videos = None

        if audios:
            conversation += [{"content": [{"type": "audio", "audio": item}]} for item in audios]
        else:
            audios = None

        if conversation:
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)

        text = self.tokenizer.decode(input_ids)

        text = text.replace("<|image_pad|>", "<|vision_start|><|image_pad|><|vision_end|>")
        text = text.replace("<|video_pad|>", "<|vision_start|><|video_pad|><|vision_end|>")
        text = text.replace("<|audio_pad|>", "<|audio_start|><|audio_pad|><|audio_end|>")

        return text, images, videos, audios

    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        # reference: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py#L80

        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def _process_single_mm_input(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):

        # audio might be included in video thus should not be processed separately
        mm_token_pos_list = []
        for index, mtoken in enumerate(input_ids):
            if mtoken in [self.image_token_id, self.video_token_id, self.audio_token_id]:
                mm_token_pos_list.append(index)

        text, images, videos, audios = self._process_mm_in_omni_utils(input_ids, images, videos, audios)

        inputs = self.processor(
            text=[text],
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        # 默认返回一个batch的数据，所以需要取0

        processed_token_ids = {
            "input_ids": inputs.pop("input_ids").tolist()[0],
            "attention_mask": inputs.pop("attention_mask").tolist()[0],
        }
        if labels:
            # similar to Qwen3OmniMoeProcessor.replace_multimodal_special_tokens, but
            # use ids instead of string and only need to consider length with position ignored
            if audios:
                audio_lengths = iter(self._get_feat_extract_output_lengths(inputs["feature_attention_mask"].sum(-1)))
            if images:
                image_grid_thw = iter(inputs["image_grid_thw"])
            if videos:
                video_grid_thw = iter(inputs["video_grid_thw"])
            for i, mindex in enumerate(mm_token_pos_list):
                if input_ids[mindex] == self.audio_token_id:
                    audio_seq_length = next(audio_lengths) + 2
                    labels[mindex] = [IGNORE_INDEX] * audio_seq_length
                elif input_ids[mindex] == self.image_token_id:
                    image_seq_length = next(image_grid_thw).prod() // self.merge_length_image + 2
                    labels[mindex] = [IGNORE_INDEX] * image_seq_length
                elif input_ids[mindex] == self.video_token_id:
                    if not self.use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // self.merge_length_video + 2
                        labels[mindex] = [IGNORE_INDEX] * video_seq_length
                    else:
                        video_seq_length = next(video_grid_thw).prod() // self.merge_length_video + 2
                        audio_seq_length = next(audio_lengths) + 2
                        labels[mindex] = [IGNORE_INDEX] * (video_seq_length + audio_seq_length)

            labels = self._flatten(labels)
            processed_token_ids["labels"] = labels

            assert len(processed_token_ids["labels"]) == len(processed_token_ids["input_ids"])

        return processed_token_ids, inputs

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        # input_features shape: (bs, freqs, frames)
        # feature_attention_mask shape: (bs, frames)
        max_length = max(
            mm_feature["input_features"].shape[-1] if "input_features" in mm_feature else 0
            for mm_feature in mm_features
        )
        for mm_feature in mm_features:
            audio_feat_keys = [k for k in mm_feature if k in ["input_features", "feature_attention_mask"]]
            for key in audio_feat_keys:
                mm_feature[key] = torch.nn.functional.pad(
                    mm_feature[key], (0, max_length - mm_feature[key].shape[-1]), "constant", 0
                )
        mm_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)
        mm_features["use_audio_in_video"] = self.use_audio_in_video
        return mm_features

    def get_mm_source_features(
        self,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        mm_features = {}
        if images is not None and len(images) > 0:
            mm_features[MediaType.IMAGE] = images
        if videos is not None and len(videos) > 0:
            mm_features[MediaType.VIDEO] = videos
        if audios is not None and len(audios) > 0:
            mm_features[MediaType.AUDIO] = audios
        return mm_features
    
    def process_single_mm_input_for_vllm(
        self,
        input_ids: List[int],
        mm_features: List[Dict[str, Any]],
    ):
        images = mm_features.get(MediaType.IMAGE, [])
        videos = mm_features.get(MediaType.VIDEO, [])
        audios = mm_features.get(MediaType.AUDIO, [])
        text, images, videos, audios = self._process_mm_in_omni_utils(input_ids, images, videos, audios)

        use_audio_in_video = True
        if not is_vllm_version_greater_than("0.16.0"):
            # v-a分离模式数据传输，需要补充audio信息
            if "<|audio_pad|>" not in text and audios:
                text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", "<|vision_start|><|video_pad|><|vision_end|><|audio_start|><|audio_pad|><|audio_end|>")
            use_audio_in_video = False

        input_ids = self.tokenizer.encode(text)
        
        if images:
            mm_features[MediaType.IMAGE] = images
        if videos:
            mm_features[MediaType.VIDEO] = videos
        if audios:
            mm_features[MediaType.AUDIO] = audios

        mm_features["mm_processor_kwargs"] = {"use_audio_in_video": use_audio_in_video}
        return input_ids, mm_features


class Qwen2AudioPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_bos_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        self.sampling_rate = 16000

    def _get_mock_single_mm_input(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        input_ids = [self.audio_token_id]
        if labels is not None:
            labels = [IGNORE_INDEX]
        if attention_mask is not None:
            attention_mask = [1]
        audios = [np.zeros(1600)]
        return self._process_single_mm_input(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, audios=audios
        )

    def _audios_to_features(self, audios: List[Any]):
        audio_inputs = self.processor.feature_extractor(
            audios,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
        return audio_inputs

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        audio_lengths = mm_features["feature_attention_mask"].sum(-1).tolist()
        audio_index = 0
        for i in range(len(input_ids)):
            if input_ids[i] == self.audio_token_id:
                audio_length = audio_lengths.pop(0)
                input_length = (audio_length - 1) // 2 + 1
                num_audio_tokens = (input_length - 2) // 2 + 1
                placeholder_sequence = [self.audio_token_id] * num_audio_tokens
                if i == 0 or input_ids[i - 1] != self.audio_bos_token_id:
                    placeholder_sequence = [self.audio_bos_token_id] + placeholder_sequence + [self.audio_eos_token_id]
                input_ids[i] = placeholder_sequence
                if labels:
                    labels[i] = [IGNORE_INDEX] * len(placeholder_sequence)
                if attention_mask:
                    attention_mask[i] = [attention_mask[i]] * len(placeholder_sequence)
                audio_index += 1
        input_ids = self._flatten(input_ids)
        labels = self._flatten(labels) if labels is not None else labels
        attention_mask = self._flatten(attention_mask) if attention_mask is not None else attention_mask
        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)



class InternVLPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids("</img>")
        self.num_image_token = self.processor.num_image_token
        self.num_patches_list = None

    def validate_input(self, input_ids: List[int], examples: Dict[str, List[Any]], index: int) -> bool:
        r"""
        Validate if the number of images, videos and audios match the number of placeholders in messages.
        """
        num_image_tokens = input_ids.count(self.image_token_id) if self.image_token_id is not None else 0
        num_video_tokens = input_ids.count(self.video_token_id) if self.video_token_id is not None else 0
        num_audio_tokens = input_ids.count(self.audio_token_id) if self.audio_token_id is not None else 0

        images = examples["image"][index]
        # TODO: merge video and video_frames
        video_key = "video_frames" if len(examples["video_frames"][index]) > 0 else "video"
        videos = examples[video_key][index]
        audios = examples["audio"][index]

        if len(images) > num_image_tokens:
            examples["image"][index] = images[:num_image_tokens]
        if len(videos) > num_video_tokens:
            examples[video_key][index] = videos[:num_video_tokens]
        if len(audios) > num_audio_tokens:
            examples["audio"][index] = audios[:num_audio_tokens]
        image_check, video_check, audio_check = None, None, None
        warning_str = ""
        if num_image_tokens > 0:
            image_check = True
            if len(images) < num_image_tokens:
                warning_str += (
                    f"Number of images {len(images)} does not match number of image tokens {num_image_tokens} "
                )
                image_check = False
        if num_video_tokens > 0:
            video_check = True
            if len(videos) < num_video_tokens:
                warning_str += (
                    f"Number of videos {len(videos)} does not match number of video tokens {num_video_tokens} "
                )
                video_check = False
        if num_audio_tokens > 0:
            audio_check = True
            if len(audios) < num_audio_tokens:
                warning_str += (
                    f"Number of audios {len(audios)} does not match number of audio tokens {num_audio_tokens} "
                )
                audio_check = False
        check_result = [image_check, video_check, audio_check]
        if True not in check_result and False in check_result:
            logger.warning(warning_str)
            return False
        return True

    def _get_video_tokens(self, num_patches: List[int]):
        placeholder_sequence = []
        for i in range(len(num_patches)):
            num_frame_token_ids = self.tokenizer.encode(f"Frame{i + 1}: ", add_special_tokens=False)
            placeholder_num = num_patches[i] * self.num_image_token
            placeholder_token_ids = [self.image_token_id] * placeholder_num
            placeholder_sequence += (
                num_frame_token_ids + [self.vision_start_token_id] + placeholder_token_ids + [self.vision_end_token_id]
            )
        return placeholder_sequence

    def _get_image_tokens(self, num_patch: int):
        placeholder_num = num_patch * self.num_image_token
        return [self.vision_start_token_id] + [self.image_token_id] * placeholder_num + [self.vision_end_token_id]

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        num_patches_list = mm_features.pop("num_patches_list", torch.Tensor([]))
        if labels:
            mm_features.update({"image_flags": torch.ones(num_patches_list.sum())})
        media_index = 0
        for i in range(len(input_ids)):
            if input_ids[i] == self.image_token_id:
                if num_patches_list.dim() == 2:
                    placeholder_sequence = self._get_video_tokens(num_patches_list[media_index].tolist())
                else:
                    placeholder_sequence = self._get_image_tokens(num_patches_list[media_index].item())
                input_ids[i] = placeholder_sequence
                if labels:
                    labels[i] = [IGNORE_INDEX] * len(placeholder_sequence)
                if attention_mask:
                    attention_mask[i] = [attention_mask[i]] * len(placeholder_sequence)
                media_index += 1
        assert media_index == num_patches_list.shape[0]
        input_ids = self._flatten(input_ids)
        labels = self._flatten(labels) if labels is not None else labels
        attention_mask = self._flatten(attention_mask) if attention_mask is not None else attention_mask
        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)

    def process_single_mm_input_for_vllm(
        self,
        input_ids: List[int],
        mm_features: List[Dict[str, Any]],
    ):
        if MediaType.VIDEO in mm_features:
            input_ids = self.tokenizer.decode(input_ids)
            mm_features["image"] = self._flatten(mm_features.pop("video"))
            video_cnt = len(mm_features["image"])
            video_place = "\n".join(f"Image-{i + 1}: <image>" for i in range(video_cnt))
            input_ids = input_ids.replace("<video>", video_place)
            input_ids = self.tokenizer.encode(input_ids, add_special_tokens=False)
        return input_ids, mm_features

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        image_flags_sum = 0
        image_flag = "image_flags" in mm_features[0]
        for mm_feature in mm_features:
            image_flags_sum += mm_feature.pop("image_flags", torch.Tensor([])).sum().int().item()
            mm_feature.pop("num_patches_list", None)
        collated_mm_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)
        if len(mm_features) > 0 and image_flag:
            collated_mm_features["image_flags"] = torch.ones((image_flags_sum))
        return collated_mm_features



class MllamaPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")
        self.pad_to_multiple_of = 8 if self.tokenizer.padding_side == "right" else 1

    def _process_single_mm_input(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        images: Optional[List["ImageInput"]] = None,
        videos: Optional[List["VideoInput"]] = None,
        audios: Optional[List["AudioInput"]] = None,
    ):
        from transformers.models.mllama.processing_mllama import get_cross_attention_token_mask

        mm_features = self.get_mm_features(images=images, videos=videos, audios=audios)
        cross_attention_token_mask = get_cross_attention_token_mask(input_ids, self.image_token_id)
        mm_features["cross_attention_token_mask"] = cross_attention_token_mask
        processed_token_ids = self.process_single_token_ids(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, mm_features=mm_features
        )
        return processed_token_ids, mm_features

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        from transformers.models.mllama.processing_mllama import convert_sparse_cross_attention_mask_to_dense

        if batch_input_ids is not None:
            if len(batch_input_ids) == 0:
                return {}
            seq_len = len(batch_input_ids[0])
            assert len(batch_input_ids) <= 1, "Only support batch size 1 for mllama in inference"
        else:
            seq_len = collated_text_features["input_ids"].size(1)
        cross_attention_token_mask = [mm_feature.pop("cross_attention_token_mask", []) for mm_feature in mm_features]
        num_tiles = [mm_feature.pop("num_tiles", []) for mm_feature in mm_features]
        mm_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)
        if len(mm_features) == 0:
            return {}
        num_tiles = self._flatten(num_tiles)
        cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
            cross_attention_token_mask,
            length=seq_len,
            num_tiles=num_tiles,
            max_num_tiles=self.processor.image_processor.max_image_tiles,
        )
        mm_features["cross_attention_mask"] = torch.from_numpy(cross_attention_mask)
        return mm_features



class LLavaPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.patch_size = getattr(self.processor, "patch_size")
        self.num_additional_image_tokens = getattr(self.processor, "num_additional_image_tokens")
        self.vision_feature_select_strategy = getattr(self.processor, "vision_feature_select_strategy")

        if self.patch_size is None:
            self.patch_size = 14
        if self.num_additional_image_tokens is None:
            self.num_additional_image_tokens = 1
        if self.vision_feature_select_strategy is None:
            self.vision_feature_select_strategy = "default"

    def _get_image_tokens(self, pixel_values: "torch.Tensor"):
        height, width = get_image_size(to_numpy_array(pixel_values[0]))
        num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + self.num_additional_image_tokens
        if self.vision_feature_select_strategy == "default":
            num_image_tokens -= 1
        return [self.image_token_id] * num_image_tokens

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        image_index = 0
        for i in range(len(input_ids)):
            placeholder_sequence = None
            if input_ids[i] == self.image_token_id:
                placeholder_sequence = self._get_image_tokens(mm_features["pixel_values"])
                input_ids[i] = placeholder_sequence
                image_index += 1
            if labels and placeholder_sequence:
                labels[i] = [IGNORE_INDEX] * len(placeholder_sequence)
            if attention_mask and placeholder_sequence:
                attention_mask[i] = [attention_mask[i]] * len(placeholder_sequence)
        input_ids, labels = self._flatten(input_ids), self._flatten(labels) if labels is not None else None
        attention_mask = self._flatten(attention_mask) if attention_mask is not None else attention_mask
        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)



class LlavaOneVisionPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<video>")

        num_image_tokens = getattr(self.processor, "num_image_tokens", 1)
        patches_height_width = int(math.sqrt(num_image_tokens))
        self.pooled_height_width = math.ceil(patches_height_width / 2)
        self.num_additional_image_tokens = 1
        self.channel_dim = ChannelDimension.FIRST

    def _get_video_tokens(self, pixel_values_videos: "torch.Tensor"):
        num_frames = pixel_values_videos.shape[1]  # frame dim is always after batch dim
        num_video_tokens = (num_frames * self.pooled_height_width * self.pooled_height_width) + 1
        return [self.video_token_id] * num_video_tokens

    def _get_image_tokens(self, pixel_values: "torch.Tensor", image_size: "torch.Tensor"):
        if not isinstance(image_size, (list, tuple)):
            # cast to list to avoid numerical precision errors when calculating unpadding
            image_size = image_size.tolist()
        orig_height, orig_width = image_size
        height, width = get_image_size(to_numpy_array(pixel_values[0][0]), channel_dim=self.channel_dim)
        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
        if self.processor.vision_feature_select_strategy == "default":
            num_image_tokens -= self.num_additional_image_tokens
        return [self.image_token_id] * num_image_tokens

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        image_index, video_index = 0, 0
        for i in range(len(input_ids)):
            placeholder_sequence = None
            if input_ids[i] == self.image_token_id:
                placeholder_sequence = self._get_image_tokens(
                    mm_features["pixel_values"], image_size=mm_features["image_sizes"][image_index]
                )
                input_ids[i] = placeholder_sequence
                image_index += 1
            elif input_ids[i] == self.video_token_id:
                placeholder_sequence = self._get_video_tokens(mm_features["pixel_values_videos"])
                input_ids[i] = placeholder_sequence
                video_index += 1
            if labels and placeholder_sequence:
                labels[i] = [IGNORE_INDEX] * len(placeholder_sequence)
            if attention_mask and placeholder_sequence:
                attention_mask[i] = [attention_mask[i]] * len(placeholder_sequence)
        input_ids, labels = self._flatten(input_ids), self._flatten(labels) if labels is not None else None
        attention_mask = self._flatten(attention_mask) if attention_mask is not None else attention_mask
        if "image_sizes" in mm_features:
            assert len(mm_features["image_sizes"]) == image_index
        if "pixel_values_videos" in mm_features:
            assert len(mm_features["pixel_values_videos"]) == video_index
        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)



class LlavaNextVideoPlugin(LlavaOneVisionPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_dim = None
        self.num_additional_image_tokens = self.processor.num_additional_image_tokens
        self.patch_size = self.processor.patch_size

    def _get_video_tokens(self, pixel_values_videos: "torch.Tensor"):
        height, width = get_image_size(pixel_values_videos[0][0])
        num_frames = pixel_values_videos.shape[1]  # frame dim is always after batch dim
        num_image_tokens = (height // self.patch_size) * (width // self.patch_size)
        num_video_tokens = num_image_tokens // 4 * num_frames  # divide by 4 needed for avg pooling layer
        return [self.video_token_id] * num_video_tokens



class TBStarsOmniPluginBase(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.image_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.end_sep_id = 213  # represent \n
        self.concat_sep_id = 127996  # represent \n\n
        self.image_placeholder_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.audio_placeholder_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.sampling_rate = self.processor.audio_processor.sampling_rate

    def _process_single_token_ids(
        self,
        input_ids: List[int],
        media_num_tokens: List[int],
        media_type: MediaType,
        labels: List[int] = None,
        attention_mask: List[int] = None,
        **kwargs,
    ):
        media_index = 0
        placeholder_token_id = (
            self.image_placeholder_id if media_type == MediaType.IMAGE else self.audio_placeholder_id
        )
        media_token_id = self.image_token_id if media_type == MediaType.IMAGE else self.audio_token_id
        media_start_id = self.image_start_id if media_type == MediaType.IMAGE else self.audio_start_id
        media_end_id = self.image_end_id if media_type == MediaType.IMAGE else self.audio_end_id
        for idx, token_id in enumerate(input_ids):
            if token_id == media_token_id:
                placeholder_token_ids = [placeholder_token_id] * media_num_tokens[media_index]
                placeholder_sequence = [media_start_id] + placeholder_token_ids + [media_end_id]
                input_ids[idx] = placeholder_sequence
                if labels:
                    labels[idx] = [IGNORE_INDEX] * len(placeholder_sequence)
                if attention_mask:
                    attention_mask[idx] = [attention_mask[idx]] * len(placeholder_sequence)
                media_index += 1
        return (
            self._flatten(input_ids),
            self._flatten(labels) if labels is not None else labels,
            self._flatten(attention_mask) if attention_mask is not None else attention_mask,
        )

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        if "image_num_tokens" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                media_type=MediaType.IMAGE,
                media_num_tokens=mm_features.pop("image_num_tokens"),
            )
        if "audio_num_tokens" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                media_type=MediaType.AUDIO,
                attention_mask=attention_mask,
                media_num_tokens=mm_features.pop("audio_num_tokens"),
            )
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        return super().process_single_token_ids(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, mm_features=mm_features
        )

    def _audios_to_features(self, audios: List[Any]):
        audio_max_segments = 20
        num_tokens_per_audio = 375
        audio_features, segment_indices = list(), list()
        for audio in audios:
            try:
                audio = torch.from_numpy(audio).unsqueeze(0)
                assert len(audio) > 0 and audio.ndim == 2 and audio.shape[1] > 400, (
                    f"audio is None or len(audio) == 0 or not hasattr(audio, 'shape')"
                )
            except:
                audio = torch.full(size=(1, 24000), fill_value=0.0, dtype=torch.float32)
            processor_output = self.processor.audio_processor(audio, self.sampling_rate, self.sampling_rate)
            if processor_output["audios"].shape[0] > audio_max_segments:
                logger.info(f"audio is too long, truncate first {30 * audio_max_segments:d}s for inputs.")
            audio_features.append(processor_output["audios"][:audio_max_segments])
            segment_indices.append(processor_output["segment_indices"][:audio_max_segments])

        audio_inputs = {
            "audios": torch.cat(audio_features, dim=0),
            "segment_indices": torch.cat(segment_indices, dim=0),
            "audio_num_tokens": [num_tokens_per_audio * len(_) for _ in segment_indices],
        }
        return audio_inputs

    def _get_image_resolution_mode(self, num_images, images_resolution_mode=None):
        # 1：动态裁切；0：不裁切，直接resize；
        if images_resolution_mode is not None:
            return images_resolution_mode
        ada_crop_mode = self.processor.image_processor.adaptive_crop_mode
        if ada_crop_mode is None:
            return [1] * num_images
        if ada_crop_mode == "ecom_detail" or (ada_crop_mode == "ecom_wiki" and num_images > 16):
            return [1] + [0] * (num_images - 1)
        return [1] * num_images

    def _images_process_adaptive_crop(self, images: List["ImageObject"]):
        num_tokens_per_image = 144
        image_processor = self.processor.image_processor

        def _process_single_image(image_source_map):
            current_image, current_processing_kwargs = image_source_map
            current_processing_kwargs.update(dict(return_tensors="pt", require_crop_position=True))
            try:
                image_processed = image_processor.preprocess([current_image], **current_processing_kwargs)
            except Exception as e:
                logger.info(f"Image processor exception for file: {current_image}, exception={str(e)}")
                current_image = Image.fromarray((np.full((224, 224, 3), OPENAI_CLIP_MEAN) * 255).astype(np.uint8))
                image_processed = image_processor.preprocess([current_image], **current_processing_kwargs)
            return dict(images=image_processed["images"][0], crop_positions=image_processed["crop_positions"][0])

        if len(images) > 0:
            resolution_mode: List[int] = self._get_image_resolution_mode(len(images))
            image_processing_kwargs = [{"do_adaptive_crop": bool(_)} for _ in resolution_mode]
            assert len(images) == len(image_processing_kwargs)
            image_source_maps = list(zip(images, image_processing_kwargs))
            # with ThreadPoolExecutor(max_workers=8) as executor:
            all_source_processed = [_process_single_image(_) for _ in image_source_maps]
            images_processed = {
                "images": torch.cat([_["images"] for _ in all_source_processed], dim=0),
                "crop_positions": torch.cat([_["crop_positions"] for _ in all_source_processed], dim=0),
                "image_num_tokens": [num_tokens_per_image * len(_["crop_positions"]) for _ in all_source_processed],
            }
        return images_processed

    def _images_process_atom_grid(self, images: List["ImageObject"]):
        image_processor = self.processor.image_processor

        patch_height = image_processor.patch_size["height"]
        patch_width = image_processor.patch_size["width"]
        atom_grid_size = image_processor.atom_grid_size
        max_image_patches = image_processor.max_image_patches

        def _process_single_image(image_source_map):
            current_image, current_processing_kwargs = image_source_map
            try:
                image_processed = image_processor.preprocess([current_image], **current_processing_kwargs)
            except Exception as e:
                logger.info(f"Image processor exception for file: {current_image}, exception={str(e)}")
                current_image = Image.fromarray((np.full((224, 224, 3), OPENAI_CLIP_MEAN) * 255).astype(np.uint8))
                image_processed = image_processor.preprocess([current_image], **current_processing_kwargs)
            return dict(images=image_processed["images"], patch_position_infos=image_processed["patch_position_infos"])

        all_max_image_patches, all_image_scale = None, None
        images_shape = [img.size for img in images]
        all_max_image_patches = [max_image_patches] * len(images_shape)
        if len(images_shape) > 50:
            raise ValueError("too many input images, please truncate number of images to `50`.")
        if any(max(_h, _w) < 56 or min(_h, _w) < 14 for _h, _w in images_shape):
            raise ValueError("images shape error, input image shape is too small")
        all_ori_num_patches = [
            math.ceil(image_shape[1] / (patch_height * atom_grid_size))
            * atom_grid_size
            * math.ceil(image_shape[0] / (patch_width * atom_grid_size))
            * atom_grid_size
            for image_shape in images_shape
        ]
        all_num_patches = min(sum(all_ori_num_patches), sum(all_max_image_patches))
        if all_num_patches > max_image_patches:
            all_image_scale = []
            for (
                idx,
                image_shape,
            ) in enumerate(images_shape):
                current_num_patches = min(all_ori_num_patches[idx], all_max_image_patches[idx])
                target_num_patches = math.floor(max_image_patches / all_num_patches * current_num_patches)
                assert target_num_patches >= 4
                target_num_pixels = math.floor(target_num_patches * patch_height * patch_width)
                original_num_pixels = math.ceil(float(image_shape[0] * image_shape[1]))
                all_image_scale.append(math.sqrt(target_num_pixels / original_num_pixels))

        image_processing_kwargs = []
        for source_index, image in enumerate(images):
            source_processing_kwargs = dict(return_tensors="pt")
            if all_max_image_patches is not None:
                source_processing_kwargs.update({"max_image_patches": all_max_image_patches[source_index]})
            if all_image_scale is not None:
                source_processing_kwargs.update({"image_scale": all_image_scale[source_index]})
            image_processing_kwargs.append(source_processing_kwargs)

        assert len(images) == len(image_processing_kwargs)
        image_source_maps = list(zip(images, image_processing_kwargs))
        # with ThreadPoolExecutor(max_workers=8) as executor:
        all_source_processed = [_process_single_image(_) for _ in image_source_maps]
        images_processed = {
            "images": torch.cat([_["images"] for _ in all_source_processed], dim=0),
            "patch_position_infos": torch.cat([_["patch_position_infos"] for _ in all_source_processed], dim=0),
            "image_num_tokens": [len(_["images"]) // (atom_grid_size**2) + 2 for _ in all_source_processed],
        }
        return images_processed

    def _images_to_features(self, images: List["ImageObject"]):
        raise NotImplementedError()


class TBStarsOmniPlugin(TBStarsOmniPluginBase):
    def _images_to_features(self, images: List["ImageObject"]):
        return self._images_process_adaptive_crop(images)


class TBStars2OmniPlugin(TBStarsOmniPluginBase):
    def _images_to_features(self, images: List["ImageObject"]):
        return self._images_process_atom_grid(images)


class TBStars2_5_OmniPlugin(TBStarsOmniPluginBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.video_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.video_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.video_placeholder_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.vision_spatial_merge_size = self.processor.image_processor.atom_grid_size

    def _process_single_token_ids(
        self,
        input_ids: List[int],
        media_num_tokens: List[int],
        media_type: MediaType,
        labels: List[int] = None,
        attention_mask: List[int] = None,
        **kwargs,
    ):
        media_index = 0
        if media_type == MediaType.IMAGE:
            placeholder_token_id = self.image_placeholder_id
            media_token_id = self.image_token_id
            media_start_id = self.image_start_id
            media_end_id = self.image_end_id
        elif media_type == MediaType.VIDEO:
            placeholder_token_id = self.video_placeholder_id
            media_token_id = self.video_token_id
            media_start_id = self.video_start_id
            media_end_id = self.video_end_id
        elif media_type == MediaType.AUDIO:
            placeholder_token_id = self.audio_placeholder_id
            media_token_id = self.audio_token_id
            media_start_id = self.audio_start_id
            media_end_id = self.audio_end_id
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        for idx, token_id in enumerate(input_ids):
            if token_id == media_token_id:
                placeholder_token_ids = [placeholder_token_id] * media_num_tokens[media_index]
                placeholder_sequence = [media_start_id] + placeholder_token_ids + [media_end_id]
                input_ids[idx] = placeholder_sequence
                if labels:
                    labels[idx] = [IGNORE_INDEX] * len(placeholder_sequence)
                if attention_mask:
                    attention_mask[idx] = [attention_mask[idx]] * len(placeholder_sequence)
                media_index += 1
        return (
            self._flatten(input_ids),
            self._flatten(labels) if labels is not None else labels,
            self._flatten(attention_mask) if attention_mask is not None else attention_mask,
        )

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        if "image_num_tokens" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                media_type=MediaType.IMAGE,
                media_num_tokens=mm_features.pop("image_num_tokens"),
            )
        elif "video_num_tokens" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                media_type=MediaType.VIDEO,
                media_num_tokens=mm_features.pop("video_num_tokens"),
            )
        elif "audio_num_tokens" in mm_features:
            input_ids, labels, attention_mask = self._process_single_token_ids(
                input_ids=input_ids,
                labels=labels,
                media_type=MediaType.AUDIO,
                attention_mask=attention_mask,
                media_num_tokens=mm_features.pop("audio_num_tokens"),
            )
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        return super().process_single_token_ids(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, mm_features=mm_features
        )

    def _images_process_atom_grid(self, images: List["ImageObject"]):
        image_processor = self.processor.image_processor

        patch_height = image_processor.patch_size["height"]
        patch_width = image_processor.patch_size["width"]
        max_image_patches = image_processor.max_image_patches

        def _process_single_image(image_source_map):
            current_image, current_processing_kwargs = image_source_map
            try:
                image_processed = image_processor.preprocess([current_image], **current_processing_kwargs)
            except Exception as e:
                logger.info(f"Image processor exception for file: {current_image}, exception={str(e)}")
                current_image = Image.fromarray((np.full((224, 224, 3), OPENAI_CLIP_MEAN) * 255).astype(np.uint8))
                image_processed = image_processor.preprocess([current_image], **current_processing_kwargs)
            return dict(images=image_processed["images"], image_grid_thw=image_processed["image_grid_thw"])

        all_max_image_patches, all_image_scale = None, None
        images_shape = [img.size for img in images]
        all_max_image_patches = [max_image_patches] * len(images_shape)
        if len(images_shape) > 50:
            raise ValueError("too many input images, please truncate number of images to `50`.")
        if any(max(_h, _w) < 56 or min(_h, _w) < 14 for _h, _w in images_shape):
            raise ValueError("images shape error, input image shape is too small")
        all_ori_num_patches = [
            math.ceil(image_shape[1] / (patch_height * self.vision_spatial_merge_size)) * self.vision_spatial_merge_size *
            math.ceil(image_shape[0] / (patch_width * self.vision_spatial_merge_size)) * self.vision_spatial_merge_size
            for image_shape in images_shape
        ]
        all_num_patches = min(sum(all_ori_num_patches), sum(all_max_image_patches))
        if all_num_patches > max_image_patches:
            all_image_scale = []
            for idx, image_shape, in enumerate(images_shape):
                current_num_patches = min(all_ori_num_patches[idx], all_max_image_patches[idx])
                target_num_patches = math.floor(max_image_patches / all_num_patches * current_num_patches)
                assert target_num_patches >= 4
                target_num_pixels = math.floor(target_num_patches * patch_height * patch_width)
                original_num_pixels = math.ceil(float(image_shape[0] * image_shape[1]))
                all_image_scale.append(math.sqrt(target_num_pixels / original_num_pixels))

        image_processing_kwargs = []
        for source_index, image in enumerate(images):
            source_processing_kwargs = dict(return_tensors="pt")
            if all_max_image_patches is not None:
                source_processing_kwargs.update({"max_image_patches": all_max_image_patches[source_index]})
            if all_image_scale is not None:
                source_processing_kwargs.update({"image_scale": all_image_scale[source_index]})
            image_processing_kwargs.append(source_processing_kwargs)

        assert len(images) == len(image_processing_kwargs)
        image_source_maps = list(zip(images, image_processing_kwargs))
        # with ThreadPoolExecutor(max_workers=8) as executor:
        all_source_processed = [_process_single_image(_) for _ in image_source_maps]
        images_processed = {
            "images": torch.cat([_["images"] for _ in all_source_processed], dim=0),
            "image_grid_thw": torch.cat([_["image_grid_thw"] for _ in all_source_processed], dim=0),
            "image_num_tokens": [len(_["images"]) // (self.vision_spatial_merge_size ** 2) for _ in all_source_processed],
        }
        return images_processed

    def _images_to_features(self, images: List["ImageObject"]):
        return self._images_process_atom_grid(images)

    def _regularize_video(self, video: "VideoInput", **kwargs) -> List["ImageObject"]:
        return self.processor._load_videos([video], videos_fps=[self.processor_args.video_fps])

    def _videos_to_features(self, videos):
        videos_input = [video[0] for video in videos]
        videos_meta = [video[1] for video in videos]
        videos_fps = [video_meta["videos_fps"][0] for video_meta in videos_meta]
        videos_processed = self.processor.video_processor(
            videos=videos_input, return_tensors="pt", videos_fps=videos_fps
        )
        video_grid_thw = videos_processed["video_grid_thw"]
        videos_processed.update(
            {
                "video_num_tokens": [
                    video_grid_thw[video_idx].prod().item() // self.vision_spatial_merge_size**2
                    for video_idx in range(video_grid_thw.shape[0])
                ]
            }
        )
        return videos_processed

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        mm_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)
        if collated_text_features is not None:
            position_ids, rope_deltas = self.processor.get_rope_index(**collated_text_features, **mm_features)
            mm_features.update(dict(position_ids=position_ids, rope_deltas=rope_deltas))
        return mm_features


class TBStarsVistaPlugin(TBStarsOmniPluginBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.image_end_id = self.tokenizer.convert_tokens_to_ids("</image>")
        self.image_placeholder_id = self.tokenizer.convert_tokens_to_ids("<|image|>")

    def _images_to_features(self, images: List["ImageObject"]):
        return self._images_process_atom_grid(images)


class TBStars008VLPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<video>")
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("</image>")
        self.end_sep_id = 213  # represent \n
        self.concat_sep_id = 127996  # represent \n\n
        self.image_place_holder_id = self.tokenizer.convert_tokens_to_ids("<PLACEHOLDER_33>")
        self.video_place_holder_id = self.tokenizer.convert_tokens_to_ids("<PLACEHOLDER_34>")

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        def _get_image_token_ids(media_len):
            return [self.vision_start_id] + [self.image_place_holder_id] * media_len + [self.vision_end_id]

        def _get_video_token_ids(media_lens):
            res = []
            for media_len in media_lens:
                if media_len <= 0:
                    continue
                res += (
                    [self.vision_start_id]
                    + [self.video_place_holder_id] * media_len
                    + [self.vision_end_id, self.end_sep_id]
                )
            return res

        image_index, video_index = 0, 0
        image_lens, video_lens = [], []
        proj_len = getattr(self.processor.image_processor, "perceiver_num_queries", 144)
        if "crop_positions" in mm_features:
            crop_positions = mm_features["crop_positions"]
            cumsum_positions = torch.tensor(
                [torch.allclose(p, crop_positions[0]) for p in crop_positions] + [True]
            ).nonzero()
            image_lens = ((cumsum_positions[1:] - cumsum_positions[:-1]) * proj_len).squeeze(1).tolist()
        if "video_chunk_sizes" in mm_features:
            video_lens = (mm_features["video_chunk_sizes"] * proj_len).int().tolist()

        for i in range(len(input_ids)):
            if input_ids[i] == self.image_token_id:
                input_ids[i] = _get_image_token_ids(image_lens[image_index])
                image_index += 1
            elif input_ids[i] == self.video_token_id:
                input_ids[i] = _get_video_token_ids(video_lens[video_index])
                video_index += 1
            else:
                continue
            if labels is not None:
                labels[i] = [IGNORE_INDEX] * len(input_ids[i])
            if attention_mask is not None:
                attention_mask[i] = [attention_mask[i]] * len(input_ids[i])
        assert image_index == len(image_lens) and video_index == len(video_lens), (
            f"image_index: {image_index}, video_index: {video_index}, image_lens: {image_lens}, video_lens: {video_lens}"
        )
        input_ids = self._flatten(input_ids)
        if labels is not None:
            labels = self._flatten(labels)
        if attention_mask is not None:
            attention_mask = self._flatten(attention_mask)
        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)

    def _videos_to_features(self, videos):
        return self.processor.image_processor(None, videos=videos, return_tensors="pt")

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        video_chunk_sizes = None
        for f in mm_features:
            video_chunk_sizes = f.pop("video_chunk_sizes", None)  # not needed in forward
        mm_collated_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)
        if video_chunk_sizes is not None:  # mock value for vllm infer
            mm_collated_features["video_chunk_sizes"] = video_chunk_sizes.unsqueeze(0)
        return mm_collated_features



class DeepseekVL2Plugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        image_index = 0
        num_image_tokens = mm_features.pop("num_image_tokens", None)
        for i in range(len(input_ids)):
            if input_ids[i] == self.image_token_id:
                input_ids[i] = [self.image_token_id] * num_image_tokens[image_index]
                if labels:
                    labels[i] = [IGNORE_INDEX] * len(input_ids[i])
                if attention_mask:
                    attention_mask[i] = [attention_mask[i]] * len(input_ids[i])
                image_index += 1
        input_ids, labels = self._flatten(input_ids), self._flatten(labels) if labels is not None else None
        attention_mask = self._flatten(attention_mask) if attention_mask is not None else None
        assert image_index == len(num_image_tokens), (
            f"image_index: {image_index}, num_image_tokens: {num_image_tokens}"
        )
        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        """padding images to max_patch_num"""
        max_n_patches = max(feature["images"].shape[0] for feature in mm_features)
        batched_images = []
        for feature in mm_features:
            images = feature["images"]
            n_pads = max_n_patches - images.shape[0]
            if n_pads > 0:
                pad_images = torch.zeros((n_pads, *images.shape[1:]), dtype=images.dtype)
                images = torch.cat([images, pad_images], dim=0)
            batched_images.append(images)
        batched_images = torch.stack(batched_images, dim=0)

        """padding images_spatial_crop to max_n_images"""
        max_n_images = max(feature["images_spatial_crop"].shape[0] for feature in mm_features)
        batched_images_spatial_crop = []
        for feature in mm_features:
            images_spatial_crop = feature["images_spatial_crop"]
            n_pads = max_n_images - feature["images_spatial_crop"].shape[0]
            if n_pads > 0:
                pad_images_spatial_crop = torch.full((n_pads, 2), 0, dtype=images_spatial_crop.dtype)
                images_spatial_crop = torch.cat([images_spatial_crop, pad_images_spatial_crop], dim=0)
            batched_images_spatial_crop.append(images_spatial_crop)
        batched_images_spatial_crop = torch.stack(batched_images_spatial_crop, dim=0)
        return {"images": batched_images, "images_spatial_crop": batched_images_spatial_crop}



class MiniCPMOPlugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio|>")
        self.sampling_rate = 16000

    def _videos_to_features(self, videos: List[List["ImageObject"]]):
        return self.processor.image_processor(videos, media_type=MediaType.VIDEO, return_tensors="pt")

    def _audios_to_features(self, audios: List[Any]):
        audio_features, audio_feature_lens, audio_phs = self.processor.audio_feature_extract(
            audios, sampling_rate=self.sampling_rate
        )
        return {
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "audio_phs": audio_phs,
        }

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        image_index, audio_idx = 0, 0
        image_sizes = mm_features.pop("image_sizes", None)
        audio_phs = mm_features.pop("audio_phs", None)
        for i in range(len(input_ids)):
            if input_ids[i] == self.image_token_id or input_ids[i] == self.video_token_id:
                image_placeholder = self.processor.image_processor.get_slice_image_placeholder(
                    image_sizes[0][image_index], image_index, max_slice_nums=None, use_image_id=True
                )
                placeholder_sequence = self.tokenizer.encode(image_placeholder)
                input_ids[i] = placeholder_sequence
                if labels:
                    labels[i] = [IGNORE_INDEX] * len(input_ids[i])
                if attention_mask:
                    attention_mask[i] = [attention_mask[i]] * len(input_ids[i])
                image_index += 1
            elif input_ids[i] == self.audio_token_id:
                audio_placeholder = audio_phs[0][audio_idx]
                placeholder_sequence = self.tokenizer.encode(audio_placeholder)
                input_ids[i] = placeholder_sequence
                if labels:
                    labels[i] = [IGNORE_INDEX] * len(input_ids[i])
        input_ids, labels = self._flatten(input_ids), self._flatten(labels) if labels is not None else None
        attention_mask = self._flatten(attention_mask) if attention_mask is not None else None
        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)

    def mm_collate_fn(self, mm_features, collated_text_features=None, batch_input_ids=None, model=None):
        if batch_input_ids is not None:
            if len(batch_input_ids) == 0:
                return {}
            assert len(batch_input_ids) <= 1, "Only support batch size 1 for MiniCPMO in inference"
        else:
            batch_input_ids = collated_text_features["input_ids"]

        image_bounds_list, audio_bounds_list, spk_bounds_list = [], [], []
        for input_ids in batch_input_ids:
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)

            ## image bound
            start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
            end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

            image_start_idx = torch.where(start_cond)[0]
            image_start_idx += 1
            image_end_idx = torch.where(end_cond)[0]

            valid_image_nums = max(len(image_start_idx), len(image_end_idx))

            image_bounds = torch.hstack(
                [
                    image_start_idx[:valid_image_nums].unsqueeze(-1),
                    image_end_idx[:valid_image_nums].unsqueeze(-1),
                ]
            )

            ##  audio bound
            audio_start_idx = torch.where(input_ids == self.tokenizer.audio_start_id)[0]
            audio_end_idx = torch.where(input_ids == self.tokenizer.audio_end_id)[0]
            assert len(audio_start_idx) == len(audio_end_idx)
            audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])

            spk_start_idx = torch.where(input_ids == self.tokenizer.spk_start_id)[0]
            spk_end_idx = torch.where(input_ids == self.tokenizer.spk_end_id)[0]
            assert len(spk_start_idx) == len(spk_end_idx)
            spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])

            image_bounds_list.append(image_bounds)
            audio_bounds_list.append(audio_bounds)
            spk_bounds_list.append(spk_bounds)

        batch_mm_features = {
            **mm_features[0],
            "image_bound": torch.stack(image_bounds_list),
            "audio_bounds": torch.stack(audio_bounds_list),
            "spk_bounds": torch.stack(spk_bounds_list),
            "tokenizer": self.tokenizer,
        }

        if "audio_features" not in batch_mm_features:
            batch_mm_features["audio_features"] = []

        if "pixel_values" not in batch_mm_features:
            batch_mm_features["pixel_values"] = [[]]

        if collated_text_features is None:
            return batch_mm_features

        bsz, seq_length = collated_text_features["input_ids"].shape
        collated_text_features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)

        batch_features = {
            "input_ids": collated_text_features["input_ids"],
            "labels": collated_text_features["labels"],
            "data": batch_mm_features,
        }
        batch_features["data"].update(collated_text_features)
        collated_text_features.clear()
        return batch_features



class Gemma3Plugin(CommonPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boi_token = "<start_of_image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.boi_token)
        self.image_placeholder_token_id = self.processor.image_token_id
        self.full_image_sequence = self.tokenizer.encode(self.processor.full_image_sequence, add_special_tokens=False)

    def process_single_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
        mm_features: Dict[str, Any] = None,
    ):
        image_index = 0
        batch_num_crops = [mm_features.pop("num_crops", [])]
        num_crops = batch_num_crops[0]
        # apply crops
        if sum(num_crops) > 0:
            for i in range(len(input_ids)):
                placeholder_sequence = None
                if input_ids[i] == self.image_token_id:
                    if num_crops[image_index]:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num_crops[image_index])
                        )
                        placeholder_sequence = self.tokenizer.encode(formatted_image_text, add_special_tokens=False)
                        input_ids[i] = placeholder_sequence
                        if labels:
                            labels[i] = [IGNORE_INDEX] * len(placeholder_sequence)
                        if attention_mask:
                            attention_mask[i] = [attention_mask[i]] * len(placeholder_sequence)
                        image_index += 1
            input_ids, labels = self._flatten(input_ids), self._flatten(labels) if labels is not None else None
            attention_mask = self._flatten(attention_mask) if attention_mask is not None else None

        # process image
        image_index = 0
        for i in range(len(input_ids)):
            placeholder_sequence = None
            if input_ids[i] == self.image_token_id:
                placeholder_sequence = self.full_image_sequence
                input_ids[i] = placeholder_sequence
                if labels:
                    labels[i] = [IGNORE_INDEX] * len(placeholder_sequence)
                if attention_mask:
                    attention_mask[i] = [attention_mask[i]] * len(placeholder_sequence)
                image_index += 1
        input_ids, labels = self._flatten(input_ids), self._flatten(labels) if labels is not None else None
        attention_mask = self._flatten(attention_mask) if attention_mask is not None else None

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        input_tensor = torch.tensor(input_ids)
        mm_token_type_ids = (input_tensor == self.image_placeholder_token_id).to(torch.int64)
        mm_features["token_type_ids"] = mm_token_type_ids

        return super().process_single_token_ids(input_ids, labels, attention_mask, mm_features)

    def mm_collate_fn(
        self,
        mm_features: List[Dict[str, Any]],
        collated_text_features: Optional[Dict[str, Any]] = None,
        batch_input_ids: Optional[List[List[int]]] = None,
        model: Optional[Any] = None,
    ):
        token_type_ids = [mm_feature.pop("token_type_ids") for mm_feature in mm_features]
        max_input_length = max(token_type_id.shape[0] for token_type_id in token_type_ids)
        new_token_type_ids = []
        mm_features = super().mm_collate_fn(mm_features, collated_text_features, batch_input_ids)

        # generation, left padding
        if batch_input_ids is not None:
            for token_type_id in token_type_ids:
                n_pads = max_input_length - token_type_id.shape[0]
                padding_sequence = torch.zeros(n_pads, dtype=token_type_id.dtype)
                token_type_id = torch.cat([padding_sequence, token_type_id], dim=0)
                new_token_type_ids.append(token_type_id)

        # training, right padding
        if collated_text_features is not None:
            for token_type_id in token_type_ids:
                n_pads = max_input_length - token_type_id.shape[0]
                padding_sequence = torch.zeros(n_pads, dtype=token_type_id.dtype)
                token_type_id = torch.cat([token_type_id, padding_sequence], dim=0)
                new_token_type_ids.append(token_type_id)

        if len(mm_features) > 0:
            mm_features["token_type_ids"] = torch.stack(new_token_type_ids)

        return mm_features


PLUGINS = {
    "base": BasePlugin,
    "glm4v": CommonPlugin,
    "chatml": CommonPlugin,
    "turing": CommonPlugin,
    "qwen3_vl_nothink": Qwen3VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "qwen2-vl": Qwen2VLPlugin,
    "qwen2-omni": Qwen2OMNIPlugin,
    "qwen3_omni": Qwen3OMNIPlugin,
    "qwen3_omni_nothink": Qwen3OMNIPlugin,
    "qwen3_5": Qwen3VLPlugin,
    "qwen3_5_nothink": Qwen3VLPlugin,
    "qvq": Qwen2VLPlugin,
    "qvq_preview": Qwen2VLPlugin,
    "qwen2-audio": Qwen2AudioPlugin,
    "internvl2": InternVLPlugin,
    "internvl3": InternVLPlugin,
    "internvl1": InternVLPlugin,
    "internvl2_Phi_3": InternVLPlugin,
    "llama3_v": MllamaPlugin,
    "vicuna": LLavaPlugin,
    "vip-llava-hf": LLavaPlugin,
    "mistral": LlavaOneVisionPlugin,
    "vicuna-next": LlavaOneVisionPlugin,
    "llava-next-yi": LlavaOneVisionPlugin,
    "llava_onevision_qwen2": LlavaOneVisionPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "tbstars_omni": TBStarsOmniPlugin,
    "tbstars008vl": TBStars008VLPlugin,
    "tbstars2_moe_vista": TBStarsVistaPlugin,
    "tbstars2_omni": TBStars2OmniPlugin,
    "tbstars2_omni_plain": TBStars2OmniPlugin,
    "tbstars2_5_omni": TBStars2_5_OmniPlugin,
    "deepseek-vl2": DeepseekVL2Plugin,
    "minicpmo": MiniCPMOPlugin,
    "minicpmo-tts": MiniCPMOPlugin,
    "gemma3": Gemma3Plugin,
}


def get_mm_plugin(
    name: str,
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    processor_args: Optional["ProcessorArguments"],
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        plugin_class = BasePlugin
    return plugin_class(tokenizer=tokenizer, processor=processor, processor_args=processor_args)