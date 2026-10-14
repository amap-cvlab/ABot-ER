# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/modeling_llava.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers.utils import logging

from ...extras.constants import MediaType
from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


logger = get_logger(__name__)
transformers_logger = logging.get_logger(__name__)


def autocast_projector_dtype(
    model: "PreTrainedModel", model_args: "ModelArguments", mm_projector_name: str = "multi_modal_projector"
) -> None:
    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if hasattr(model, mm_projector_name) and getattr(model, "quantization_method", None):
        logger.info("Casting multimodal projector outputs in {}.".format(model_args.compute_dtype))
        mm_projector: "torch.nn.Module" = getattr(model, mm_projector_name)
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)


def configure_visual_model(config: "PretrainedConfig") -> None:
    text_model_config_names = ["text_config", "language_config", "llm_config"]
    for text_model_config_name in text_model_config_names:
        if getattr(config, text_model_config_name, None) and not getattr(config, "hidden_size", None):
            # required for ds zero3 and valuehead models
            setattr(config, "hidden_size", getattr(config, text_model_config_name).hidden_size)


class CommonVisualProcessor(object):
    def __init__(self, vision_config):
        self.image_size: int = vision_config['image_size']
        self.patch_size = vision_config['patch_size'] if 'patch_size' in vision_config else None

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.image_processor = self

    def save_pretrained(self, *args, **kwargs):
        pass

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs):
        images_after = []
        for image_paths in images:
            for image in image_paths:
                images_after.append(self.image_transform(image).to(torch.float16))
        images_after = torch.stack(images_after, dim=0)
        return {'images': images_after}


class InternVL2VisualProcessor(object):
    def __init__(self, vision_config):
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.vision_config = vision_config
        self.input_size = vision_config.image_size # 448
        self.patch_size = vision_config.patch_size # 14
        self.image_transform = self._build_transform(input_size=self.input_size)
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        self.downsample_ratio = 0.5
        self.num_image_token = int((vision_config.image_size // vision_config.patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.image_processor = self

    def save_pretrained(self, *args, **kwargs):
        pass

    def _build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(self, multimedia_objs, media_type=MediaType.IMAGE, **kwargs):
        pixel_values_list = []
        num_patches_list = []
        for multimedia_obj in multimedia_objs:
            if media_type == MediaType.IMAGE:
                images = self._dynamic_preprocess(multimedia_obj, image_size=self.input_size, use_thumbnail=True)
                pixel_values = [self.image_transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                num_patches_list.append(pixel_values.size(0))
                pixel_values_list.append(pixel_values)
            elif media_type == MediaType.VIDEO:
                sub_video_num_patches_list = []
                for frame in multimedia_obj:
                    img = self._dynamic_preprocess(frame, image_size=self.input_size, use_thumbnail=True, max_num=1)
                    pixel_values = [self.image_transform(tile) for tile in img]
                    pixel_values = torch.stack(pixel_values)
                    sub_video_num_patches_list.append(pixel_values.shape[0])
                    pixel_values_list.append(pixel_values)
                num_patches_list.append(sub_video_num_patches_list)
        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = torch.LongTensor(num_patches_list)
        return {'pixel_values': pixel_values, 'num_patches_list': num_patches_list}


class InternLMXcomposerVisualProcessor(object):
    def __init__(self) -> None:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.font = self.get_font()
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.image_resize = transforms.Resize((1120, 1120), interpolation=InterpolationMode.BICUBIC)
        self.image_processor = self

    def get_font(self):
        truetype_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/SimHei.ttf'
        ff = urlopen(truetype_url)
        font = ImageFont.truetype(ff, size=40)
        return font

    def save_pretrained(self, *args, **kwargs):
        pass

    def padding_336(self, b, pad=336):
        width, height = b.size
        tar = int(np.ceil(height / pad) * pad)
        top_padding = 0 # int((tar - height)/2)
        bottom_padding = tar - height - top_padding
        left_padding = 0
        right_padding = 0
        b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])
        return b

    def Image_transform(self, img, hd_num=25):
        width, height = img.size
        trans = False
        if width < height:
            img = img.transpose(Image.TRANSPOSE)
            trans = True
            width, height = img.size
        ratio = (width/ height)
        scale = 1
        while scale*np.ceil(scale/ratio) <= hd_num:
            scale += 1
        scale -= 1
        scale = min(np.ceil(width / 560), scale)
        new_w = int(scale * 560)
        new_h = int(new_w / ratio)
        #print (scale, f'{height}/{new_h}, {width}/{new_w}')
        img = transforms.functional.resize(img, [new_h, new_w],)
        img = self.padding_336(img, 560)
        width, height = img.size
        if trans:
            img = img.transpose(Image.TRANSPOSE)
        return img

    def Video_transform(self, img, hd_num=25):
        width, height = img.size
        trans = False
        if width < height:
            img = img.transpose(Image.TRANSPOSE)
            trans = True
            width, height = img.size
        ratio = (width/ height)
        scale = 1
        new_h = int(scale * 560)
        new_w = int(new_h * ratio)
        #print (new_h, new_w)
        img = transforms.functional.resize(img, [new_h, new_w],)
        img = img.transpose(Image.TRANSPOSE)
        img = self.padding_336(img, 560)
        width, height = img.size
        if not trans:
            img = img.transpose(Image.TRANSPOSE)
        return img

    def frame2img(self, imgs, font):
        new_imgs = []
        for img in imgs:
            w, h = img.size
            scale = w/h
            if w > h:
                new_w = 560 * 2
                new_h = int(560 * 2 / scale)
            else:
                new_w = int(560 * 2 * scale)
                new_h = 560 * 2
            img = transforms.functional.resize(img, [new_h, new_w],)
            new_imgs.append(img)
        imgs = new_imgs
        new_w = 0
        new_h = 0
        pad = 40
        if w > h:
            for im in imgs:
                w,h = im.size
                new_w = max(new_w, w)
                new_h += h + 10 + pad
            new_img = Image.new('RGB', (new_w, new_h), 'white')
            draw = ImageDraw.Draw(new_img)
            curr_h = 0
            for idx, im in enumerate(imgs):
                w,h = im.size
                new_img.paste(im, (0, pad + curr_h))
                draw.text((0, curr_h ), f'<IMAGE {idx}>', font=font, fill='black')
                if idx + 1 < len(imgs):
                    draw.line([(0, pad +curr_h + h +5), (new_w, pad +curr_h + h +5)], fill = 'black', width=2)
                curr_h += h + 10 + pad
            #print (new_w, new_h)
        else:
            for im in imgs:
                w,h = im.size
                new_w += w + 10
                new_h = max(new_h, h)
            new_h += pad
            new_img = Image.new('RGB', (new_w, new_h), 'white')
            draw = ImageDraw.Draw(new_img)
            curr_w = 0
            for idx, im in enumerate(imgs):
                w,h = im.size
                new_img.paste(im, (curr_w, pad))
                draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
                if idx + 1 < len(imgs):
                    draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill = 'black', width=2)
                curr_w += w + 10
        return new_img

    def load_video(self, vid, num_frm=32, start=None, end=None):
        fps = vid.get_avg_fps()
        t_stride = int(round(float(fps) / int(1)))
        start_idx = 0 if start is None else start
        end_idx = len(vid) if end is None else end
        all_pos = list(range(start_idx, end_idx, t_stride))
        try:
            images = [vid[i].numpy() for i in all_pos]
        except:
            images = [vid[i].asnumpy() for i in all_pos]
        if len(images) > num_frm:
            num_frm = min(num_frm, len(images))
            step_size = len(images) / (num_frm + 1)
            indices = [int(i*step_size) for i in range(num_frm)]
            images = [images[i] for i in indices]
        images = [Image.fromarray(arr) for arr in images]
        return images

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(self, multimedia_objs, media_type, **kwargs):
        hd_num = 6 if len(multimedia_objs) > 1 else 24
        if media_type == MediaType.IMAGE:
            images_after = []
            for multimedia_obj in multimedia_objs:
                multimedia_obj = self.Image_transform(multimedia_obj, hd_num = hd_num)
                multimedia_obj = self.image_resize(multimedia_obj)
                images_after.append(self.image_transform(multimedia_obj))
            images_after = torch.stack(images_after)
            return {'images': images_after}
        elif media_type == MediaType.VIDEO:
            videos = []
            for multimedia_obj in multimedia_objs:
                # multimedia_obj = self.load_video(multimedia_obj)
                multimedia_obj = self.frame2img(multimedia_obj, self.font)
                multimedia_obj = self.Video_transform(multimedia_obj, hd_num = hd_num)
                videos.append(self.image_transform(multimedia_obj))
            videos = torch.stack(videos)
            return {'images': videos}
        else:
            raise Exception(f"Unknown media type: {media_type}")


class DeepseekVL2Processor(object):
    def __init__(self, processor):
        self.deepseek_vl2_processor = processor

        self.candidate_resolutions = processor.candidate_resolutions
        self.image_size = processor.candidate_resolutions[0][0]
        self.patch_size = processor.patch_size
        self.image_mean = processor.image_mean
        self.image_std = processor.image_std
        self.normalize = processor.normalize
        self.downsample_ratio = processor.downsample_ratio

        self.image_transform = self.build_transform(self.image_mean, self.image_std, self.normalize)
        self.image_processor = self

    def build_transform(self, mean, std, normalize):
        transform = [transforms.ToTensor()]
        if normalize:
            transform.append(transforms.Normalize(mean=mean, std=std))
        transform = transforms.Compose(transform)

        return transform

    def select_best_resolution(self, image_size, candidate_resolutions):
        # used for cropping
        original_width, original_height = image_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float("inf")

        for width, height in candidate_resolutions:
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
            effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (width, height)

        return best_fit

    def tokenize_with_image(self, images: List[Image.Image], cropping: bool = True):
        images_list, images_spatial_crop, num_image_tokens= [], [], []
        for image in images:
            """select best resolution for anyres"""
            if cropping:
                best_width, best_height = self.select_best_resolution(image.size, self.candidate_resolutions)
            else:
                best_width, best_height = self.image_size, self.image_size

            """process the global view"""
            global_view = ImageOps.pad(image, (self.image_size, self.image_size),
                                       color=tuple(int(x * 255) for x in self.image_mean))
            images_list.append(self.image_transform(global_view))

            """process the local views"""
            local_view = ImageOps.pad(image, (best_width, best_height),
                                      color=tuple(int(x * 255) for x in self.image_mean))
            for i in range(0, best_height, self.image_size):
                for j in range(0, best_width, self.image_size):
                    images_list.append(
                        self.image_transform(local_view.crop((j, i, j + self.image_size, i + self.image_size))))

            """record height / width crop num"""
            num_width_tiles, num_height_tiles = best_width // self.image_size, best_height // self.image_size
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            """add image tokens"""
            h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            # global views tokens h * (w + 1), 1 is for line seperator
            tokenized_image_length = h * (w + 1)
            # add a seperator between global and local views
            tokenized_image_length += 1
            # local views tokens, (num_height_tiles * h) * (num_width_tiles * w + 1)
            tokenized_image_length += (num_height_tiles * h) * (num_width_tiles * w + 1)

            num_image_tokens.append(tokenized_image_length)

        if len(images_list) == 0:
            images = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        else:
            images = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)

        return {
            "images": images,
            "images_spatial_crop": images_spatial_crop,
            "num_image_tokens": num_image_tokens
        }

    def save_pretrained(self, *args, **kwargs):
        self.deepseek_vl2_processor.save_pretrained(*args, **kwargs)

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs):
        return self.tokenize_with_image(images)
