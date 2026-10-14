# Copyright 2024 the LlamaFactory team.
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

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING
import os
import json
import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer
    from ...hparams import ModelArguments


logger = get_logger(__name__)


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""
    Resize token embeddings.
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
            _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

        logger.info("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))


def configure_embedding_model(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    embedding_config_dir = os.path.join(model_args.model_name_or_path, 'config_sentence_transformers.json')
    if os.path.isfile(embedding_config_dir):
        setattr(config, "is_embedding_model", True)
        model_args.resize_vocab = False
        pooling_config_file = os.path.join(model_args.model_name_or_path, '1_Pooling', 'config.json')
        assert os.path.isfile(pooling_config_file), 'Pooling file not exist, do not know how to pooling'
        with open(pooling_config_file, 'r', encoding='utf-8') as pooling_file:
            pooling_config = json.load(pooling_file)
        # pooling方法来源：https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
        if pooling_config.get('pooling_mode_cls_token', False):
            emb_pooling_method = 'CLS'
        elif pooling_config.get('pooling_mode_lasttoken', False):
            emb_pooling_method = 'LAST'
        else:
            raise Exception(f'Not implemented pooling method, please contact to MDL for more support!')
        setattr(config, "emb_pooling_method", emb_pooling_method)