from typing import Optional, List, Union, Tuple
import math
import functools

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import clone_hook

def dispatch_model(model_config):
    device_map = {}
    device_count = torch.cuda.device_count()
    num_layers = model_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (device_count - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * device_count
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    # place same device for language_model output and vit
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def forward_wrapper(forward_fn):
    @functools.wraps(forward_fn)
    def wrapper(*args, **kwargs):
        if 'inputs_embeds' in kwargs:
            kwargs.pop('inputs_embeds')
        return forward_fn(*args, **kwargs)

    return wrapper

def patch_internvl_chat_model(model):
    setattr(type(model), 'supports_gradient_checkpointing', True)
    embedding = model.get_input_embeddings()
    embedding.register_forward_hook(clone_hook)
    setattr(model, 'forward', forward_wrapper(getattr(model, "forward")))