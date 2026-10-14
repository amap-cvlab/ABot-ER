# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

@dataclass
class XcomposerOutputWithPast(CausalLMOutputWithPast):
    attention_mask: Optional[torch.Tensor] = None

def Xcomposer_forward(self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
) -> Union[Tuple, XcomposerOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    Returns:
    """

    if past_key_values is None and images is not None:
        inputs_embeds, attention_mask, labels, im_mask = self.input2emb(input_ids, images, labels)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        input_ids = None
    else:
        im_mask = None
    
    infer_mode = 'base'
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        im_mask=im_mask,
        infer_mode=infer_mode,
    )

    hidden_states = outputs[0]
    logits = self.output(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits, ) + outputs[1:]
        return (loss, ) + output if loss is not None else output

    return XcomposerOutputWithPast(
        loss=loss,
        logits=logits,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

    
def Xcomposer_prepare_inputs_for_generation(self,
        input_ids,
        images=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        im_mask=None,
        infer_mode='base',
        **kwargs,
):
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids, 'images': images}

    im_mask = im_mask

    model_inputs.update({
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'use_cache': kwargs.get('use_cache'),
        'attention_mask': attention_mask,
        'im_mask': im_mask,
        'infer_mode': infer_mode, 
    })
    return model_inputs


def _update_model_kwargs_for_generation(self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
) -> Dict[str, Any]:
    # update past_key_values keeping its naming used in model code
    cache_name, cache = self._extract_past_from_model_output(
        outputs, standardize_cache_format=standardize_cache_format
    )
    model_kwargs[cache_name] = cache
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            if getattr(outputs, "attention_mask", None) is not None:
                if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
                    past_length = model_kwargs["past_key_values"][0][0].shape[2]
                    if model_kwargs["attention_mask"].shape[1] != past_length:
                        model_kwargs["attention_mask"] = outputs.attention_mask
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    if (
        model_kwargs.get("use_cache", True)
        and "cache_position" in model_kwargs
        and model_kwargs["cache_position"] is not None
    ):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

    return model_kwargs


def input2emb(self, input_ids, images, labels):
    image_token_id = self.image_token_id
    bos_token_id = self.tokenizer.bos_token_id
    eos_token_id = self.tokenizer.eos_token_id
    pad_token_id = self.tokenizer.pad_token_id

    wrap_embeds_list, wrap_atts_list, wrap_labels_list, wrap_im_mask_list = [], [], [], []
    image_features = []
    for i in range(images.shape[0]):
        image = images[i][None]
        image = self.img2emb(image)[0]
        assert image.shape[0] == 1
        image_features.append(image[0])

    image_index = 0
    for idx in range(input_ids.shape[0]):
        wrap_embeds, wrap_atts, wrap_labels, wrap_im_mask = [], [], [], []
        input_id = input_ids[idx].tolist()
        input_start_index = input_id.index(bos_token_id)
        input_id = input_id[input_start_index:]
        if labels is not None:
            label = labels[idx].tolist()

        image_num = input_id.count(image_token_id)
        if image_num > 0:
            input_id = input_id[1:]
            if labels is not None:
                label = label[1:]
                
        input_id.append(eos_token_id)
        if labels is not None:
            label.append(eos_token_id)
        else:
            label = []

        pre_i, i = 0, 0
        for _ in range(image_num):
            i = input_id.index(image_token_id, pre_i)
            res_input_id = torch.tensor([bos_token_id] + input_id[pre_i:i], device=self.device)
            wrap_embeds.append(self.model.tok_embeddings(res_input_id))
            wrap_im_mask += [0] * len(res_input_id)
            if labels is not None:
                wrap_labels += [-100] + label[pre_i:i] 
            pre_i = i + 1

            if image_index < len(image_features):
                image = image_features[image_index]
                wrap_embeds.append(image)
                wrap_im_mask += [1] * image.shape[0]
                if labels is not None:
                    wrap_labels += [-100] * image.shape[0]
                image_index += 1
        res_input_id = torch.tensor([bos_token_id] + input_id[pre_i:-1], device=self.device)
        wrap_embeds.append(self.model.tok_embeddings(res_input_id))
        wrap_im_mask += [0] * len(res_input_id)
        if labels is not None:
            wrap_labels += [-100] + label[pre_i:-1]

        wrap_embeds = torch.concat(wrap_embeds, dim=0)
        wrap_im_mask = torch.tensor(wrap_im_mask, dtype=torch.bool, device=self.device)
        wrap_atts = torch.ones(wrap_embeds.shape[0], dtype=torch.int64, device=self.device)

        wrap_embeds = wrap_embeds[:self.max_length]
        wrap_atts = wrap_atts[:self.max_length]
        wrap_im_mask = wrap_im_mask[:self.max_length]

        wrap_embeds_list.append(wrap_embeds)
        wrap_atts_list.append(wrap_atts)
        wrap_im_mask_list.append(wrap_im_mask)

        if labels is not None:
            wrap_labels = torch.tensor(wrap_labels, device=self.device)
            wrap_labels = wrap_labels[:self.max_length]
            wrap_labels_list.append(wrap_labels)
        
    max_embeds_len = max(map(len, wrap_embeds_list))
    if labels is not None:
        max_labels_len = max(map(len, wrap_labels_list))
    else:        
        max_labels_len = 0
    max_len = max(max_embeds_len, max_labels_len)
    
    pad_embeds = self.model.tok_embeddings(torch.tensor(pad_token_id, device=self.device))
    wrap_embeds_list = [
        torch.concat((pad_embeds.repeat(max_len - wrap_embeds.shape[0], 1), wrap_embeds), dim=0)
        for wrap_embeds in wrap_embeds_list
    ]
    wrap_atts_list = [
        torch.concat((torch.zeros(max_len - wrap_atts.shape[0], dtype=torch.int64, device=self.device), wrap_atts), dim=0)
        for wrap_atts in wrap_atts_list
    ]
    wrap_im_mask_list = [
        torch.concat((torch.zeros(max_len - wrap_im_mask.shape[0], dtype=torch.bool, device=self.device), wrap_im_mask), dim=0)
        for wrap_im_mask in wrap_im_mask_list
    ]
    inputs_embeds = torch.stack(wrap_embeds_list, dim=0)
    attention_mask = torch.stack(wrap_atts_list, dim=0)
    im_mask = torch.stack(wrap_im_mask_list, dim=0)

    if labels is not None:
        wrap_labels_list = [                
            torch.concat((torch.zeros(max_len - wrap_labels.shape[0], dtype=torch.int64, device=self.device), wrap_labels), dim=0)
            for wrap_labels in wrap_labels_list
        ]
        labels = torch.stack(wrap_labels_list, dim=0)

    return inputs_embeds, attention_mask, labels, im_mask

def patch_internlm_xcomposer_forward_impl(model):
    setattr(type(model), 'forward', Xcomposer_forward)
    setattr(type(model), 'prepare_inputs_for_generation', Xcomposer_prepare_inputs_for_generation)
    setattr(type(model), 'input2emb', input2emb)
    setattr(type(model), '_update_model_kwargs_for_generation', _update_model_kwargs_for_generation)
