# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# patch this method for generate kwargs conflict
import os
import shutil
import torch

audio_generation_counter = 0

def decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
    outputs = self.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        **kwargs,
    )
    return outputs

def generate(
    self,
    input_ids=None,
    pixel_values=None,
    tgt_sizes=None,
    audio_features=None,
    audio_feature_lens=None,
    image_bound=None,
    audio_bounds=None,
    spk_bounds=None,
    attention_mask=None,
    tokenizer=None,
    vision_hidden_states=None,
    stream=False,
    audio_output_path=None,
    **kwargs,
):
    assert input_ids is not None
    assert len(input_ids) == len(pixel_values)

    model_inputs = {
        "input_ids": input_ids,
        "audio_features": audio_features,
        "audio_feature_lens": audio_feature_lens,
        "image_bound": image_bound,
        "audio_bounds": audio_bounds,
        "spk_bounds": spk_bounds,
    }

    if vision_hidden_states is None:
        model_inputs["pixel_values"] = pixel_values
        model_inputs["tgt_sizes"] = tgt_sizes
    else:
        model_inputs["vision_hidden_states"] = vision_hidden_states

    with torch.inference_mode():
        model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
        model_inputs["inputs_embeds"] = self.get_omni_embedding(
            model_inputs,
            input_embeddings=model_inputs["inputs_embeds"],
            chunk_length=self.config.audio_chunk_length,
        )

        if stream:
            outputs = {}
        else:
            outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)

    if audio_output_path is None:
        return outputs

    global audio_generation_counter
    audio_output_dir = os.path.dirname(audio_output_path)
    tmp_output_dir = "/tmp/tts_gen/"
    if not os.path.exists(tmp_output_dir):
        os.makedirs(tmp_output_dir, exist_ok=True)
    audio_output_path = os.path.join(audio_output_dir, f"{audio_generation_counter}.wav")
    tmp_output_path = os.path.join(tmp_output_dir, f"{audio_generation_counter}.wav")
    audio_generation_counter += 1

    tts_inputs = {"input_ids": input_ids, "spk_bounds": spk_bounds}
    tts_text = self._decode_text(outputs.sequences, tokenizer)
    if isinstance(tts_text, list):
        tts_text = tts_text[0]
    mel_spec = self._generate_mel_spec(tts_inputs, outputs, tts_text)
    wav_numpy, sr = self.decode_mel_to_audio(mel_spec, tmp_output_path)
    shutil.copy(tmp_output_path, audio_output_path)
    outputs.multimodal_output_path = [os.path.abspath(audio_output_path)]

    return outputs

def patch_minicpmo_generate_impl(model):
    setattr(type(model), '_decode', decode)
    setattr(type(model), 'generate', generate)
