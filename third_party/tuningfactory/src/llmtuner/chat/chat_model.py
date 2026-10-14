from dataclasses import dataclass
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import GenerationConfig, HfArgumentParser, TextIteratorStreamer

from llmtuner.generate.dataloader import ChatApiDataloader
from llmtuner.generate.infer_args import ChatInferArguments

from ..generate.engine import CallMode, create_generate_engine
from ..hparams import FinetuningArguments, GeneratingArguments, ModelArguments


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..generate.data import OutputDataAfterEngine
    from ..generate.engine import HuggingFaceGenerateEngine, VllmGenerateEngine


@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]


class ChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        parser = HfArgumentParser((ModelArguments, FinetuningArguments, GeneratingArguments, ChatInferArguments))
        (
            model_args,
            finetuning_args,
            generating_args,
            infer_args,
        ) = parser.parse_args_into_dataclasses()
        self.can_generate = finetuning_args.stage == "sft"
        self.app_timeout = infer_args.timeout
        self.infer_engine: "HuggingFaceGenerateEngine|VllmGenerateEngine" = create_generate_engine(
            model_args, infer_args, finetuning_args, generating_args, CallMode.CHAT_API
        )
        self.infer_mode = infer_args.infer_mode
        self.model, self.tokenizer, self.processor, self.template = (
            self.infer_engine.model,
            self.infer_engine.tokenizer,
            self.infer_engine.processor,
            self.infer_engine.template,
        )
        self.dataloader = ChatApiDataloader(
            tokenizer=self.infer_engine.get_tokenizer(),
            template=self.infer_engine.get_template(),
            infer_args=infer_args,
            processor=self.infer_engine.get_processor(),
            model_dtype=self.infer_engine.get_model_dtype(),
        )

    def _process_args(
        self,
        messages: Sequence[Union[Dict[str, str], str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[List["NDArray"]] = [],
        **input_kwargs,
    ) -> Dict[str, Any]:
        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        num_return_sequences = input_kwargs.pop("num_return_sequences", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)
        stop = input_kwargs.pop("stop", None)
        raw_message_mode = input_kwargs.pop("raw_message_mode", False)

        generating_args = dict(
            do_sample=do_sample if do_sample is not None else self.infer_engine.gen_kwargs["do_sample"],
            temperature=temperature or self.infer_engine.gen_kwargs["temperature"],
            top_p=top_p or self.infer_engine.gen_kwargs["top_p"],
            top_k=top_k or self.infer_engine.gen_kwargs["top_k"],
            num_return_sequences=num_return_sequences or 1,
            repetition_penalty=repetition_penalty or self.infer_engine.gen_kwargs["repetition_penalty"],
        )

        if isinstance(num_return_sequences, int) and num_return_sequences > 1:
            generating_args["do_sample"] = True

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens
        if stop and "vllm" in self.infer_mode:
            # huggingface not support stop?
            generating_args["stop"] = stop
        # 每次调用，更新引擎的generate 配置。
        self.infer_engine.gen_kwargs.update(generating_args)
        if raw_message_mode:
            preprocess_result = {}
            # add_special_tokens 不确定这个特殊影响
            # 这里需要放到list中，与dataloader.prompts_preprocess 返回保持一致。
            preprocess_result["input_ids"] = [self.dataloader.tokenizer.encode(
                messages, add_special_tokens=False, padding=True, truncation=True
            )]
            preprocess_result["samples"] = {"fake_prompt_column": [messages]}
            return preprocess_result
        else:
            return self.dataloader.prompts_preprocess(
                prompts=messages, images=images, system_prompts=system, videos=None, tools=tools
            )

    @torch.inference_mode()
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[List["NDArray"]] = [],
        **input_kwargs,
    ) -> List[Response]:
        if not self.can_generate:
            raise ValueError("The current model does not support `chat`.")

        sample = self._process_args(messages, system, tools, images, **input_kwargs)
        prompt_length = len(sample["input_ids"][0])
        outputs: "OutputDataAfterEngine" = self.infer_engine.predict_batch(sample)
        response = outputs.predict
        results = []

        for i in range(len(response)):
            results.append(
                Response(
                    response_text=response[i],
                    response_length=int(outputs.predict_response_lengths[i]),
                    prompt_length=prompt_length,
                    finish_reason=outputs.predict_finish_reasons[i],
                )
            )

        return results

    @torch.inference_mode()
    def stream_chat(
        self,
        messages: Sequence[Union[Dict[str, str], str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[List["NDArray"]] = [],
        **input_kwargs,
    ) -> Generator[str, None, None]:
        if not self.can_generate:
            raise ValueError("The current model does not support `stream_chat`.")

        gen_kwargs, _ = self._process_args(messages, system, tools, images, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        yield from streamer

    @torch.inference_mode()
    def get_scores(self, batch_input: List[str], **input_kwargs) -> List[float]:
        if self.can_generate:
            raise ValueError("Cannot get scores using an auto-regressive model.")

        max_length = input_kwargs.pop("max_length", None)
        device = getattr(self.model.pretrained_model, "device", "cuda")
        inputs = self.tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            max_length=max_length or getattr(self.model.config, "max_position_embeddings", 1024),
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        input_ids: torch.Tensor = inputs["input_ids"]
        _, _, values = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if getattr(self.model.config, "model_type", None) == "chatglm":
            values = torch.transpose(values, 0, 1)

        scores = []
        for i in range(input_ids.size(0)):
            end_indexes = (input_ids[i] != self.tokenizer.pad_token_id).nonzero()
            end_index = end_indexes[-1].item() if len(end_indexes) else 0
            scores.append(values[i, end_index].nan_to_num().item())

        return scores

    def match_template(self, message: str, system: Optional[str] = None, tools: Optional[str] = None) -> bool:
        template = self.template
        tokenizer = self.tokenizer
        # compare message with a dummy templated message to discriminate whether
        # the message has been templated
        # construct a dummy templated message according to template._encode
        system = system or template.default_system
        elements = []
        elements += template.format_prefix.apply()
        if system or tools:
            tool_text = template.format_tools.apply(content=tools)[0] if tools else ""
            elements += template.format_system.apply(content=(system + tool_text))
        dummy_content = self.model.__class__.__name__
        elements += template.format_user.apply(content=dummy_content, idx=str(0))
        # refer to template._convert_elements_to_ids but use str
        for i, elem in enumerate(elements):
            if isinstance(elem, dict):
                elements[i] = elem.get("token")
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token is not None:
                    elements[i] = tokenizer.bos_token
                elif "eos_token" in elem and tokenizer.eos_token is not None:
                    elements[i] = tokenizer.eos_token
            else:
                assert isinstance(elem, str), "Input must be string, set[str] or dict[str, str], got {}".format(
                    type(elem)
                )
        dummy_message = "".join(elements)
        # longest common prefix
        for index, (char_a, char_b) in enumerate(zip(message, dummy_message)):
            if char_a != char_b:
                break
        return dummy_message[index:].startswith(dummy_content)

    @torch.inference_mode()
    def get_token_candidates(
        self,
        message: str,
        system: Optional[str] = None,
        **input_kwargs,
    ) -> Tuple[List[List], List[Dict]]:
        if not self.can_generate:
            raise ValueError("The current model does not support `debug`.")
        if self.match_template(message, system):  # formated message
            # does tokenizer encoded prompt consistent with template formated
            prompt = self.tokenizer(message)["input_ids"]
        else:
            paired_messages = [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            prompt, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, system=system)
        prompt_length = len(prompt)
        input_ids = torch.tensor([prompt], device=self.model.device)
        # use greedy search and top_k is not for sampling thus pop it out
        top_k = input_kwargs.pop("top_k", 5)
        max_new_tokens = input_kwargs.pop("max_new_tokens", 128)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1,
            top_k=1,
            output_scores=True,
            early_stopping=True,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        outputs = self.model.generate(input_ids, generation_config)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        # every step topk candidate (token + prob)
        logits = self.model(outputs.sequences).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_indices = probs.topk(top_k, dim=-1).indices
        # on CPU from now on
        outputs.sequences = outputs.sequences.to("cpu")
        transition_scores = transition_scores.to("cpu")
        probs = probs.to("cpu")
        top_indices = top_indices.to("cpu")
        input_tokens_with_prob = [
            {
                "id": outputs.sequences[0][0].item(),
                "text": self.tokenizer.decode(outputs.sequences[0][0].item()),
                "probability": float(1),
            }
        ]
        next_tokens_with_prob = []
        for i in range(0, len(outputs.sequences[0]) - 1):
            token_id = outputs.sequences[0][i + 1]
            token_str = self.tokenizer.decode(token_id)
            token_prob = probs[0][i][token_id]
            temp = {
                "id": int(token_id.item()),
                "text": token_str,
                "probability": float(token_prob),
            }
            if i < prompt_length - 1:
                input_tokens_with_prob.append(temp)

            next_token_sub = []
            next_token_sub.append(temp)  # put current token in first
            for j in top_indices[0][i]:
                if len(next_token_sub) >= top_k:
                    break
                if j == token_id:
                    continue
                temp = {
                    "id": int(j),
                    "text": self.tokenizer.decode(j),
                    "probability": probs[0][i][j].cpu().item(),
                }
                next_token_sub.append(temp)
            next_tokens_with_prob.append(next_token_sub)

        generated_tokens = outputs.sequences[:, prompt_length:]
        generated_tokens_with_prob = [
            {
                "id": int(token.cpu()),
                "text": self.tokenizer.decode(token),
                "probability": float(np.exp(score.cpu().numpy())),
            }
            for token, score in zip(generated_tokens[0], transition_scores[0])
        ]
        # every step topk candidate (token + prob), greedy search result with probs
        return next_tokens_with_prob, input_tokens_with_prob + generated_tokens_with_prob

    @torch.inference_mode()
    def get_sequence_candidates(
        self,
        message: str,
        system: Optional[str] = None,
        **input_kwargs,
    ) -> Tuple[List[List], List[Dict]]:
        if not self.can_generate:
            raise ValueError("The current model does not support `debug`.")
        if self.match_template(message, system):  # formated message
            # does tokenizer encoded prompt consistent with template formated
            prompt = self.tokenizer(message)["input_ids"]
        else:
            paired_messages = [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            prompt, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, system=system)
        input_ids = torch.tensor([prompt], device=self.model.device)
        # use beam search
        num_beams = input_kwargs.pop("num_beams", 4)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=False,
            early_stopping=True,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        outputs = self.model.generate(input_ids, generation_config)
        end_indice = (outputs.sequences != generation_config.pad_token_id).cumsum(-1).argmax(-1) + 1
        seq_candidates = []
        for i, sequence in enumerate(outputs.sequences):
            input_ids = sequence[: end_indice[i]].unsqueeze(0)
            logits = self.model(input_ids).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output_tokens_with_prob = [
                {
                    "id": input_ids[0][0].item(),
                    "text": self.tokenizer.decode(input_ids[0][0].item()),
                    "probability": float(1),
                }
            ]

            for i in range(0, len(input_ids[0]) - 1):
                token_id = input_ids[0][i + 1]
                token_prob = probs[0][i][token_id]
                token_str = self.tokenizer.decode(token_id)
                temp = {
                    "id": int(token_id.item()),
                    "text": token_str,
                    "probability": float(token_prob),
                }
                output_tokens_with_prob.append(temp)
            seq_candidates.append(output_tokens_with_prob)

        return seq_candidates
