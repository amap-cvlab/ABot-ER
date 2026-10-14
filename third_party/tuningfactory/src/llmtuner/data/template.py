import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from ..extras.constants import IGNORE_INDEX
from ..extras.logging import get_logger
from .formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from .mm_plugin import get_mm_plugin
from .utils import Role


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import ProcessorArguments
    from .formatter import SLOTS, Formatter
    from .mm_plugin import BasePlugin


logger = get_logger(__name__)


@dataclass
class Template:
    name: str
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: List[str]
    thought_words: tuple[str, str]
    image_token: str
    video_token: str
    audio_token: str
    efficient_eos: bool
    replace_eos: bool
    enable_thinking: Optional[bool]
    special_tokens: List[str]

    tokenizer: Optional["PreTrainedTokenizer"] = None
    processor: Optional["ProcessorMixin"] = None
    plugin: Optional["BasePlugin"] = None

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        r"""
        Extracts tool message.
        """
        return self.format_tools.extract(content)

    def source_mask(
        self,
        tokenizer: "PreTrainedTokenizer",
        source_ids: List[int],
        turn_idx: int,
    ):
        if turn_idx != 0 and self.efficient_eos:
            source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
        else:
            source_mask = [IGNORE_INDEX] * len(source_ids)
        return source_mask

    def _format_user(self, content: str, idx: str):
        return self.format_user.apply(content=content, idx=idx)

    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}\n\n{self.thought_words[1]}\n\n" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self._format_user(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids


@dataclass
class Llama2Template(Template):
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: str,
        tools: str,
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


@dataclass
class Internvl3Template(Template):
    def change_to_vllm(self):
        self.image_token = "<image>"
        self.video_token = "<video>"


@dataclass
class Qwen2vlTemplate(Template):
    def change_to_vllm(self):
        self.image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        self.video_token = "<|vision_start|><|video_pad|><|vision_end|>"


@dataclass
class Qwen2_5OmniTemplate(Template):
    def change_to_vllm(self):
        self.image_token = "<|vision_bos|><|IMAGE|><|vision_eos|>"
        self.video_token = "<|vision_bos|><|VIDEO|><|vision_eos|>"


@dataclass
class Qwen3_5Template(Template):
    def change_to_vllm(self):
        self.image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        self.video_token = "<|vision_start|><|video_pad|><|vision_end|>"
        self.encode_oneturn = self.encode_oneturn_with_think_in_end

    def encode_oneturn(self, *args, **kwargs):
        return super().encode_oneturn(*args, **kwargs)

    def encode_oneturn_with_think_in_end(self, *args, **kwargs):
        prompt_ids, _ = super().encode_oneturn(*args, **kwargs)
        prompt_ids += self.get_thought_word_ids(args[0])
        return prompt_ids, _

@dataclass
class ReasoningTemplate(Template):
    r"""A template that add thought to assistant message."""

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(1, len(messages) - 2, 2):
            messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0] not in messages[-1]["content"]
            and self.thought_words[1] not in messages[-1]["content"]
        ):
            if not self.enable_thinking:  # do not compute loss
                prompt_ids = prompt_ids + self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools)
        for i in range(0, len(messages), 2):
            if (
                self.thought_words[0] not in messages[i + 1]["content"]
                and self.thought_words[1] not in messages[i + 1]["content"]
            ):
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


@dataclass
class Qwen2vlReasoningTemplate(ReasoningTemplate):
    def change_to_vllm(self):
        self.image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        self.video_token = "<|vision_start|><|video_pad|><|vision_end|>"


@dataclass
class Qwen3_5ReasoningTemplate(ReasoningTemplate):
    def change_to_vllm(self):
        self.image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        self.video_token = "<|vision_start|><|video_pad|><|vision_end|>"
        self.encode_oneturn = self.encode_oneturn_with_think_in_end

    def encode_oneturn(self, *args, **kwargs):
        return super().encode_oneturn(*args, **kwargs)

    def encode_oneturn_with_think_in_end(self, *args, **kwargs):
        prompt_ids, _ = super().encode_oneturn(*args, **kwargs)
        think_str = f"{self.thought_words[0]}\n"
        think_ids = args[0].encode(think_str, add_special_tokens=False)
        prompt_ids += think_ids
        return prompt_ids, _



TEMPLATES: Dict[str, Template] = {}


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Sequence[str] = [],
    thought_words: Optional[tuple[str, str]] = None,
    image_token: str = "<image>",
    audio_token: str = "<audio>",
    video_token: str = "<video>",
    efficient_eos: bool = False,
    replace_eos: bool = False,
    enable_thinking: Optional[bool] = True,
    special_tokens: Optional[List[str]] = [],
    template_class: type["Template"] = Template,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=eos_slots, tool_format="default")
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        name=name,
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words,
        thought_words=thought_words or ("<think>", "</think>"),
        image_token=image_token,
        video_token=video_token,
        audio_token=audio_token,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        enable_thinking=enable_thinking,
        special_tokens=special_tokens,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def _jinja_escape(content: str) -> str:
    return content.replace("'", r"\'")


def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            slot_pieces = slot.split("{{content}}")
            if slot_pieces[0]:
                slot_items.append("'" + _jinja_escape(slot_pieces[0]) + "'")
            if len(slot_pieces) > 1:
                slot_items.append(placeholder)
                if slot_pieces[1]:
                    slot_items.append("'" + _jinja_escape(slot_pieces[1]) + "'")
        elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
            if "bos_token" in slot and tokenizer.bos_token_id is not None:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError("Dict is not supported.")

    return " + ".join(slot_items)


def _get_jinja_template(template: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    jinja_template = ""

    prefix = _convert_slots_to_jinja(template.format_prefix.apply(), tokenizer)
    if prefix:
        jinja_template += "{{ " + prefix + " }}"

    if template.default_system:
        jinja_template += "{% set system_message = '" + _jinja_escape(template.default_system) + "' %}"

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    )

    system_message = _convert_slots_to_jinja(template.format_system.apply(), tokenizer, placeholder="system_message")
    if not isinstance(template, Llama2Template):
        jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"

    jinja_template += "{% for message in messages %}"
    jinja_template += "{% set content = message['content'] %}"
    if isinstance(template, Llama2Template):
        jinja_template += "{% if loop.index0 == 0 and system_message is defined %}"
        jinja_template += "{% set content = " + system_message + " + message['content'] %}"
        jinja_template += "{% endif %}"

    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_slots_to_jinja(template.format_user.apply(), tokenizer)
    jinja_template += "{{ " + user_message + " }}"

    jinja_template += "{% elif message['role'] == 'assistant' %}"
    assistant_message = _convert_slots_to_jinja(
        template.format_assistant.apply() + template.format_separator.apply(), tokenizer
    )
    jinja_template += "{{ " + assistant_message + " }}"
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    return jinja_template


def get_template_and_fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: str = None,
    enable_thinking: bool = True,
    tool_format: Optional[str] = None,
    processor: Optional["ProcessorMixin"] = None,
    processor_args: Optional["ProcessorArguments"] = None,
    vllm_mode=None,
) -> Template:
    if name is None:
        template = TEMPLATES["empty"]  # placeholder
    else:
        template = TEMPLATES.get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))

    if tool_format is not None:
        logger.info("Using tool format: {}.".format(tool_format))
        eos_slots = [] if template.efficient_eos else [{"eos_token"}]
        template.format_tools = ToolFormatter(tool_format=tool_format)
        template.format_function = FunctionFormatter(slots=eos_slots, tool_format=tool_format)

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")
        pre_eos_token = tokenizer.eos_token
        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]
        if pre_eos_token is not None and pre_eos_token not in template.stop_words:
            template.stop_words.append(pre_eos_token)

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    if template.special_tokens:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=template.special_tokens),
            replace_additional_special_tokens=False,
        )

    try:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = _get_jinja_template(template, tokenizer)
    except ValueError as e:
        logger.info(f"Cannot add this chat template to tokenizer. Error: {e}")

    template.enable_thinking = enable_thinking
    template.tokenizer = tokenizer
    template.processor = processor
    if vllm_mode and hasattr(template, "change_to_vllm"):
        template.change_to_vllm()
    template.plugin = get_mm_plugin(name=name, tokenizer=tokenizer, processor=processor, processor_args=processor_args)
    return template


_register_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    ),
)


_register_template(
    name="aquila",
    format_user=StringFormatter(slots=["Human: {{content}}###Assistant:"]),
    format_separator=EmptyFormatter(slots=["###"]),
    default_system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    stop_words=["</s>"],
    efficient_eos=True,
)


_register_template(
    name="atom",
    format_user=StringFormatter(
        slots=[{"bos_token"}, "Human: {{content}}\n", {"eos_token"}, {"bos_token"}, "Assistant:"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}\n", {"eos_token"}]),
)


_register_template(
    name="baichuan",
    format_user=StringFormatter(slots=[{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}]),
    efficient_eos=True,
)


_register_template(
    name="baichuan2",
    format_user=StringFormatter(slots=["<reserved_106>{{content}}<reserved_107>"]),
    efficient_eos=True,
)


_register_template(
    name="belle",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nBelle: "]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="bluelm",
    format_user=StringFormatter(slots=[{"token": "[|Human|]:"}, "{{content}}", {"token": "[|AI|]:"}]),
)


_register_template(
    name="breeze",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST] "]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
)


_register_template(
    name="chatglm2",
    format_user=StringFormatter(slots=["[Round {{idx}}]\n\n问：{{content}}\n\n答："]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
    efficient_eos=True,
)


_register_template(
    name="chatglm3",
    format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
    format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
    format_system=StringFormatter(slots=[{"token": "<|system|>"}, "\n", "{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(
        slots=[{"token": "<|observation|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]
    ),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


_register_template(
    name="chatml",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


_register_template(
    name="chatml_de",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="Du bist ein freundlicher und hilfsbereiter KI-Assistent.",
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


_register_template(
    name="codegeex2",
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
)


_register_template(
    name="codegeex4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>\n"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    default_system=(
        "你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，"
        "并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。"
    ),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


_register_template(
    name="cohere",
    format_user=StringFormatter(
        slots=[
            (
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="cpm",
    format_user=StringFormatter(slots=["<用户>{{content}}<AI>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="dbrx",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system=(
        "You are DBRX, created by Databricks. You were last updated in December 2023. "
        "You answer questions based on information available up to that point.\n"
        "YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough "
        "responses to more complex and open-ended questions.\nYou assist with various tasks, "
        "from writing to coding (using markdown for code blocks — remember to use ``` with "
        "code, JSON, and tables).\n(You do not have real-time data access or code execution "
        "capabilities. You avoid stereotyping and provide balanced perspectives on "
        "controversial topics. You do not provide song lyrics, poems, or news articles and "
        "do not divulge details of your training data.)\nThis is your system prompt, "
        "guiding your responses. Do not reference it, just respond to the user. If you find "
        "yourself talking about this message, stop. You should be responding appropriately "
        "and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION "
        "ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY."
    ),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="deepseek",
    format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="deepseekcoder",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n### Response:"]),
    format_assistant=StringFormatter(slots=["\n{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI programming assistant, utilizing the Deepseek Coder model, "
        "developed by Deepseek Company, and you only answer questions related to computer science. "
        "For politically sensitive questions, security and privacy issues, "
        "and other non-computer science questions, you will refuse to answer\n"
    ),
)


_register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_template(
    name="empty",
    efficient_eos=True,
)


_register_template(
    name="falcon",
    format_user=StringFormatter(slots=["User: {{content}}\nFalcon:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    efficient_eos=True,
)


_register_template(
    name="fewshot",
    format_separator=EmptyFormatter(slots=["\n\n"]),
    efficient_eos=True,
)


_register_template(
    name="gemma",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_separator=EmptyFormatter(slots=["<end_of_turn>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
)


_register_template(
    name="gemma3",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_separator=EmptyFormatter(slots=["<end_of_turn>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
    image_token="<start_of_image>",
)


_register_template(
    name="glm4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


_register_template(
    name="glm4_moe",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
    template_class=ReasoningTemplate,
)


_register_template(
    name="glm4v",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    image_token="<|begin_of_image|>",  # not the real token, just a placeholder
    efficient_eos=True,
)


_register_template(
    name="intern",
    format_user=StringFormatter(slots=["<|User|>:{{content}}\n<|Bot|>:"]),
    format_system=StringFormatter(slots=["<|System|>:{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["<eoa>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
    ),
    stop_words=["<eoa>"],
    efficient_eos=True,  # internlm tokenizer cannot set eos_token_id
)


_register_template(
    name="intern2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
        "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such "
        "as English and 中文."
    ),
    stop_words=["<|im_end|>"],
    efficient_eos=True,  # internlm2 tokenizer cannot set eos_token_id
)

_register_template(
    name="intern3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
    ),
    stop_words=["<|im_end|>"],
    efficient_eos=True,  # internlm2 tokenizer cannot set eos_token_id
)

_register_template(
    name="internlm_xcomposer",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n"
        "- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model "
        "that is developed by Shanghai AI Laboratory (上海人工智能实验室). "
        "It is designed to be helpful, honest, and harmless.\n"
        "- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen "
        "by the user such as English and 中文.\n"
        "- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively "
        "based on the provided image."
    ),
    stop_words=["<|im_end|>"],
    efficient_eos=True,  # internlm2 tokenizer cannot set eos_token_id
)


_register_template(
    name="internvl1",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|><|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>"]),
    format_separator=EmptyFormatter(slots=["<|im_end|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=("You are an AI assistant whose name is InternLM (书生·浦语)."),
    stop_words=["<|im_end|>"],
    efficient_eos=True,  # internlm2 tokenizer cannot set eos_token_id
    image_token="<IMG_CONTEXT>",
    video_token="<IMG_CONTEXT>",
)

_register_template(
    name="internvl2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|><|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>"]),
    format_assistant=StringFormatter(slots=["{{content}}", "<|im_end|>"]),
    format_separator=EmptyFormatter(slots=["<|im_end|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"
    ),
    stop_words=["<|im_end|>"],
    image_token="<IMG_CONTEXT>",
    video_token="<IMG_CONTEXT>",
    template_class=Internvl3Template,
)

_register_template(
    name="internvl3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}", "<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["<|im_end|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    ),
    stop_words=["<|im_end|>"],
    image_token="<IMG_CONTEXT>",
    video_token="<IMG_CONTEXT>",
    template_class=Internvl3Template,
)

_register_template(
    name="internvl2_Phi_3",
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>"]),
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|><|assistant|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"
    ),
    stop_words=["<|endoftext|>", "<|end|>", "</s>"],
    replace_eos=True,
    image_token="<IMG_CONTEXT>",
    video_token="<IMG_CONTEXT>",
)


_register_template(
    name="llama2",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    template_class=Llama2Template,
)


_register_template(
    name="llama2_zh",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
    template_class=Llama2Template,
)


_register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_function=FunctionFormatter(slots=["{{content}}", "<|eot_id|>"], tool_format="llama3"),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_template(
    name="llama3_v",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_function=FunctionFormatter(slots=["{{content}}", "<|eot_id|>"], tool_format="llama3"),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
    image_token="<|image|>",
)


_register_template(
    name="mistral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="olmo",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"eos_token"}]),
)


_register_template(
    name="openchat",
    format_user=StringFormatter(slots=["GPT4 Correct User: {{content}}", {"eos_token"}, "GPT4 Correct Assistant:"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="openchat-3.6",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)


_register_template(
    name="orion",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: ", {"eos_token"}]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="phi",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


_register_template(
    name="phi4",
    format_user=StringFormatter(
        slots=["<|im_start|>user<|im_sep|>{{content}}<|im_end|><|im_start|>assistant<|im_sep|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_start|>system<|im_sep|>{{content}}<|im_end|>"]),
    stop_words=["<|im_end|>"],
)


_register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="qwen2_5",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_function=FunctionFormatter(slots=["{{content}}", "<|im_end|>"], tool_format="qwen2_5"),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="qwen3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen2_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)

_register_template(
    name="qwen3_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen2_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)

# copied from qwen template
_register_template(
    name="qwen3_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen2_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    template_class=Qwen2vlReasoningTemplate,
)

_register_template(
    name="qwen3_omni",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    default_system="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen2_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    audio_token="<|audio_pad|>",
    template_class=ReasoningTemplate,
)

_register_template(
    name="qwen3_omni_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    default_system="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen2_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    audio_token="<|audio_pad|>",
)

# copied from qwen template
_register_template(
    name="qwen3_vl_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen2_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    template_class=Qwen2vlTemplate,
)


_register_template(
    name="qwen3_5",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen3_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen3_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    template_class=Qwen3_5ReasoningTemplate,
)


_register_template(
    name="qwen3_5_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen3_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen3_5"),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    template_class=Qwen3_5Template,
)


_register_template(
    name="qwen2_5_math_cot",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="Please reason step by step, and put your final answer within \\boxed{}.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="qwen2_5_math_tir",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="qwen_o1_preview",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="solar",
    format_user=StringFormatter(slots=["### User:\n{{content}}\n\n### Assistant:\n"]),
    format_system=StringFormatter(slots=["### System:\n{{content}}\n\n"]),
    efficient_eos=True,
)


_register_template(
    name="starchat",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


_register_template(
    name="telechat",
    format_user=StringFormatter(slots=["<_user>{{content}}<_bot>"]),
    format_system=StringFormatter(slots=["<_system>{{content}}<_end>"]),
    stop_words=["<_end>"],
    replace_eos=True,
)


r"""
Legacy model chat-template support.
"""
_register_template(
    name="turing",
    format_system=StringFormatter(slots=["\n\nSystem: {{content}}"]),
    format_user=StringFormatter(slots=["\n\nHuman: {{content}}\n\nAssistant:"]),
    format_assistant=StringFormatter(slots=[" {{content}}", {"eos_token"}]),  # a blank before answer
)


_register_template(
    name="tbstars",
    format_user=StringFormatter(slots=["\n\nHuman: {{content}}\n\nAssistant: "]),
    special_tokens=[
        "<pad>",
        "<|beginoftext|>",
        "<|endoftext|>",
        "<unk>",
        "<mask>",
        "\n\nSystem: ",
        "\n\nHuman: ",
        "\n\nAssistant: ",
        "\n\nFunction: ",
    ],
)


_register_template(
    name="tbstars008vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


_register_template(
    name="turing_v1",
    format_user=StringFormatter(slots=["\n\nHuman: {{content}}\n\nAssistant: "]),
)


_register_template(
    name="tbstars2",
    format_user=StringFormatter(
        slots=[
            {"token": "<|im_start|>"},
            "user\n{{content}}",
            {"token": "<|im_end|>"},
            "\n",
            {"token": "<|im_start|>"},
            "assistant\n",
        ]
    ),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    special_tokens=["<pad>", "<|beginoftext|>", "<|endoftext|>", "<unk>", "<mask>", "<|im_start|>", "<|im_end|>"],
)


_register_template(
    name="tbstars2_moe_vista",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="tbstars_omni",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    audio_token="<|audio_pad|>",
)


_register_template(
    name="tbstars2_omni",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    audio_token="<|audio_pad|>",
)


_register_template(
    name="tbstars2_5_omni",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    audio_token="<|audio_pad|>",
)


_register_template(
    name="tbstars2_omni_plain",
    format_user=StringFormatter(slots=["{{content}}"]),
    format_separator=EmptyFormatter(slots=[""]),
    stop_words=["<|im_end|>"],
    replace_eos=False,
    image_token="<|image_pad|>",
    audio_token="<|audio_pad|>",
)


_register_template(
    name="vicuna",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
)


_register_template(
    name="vicuna-next",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
)


_register_template(
    name="xuanyuan",
    format_user=StringFormatter(slots=["Human: {{content}} Assistant:"]),
    default_system=(
        "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，"
        "会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、"
        "不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
    ),
)


_register_template(
    name="xverse",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: "]),
)


_register_template(
    name="yayi",
    format_user=StringFormatter(slots=[{"token": "<|Human|>"}, ":\n{{content}}\n\n", {"token": "<|YaYi|>"}, ":"]),
    format_system=StringFormatter(slots=[{"token": "<|System|>"}, ":\n{{content}}\n\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "You are a helpful, respectful and honest assistant named YaYi "
        "developed by Beijing Wenge Technology Co.,Ltd. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    stop_words=["<|End|>"],
)


_register_template(
    name="yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="yi_vl",
    format_user=StringFormatter(slots=["### Human: {{content}}\n### Assistant:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system=(
        "This is a chat between an inquisitive human and an AI assistant. "
        "Assume the role of the AI assistant. Read all the images carefully, "
        "and respond to the human's questions with informative, helpful, detailed and polite answers. "
        "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。"
        "仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
    ),
    stop_words=["###"],
    efficient_eos=True,
)


_register_template(
    name="yuan",
    format_user=StringFormatter(slots=["{{content}}", {"token": "<sep>"}]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<eod>"],
    replace_eos=True,
)


_register_template(
    name="zephyr",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}", {"eos_token"}, "<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}", {"eos_token"}]),
    default_system="You are Zephyr, a helpful assistant.",
)


_register_template(
    name="ziya",
    format_user=StringFormatter(slots=["<human>:{{content}}\n<bot>:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_template(
    name="deepseek-vl2",
    format_prefix=EmptyFormatter(slots=["<｜begin▁of▁sentence｜>"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_user=StringFormatter(slots=["<|User|>: {{content}}\n\n<|Assistant|>:"]),
    image_token="<image>",
)


_register_template(
    name="deepseek-v3",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


_register_template(
    name="deepseek-r1",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=ReasoningTemplate,
)


# same as "llava-hf" except for default_system
_register_template(
    name="vip-llava-hf",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
)


_register_template(
    name="llava-next-yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    default_system="Answer the questions.",
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


# supports: https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf
_register_template(
    name="llava_onevision_qwen2",
    format_user=StringFormatter(
        slots=[
            {"token": "<|im_start|>"},
            "user\n{{content}}",
            {"token": "<|im_end|>"},
            "\n",
            {"token": "<|im_start|>"},
            "assistant\n",
        ]
    ),
    format_system=StringFormatter(
        slots=[{"token": "<|im_start|>"}, "system\n{{content}}", {"token": "<|im_end|>"}, "\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<image>",
    video_token="<video>",
)


# supports: https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf
_register_template(
    name="llava_next_video",
    format_prefix=EmptyFormatter(slots=[{"<s>"}]),
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    image_token="<image>",
    video_token="<video>",
)


_register_template(
    name="qwen2-vl",
    format_user=StringFormatter(
        slots=[
            {"token": "<|im_start|>"},
            "user\n{{content}}",
            {"token": "<|im_end|>"},
            "\n",
            {"token": "<|im_start|>"},
            "assistant\n",
        ]
    ),
    format_system=StringFormatter(
        slots=[{"token": "<|im_start|>"}, "system\n{{content}}", {"token": "<|im_end|>"}, "\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_function=FunctionFormatter(slots=["{{content}}", "<|im_end|>"], tool_format="qwen2_5"),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    template_class=Qwen2vlTemplate,
)

_register_template(
    name="qwen2-omni",
    format_user=StringFormatter(
        slots=[
            {"token": "<|im_start|>"},
            "user\n{{content}}",
            {"token": "<|im_end|>"},
            "\n",
            {"token": "<|im_start|>"},
            "assistant\n",
        ]
    ),
    format_system=StringFormatter(
        slots=[{"token": "<|im_start|>"}, "system\n{{content}}", {"token": "<|im_end|>"}, "\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_function=FunctionFormatter(slots=["{{content}}", "<|im_end|>"], tool_format="qwen2_5"),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    default_system="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|IMAGE|>",
    video_token="<|VIDEO|>",
    audio_token="<|AUDIO|>",
    template_class=Qwen2_5OmniTemplate,
)

_register_template(
    name="qvq",
    format_user=StringFormatter(
        slots=[
            {"token": "<|im_start|>"},
            "user\n{{content}}",
            {"token": "<|im_end|>"},
            "\n",
            {"token": "<|im_start|>"},
            "assistant\n",
        ]
    ),
    format_system=StringFormatter(
        slots=[{"token": "<|im_start|>"}, "system\n{{content}}", {"token": "<|im_end|>"}, "\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_function=FunctionFormatter(slots=["{{content}}", "<|im_end|>"], tool_format="qwen2_5"),
    format_tools=ToolFormatter(tool_format="qwen2_5"),
    default_system="A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nThe assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nThe reasoning process is enclosed within <think> </think>.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    template_class=ReasoningTemplate,
)

_register_template(
    name="qvq_preview",
    format_user=StringFormatter(
        slots=[
            {"token": "<|im_start|>"},
            "user\n{{content}}",
            {"token": "<|im_end|>"},
            "\n",
            {"token": "<|im_start|>"},
            "assistant\n",
        ]
    ),
    format_system=StringFormatter(
        slots=[{"token": "<|im_start|>"}, "system\n{{content}}", {"token": "<|im_end|>"}, "\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. Answer in the language of the question. You should think step-by-step.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    image_token="<|image_pad|>",
    video_token="<|image_pad|>",  # TODO: change it as a real video token
)

_register_template(
    name="qwen2-audio",
    format_user=StringFormatter(
        slots=[
            {"token": "<|im_start|>"},
            "user\n{{content}}",
            {"token": "<|im_end|>"},
            "\n",
            {"token": "<|im_start|>"},
            "assistant\n",
        ]
    ),
    format_system=StringFormatter(
        slots=[{"token": "<|im_start|>"}, "system\n{{content}}", {"token": "<|im_end|>"}, "\n"]
    ),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    audio_token="<|AUDIO|>",
)


_register_template(
    name="microsoft/Phi-3",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


_register_template(
    name="minicpmo",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    audio_token="<|audio|>",
)


_register_template(
    name="minicpmo-tts",
    format_user=StringFormatter(
        slots=[
            "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"
        ]
    ),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    default_system="Clone the voice in the provided audio prompt.",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    audio_token="<|audio|>",
)
