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

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple, Union

from typing_extensions import override


class FunctionCall(NamedTuple):
    name: str
    arguments: str


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}]).\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```).\n"""
    "```\n"
)

GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具{tool_text}"
)

GLM4_MOE_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:"
    "\n<tool_call>{{function-name}}"
    "\n<arg_key>{{arg-key-1}}</arg_key>"
    "\n<arg_value>{{arg-value-1}}</arg_value>"
    "\n<arg_key>{{arg-key-2}}</arg_key>"
    "\n<arg_value>{{arg-value-2}}</arg_value>"
    "\n...\n</tool_call>\n"
)

QWEN2_5_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query."
    "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    "{tool_text}"
    """\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>"""
)

QWEN35_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>{tool_text}"
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: "
    "an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)

LLAMA3_TOOL_PROMPT = (
    "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
    "You have access to the following functions. To call a function, please respond with JSON for a function call. "
    """Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. """
    "Do not use variables.\n\n{tool_text}"
)


@dataclass
class ToolUtils(ABC):
    @staticmethod
    @abstractmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        r"""
        Generates the system message describing all the available tools.
        """
        ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        r"""
        Generates the assistant message including all the tool calls.
        """
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        r"""
        Extracts all the function calls from the assistant message.

        It should be an inverse function of `function_formatter`.
        """
        ...


class DefaultToolUtils(ToolUtils):
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_text = ""
        for name, arguments in functions:
            function_text += f"Action: {name}\nAction Input: {arguments}\n"

        return function_text

    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @staticmethod
    def tool_extractor(content: str) -> Union[str, list[tuple[str, str]]]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: list[tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append((tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content

        return results


class GLM4ToolUtils(ToolUtils):
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("GLM-4 does not support parallel functions.")

        return f"{functions[0].name}\n{functions[0].arguments}"

    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @staticmethod
    def tool_extractor(content: str) -> Union[str, list[tuple[str, str]]]:
        if "\n" not in content:
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)
        try:
            arguments = json.loads(tool_input)
        except json.JSONDecodeError:
            return content

        return [(tool_name, json.dumps(arguments, ensure_ascii=False))]


class QWEN25ToolUtils(ToolUtils):
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = []
        for name, arguments in functions:
            function_texts.append(
                "<tool_call>\n" + f'{{"name": "{name}", "arguments": {arguments}}}' + "\n</tool_call>"
            )

        return "\n".join(function_texts)

    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN2_5_TOOL_PROMPT.format(tool_text=tool_text)

    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<tool_call>(.+?)</tool_call>(?=\s*<tool_call>|\s*$)", re.DOTALL)
        tool_match: list[str] = re.findall(regex, content)
        if not tool_match:
            return content

        results = []
        for tool in tool_match:
            try:
                tool = json.loads(tool.strip())
            except json.JSONDecodeError:
                return content

            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


class QWEN35ToolUtils(ToolUtils):
    r"""Qwen 3.5 tool using template. ref llama-factory"""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_text += "\n" + json.dumps(tool, ensure_ascii=False)

        return QWEN35_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = []
        for func in functions:
            name, arguments = func.name, json.loads(func.arguments)
            prompt = f"<tool_call>\n<function={name}>"
            for key, value in arguments.items():
                prompt += f"\n<parameter={key}>"
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += f"\n{value}\n</parameter>"
            prompt += "\n</function>\n</tool_call>"
            function_texts.append(prompt)

        return "\n".join(function_texts)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        results = []
        regex = re.compile(r"<tool_call>\s*<function=\s*([^\s<>]+)\s*(.*?)\s*</function>\s*</tool_call>", re.DOTALL)
        for func_name, params_block in re.findall(regex, content):
            args_dict = {}
            param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
            for key, raw_value in re.findall(param_pattern, params_block.strip()):
                value = raw_value.strip()
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = raw_value.strip()
                args_dict[key] = parsed_value

            results.append(FunctionCall(func_name.strip(), json.dumps(args_dict, ensure_ascii=False)))

        return results if results else content


class Llama3ToolUtils(ToolUtils):
    r"""
    Llama 3.x tool using template with `tools_in_user_message=False`.

    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
    """

    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        date = datetime.now().strftime("%d %b %Y")
        tool_text = ""
        for tool in tools:
            wrapped_tool = {"type": "function", "function": tool}
            tool_text += json.dumps(wrapped_tool, indent=4, ensure_ascii=False) + "\n\n"

        return LLAMA3_TOOL_PROMPT.format(date=date, tool_text=tool_text)

    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("Llama-3 does not support parallel functions.")

        return f'{{"name": "{functions[0].name}", "parameters": {functions[0].arguments}}}'

    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        try:
            tool = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        if "name" not in tool or "parameters" not in tool:
            return content

        return [FunctionCall(tool["name"], json.dumps(tool["parameters"], ensure_ascii=False))]


class GLM4MOEToolUtils(QWEN25ToolUtils):
    r"""GLM-4-MOE tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return GLM4_MOE_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_json = [
            {"func_name": name, "func_key_values": json.loads(arguments)} for name, arguments in functions
        ]
        function_texts = []
        for func in function_json:
            prompt = "\n<tool_call>" + func["func_name"]
            for key, value in func["func_key_values"].items():
                prompt += "\n<arg_key>" + key + "</arg_key>"
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += "\n<arg_value>" + value + "</arg_value>"
            function_texts.append(prompt)

        return "\n".join(function_texts)


TOOLS = {
    "default": DefaultToolUtils(),
    "glm4": GLM4ToolUtils(),
    "llama3": Llama3ToolUtils(),
    "qwen2_5": QWEN25ToolUtils(),
    "qwen3_5": QWEN35ToolUtils(),
    "glm4_moe": GLM4MOEToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")

    return tool_utils
