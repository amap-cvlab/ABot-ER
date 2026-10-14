import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel
from transformers.image_utils import load_image

from ..chat import ChatModel
from ..data import Role as DataRole
from ..extras.logging import get_logger
from ..extras.misc import torch_gc
from ..extras.packages import is_fastapi_availble, is_starlette_available, is_uvicorn_available
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseUsage,
    ChatCompletionStreamResponse,
    Finish,
    Function,
    FunctionCall,
    ModelCard,
    ModelList,
    Role,
    ScoreEvaluationRequest,
    ScoreEvaluationResponse,
    SequenceCandidatesResponse,
    TokenCandidatesResponse,
    TokenDebugRequest,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray

if is_fastapi_availble():
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware


if is_starlette_available():
    from sse_starlette import EventSourceResponse


if is_uvicorn_available():
    import uvicorn

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def dictify(data: "BaseModel") -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


def create_app(chat_model: "ChatModel", **kwargs) -> "FastAPI":
    raise RuntimeError("Legacy API server is disabled in ABot-ER. Use the documented loopback-only TP server.")
    app = FastAPI(lifespan=lifespan, **kwargs)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", 1)))
    role_mapping = {
        Role.USER: DataRole.USER.value,
        Role.ASSISTANT: DataRole.ASSISTANT.value,
        Role.SYSTEM: DataRole.SYSTEM.value,
        Role.FUNCTION: DataRole.FUNCTION.value,
        Role.TOOL: DataRole.OBSERVATION.value,
    }

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: ChatCompletionRequest):
        if not chat_model.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")
        if request.messages and request.raw_messages:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="raw messages and messages cannot both use.")

        raw_message_mode = False
        if request.messages:
            if len(request.messages) == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

            if request.messages[0].role == Role.SYSTEM:
                system = request.messages.pop(0).content
            else:
                system = ""

            if len(request.messages) % 2 == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

            input_messages = []
            images = []
            for i, message in enumerate(request.messages):
                if i % 2 == 0 and message.role not in [Role.USER, Role.TOOL]:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
                elif i % 2 == 1 and message.role not in [Role.ASSISTANT, Role.FUNCTION]:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
                if message.role == Role.ASSISTANT and isinstance(message.tool_calls, list) and len(message.tool_calls):
                    tool_calls = [
                        {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                        for tool_call in message.tool_calls
                    ]
                    content = json.dumps(tool_calls, ensure_ascii=False)
                    input_messages.append({"role": role_mapping[Role.FUNCTION], "content": content})
                elif isinstance(message.content, list):
                    for input_item in message.content:
                        if input_item.type == "text":
                            input_messages.append({"role": role_mapping[message.role], "content": input_item.text})
                        elif input_item.type == "image_url":
                            if input_item.image_url:
                                images.append(load_image(input_item.image_url.url))
                            else:
                                raise ValueError("image type need image url param")
                else:
                    input_messages.append({"role": role_mapping[message.role], "content": message.content})
        elif request.raw_messages:
            input_messages = request.raw_messages
            system = ""
            images = []
            raw_message_mode = True
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="can not find messages and raw_messages params.please check it")

        tool_list = request.tools
        if isinstance(tool_list, list) and len(tool_list):
            try:
                tools = json.dumps([dictify(tool.function) for tool in tool_list], ensure_ascii=False)
            except Exception:
                logger.exception("Invalid tools")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tools")
        else:
            tools = None

        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, chat_completion, input_messages, system, tools, images, request, raw_message_mode)  # type: ignore

    def chat_completion(
        messages: Sequence[Union[Dict[str, str],str]],
        system: str,
        tools: str,
        images: Optional[List["NDArray"]],
        request: ChatCompletionRequest,
        raw_message_mode:bool = False
    ):
        completion_id = "chatcmpl-{}".format(uuid.uuid4().hex)

        if request.stream:
            # TODO support to stream api
            # stream has cause asyncio.exceptions.CancelledError: Cancelled by cancel scope 7f9bc9d151e0
            if tools:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")
            if request.n > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream multiple responses."
                )

            generate = stream_chat_completion(completion_id, messages, system, tools, images, request)
            return EventSourceResponse(generate, media_type="text/event-stream")

        responses = chat_model.chat(
            messages,
            system,
            tools,
            images,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            num_return_sequences=request.n,
            stop=request.stop,
            raw_message_mode=raw_message_mode
        )

        prompt_length, response_length = 0, 0
        choices = []
        for i, response in enumerate(responses):
            if tools:
                result = chat_model.template.format_tools.extract(response.response_text)
            else:
                result = response.response_text

            if isinstance(result, list):
                tool_calls = []
                for tool in result:
                    function = Function(name=tool[0], arguments=tool[1])
                    tool_calls.append(FunctionCall(id="call_{}".format(uuid.uuid4().hex), function=function))

                response_message = ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=tool_calls)
                finish_reason = Finish.TOOL
            else:
                response_message = ChatCompletionMessage(role=Role.ASSISTANT, content=result)
                finish_reason = Finish.STOP if response.finish_reason == "stop" else Finish.LENGTH

            choices.append(
                ChatCompletionResponseChoice(index=i, message=response_message, finish_reason=finish_reason)
            )
            prompt_length = response.prompt_length
            response_length += response.response_length

        usage = ChatCompletionResponseUsage(
            prompt_tokens=prompt_length,
            completion_tokens=response_length,
            total_tokens=prompt_length + response_length,
        )

        return ChatCompletionResponse(id=completion_id, model=request.model, choices=choices, usage=usage)

    def stream_chat_completion(
        completion_id: str,
        messages: Sequence[Dict[str, str]],
        system: str,
        tools: str,
        images: Optional[List["NDArray"]],
        request: ChatCompletionRequest,
    ):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=ChatCompletionMessage(role=Role.ASSISTANT, content=""), finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(id=completion_id, model=request.model, choices=[choice_data])
        yield jsonify(chunk)

        for new_text in chat_model.stream_chat(
            messages,
            system,
            tools,
            images,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
        ):
            if len(new_text) == 0:
                continue

            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=ChatCompletionMessage(content=new_text), finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(id=completion_id, model=request.model, choices=[choice_data])
            yield jsonify(chunk)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=ChatCompletionMessage(), finish_reason=Finish.STOP
        )
        chunk = ChatCompletionStreamResponse(id=completion_id, model=request.model, choices=[choice_data])
        yield jsonify(chunk)
        yield "[DONE]"

    @app.post("/v1/score/evaluation", response_model=ScoreEvaluationResponse, status_code=status.HTTP_200_OK)
    async def create_score_evaluation(request: ScoreEvaluationRequest):
        if chat_model.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        if len(request.messages) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request")

        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, get_score, request)

    def get_score(request: ScoreEvaluationRequest):
        scores = chat_model.get_scores(request.messages, max_length=request.max_length)
        return ScoreEvaluationResponse(model=request.model, scores=scores)

    @app.post("/v1/debug/token", response_model=TokenCandidatesResponse, status_code=status.HTTP_200_OK)
    async def create_token_debug(request: TokenDebugRequest):
        if not chat_model.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, get_token_candidates, request)

    def get_token_candidates(request: TokenDebugRequest):
        next_tokens_with_prob, output_tokens_with_prob = chat_model.get_token_candidates(
            message=request.prompt,
            system=request.system,
            max_new_tokens=request.max_new_tokens,
            top_k=request.topk)
        return TokenCandidatesResponse(
            result=dict(next_token=next_tokens_with_prob, generation_seq=output_tokens_with_prob)
        )

    @app.post("/v1/debug/sequence", response_model=SequenceCandidatesResponse, status_code=status.HTTP_200_OK)
    async def create_sequence_debug(request: TokenDebugRequest):
        if not chat_model.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, get_sequence_candidates, request)

    def get_sequence_candidates(request: TokenDebugRequest):
        result = chat_model.get_sequence_candidates(
            message=request.prompt,
            system=request.system,
            max_new_tokens=request.max_new_tokens)
        return SequenceCandidatesResponse(result=result)

    @app.get("/v1/debug/checkpreload.htm", status_code=status.HTTP_200_OK)
    async def health_check():
        # template would be set at the last step of chat_model initialization
        if not chat_model.can_generate or not hasattr(chat_model, "template"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

        return {"status": "healthy"}

    return app


if __name__ == "__main__":
    chat_model = ChatModel()
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8000)), workers=1)
