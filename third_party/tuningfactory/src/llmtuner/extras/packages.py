import importlib.metadata
import importlib.util
from functools import lru_cache

from packaging import version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "0.0.0"


def is_fastapi_availble():
    return _is_package_available("fastapi")


def is_librosa_available():
    return _is_package_available("librosa")


def is_jieba_available():
    return _is_package_available("jieba")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


def is_nltk_available():
    return _is_package_available("nltk")


def is_requests_available():
    return _is_package_available("requests")


def is_rouge_available():
    return _is_package_available("rouge_chinese")


def is_starlette_available():
    return _is_package_available("sse_starlette")


def is_uvicorn_available():
    return _is_package_available("uvicorn")


def is_turbo_available():
    return _is_package_available("transformers_turbo")


def is_pyreft_available():
    return _is_package_available("pyreft")


def is_safetensors_available():
    return _is_package_available("safetensors")


@lru_cache
def is_transformers_version_greater_than(content: str):
    return version.parse(_get_package_version("transformers")) >= version.parse(content)


@lru_cache
def is_trl_version_greater_than(content: str):
    return version.parse(_get_package_version("trl")) >= version.parse(content)


@lru_cache
def is_flash_attn_version_greater_than(content: str):
    return version.parse(_get_package_version("flash_attn")) >= version.parse(content)


@lru_cache
def is_vllm_version_greater_than(content: str):
    return version.parse(_get_package_version("vllm")) >= version.parse(content)


def is_mcore_version_greater_than(content: str):
    return version.parse(_get_package_version("megatron.core")) >= version.parse(content)


@lru_cache
def is_torch_version_greater_than(content: str):
    return version.parse(_get_package_version("torch")) >= version.parse(content)
