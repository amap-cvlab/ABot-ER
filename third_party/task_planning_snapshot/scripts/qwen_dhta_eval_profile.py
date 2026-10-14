"""Frozen historical Qwen template for DHTA checkpoints with reasoning-effort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

NATIVE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
LEGACY_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
GENERATION_SHA256 = "e8d5f26ca96fba21b24d011a5524d93136bef5c17ec7a64fe355085f142ecdb6"


def historical_dhta_profile(model_path):
    """Return the audited profile, leaving ordinary checkpoint templates intact."""
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    template_path = model_path / "chat_template.jinja"
    if not config_path.is_file() or not template_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    from abot_er.checkpoint_contract import normalize_checkpoint_config
    if normalize_checkpoint_config(config) is None or config.get("model_type") != "qwen3_5":
        return None
    native = template_path.read_bytes()
    if b"reasoning_effort" not in native:
        return None
    native_hash = hashlib.sha256(native).hexdigest()
    if native_hash != NATIVE_SHA256:
        raise ValueError("Unaudited DHTA reasoning-effort template; review before evaluation")
    generation_path = model_path / "generation_config.json"
    generation_hash = hashlib.sha256(generation_path.read_bytes()).hexdigest()
    if generation_hash != GENERATION_SHA256:
        raise ValueError("Unaudited DHTA generation config; review before evaluation")
    legacy_path = Path(__file__).with_name("qwen35_legacy_chat_template.jinja")
    if hashlib.sha256(legacy_path.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise ValueError("Historical DHTA template checksum mismatch")
    return {
        "name": "qwen_dhta_historical_v1",
        "chat_template": str(legacy_path),
        "chat_template_sha256": LEGACY_SHA256,
        "native_chat_template_sha256": native_hash,
        "generation_config_sha256": generation_hash,
        "generation_defaults": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "repetition_penalty": 1.0,
        },
    }
