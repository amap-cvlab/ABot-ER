import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

HELPER_PATH = Path(__file__).resolve().parents[1] / "embodied_eval/models/qwen_dhta_eval_profile.py"
spec = importlib.util.spec_from_file_location("profile_under_test", HELPER_PATH)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def stage(tmp_path, monkeypatch):
    native = b"{# reasoning_effort #}native"
    generation = b'{"temperature": 1.0, "top_p": 0.95, "top_k": 20}'
    (tmp_path / "config.json").write_text(json.dumps(
        {"dhta_enabled": True, "model_type": "qwen3_5"}))
    (tmp_path / "chat_template.jinja").write_bytes(native)
    (tmp_path / "generation_config.json").write_bytes(generation)
    monkeypatch.setattr(profile, "NATIVE_SHA256", hashlib.sha256(native).hexdigest())
    monkeypatch.setattr(profile, "GENERATION_SHA256", hashlib.sha256(generation).hexdigest())


@pytest.mark.parametrize("thinking", [True, False])
def test_frozen_template_preserves_historical_thinking_without_effort(tmp_path, monkeypatch, thinking):
    stage(tmp_path, monkeypatch)
    result = profile.historical_dhta_profile(tmp_path)
    template = Path(result["chat_template"]).read_text()
    rendered = _compile_jinja_template(template).render(
        messages=[{"role": "user", "content": "[TASK_PLAN]\nfind cup"}],
        add_generation_prompt=True, enable_thinking=thinking)
    assert "reasoning_effort" not in template
    assert "xhigh" not in rendered
    assert rendered.endswith("<think>\n") if thinking else rendered.endswith("<think>\n\n</think>\n\n")
    assert "[TASK_PLAN]" in rendered


@pytest.mark.parametrize("filename", ["chat_template.jinja", "generation_config.json"])
def test_unaudited_inputs_fail_before_model_service(tmp_path, monkeypatch, filename):
    stage(tmp_path, monkeypatch)
    with (tmp_path / filename).open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="Unaudited"):
        profile.historical_dhta_profile(tmp_path)


def test_ordinary_and_moe_templates_are_unchanged(tmp_path, monkeypatch):
    stage(tmp_path, monkeypatch)
    (tmp_path / "chat_template.jinja").write_text("ordinary template")
    assert profile.historical_dhta_profile(tmp_path) is None
    (tmp_path / "config.json").write_text(json.dumps(
        {"dhta_enabled": True, "model_type": "qwen3_5_moe"}))
    assert profile.historical_dhta_profile(tmp_path) is None
