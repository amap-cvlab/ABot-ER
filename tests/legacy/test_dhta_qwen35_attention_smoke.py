"""Standalone Qwen3.5 attention integration smoke test for DHTA.

This file intentionally avoids importing ``llmtuner.__init__`` so it can run on
login nodes where optional serving/MOS dependencies are unavailable.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import logging
from pathlib import Path
import sys
import types

import torch


def _load_dhta(repo_root: Path):
    for package_name in ("llmtuner", "llmtuner.model", "llmtuner.model.model_utils", "llmtuner.extras"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules.setdefault(package_name, package)
    logging_module = types.ModuleType("llmtuner.extras.logging")
    logging_module.get_logger = logging.getLogger
    sys.modules["llmtuner.extras.logging"] = logging_module
    module_name = "llmtuner.model.model_utils.dhta"
    spec = importlib.util.spec_from_file_location(module_name, repo_root / "src/llmtuner/model/model_utils/dhta.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stub_broken_optional_torchaudio() -> None:
    # Some login-node conda envs have an unused torchaudio build with a mismatched
    # libtorch ABI. Qwen3.5 itself does not use it, so isolate that environment
    # issue from this architecture test.
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    torchaudio.functional = types.SimpleNamespace(rnnt_loss=lambda *args, **kwargs: None)
    sys.modules["torchaudio"] = torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2] / "third_party/tuningfactory"
    dhta = _load_dhta(repo_root)
    _stub_broken_optional_torchaudio()

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    config = AutoConfig.from_pretrained(args.base_config, trust_remote_code=True)
    values = {
        "dhta_enabled": True,
        "dhta_layers": [19, 23, 27, 31],
        "dhta_rank": 16,
        "dhta_alpha": 16.0,
        "dhta_qa_token_id": 151700,
        "dhta_task_plan_token_id": 151701,
        "dhta_require_route_token": True,
    }
    for target in (config, config.text_config):
        for key, value in values.items():
            setattr(target, key, value)

    dhta.prepare_dhta_model_class(config)
    attention = Qwen3_5Attention(config.text_config, layer_idx=19)
    assert attention.q_proj.dhta._z3_leaf is True
    assert attention.v_proj.dhta._z3_leaf is True
    attention.eval()
    hidden = torch.randn(1, 3, config.text_config.hidden_size)
    attention.q_proj._dhta_route = dhta.DHTA_QA_ROUTE
    base_q = torch.nn.functional.linear(hidden, attention.q_proj.weight, attention.q_proj.bias)
    routed_q_at_init = attention.q_proj(hidden)
    torch.testing.assert_close(base_q, routed_q_at_init)

    with torch.no_grad():
        attention.q_proj.dhta.qa_b.weight.normal_(std=0.01)
    qa_q = attention.q_proj(hidden)
    attention.q_proj._dhta_route = dhta.DHTA_TASK_PLAN_ROUTE
    task_plan_q = attention.q_proj(hidden)
    assert not torch.equal(qa_q, base_q)
    torch.testing.assert_close(task_plan_q, base_q)

    state = attention.state_dict()
    assert "q_proj.weight" in state
    assert "q_proj.dhta.qa_a.weight" in state
    assert "v_proj.dhta.task_plan_b.weight" in state
    restored = Qwen3_5Attention(config.text_config, layer_idx=19)
    restored.load_state_dict(state, strict=True)
    restored.q_proj._dhta_route = dhta.DHTA_QA_ROUTE
    torch.testing.assert_close(restored.q_proj(hidden), qa_q)

    # Starting from an ordinary backbone must leave its constructor untouched,
    # then inject fresh DHTA tensors after from_pretrained has completed.
    postload_config = AutoConfig.from_pretrained(args.base_config, trust_remote_code=True)
    for target in (postload_config, postload_config.text_config):
        for key, value in values.items():
            setattr(target, key, value)
    dhta.set_dhta_injection_mode(postload_config, False)
    postload_attention = Qwen3_5Attention(postload_config.text_config, layer_idx=19)
    assert not isinstance(postload_attention.q_proj, dhta.DHTARoutedLinear)
    dhta._configure_attention_instance(postload_attention, postload_config, 19)
    assert isinstance(postload_attention.q_proj, dhta.DHTARoutedLinear)
    assert postload_attention.q_proj.dhta.qa_a.weight.shape == (16, postload_config.text_config.hidden_size)
    assert postload_attention.q_proj.dhta.qa_a.weight.dtype == postload_attention.q_proj.weight.dtype
    postload_attention.q_proj._dhta_route = dhta.DHTA_QA_ROUTE
    postload_hidden = torch.randn(1, 3, postload_config.text_config.hidden_size)
    assert torch.isfinite(postload_attention.q_proj(postload_hidden)).all()
    print(
        "DHTA_QWEN35_SMOKE_OK "
        f"q_shape={tuple(base_q.shape)} residual_state_keys={sum('.dhta.' in key for key in state)} "
        "postload_injection=ok"
    )


if __name__ == "__main__":
    main()
