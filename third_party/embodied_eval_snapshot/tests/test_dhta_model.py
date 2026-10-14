from types import SimpleNamespace

import pytest
import torch

from embodied_eval.models.dhta import (
    DHTA_QA_ROUTE,
    DHTA_TASK_PLAN_ROUTE,
    DHTALowRankResidual,
    inject_dhta_route,
    prepare_dhta_model_class,
    validate_dhta_loading_info,
    validate_dhta_tokenizer,
)


def test_route_injection_matches_training_semantics_for_multimodal_message() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "  where is the cup?"},
            ],
        }
    ]
    assert inject_dhta_route(messages, "2d_qa") is messages
    assert messages[0]["content"][0] == {"type": "text", "text": "[2D_QA]\n"}
    assert messages[0]["content"][2]["text"] == "where is the cup?"
    inject_dhta_route(messages, "2d_qa")
    assert sum(
        item.get("text", "").count("[2D_QA]")
        for item in messages[0]["content"]
        if item.get("type") == "text"
    ) == 1


def test_route_injection_fails_closed_on_conflict() -> None:
    messages = [{"role": "user", "content": "[TASK_PLAN]\nquestion"}]
    with pytest.raises(ValueError, match="Invalid DHTA route"):
        inject_dhta_route(messages, "2d_qa")


def test_residual_routes_only_the_selected_branch() -> None:
    residual = DHTALowRankResidual(4, 3, rank=2, alpha=2.0)
    with torch.no_grad():
        residual.qa_b.weight.fill_(0.5)
        residual.task_plan_b.weight.fill_(-0.25)
    hidden = torch.randn(2, 3, 4)
    routes = torch.tensor([DHTA_QA_ROUTE, DHTA_TASK_PLAN_ROUTE])
    mixed = residual(hidden, routes)
    torch.testing.assert_close(mixed[0], residual(hidden[:1], DHTA_QA_ROUTE)[0])
    torch.testing.assert_close(mixed[1], residual(hidden[1:], DHTA_TASK_PLAN_ROUTE)[0])


def test_standard_checkpoint_does_not_patch_transformers() -> None:
    prepare_dhta_model_class(SimpleNamespace(dhta_enabled=False))


@pytest.mark.parametrize('family', ['qwen3_5', 'qwen3_5_moe'])
@pytest.mark.parametrize('gate_type', ['sigmoid', 'swish'])
def test_dense_and_moe_checkpoint_roundtrip(tmp_path, family, gate_type):
    from transformers import AutoConfig
    from importlib import import_module
    from embodied_eval.models.dhta import validate_dhta_model
    if family.endswith('moe') and gate_type == 'swish':
        pytest.skip('swish is a dense checkpoint feature')

    fields = dict(dhta_enabled=True, dhta_layers=[0], dhta_rank=2,
                  dhta_alpha=2., dhta_qa_token_id=5, dhta_task_plan_token_id=6,
                  dhta_require_route_token=True)
    config = AutoConfig.for_model(family, text_config=dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        layer_types=['full_attention'], num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_experts=2, num_experts_per_tok=1, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, output_gate_type=gate_type, **fields),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                           num_heads=2, out_hidden_size=16), **fields)
    prepare_dhta_model_class(config)
    name = 'Qwen3_5MoeForConditionalGeneration' if family.endswith('moe') else 'Qwen3_5ForConditionalGeneration'
    cls = getattr(import_module(f'transformers.models.{family}.modeling_{family}'), name)
    model = cls(config).eval()
    validate_dhta_model(model)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if '.dhta.' in key:
                parameter.fill_(0.03 if 'qa_' in key else -0.02)
    model.save_pretrained(tmp_path)
    restored, info = cls.from_pretrained(tmp_path, output_loading_info=True)
    validate_dhta_loading_info(info)
    validate_dhta_model(restored)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    for token, route in ((5, 0), (6, 1)):
        with torch.no_grad():
            output = restored(input_ids=torch.tensor([[token, 7, 8]]), use_cache=False)
        assert torch.isfinite(output.logits).all()
        assert restored._dhta_current_route == route
    # Patching the same family must leave a subsequent ordinary model untouched.
    config.dhta_enabled = config.text_config.dhta_enabled = False
    standard = cls(config)
    assert not any('.dhta.' in k for k in standard.state_dict())


def test_loading_info_rejects_unrestored_dhta_weights() -> None:
    validate_dhta_loading_info({"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []})
    with pytest.raises(RuntimeError, match="did not load exactly"):
        validate_dhta_loading_info({"missing_keys": ["model.layers.19.q_proj.dhta.qa_a.weight"]})


class _Tokenizer:
    ids = {"[2D_QA]": 101, "[TASK_PLAN]": 102}

    def encode(self, token, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.ids[token]]


def test_gdn_swish_config_does_not_change_full_attention_sigmoid(monkeypatch):
    from transformers import Qwen3_5Config
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    config = Qwen3_5Config(text_config=dict(hidden_size=16, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, output_gate_type='swish'))
    config.text_config._attn_implementation = 'eager'
    attn = module.Qwen3_5Attention(config.text_config, 0).eval()
    def ones_attention(_self, q, k, v, mask, **kwargs):
        return torch.ones(q.shape[0], q.shape[2], q.shape[1], q.shape[3]), None
    monkeypatch.setattr(module, 'eager_attention_forward', ones_attention)
    hidden = torch.randn(1, 3, 16)
    positions = (torch.ones(1, 3, 8), torch.zeros(1, 3, 8))
    gate = attn.q_proj(hidden).view(1, 3, 2, 16).chunk(2, dim=-1)[1].reshape(1, 3, 16)
    for name in ['swish', 'sigmoid']:
        attn.config.output_gate_type = name
        actual, _ = attn(hidden, positions, None)
        torch.testing.assert_close(actual, attn.o_proj(torch.sigmoid(gate)))
    norm = module.Qwen3_5RMSNormGated(16)
    ones = torch.ones_like(gate)
    expected = ones * torch.rsqrt(ones.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    expected = expected * norm.weight * torch.nn.functional.silu(gate)
    torch.testing.assert_close(norm(ones, gate), expected)


def test_tokenizer_must_match_checkpoint_route_ids() -> None:
    config = SimpleNamespace(
        dhta_enabled=True,
        dhta_qa_token="[2D_QA]",
        dhta_task_plan_token="[TASK_PLAN]",
        dhta_qa_token_id=101,
        dhta_task_plan_token_id=102,
    )
    validate_dhta_tokenizer(config, _Tokenizer())
    config.dhta_qa_token_id = 999
    with pytest.raises(ValueError, match="resolve atomically"):
        validate_dhta_tokenizer(config, _Tokenizer())
