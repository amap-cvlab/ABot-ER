import copy
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
import weakref

import torch


ROOT = Path(__file__).resolve().parents[2] / "third_party/tuningfactory"


def _load_without_llmtuner_side_effects(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in (
    "llmtuner",
    "llmtuner.data",
    "llmtuner.model",
    "llmtuner.model.model_utils",
    "llmtuner.extras",
):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules.setdefault(package_name, package)

logging_module = types.ModuleType("llmtuner.extras.logging")
logging_module.get_logger = logging.getLogger
sys.modules["llmtuner.extras.logging"] = logging_module

fixers = _load_without_llmtuner_side_effects("llmtuner.data.fixers", "src/llmtuner/data/fixers.py")
dhta = _load_without_llmtuner_side_effects("llmtuner.model.model_utils.dhta", "src/llmtuner/model/model_utils/dhta.py")
FixerContext = fixers.FixerContext
PrependDHTATaskTokenFixer = fixers.PrependDHTATaskTokenFixer
DHTA_QA_ROUTE = dhta.DHTA_QA_ROUTE
DHTA_TASK_PLAN_ROUTE = dhta.DHTA_TASK_PLAN_ROUTE
DHTALowRankResidual = dhta.DHTALowRankResidual
DHTARoutedLinear = dhta.DHTARoutedLinear


class DHTATest(unittest.TestCase):
    def setUp(self):
        self.fixer_context = FixerContext(
            messages_key="conversations",
            image_key="images",
            role_tag="from",
            content_tag="value",
            user_tag="human",
        )

    def test_fixer_injects_exactly_one_token_without_touching_assistant(self):
        for task_family, expected_token in (("2d_qa", "[2D_QA]"), ("task_plan", "[TASK_PLAN]")):
            with self.subTest(task_family=task_family):
                record = {
                    "task_family": task_family,
                    "images": ["a.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nquestion"},
                        {"from": "gpt", "value": "unchanged answer"},
                    ],
                }
                assistant_before = copy.deepcopy(record["conversations"][1])
                fixer = PrependDHTATaskTokenFixer()
                fixed = fixer(record, self.fixer_context)
                fixed_again = fixer(fixed, self.fixer_context)
                self.assertIs(fixed_again, fixed)
                self.assertEqual(fixed["conversations"][0]["value"].count(expected_token), 1)
                self.assertEqual(fixed["conversations"][1], assistant_before)

    def test_fixer_fails_closed_on_unknown_or_conflicting_route(self):
        fixer = PrependDHTATaskTokenFixer()
        unknown = {
            "task_family": "unknown",
            "conversations": [{"from": "human", "value": "question"}],
        }
        conflict = {
            "task_family": "2d_qa",
            "conversations": [{"from": "human", "value": "[TASK_PLAN]\nquestion"}],
        }
        self.assertIsNone(fixer(unknown, self.fixer_context))
        self.assertIsNone(fixer(conflict, self.fixer_context))

    def test_residual_is_zero_initialized_and_routes_gradients(self):
        torch.manual_seed(7)
        residual = DHTALowRankResidual(8, 6, rank=2, alpha=2.0)
        hidden = torch.randn(2, 3, 8)
        self.assertEqual(torch.count_nonzero(residual(hidden, DHTA_QA_ROUTE)).item(), 0)
        self.assertEqual(torch.count_nonzero(residual(hidden, DHTA_TASK_PLAN_ROUTE)).item(), 0)
        residual(hidden, DHTA_QA_ROUTE).sum().backward()
        self.assertIsNotNone(residual.qa_b.weight.grad)
        self.assertIsNotNone(residual.task_plan_a.weight.grad)
        self.assertIsNotNone(residual.task_plan_b.weight.grad)
        self.assertEqual(torch.count_nonzero(residual.task_plan_a.weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(residual.task_plan_b.weight.grad).item(), 0)

    def test_residual_children_are_frozen_against_hf_post_init(self):
        residual = DHTALowRankResidual(8, 6, rank=2, alpha=2.0)
        projections = (
            residual.qa_a,
            residual.qa_b,
            residual.task_plan_a,
            residual.task_plan_b,
        )
        self.assertTrue(getattr(residual, "_is_hf_initialized", False))
        self.assertIs(getattr(residual, "_z3_leaf", None), True)
        self.assertTrue(all(getattr(module, "_is_hf_initialized", False) for module in projections))
        self.assertTrue(all(getattr(module.weight, "_is_hf_initialized", False) for module in projections))
        self.assertEqual(torch.count_nonzero(residual.qa_b.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(residual.task_plan_b.weight).item(), 0)

    def test_residual_state_dict_roundtrip_preserves_both_branches(self):
        torch.manual_seed(11)
        source = DHTALowRankResidual(8, 6, rank=2, alpha=4.0)
        with torch.no_grad():
            source.qa_b.weight.normal_()
            source.task_plan_b.weight.normal_()
        restored = DHTALowRankResidual(8, 6, rank=2, alpha=4.0)
        restored.load_state_dict(source.state_dict(), strict=True)
        hidden = torch.randn(1, 4, 8)
        for route in (DHTA_QA_ROUTE, DHTA_TASK_PLAN_ROUTE):
            torch.testing.assert_close(source(hidden, route), restored(hidden, route))

    def test_routed_linear_preserves_base_keys_and_executes_explicit_child(self):
        torch.manual_seed(13)
        base = torch.nn.Linear(8, 6, bias=False)
        base_weight = base.weight
        projection = DHTARoutedLinear(base, rank=2, alpha=2.0)
        self.assertIs(projection.weight, base_weight)
        self.assertIn("weight", projection.state_dict())
        self.assertIn("dhta.qa_a.weight", projection.state_dict())
        self.assertNotIn("base_projection.weight", projection.state_dict())

        hidden = torch.randn(1, 3, 8)
        shared = torch.nn.functional.linear(hidden, projection.weight, projection.bias)
        projection._dhta_route = DHTA_QA_ROUTE
        torch.testing.assert_close(projection(hidden), shared)
        with torch.no_grad():
            projection.dhta.qa_b.weight.normal_()
        self.assertFalse(torch.equal(projection(hidden), shared))
        projection._dhta_route = DHTA_TASK_PLAN_ROUTE
        torch.testing.assert_close(projection(hidden), shared)

    def test_mixed_batch_uses_exactly_one_branch_per_row(self):
        torch.manual_seed(19)
        residual = DHTALowRankResidual(8, 6, rank=2, alpha=2.0)
        with torch.no_grad():
            residual.qa_b.weight.fill_(0.5)
            residual.task_plan_b.weight.fill_(-0.25)
        hidden = torch.randn(2, 3, 8)
        route = torch.tensor([DHTA_QA_ROUTE, DHTA_TASK_PLAN_ROUTE])
        mixed = residual(hidden, route)
        torch.testing.assert_close(mixed[0], residual(hidden[:1], DHTA_QA_ROUTE)[0])
        torch.testing.assert_close(mixed[1], residual(hidden[1:], DHTA_TASK_PLAN_ROUTE)[0])

    def test_route_controller_is_fail_closed_and_persists_for_decode(self):
        projection = DHTARoutedLinear(torch.nn.Linear(4, 4, bias=False), rank=2, alpha=2.0)
        model = torch.nn.Module()
        model.config = types.SimpleNamespace(
            dhta_qa_token_id=101,
            dhta_task_plan_token_id=102,
            dhta_require_route_token=True,
        )
        model._dhta_projection_refs = (weakref.ref(projection),)
        model._dhta_current_route = None

        dhta._set_model_route(model, torch.tensor([[101, 7, 8]]), past_key_values=None)
        self.assertEqual(projection._dhta_route, DHTA_QA_ROUTE)
        dhta._set_model_route(model, torch.tensor([[9]]), past_key_values=object())
        self.assertEqual(projection._dhta_route, DHTA_QA_ROUTE)

        with self.assertRaises(ValueError):
            dhta._set_model_route(model, torch.tensor([[7, 8]]), past_key_values=None)
        with self.assertRaises(ValueError):
            dhta._set_model_route(model, torch.tensor([[101, 102]]), past_key_values=None)

    def test_validate_model_checks_explicit_projection_inventory(self):
        model = torch.nn.Module()
        model.config = types.SimpleNamespace(
            dhta_enabled=True,
            dhta_layers=[19, 23],
            dhta_rank=2,
        )
        model.projections = torch.nn.ModuleList()
        for layer_idx in model.config.dhta_layers:
            for projection_name in ("q_proj", "v_proj"):
                projection = DHTARoutedLinear(torch.nn.Linear(4, 4, bias=False), rank=2, alpha=2.0)
                projection.dhta_layer_idx = layer_idx
                projection.dhta_projection_name = projection_name
                model.projections.append(projection)

        dhta.validate_dhta_model(model)
        model.projections[-1].dhta_projection_name = "q_proj"
        with self.assertRaisesRegex(RuntimeError, "DHTA projection mismatch"):
            dhta.validate_dhta_model(model)

    def test_injection_mode_is_propagated_to_nested_text_config(self):
        text_config = types.SimpleNamespace()
        config = types.SimpleNamespace(dhta_enabled=True, text_config=text_config)
        dhta.set_dhta_injection_mode(config, False)
        self.assertIs(config.dhta_inject_during_init, False)
        self.assertIs(text_config.dhta_inject_during_init, False)
        dhta.set_dhta_injection_mode(config, True)
        self.assertIs(config.dhta_inject_during_init, True)
        self.assertIs(text_config.dhta_inject_during_init, True)

    def test_attention_injection_is_idempotent_and_rejects_partial_wrap(self):
        config = types.SimpleNamespace(dhta_layers=[3], dhta_rank=2, dhta_alpha=2.0)
        attention = torch.nn.Module()
        attention.q_proj = torch.nn.Linear(8, 8, bias=False)
        attention.v_proj = torch.nn.Linear(8, 8, bias=False)
        dhta._configure_attention_instance(attention, config, 3)
        q_proj = attention.q_proj
        v_proj = attention.v_proj
        dhta._configure_attention_instance(attention, config, 3)
        self.assertIs(attention.q_proj, q_proj)
        self.assertIs(attention.v_proj, v_proj)

        partial = torch.nn.Module()
        partial.q_proj = DHTARoutedLinear(torch.nn.Linear(8, 8, bias=False), rank=2, alpha=2.0)
        partial.v_proj = torch.nn.Linear(8, 8, bias=False)
        with self.assertRaisesRegex(RuntimeError, "partially wrapped"):
            dhta._configure_attention_instance(partial, config, 3)


if __name__ == "__main__":
    unittest.main()
