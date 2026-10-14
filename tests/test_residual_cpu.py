"""Small CPU tensor checks; no downloaded weights, benchmark data or GPU runs."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import typing
import unittest

from abot_er.reproduction import ROOT

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


def residual_classes(path):
    # Execute the actual class definitions without unrelated training imports.
    import torch
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in {"DHTALowRankResidual", "DHTARoutedLinear"}]
    scope = dict(torch=torch, nn=torch.nn, math=math, os=os, Union=typing.Union,
                 DHTA_QA_ROUTE=0, DHTA_TASK_PLAN_ROUTE=1)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["DHTARoutedLinear"]


@unittest.skipUnless(HAS_TORCH, "CPU tensor tests require torch")
class ResidualTensorTests(unittest.TestCase):
    def test_both_implementations_and_namespaces(self):
        import torch
        paths = [ROOT / "third_party/tuningfactory/src/llmtuner/model/model_utils/dhta.py",
                 ROOT / "third_party/embodied_eval_snapshot/embodied_eval/models/dhta.py"]
        torch.manual_seed(17)
        reference = None
        for path in paths:
            linear = residual_classes(path)
            for namespace in ("dhta", "task_conditioned"):
                with self.subTest(path=path.name, namespace=namespace):
                    layer = linear(torch.nn.Linear(3, 4, bias=False), 2, 4., namespace)
                    with torch.no_grad():
                        layer.weight.fill_(.25)
                        layer.residual().qa_a.weight.fill_(.1)
                        layer.residual().qa_b.weight.fill_(.2)
                        layer.residual().task_plan_a.weight.fill_(.3)
                        layer.residual().task_plan_b.weight.fill_(.4)
                    keys = set(layer.state_dict())
                    self.assertEqual(keys, {"weight"} | {
                        f"{namespace}.{route}_{side}.weight" for route in ("qa", "task_plan")
                        for side in ("a", "b")})
                    x = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 10
                    layer._dhta_route = torch.tensor([0, 1])
                    y = layer(x)
                    expected = []
                    for i, route in enumerate(("qa", "task_plan")):
                        residual = layer.residual()
                        a = getattr(residual, route + "_a").weight
                        b = getattr(residual, route + "_b").weight
                        expected.append(torch.nn.functional.linear(x[i], layer.weight + 2 * b @ a))
                    torch.testing.assert_close(y, torch.stack(expected))
                    if reference is None:
                        reference = y.detach().clone()
                    torch.testing.assert_close(y, reference)
                    y.sum().backward()
                    self.assertIsNotNone(layer.weight.grad)
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in layer.residual().parameters()))
                    restored = linear(torch.nn.Linear(3, 4, bias=False), 2, 4., namespace)
                    restored.load_state_dict(layer.state_dict(), strict=True)
                    restored._dhta_route = torch.tensor([0, 1])
                    torch.testing.assert_close(restored(x), y)
                    restored._dhta_route = None
                    with self.assertRaises(RuntimeError): restored(x)

    @unittest.skipUnless(HAS_SAFETENSORS, "Merge checks require safetensors")
    def test_task_plan_merge_removes_both_routes_without_double_application(self):
        import torch
        from safetensors.torch import load_file, save_file
        path = ROOT / "third_party/task_planning_snapshot/scripts/dhta_runtime.py"
        spec = importlib.util.spec_from_file_location("tp_merge_cpu", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for namespace in ("dhta", "task_conditioned"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tensors, expected = {}, {}
                info = dict(layers=[0], rank=2, alpha=4., qa_token_id=10,
                            task_plan_token_id=11, namespace=namespace)
                for proj in ("q_proj", "v_proj"):
                    prefix = f"model.language_model.layers.0.self_attn.{proj}"
                    base = torch.ones(4, 3)
                    tensors[prefix + ".weight"] = base
                    for route, factor in (("qa", 9.), ("task_plan", .5)):
                        tensors[f"{prefix}.{namespace}.{route}_a.weight"] = torch.full((2, 3), factor)
                        tensors[f"{prefix}.{namespace}.{route}_b.weight"] = torch.full((4, 2), factor)
                    expected[prefix + ".weight"] = base + 1.
                save_file(tensors, root / "model.safetensors")
                result = module.merge_dhta_task_plan_weights(root, info)
                self.assertEqual(result["removed_dhta_tensor_count"], 8)
                actual = load_file(root / "model.safetensors")
                self.assertEqual(set(actual), set(expected))
                for key in expected: torch.testing.assert_close(actual[key], expected[key])
                before = (root / "model.safetensors").read_bytes()
                module.merge_dhta_task_plan_weights(root, info)
                self.assertEqual(before, (root / "model.safetensors").read_bytes())
                with self.assertRaises(RuntimeError):
                    module.merge_dhta_task_plan_weights(root, {**info, "alpha": 8.})


if __name__ == "__main__":
    unittest.main()
