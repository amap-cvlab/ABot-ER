"""Synthetic regression tests against byte-identical upstream scoring functions.

No benchmark samples, labels, model service, or judge service are used.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "third_party/embodied_eval_snapshot"
sys.path.insert(0, str(SNAPSHOT))

from PIL import Image
def load_processor(name):
    # Load the unchanged processor directly: importing the task registry would
    # unnecessarily require dataset downloads and the full evaluation runtime.
    path = SNAPSHOT / "embodied_eval/tasks" / name / "process.py"
    spec = importlib.util.spec_from_file_location(f"point_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vabench = load_processor("vabench")
vabench_process_results, vabench_doc_to_text = vabench.vabench_process_results, vabench.vabench_doc_to_text
where2place_process_results = load_processor("where2place").where2place_process_results


class PointScorerSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = json.loads((ROOT / "configs/eval/vabench_runtime_contract.json").read_text())["dataset_kwargs"]

    def doc(self):
        return {"id": "synthetic", "problem": "Select safe points.",
                "image": Image.new("RGB", (100, 100)),
                "mask": Image.new("L", (100, 100), 255)}

    def va(self, points):
        answer = "<Answer><point>" + json.dumps(points) + "</point></Answer>"
        return vabench_process_results(self.doc(), [answer], self.kwargs)

    def test_source_integrity(self):
        manifest = json.loads((SNAPSHOT / "SOURCE_MANIFEST.json").read_text())
        for entry in manifest["files"]:
            self.assertEqual(hashlib.sha256((SNAPSHOT / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])

    def test_vabench_eight_repeated_valid_points(self):
        self.assertEqual(self.va([[0.5, 0.5]] * 8)["accuracy"], 1.0)

    def test_vabench_one_point_is_one_eighth(self):
        self.assertEqual(self.va([[0.5, 0.5]])["accuracy"], 0.125)

    def test_vabench_partial_credit(self):
        self.assertEqual(self.va([[0.5, 0.5]] * 4 + [[1.1, 1.1]] * 4)["accuracy"], 0.5)

    def test_vabench_empty(self):
        self.assertEqual(self.va([])["accuracy"], 0.0)

    def test_coordinate_one_maps_to_width_not_width_minus_one(self):
        result = self.va([[1.0, 1.0]] * 8)
        self.assertEqual(result["processed_points"][0], [100, 100])
        self.assertEqual(result["accuracy"], 0.0)

    def test_runtime_prompt_not_raw_yaml(self):
        prompt = vabench_doc_to_text(self.doc(), self.kwargs)
        self.assertIn("decimal between 0 and 1", prompt)
        self.assertNotIn("0-1000", prompt)
        self.assertEqual(self.kwargs["expected_points"], 8)

    def w2p(self, answer):
        return where2place_process_results(self.doc(), [answer], {"backbone": "gpt", "use_kit_prompt": False})

    def test_w2p_one_point_full_credit(self):
        self.assertEqual(self.w2p("[(0.5, 0.5)]")["accuracy"], 1.0)

    def test_w2p_actual_point_denominator(self):
        self.assertEqual(self.w2p("[(0.5, 0.5), (1.1, 1.1)]")["accuracy"], 0.5)

    def test_w2p_empty_zero(self):
        self.assertEqual(self.w2p("")["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
