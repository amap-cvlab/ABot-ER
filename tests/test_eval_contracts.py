import json
from pathlib import Path
import tempfile
import unittest

from abot_er.evaluation import audit_response_file, has_response_text, resolve_contracts, task_index


class EvalContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "datasets.json"

    def manifest(self, task, **kwargs):
        mapping = {"dataset_path": str(self.root), "revision": "synthetic-v1", "expected_count": 2, **kwargs}
        self.path.write_text(json.dumps({"tasks": {task: mapping}}))
        return self.path

    def test_thirteen_tasks_nine_benchmarks(self):
        self.assertEqual(len(task_index()), 13)

    def test_runtime_vabench_contract(self):
        path = self.manifest("vabench_aboter")
        cfg = resolve_contracts(path, ["vabench_aboter"])["vabench_aboter"]
        self.assertEqual(cfg["dataset_kwargs"]["coord_mode"], "normalized_0_1")
        self.assertEqual(cfg["dataset_kwargs"]["expected_points"], 8)
        self.assertIn("decimal between 0 and 1", cfg["dataset_kwargs"]["post_prompt"])

    def test_rejects_prompt_override(self):
        path = self.manifest("vabench_aboter", post_prompt="new prompt")
        with self.assertRaisesRegex(ValueError, "Unsupported overrides"):
            resolve_contracts(path, ["vabench_aboter"])

    def test_requires_explicit_video_root(self):
        with self.assertRaisesRegex(ValueError, "video_dir"):
            resolve_contracts(self.manifest("vsibench"), ["vsibench"])

    def test_requires_revision(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            resolve_contracts(self.manifest("erqa", revision=""), ["erqa"])

    def test_requires_coverage(self):
        with self.assertRaisesRegex(ValueError, "expected_count"):
            resolve_contracts(self.manifest("erqa", expected_count=0), ["erqa"])

    def test_nested_empty_responses_are_not_generation_success(self):
        for value in (None, [], [[]], [[""]], [["  "]], {"error": "failure"}):
            self.assertFalse(has_response_text(value))
        self.assertTrue(has_response_text([["A"]]))

    def test_response_audit_does_not_rewrite_scores(self):
        path = self.root / "samples.json"
        rows = [{"doc_id": 0, "resps": [[""]], "score": 0},
                {"doc_id": 1, "resps": [["A"]], "score": 1},
                {"doc_id": 2, "score": 0}]
        content = "\n".join(json.dumps(row) for row in rows)
        path.write_text(content)
        self.assertEqual(audit_response_file(path),
                         {"records": 3, "empty_responses": 2, "missing_response_field": 1})
        self.assertEqual(path.read_text(), content)


if __name__ == "__main__":
    unittest.main()
