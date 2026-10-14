"""Offline synthetic tests; no model download, inference, judge or simulator."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from abot_er.checkpoint_contract import normalize_checkpoint_config, require_task_conditioned_checkpoint, validate_loading_info, validate_tensor_index
from abot_er.public_data import verify_publication
from abot_er.reproduction import ROOT, prepare_release, render_recipe, sha256, write_json


def config_fixture(prefix="task_conditioned", depth=32, layers=None):
    values = {"enabled": True, "layers": layers or [19, 23, 27, 31], "rank": 16,
              "alpha": 16., "qa_token": "[2D_QA]", "qa_token_id": 100,
              "task_plan_token": "[TASK_PLAN]", "task_plan_token_id": 101,
              "require_route_token": True, "inject_during_init": True,
              "schema_version": prefix + "-qv-v1"}
    text = {"num_hidden_layers": depth, "layer_types": ["full_attention"] * depth,
            **{prefix + "_" + key: value for key, value in values.items()}}
    return {"model_type": "qwen3_5", "text_config": text,
            "architectures": ["ABotERForConditionalGeneration" if prefix == "task_conditioned"
                              else "Qwen3_5ForConditionalGeneration"],
            **{prefix + "_" + key: value for key, value in values.items()}}


class CheckpointContractTests(unittest.TestCase):
    def test_all_published_layer_layouts(self):
        for depth, layers in [(24, [11, 15, 19, 23]), (32, [19, 23, 27, 31]),
                              (64, [51, 55, 59, 63]), (40, [27, 31, 35, 39])]:
            cfg = config_fixture(depth=depth, layers=layers)
            self.assertEqual(normalize_checkpoint_config(cfg), "task_conditioned")
            self.assertEqual(cfg["text_config"]["dhta_layers"], layers)
            self.assertEqual(cfg["abot_residual_namespace"], "task_conditioned")
            self.assertEqual(normalize_checkpoint_config(cfg), "task_conditioned")

    def test_legacy_and_unadapted_base(self):
        self.assertEqual(normalize_checkpoint_config(config_fixture("dhta")), "dhta")
        self.assertIsNone(normalize_checkpoint_config({"model_type": "qwen3_5"}))

    def test_public_evaluated_model_requires_conditioning(self):
        for namespace in ("dhta", "task_conditioned"):
            self.assertEqual(require_task_conditioned_checkpoint(config_fixture(namespace)), namespace)
        for cfg in ({"model_type": "qwen3_5"}, {"dhta_enabled": False}):
            with self.assertRaises(ValueError):
                require_task_conditioned_checkpoint(cfg)

    def test_base_is_allowed_only_as_independent_judge(self):
        from abot_er.evaluation import model_identity
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_json(path / "config.json", {"model_type": "qwen3_5"})
            with self.assertRaisesRegex(ValueError, "task-conditioned"):
                model_identity(path)
            self.assertFalse(model_identity(path, is_judge=True)["dhta_enabled"])

    def test_tp_runtime_rejects_unconditioned_source(self):
        path = ROOT / "third_party/task_planning_snapshot/scripts/dhta_runtime.py"
        spec = importlib.util.spec_from_file_location("tp_no_base_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            write_json(model / "config.json", {"model_type": "qwen3_5"})
            with self.assertRaisesRegex(ValueError, "task-conditioned"):
                module.prepare_dhta_task_plan_model(model)

    def test_ambiguous_architecture_rejected(self):
        with self.assertRaises(ValueError):
            normalize_checkpoint_config({"architectures": ["ABotERForConditionalGeneration"]})

    def test_invalid_contracts_fail_closed(self):
        for field, value in [("enabled", "true"), ("schema_version", "unknown"),
                              ("layers", [32]), ("layers", [19, 19]), ("rank", 0),
                              ("alpha", float("nan")), ("require_route_token", False),
                              ("inject_during_init", False),
                              ("qa_token", "[EQA]"), ("qa_token_id", 101)]:
            cfg = config_fixture()
            cfg["task_conditioned_" + field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                normalize_checkpoint_config(cfg)

    def test_top_text_and_alias_conflicts(self):
        for location, key, value in [("top", "dhta_enabled", False),
                                     ("top", "dhta_rank", 8),
                                     ("text", "task_conditioned_rank", 8)]:
            cfg = config_fixture()
            target = cfg if location == "top" else cfg["text_config"]
            target[key] = value
            with self.assertRaises(ValueError): normalize_checkpoint_config(cfg)

    def test_non_full_attention_rejected(self):
        cfg = config_fixture()
        cfg["text_config"]["layer_types"][19] = "linear_attention"
        with self.assertRaises(ValueError): normalize_checkpoint_config(cfg)

    def test_short_layer_types_rejected(self):
        cfg = config_fixture()
        cfg["text_config"]["layer_types"] = ["full_attention"]
        with self.assertRaises(ValueError): normalize_checkpoint_config(cfg)

    def test_loading_errors_both_namespaces(self):
        for namespace in ("dhta", "task_conditioned"):
            for kind in ("missing_keys", "unexpected_keys", "mismatched_keys"):
                with self.assertRaises(RuntimeError):
                    validate_loading_info({kind: [f"q_proj.{namespace}.qa_a.weight"]})
        validate_loading_info({"missing_keys": []})

    def test_index_prevents_residual_omission_or_partial_weights(self):
        cfg = config_fixture()
        weights = {}
        for layer in cfg["task_conditioned_layers"]:
            for projection in ("q_proj", "v_proj"):
                prefix = f"model.language_model.layers.{layer}.self_attn.{projection}"
                weights[prefix + ".weight"] = "fixture.safetensors"
                for route in ("qa", "task_plan"):
                    for side in ("a", "b"):
                        weights[f"{prefix}.task_conditioned.{route}_{side}.weight"] = "fixture.safetensors"
        validate_tensor_index(cfg, weights)
        with self.assertRaises(ValueError): validate_tensor_index({}, weights)
        weights.pop(next(key for key in weights if ".task_conditioned." in key))
        with self.assertRaises(ValueError): validate_tensor_index(cfg, weights)

    def test_tp_namespace_and_architecture(self):
        path = ROOT / "third_party/task_planning_snapshot/scripts/dhta_runtime.py"
        spec = importlib.util.spec_from_file_location("tp_contract_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # stdlib only; no model operation
        for namespace in ("dhta", "task_conditioned"):
            cfg = config_fixture(namespace)
            normalize_checkpoint_config(cfg)
            info = module._validate_config(cfg)
            self.assertEqual(info["namespace"], namespace)
            keys = module._expected_dhta_keys(info["layers"], namespace)
            self.assertEqual(len(keys), 32)
            self.assertTrue(all(f".{namespace}." in key for key in keys))

    def test_route_injection_unchanged_and_exclusive(self):
        # Extract this pure function without importing torch/the evaluator.
        path = ROOT / "third_party/embodied_eval_snapshot/embodied_eval/models/dhta.py"
        tree = ast.parse(path.read_text())
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "inject_dhta_route")
        scope = {"DHTA_QA_TOKEN": "[2D_QA]", "DHTA_TASK_PLAN_TOKEN": "[TASK_PLAN]"}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        inject = scope["inject_dhta_route"]
        for task, token in [("2d_qa", "[2D_QA]"), ("task_plan", "[TASK_PLAN]")]:
            result = inject([{"role": "user", "content": "<image> Question"}], task)
            self.assertEqual(result[0]["content"], token + "\n<image> Question")
            self.assertEqual(inject(deepcopy(result), task), result)
        with self.assertRaises(ValueError):
            inject([{"role": "user", "content": "[TASK_PLAN] Question"}], "2d_qa")


class PublicDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "download"
        stage = self.data / "data/stage_a_pretrain"
        stage.mkdir(parents=True)
        self.shard = stage / "part-00000-of-00001.jsonl"
        row = {"task_family": "2d_qa", "images": ["assets/a.png"],
               "conversations": [{"from": "human", "value": "<image> Question"}, {"from": "gpt", "value": "A"}]}
        self.shard.write_text(json.dumps(row) + "\n")
        sums = self.data / "SHA256SUMS.txt"
        sums.write_text(f"{sha256(self.shard)}  data/stage_a_pretrain/{self.shard.name}\n")
        release = self.data / "RELEASE_MANIFEST.json"
        write_json(release, {"release_version": "synthetic-test", "stages": {
            "stage_a_pretrain": {"physical_records": 1, "shards": 1}}})
        self.pointer = self.root / "pointer.json"
        write_json(self.pointer, {"repository": "synthetic/fixture", "revision": "fixture-revision",
            "release_version": "synthetic-test", "checksums_sha256": sha256(sums),
            "release_manifest_sha256": sha256(release),
            "stages": {"stage_a_pretrain": {"physical_records": 1, "shards": 1}}})

    def verify(self):
        return verify_publication(self.data, "stage_a_pretrain", self.pointer)

    def test_verify_and_prepare_preserve_labels(self):
        before = self.shard.read_bytes()
        meta = self.verify()
        manifest = self.root / "verified.json"
        write_json(manifest, meta)
        media = self.root / "assets"
        media.mkdir()
        (media / "a.png").write_bytes(b"fixture")
        mapping = self.root / "mapping.json"
        write_json(mapping, {"assets": str(media)})
        prepared = prepare_release(self.shard.parent, manifest, self.root / "prepared", mapping)
        self.assertEqual(self.shard.read_bytes(), before)
        self.assertEqual(prepared["publication"], meta["publication"])
        actual = json.loads((self.root / "prepared/shards" / self.shard.name).read_text())
        self.assertEqual(actual["conversations"], json.loads(before)["conversations"])
        self.assertEqual(actual["task_family"], "2d_qa")

    def test_changed_checksum_list_rejected(self):
        (self.data / "SHA256SUMS.txt").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "identity"): self.verify()

    def test_changed_shard_rejected(self):
        row = json.loads(self.shard.read_text())
        row["conversations"][-1]["value"] = "B"
        self.shard.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"): self.verify()

    def test_extra_shard_rejected(self):
        (self.shard.parent / "extra.jsonl").write_bytes(self.shard.read_bytes())
        with self.assertRaises(ValueError): self.verify()

    def test_symlink_rejected(self):
        content = self.shard.read_bytes()
        self.shard.unlink()
        outside = self.root / "outside.jsonl"
        outside.write_bytes(content)
        self.shard.symlink_to(outside)
        with self.assertRaises(ValueError): self.verify()

    def test_count_mismatch_rejected(self):
        pointer = json.loads(self.pointer.read_text())
        pointer["stages"]["stage_a_pretrain"]["physical_records"] = 2
        write_json(self.pointer, pointer)
        with self.assertRaises(ValueError): self.verify()

    def test_public_recipe_requires_explicit_scope(self):
        pinned = json.loads((ROOT / "data/modelscope_release.json").read_text())
        meta = json.loads((ROOT / "data/manifests/pretrain_vabxml8.json").read_text())
        meta.update(release_id=f"ABot-ER-{pinned['release_version']}-stage_a_pretrain",
                    publication={key: pinned[key] for key in ("repository", "revision", "checksums_sha256")},
                    publication_stage="stage_a_pretrain", physical_records=462385)
        meta["shards"][0]["rows"] -= 5
        model, data = self.root / "model", self.root / "prepared"
        write_json(model / "config.json", {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 32}})
        write_json(data / "manifest.json", meta)
        args = (ROOT / "configs/pretrain/q35_9b_dhta.yaml", model, data, self.root / "run", 32)
        with self.assertRaises(ValueError): render_recipe(*args)
        cfg, summary = render_recipe(*args, public_data=True)
        self.assertEqual(summary["data_scope"], "modelscope_public_export")
        self.assertEqual(cfg["max_steps"], 3600)
        meta["publication"]["revision"] = "wrong"
        write_json(data / "manifest.json", meta)
        with self.assertRaises(ValueError): render_recipe(*args, public_data=True)


class DisabledLegacyTests(unittest.TestCase):
    def test_api_ui_fail_before_initialization(self):
        base = ROOT / "third_party/tuningfactory/src"
        for relative, functions in [("llmtuner/api/app.py", ["create_app"]),
                                     ("llmtuner/webui/interface.py", ["create_ui", "create_web_demo"])]:
            tree = ast.parse((base / relative).read_text())
            for name in functions:
                node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
                self.assertIsInstance(node.body[0], ast.Raise)
        source = (base / "llmtuner/model/model_utils/valuehead.py").read_text()
        self.assertNotIn("torch.load(", source)


if __name__ == "__main__":
    unittest.main()
