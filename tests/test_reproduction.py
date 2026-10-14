import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from abot_er.reproduction import ROOT, prepare_release, rank_capacity, relocate_media, render_recipe, verify_release, write_json


class ReproductionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.media = self.root / "media"
        self.media.mkdir()
        (self.media / "image.jpg").write_bytes(b"synthetic fixture, not a real image")
        self.record = {"images": ["/old/image.jpg"], "task_family": "2d_qa",
                       "conversations": [{"from": "human", "value": "<image> Question"},
                                         {"from": "gpt", "value": "A"}]}
        raw = (json.dumps(self.record) + "\n").encode()
        (self.source / "part-000.jsonl").write_bytes(raw)
        self.manifest = self.root / "manifest.json"
        self.meta = {"release_id": "test", "dataset_key": "test", "physical_records": 1,
                     "ordered_sha256": hashlib.sha256(raw).hexdigest(), "shards": [
                         {"file": "part-000.jsonl", "rows": 1, "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest()}]}
        write_json(self.manifest, self.meta)

    def test_verify(self):
        self.assertEqual(verify_release(self.source, self.manifest)["physical_records"], 1)

    def test_corrupt(self):
        (self.source / "part-000.jsonl").write_text('{}\n')
        with self.assertRaises(ValueError):
            verify_release(self.source, self.manifest)

    def test_extra_shard(self):
        (self.source / "unexpected.jsonl").write_text('{}\n')
        with self.assertRaises(ValueError):
            verify_release(self.source, self.manifest)

    def test_duplicate_manifest(self):
        self.meta["shards"] *= 2
        write_json(self.manifest, self.meta)
        with self.assertRaises(ValueError):
            verify_release(self.source, self.manifest)

    def test_missing_media(self):
        with self.assertRaises(FileNotFoundError):
            relocate_media("/old/missing.jpg", {"/old": str(self.media)})

    def test_escape(self):
        with self.assertRaises(ValueError):
            relocate_media("/old/../escape", {"/old": str(self.media)})

    def test_unmapped(self):
        with self.assertRaises(ValueError):
            relocate_media("/unknown/image.jpg", {"/old": str(self.media)})

    def test_prepare_preserves_labels(self):
        mapping = self.root / "roots.json"
        write_json(mapping, {"/old": str(self.media)})
        output = self.root / "prepared"
        result = prepare_release(self.source, self.manifest, output, mapping)
        actual = json.loads((output / "shards/part-000.jsonl").read_text())
        self.assertEqual(actual["conversations"], self.record["conversations"])
        self.assertEqual(actual["images"], [str((self.media / "image.jpg").resolve())])
        self.assertEqual(result["source_ordered_sha256"], self.meta["ordered_sha256"])
        verify_release(output / "shards", output / "manifest.json")
        with self.assertRaises(FileExistsError):
            prepare_release(self.source, self.manifest, output, mapping)

    def test_capacity(self):
        meta = {"shards": [{"file": f"{i:03}.jsonl", "rows": 820} for i in range(128)]}
        self.assertEqual(rank_capacity(meta, 32, 4), [820] * 32)
        with self.assertRaises(ValueError):
            rank_capacity(meta, 48, 4)

    def test_historical_hash_method(self):
        raw = (self.source / "part-000.jsonl").read_bytes().rstrip(b"\r\n")
        self.meta["hash_method"] = "round_robin_length_prefix_v1"
        self.meta["ordered_sha256"] = hashlib.sha256(len(raw).to_bytes(8, "big") + raw).hexdigest()
        write_json(self.manifest, self.meta)
        verify_release(self.source, self.manifest)

    def recipe_fixture(self):
        model = self.root / "model"
        model.mkdir()
        write_json(model / "config.json", {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 32}})
        metadata = json.loads((ROOT / "data/manifests/pretrain_vabxml8.json").read_text())
        prepared = self.root / "prepared"
        write_json(prepared / "manifest.json", metadata)
        return model, prepared

    def test_frozen_recipe(self):
        model, prepared = self.recipe_fixture()
        cfg, summary = render_recipe(ROOT / "configs/pretrain/q35_9b_dhta.yaml", model, prepared, self.root / "out", 32)
        self.assertEqual(cfg["gradient_accumulation_steps"], 4)
        self.assertEqual(summary["global_batch"], 128)
        self.assertGreater(summary["tail_steps"], 0)

    def test_wrong_architecture(self):
        model, prepared = self.recipe_fixture()
        write_json(model / "config.json", {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 24}})
        with self.assertRaises(ValueError):
            render_recipe(ROOT / "configs/pretrain/q35_9b_dhta.yaml", model, prepared, self.root / "out", 32)

    def test_recipe_cannot_disable_conditioning(self):
        import yaml
        model, prepared = self.recipe_fixture()
        spec = yaml.safe_load((ROOT / "configs/pretrain/q35_9b_dhta.yaml").read_text())
        spec["training"]["enable_dhta"] = False
        recipe = self.root / "invalid.yaml"
        recipe.write_text(yaml.safe_dump(spec))
        with self.assertRaisesRegex(ValueError, "require task-conditioned"):
            render_recipe(recipe, model, prepared, self.root / "out", 32)

    def test_wrong_release(self):
        model, prepared = self.recipe_fixture()
        write_json(prepared / "manifest.json", self.meta)
        with self.assertRaises(ValueError):
            render_recipe(ROOT / "configs/pretrain/q35_9b_dhta.yaml", model, prepared, self.root / "out", 32)

    def test_invalid_gpu_count(self):
        model, prepared = self.recipe_fixture()
        with self.assertRaises(ValueError):
            render_recipe(ROOT / "configs/pretrain/q35_9b_dhta.yaml", model, prepared, self.root / "out", 48)


if __name__ == "__main__":
    unittest.main()
