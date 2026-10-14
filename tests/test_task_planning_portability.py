import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from abot_er.task_planning import LINKS, SNAPSHOT, prepare, serve, validate_assets
from abot_er.reproduction import sha256


class TaskPlanningPortabilityTest(unittest.TestCase):
    def manifest(self, root):
        paths = {}
        for key in set(LINKS) | {"ai2thor_releases"}:
            directory = root / key
            directory.mkdir()
            paths[key] = str(directory)
        return {"revision": "fixture-only-v1", "paths": paths}

    def test_requires_revision(self):
        with self.assertRaises(ValueError): validate_assets({"paths": {}})

    def test_requires_all_assets(self):
        with self.assertRaises(ValueError): validate_assets({"revision": "x", "paths": {}})

    def test_server_pins_dhta_thinking_and_refuses_mutated_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = root / "model.safetensors"
            weight.write_bytes(b"synthetic fixture, not weights")
            index = root / "server.json"
            index.write_text(json.dumps({"model_path": str(root), "route": "[TASK_PLAN]",
                                        "dhta_enabled": True, "prepared_sha256": {weight.name: sha256(weight)}}))
            args = argparse.Namespace(server_manifest=str(index), tensor_parallel_size=8,
                                      gpu_memory_utilization=.9, port=8000, served_model_name="abot-er-tp", dry_run=True)
            with patch("builtins.print"):
                result = serve(args)
            command = result["command"]
            value = command[command.index("--default-chat-template-kwargs") + 1]
            self.assertEqual(json.loads(value), {"enable_thinking": True})
            weight.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"): serve(args)

    def test_server_rejects_unconditioned_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = root / "model.safetensors"
            weight.write_bytes(b"fixture")
            index = root / "server.json"
            index.write_text(json.dumps({"model_path": str(root), "route": "[TASK_PLAN]",
                                        "dhta_enabled": False, "prepared_sha256": {weight.name: sha256(weight)}}))
            args = argparse.Namespace(server_manifest=str(index), tensor_parallel_size=8,
                                      gpu_memory_utilization=.9, port=8000, served_model_name="abot-er-tp", dry_run=True)
            with self.assertRaisesRegex(ValueError, "Verified TP-only"):
                serve(args)

    def test_staging_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self.manifest(root)
            index = root / "assets.json"
            index.write_text(json.dumps(manifest))
            out = root / "runtime"
            with patch("builtins.print"):
                prepare(argparse.Namespace(assets=str(index), output=str(out)))
            config = json.loads((out / "runtime.json").read_text())
            self.assertNotIn("HOME", config["environment"])
            for key, relative in LINKS.items():
                self.assertEqual((out / "source" / relative).resolve(), Path(manifest["paths"][key]))
            with self.assertRaises(FileExistsError):
                prepare(argparse.Namespace(assets=str(index), output=str(out)))

    def test_refuses_nested_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = self.manifest(root)
            index = root / "assets.json"
            index.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                prepare(argparse.Namespace(assets=str(index), output=str(root / "et_benchmarks" / "runtime")))

    def test_no_home_delete_and_no_default_pruning(self):
        source = (SNAPSHOT / "scripts/run_full_eval_local.py").read_text()
        self.assertNotIn("export HOME=", source)
        self.assertNotIn('rm -rf \\"$HOME', source)
        pipeline = (SNAPSHOT / "scripts/run_tp_full_eval_pipeline.py").read_text()
        self.assertIn("if args.prune_artifacts and args.max_jobs is None", pipeline)

    def test_per_shard_cache_without_home_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            releases = root / "releases"
            releases.mkdir()
            sys.path.insert(0, str(SNAPSHOT / "scripts"))
            spec = importlib.util.spec_from_file_location("tp_portability_fixture", SNAPSHOT / "scripts/run_full_eval_local.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch.object(module, "AI2THOR_RELEASES_ROOT", releases):
                prefix = module.alfred_ai2thor_home_prefix(root / "scratch", "sample")
                self.assertIn("ABOT_AI2THOR_BASE_DIR=", prefix)
                self.assertNotIn("HOME", prefix)
                with self.assertRaises(ValueError): module.alfred_ai2thor_home_prefix(root, "../escape")


if __name__ == "__main__": unittest.main()
