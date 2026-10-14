"""Dependency-free checks of frozen renderer/parser/scorer source identities."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SnapshotIntegrityTests(unittest.TestCase):
    def test_frozen_sources(self):
        for name in ("embodied_eval_snapshot", "task_planning_snapshot"):
            root = ROOT / "third_party" / name
            manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
            paths = set()
            for entry in manifest["files"]:
                relative = Path(entry["path"])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                self.assertNotIn(entry["path"], paths)
                paths.add(entry["path"])
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                                 entry["sha256"], f"Frozen source changed: {name}/{relative}")


if __name__ == "__main__":
    unittest.main()
