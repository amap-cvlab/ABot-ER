import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from abot_er.artifacts import checked_path, file_matches, validate_index


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.index = {"schema_version": 1, "status": "published", "repo_type": "dataset",
                      "repo_id": "test/fixture", "revision": "a" * 40,
                      "files": [{"path": "shards/part.jsonl", "size_bytes": 3,
                                 "sha256": hashlib.sha256(b"abc").hexdigest()}]}

    def test_pinned_allowlist(self):
        self.assertEqual(len(validate_index(self.index)), 1)

    def test_reject_unpublished(self):
        self.index["status"] = "pending"
        with self.assertRaises(ValueError): validate_index(self.index)

    def test_reject_mutable_revision(self):
        for revision in ("main", "v1", "abcdef0", ""):
            self.index["revision"] = revision
            with self.assertRaises(ValueError): validate_index(self.index)

    def test_unsafe_paths(self):
        for path in ("../secret", "/tmp/file", "a/../b", ".env", "a\\b", "a//b", "a/./b",
                     "abot_download_receipt.json", "a\nb"):
            self.index["files"][0]["path"] = path
            with self.assertRaises(ValueError): validate_index(self.index)

    def test_duplicate(self):
        self.index["files"].append(copy.deepcopy(self.index["files"][0]))
        with self.assertRaises(ValueError): validate_index(self.index)

    def test_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "shards").symlink_to(root.parent)
            with self.assertRaises(ValueError): checked_path(root, "shards/part.jsonl")

    def test_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file"
            path.write_bytes(b"abc")
            self.assertTrue(file_matches(path, self.index["files"][0]))
            path.write_bytes(b"abd")
            self.assertFalse(file_matches(path, self.index["files"][0]))


if __name__ == "__main__": unittest.main()
