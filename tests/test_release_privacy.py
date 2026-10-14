import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "privacy", Path(__file__).resolve().parents[1] / "scripts/check_release_privacy.py"
)
privacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy)


class ReleasePrivacyTests(unittest.TestCase):
    def test_paths_and_keys(self):
        examples = [
            "/" + "Users/example/project/file",
            "/" + "mnt/shared/project/file",
            "model." + "amap_common_resource_pool.example",
            "ghp_" + "X" * 36,
            "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
            "https://" + "user:secret@example.invalid/",
            "http://" + "10.20.30.40/",
        ]
        for text in examples:
            self.assertTrue(privacy.inspect_text(text))

    def test_public_examples_and_versions(self):
        for text in ["/path/to/model", "os.getenv('API_KEY')", "libcurand=10.3.7.37",
                     "https://www.modelscope.cn/models/amap_cvlab/ABot-ER-9B",
                     "http://127.0.0.1:8000", "Copyright 2024 Example Authors"]:
            self.assertEqual([], privacy.inspect_text(text))

    def test_only_tracked_files_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", tmp], check=True)
            (root / "README.md").write_text("Public example\n")
            (root / ".env").write_text("LOCAL_ONLY=example\n")
            (root / "image.png").write_bytes(b"\x89PNG\x00")
            subprocess.run(["git", "-C", tmp, "add", "README.md", "image.png"], check=True)
            count, findings, binaries = privacy.scan(root)
            self.assertEqual(2, count)
            self.assertEqual([], findings)
            self.assertEqual(["image.png"], binaries)

    def test_binary_and_private_filenames(self):
        self.assertTrue(privacy.BLOCKED_PATHS.search("archive/" + "independent-reproduction/report.md"))
        self.assertTrue(privacy.BLOCKED_PATHS.search(".env"))


if __name__ == "__main__":
    unittest.main()
