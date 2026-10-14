"""Screen tracked release files without printing matched sensitive values.

This is a preventive check, not a proof of absence of all sensitive information.
Review binary assets and publication rights separately. Public attribution and
environment-variable names are not credentials and are intentionally retained.
"""
import argparse
from pathlib import Path
import re
import subprocess


RULES = {
    "personal-home-path": r"/(?:Users|home)/[A-Za-z0-9_.-]+/",
    "mounted-internal-path": r"/mnt/[A-Za-z0-9_.-]+/",
    "internal-host": r"(?:code\.[a-z-]+-inc\.com|[\w.-]+\.(?:aliyun-inc|antfin)\.com)",
    "internal-model-uri": r"model\.amap_common_resource_pool\.",
    "private-key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github-token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b",
    "cloud-access-key": r"\b(?:AKIA[A-Z0-9]{16}|LTAI[A-Za-z0-9]{16,})\b",
    "service-secret": r"\bsk-[A-Za-z0-9_-]{24,}\b",
    "credential-in-url": r"https?://[^\s/:<>]+:[^\s/@<>]+@",
    "private-ip-url": r"https?://(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d+\.\d+(?:\.\d+)?(?=[:/\s])",
}
PATTERNS = {name: re.compile(pattern) for name, pattern in RULES.items()}
BLOCKED_PATHS = re.compile(
    r"(?:^|/)(?:\.env(?:\..+)?|id_(?:rsa|ed25519)(?:\.pub)?|"
    r"AGENT_REPRODUCTION[^/]*|independent-reproduction|scheduler_envfix[^/]*|"
    r"permissions-before\.json|[^/]*\.bundle)(?:/|$)", re.I
)


def inspect_text(text):
    findings = []
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((category, text.count("\n", 0, match.start()) + 1))
    return sorted(set(findings))


def scan(root):
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"],
    ).decode().split("\0")
    findings, binaries, count = [], [], 0
    for name in filter(None, tracked):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue  # Deleted in this working tree.
        count += 1
        if BLOCKED_PATHS.search(name) and not name.endswith(".env.example"):
            findings.append((name, 0, "private-artifact-name"))
        for category, _ in inspect_text(name):
            findings.append((name, 0, category))
        if path.is_symlink():
            findings.append((name, 0, "symlink-needs-review"))
            continue
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
            if "\0" in text:
                raise UnicodeError()
        except UnicodeError:
            binaries.append(name)
            continue
        for category, line in inspect_text(text):
            findings.append((name, line, category))
    return count, findings, binaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    count, findings, binaries = scan(args.root.resolve())
    for name, line, category in findings:
        print(f"{name}:{line}: {category}")
    print(f"Scanned {count} tracked files; {len(findings)} findings; "
          f"{len(binaries)} binary assets require separate review.")
    for name in binaries:
        print(f"Binary review: {name}")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
