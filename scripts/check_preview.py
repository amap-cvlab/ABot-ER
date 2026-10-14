"""Check local Markdown links and JSON syntax in the release."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
errors = []
for path in root.rglob("*.json"):
    if ".git" not in path.parts:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
for path in root.rglob("*.md"):
    if ".git" in path.parts or "third_party" in path.parts:
        continue
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        if not (path.parent / target.split("#", 1)[0]).exists():
            errors.append(f"{path.relative_to(root)}: missing link {target}")
if errors:
    raise SystemExit("\n".join(errors))
print("Release documentation checks passed (links and JSON only).")
