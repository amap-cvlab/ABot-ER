"""Record installed package versions without collecting environment variables or credentials."""
import importlib.metadata
import json
import platform
import sys

packages = ["torch", "torchvision", "transformers", "tokenizers", "datasets", "pyarrow",
            "deepspeed", "accelerate", "trl", "peft", "Pillow", "numpy", "decord", "av",
            "safetensors", "flash-linear-attention", "fla-core", "causal-conv1d"]
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": sys.version, "platform": platform.platform(), "packages": versions}, indent=2))
