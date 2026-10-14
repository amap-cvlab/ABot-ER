"""Backbone-aware coordinate conversion for pointing benchmarks."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from embodied_eval.pointing.point_utils import check_points_in_mask, omni_decode_points


def decode_points_to_absolute(
    text: str,
    width: int,
    height: int,
    backbone: Optional[str] = None,
) -> List[List[int]]:
    """Parse model output and convert to absolute pixel coordinates."""
    if not text or not str(text).strip():
        return []

    points = omni_decode_points(str(text).strip())
    if not points:
        return []

    backbone = (backbone or "gpt").lower().replace("qwen2.5", "qwen2_5")
    abs_points: List[List[int]] = []

    if backbone in ("qwen2.5", "qwen2_5", "mimo"):
        for point in points:
            abs_points.append([int(round(point[0])), int(round(point[1]))])
    elif backbone in ("qwen3", "gemini-2.5"):
        for point in points:
            abs_points.append([
                int(round(point[0] / 1000.0 * width)),
                int(round(point[1] / 1000.0 * height)),
            ])
    elif backbone == "gemini_robotics":
        for point in points:
            abs_points.append([
                int(round(point[1] / 1000.0 * width)),
                int(round(point[0] / 1000.0 * height)),
            ])
    elif backbone == "molmo":
        for point in points:
            abs_points.append([
                int(round(point[0] / 100.0 * width)),
                int(round(point[1] / 100.0 * height)),
            ])
    elif backbone in ("gpt", "pelican", "internvl", "magma"):
        for point in points:
            abs_points.append([
                int(round(point[0] * width)),
                int(round(point[1] * height)),
            ])
    else:
        raise ValueError(f"Unsupported backbone for pointing: {backbone}")

    return abs_points


def mask_accuracy_from_points(
    abs_points: List[List[int]],
    mask: Image.Image,
) -> Tuple[float, int, int]:
    points_in_mask, total_points = check_points_in_mask(abs_points, mask)
    if total_points == 0:
        return 0.0, 0, 0
    return points_in_mask / total_points, points_in_mask, total_points


def ensure_pil_mask(mask, width: int, height: int) -> Image.Image:
    if isinstance(mask, Image.Image):
        pil_mask = mask.convert("L")
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.max() > 1:
            arr = (arr > 127).astype("uint8") * 255
        else:
            arr = (arr > 0.5).astype("uint8") * 255
        pil_mask = Image.fromarray(arr.astype("uint8"), mode="L")

    if pil_mask.size != (width, height):
        pil_mask = pil_mask.resize((width, height), Image.NEAREST)
    return pil_mask
