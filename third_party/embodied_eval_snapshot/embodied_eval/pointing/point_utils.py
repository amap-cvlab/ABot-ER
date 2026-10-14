"""Parse 2D points from VLM outputs and check against masks (self-contained)."""

import ast
import re
from typing import Any, List, Tuple

import numpy as np
from PIL import Image


def omni_decode_points(output: str) -> List[List[float]]:
    """Parse [x, y] coordinates from diverse VLM output formats."""
    if not isinstance(output, str) or not output.strip():
        return []

    if "<point" in output.lower():
        points = _extract_from_xml_attributes(output)
        if points:
            return points

    text = _preprocess_text(output)

    try:
        clean_text = re.sub(r"^[a-zA-Z0-9_\s]+:\s*", "", text)
        data = ast.literal_eval(clean_text)
        points = _parse_structured_data(data)
        if points:
            return points
    except (ValueError, SyntaxError, MemoryError):
        pass

    return _extract_points_by_regex(text)


def _preprocess_text(text: str) -> str:
    text = re.sub(r"```(?:json|python|html)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL)
    tag_match = re.search(
        r"<(?:point|points)>(.*?)</(?:point|points)>", text, re.DOTALL | re.IGNORECASE
    )
    if tag_match:
        text = tag_match.group(1)
    return text.strip()


def _parse_structured_data(data: Any) -> List[List[float]]:
    points = []
    if isinstance(data, dict):
        for key in ("point_2d", "points", "point", "coordinates"):
            if key in data:
                return _parse_structured_data(data[key])
    elif isinstance(data, (list, tuple)):
        if not data:
            return []
        if len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
            return [[float(data[0]), float(data[1])]]
        for item in data:
            extracted = _parse_structured_data(item)
            if extracted:
                points.extend(extracted)
    return points


def _extract_from_xml_attributes(text: str) -> List[List[float]]:
    all_points = []
    for pattern in (
        r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)",
        r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)",
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
        r"(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})",
    ):
        for match in re.finditer(pattern, text):
            try:
                if pattern.endswith(r"([0-9]{3})"):
                    point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
                else:
                    point = [float(match.group(i)) for i in range(1, 3)]
                all_points.append(point)
            except ValueError:
                pass
    return all_points


def _extract_points_by_regex(text: str) -> List[List[float]]:
    points = []
    bracket_pattern = r"[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]"
    matches = re.findall(bracket_pattern, text)
    if matches:
        for m in matches:
            points.append([float(m[0]), float(m[1])])
    else:
        for m in re.findall(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", text):
            points.append([float(m[0]), float(m[1])])
    return points


def check_points_in_mask(points: List[List[float]], mask: Image.Image) -> Tuple[int, int]:
    """Return (points inside foreground, total points). Coordinates are absolute pixels."""
    if not points or mask is None:
        return 0, 0

    if mask.mode != "L":
        mask = mask.convert("L")

    width, height = mask.size
    mask_array = np.array(mask)
    points_in_mask = 0

    for point in points:
        x_pixel = int(round(point[0]))
        y_pixel = int(round(point[1]))
        if 0 <= x_pixel < width and 0 <= y_pixel < height:
            if mask_array[y_pixel, x_pixel] > 0:
                points_in_mask += 1

    return points_in_mask, len(points)
