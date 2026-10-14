'''
VABench-P point-selection evaluation process functions.

Dataset: IffYuan/VABench-P (300 samples, test split)
  Fields: id (str), problem (str), image (PIL.Image), mask (PIL.Image)

Task: model outputs 8 normalized points (0-1 or 0-1000) or absolute pixel coords
  wrapped in <Answer><point>[[x1,y1],...,[x8,y8]]</point></Answer>.
  Accuracy = fraction of points that land inside the ground-truth mask.
'''

import ast
import re

import pandas as pd
from loguru import logger as eval_logger
from PIL import Image

from embodied_eval.pointing.coords import (
    ensure_pil_mask,
    mask_accuracy_from_points,
)
from embodied_eval.pointing.point_utils import omni_decode_points


def vabench_doc_to_visual(doc, dataset_kwargs=None):
    image = doc.get("image")
    if image is None:
        return []
    if not isinstance(image, Image.Image):
        return []
    if image.mode != "RGB":
        image = image.convert("RGB")
    return [image]


def vabench_doc_to_text(doc, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    instruction = doc.get("problem", doc.get("question", ""))
    pre = dataset_kwargs.get("pre_prompt", "")
    if pre:
        instruction = f"{pre} {instruction}"
    post = dataset_kwargs.get("post_prompt", "")
    if post:
        instruction = f"{instruction} {post}"
    return instruction


def vabench_process_results(doc, results, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    raw_prediction = results[0] if results else ""
    doc["prediction"] = raw_prediction

    points = _extract_vabench_points(raw_prediction)

    image = doc.get("image")
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        width, height = 640, 480

    coord_mode = dataset_kwargs.get("coord_mode")
    abs_points = vabench_points_to_absolute(points, width, height, coord_mode)

    points_mode = dataset_kwargs.get("points_mode", "adaptive")

    if points_mode == "fixed":
        expected_points = dataset_kwargs.get("expected_points")
        if expected_points is None:
            raise ValueError("expected_points is required when points_mode is 'fixed'")
        expected_points = int(expected_points)
    elif points_mode == "adaptive":
        if "expected_points" in dataset_kwargs:
            raise ValueError("expected_points must NOT be set when points_mode is 'adaptive'")
    else:
        raise ValueError(f"Unknown points_mode: {points_mode}")

    if points_mode == "fixed":
        eval_points = abs_points[:expected_points]
    else:
        eval_points = abs_points
        expected_points = len(abs_points) if len(abs_points) > 0 else 1

    metric_mode = dataset_kwargs.get("metric_mode")
    if metric_mode is None:
        raise ValueError("metric_mode is required in dataset_kwargs")
    mask = doc.get("mask")

    if metric_mode == "mask":
        if mask is not None and len(eval_points) > 0:
            pil_mask = ensure_pil_mask(mask, width, height)
            _, points_in_mask, _ = mask_accuracy_from_points(eval_points, pil_mask)
            total_points = expected_points
            acc = points_in_mask / total_points
            scored_count = points_in_mask
        else:
            acc = 0.0
            points_in_mask = 0
            total_points = expected_points
            scored_count = 0
    else:
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    result_dict = {
        "target": doc.get("id", ""),
        "question_type": doc.get("question_type", "vabench"),
        "accuracy": acc,
        "processed_points": eval_points,
        "parsed_points": len(abs_points),
        "points_in_mask": points_in_mask,
        "total_points": total_points,
        "scored_points": scored_count,
    }

    doc["accuracy"] = acc
    return result_dict


def vabench_points_to_absolute(points, width, height, coord_mode):
    """Convert parsed VABench points to absolute image coordinates.

    ``normalized`` is retained as the legacy name for 0-1000 normalized
    coordinates.  ``normalized_0_1`` is explicit so checkpoints trained with
    fractional coordinates can be evaluated without changing official/default
    task behavior.
    """
    if coord_mode == "pixel":
        x_scale = y_scale = 1.0
    elif coord_mode in {"normalized", "normalized_0_1000"}:
        x_scale = width / 1000.0
        y_scale = height / 1000.0
    elif coord_mode == "normalized_0_1":
        x_scale = float(width)
        y_scale = float(height)
    else:
        raise ValueError(f"Unknown or missing coord_mode: {coord_mode}")

    return [
        [int(round(point[0] * x_scale)), int(round(point[1] * y_scale))]
        for point in points
    ]


def vabench_aggregate_results(results):
    if not results:
        return {}

    results_df = pd.DataFrame(results)
    output = {}

    if "question_type" in results_df.columns:
        for qtype, indices in results_df.groupby("question_type").groups.items():
            per_type = results_df.iloc[indices]
            if "accuracy" in per_type.columns:
                output[f"{qtype}_accuracy"] = per_type["accuracy"].mean()

    if "accuracy" in results_df.columns:
        overall = results_df["accuracy"].mean()
        output["accuracy_average"] = overall
        output["overall"] = overall

    eval_logger.info(f"VABench-P evaluation results: {output}")
    return output


def _extract_vabench_points(text):
    if not text or not str(text).strip():
        return []

    text = str(text).strip()

    text = re.sub(
        r"^(.*?</think>|<think>.*?</think>)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        inner = answer_match.group(1).strip()
    else:
        inner = text

    point_match = re.search(
        r"<point>(.*?)</point>",
        inner,
        re.DOTALL | re.IGNORECASE,
    )
    if point_match:
        inner = point_match.group(1).strip()

    bracket_match = re.search(r"\[\[.*\]\]", inner, re.DOTALL)
    if bracket_match:
        raw_points = bracket_match.group(0)
    else:
        raw_match = re.search(r"\[\[.*\]\]", text, re.DOTALL)
        if raw_match:
            raw_points = raw_match.group(0)
        else:
            return omni_decode_points(text)

    points = _try_parse_points_array(raw_points)
    if points is not None:
        return points

    return omni_decode_points(text)


def _try_parse_points_array(raw):
    try:
        points_list = ast.literal_eval(raw)
        if isinstance(points_list, list) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in points_list
        ):
            return [[float(p[0]), float(p[1])] for p in points_list]
    except (ValueError, SyntaxError):
        pass

    point_pairs = re.findall(r"\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]", raw)
    if point_pairs:
        return [[float(x), float(y)] for x, y in point_pairs]

    return None
