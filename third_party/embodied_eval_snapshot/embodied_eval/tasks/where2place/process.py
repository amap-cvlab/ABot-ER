'''
Where2Place task metrics — pointing parse/coords in embodied_eval.pointing.

Fixes applied vs the original utils/process_back.py:
  Bug1/2/3 JSON format: handle both "point" and "point_2d" keys (Qwen3-VL / R1.5 native)
           removed the erroneous `points = []` that wiped parsed results
           if json_match: block now lives inside the else branch (indentation fix)
  Bug4     pixel coords: backbone-aware scaling via decode_points_to_absolute (0-1000 -> px)
  Bug5     bbox_order: where2place-bbox.yaml declares bbox_order=min_x_max_x_min_y_max_y
'''

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger as eval_logger
from PIL import Image

from embodied_eval.pointing.coords import (
    decode_points_to_absolute,
    ensure_pil_mask,
    mask_accuracy_from_points,
)

METRICS_FOR_WHERE2PLACE = {"accuracy": "spatial_reference"}

_KIT_POST_PROMPTS = {
    "qwen3": 'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].',
    "gemini-2.5": 'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].',
    "qwen2_5": 'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].',
    "qwen2.5": 'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].',
    "mimo": 'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].',
    "gpt": (
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
        "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
    ),
    "pelican": (
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
        "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
    ),
    "internvl": (
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
        "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
    ),
    "magma": (
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
        "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
    ),
}


def _resolve_backbone(dataset_kwargs=None) -> str:
    if dataset_kwargs and dataset_kwargs.get("backbone"):
        return str(dataset_kwargs["backbone"]).lower().replace("qwen2.5", "qwen2_5")
    env = os.environ.get("WHERE2PLACE_BACKBONE") or os.environ.get("EMBODIED_EVAL_BACKBONE")
    if env:
        return env.lower().replace("qwen2.5", "qwen2_5")
    return "gpt"


def _resolve_post_prompt(dataset_kwargs=None) -> str:
    backbone = _resolve_backbone(dataset_kwargs)
    if dataset_kwargs:
        if dataset_kwargs.get("post_prompt"):
            return dataset_kwargs["post_prompt"]
        if dataset_kwargs.get("use_kit_prompt", True) and backbone in _KIT_POST_PROMPTS:
            return _KIT_POST_PROMPTS[backbone]
    return _KIT_POST_PROMPTS.get("gpt", "")


def where2place_doc_to_visual(doc, dataset_kwargs=None):
    return [doc["image"].convert("RGB")]


def where2place_doc_to_text(doc, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    question = doc["question"]
    pre = dataset_kwargs.get("pre_prompt", "")
    if pre:
        question = f"{pre} {question}"
    post = _resolve_post_prompt(dataset_kwargs)
    if post:
        question = f"{question} {post}"
    return question


def where2place_process_results(doc, results, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    doc["prediction"] = results[0]

    target = np.array(doc["mask"]) / 255.0
    result_dict = {"target": mask_to_bbox(target)}
    result_dict["question_type"] = doc.get("question_type", "where2place")

    image = doc["image"].convert("RGB") if doc.get("image") else None
    width, height = image.size if image else (640, 480)
    backbone = _resolve_backbone(dataset_kwargs)

    acc, points_in_mask, total_points = spatial_reference(
        doc["prediction"],
        doc["mask"],
        width=width,
        height=height,
        backbone=backbone,
        bbox_order=dataset_kwargs.get("bbox_order"),
    )
    doc["accuracy"] = acc
    result_dict["accuracy"] = acc
    result_dict["processed_points"] = getattr(spatial_reference, "_last_abs_points", [])
    result_dict["points_in_mask"] = points_in_mask
    result_dict["total_points"] = total_points
    result_dict["backbone"] = backbone

    return result_dict


def where2place_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        for metric in METRICS_FOR_WHERE2PLACE.keys():
            output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            _qtype, metric_name = key.rsplit("_", 1)
            if metric_name in METRICS_FOR_WHERE2PLACE and isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    for metric_name, vals in metric_to_values.items():
        if vals:
            output[f"{metric_name}_average"] = sum(vals) / len(vals)

    acc_vals = [v for k, v in output.items() if k.endswith("_accuracy") and isinstance(v, (float, int))]
    if acc_vals:
        output["overall"] = sum(acc_vals) / len(acc_vals)
    eval_logger.info(f"Evaluation results: {output}")
    return output


def spatial_reference(
    pred,
    mask,
    width=640,
    height=480,
    threshold=0.5,
    backbone=None,
    bbox_order=None,
):
    """Kit-style spatial reference: omni_decode_points + backbone scaling + mask hit rate."""
    spatial_reference._last_abs_points = []

    if not pred or not str(pred).strip():
        return 0.0, 0, 0

    backbone = (backbone or "gpt").lower().replace("qwen2.5", "qwen2_5")

    text = str(pred).strip()
    text = re.sub(
        r"^(.*?</think>|<think>.*?</think>)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        text = answer_match.group(1).strip()

    try:
        abs_points = decode_points_to_absolute(text, width, height, backbone=backbone)
    except ValueError:
        eval_logger.warning(
            f"where2place: unsupported backbone '{backbone}', falling back to 'gpt'"
        )
        abs_points = decode_points_to_absolute(text, width, height, backbone="gpt")
    spatial_reference._last_abs_points = abs_points

    if not abs_points:
        eval_logger.debug("where2place: no points parsed from prediction")
        return 0.0, 0, 0

    pil_mask = ensure_pil_mask(mask, width, height)

    if len(abs_points) >= 1 and bbox_order:
        abs_points = _expand_bbox_points(abs_points, width, height, bbox_order, text, backbone)

    acc, points_in_mask, total_points = mask_accuracy_from_points(abs_points, pil_mask)
    return acc, points_in_mask, total_points


def _expand_bbox_points(abs_points, width, height, bbox_order, text, backbone):
    """If model returned a single 4-vector bbox, sample points inside the box."""
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    for match in re.findall(pattern, text):
        vector = [float(n) if "." in n else int(n) for n in match.split(",")]
        if len(vector) != 4:
            continue
        if bbox_order == "min_x_max_x_min_y_max_y":
            min_x, max_x, min_y, max_y = vector
            x0, x1 = min_x, max_x
            y0, y1 = min_y, max_y
        else:
            x0, y0, x1, y1 = vector

        # Scale if floats (normalized) OR if integers that exceed image bounds (0-1000 style)
        needs_scale = all(isinstance(v, float) for v in (x0, y0, x1, y1)) or any(
            v > max(width, height) for v in (x0, x1, y0, y1)
        )
        if needs_scale:
            if backbone in ("qwen3", "gemini-2.5", "gemini_robotics"):
                x0, x1 = x0 / 1000.0 * width, x1 / 1000.0 * width
                y0, y1 = y0 / 1000.0 * height, y1 / 1000.0 * height
            elif backbone in ("gpt", "pelican", "internvl", "magma"):
                x0, x1 = x0 * width, x1 * width
                y0, y1 = y0 * height, y1 * height
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        binary = np.zeros((height, width), dtype=bool)
        binary[y0:y1, x0:x1] = True
        ys, xs = np.where(binary)
        if len(xs):
            return np.stack([xs, ys], axis=1).tolist()
    return abs_points


def mask_to_bbox(mask, threshold=0.5):
    binary_mask = mask > threshold
    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return (x0, y0, x1, y1)
