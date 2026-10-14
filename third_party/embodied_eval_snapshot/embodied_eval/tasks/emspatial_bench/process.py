'''
EmbSpatial-Bench benchmark adaptation for embodied-eval framework
Official Repo: https://github.com/mengfeidu/EmbSpatial-Bench

Implementation Note:
- EmbSpatial-Bench evaluates spatial understanding from an egocentric perspective
- Questions are formatted as multiple-choice problems
- Images are stored as base64 encoded strings in the dataset
- Evaluation uses accuracy metric, matching predictions to answer options
'''

import os
import base64
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, List, Dict
from io import BytesIO
from PIL import Image
from loguru import logger as eval_logger
from tqdm import tqdm

# Spatial relation types in EmbSpatial-Bench
EMSPATIAL_RELATIONS = [
    "close",
    "under",
    "right",
    "left",
    "above",
    "behind",
]

# Metrics for EmbSpatial-Bench evaluation
METRICS_FOR_EMSPATIAL = {
    "accuracy": "compute_accuracy"
}


def emspatial_doc_to_visual(doc, dataset_kwargs=None):
    """
    Extract visual data from EmbSpatial-Bench document.
    Images can be either PIL Image objects (from HuggingFace dataset) or base64 encoded strings.
    
    Args:
        doc: Document containing 'image' field (PIL Image) or 'image_base64' field (base64 string)
        dataset_kwargs: Dataset configuration
    
    Returns:
        List containing a PIL Image object
    """
    # First try to get PIL Image directly (HuggingFace format)
    image = doc.get("image")
    if image is not None:
        # If it's already a PIL Image, use it directly
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            return [image]
        # If it's a dict with 'bytes' or 'path', handle accordingly
        elif isinstance(image, dict):
            if 'bytes' in image:
                image = Image.open(BytesIO(image['bytes']))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return [image]
    
    # Fallback to base64 decoding if image_base64 field exists
    image_base64 = doc.get("image_base64", "")
    if not image_base64 and isinstance(image, str):
        image_base64 = image
    if image_base64:
        try:
            # Decode base64 image
            # Handle both raw base64 and data URI formats
            if image_base64.startswith("data:image"):
                # Extract base64 part from data URI
                base64_data = image_base64.split(",")[1]
            else:
                base64_data = image_base64
            
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes))
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return [image]
        except Exception as e:
            eval_logger.error(f"Failed to decode base64 image for {doc.get('question_id', 'unknown')}: {e}")
    
    eval_logger.warning(f"No image found in document {doc.get('question_id', 'unknown')}")
    return []


def emspatial_doc_to_text(doc, dataset_kwargs=None):
    """
    Extract question text from EmbSpatial-Bench document.
    Formats the question with answer options for the model.
    
    Args:
        doc: Document containing question and answer_options
    
    Returns:
        Formatted question string with options
    """
    question = doc.get("question", "")
    answer_options = doc.get("answer_options", [])
    
    if not question:
        return ""
    
    # Format question with options
    if answer_options:
        options_text = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(answer_options)])
        formatted_question = f"{question}\n\nOptions:\n{options_text}\n\nPlease select the correct answer (A, B, C, or D).\n"
        # you need to answer by only outputting the letter corresponding to your choice. your answer should be one of A, B, C, or D. you mast give a letter as your final answer rather than don't output anything else. This question mainly examines the relative positions of objects.
    else:
        formatted_question = question
    
    return formatted_question


def emspatial_doc_to_target(doc, dataset_kwargs=None):
    """
    Extract target answer from EmbSpatial-Bench document.
    
    Args:
        doc: Document containing answer (index) and answer_options
    
    Returns:
        Target answer string (the actual option text)
    """
    answer_idx = doc.get("answer")
    answer_options = doc.get("answer_options", [])
    
    if answer_idx is not None and isinstance(answer_idx, int) and 0 <= answer_idx < len(answer_options):
        return answer_options[answer_idx]
    elif answer_idx is not None:
        # If answer is already a string, return it
        return str(answer_idx)
    else:
        return ""


def extract_prediction_from_text(prediction: str, answer_options: List[str]) -> Optional[int]:
    """
    Extract the predicted answer index from model's text response.
    Tries multiple strategies to match the prediction to answer options.
    
    Args:
        prediction: Model's raw text response
        answer_options: List of answer option strings
    
    Returns:
        Index of matched option, or None if no match found
    """
    if not prediction or not answer_options:
        return None

    strict_mode = str(os.getenv("EMSPATIAL_SCORING_STRICT", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    def _norm_text_for_match(text: str) -> str:
        text = (text or "").lower()
        # Keep letters/numbers/spaces only; collapse whitespace
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    pred_strip = re.sub(
            r"<think>.*?</think>",
            "",
            prediction,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
    pred_lower = pred_strip.lower()
    # Strategy 1: Direct letter match (A, B, C, D) — robust to case and punctuation
    # Match standalone letters like 'A', 'B.', '(C)', 'final answer is D', etc.
    # Use findall and take the last match to prioritize answers at the end of the text
    letter_matches = re.findall(r"\b([A-D])\b", pred_strip, re.IGNORECASE)
    if letter_matches:
        idx = ord(letter_matches[-1].upper()) - ord('A')
        if 0 <= idx < len(answer_options):
            return idx

    # Also catch forms like 'final answer: C' or 'final answer is C'
    fa_match = re.search(r"final answer[:\s]*([A-D])\b", pred_strip, re.IGNORECASE)
    if fa_match:
        idx = ord(fa_match.group(1).upper()) - ord('A')
        if 0 <= idx < len(answer_options):
            return idx
    
    # If strict mode enabled, do not attempt heuristic/fuzzy inference.
    # Only accept explicit option selection signals (letter/final answer/option text/quoted text/number).
    if strict_mode:
        return None
    
    # Strategy 2: Match option text directly (case-insensitive, full option match)
    pred_norm = _norm_text_for_match(pred_strip)
    if pred_norm:
        for i, option in enumerate(answer_options):
            opt_norm = _norm_text_for_match(option)
            if not opt_norm:
                continue
            # Require full normalized option text to appear in the prediction
            if opt_norm in pred_norm:
                return i
    
    # Strategy 3: Extract quoted text and match
    quoted_match = re.search(r'["\']([^"\']+)["\']', prediction)
    if quoted_match:
        quoted_text = quoted_match.group(1).lower().strip()
        for i, option in enumerate(answer_options):
            if quoted_text in option.lower() or option.lower() in quoted_text:
                return i
    
    # Strategy 4: Number match (1, 2, 3, 4)
    number_match = re.search(r'\b([1-4])\b', pred_strip)
    if number_match:
        idx = int(number_match.group(1)) - 1
        if 0 <= idx < len(answer_options):
            return idx

    
    # Strategy 5: Fuzzy match using word overlap
    pred_words = set(re.findall(r'\b\w+\b', pred_lower))
    best_match_idx = None
    best_match_score = 0
    
    for i, option in enumerate(answer_options):
        option_words = set(re.findall(r'\b\w+\b', option.lower()))
        overlap = len(pred_words & option_words)
        if overlap > best_match_score and overlap > 0:
            best_match_score = overlap
            best_match_idx = i
    
    return best_match_idx


def compute_accuracy(doc, target, prediction):
    """
    Compute accuracy by matching prediction to the correct answer option.
    
    Args:
        doc: Document containing answer_options and answer (index)
        target: Target answer string (unused, we use doc['answer'] instead)
        prediction: Model's raw text prediction
    
    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    answer_idx = doc.get("answer")
    answer_options = doc.get("answer_options", [])

    eval_logger.info(f"compute_accuracy called for doc_id={doc.get('question_id')} relation={doc.get('relation')} answer_present={'yes' if answer_idx is not None else 'no'} options_present={len(answer_options)}")

    # Helper: try to parse options from the raw question text if not provided
    def _parse_options_from_doc_text(doc_text: str) -> List[str]:
        if not doc_text or "Options" not in doc_text:
            return []
        parts = doc_text.split("Options:")[-1].strip().splitlines()
        opts = []
        for line in parts:
            line = line.strip()
            if not line:
                continue
            # lines like 'A. text' or 'A) text'
            m = re.match(r'^[A-D][\.)]\s*(.*)$', line)
            if m:
                opts.append(m.group(1).strip())
            else:
                # if the line doesn't start with letter, try to take whole line
                opts.append(line)
            # stop when we gathered 4 options
            if len(opts) >= 4:
                break
        return opts

    # If answer_options missing, try to extract from doc string
    if not answer_options:
        raw_doc_text = doc.get("doc") or doc.get("question") or ""
        parsed = _parse_options_from_doc_text(raw_doc_text)
        if parsed:
            answer_options = parsed
            eval_logger.info(f"Parsed answer_options from doc text: {answer_options}")

    # If numeric answer index missing, try to derive from 'target' (text) or doc['target']
    if answer_idx is None:
        t = target or doc.get("target")
        eval_logger.info(f"No numeric answer index; trying to infer from target='{t}'")
        if t and answer_options:
            for i, opt in enumerate(answer_options):
                if str(t).strip().lower() == str(opt).strip().lower():
                    answer_idx = i
                    break
        if answer_idx is not None:
            eval_logger.info(f"Inferred answer_idx={answer_idx} from target text")
        else:
            eval_logger.info("Could not infer numeric answer index from target/text")

    # If still missing, give up
    if answer_idx is None or not answer_options:
        eval_logger.warning(f"Unable to determine ground-truth answer for doc_id={doc.get('question_id')}; answer_idx={answer_idx} answer_options_len={len(answer_options)}")
        return 0.0

    # Extract predicted index from text
    eval_logger.info(f"Extracting prediction from text. prediction_preview='{str(prediction)[:200]}'")
    pred_idx = extract_prediction_from_text(prediction, answer_options)
    if pred_idx is None:
        eval_logger.info(f"Prediction could not be mapped to any option. doc_id={doc.get('question_id')}")
        return 0.0

    # Log predicted option and comparison
    pred_text = answer_options[pred_idx] if 0 <= pred_idx < len(answer_options) else None
    true_text = answer_options[answer_idx] if 0 <= answer_idx < len(answer_options) else None
    eval_logger.info(f"doc_id={doc.get('question_id')} predicted_idx={pred_idx} predicted_text='{pred_text}' answer_idx={answer_idx} answer_text='{true_text}'")

    score = 1.0 if pred_idx == answer_idx else 0.0
    eval_logger.info(f"Result for doc_id={doc.get('question_id')}: score={score}")
    return score


def emspatial_process_results(doc, results, dataset_kwargs=None):
    """
    Process model prediction results and compute evaluation metrics.
    
    Args:
        doc: Original document
        results: Model prediction results (list of strings)
        dataset_kwargs: Dataset configuration
    
    Returns:
        Dictionary containing evaluation results
    """
    raw_prediction = results[0] if results else ""
    doc["prediction"] = raw_prediction
    
    # Get ground truth answer
    target = emspatial_doc_to_target(doc, dataset_kwargs)
    
    # Get spatial relation type for categorization
    relation = doc.get("relation", "unknown")
    
    result_dict = {"target": target}
    result_dict["relation"] = relation
    
    # Compute metrics
    for key, value in METRICS_FOR_EMSPATIAL.items():
        score = eval(value)(doc, target, raw_prediction)
        doc[key] = {key: score}
        result_dict[key] = doc[key]
    
    return result_dict


def emspatial_aggregate_results(results):
    """
    Aggregate evaluation results by spatial relation type.
    
    Args:
        results: List of result dictionaries containing relation and scores
    
    Returns:
        Dictionary containing:
        - Per-relation accuracy scores
        - Overall accuracy
    """
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    output = {}
    
    # Compute per-relation metrics
    if "relation" in results_df.columns:
        for relation, indices in results_df.groupby("relation").groups.items():
            if relation == "None" or relation is None:
                continue
            
            per_relation = results_df.iloc[indices]
            metric_name = "accuracy"
            
            if metric_name in per_relation.columns:
                metric_data = per_relation[metric_name].tolist()
                avg_score = np.mean([x[metric_name] for x in metric_data])
                output[f"{relation}_{metric_name}"] = avg_score
    
    # Compute overall accuracy
    if "accuracy" in results_df.columns:
        all_accuracy = results_df["accuracy"].tolist()
        overall_accuracy = np.mean([x["accuracy"] for x in all_accuracy])
        output["accuracy_average"] = overall_accuracy
        output["overall"] = overall_accuracy
    
    eval_logger.info(f"EmbSpatial-Bench Evaluation Results: {output}")
    return output


def post_evaluate_results(sample_file_path, results_file_path):
    """
    Post-process evaluation results for offline analysis.
    
    Args:
        sample_file_path: Path to samples JSON file (jsonl format)
        results_file_path: Path to save results JSON file
    """
    import json

    if not os.path.exists(sample_file_path):
        eval_logger.error(f"Sample file not found: {sample_file_path}")
        return
    
    # Read all samples
    data = []
    with open(sample_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    eval_logger.info(f"Loaded {len(data)} samples from {sample_file_path}. Starting re-evaluation...")
    
    results = []
    for doc in tqdm(data, desc="Re-evaluating emspatial samples"):
        # Extract prediction from sample
        pred_raw = ""
        if isinstance(doc.get("resps"), list) and doc["resps"] and doc["resps"][0]:
            pred_raw = doc["resps"][0][0]
        elif doc.get("prediction"):
            pred_raw = doc["prediction"]
        
        target = doc.get("target", emspatial_doc_to_target(doc))
        relation = doc.get("relation", doc.get("relation", "unknown"))
        
        # Re-compute metrics
        result_dict = {"target": target, "relation": relation}
        
        for key, value in METRICS_FOR_EMSPATIAL.items():
            score = eval(value)(doc, target, pred_raw)
            doc[key] = {key: score}
            result_dict[key] = doc[key]
        
        results.append(result_dict)
    
    # Save updated samples
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    eval_logger.info(f"Updated samples saved to {sample_file_path}")
    
    # Save aggregated results
    output = emspatial_aggregate_results(results)
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    eval_logger.info(f"Updated aggregated results saved to {results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-evaluate EmbSpatial-Bench results from saved samples (jsonl)."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="日志目录，例如 .../logs/emspatial-bench/<model>/<run_id>/",
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default=None,
        help="samples_emspatial-bench.json 的完整路径（优先级高于 base_dir）",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="results_emspatial-bench.json 的完整路径（优先级高于 base_dir）",
    )
    args = parser.parse_args()

    sample_path = args.sample_file
    results_path = args.results_file

    # 如果只给了 base_dir，则自动拼接出样本和结果文件路径
    if args.base_dir and (not sample_path or not results_path):
        base_dir = args.base_dir.rstrip("/")
        if not sample_path:
            sample_path = f"{base_dir}/samples_emspatial-bench.json"
        if not results_path:
            results_path = f"{base_dir}/results_emspatial-bench.json"

    if not sample_path or not results_path:
        raise ValueError(
            "You must provide --base_dir or both --sample_file and --results_file."
        )

    post_evaluate_results(
        sample_file_path=sample_path,
        results_file_path=results_path,
    )
