'''
Cosmos Reasoning Benchmark - Embodied Reasoning Evaluation
Official Repo: https://github.com/nvidia-cosmos/cosmos-reason1
Paper: https://arxiv.org/abs/2503.15558

This benchmark evaluates embodied reasoning capabilities across multiple robotics domains:
- BridgeV2: Robot manipulation tasks
- RoboVQA: Visual question answering for robotics
- AgibotWorld: Industrial robot tasks
- HoloAssist: Egocentric human demonstration
- RoboFail: Failure detection in robotics
'''

import os
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from loguru import logger as eval_logger
from pathlib import Path

# Cosmos sub-tasks
COSMOS_SUB_TASKS = [
    "bridgev2",
    "robovqa",
    "agibot",
    "holoassist",
    "robofail"
]

# Metrics for Cosmos tasks
METRICS_FOR_COSMOS = {
    "accuracy": "compute_mcq_accuracy"
}


def parse_letter_response(output_text: str) -> tuple[str, str]:
    """
    Parses model output text to extract answer and reasoning.
    
    Supports multiple formats:
    1. Official Cosmos format with <answer> and <think> tags
    2. Plain text with single letter answer (A-Z)
    
    Args:
        output_text: The raw text response from the model.
    
    Returns:
        A tuple containing (answer: str, reasoning: str).
        Returns empty strings if no answer/reasoning found.
    """
    if not output_text:
        return "", ""
    
    answer = ""
    reasoning = ""
    
    # Try to extract from <answer> tags (official Cosmos format)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_match = ANSWER_PATTERN.search(output_text)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Extract single letter from answer content
        SINGLE_LETTER_PATTERN = re.compile(r"[A-Z]")
        letter_match = SINGLE_LETTER_PATTERN.search(answer_content)
        if letter_match:
            answer = letter_match.group(0)
    
    # Try to extract reasoning from <think> tags
    REASONING_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    reasoning_match = REASONING_PATTERN.search(output_text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Fallback: extract option letter from plain text.
    #
    # IMPORTANT: do NOT take the first capital letter (e.g. "So..." -> "S").
    # We instead look for standalone option letters like "A", "(A)", "Answer: A",
    # and take the LAST occurrence to handle chain-of-thought + final answer.
    if not answer:
        text = output_text.strip()

        # Prefer A-D (Cosmos MCQ); take last standalone hit.
        m_all = re.findall(r"(?<![A-Za-z0-9])\(?\s*([A-D])\s*\)?(?![A-Za-z0-9])", text, flags=re.IGNORECASE)
        if m_all:
            answer = m_all[-1].upper()
        else:
            # Some sub-tasks are binary A/B.
            m_bin = re.findall(r"(?<![A-Za-z0-9])\(?\s*([A-B])\s*\)?(?![A-Za-z0-9])", text, flags=re.IGNORECASE)
            if m_bin:
                answer = m_bin[-1].upper()
    
    return answer, reasoning


def cosmos_doc_to_visual(doc, dataset_kwargs=None):
    """Extract video path for Cosmos task"""
    video_path = doc.get("video", "")
    if not video_path:
        eval_logger.warning("No video path found in document")
        return []
    
    # Get video directory from dataset_kwargs
    video_dir = dataset_kwargs.get("video_dir", "") if dataset_kwargs else ""
    if not video_dir:
        eval_logger.warning("No video_dir specified in dataset_kwargs")
        return []
    
    full_video_path = os.path.join(video_dir, video_path)
    
    if not os.path.exists(full_video_path):
        eval_logger.warning(f"Video file not found: {full_video_path}")
        return []
    
    return [full_video_path]


def cosmos_doc_to_text(doc, dataset_kwargs=None):
    """Extract question text with choices for Cosmos task"""
    qa_pairs = doc.get("qa_pairs", {})
    question = qa_pairs.get("question", "")
    choices = qa_pairs.get("index2ans", {})
    
    # Format question with choices
    prompt_text = question + "\n"
    prompt_text += "\n".join([f"({key}) {value}" for key, value in sorted(choices.items())])
    
    # Add pre/post prompts if specified
    pre_prompt = ""
    post_prompt = ""
    if dataset_kwargs:
        pre_prompt = dataset_kwargs.get("pre_prompt", "")
        post_prompt = dataset_kwargs.get("post_prompt", "\nAnswer with the option's letter from the given choices directly.")
    
    return f"{pre_prompt}{prompt_text}{post_prompt}"


def cosmos_doc_to_target(doc, dataset_kwargs=None):
    """Extract target answer for Cosmos task"""
    qa_pairs = doc.get("qa_pairs", {})
    answer = qa_pairs.get("answer", "")
    return answer.upper() if answer else ""


def cosmos_process_results(doc, results, dataset_kwargs=None):
    """
    Process Cosmos results and compute metrics.
    
    Expected result format:
    - results[0]: predicted answer text (may contain <answer> and <think> tags)
    
    Follows official Cosmos-Reason1 benchmark format.
    """
    prediction = results[0] if results and results[0] else ""
    
    # Parse the answer and reasoning from model output (supports official format)
    pred_letter, reasoning = parse_letter_response(prediction)
    doc["prediction"] = pred_letter
    doc["reasoning"] = reasoning
    doc["full_response"] = prediction
    
    # Get ground truth
    target = cosmos_doc_to_target(doc, dataset_kwargs)
    
    # Get sub-task name for grouping
    sub_task = dataset_kwargs.get("sub_task", "unknown") if dataset_kwargs else "unknown"
    
    # Check if answer is correct (following official benchmark)
    is_correct = (pred_letter.strip().upper() == target.strip().upper()) if pred_letter and target else False
    
    result_dict = {
        "target": target,
        "prediction": pred_letter,
        "reasoning": reasoning,
        "full_response": prediction,
        "sub_task": sub_task,
        "is_correct": is_correct  # Official format field
    }
    
    # Compute metrics
    for metric_name, metric_func in METRICS_FOR_COSMOS.items():
        try:
            score = eval(metric_func)(pred_letter, target)
            doc[metric_name] = score
            result_dict[metric_name] = score
        except Exception as e:
            eval_logger.warning(f"Failed to compute {metric_name}: {e}")
            doc[metric_name] = 0.0
            result_dict[metric_name] = 0.0
    
    return result_dict


def cosmos_aggregate_results(results):
    """
    Aggregate Cosmos results by sub-task and overall.
    """
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    output = {}
    
    # Aggregate by sub-task
    if "sub_task" in results_df.columns:
        for sub_task, indices in results_df.groupby("sub_task").groups.items():
            per_sub_task = results_df.iloc[indices]
            
            for metric_name in METRICS_FOR_COSMOS.keys():
                if metric_name in per_sub_task.columns:
                    avg_score = per_sub_task[metric_name].mean()
                    output[f"{sub_task}_{metric_name}"] = avg_score
    
    # Compute average across sub-tasks for each metric
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            sub_task, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            output[f"{metric_name}_average"] = sum(vals) / len(vals)
    
    # Overall metrics (computed from all samples, not just sub-task averages)
    for metric_name in METRICS_FOR_COSMOS.keys():
        if metric_name in results_df.columns:
            overall_score = results_df[metric_name].mean()
            output[f"overall_{metric_name}"] = overall_score
    
    # Add overall score (average of all overall metrics)
    overall_metrics = [v for k, v in output.items() if k.startswith("overall_") and isinstance(v, (int, float))]
    if overall_metrics:
        output["overall"] = sum(overall_metrics) / len(overall_metrics)
    
    eval_logger.info(f"Cosmos Evaluation Results: {output}")
    return output


def compute_mcq_accuracy(prediction: str, target: str) -> float:
    """
    Compute multiple choice question accuracy.
    
    Args:
        prediction: Predicted answer letter (e.g., "A", "B", "C", "D")
        target: Ground truth answer letter
    
    Returns:
        1.0 if prediction matches target, 0.0 otherwise
    """
    if not prediction or not target:
        return 0.0
    
    pred_normalized = prediction.strip().upper()
    target_normalized = target.strip().upper()
    
    return 1.0 if pred_normalized == target_normalized else 0.0


def post_process_results(sample_file_path: str, results_file_path: str):
    """
    Post-process results from saved samples file.
    
    Args:
        sample_file_path: Path to samples JSON file
        results_file_path: Path to save aggregated results
    """
    if not os.path.exists(sample_file_path):
        eval_logger.error(f"Sample file not found: {sample_file_path}")
        return
    
    # Read samples (jsonl format)
    data = []
    with open(sample_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    eval_logger.info(f"Loaded {len(data)} samples from {sample_file_path}")
    
    # Recompute metrics for each sample
    results = []
    for doc in data:
        # Extract prediction from resps field
        pred_raw = ""
        if "resps" in doc and doc["resps"] and doc["resps"][0]:
            pred_raw = doc["resps"][0][0]
        
        # Parse prediction (supports official format with <answer> tags)
        pred_letter, reasoning = parse_letter_response(pred_raw)
        
        # Get target
        target = doc.get("target", "")
        
        # Get sub-task
        # NOTE: in our saved Cosmos samples, `doc` is typically a string prompt (not a dict),
        # so we must not assume `.get()` exists on it.
        sub_task = doc.get("sub_task", "unknown")
        
        # Compute accuracy
        accuracy = compute_mcq_accuracy(pred_letter, target)
        
        # Check if correct (official format)
        is_correct = (pred_letter.strip().upper() == target.strip().upper()) if pred_letter and target else False
        
        # Update doc
        doc["prediction"] = pred_letter
        doc["reasoning"] = reasoning
        doc["full_response"] = pred_raw
        doc["accuracy"] = accuracy
        doc["is_correct"] = is_correct
        
        # Add to results
        result_dict = {
            "target": target,
            "prediction": pred_letter,
            "reasoning": reasoning,
            "full_response": pred_raw,
            "sub_task": sub_task,
            "accuracy": accuracy,
            "is_correct": is_correct
        }
        results.append(result_dict)
    
    # Save updated samples
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    eval_logger.info(f"Updated samples saved to {sample_file_path}")
    
    # Aggregate results
    aggregated_results = cosmos_aggregate_results(results)
    
    # Save aggregated results
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, ensure_ascii=False, indent=2)
    
    eval_logger.info(f"Aggregated results saved to {results_file_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-process Cosmos evaluation results")
    parser.add_argument("--sample_file", type=str, required=True, help="Path to samples JSON file")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save aggregated results")
    args = parser.parse_args()
    
    post_process_results(
        sample_file_path=args.sample_file,
        results_file_path=args.results_file
    )

