'''
modified from "https://github.com/embodiedreasoning/ERQA/blob/main/eval_harness.py"
is_correct = response_text.replace(".", "").strip().lower() == answer.strip().lower()
'''
import os

import numpy as np
import pandas as pd

from collections import defaultdict
from loguru import logger as eval_logger

# MCA
METRICS_FOR_ERQA = {"accuracy": "exact_match"}
ERQA_QUESTION_TYPES = [
    "Action Reasoning",
    "Multi-view Reasoning",
    "Other",
    "Pointing",
    "State Estimation",
    "Spatial Reasoning",
    "Trajectory Reasoning",
    "Task Reasoning",
]

def erqa_doc_to_visual(doc, dataset_kwargs=None):
    images = doc["images"]
    if not isinstance(images, (list, tuple)):
        images = [images]
    
    if len(doc["visual_indices"]) == 0:
        return images
    else:
        return (images, doc["visual_indices"])


def erqa_doc_to_text(doc, dataset_kwargs=None):
    return doc["question"]

def extract_single_word_option(text):
    """
    Extract a single word option from the prediction text.
    Handles various response formats including thinking mode outputs.
    """
    if not text:
        return ""
    
    # Clean the text
    text = text.strip()
    
    import re
    
    # First, handle thinking mode: extract content after </think> tag
    think_pattern = r'</think>\s*(.+?)$'
    think_match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if think_match:
        # Extract text after </think> tag
        answer_text = think_match.group(1).strip()
        
        # If the answer after </think> is a single letter (A-D), return it directly
        single_letter_match = re.match(r'^([A-D])\.?\s*$', answer_text, re.IGNORECASE)
        if single_letter_match:
            return single_letter_match.group(1).upper()
        
        # Otherwise, continue with pattern matching on the extracted text
        text = answer_text
    
    # Common patterns for extracting answers
    answer_patterns = [
        r'(?:answer|Answer)(?:\s*is)?\s*:\s*([A-D])',  # Answer: A or Answer is A
        r'(?:the|The)\s+answer\s+is\s+([A-D])',  # The answer is A
        r'^\s*([A-D])\s*[\.\):]',  # Single letter at start with punctuation
        r'\b([A-D])\s*[\.\):]?\s*$',  # Single letter at end
        r'^([A-D])$',  # Just a single letter
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no pattern matches, try to find any single uppercase letter A-D
    fallback_match = re.search(r'\b([A-D])\b', text)
    if fallback_match:
        return fallback_match.group(1).upper()
    
    # Return empty string if no valid answer found
    return ""

def erqa_process_results(doc, results, dataset_kwargs=None):
    pred_raw = results[0]
    # Process the raw prediction to extract single word option
    processed_pred = extract_single_word_option(pred_raw)
    doc["prediction"] = processed_pred
    target = doc["answer"]

    result_dict = {"target": target}
    result_dict["question_type"] = doc.get("question_type", "erqa")
    for key, value in METRICS_FOR_ERQA.items():
        doc[key] = eval(value)(doc["prediction"], target)
        result_dict[key] = doc[key]

    return result_dict

def erqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in ERQA_QUESTION_TYPES:
            for metric in METRICS_FOR_ERQA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
    
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            qtype, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_average"] = avg_val
            
    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output

def exact_match(response_text, answer):
    return response_text.replace(".", "").strip().lower() == answer.strip().lower()

def post_evaluate_results(sample_file_path, results_file_path):
    import json
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for doc in data:
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        # Process the raw prediction to extract single word option
        processed_pred = extract_single_word_option(pred_raw)
        target = doc["target"]
        question_type = doc["question_type"]

        for key, value in METRICS_FOR_ERQA.items():
            doc[key] = eval(value)(processed_pred, target)

            result_dict = {
                "question_type": question_type,
                key: doc[key]
            }
            results.append(result_dict)
    
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    output = erqa_aggregate_results(results)

    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    post_evaluate_results(
        sample_file_path="",
        results_file_path=""
    )