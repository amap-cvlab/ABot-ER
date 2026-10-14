'''
modified from "https://github.com/facebookresearch/open-eqa/blob/main/openeqa/evaluation/llm_match.py"

'''
import os
import numpy as np
import pandas as pd
import time
import json
import argparse

from collections import defaultdict
from openai import OpenAI
from typing import Optional
from loguru import logger as eval_logger
from tqdm import tqdm

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"), 
    base_url = os.getenv("OPENAI_API_BASE"),
    default_headers = {"x-foo": "true"}
)
# openai.api_key = os.getenv("OPENAI_API_KEY")

# openai.base_url = os.getenv("OPENAI_API_BASE")
# openai.default_headers = {"x-foo": "true"}

METRICS_FOR_OPENEQA_EMEQA = {
    "llm_match_score": "llm_match"
}
OPENEQA_EMEQA_QUESTION_TYPES = [
    "attribute recognition",
    "functional reasoning",
    "object localization",
    "object recognition",
    "object state recognition",
    "spatial understanding",
    "world knowledge",
]

def openeqa_emeqa_doc_to_visual(doc, dataset_kwargs=None):
    video_path = os.path.join(dataset_kwargs["video_dir"], f'{doc["episode_history"]}.mp4')
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def openeqa_emeqa_doc_to_text(doc, dataset_kwargs=None):
    question = doc["question"]
    return question

def openeqa_emeqa_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]
    pred_raw = results[0]
    
    target = doc["answer"]
    extra_answers = doc.get("extra_answers", None)
    question_type = doc["category"]

    result_dict = {"target": target}
    if extra_answers:
        result_dict = {"extra_answers": extra_answers}
    result_dict["question_type"] = question_type

    for key, value in METRICS_FOR_OPENEQA_EMEQA.items():
        score, judge_error = eval(value)(doc["question"], target, pred_raw, return_error=True)
        doc[key] = {key: score}
        result_dict[key] = doc[key]
        if judge_error:
            result_dict["judge_failed"] = True
            result_dict["judge_error"] = judge_error

    return result_dict

def openeqa_emeqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}
    # key: {question_type}_{metric_name}
    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in OPENEQA_EMEQA_QUESTION_TYPES:
            for metric in METRICS_FOR_OPENEQA_EMEQA.keys():
                metric_data = per_question_type[metric].tolist()
                avg_score = np.mean([x[metric] for x in metric_data])
                output[f"{question_type}_{metric}"] = avg_score
    
    metric_to_values = defaultdict(list)
    metric_to_all_scores = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            qtype, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    for metric_name in METRICS_FOR_OPENEQA_EMEQA.keys():
        if metric_name in results.columns:
            metric_data = results[metric_name].tolist()
            metric_to_all_scores[metric_name].extend([x[metric_name] for x in metric_data])
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_per_type_average"] = avg_val
    for metric_name, vals in metric_to_all_scores.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_all_samples_average"] = avg_val
    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    output["100score_overall"] = (output["overall"] - 1) * 25
    eval_logger.info(f"Evaluation results: {output}")
    return output

def llm_match(
        question: str,
        answer: str,
        prediction: str,
        extra_answers: Optional[list] = None,
        openai_model: str = "gpt-4o-1120-global",
        openai_seed: int = 1234,
        openai_max_tokens: int = 32,
        openai_temperature: float = 0.2,
        verbose: bool = False,
        return_error: bool = False,
    ):
    if prediction is None:
        return (1, None) if return_error else 1
    
    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    messages = prepare_openai_messages(
        prompt.format(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
        ),
    )
    try:
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            verbose=verbose,
        )
        score = parse_score(output)
        if score < 1 or score > 5:
            error = f"Invalid judge score {score} from output: {output!r}"
            eval_logger.error(error)
            return (1, error) if return_error else 1
        score = clamp_llm_score(score)
        return (score, None) if return_error else score
    except Exception as e:
        eval_logger.info(f"Evaluation Error: {e}，")
        return (1, str(e)) if return_error else 1

def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]

def call_openai_api(
    messages: list,
    model: str = "gpt-4o-1120-global",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content

def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())


def clamp_llm_score(score: int) -> int:
    return min(max(int(score), 1), 5)

def load_prompt(name: str):
    if name == "mmbench":
        prompt = """
        You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
        To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
        5 means that the response perfectly matches the answer.
        1 means that the response is completely different from the answer.
        Output EXACTLY one number X and X must be in [1, 5].
        Output nothing else.

        Example 1:
        Question: Is it overcast?
        Answer: no
        Response: yes
        Your mark: 1

        Example 2:
        Question: Who is standing at the table?
        Answer: woman
        Response: Jessica
        Your mark: 3

        Example 3:
        Question: Are there drapes to the right of the bed?
        Answer: yes
        Response: yes
        Your mark: 5

        Your Turn:
        Question: {question}
        Answer: {answer}
        Response: {prediction}    
        """
    else:
        prompt = """
        You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
        To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
        5 means that the response perfectly matches the answer or any of the extra answers.
        1 means that the response is completely different from the answer and all of the extra answers.
        Output EXACTLY one number X and X must be in [1, 5].
        Output nothing else.

        Example 1:
        Question: Is it overcast?
        Answer: no
        Extra Answers: ['doesn't look like it', 'no',' it's sunny']
        Response: yes
        Your mark: 1

        Example 2:
        Question: Who is standing at the table?
        Answer: woman
        Extra Answers: ['a woman', 'a lady', 'woman']
        Response: Jessica
        Your mark: 3

        Example 3:
        Question: Are there drapes to the right of the bed?
        Answer: yes
        Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
        Response: yes
        Your mark: 5

        Your Turn:
        Question: {question}
        Answer: {answer}
        Extra Answers: {extra_answers}
        Response: {prediction}
        """
    return prompt.strip()


def _extract_prediction_from_sample(sample: dict) -> str:
    """
    samples_*.json 里常见字段：
    - prediction: str（有些任务会写入）
    - resps: [[str]]（框架标准输出）
    """
    pred_raw = sample.get("prediction", "") or ""
    if pred_raw:
        return pred_raw
    resps = sample.get("resps")
    if isinstance(resps, list) and len(resps) > 0:
        try:
            # 结构通常是 [[answer_str]]
            return resps[0][0] or ""
        except Exception:
            return ""
    return ""


def post_evaluate_results(
    sample_file_path: str,
    results_file_path: str,
    openai_model: str = "gpt-4o-1120-global",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    verbose: bool = False,
):
    """
    根据已经保存的 samples_openeqa-emeqa.json（jsonl）重新发出 LLM judge 请求打分，
    并覆盖写入 results_openeqa-emeqa.json 的聚合结果。

    语义：对每一条 sample 都重新打分（即使已有旧分数），再做一次聚合。
    """
    if not os.path.exists(sample_file_path):
        eval_logger.error(f"Sample file not found: {sample_file_path}")
        return

    # 读取所有 sample(jsonl)
    data = []
    with open(sample_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    eval_logger.info(f"Loaded {len(data)} samples from {sample_file_path}. Starting re-evaluation...")

    results = []
    for doc in tqdm(data, desc="Re-evaluating openeqa samples"):
        question = doc.get("question", doc.get("doc", ""))
        target = doc.get("target", doc.get("answer", ""))
        extra_answers = doc.get("extra_answers", None)
        question_type = doc.get("question_type", doc.get("category", ""))

        pred_raw = _extract_prediction_from_sample(doc)

        # 重新进行 LLM 评分
        score = llm_match(
            question=question,
            answer=target,
            prediction=pred_raw,
            extra_answers=extra_answers,
            openai_model=openai_model,
            openai_seed=openai_seed,
            openai_max_tokens=openai_max_tokens,
            openai_temperature=openai_temperature,
            verbose=verbose,
        )

        # 回写到 sample，方便后续再次 post-eval
        doc["prediction"] = pred_raw
        doc["llm_match_score"] = {"llm_match_score": score}
        doc["question_type"] = question_type
        if extra_answers is not None:
            doc["extra_answers"] = extra_answers

        # 构造用于聚合的 dict（字段尽量与在线 process_results 对齐）
        result_dict = {
            "target": target,
            "question_type": question_type,
            "llm_match_score": {"llm_match_score": score},
        }
        if extra_answers is not None:
            result_dict["extra_answers"] = extra_answers
        results.append(result_dict)

    # 1) 覆盖写回 samples（带新分数）
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    eval_logger.info(f"Updated samples saved to {sample_file_path}")

    # 2) 重新聚合并覆盖写回 results
    output = openeqa_emeqa_aggregate_results(results)
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    eval_logger.info(f"Updated aggregated results saved to {results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-evaluate OpenEQA results from saved samples (jsonl).")
    parser.add_argument("--base_dir", type=str, default=None, help="日志目录，例如 .../logs/openeqa/<model>/<run_id>/")
    parser.add_argument("--sample_file", type=str, default=None, help="samples_openeqa-emeqa.json 的路径（优先级高于 base_dir）")
    parser.add_argument("--results_file", type=str, default=None, help="results_openeqa-emeqa.json 的路径（优先级高于 base_dir）")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-1120-global")
    parser.add_argument("--openai_seed", type=int, default=1234)
    parser.add_argument("--openai_max_tokens", type=int, default=32)
    parser.add_argument("--openai_temperature", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sample_path = args.sample_file
    results_path = args.results_file

    if args.base_dir and (not sample_path or not results_path):
        base_dir = args.base_dir.rstrip("/")
        if not sample_path:
            sample_path = f"{base_dir}/samples_openeqa-emeqa.json"
        if not results_path:
            results_path = f"{base_dir}/results_openeqa-emeqa.json"

    if not sample_path or not results_path:
        raise ValueError("You must provide --base_dir or both --sample_file and --results_file.")

    post_evaluate_results(
        sample_file_path=sample_path,
        results_file_path=results_path,
        openai_model=args.openai_model,
        openai_seed=args.openai_seed,
        openai_max_tokens=args.openai_max_tokens,
        openai_temperature=args.openai_temperature,
        verbose=args.verbose,
    )
