'''
modified from "https://github.com/google-deepmind/robovqa/blob/main/data_loading_and_eval.ipynb"
bleu = sacrebleu.sentence_bleu(answer, pred_answer)
1. remove discrete question for BELU2-4
2. smooth_method='exp' (Method 7) <- "https://github.com/unira-zwj/PhysVLM/issues/2"
'''
import os
import numpy as np
import re
import pandas as pd

from collections import defaultdict
from openai import OpenAI
from typing import Optional
from tqdm import tqdm
from loguru import logger as eval_logger

# 注意：不要在这里覆盖环境变量，应该从环境变量中读取
# 如果环境变量未设置，client 会在调用时失败，这样更安全
_client = None

def get_openai_client():
    """获取 OpenAI 客户端（延迟初始化，使用环境变量）"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running evaluation: "
                "export OPENAI_API_KEY='your-api-key'"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=api_base if api_base else None
        )
    return _client

# 为了向后兼容，提供一个 client 属性访问方式
# 注意：直接访问 client 会触发延迟初始化
class _ClientProxy:
    def __getattr__(self, name):
        return getattr(get_openai_client(), name)

client = _ClientProxy()

METRICS_FOR_ROBOVQA = {
    # "BELU": "BELU_Eval"
    "llm_match_score": "llm_match"
}
ROBOVQA_QUESTION_TYPES = [
    "past_description:freeform",
    "immediate_planning_with_context20:freeform",
    "affordance:discriminative:discrete:False",
    "success:discrete:True",
    "future_prediction:freeform",
    "affordance:generative:positive:freeform",
    "success:discrete:False",
    "remaining5_planning_with_context20:freeform",
    "planning:freeform",
    "affordance:discriminative:discrete:True",
]

def robovqa_doc_to_visual(doc, dataset_kwargs=None):
    video_path = os.path.join(dataset_kwargs["video_dir"], doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def robovqa_doc_to_text(doc, dataset_kwargs=None):
    question = doc["question"]
    return question

def robovqa_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]
    pred_raw = results[0]
    
    target = doc["answer"]
    question_type = doc["question_type"]

    result_dict = {"target": target}
    result_dict["question_type"] = question_type
    for key, value in METRICS_FOR_ROBOVQA.items():
        pred = pred_raw
        # score = eval(value)(pred, target)
        # doc[key] = {'score': score.score, 'precisions': score.precisions, "bp": score.bp}
        score, judge_error = eval(value)(doc["question"], target, pred, return_error=True)
        doc[key] = {key: score}

        result_dict[key] = doc[key]
        if judge_error:
            result_dict["judge_failed"] = True
            result_dict["judge_error"] = judge_error

    return result_dict

def robovqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}
    # key: {question_type}_{metric_name}
    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in ROBOVQA_QUESTION_TYPES:
            for metric in METRICS_FOR_ROBOVQA.keys():
                metric_data = per_question_type[metric].tolist()
                # avg_score = np.mean([x['score'] for x in metric_data])
                # avg_bp = np.mean([x['bp'] for x in metric_data])
                # avg_precisions = np.mean([x['precisions'] for x in metric_data], axis=0)  # element-wise mean for 4-gram precisions

                # output[f"{question_type}_{metric}"] = avg_score
                # output[f"{question_type}_{metric}-bp"] = avg_bp
                # output[f"{question_type}_{metric}1"] = avg_precisions[0]

                # if 'freeform' in question_type:
                #     output[f"{question_type}_{metric}2"] = avg_precisions[1]
                #     output[f"{question_type}_{metric}3"] = avg_precisions[2]
                #     output[f"{question_type}_{metric}4"] = avg_precisions[3]
                avg_score = np.mean([x[metric] for x in metric_data])
                output[f"{question_type}_{metric}"] = avg_score
    
    metric_to_values = defaultdict(list)
    # 从右到左匹配metric名称，因为metric名称可能包含下划线（如 llm_match_score）
    for key, val in output.items():
        if "_" in key and isinstance(val, (float, int)):
            # 尝试匹配所有已知的metric名称
            matched = False
            for metric in METRICS_FOR_ROBOVQA.keys():
                if key.endswith(f"_{metric}"):
                    metric_name = metric
                    metric_to_values[metric_name].append(val)
                    matched = True
                    break
            # 如果没有匹配到已知metric，使用原来的逻辑（向后兼容）
            if not matched:
                qtype, metric_name = key.rsplit("_", 1)
                metric_to_values[metric_name].append(val)
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_average"] = avg_val
            # 仅 freeform 题型平均（与官方 BLEU 通常只报 freeform 可比）
            freeform_vals = [v for k, v in output.items() if isinstance(v, (float, int)) and "freeform" in k and "_average" not in k and k.endswith(f"_{metric_name}")]
            if freeform_vals:
                output[f"{metric_name}_freeform_average"] = sum(freeform_vals) / len(freeform_vals)

    # overall 与 llm_match_score_average 一致，避免重复计入导致语义混淆
    output["overall"] = output.get("llm_match_score_average", sum([v for v in output.values() if isinstance(v, (float, int))]) / max(1, sum(1 for v in output.values() if isinstance(v, (float, int)))))
    output["100score_overall"] = (output["overall"] - 1) * 25
    eval_logger.info(f"Evaluation results: {output}")
    return output

def BELU_Eval(pred_answer, answer):
    import sacrebleu
    bleu = sacrebleu.sentence_bleu(
        pred_answer, 
        [answer],
        smooth_method='exp'
    )
    return bleu

def llm_match(
        question: str,
        answer: str,
        prediction: str,
        extra_answers = None,
        openai_model: str = "gpt-4o-mini-0718-global",
        openai_seed: int = 1234,
        openai_max_tokens: int = 128,
        openai_temperature: float = 0.2,
        verbose: bool = False,
        max_tries: int = 3,
        return_error: bool = False,
    ):
    import time
    
    if prediction is None:
        return (1, None) if return_error else 1
    
    prompt = load_prompt()

    messages = prepare_openai_messages(
        prompt.format(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
        ),
    )
    
    for attempt in range(max_tries):
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
            if attempt < max_tries - 1:
                eval_logger.warning(f"LLM evaluation failed (attempt {attempt + 1}/{max_tries}): {e}")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                eval_logger.error(f"LLM evaluation failed after {max_tries} attempts: {e}")
                return (1, str(e)) if return_error else 1  # Lowest valid score if all attempts fail.


def load_prompt():
    prompt = """
    You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
    To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
    5 means that the response perfectly matches the answer.
    1 means that the response is completely different from the answer.

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
    return prompt

def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]

def call_openai_api(
    messages: list,
    model: str = "gpt-4o-1120-global",
    seed = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    client = get_openai_client()
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


def parse_score(output: str, tag: str = "Your mark:") -> int:
    """解析 LLM 返回的 1-5 分，支持整数字符串或浮点字符串（如 '4' 或 '4.0'）。"""
    raw = output.strip()
    if not raw:
        return 0
    if raw.isdigit():
        return int(raw)
    try:
        return int(float(raw))
    except ValueError:
        pass
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        raw = output[start_idx:].replace(tag, "").strip()
    else:
        raw = output[start_idx:end_idx].replace(tag, "").strip()
    try:
        return int(float(raw)) if raw else 0
    except ValueError:
        raise ValueError("Invalid output string: {}".format(output))


def clamp_llm_score(score: int) -> int:
    return min(max(int(score), 1), 5)

def post_evaluate_results(sample_file_path, results_file_path):
    """
    根据已经保存的 json 文件「重新评测所有 sample 并聚合」 robovqa 结果。

    注意：这里的“重新评测”语义是——对每一条 sample 都重新打分（即使
    `samples_robovqa.json` 里已经有旧的分数），然后再做一次聚合。
    """
    import json

    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for doc in tqdm(data):
        target = doc["target"]
        question_type = doc["question_type"]
        question = doc.get("question", doc.get("doc", ""))

        result_dict = {"target": target, "question_type": question_type}

        for metric_name, metric_func_name in METRICS_FOR_ROBOVQA.items():
            pred_raw = ""
            if isinstance(doc.get("resps"), list) and doc["resps"] and doc["resps"][0]:
                pred_raw = doc["resps"][0][0]

            eval_func = eval(metric_func_name)
            score_val = eval_func(question, target, pred_raw)
            score_dict = {metric_name: score_val}

            # 更新到当前 doc（方便之后如果还需要再次 post-eval）
            doc[metric_name] = score_dict
            result_dict[metric_name] = score_dict

        results.append(result_dict)

    # 回写 samples_robovqa.json（带上补全后的 metric）
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 按 robovqa 的官方聚合方式重新计算结果
    output = robovqa_aggregate_results(results)

    # 覆盖写入新的聚合结果
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    base_dir = "/your/path/to/embodied-eval-main/logs/robovqa/mimo_embodied/20260123_040922"
    post_evaluate_results(
        sample_file_path=f"{base_dir}/samples_robovqa.json",
        results_file_path=f"{base_dir}/results_robovqa.json"
    )
