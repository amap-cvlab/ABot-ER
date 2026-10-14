import os
import json
import re
import asyncio
import numpy as np
from typing import Dict, Any, List
from openai import OpenAI, AsyncOpenAI
from loguru import logger as eval_logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

# --- GPT Evaluation Client ---
_client = None
_async_client = None
SYSTEM_PROMPT = (
    "You, as an AI system, have been equipped with a vast amount of knowledge "
    "and an understanding of logical reasoning. Your task is to use these "
    "capabilities to assess academic responses."
)

def get_openai_client(base_url=None):
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    actual_base = base_url if base_url else os.getenv("OPENAI_API_BASE")
    if actual_base:
        actual_base = actual_base.rstrip("/")
    if _client is None or base_url:
        return OpenAI(api_key=api_key, base_url=actual_base)
    return _client

def get_async_openai_client(base_url=None):
    api_key = os.getenv("OPENAI_API_KEY")
    actual_base = base_url if base_url else os.getenv("OPENAI_API_BASE")
    if actual_base:
        actual_base = actual_base.rstrip("/")
    return AsyncOpenAI(api_key=api_key, base_url=actual_base)

# --- Big Capability Mapping ---
BIG_CAPABILITY_MAP = {
    "object_perception": ["object_type", "object_property", "object_state"],
    "spatial_perception": ["spatial_perception"],
    "temporal_perception": ["temporal_perception", "action_perception"],
    "embodied_knowledge": ["affordance", "world_knowledge"],
    "embodied_reasoning": ["closed-loop_planning", "open-loop_planning", "task_related_object", "situated_reasoning"],
}

# 12 个细粒度能力维度，用于 overall = 12 维简单平均
UNIEQA_FINE_DIMENSIONS = [
    "object_type", "object_property", "object_state",
    "spatial_perception", "action_perception", "temporal_perception",
    "affordance", "world_knowledge", "task_related_object",
    "situated_reasoning", "closed-loop_planning", "open-loop_planning",
]

# --- UniEQA Specific Prompts (aligned with official UniEQA evaluator) ---
UNIEQA_PROMPTS = {
    "open-ended": {
        "object_type": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to object type recognition. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the model's prediction is not exactly the answer, but the two objects are similar, especially in terms of appearance and function, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(0.5 or 0)"}'. The correct response and student's answer is provided below.

Example1:
The Student's Answer is:"armchair"
The Correct Answer is:"sofa"
Your output:
{
"Assessment": "The student's answer 'armchair' is similar with the correct answer 'sofa' in terms of appearance and function, so the score will be 0.5.",
 "Final Score": "0.5"
}

Example2:
The Student's Answer is:"fridge"
The Correct Answer is:"refrigerator."
Your output:
{
"Assessment": "The student's answer is semantically correct, as fridge and refrigerator are different names of the same type of object, so its score will be '1'.",
 "Final Score": "1"
}

Example3:
The Student's Answer is:"lamp."
The Correct Answer is:"hat."
Your output:
{
"Assessment": "The student's response is wrong. Therefore, it gets a score of '0'.",
 "Final Score": "0"
}

Your Turn:""",
        "affordance": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related with affordances, which are the characteristics or properties of an object that suggest how it can be used. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: Focus on the key points of the answer, while the differences in pronouns should be ignored. If the student's answer is correct semantically, assign a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Otherwise, assign a score of '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.

Example1:
The Student's Answer is:"You wear this to protect your hands"
The Correct Answer is:"to protect my hands"
Your output:
{
"Assessment": "The student's answer aligns with the correct answer 'protect hands' semantically, and the differences in pronouns should be ignored, so it's semantically correct. Its score will be 1.",
 "Final Score": "1"
}

Example2:
The Student's Answer is:"yes"
The Correct Answer is:"yes."
Your output:
{
"Assessment": "The student's answer is completely consistent with the correct answer, so the score will be '1'.",
 "Final Score": "1"
}

Example3:
The Student's Answer is:"yes."
The Correct Answer is:"no."
Your output:
{
"Assessment": "The student's response is wrong. Therefore, it gets a score of '0'.",
 "Final Score": "0"
}

Your Turn:""",
        "object_state": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to identifying the state of objects, such as open or close. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "object_property": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to identifying the property of objects, such as color. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "spatial_perception": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to spatial reasoning. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is correct semantically, assign a score of '1'. If the student's answer is correct semantically, but contains further information, assign a score of '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. Otherwise, assign a score of '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(0.5 or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(0.5 or 0)"}'.

Example1:
The Student's Answer is:"The microwave is located in the kitchen area. It appears built into the cabinetry above the countertop, near the oven."
The Correct Answer is:"Above the oven"
Your output:
{
"Assessment": "The student's answer partially overlaps with the correct answer, but 'near' is not as accurate as 'above', so the score will be 0.5.",
 "Final Score": "0.5"
}

Example2:
The Student's Answer is:"The doll is on top of the wooden dresser near the entrance with the arched wooden doors."
The Correct Answer is:"On top of the light brown wooden cabinet"
Your output:
{
"Assessment": "Although 'wooden cabinet' is referred to as 'wooden dresser', the student's overall answer is correct, that is, on top of a certain wooden item, so it gets 1 point.",
 "Final Score": "1"
}

Example3:
The Student's Answer is:"The standing lamp is located near the entrance of what appears to be a bathroom or a hallway. It is on the left side, adjacent to the doorway."
The Correct Answer is:"next to the bed in the bedroom."
Your output:
{
"Assessment": "The student's answer describes the location of a standing lamp, indicating it is near the entrance of what seems to be a bathroom or hallway and on the left side adjacent to a doorway. However, this description does not align with the correct answer, which specifies that the lamp is next to the bed in the bedroom. Consequently, the student's answer is inaccurate as it does not match the specified location of the lamp. Therefore, the student receives a score of 0.",
 "Final Score": "0"
}

Your Turn:""",
        "task_related_object": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to identifying task-relevant objects. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "temporal_perception": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to temporal perception. For prediction tasks, only predict the change after executing one step of action, and it should be accurate and concise. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "action_perception": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to recognizing current action. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "world_knowledge": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to common knowledge. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "situated_reasoning": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to situated reasoning. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "closed-loop_planning": """Now, you will be presented with a correct response, and a student's answer to an open-ended question about generating task plans. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the plan provided by the student matches the correct answer, assign a score of '1'. If the student provides additional but reasonable plan and includes the content of the answer, assign a score of '0.5'. Otherwise, assign a score of '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1 (0.5 or 0)' as your final judgment. Output in JSON format, for example: '{"Assessment": "xxxxx", "Final Score": "1(0.5 or 0)"}'.

Example1:
The Student's Answer is:"To reach your main goal of dropping a cooked apple slice in the fridge, follow these steps based on your current environment:

1. **Current Step Analysis:**
   - You have already picked up the knife.
   - You are at the sink, where you can slice the apple.

2. **Next Step Decision:**
   - Slice the apple in the sink. Use the knife to cut a slice of the apple you have with you.

3. **Subsequent Actions:**
   - Cook the apple slice using an appropriate method available in your environment (e.g., using a stove, microwave, or oven if accessible).
   - Once cooked, find the fridge in your environment and place the cooked apple slice inside it to complete your main goal."
The Correct Answer is:"turn around to the fridge."
Your output:
{
"Assessment": "The student's answer partially aligns with the correct answer 'turn to the fridge'. The additional information is reasonable but not accurate, so the score will be 0.5.",
 "Final Score": "0.5"
}

Example2:
The Student's Answer is:"Based on the current images and steps completed:

You have already walked to the sofa and picked up the cushion. Now, you need to:

**Current Step:**
- Turn around and walk to the tall lamp by the picture on the wall.

Proceed with that action to reach the final goal."
The Correct Answer is:"Turn on the tall lamp."
Your output:
{
"Assessment": "The student's answer indicates a sequence of actions related to walking to a lamp, but it does not correctly address the task of turning on the lamp, which is the primary objective of the correct answer. While the student correctly outlines steps preceding the final action, the ultimate goal of the task is not achieved, as they described walking to the lamp rather than actually turning it on. Therefore, the answer does not align with the correct response in essence.",
 "Final Score": "0"
}

Example3:
The Student's Answer is:"You are at the final step. To achieve your goal, make sure to raise the bowl up near the lamp and then turn on the lamp."
The Correct Answer is:"Turn on the lamp."
Your output:
{
"Assessment": "The student's response matches the correct answer 'turn on the lamp'. Therefore, it warrants a score of '1'.",
 "Final Score": "1"
}

Your Turn:""",
        "open-loop_planning": """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to open-loop planning. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. If the student's answer partially overlaps with the correct answer and does not conflict with it, assign a score of '0.5'. If the answer is incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'. The correct response and student's answer is provided below.""",
        "default": """Now, you will be presented with a correct response, and a student's answer to an open-ended question. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)'. Output in JSON format. For instance: {"Assessment": "xxxxx", "Final Score": "1(or 0)"}. The correct response and student's answer is provided below."""
    },
    "multi-choice": {
        "closed-loop_planning": """Now, you will be presented with the correct answer and a student's response to a multiple-choice question about generating task plans, where multiple options are provided but only one is correct. Your task is to first match the student's answer to the most relevant option and then evaluate its correctness based on the following rules: If the plan provided by the student matches the correct answer, assign a score of '1'. Otherwise, assign a score of '0'. Begin your evaluation with an 'Assessment:' paragraph, where you explain your reasoning. Conclude with 'Final Score: 1 (or 0)' as your final judgment. Output in JSON format, for example: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'.

Example1:
The Student's Answer is:"Based on your observation in the last frame, the next logical action would be:

D. pick up potato

This allows you to proceed with the peeling and chopping processes."
The Correct Answer is:"D. pick up potato"
Your output:
{
"Assessment": "The student's response clearly states that the next logical action based on observation is to 'pick up potato', which matches the correct answer provided. Therefore, it warrants a score of '1'.",
 "Final Score": "1"
}

Example2:
The Student's Answer is:"Based on your description and the goal of cleaning and replacing the chopping blades, the next best action is:

**D. remove chopping blade**

This step is necessary to clean it before replacement."
The Correct Answer is:"A. open cupboard"
Your output:
{
"Assessment": "The student's response identifies option 'D' (remove chopping blade) as the necessary next action based on the described goal, which does not align with the correct answer, 'A' (open cupboard), which indicates that the student's answer is incorrect. Therefore, it results in a score of 0.",
 "Final Score": "0"
}

Your Turn:""",
        "default": """Now, you will be presented with the correct answer and a student's response to a multiple-choice question, where multiple options are provided but only one is correct. Your task is to first match the student's answer to the most relevant option and then evaluate its correctness based on the following rules: If the most relevant option corresponds to the correct answer, assign a score of '1'. If the most relevant option does not match the correct answer, assign a score of '0'. Begin your evaluation with an 'Assessment:' paragraph, where you explain your reasoning. Conclude with 'Final Score: 1 (or 0)' as your final judgment. Output in JSON format, for example: '{"Assessment": "xxxxx", "Final Score": "1(or 0)"}'.

Example1:
The Student's Answer is:"yes"
The Correct Answer is:"yes"
Your output:
{
"Assessment": "The student's answer matches the correct answer 'yes', so the score will be 1.",
 "Final Score": "1"
}

Example2:
The Student's Answer is:"The correct choice is A"
The Correct Answer is:"A"
Your output:
{
"Assessment": "The most relevant option of the student's answer is 'A', which matches the correct answer, so it's a '1'.",
 "Final Score": "1"
}

Example3:
The Student's Answer is:"C"
The Correct Answer is:"D"
Your output:
{
"Assessment": "The student's answer 'C' is wrong. The score will be '0'.",
 "Final Score": "0"
}

Your Turn:"""
    },
    "multi-selection": {
        "default": """Now, you will be given the correct answer(s) and a student's response to a multiple-selection question, where multiple options are provided, and one or more may be correct. Your task is to first match the student's answer to the most relevant option(s) and then evaluate its correctness based on the following rules: If the student's answer includes all correct options and no incorrect ones, assign a score of '1'. If the student's answer contains any incorrect options, assign a score of '0'. If the student's answer includes some but not all of the correct options and no incorrect ones, assign a score of '0.5'. Begin your evaluation with an 'Assessment:' paragraph, where you explain your reasoning. Conclude with 'Final Score: 1, 0.5, or 0' as your final judgment. Output in JSON format, for example: '{"Assessment": "xxxxx", "Final Score": "1(0.5, or 0)"}'.

Example1:
The Student's Answer is:"openable"
The Correct Answer is:"openable, breakable"
Your output:
{
"Assessment": "The student's answer contains right answer 'openable', but lacks 'breakable', so the score will be 0.5.",
 "Final Score": "0.5"
}

Example2:
The Student's Answer is:"A, C"
The Correct Answer is:"A C"
Your output:
{
"Assessment": "The student's answer includes all correct options (A and C) and no incorrect ones, so it's a '1'.",
 "Final Score": "1"
}

Example3:
The Student's Answer is:"A, C"
The Correct Answer is:"C, D"
Your output:
{
"Assessment": "The student's answer contains right answer 'openable', and the wrong option 'A', so it's wrong. The score will be '0'.",
 "Final Score": "0"
}

Your Turn:"""
    },
    "default": """Now, you will be presented with a correct response, and a student's answer to some question. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)', which is your final judgement. Output in JSON format. For instance: {"Assessment": "xxxxx", "Final Score": "1(or 0)"}. The correct response and student's answer is provided below."""
}

# --- Framework Hooks (Synchronous) ---

def unieqa_doc_to_visual(doc, dataset_kwargs=None):
    from PIL import Image
    import os as _os
    visuals = []
    image_root = (dataset_kwargs or {}).get('image_root', '')
    if 'images' in doc and isinstance(doc['images'], list):
        img_paths = doc['images']
        
        # 智能采样逻辑：
        # 1. 如果图片数量在 12 张以内（绝大部分 UniEQA 样本是 8 张），则全量保留，不做采样
        # 2. 只有对于图片数量很多的场景（如 ScanNet 场景有 600 张），才采样到 12 张
        max_frames = 15
        if len(img_paths) > max_frames:
            import numpy as np
            indices = np.linspace(0, len(img_paths) - 1, max_frames, dtype=int)
            img_paths = [img_paths[i] for i in indices]
            
        for img_path in img_paths:
            try:
                if hasattr(img_path, 'convert'):
                    visuals.append(img_path.convert('RGB'))
                else:
                    resolved = img_path
                    if image_root and not _os.path.exists(img_path):
                        norm = img_path.replace('\\', '/')
                        for pn in ['Part1', 'Part2', 'Part3', 'Part4', 'Part5', 'Part6']:
                            anchor = '/' + pn + '/'
                            if anchor in norm:
                                idx = norm.index(anchor)
                                resolved = _os.path.join(image_root, norm[idx+1:])
                                break
                    visuals.append(Image.open(resolved).convert('RGB'))
            except Exception as e: eval_logger.warning(f"Failed to load image {img_path}: {e}")
        return visuals
    if 'image' in doc:
        img = doc['image']
        if hasattr(img, 'convert'): return [img.convert('RGB')]
        elif isinstance(img, str):
            try:
                resolved = img
                if image_root and not _os.path.exists(img):
                    norm = img.replace('\\', '/')
                    for pn in ['Part1', 'Part2', 'Part3', 'Part4', 'Part5', 'Part6']:
                        anchor = '/' + pn + '/'
                        if anchor in norm:
                            idx = norm.index(anchor)
                            resolved = _os.path.join(image_root, norm[idx+1:])
                            break
                return [Image.open(resolved).convert('RGB')]
            except: pass
    return []

def unieqa_doc_to_text(doc, dataset_kwargs=None):
    return doc.get('question', "Please answer the question about this image.")

def unieqa_process_results(doc, results, dataset_kwargs=None):
    pred_raw = results[0] if results else ""
    target = doc.get('answer', "")
    dimension = doc.get('capability_dimension', 'default')
    question_type = doc.get("question_type", "open-ended")
    score, assessment, judge_error = llm_match(target, pred_raw, dimension, question_type, return_error=True)
    result = {
        "target": target,
        "prediction": pred_raw,
        "dimension": dimension,
        "question_type": question_type,
        "score": score,
        "assessment": assessment,
    }
    if judge_error:
        result["judge_failed"] = True
        result["judge_error"] = judge_error
    return result

def _resolve_prompt(dimension, question_type):
    if not question_type:
        question_type = "open-ended"
    prompt_group = UNIEQA_PROMPTS.get(question_type, UNIEQA_PROMPTS["default"])
    if isinstance(prompt_group, dict):
        return prompt_group.get(dimension, prompt_group.get("default", UNIEQA_PROMPTS["default"]))
    return prompt_group


def llm_match(target, prediction, dimension, question_type, return_error=False):
    client = get_openai_client()
    prompt_template = _resolve_prompt(dimension, question_type)
    prompt = prompt_template + f"\nThe Student's Answer is:\"{prediction}\"\nThe Correct Answer is:\"{target}\""
    try:
        scores = []
        assessments = []
        for _ in range(3):
            response = client.chat.completions.create(
                model="gpt-4o-mini-0718-global",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = json.loads(response.choices[0].message.content)
            score_str = str(content.get("Final Score", "0"))
            score_match = re.search(r"([0-1](\.\d+)?)", score_str)
            if not score_match:
                raise ValueError(f"Invalid judge score from output: {response.choices[0].message.content!r}")
            score = float(score_match.group(1))
            scores.append(score)
            assessments.append(content.get("Assessment", ""))
        avg_score = sum(scores) / len(scores) if scores else 0.0
        assessment_blob = json.dumps({"scores": scores, "assessments": assessments}, ensure_ascii=False)
        return (avg_score, assessment_blob, None) if return_error else (avg_score, assessment_blob)
    except Exception as e:
        error = str(e)
        result = (0, f"Error: {error}")
        return (*result, error) if return_error else result

def unieqa_aggregate_results(results):
    if not results: return {"overall": 0.0}
    dim_scores = {}
    for r in results:
        dim = r.get('dimension', 'default')
        if dim not in dim_scores: dim_scores[dim] = []
        dim_scores[dim].append(r.get('score', 0))
    output = {}
    for dim, scores in sorted(dim_scores.items()):
        avg = sum(scores) / len(scores)
        output[f"{dim}_score"] = avg
        print(f"  {dim:25s}: {avg:.4f} ({len(scores)} samples)")
    
    # 大能力项统计
    big_scores = {}
    for big_cap, small_caps in BIG_CAPABILITY_MAP.items():
        vals = [output[f"{s}_score"] for s in small_caps if f"{s}_score" in output]
        if vals:
            v = sum(vals) / len(vals)
            output[f"big_{big_cap}_score"] = v
            big_scores[big_cap] = v
    if big_scores:
        print("\n" + "-" * 40 + "\nUniEQA 聚合评分结果 (按大能力项):\n" + "-" * 40)
        for big_cap, v in sorted(big_scores.items()):
            print(f"  {big_cap:25s}: {v:.4f}")

    # overall = 12 个细粒度能力维度的简单平均（不再按样本数加权）
    dim_avgs = [output[f"{d}_score"] for d in UNIEQA_FINE_DIMENSIONS if f"{d}_score" in output]
    overall = sum(dim_avgs) / len(dim_avgs) if dim_avgs else 0.0
    output.update({"overall": overall, "score_average": overall})
    print("-" * 40 + f"\n  {'Overall':25s}: {overall:.4f} (12-dim mean)\n" + "=" * 40 + "\n")
    return output

# --- Async Utils for Post-Evaluation ---

async def llm_match_async(target, prediction, dimension, question_type, semaphore):
    async with semaphore:
        prompt_template = _resolve_prompt(dimension, question_type)
        prompt = prompt_template + f"\nThe Student's Answer is:\"{prediction}\"\nThe Correct Answer is:\"{target}\""
        base_url = os.getenv("OPENAI_API_BASE", "").rstrip("/")
        urls_to_try = [base_url]
        if base_url.endswith("/v1"): urls_to_try.append(base_url[:-3].rstrip("/"))
        else: urls_to_try.append(base_url + "/v1")
        scores = []
        assessments = []
        for _ in range(3):
            last_error = None
            for attempt in range(3):
                for url in urls_to_try:
                    try:
                        client = get_async_openai_client(base_url=url)
                        response = await client.chat.completions.create(
                            model="gpt-4o-mini-0718-global",
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"},
                            timeout=30
                        )
                        content = json.loads(response.choices[0].message.content)
                        score_match = re.search(r"([0-1](\.\d+)?)", str(content.get("Final Score", "0")))
                        score = float(score_match.group(1)) if score_match else 0.0
                        scores.append(score)
                        assessments.append(content.get("Assessment", ""))
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        if "429" in str(e):
                            eval_logger.warning(f"Rate limited (429), retrying {attempt+1}/3...")
                            await asyncio.sleep((attempt + 1) * 2)
                            continue
                        eval_logger.error(f"LLM eval failed for sample. Error: {e}")
                        break
                if last_error is None:
                    break
            if last_error is not None and not scores:
                return 0, f"Error: {last_error}"
        avg_score = sum(scores) / len(scores) if scores else 0.0
        assessment_blob = json.dumps({"scores": scores, "assessments": assessments}, ensure_ascii=False)
        return avg_score, assessment_blob

async def post_evaluate_results_async(base_dir, force=False):
    sample_file_path = os.path.join(base_dir, "samples_unieqa.json")
    inference_file_path = os.path.join(base_dir, "inference_unieqa.json")
    input_file = sample_file_path if os.path.exists(sample_file_path) else inference_file_path
    if not os.path.exists(input_file): return
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    eval_logger.info(f"Loaded {len(data)} samples. Starting async evaluation...")
    dataset = None
    if data and ("target" not in data[0] or "dimension" not in data[0]):
        try:
            from datasets import load_from_disk
            # 修正为最新的 v3 数据集路径
            ds_path = "/your/path/to/embodied-eval-main/embodied_eval/data/unieqa/unieqa_full_multi_v3"
            dataset = load_from_disk(ds_path)
            if 'train' in dataset: dataset = dataset['train']
            eval_logger.info(f"Loaded GT dataset from {ds_path}")
        except Exception as e: 
            eval_logger.warning(f"Failed to load GT dataset: {e}")
            pass
    semaphore = asyncio.Semaphore(5)
    tasks = []
    for doc in data:
        pred_raw = doc.get("prediction", "")
        if not pred_raw and "resps" in doc and doc["resps"]:
            try:
                r = doc["resps"][0]
                pred_raw = r[0] if isinstance(r, list) else r
            except: pass
        target = doc.get("target", doc.get("answer", ""))
        dimension = doc.get("dimension", doc.get("capability_dimension", "default"))
        question_type = doc.get("question_type")
        if (not target or dimension == "default") and dataset is not None:
            idx = doc.get("doc_id")
            if idx is not None and idx < len(dataset):
                target = dataset[idx].get("answer", "")
                dimension = dataset[idx].get("capability_dimension", "default")
                if question_type is None:
                    question_type = dataset[idx].get("question_type")
        if not question_type:
            question_type = "open-ended"
        score, assessment = doc.get("score"), doc.get("assessment")
        should_re_eval = force or score is None or (isinstance(score, (int, float)) and score == 0 and (not assessment or "blocked" in str(assessment).lower() or "error" in str(assessment).lower()))
        if should_re_eval:
            tasks.append((doc, target, pred_raw, dimension, question_type))
        else:
            doc.update({
                "target": target,
                "dimension": dimension,
                "prediction": pred_raw,
                "question_type": question_type,
            })
    if tasks:
        async def eval_task(tup):
            doc, target, pred, dim, qtype = tup
            s, a = await llm_match_async(target, pred, dim, qtype, semaphore)
            doc.update({
                "target": target,
                "dimension": dim,
                "prediction": pred,
                "question_type": qtype,
                "score": s,
                "assessment": a,
            })
        await tqdm_async.gather(*(eval_task(t) for t in tasks), desc="Evaluating")
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for d in data: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    output = unieqa_aggregate_results([{"dimension": d["dimension"], "score": d["score"]} for d in data])
    with open(os.path.join(base_dir, "results_unieqa.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    eval_logger.info("Done.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(post_evaluate_results_async(args.base_dir, force=args.force))
