import re

from openai import OpenAI

from utils.helper import retry_fn
from constant.llm import GPT_KEY, MODEL
from constant.dirs import LLAVA_MODEL_BASE_DIR, LLAVA_MODEL_PATH, LLAVA_IMAGE_PATH

client = OpenAI(
    api_key=GPT_KEY,
)



def llava_single(text_question):
    from llava.eval.run_llava import eval_model_CA
    from data_gen.visible_single_task_type import tokenizer, model, image_processor, model_name

    args = type('Args', (), {
    "model_path": LLAVA_MODEL_PATH,
    "model_base": LLAVA_MODEL_BASE_DIR, # None
    "model_name": model_name,
    "query": text_question,
    "conv_mode": None,
    "image_file": LLAVA_IMAGE_PATH,
    "sep": ",",
    "temperature": 0.3,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
    })()

    return eval_model_CA(args, tokenizer, model, image_processor).lower()


def query_gpt_single(
    prompt,
    model=MODEL,
    max_tokens=2048,
    temperature=None,
    user_role=True,
    full_response=False,
):
    def f(prompt, user_role=True):
        if user_role:
            message = [{"role": "user", "content": prompt}]
        else:
            message = prompt
        response = client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if full_response:
            return response

        return response.choices[0].message.content

    response = retry_fn(lambda: f(prompt, user_role), max_failures=10, sleep_time=5)

    return response


def get_response(messages, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def process_gpt_answer(answer, to_list=False):
    def removals(text):
        text = text.replace("'", "")
        text = text.replace('"', "")
        text = text.replace("[", "")
        text = text.replace("]", "")

        return text

    answer = answer.replace("\n", "")

    if not to_list:
        try:
            processed_answer = re.search(r"\[.*\]", answer).group()
        except Exception:
            return None
        processed_answer = removals(processed_answer)
    else:
        processed_answer = []
        answers = re.findall(r"\[.*?\]", answer)

        if len(answers) == 0:
            return None
        else:
            for answer in answers:
                processed_answer.extend(answer.split(","))

        processed_answer = [removals(ans).strip() for ans in processed_answer]

    return processed_answer
