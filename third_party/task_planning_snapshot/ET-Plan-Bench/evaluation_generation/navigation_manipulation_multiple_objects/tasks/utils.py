from openai import OpenAI
import os
import subprocess
import json
import matplotlib.pyplot as plt
import time
import jsonlines

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

def get_response(messages, model="gpt-4-1106-preview"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def show_image_list(images):
    num = len(images)
    fig = plt.figure(figsize=(18, 9))
    for i, image in enumerate(images):
        ax = plt.subplot(1, num, i + 1)
        ax.imshow(image[:, :, ::-1])
    plt.show()


def start_unity():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    exec_path = os.path.join(
        root_dir,
        "simulator/virtualhome/virtualhome/simulation/unity_simulator_2.3.0/linux_exec.v2.3.0.x86_64",
    )
    os.environ["DISPLAY"] = ":1"
    p = subprocess.Popen(
        [
            exec_path,
            "-batchmode",
        ],
        start_new_session=True,
    )
    time.sleep(4)
    return p


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in jsonlines.Reader(f):
            data.append(line)
    return data


def auto_kill_unity(kill_before_return=True):
    def decorator(func):
        def new_func(*args, **kwargs):
            kill_cmd = "ps -ef | grep linux_exec | awk '{print $2}' | xargs kill -9"
            try:
                result = func(*args, **kwargs)
            except KeyboardInterrupt as e:
                os.system(kill_cmd)
                print(f"kill unity automaticly for keyboard interrupt.")
                raise e
            except Exception as e:
                os.system(kill_cmd)
                print(f"kill unity automaticly for exception.")
                raise e
            else:
                if kill_before_return:
                    os.system(kill_cmd)
                    print(f"kill unity before function return.")
                return result

        return new_func

    return decorator
