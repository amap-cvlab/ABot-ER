import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import argparse
import random
import json
from collections import defaultdict

from utils.robot_task import nav_manipulate_single_ground_truth
from utils.unity import auto_kill_unity
from utils.io import json_iter, load_json, save_json
from utils.log import get_logger
from utils.helper import get_task_ids
from envs.env import VirtualHomeEnv, TaskEnv
from constant.dirs import (
    GENERATED_DATA_GROUND_TRUTH_DIR,
    GENERATED_TASK_SOURCE_DIR,
    GENERATED_TASK_FILTERED_DIR,
)


logger = get_logger(__file__)

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--task_name_id", type=int, help = "0 to 3")
parser.add_argument("-e", "--env_id", type=int)

args = parser.parse_args()

IDX = args.task_name_id

config = {
    "port_num": 4915,
    "task_name": ["nav_on", "nav_inside", "nav_on_cons", "nav_inside_cons"][IDX],
    "data_to_generate": 200,
    "increment": False,
    "resume": True,
    "save_img": True,
    "save_success_task": True,
    "init_room_pool": ["bathroom", "bedroom", "kitchen", "livingroom"],
}

logger.info("CONFIGS: %s", config)

PORT_NUM = config["port_num"]
TASK_NAME = config["task_name"]
DATA_TO_GENERATE = config["data_to_generate"]
INCREMENT = config["increment"]
RESUME = config["resume"]
SAVE_IMG = config["save_img"]
SAVE_SUCCESS_TASK = config["save_success_task"]
init_room_pool = config["init_room_pool"]

if SAVE_SUCCESS_TASK:
    filtered_task_save_dir = os.path.join(GENERATED_TASK_FILTERED_DIR, TASK_NAME)
    os.makedirs(filtered_task_save_dir, exist_ok=True)

source_task_dir = os.path.join(GENERATED_TASK_SOURCE_DIR, TASK_NAME)

# files_env = [
#     file for file in os.listdir(source_task_dir) if file.endswith((".json", ".jsonl"))
# ]
# files_env.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
file_env = os.path.join(source_task_dir, f"env_{args.env_id}.json")

logger.info("files to process: %s", file_env)


progress_file_path = os.path.join(source_task_dir, "progress")
if os.path.exists(progress_file_path):
    progress = load_json(progress_file_path)
    progress = defaultdict(int, progress)
    if RESUME:
        logger.info("resuming from: %s", progress)
else:
    progress = defaultdict(int)


def nav_manipulation_single_from_task(task_name, env, save_dir, task_complete):
    task_id = task_complete["index"]
    if "cons" in task_name:
        target_object = task_complete["id1"]
        target_container_object = task_complete["id2"]
    else:
        target_object = task_complete["object1"]
        target_container_object = task_complete["object2"]

    task_env = TaskEnv(env, task_complete, task_id, save_dir, use_partial_graph=False)

    random_init_room = random.choice(init_room_pool)
    logger.info("initial room: %s", random_init_room)
    task_env.reset_env_room(room=random_init_room, port_num=PORT_NUM)

    if "nav_on" in task_name:
        success = nav_manipulate_single_ground_truth(
            env, task_env, target_object, target_container_object, save_img=SAVE_IMG
        )
    elif "nav_inside" in task_name:
        success = nav_manipulate_single_ground_truth(
            env, task_env, target_object, target_container_object, save_img=SAVE_IMG
        )

    return success


@auto_kill_unity(kill_before_return=True)
def for_loop_data_generation():
    filtered_task_file_path = os.path.join(filtered_task_save_dir, file_env)

    env_id = file_env.split("_")[-1].split(".")[0]
    source_counter = progress[env_id]

    save_dir = os.path.join(
        GENERATED_DATA_GROUND_TRUTH_DIR, TASK_NAME, f"env_{env_id}"
    )
    os.makedirs(save_dir, exist_ok=True)

    if INCREMENT:
        success_counter = 0
    else:
        success_counter = len(
            [
                file
                for file in os.listdir(save_dir)
                if file.endswith((".json", ".jsonl"))
            ]
        )

    if DATA_TO_GENERATE != -1 and success_counter >= DATA_TO_GENERATE:
        logger.info(
            "DATA_TO_GENERATE (%s) reached for file: %s", DATA_TO_GENERATE, file_env
        )
        return

    source_task_file_path = os.path.join(source_task_dir, file_env)

    task_id_continue = get_task_ids(save_dir)

    if RESUME:
        resume_from = max(task_id_continue + [source_counter])

    logger.info("processing: %s", file_env)

    env = VirtualHomeEnv(port=str(PORT_NUM))
    logger.info("initial port after VirtualHomeEnv: %s", PORT_NUM)

    for task_complete in json_iter(source_task_file_path):
        task_id = task_complete["index"]
        # if task_id in task_id_continue:
        if (RESUME and task_id < resume_from) or task_id in task_id_continue:
            print("[SKIP] task", task_id)
            continue

        logger.info("task_id: %s", task_id)
        logger.info("task_complete: %s", task_complete)

        success = nav_manipulation_single_from_task(
            TASK_NAME, env, save_dir, task_complete
        )

        if task_id >= source_counter:
            progress[env_id] = task_id + 1
            save_json(progress, progress_file_path)

        if success:
            success_counter += 1
            if SAVE_SUCCESS_TASK:
                with open(filtered_task_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(task_complete, ensure_ascii=False) + "\n")

            if DATA_TO_GENERATE != -1 and success_counter >= DATA_TO_GENERATE:
                logger.info(
                    "DATA_TO_GENERATE (%s) reached for file: %s",
                    DATA_TO_GENERATE,
                    file_env,
                )

                env.close()
                break

    if success_counter < DATA_TO_GENERATE:
        logger.warning(
            "%s success from file: %s, failed to reach %s",
            success_counter,
            file_env,
            DATA_TO_GENERATE,
        )

    env.close()


if __name__ == "__main__":
    try:
        for_loop_data_generation()
    except KeyboardInterrupt:
        sys.exit(0)
