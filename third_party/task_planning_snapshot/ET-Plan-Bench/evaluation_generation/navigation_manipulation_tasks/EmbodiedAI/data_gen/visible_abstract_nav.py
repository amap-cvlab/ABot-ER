import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import random
from collections import defaultdict

from utils.robot_task import nav_simple_abstract
from utils.unity import auto_kill_unity
from utils.io import json_iter, load_json, save_json
from utils.log import get_logger
from utils.helper import get_task_ids, count_success
from envs.env import VirtualHomeEnv, TaskEnv
from constant.dirs import (
    GENERATED_DATA_VISIBLE_DIR,
    GENERATED_DATA_VISIBLE_LLAVA_DIR,
    GENERATED_TASK_ABSTRACT_DIR,
)


logger = get_logger(__file__)


config = {
    "port_num": 6789,
    "data_to_generate": -1,
    "model": "gpt",
    "stop_metric": "process",
    "increment": False,
    "resume": True,
    "save_img": False,
    "init_room_pool": ["bathroom", "bedroom", "kitchen", "livingroom"],
}

logger.info("CONFIGS: %s", config)

PORT_NUM = config["port_num"]
DATA_TO_GENERATE = config["data_to_generate"]
MODEL = config["model"]
STOP_METRIC = config["stop_metric"]
INCREMENT = config["increment"]
RESUME = config["resume"]
SAVE_IMG = config["save_img"]
init_room_pool = config["init_room_pool"]

abstract_task_dir = os.path.join(GENERATED_TASK_ABSTRACT_DIR, "nav")

envs_test_eval = [37, 0, 32, 39, 19, 20, 48, 49, 17, 26]
# envs_test_eval.sort()
files_env = [f"env_{env_id}.json" for env_id in envs_test_eval]

envs_skip = [9]

logger.info("files to process: %s", files_env)


progress_file_path = os.path.join(abstract_task_dir, f"progress_{MODEL}")
if os.path.exists(progress_file_path):
    progress = load_json(progress_file_path)
    progress = defaultdict(int, progress)
    if RESUME:
        logger.info("resuming from: %s", progress)
else:
    progress = defaultdict(int)


success_count_path = os.path.join(abstract_task_dir, f"success_{MODEL}")
if os.path.exists(success_count_path):
    success_count_dict = load_json(success_count_path)
    success_count_dict = defaultdict(int, success_count_dict)
else:
    success_count_dict = defaultdict(int)


@auto_kill_unity(kill_before_return=True)
def for_loop_data_generation():

    for file_env in files_env:

        env_id = file_env.split("_")[-1].split(".")[0]

        if int(env_id) in envs_skip:
            logger.info("env skip: %s", env_id)
            continue

        source_counter = progress[env_id]

        if MODEL == "gpt":
            save_dir = os.path.join(
                GENERATED_DATA_VISIBLE_DIR, "abstract_nav", f"env_{env_id}"
            )
        else:
            save_dir = os.path.join(
                GENERATED_DATA_VISIBLE_LLAVA_DIR, "abstract_nav", f"env_{env_id}"
            )
        os.makedirs(save_dir, exist_ok=True)

        if INCREMENT:
            success_counter = 0
        else:
            success_counter = success_count_dict[env_id]
            if success_counter == 0:
                success_counter = count_success(save_dir)
                success_count_dict[env_id] = success_counter
                save_json(success_count_dict, success_count_path)

        process_counter = len(
            [file for file in os.listdir(save_dir) if file.endswith(("json", "jsonl"))]
        )

        if DATA_TO_GENERATE != -1:
            if (STOP_METRIC == "success" and success_counter >= DATA_TO_GENERATE) or (
                STOP_METRIC == "process" and process_counter >= DATA_TO_GENERATE
            ):
                logger.info(
                    "DATA_TO_GENERATE (%s) reached for file: %s",
                    DATA_TO_GENERATE,
                    file_env,
                )
                continue

        filtered_task_file_path = os.path.join(abstract_task_dir, file_env)

        task_id_continue = get_task_ids(save_dir)

        if RESUME:
            resume_from = max(task_id_continue + [source_counter])
        else:
            resume_from = 0

        logger.info("processing: %s", file_env)

        env = VirtualHomeEnv(port=str(PORT_NUM))
        logger.info("initial port after VirtualHomeEnv: %s", PORT_NUM)

        for task_count, task_complete in enumerate(json_iter(filtered_task_file_path)):
            task_id = task_complete["index"]
            # if task_id in task_id_continue:
            if (RESUME and task_id < resume_from) or task_id in task_id_continue:
                print("[SKIP] task", task_id)
                continue

            logger.info("task_count: %s, task_id: %s", task_count, task_id)
            logger.info("task_complete: %s", task_complete)

            task_id = task_complete["index"]

            task_env = TaskEnv(
                env, task_complete, task_id, save_dir, use_partial_graph=False
            )

            random_init_room = random.choice(init_room_pool)
            logger.info("initial room: %s", random_init_room)
            task_env.reset_env_room(room=random_init_room, port_num=PORT_NUM)
            success = nav_simple_abstract(
                task_env, task_complete, save_img=SAVE_IMG, model=MODEL
            )

            process_counter += 1

            if task_id >= source_counter:
                progress[env_id] = task_id + 1
                save_json(progress, progress_file_path)

            if success:
                success_counter += 1
                success_count_dict[env_id] = success_counter
                save_json(success_count_dict, success_count_path)

            # if DATA_TO_GENERATE != -1 and process_counter >= DATA_TO_GENERATE:
            if DATA_TO_GENERATE != -1:
                if (
                    STOP_METRIC == "success" and success_counter >= DATA_TO_GENERATE
                ) or (STOP_METRIC == "process" and process_counter >= DATA_TO_GENERATE):
                    logger.info(
                        "DATA_TO_GENERATE (%s) reached for file: %s",
                        DATA_TO_GENERATE,
                        file_env,
                    )

                    break

        if DATA_TO_GENERATE != -1:
            if STOP_METRIC == "success" and success_counter < DATA_TO_GENERATE:
                logger.warning(
                    "%s success from file: %s, failed to reach %s",
                    success_counter,
                    file_env,
                    DATA_TO_GENERATE,
                )
            elif STOP_METRIC == "process" and process_counter < DATA_TO_GENERATE:
                logger.warning(
                    "%s processed from file: %s, failed to reach %s",
                    process_counter,
                    file_env,
                    DATA_TO_GENERATE,
                )

        env.close()


if __name__ == "__main__":
    try:
        for_loop_data_generation()
    except KeyboardInterrupt:
        sys.exit(0)
