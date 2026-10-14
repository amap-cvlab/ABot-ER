import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import random

from utils.robot_task import nav_manipulate_single_constrained
from utils.unity import auto_kill_unity
from utils.io import json_iter
from utils.log import get_logger
from envs.env import VirtualHomeEnv, TaskEnv
from constant.dirs import GENERATED_DATA_DEBUG_DIR, GENERATED_TASK_FILTERED_DIR


logger = get_logger(__file__)

PORT_NUM = 3133
TASK_TYPE = "nav_on_cons"
ENV_ID = 0
init_room_pool = ["bathroom", "bedroom", "kitchen", "livingroom"]
task_id_to_run = [0]

@auto_kill_unity(kill_before_return=True)
def for_loop_data_generation():

    generated_tasks_file_name = f"env_{ENV_ID}.json"

    generated_tasks_file_path = os.path.join(
        GENERATED_TASK_FILTERED_DIR, TASK_TYPE, generated_tasks_file_name
    )
    save_dir = os.path.join(GENERATED_DATA_DEBUG_DIR, TASK_TYPE)
    os.makedirs(save_dir, exist_ok=True)

    env = VirtualHomeEnv(port=str(PORT_NUM))
    logger.info("initial port after VirtualHomeEnv: %s", PORT_NUM)

    # task_id_continue = get_task_ids(save_dir)

    for task_complete in json_iter(generated_tasks_file_path):
        task_id = task_complete["index"]
        # if task_id in task_id_continue:
        if task_id not in task_id_to_run:
            continue

        logger.info("task_id: %s", task_id)
        logger.info("task_complete: %s", task_complete)
        # task = task_complete['task']
        target_object = task_complete["object1"]
        target_constraint_object = task_complete["object3"]
        target_object_rel = task_complete["rel1"]
        target_container_object = task_complete["object2"]
        target_container_constraint_object = task_complete["object4"]
        target_container_rel = task_complete["rel2"]

        task_env = TaskEnv(
            env, task_complete, task_id, save_dir, use_partial_graph=False
        )

        random_init_room = random.choice(init_room_pool)
        logger.info("initial room: %s", random_init_room)
        task_env.reset_env_room(room=random_init_room, port_num=PORT_NUM)

        _ = nav_manipulate_single_constrained(
            task_env,
            target_object,
            target_container_object,
            target_constraint_object,
            target_container_constraint_object,
            target_object_rel,
            target_container_rel,
            "in",
            save_img=True,
        )

        # break
    env.close()


if __name__ == "__main__":
    try:
        for_loop_data_generation()
    except KeyboardInterrupt:
        sys.exit(0)
