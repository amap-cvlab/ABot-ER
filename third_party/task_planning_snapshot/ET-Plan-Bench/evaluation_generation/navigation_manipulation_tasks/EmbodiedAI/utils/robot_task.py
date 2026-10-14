import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from envs.env import TaskEnv, VirtualHomeEnv
from utils.robot import (
    robot_exploration,
    robot_put_in,
    robot_put_on,
    robot_single_action,
    robot_exploration_ground_truth,
    robot_exploration_id_ground_truth,
    robot_exploration_constraint,
    robot_exploration_abstract,
    get_cmd_from_action,
)


def put_in_close(task_env, object_id, container_id, save_img=False):
    _, success_open, _ = robot_single_action(task_env, "open", container_id)

    _, success, _ = robot_put_in(task_env, object_id, container_id, save_img=save_img)
    if success_open:
        _, _, _ = robot_single_action(task_env, "close", container_id)

    return success


def nav_manipulate_single(
    task_env: TaskEnv, object_name, container, task_type="on", save_img=False, model="gpt"
):
    if task_type not in ["in", "on"]:
        raise ValueError(f"Task type should be 'in' or 'on'! task type: {task_type}")

    object_id = robot_exploration(task_env, object_name, save_img=save_img, model=model)

    if not object_id:
        task_env.save()
        return False

    if (
        len(task_env.action_history) > 2
        and get_cmd_from_action(task_env.action_history[-2]) == "[open]"
    ):

        close_action = task_env.action_history[-2].replace("[open]", "[close]", 1)
        _, success, _ = robot_single_action(task_env, "grab", object_id)
        _, _, _ = task_env.llm_step_save(
            close_action, prompt="", response="", save_img=save_img
        )
    else:
        _, success, _ = robot_single_action(task_env, "grab", save_img=save_img)

    if not success:
        task_env.save()
        return False

    container_id = robot_exploration(
        task_env, container, save_img=save_img, model=model
    )

    if not container_id:
        task_env.save()
        return False

    if get_cmd_from_action(task_env.action_history[-2]) == "[open]":

        close_action = task_env.action_history[-2].replace("[open]", "[close]", 1)
        if task_type == "in":
            success = put_in_close(task_env, object_id, container_id, save_img)
        else:
            _, success, _ = robot_put_on(task_env, object_id, container_id)
        _, _, _ = task_env.llm_step_save(
            close_action, prompt="", response="", save_img=save_img
        )
    else:
        if task_type == "in":
            success = put_in_close(task_env, object_id, container_id, save_img)
        else:
            _, success, _ = robot_put_on(
                task_env, object_id, container_id, save_img=save_img
            )

    if success:
        task_env.done = success
        task_env.save()

    return success


def nav_manipulate_single_constrained(
    task_env: TaskEnv,
    object_name: str,
    container_name: str,
    object_constraint_name: str,
    container_constraint_name: str,
    object_rel: str,
    container_rel: str,
    task_type="on",
    save_img=False,
    model="gpt",
):
    if task_type not in ["in", "on"]:
        raise ValueError(f"Task type should be 'in' or 'on'! task type: {task_type}")

    object_id, _ = robot_exploration_constraint(
        task_env,
        object_name,
        object_constraint_name,
        object_rel,
        save_img=save_img,
        model=model,
    )

    if not object_id:
        task_env.save()
        return False

    if (
        len(task_env.action_history) > 2
        and get_cmd_from_action(task_env.action_history[-2]) == "[open]"
    ):

        close_action = task_env.action_history[-2].replace("[open]", "[close]", 1)
        _, success, _ = robot_single_action(task_env, "grab", object_id)
        _, _, _ = task_env.llm_step_save(
            close_action, prompt="", response="", save_img=save_img
        )
    else:
        _, success, _ = robot_single_action(task_env, "grab", save_img=save_img)

    if not success:
        task_env.save()
        return False

    container_id, _ = robot_exploration_constraint(
        task_env,
        container_name,
        container_constraint_name,
        container_rel,
        save_img=save_img,
        model=model,
    )

    if not container_id:
        task_env.save()
        return False

    if get_cmd_from_action(task_env.action_history[-2]) == "[open]":

        close_action = task_env.action_history[-2].replace("[open]", "[close]", 1)
        if task_type == "in":
            success = put_in_close(task_env, object_id, container_id, save_img)
        else:
            _, success, _ = robot_put_on(
                task_env, object_id, container_id, save_img=save_img
            )
        _, _, _ = task_env.llm_step_save(
            close_action, prompt="", response="", save_img=save_img
        )
    else:
        if task_type == "in":
            success = put_in_close(task_env, object_id, container_id, save_img)
        else:
            _, success, _ = robot_put_on(
                task_env, object_id, container_id, save_img=save_img
            )

    if success:
        task_env.done = success
        task_env.save()

    return success


def nav_manipulate_single_ground_truth(
    vh_env: VirtualHomeEnv,
    task_env: TaskEnv,
    obj,
    container,
    task_type="on",
    save_img=False,
):
    if task_type not in ["in", "on"]:
        raise ValueError(f"Task type should be 'in' or 'on'! task type: {task_type}")

    if isinstance(obj, str):
        exploration_success = robot_exploration_ground_truth(
            vh_env, task_env, obj, save_img=save_img
        )
    elif isinstance(obj, int):
        exploration_success = robot_exploration_id_ground_truth(
            vh_env, task_env, obj, save_img=save_img
        )
    else:
        raise ValueError

    if not exploration_success:
        return False

    object_id = exploration_success

    if (
        len(task_env.action_history) > 2
        and get_cmd_from_action(task_env.action_history[-2]) == "[open]"
    ):

        close_action = task_env.action_history[-2].replace("[open]", "[close]", 1)
        _, success, _ = robot_single_action(task_env, "grab", object_id)
        _, _, _ = task_env.llm_step_save(
            close_action, prompt="", response="", save_img=save_img
        )
    else:
        _, success, _ = robot_single_action(task_env, "grab", save_img=save_img)

    if not success:
        return False

    if isinstance(container, str):
        exploration_success = robot_exploration_ground_truth(
            vh_env, task_env, container, save_img=save_img
        )
    elif isinstance(container, int):
        exploration_success = robot_exploration_id_ground_truth(
            vh_env, task_env, container, save_img=save_img
        )
    else:
        raise ValueError

    if not exploration_success:
        return False

    container_id = exploration_success

    if "open" in task_env.action_history[-2]:

        close_action = task_env.action_history[-2].replace("[open]", "[close]", 1)

        if task_type == "in":
            success = put_in_close(task_env, object_id, container_id, save_img)
        else:
            _, success, _ = robot_put_on(
                task_env, object_id, container_id, save_img=save_img
            )

        _, _, _ = task_env.llm_step_save(
            close_action, prompt="", response="", save_img=save_img
        )
    else:

        if task_type == "in":
            success = put_in_close(task_env, object_id, container_id, save_img)
        else:
            _, success, _ = robot_put_on(
                task_env, object_id, container_id, save_img=save_img
            )

    if success:
        task_env.done = success
        task_env.save()

    return success


def nav_simple_abstract(
    task_env: TaskEnv, task_abstract_complete, save_img=False, model="gpt"
):
    task_abstract = task_abstract_complete["task"]
    object_id = robot_exploration_abstract(task_env, task_abstract, save_img=save_img, model=model)

    if not object_id:
        task_env.save_abstract(task_abstract_complete, object_id)
        return False

    task_env.save_abstract(task_abstract_complete, object_id)
    return True
