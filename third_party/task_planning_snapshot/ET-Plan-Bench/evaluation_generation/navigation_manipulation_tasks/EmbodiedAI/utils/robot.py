import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import re

import networkx as nx

from envs.env import TaskEnv, VirtualHomeEnv
from utils.gpt_call import query_gpt_single, process_gpt_answer, llava_single
from utils.log import get_logger
from utils.helper import get_main_file_abs_path
from constant.prompt.data_gen import (
    PROMPT_IF_HIDDEN_ORDERED,
    PROMPT_ROOMS_ORDEDED_WITH_OBJECT_NAME,
    PROMPT_IF_HIDDEN_ORDERED_ABSTRACT,
    PROMPT_ROOMS_ORDEDED_ABSTRACT,
    PROMPT_IF_EXIST_ITEM_CORRECT,
)
from constant.params import BLOCKER_MAX, STEP_MAX


logger = get_logger(get_main_file_abs_path())


# def get_tmp_name_id_dict(id_name_dict:dict):
#     tmp_name_id_dict = {}
#     name_ids_dict = {}

#     for id, name in id_name_dict.items():
#         name_ids_dict = populate_dict_list(name_ids_dict, id, name)

#     for name, ids in name_ids_dict.items():
#         if len(ids) == 1:
#             tmp_name_id_dict[name] = ids[0]
#         elif len(ids) > 1:
#             for id in ids:
#                 tmp_name = name + f"#{id}"
#                 tmp_name_id_dict[tmp_name] = id

#     return tmp_name_id_dict


def get_cmd_from_action(action: str):
    return re.search(r"\[.*?\]", action).group()


def LLM_find_rooms(prompt: str, model="gpt"):
    if model == "gpt":
        response = query_gpt_single(prompt)
    else:
        response = llava_single(prompt)
    found_rooms = process_gpt_answer(response, to_list=True)

    logger.info("gpt4 recommand rooms in order: %s", found_rooms)

    return found_rooms


def LLM_find_room(prompt: str, room_names_to_check: list, model="gpt", max_retry=5):
    found_room = None

    while found_room not in room_names_to_check:
        if max_retry == 0:
            return None

        max_retry -= 1

        if model == "gpt":
            response = query_gpt_single(prompt)
        else:
            response = llava_single(prompt)
        found_room = process_gpt_answer(response)

    logger.info("gpt4 recommand room: %s", found_room)

    return found_room


def LLM_find_blockers(task_env: TaskEnv, object_name: str, abstract=False, model="gpt"):
    blockers_unique_names = TaskEnv.get_unique_value_class_name(
        task_env.items_remaining
    )
    logger.info("Blocking items for llm to choose: %s", blockers_unique_names)

    if abstract:
        prompt_if_hidden = PROMPT_IF_HIDDEN_ORDERED_ABSTRACT.format(
            task=object_name, blockers=blockers_unique_names
        )
    else:
        prompt_if_hidden = PROMPT_IF_HIDDEN_ORDERED.format(
            object_name=object_name, blockers=blockers_unique_names
        )
        # prompt_check_if_inside_object = task_env.prompt_meta_task + prompt_if_inside
    prompt_check_if_hidden_object = prompt_if_hidden

    if model == "gpt":
        llm_decided_blockers = query_gpt_single(prompt_check_if_hidden_object)
    else:
        llm_decided_blockers = llava_single(prompt_check_if_hidden_object)

    logger.info("Blocking items chosen by llm (raw response): %s", llm_decided_blockers)

    llm_decided_blockers_processed = process_gpt_answer(
        llm_decided_blockers, to_list=True
    )
    logger.info(
        "Blocking items chosen by llm (processed): %s", llm_decided_blockers_processed
    )

    if llm_decided_blockers_processed is None:
        return (
            [],
            [],
            prompt_check_if_hidden_object,
            llm_decided_blockers_processed,
        )

    blockers_ids = []
    blockers_name = []
    for blocker_name in llm_decided_blockers_processed[:BLOCKER_MAX]:
        blocker_ids = TaskEnv.get_listed_object_ids_from_name(
            blocker_name, task_env.items_remaining
        )

        logger.info(
            "blocker_ids of blockers %s: %s",
            blocker_name,
            blocker_ids,
        )

        if len(blocker_ids) > 0:
            blockers_ids.append(blocker_ids)
            blockers_name.append(blocker_name)

    logger.info(
        "LLM_decision_if_inside: %s; ids: %s",
        llm_decided_blockers_processed,
        blockers_ids,
    )

    return (
        blockers_ids,
        blockers_name,
        prompt_check_if_hidden_object,
        llm_decided_blockers_processed,
    )


def LLM_determines_abstract_items(prompt: str, model="gpt"):
    if model == "gpt":
        response = query_gpt_single(prompt)
    else:
        response = llava_single(prompt)

    abstract_correct = process_gpt_answer(response, to_list=True)

    logger.info("LLM determines acceptable items: %s", abstract_correct)

    return abstract_correct


def robot_self_action(
    task_env: TaskEnv, action_cmd: str, prompt="", response="", save_img=False
):
    action = f"<char0> [{action_cmd}]"

    obs, success, info = task_env.llm_step_save(
        action, prompt=prompt, response=response, save_img=save_img
    )

    return obs, success, info


def robot_single_action(
    task_env: TaskEnv,
    action_cmd: str,
    object_id=None,
    prompt="",
    response="",
    save_img=False,
):
    if action_cmd not in task_env.env.actions_available:
        logger.fatal("Action not allowed: %s", action_cmd)
        raise ValueError(f"Action not allowed: {action_cmd}")
    if object_id is None:
        # use object in last action
        last_action = task_env.action_history[-1]

        logger.info("Using last action: %s", last_action)

        if last_action.count(" ") != 3:
            logger.error("Last action is not a sinlge-object action")

            last_action_split = last_action.split(" ")
            last_action = " ".join(last_action_split[:4])

        action = re.sub(r"\[.*\]", f"[{action_cmd}]", last_action)
    else:
        object_name = task_env.env.get_object_name_by_id(object_id)

        action = f"<char0> [{action_cmd}] <{object_name}> ({object_id})"

    obs, success, info = task_env.llm_step_save(
        action, prompt=prompt, response=response, save_img=save_img
    )

    return obs, success, info


def robot_cos_action(
    task_env: TaskEnv,
    action_cmd: str,
    object1_id: int,
    object2_id: int,
    prompt="",
    response="",
    save_img=False,
):
    object1_name = task_env.env.get_object_name_by_id(object1_id)
    object2_name = task_env.env.get_object_name_by_id(object2_id)

    action = (
        f"<char0> [{action_cmd}] <{object1_name}> ({object1_id}) "
        + f"<{object2_name}> ({object2_id})"
    )

    obs, success, info = task_env.llm_step_save(
        action, prompt=prompt, response=response, save_img=save_img
    )

    return obs, success, info


def robot_walk_to_object_ids(
    task_env: TaskEnv, object_ids: list, prompt="", response="", save_img=False
):
    if len(object_ids) > 1:
        object_ids = task_env.get_object_ids_sorted_distance(object_ids)

    for object_id in object_ids:
        obs, success, info = robot_single_action(
            task_env,
            "walk",
            object_id,
            prompt=prompt,
            response=response,
            save_img=save_img,
        )

        task_env.update_items_explored(object_id)

        if success:
            return object_id, obs, success, info

    logger.warning(
        "Failed to walk to %s; ids: %s",
        task_env.env.get_object_name_by_id(object_ids[0]),
        object_ids,
    )

    return None, None, False, None


def robot_walk_to_object_name(
    task_env: TaskEnv,
    object_name: str,
    available_object_dict: dict,
    prompt="",
    response="",
    save_img=False,
):
    object_ids = TaskEnv.get_listed_object_ids_from_name(
        object_name, available_object_dict
    )
    if len(object_ids) > 0:
        object_id, _, success, _ = robot_walk_to_object_ids(
            task_env, object_ids, prompt=prompt, response=response, save_img=save_img
        )

        if success:
            return object_id
        else:
            return success
    else:
        return None


def robot_walk_to_object_abstract(
    task_env: TaskEnv,
    task_abstract: str,
    model="gpt",
    save_img=False,
):
    prompt_if_exist_item_correct = PROMPT_IF_EXIST_ITEM_CORRECT.format(
        task=task_abstract,
        items=task_env.get_unique_value_class_name(task_env.items_seen),
    )

    correct_items = LLM_determines_abstract_items(
        prompt_if_exist_item_correct, model=model
    )

    if correct_items is not None:
        correct_item = correct_items[0]
        object_id = robot_walk_to_object_name(
            task_env,
            correct_item,
            task_env.items_seen,
            prompt_if_exist_item_correct,
            correct_items,
            save_img=save_img,
        )
        return object_id

    return None


def robot_walk_to_constrained_item_name(
    task_env: TaskEnv,
    item_name: str,
    constraint_item_name: str,
    constraint: str,
    save_img=False,
):
    success, item_ids, constraint_item_ids = visible_constrained_items(
        task_env, item_name, constraint_item_name, constraint
    )

    if success:
        logger.info(
            "found constrained item: %s, constraining item: %s, constraint: %s",
            item_name,
            constraint_item_name,
            constraint,
        )

        _, success, _ = robot_single_action(
            task_env,
            "walk",
            item_ids[0],
            save_img=save_img,
        )

        task_env.update_items_explored(item_ids[0])

        if success:
            return True, item_ids, constraint_item_ids

        return False, None, None

    logger.info("not both items visible")

    return None, item_ids, constraint_item_ids


def robot_walk_search_constrained_items_name(
    task_env: TaskEnv,
    item_name: str,
    constraint_item_name: str,
    constraint: str,
    save_img=False,
):
    success, item_ids, constraint_item_ids = robot_walk_to_constrained_item_name(
        task_env, item_name, constraint_item_name, constraint, save_img=save_img
    )

    if success:
        return item_ids[0], constraint_item_ids[0]

    if success is False:
        return False, False

    # assert len(set(item_ids).intersection(set(constraint_item_ids))) == 0

    visited_items_id = []

    for item_id in set(item_ids + constraint_item_ids):
        logger.info("one of the items visible, walking to item_id %s", item_id)
        _, success, _ = robot_single_action(
            task_env,
            "walk",
            item_id,
            save_img=save_img,
        )

        visited_items_id.append(item_id)

        if not success:
            return False, False

        if constraint == "INSIDE" and item_id in constraint_item_ids:
            _, _, _ = robot_single_action(
                task_env,
                "open",
                item_id,
                save_img=save_img,
            )

        success, item_ids, constraint_item_ids = robot_walk_to_constrained_item_name(
            task_env,
            item_name,
            constraint_item_name,
            constraint,
            save_img=save_img,
        )

        if success:
            for item_id in visited_items_id:
                task_env.update_items_explored(item_id)

            return item_ids[0], constraint_item_ids[0]

    for item_id in visited_items_id:
        task_env.update_items_explored(item_id)

    return None, None


def robot_put_in(task_env: TaskEnv, object_id: int, container_id: int, save_img=False):
    obs, success, info = robot_cos_action(
        task_env, "putin", object_id, container_id, save_img=save_img
    )

    if not success:
        obs, success, info = robot_cos_action(
            task_env, "put", object_id, container_id, save_img=save_img
        )

    return obs, success, info


def robot_put_on(task_env: TaskEnv, object_id: int, container_id: int, save_img=False):
    obs, success, info = robot_cos_action(
        task_env, "put", object_id, container_id, save_img=save_img
    )

    if not success:
        obs, success, info = robot_cos_action(
            task_env, "putin", object_id, container_id, save_img=save_img
        )

    return obs, success, info


def visible_constrained_items(
    task_env: TaskEnv, item_name: str, constraint_item_name: str, constraint: str
):
    item_ids = TaskEnv.get_listed_object_ids_from_name(
        item_name, task_env.items_remaining
    )

    constraint_item_ids = TaskEnv.get_listed_object_ids_from_name(
        constraint_item_name, task_env.items_remaining
    )

    if len(item_ids) > 0 and len(constraint_item_ids) > 0:
        for item_idx, item_id in enumerate(item_ids):
            for constraint_item_idx, constraint_item_id in enumerate(
                constraint_item_ids
            ):
                if task_env.check_relation(
                    object_id_1=item_id,
                    object_id_2=constraint_item_id,
                    relation=constraint,
                ):
                    item_ids[0], item_ids[item_idx] = (
                        item_ids[item_idx],
                        item_ids[0],
                    )
                    (
                        constraint_item_ids[0],
                        constraint_item_ids[constraint_item_idx],
                    ) = (
                        constraint_item_ids[constraint_item_idx],
                        constraint_item_ids[0],
                    )
                    return True, item_ids, constraint_item_ids

    return None, item_ids, constraint_item_ids


def robot_exploration_single_room_id(
    task_env: TaskEnv,
    object_name: str,
    room_id: int = None,
    prompt="",
    response="",
    save_img=False,
    model="gpt",
):
    if room_id:
        _, success, _ = robot_single_action(
            task_env, "walk", room_id, prompt, response, save_img=save_img
        )

        if not success:
            return False

    object_id = robot_walk_to_object_name(
        task_env, object_name, task_env.items_seen, save_img=save_img
    )

    if object_id:
        logger.info("Found object!")
        return object_id
    if object_id is False:
        return False

    # not visible, find containers (large objects)
    blockers_ids, blockers_name, prompt_check_if_hidden_object, llm_decision = (
        LLM_find_blockers(task_env, object_name, model=model)
    )

    for blocker_name, blocker_ids in zip(blockers_name, blockers_ids):
        if task_env.step_num > STEP_MAX:
            logger.info("maximum number of steps reached! step: %s", task_env.step_num)
            break
        while len(blocker_ids) > 0:
            if task_env.step_num > STEP_MAX:
                logger.info(
                    "maximum number of steps reached! step: %s", task_env.step_num
                )
                break
            if len(blocker_ids) > 1:
                blocker_ids = task_env.get_object_ids_sorted_distance(blocker_ids)
                logger.info(
                    "Re-ordered ids for container %s: %s",
                    blocker_name,
                    blocker_ids,
                )

            blocker_id = blocker_ids.pop(0)

            _, success, _ = robot_single_action(
                task_env,
                "walk",
                blocker_id,
                prompt=prompt_check_if_hidden_object,
                response=llm_decision,
                save_img=save_img,
            )

            task_env.update_items_explored(blocker_id)

            if success:
                object_id = robot_walk_to_object_name(
                    task_env,
                    object_name,
                    task_env.items_seen,
                    save_img=save_img,
                )
                if object_id:
                    logger.info("Found object!")
                    return object_id
                if object_id is False:
                    return False

                _, _, _ = robot_single_action(
                    task_env, "open", blocker_id, save_img=save_img
                )

                object_id = robot_walk_to_object_name(
                    task_env,
                    object_name,
                    task_env.items_seen,
                    save_img=save_img,
                )
                if object_id:
                    logger.info("Found object!")
                    return object_id
                if object_id is False:
                    return False

                _, _, _ = robot_single_action(
                    task_env,
                    "close",
                    blocker_id,
                    save_img=save_img,
                )
                continue  # not necessary

    return None


def robot_exploration_constrained_single_room_id(
    task_env: TaskEnv,
    object_name: str,
    constraint_object_name: str,
    constraint: str,
    room_id: int = None,
    prompt="",
    response="",
    save_img=False,
    model="gpt",
):
    if room_id:
        _, success, _ = robot_single_action(
            task_env, "walk", room_id, prompt, response, save_img=save_img
        )

        if not success:
            return False, False

    item_id, constraint_item_id = robot_walk_search_constrained_items_name(
        task_env, object_name, constraint_object_name, constraint, save_img=save_img
    )

    if item_id and constraint_item_id:
        return item_id, constraint_item_id
    if item_id is False:
        return False, False

    # not visible, find containers (large objects)
    object_name_prompt = object_name + " and " + constraint_object_name
    blockers_ids, blockers_name, prompt_check_if_hidden_object, llm_decision = (
        LLM_find_blockers(task_env, object_name_prompt, model=model)
    )

    for blocker_name, blocker_ids in zip(blockers_name, blockers_ids):
        if task_env.step_num > STEP_MAX:
            logger.info("maximum number of steps reached! step: %s", task_env.step_num)
            break
        while len(blocker_ids) > 0:
            if task_env.step_num > STEP_MAX:
                logger.info(
                    "maximum number of steps reached! step: %s", task_env.step_num
                )
                break
            if len(blocker_ids) > 1:
                blocker_ids = task_env.get_object_ids_sorted_distance(blocker_ids)
                logger.info(
                    "Re-ordered ids for container %s: %s",
                    blocker_name,
                    blocker_ids,
                )

            blocker_id = blocker_ids.pop(0)

            _, success, _ = robot_single_action(
                task_env,
                "walk",
                blocker_id,
                prompt=prompt_check_if_hidden_object,
                response=llm_decision,
                save_img=save_img,
            )

            task_env.update_items_explored(blocker_id)

            item_id, constraint_item_id = robot_walk_search_constrained_items_name(
                task_env,
                object_name,
                constraint_object_name,
                constraint,
                save_img=save_img,
            )

            if item_id and constraint_item_id:
                return item_id, constraint_item_id
            if item_id is False:
                return False, False

    return None, None


def robot_exploration_single_room_name(
    task_env: TaskEnv,
    object_name: str,
    room_name: str,
    prompt="",
    response="",
    save_img=False,
    model="gpt",
):
    room_ids = TaskEnv.get_listed_object_ids_from_name(
        room_name, task_env.rooms_remaining
    )

    if len(room_ids) > 1:
        room_ids = task_env.get_object_ids_sorted_distance(room_ids)

    for room_id in room_ids:
        object_id = robot_exploration_single_room_id(
            task_env,
            object_name,
            room_id,
            prompt,
            response,
            save_img=save_img,
            model=model,
        )
        if object_id:
            return object_id, room_id
        if object_id is False:
            return False, False

    return None, None


def robot_exploration_single_room_id_abstract(
    task_env: TaskEnv,
    task_abstract: str,
    room_id: int,
    prompt="",
    response="",
    save_img=False,
    model="gpt",
):
    _, success, _ = robot_single_action(
        task_env, "walk", room_id, prompt, response, save_img=save_img
    )

    if not success:
        return False

    object_id = robot_walk_to_object_abstract(
        task_env, task_abstract, model=model, save_img=save_img
    )

    if object_id:
        logger.info("Found object!")
        return object_id
    if object_id is False:
        return False

    # not visible, find containers (large objects)
    blockers_ids, blockers_name, prompt_check_if_hidden_object, llm_decision = (
        LLM_find_blockers(task_env, task_abstract, abstract=True, model=model)
    )

    for blocker_name, blocker_ids in zip(blockers_name, blockers_ids):
        if task_env.step_num > STEP_MAX:
            logger.info("maximum number of steps reached! step: %s", task_env.step_num)
            break
        while len(blocker_ids) > 0:
            if task_env.step_num > STEP_MAX:
                logger.info(
                    "maximum number of steps reached! step: %s", task_env.step_num
                )
                break
            if len(blocker_ids) > 1:
                blocker_ids = task_env.get_object_ids_sorted_distance(blocker_ids)
                logger.info(
                    "Re-ordered ids for container %s: %s",
                    blocker_name,
                    blocker_ids,
                )

            blocker_id = blocker_ids.pop(0)

            _, success, _ = robot_single_action(
                task_env,
                "walk",
                blocker_id,
                prompt=prompt_check_if_hidden_object,
                response=llm_decision,
                save_img=save_img,
            )

            task_env.update_items_explored(blocker_id)

            if success:
                object_id = robot_walk_to_object_abstract(
                    task_env, task_abstract, model=model, save_img=save_img
                )

                if object_id:
                    logger.info("Found object!")
                    return object_id
                if object_id is False:
                    return False

                _, _, _ = robot_single_action(
                    task_env, "open", blocker_id, save_img=save_img
                )

                object_id = robot_walk_to_object_abstract(
                    task_env, task_abstract, model=model, save_img=save_img
                )

                if object_id:
                    logger.info("Found object!")
                    return object_id
                if object_id is False:
                    return False

                _, _, _ = robot_single_action(
                    task_env,
                    "close",
                    blocker_id,
                    save_img=save_img,
                )
                continue  # not necessary

    return None


def robot_exploration(task_env: TaskEnv, object_name: str, save_img=False, model="gpt"):
    logger.info("Finding %s", object_name)

    object_id = robot_walk_to_object_name(
        task_env, object_name, task_env.items_seen, save_img=save_img
    )
    if object_id:
        logger.info("Found object!")
        return object_id
    if object_id is False:
        return False

    prompt_rooms_with_object_name = PROMPT_ROOMS_ORDEDED_WITH_OBJECT_NAME.format(
        object_name=object_name, room_names=task_env.env.room_names
    )
    prompt_find_rooms = prompt_rooms_with_object_name

    logger.info("Finding %s by LLM", object_name)
    rooms_found = LLM_find_rooms(prompt_find_rooms, model=model)
    logger.info("Rooms to explore: %s", rooms_found)
    if rooms_found is None:
        return None
    if isinstance(rooms_found, str):
        rooms_found = [rooms_found]

    for room_name in rooms_found:
        object_id, _ = robot_exploration_single_room_name(
            task_env,
            object_name,
            room_name,
            prompt_find_rooms,
            rooms_found,
            save_img=save_img,
            model=model,
        )
        if object_id:
            return object_id
        if object_id is False:
            return False

    return None


def robot_exploration_id_ground_truth(
    vh_env: VirtualHomeEnv, task_env: TaskEnv, object_id: int, save_img=False
):
    room_id = vh_env.object_id_in_room_id_dict[object_id]
    path_object_room = list(
        nx.all_simple_paths(vh_env.scene_graph_inside, object_id, room_id)
    )[0]

    current_room_id = vh_env.get_agent_room_id()

    if current_room_id != room_id:
        _, success, _ = robot_single_action(
            task_env, "walk", room_id, save_img=save_img
        )

        if not success:
            return False

    for intermediate_object_id in reversed(list(path_object_room)[:-1]):
        if len(path_object_room) > 2 and len(task_env.action_history) > 0:
            _, _, _ = robot_single_action(task_env, "open", save_img=save_img)
        _, success, _ = robot_single_action(
            task_env, "walk", intermediate_object_id, save_img=save_img
        )

        if not success:
            return False

    return object_id


def robot_exploration_ground_truth(
    vh_env: VirtualHomeEnv, task_env: TaskEnv, object_name: str, save_img=False
):
    logger.info("Finding %s with ground truth", object_name)
    object_ids = TaskEnv.get_listed_object_ids_from_name(
        object_name, vh_env.full_nodes_list
    )

    if len(object_ids) == 0:
        return False

    object_ids = task_env.get_object_ids_sorted_distance(object_ids)
    object_id = object_ids[0]

    return robot_exploration_id_ground_truth(vh_env, task_env, object_id, save_img)


def robot_exploration_constraint(
    task_env: TaskEnv,
    object_name: str,
    constraint_object_name: str,
    constraint: str,
    save_img=False,
    model="gpt",
):
    logger.info(
        "Finding %s constrained to %s with constraint %s",
        object_name,
        constraint_object_name,
        constraint,
    )

    if constraint_object_name in task_env.env.room_names:
        logger.info("constraint object is a room, search in %s", constraint_object_name)
        return robot_exploration_single_room_name(
            task_env,
            object_name,
            constraint_object_name,
            save_img=save_img,
            model=model,
        )

    item_id, constraint_item_id = robot_walk_search_constrained_items_name(
        task_env, object_name, constraint_object_name, constraint, save_img=save_img
    )

    if item_id and constraint_item_id:
        return item_id, constraint_item_id
    if item_id is False:
        return False, False

    # current_room_id = task_env.env.get_agent_room_id()
    # task_env.update_rooms_explored(current_room_id)

    object_name_prompt = object_name + " and " + constraint_object_name
    prompt_rooms_with_object_name = PROMPT_ROOMS_ORDEDED_WITH_OBJECT_NAME.format(
        object_name=object_name_prompt,
        room_names=task_env.get_unique_value_class_name(task_env.rooms_remaining),
    )
    prompt_find_rooms = prompt_rooms_with_object_name

    logger.info("Finding %s", object_name_prompt)
    rooms_found = LLM_find_rooms(prompt_find_rooms, model=model)
    logger.info("Rooms to explore: %s", rooms_found)
    if rooms_found is None:
        return None, None
    if isinstance(rooms_found, str):
        rooms_found = [rooms_found]

    for room_name in rooms_found:
        room_ids = TaskEnv.get_listed_object_ids_from_name(
            room_name, task_env.rooms_remaining
        )

        if len(room_ids) > 1:
            room_ids = task_env.get_object_ids_sorted_distance(room_ids)

        for room_id in room_ids:
            item_id, constraint_item_id = robot_exploration_constrained_single_room_id(
                task_env,
                object_name,
                constraint_object_name,
                constraint,
                room_id,
                prompt_find_rooms,
                rooms_found,
                save_img,
                model,
            )

            if item_id and constraint_item_id:
                return item_id, constraint_item_id

            if item_id is False:
                return False, False

    return None, None


def robot_exploration_abstract(
    task_env: TaskEnv,
    task_abstract: str,
    model="gpt",
    save_img=False,
):
    object_id = robot_walk_to_object_abstract(
        task_env, task_abstract, model=model, save_img=save_img
    )

    if object_id:
        logger.info("Found object!")
        return object_id
    if object_id is False:
        return False

    # current_room_id = task_env.env.get_agent_room_id()
    # task_env.update_rooms_explored(current_room_id)

    prompt_rooms_ordered_abstract = PROMPT_ROOMS_ORDEDED_ABSTRACT.format(
        task=task_abstract,
        room_names=task_env.get_unique_value_class_name(task_env.rooms_remaining),
    )

    logger.info("Abstract task %s", task_abstract)
    rooms_found = LLM_find_rooms(prompt_rooms_ordered_abstract, model=model)
    logger.info("Rooms to explore: %s", rooms_found)
    if rooms_found is None:
        return None
    if isinstance(rooms_found, str):
        rooms_found = [rooms_found]

    for room_name in rooms_found:
        room_ids = TaskEnv.get_listed_object_ids_from_name(
            room_name, task_env.rooms_remaining
        )

        if len(room_ids) > 1:
            room_ids = task_env.get_object_ids_sorted_distance(room_ids)

        for room_id in room_ids:
            item_id = robot_exploration_single_room_id_abstract(
                task_env,
                task_abstract,
                room_id,
                prompt_rooms_ordered_abstract,
                rooms_found,
                save_img,
                model,
            )

            if item_id:
                return item_id

            if item_id is False:
                return False

    return None
