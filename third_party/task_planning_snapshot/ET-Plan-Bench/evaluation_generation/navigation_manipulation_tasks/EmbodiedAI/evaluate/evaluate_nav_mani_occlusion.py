import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)


import argparse
from tqdm import tqdm

import numpy as np

from utils.eval import lcs, get_action_object_list
from utils.io import load_json
from constant.params import STEP_MAX
from constant.dirs import GENERATED_DATA_GROUND_TRUTH_DIR


parser = argparse.ArgumentParser()

parser.add_argument("-c", "--constraint", action='store_true')
parser.add_argument("-o", "--occlusion", type=str, default="item_size", choices=["item_size", "init_distance"])
parser.add_argument("-q", "--quantile", type=int, required=False, choices=range(0,101))
parser.add_argument("-m", "--model", type=str, default="gpt", choices=["gpt", "llava"])

args, leftovers = parser.parse_known_args()

if args.quantile is None:
    if args.occlusion == "item_size":
        args.quantile = 20
    else:
        args.quantile = 80

nav_mani_task_categories = ["nav_inside", "nav_on"]
nav_mani_cons_task_categories = ["nav_inside_cons", "nav_on_cons"]
if args.constraint:
    task_categories = nav_mani_cons_task_categories
else:
    task_categories = nav_mani_task_categories


testing_environment_ids = [37, 0, 32, 39, 19, 20, 48, 49, 17, 26]


stats = {}
COUNT = 0

print("calculating threshold...")

for env_id in tqdm(testing_environment_ids):

    env = f"env_{env_id}"

    for task_name in task_categories:
        task_dir = os.path.join(GENERATED_DATA_GROUND_TRUTH_DIR, task_name)
        env_path = os.path.join(task_dir, env)
        for data in os.listdir(env_path):
            if data == "RGB":
                continue
            data_path = os.path.join(env_path, data)
            if args.model == "gpt":
                data_path_plan = data_path.replace("ground_truth", "visible")
            else:
                data_path_plan = data_path.replace("ground_truth", "visible_llava")
            if not os.path.exists(data_path_plan):
                continue

            full_data = load_json(data_path)
            KEY = str(COUNT) + "_" + str(full_data[args.occlusion])
            stats[KEY] = data_path
            COUNT += 1

threshold = np.percentile([float(key.split("_")[-1]) for key in stats.keys()], args.quantile)

total_num = 0
success_num = 0

distance_list = []
lsc_value_list = []
lsc_value_ratio_list = []
steps_list = []

for key, GT_task_file in tqdm(stats.items()):
    metric = float(key.split("_")[-1])


    if args.occlusion == "item_size":
        if metric >= threshold:
            continue
    else:
        if metric <= threshold:
            continue
    if args.model == "gpt":
        Plan_task_file = GT_task_file.replace("ground_truth", "visible")
    else:
        Plan_task_file = GT_task_file.replace("ground_truth", "visible_llava")

    if not os.path.exists(GT_task_file):
        print("[WARNING] GT_task_file not exists", GT_task_file)
        continue

    if not os.path.exists(Plan_task_file):
        continue

    GT_task_data = load_json(GT_task_file)  # modify here
    Plan_task_data = load_json(Plan_task_file)

    traverse_distance = Plan_task_data[
        "traverse_distance"
    ]  # only calculate the traverse distance for success cases

    if traverse_distance == 0:  # could be vittualhome little brain issue
        continue

    GT_action_history = GT_task_data["action_history"]
    GT_action_history_parse = [
        get_action_object_list(action) for action in GT_action_history
    ]

    Plan_action_history = Plan_task_data["action_history"]
    Plan_action_history_parse = [
        get_action_object_list(action) for action in Plan_action_history
    ]

    # print("GT_action_history_parse:", GT_action_history_parse)
    # print("Plan_action_history_parse:", Plan_action_history_parse)

    lcs_string = lcs(GT_action_history_parse, Plan_action_history_parse)
    lcs_value = len(lcs_string) - 1
    lsc_value_list.append(lcs_value)

    # print("lcs_string:", lcs_string)
    # print("GT_action_history_parse:", GT_action_history_parse)

    lsc_value_ratio = lcs_value / len(GT_action_history_parse)
    # print("lsc_value_ratio:", lsc_value_ratio)

    if len(GT_action_history_parse) > len(Plan_action_history_parse):
        lsc_value_ratio = 1.0

    lsc_value_ratio_list.append(lsc_value_ratio)

    success_tag = str(Plan_task_data["success"])

    total_num += 1
    if success_tag == "True":
        distance_list.append(traverse_distance)
        success_num += 1
        steps_list.append(len(Plan_action_history_parse))
    else:  # seq length for failed cases is MAX_STEPS
        steps_list.append(STEP_MAX)

print("total_num:", total_num)
print("success_num:", success_num)
print("success rate:", success_num / total_num)

distance_sum = 0
for the_distance in distance_list:
    distance_sum += the_distance
average_distance = distance_sum / len(distance_list)
print("average_distance for success cases:", average_distance)

lcs_sum = 0
for the_lcs in lsc_value_list:
    lcs_sum += the_lcs
average_lcs = lcs_sum / len(lsc_value_list)
print("average_lcs for all cases:", average_lcs)

lcs_ratio_sum = 0
for the_lcs_ratio in lsc_value_ratio_list:
    lcs_ratio_sum += the_lcs_ratio
average_lcs_ratio = lcs_ratio_sum / len(lsc_value_ratio_list)
print("average_lcs_ratio for all cases:", average_lcs_ratio)

steps_sum = 0
for the_step_num in steps_list:
    steps_sum += the_step_num
average_steps_num = steps_sum / len(steps_list)
print("average_steps_num for all cases:", average_steps_num)
