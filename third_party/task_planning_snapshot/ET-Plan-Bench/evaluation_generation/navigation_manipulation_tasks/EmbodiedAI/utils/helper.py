import os
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy


def retry_fn(fn, max_failures=10, sleep_time=5):
    failures = 0
    while failures < max_failures:
        try:
            return fn()
        except KeyboardInterrupt:
            print("Interrupted")
            sys.exit(0)
        except Exception as e:
            failures += 1
            print("Failed with exception:")
            print(e)
            print(f"Failed {failures} times, waiting {sleep_time}s to retry")
            time.sleep(sleep_time)
            if failures >= max_failures:
                raise RuntimeError("Max failures exceeded!") from e
            time.sleep(2)


def show_image_list(images):
    num = len(images)
    # fig = plt.figure(figsize=(18, 9))
    for i, image in enumerate(images):
        ax = plt.subplot(1, num, i + 1)
        ax.imshow(image[:, :, ::-1])
    plt.show()


def flatten(list_of_list):
    return [ele for lst in list_of_list for ele in lst]


def get_file_id(file_name):
    return int(file_name.split(".")[0].split("_")[-1])


def get_time_now(time_format="%Y%m%d_%H:%M:%S"):
    return datetime.now().strftime(time_format)


def get_task_ids(save_dir):
    task_ids = []
    for file in os.listdir(save_dir):
        if len(file.split(".")) > 1 and len(file.split(".")[0].split("_")) > 1:
            task_ids.append(int(file.split(".")[0].split("_")[1]))

    return task_ids


def count_success(save_dir):
    success_counter = 0

    for file in os.listdir(save_dir):
        if file.endswith((".json", ".jsonl")):
            file_path = os.path.join(save_dir, file)
            with open(file_path) as file:
                for line in file:
                    if '"success": true' in line:
                        success_counter += 1
                    break

    return success_counter


def eval_success_env(env_dir, out_of=50):
    success_counter = 0
    data_counter = 0
    for data in os.listdir(env_dir):
        if data_counter == out_of:
            break
        if data == "RGB":
            continue
        data_path = os.path.join(env_dir, data)
        with open(data_path) as file:
            for line in file:
                if '"success": true' in line:
                    success_counter += 1
                break
        data_counter += 1

    if data_counter < out_of:
        print("[WARNING] data in %s is less than %s", env_dir, out_of)

    return success_counter / out_of


def get_unique_value(dictionary):
    return list(set(dictionary.values()))


def get_unique_key(dictionary):
    return list(set(dictionary.keys()))


def get_reverse_dict(dictionary):
    reversed_dict = {}

    for key, value in dictionary.items():
        reversed_dict[value] = key

    return reversed_dict


def diff_dict_update(diff_dict, source_dict, update_dict):
    update_dict = {
        key: value for key, value in update_dict.items() if key not in source_dict
    }
    diff_dict.update(update_dict)

    return update_dict


def distance(point_source, points_destination):
    point_source_expand = np.expand_dims(point_source, axis=0)

    distances = np.squeeze(scipy.spatial.distance.cdist(point_source_expand,points_destination), 0)

    return distances


def get_main_file_abs_path():
    file = sys.argv[0]
    file_abs_path = os.path.abspath(file)

    return file_abs_path
