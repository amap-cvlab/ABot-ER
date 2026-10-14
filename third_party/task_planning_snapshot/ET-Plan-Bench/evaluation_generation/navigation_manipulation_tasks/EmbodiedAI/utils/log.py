import logging
import os

from utils.helper import get_time_now
from constant.dirs import LOG_DIR, REPO_NAME


def get_logger(file_name):
    date = get_time_now('%Y%m%d')
    if REPO_NAME in file_name:
        relative_path = file_name.split(REPO_NAME, 1)[1][1:]
    else:
        relative_path = os.path.basename(file_name)
    log_folder = os.path.join(LOG_DIR, relative_path)
    folders_list = log_folder.split("/")
    folders_list.insert(-2, date)

    log_folder_path = "/".join(folders_list[:-1])
    log_path_tmp = "/".join(folders_list)

    os.makedirs(log_folder_path, exist_ok=True)

    pid = os.getpid()

    log_file_path = log_path_tmp.split(".")[0] + f"_{pid}.log"

    logging.basicConfig(
        format="%(asctime)s,%(levelname)s,[%(filename)s:%(funcName)s:%(lineno)d],%(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(file_name)

    return logger
