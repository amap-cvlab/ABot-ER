import json
import os
import traceback

import requests

from .logging import get_logger
from ..generate.utils import is_external_cluster

logger = get_logger(__name__)


MODULE = 'InteractiveCkpt'
SUBMODULE_ENABLE = 'Enable'
SUBMODULE_SHOULD_SAVE = 'ShouldSave'
SUBMODULE_FINISH_SAVE = 'FinishSave'


def interactive_ckpt_enable():
    response = to_scheduler(MODULE, SUBMODULE_ENABLE)
    if response['status'] != 'success' or response['response']['code'] != 0:
        logger.warning(f"{MODULE} {SUBMODULE_ENABLE} failed, response:{response}")
        return False
    logger.debug(f"{MODULE} {SUBMODULE_ENABLE} , response:{response}")
    return True


def interactive_ckpt_should_save(rank, global_step):
    message = {
        "rank": rank,
        "global_step": global_step
    }
    response = to_scheduler(MODULE, SUBMODULE_SHOULD_SAVE, message=message)
    if response['status'] != 'success' or response['response']['code'] != 0:
        logger.warning(f"{MODULE} {SUBMODULE_SHOULD_SAVE} failed, response:{response}")
        return False
    should_save = response['response']['data']['should_save']
    logger.debug(f"{MODULE} {SUBMODULE_SHOULD_SAVE}, response:{response}")
    return should_save


def interactive_ckpt_finish_save(rank, global_step, ckpt_id):
    message = {
        "rank": rank,
        "global_step": global_step,
        "ckpt_id": ckpt_id
    }
    response = to_scheduler(MODULE, SUBMODULE_FINISH_SAVE, message=message)
    if response['status'] != 'success' or response['response']['code'] != 0:
        logger.warning(f"{MODULE} {SUBMODULE_FINISH_SAVE} failed, request:{message}, response:{response}")
    logger.debug(f"{MODULE} {SUBMODULE_FINISH_SAVE}, message:{message}, response:{response}")


def to_scheduler(module: str, submodule: str, message: dict = None, retry_times=3):
    ip, app_id, scheduler_prefix = os.getenv("IP", None), os.getenv("APP_ID", None), os.getenv("SCHEDULER_URL", None)
    if not ip or not app_id or not scheduler_prefix:
        return {"status": "error", "err_msg": "Environment variables not set", "response": None}

    headers = {'ip': ip, 'app_id': app_id, 'submodule': submodule}
    url = f'{scheduler_prefix}{module}'
    message = dict() if not message else message
    data = json.dumps(message)
    if is_external_cluster():
        timeout = 120
    else:
        timeout = 10
    while True:
        try:
            response = requests.post(url, data=data, headers=headers, timeout=timeout)
            response = response.json()
            return response
        except Exception as e:
            retry_times -= 1
            if retry_times == 0:
                response = {'status': 'internal_exception',
                            'err_msg': f'Exception at to_scheduler: {repr(e)} \n {traceback.format_exc()}',
                            'response': None}
                return response
