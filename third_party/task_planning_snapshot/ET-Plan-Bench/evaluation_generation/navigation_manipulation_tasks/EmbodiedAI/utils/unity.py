from constant.dirs import UNITY_EXEC_FILE


import os
import subprocess
import time


def start_unity():
    os.environ["DISPLAY"] = ":1"
    p = subprocess.Popen(
        [
            UNITY_EXEC_FILE,
            "-batchmode",
        ],
        start_new_session=True,
    )
    time.sleep(4)
    return p


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