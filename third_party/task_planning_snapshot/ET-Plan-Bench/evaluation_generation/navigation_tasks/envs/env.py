import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import numpy as np
import random
from tasks.utils import start_unity, auto_kill_unity
from tasks.scene_graph import VirtualHomeSceneGraph
from tasks.task_checker import TaskChecker
from simulator.virtualhome.virtualhome.simulation import unity_simulator
from simulator.virtualhome.virtualhome.simulation.unity_simulator import comm_unity
from simulator.virtualhome.virtualhome.simulation.evolving_graph.utils import (
    get_visible_nodes as get_visible_nodes_in_same_room,
)
import jsonlines, json
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

from PIL import Image

class VirtualHomeNavigationEnv:
    def __init__(
        self,
        port=8080,
        max_episode_length=20,
        seed=0,
        image_width=640,
        image_height=480,
        task_finish_condition: str = "",
        input_data_file=None,
    ):
        self.max_episode_length = max_episode_length
        self.seed = seed
        self.image_height = image_height
        self.image_width = image_width
        self.steps = 0
        self.task_finish_condition = task_finish_condition
        self.input_data_file = input_data_file

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.actions_available = [
            "walk",
            "run",
            "walktowards",
            "walkforward",
            "turnleft",
            "turnright",
            "sit",
            "standup",
            "grab",
            "open",
            "close",
            "put",
            "putin",
            "switchon",
            "switchoff",
            "drink",
            "touch",
            "lookat",
        ]
        if(self.input_data_file.split('/')[-1] == 'navigation_tasks.json'):
            self._load_tasks_simple(self.input_data_file)
        else:
            self._load_tasks_CA(self.input_data_file)
        self.unity_process = None
        if os.environ.get("ET_PLAN_SKIP_START_UNITY", "1") != "1":
            self.unity_process = start_unity()

        unity_exec_file = os.path.abspath(
            os.path.join(
                root_dir,
                "../../simulator/virtualhome/virtualhome/simulation/"
                "unity_simulator_2.3.0/linux_exec.v2.3.0.x86_64",
            )
        )
        no_graphics = os.environ.get("ET_PLAN_NO_GRAPHICS", "1") == "1"
        x_display = None if no_graphics else os.environ.get("ET_PLAN_X_DISPLAY", "0")
        timeout_wait = int(os.environ.get("ET_PLAN_TIMEOUT_WAIT", "60"))
        print("unity_exec_file:", unity_exec_file)
        self.comm = unity_simulator.UnityCommunication(
            file_name=unity_exec_file,
            port=str(port),
            x_display=x_display,
            no_graphics=no_graphics,
            timeout_wait=timeout_wait,
        )

    @property
    def task_num(self):
        return len(self.tasks)
    
    def _load_tasks_simple(self, task_file_path):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        task_file = task_file_path
        tasks = []
        with open(task_file, "r", encoding="utf-8") as f:
            for line in jsonlines.Reader(f):
                task_conditions = TaskChecker.parse_conditions(
                    line["task_completion_criterion"], pass_preprocess=True
                )
                if task_conditions is None:
                    continue
                tasks.append(
                    {
                        "env_id": line["env_id"],
                        "task_description": line["task"],
                        "task_conditions": task_conditions,
                    }
                )
            self.tasks = tasks
            print(
                f"load tasks total: {len(self.tasks)}"
            )

    def _load_tasks_CA(self, task_file_path):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        task_file = task_file_path

        tasks = []
        with open(task_file, 'r') as json_file:
            json_list = list(json_file)

        for task_id, json_str in enumerate(json_list):
            line = json.loads(json_str)

            tasks.append({
                "env_id": int(line["env_id"]),
                "task_description": line["task"],
                "task_completion_criterion": line["task_completion_criterion"],
                "object1": line["object_1"],
                "object2": line["object_2"],
            })
        
        self.tasks = tasks
        print(
            f"load tasks total: {len(self.tasks)}"
        )

    def get_observation_CA(self, save_img_path=None) -> dict:

        def get_first_person_image(mode):
            _, images = self.comm.camera_image(
                [self.first_person_camera_id],
                mode=mode,
                image_width=self.image_width,
                image_height=self.image_height,
            )
            return images[0]

        def get_visible_graph(full_graph: dict, visible_ids: list):
            visible_ids = list(map(int, visible_ids))
            nodes = [n for n in full_graph["nodes"] if n["id"] in visible_ids]
            edges = [
                e
                for e in full_graph["edges"]
                if e["from_id"] in visible_ids and e["to_id"] in visible_ids
            ]
            return dict(edges=edges, nodes=nodes)

        _, full_graph = self.comm.environment_graph()
        _, visible_objects = self.comm.get_visible_objects(self.first_person_camera_id)

        first_person_view_image = get_first_person_image(mode="normal")
        first_person_view_image = [first_person_view_image[:,:,2], first_person_view_image[:,:,1], first_person_view_image[:,:,0]]
        first_person_view_image = np.array(first_person_view_image)
        first_person_view_image = np.transpose(first_person_view_image, (1, 2, 0))

        img = Image.fromarray(np.uint8(np.array(first_person_view_image)), 'RGB')
        if(save_img_path == None):
            obs = {
                # "task_description": self.task_info["task_description"],
                "full_graph": full_graph,
                "partial_graph": get_visible_nodes_in_same_room(full_graph, agent_id=1),
                "visible_graph": get_visible_graph(
                    full_graph, list(visible_objects.keys())
                ),
            }
        else:
            img.save(save_img_path)

            obs = {
                # "task_description": self.task_info["task_description"],
                "full_graph": full_graph,
                "partial_graph": get_visible_nodes_in_same_room(full_graph, agent_id=1),
                "visible_graph": get_visible_graph(
                    full_graph, list(visible_objects.keys())
                ),
                "rgb": save_img_path,
            }
        
        return obs
    
    def reset_CA(
        self,
        env_id=0,
        task_id=None,
        init_room=None,
    ) -> dict:
        self.steps += 1

        if task_id is None:
            task_id = random.randint(0, self.task_num - 1)
        else:
            assert task_id < self.task_num, f"task id should in [0, {self.task_num-1}]"
        self.task_info = self.tasks[task_id]

        self.comm.reset(env_id)

        if init_room is not None:
            self.comm.add_character(initial_room=init_room)
        else:
            self.comm.add_character()

        _, camera_count = self.comm.camera_count()
        self.first_person_camera_id = camera_count - 6

        obs = self.get_observation_CA()
        return obs

    def step_CA(
        self, script: Union[str, List[str]], save_img_path, recording=False
    ) -> Tuple[dict, float, bool, bool, dict]:
        if isinstance(script, str):
            script = [script]
        success, message = self.comm.render_script(
            script,
            recording=recording,
            skip_animation=not recording,
            # camera_mode="FIRST_PERSON",
        )
        # print("success inside step_CA:", success)
        # print("message inside step_CA:", message)
        if not success:
            print("Not success")
            print(message)
            info = {'msg': message}
        else:
            info = {'msg': None}

        obs = self.get_observation_CA(save_img_path)

        truncated = self.is_truncated()

        return obs, None, None, truncated, info

    def is_truncated(self):
        return self.steps >= self.max_episode_length

    def close(self):
        self.comm.close()
        if self.unity_process is not None:
            self.unity_process.kill()
