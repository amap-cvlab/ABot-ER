import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import numpy as np
import random
from tasks.utils import start_unity, auto_kill_unity
from task_generation_codes.scene_graph import VirtualHomeSceneGraph, MemorySceneGraph
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
        task_list=None
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
        # if(self.input_data_file == None):
        #     self._load_tasks()
        # else:
        #     self._load_tasks_CA()
        # self._load_tasks()

        self._scene_graph = None
        self.full_graph = None
        self._load_tasks_multi_obj()
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

    @property
    def scene_graph(self):
        if self._scene_graph is None:
            self.update_scene_graph()

        return self._scene_graph

    @property
    def room_names(self):
        return self.scene_graph.room_names

    @property
    def room_ids(self):
        return self.scene_graph.room_ids

    @property
    def object_name_ids_dict(self):
        return self.scene_graph.object_name_ids_dict

    @property
    def id_instance_name_dict(self):
        return self.scene_graph.id_instance_name_dict

    @property
    def instance_name_id_dict(self):
        return self.scene_graph.instance_name_id_dict

    @property
    def characters(self):
        return self.scene_graph.characters

    @property
    def object_id_in_room_id_dict(self):
        return self.scene_graph.object_id_in_room_id_dict

    @property
    def room_id_object_id_dict(self):
        return self.scene_graph.room_id_object_id_dict

    @property
    def full_nodes_list(self):
        return self.scene_graph.full_nodes_list

    def get_object_name_by_id(self, object_id: int):
        return self.scene_graph.nodes()[object_id]["class_name"]

    def get_object_by_id(self, object_id: int):
        return self.scene_graph.get_full_node(object_id)

    def update_scene_graph(self):
        self._scene_graph = VirtualHomeSceneGraph(self.full_graph)

    def get_agent_room_id(self):
        room_ids = []
        room_names = []
        for edge in self.full_graph["edges"]:
            if (
                edge["from_id"] == 1 and edge["relation_type"] == "INSIDE"
            ):  # agent id assumed to be 1
                room_id = edge["to_id"]
                room_name = self.get_object_name_by_id(room_id)
                if room_name not in self.room_names:
                    print(
                        "Agent in non-room object: %s; id: %s".format(room_name, room_id
                    ))
                else:
                    room_ids.append(room_id)
                    room_names.append(room_name)

        if len(room_ids) > 1:
            print(
                "Agent in more than one room: %s; ids: %s".format(room_names, room_ids
            ))
        elif len(room_ids) == 0:
            print("Agent not in room!")
            raise RuntimeError("Agent not in room!")

        # self.logger.info(
        #     "Agent is in room: %s; id: %s",
        #     [self.get_object_name_by_id(room_id) for room_id in room_ids],
        #     room_ids,
        # )

        return room_ids[0]

        
    def _load_tasks(self):
        simple_tasks = self._load_simple_tasks_info()
        complex_tasks = self._load_complex_tasks_info()
        self.tasks = simple_tasks + complex_tasks
        print(
            f"load simple task num: {len(simple_tasks)}, complex task num: {len(complex_tasks)}, total: {len(self.tasks)}"
        )

    def _load_tasks_CA(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        task_file = os.path.join(
            root_dir,
            "tasks/data/virtualhome/generated_tasks/"+self.input_data_file,
        )

        tasks = []
        with open(task_file, 'r') as json_file:
            json_list = list(json_file)

        for task_id, json_str in enumerate(json_list):
            line = json.loads(json_str)

            tasks.append({
                "env_id": int(line["env_id"]),
                "task_description": line["task"],
                "task_conditions": TaskChecker.parse_conditions(
                    line["task_completion_criterion"], pass_preprocess=True
                ),
                "task_completion_criterion": line["task_completion_criterion"],
                "object1": line["object1"],
                "object2": line["object2"],
            })
        
        self.tasks = tasks
        print(
            f"load tasks total: {len(self.tasks)}"
        )
    
    def _load_tasks_multi_obj(self):
        with open(self.input_data_file, 'r') as json_file:
            json_list = list(json_file)
        tasks = []
        for task_id, json_str in enumerate(json_list):
            line = json.loads(json_str)

            tasks.append({
                "env_id": int(line["env_id"]),
                "task_description": line["task"],
                "task_conditions": TaskChecker.parse_conditions(
                    line["task_completion_criterion"], pass_preprocess=True
                ),
                "task_completion_criterion": line["task_completion_criterion"],
                "object1": line["object1"],
                "object2": line["object2"],
                "object3": line["object3"]
            })
        
        self.tasks = tasks
        print(
            f"load tasks total: {len(self.tasks)}"
        )
        

    def _load_simple_tasks_info(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        task_file = os.path.join(
            root_dir,
            # "tasks/data/virtualhome/generated_tasks/simple_single_object_tasks_env_0.jsonl",
            "tasks/data/virtualhome/generated_tasks/all_simple_task.json",
        )
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
        return tasks

    def _load_complex_tasks_info(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        task_file = os.path.join(
            root_dir,
            "tasks/data/virtualhome/generated_tasks/complex_single_object_tasks_env_0.jsonl",
        )
        tasks = []
        with open(task_file, "r", encoding="utf-8") as f:
            for line in jsonlines.Reader(f):
                response = line["response"]
                task_description = TaskChecker.parse_task_description(response)
                if task_description is None:
                    continue
                task_conditions = TaskChecker.parse_conditions(response)
                if task_conditions is None:
                    continue
                tasks.append(
                    {
                        "env_id": line["env_id"],
                        "task_description": task_description,
                        "task_conditions": task_conditions,
                    }
                )
                # print(f"task: {task_description}, condition: {task_conditions}")
        return tasks

    def get_observation(self) -> dict:

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

        obs = {
            "task_description": self.task_info["task_description"],
            "full_graph": full_graph,
            "partial_graph": get_visible_nodes_in_same_room(full_graph, agent_id=1),
            "visible_graph": get_visible_graph(
                full_graph, list(visible_objects.keys())
            ),
            "rgb": get_first_person_image(mode="normal"),
            "seg_inst": get_first_person_image(mode="seg_inst"),
            "depth": get_first_person_image(mode="depth"),
        }
        
        return obs

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
        self.full_graph = full_graph
        _, visible_objects = self.comm.get_visible_objects(self.first_person_camera_id)

        # print(get_first_person_image(mode="normal"))

        img = Image.fromarray(np.uint8(np.array(get_first_person_image(mode="normal"))), 'RGB')
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

    def reset(
        self,
        task_id=None,
        init_room=None,
    ) -> dict:
        self.steps += 1

        if task_id is None:
            task_id = random.randint(0, self.task_num - 1)
        else:
            assert task_id < self.task_num, f"task id should in [0, {self.task_num-1}]"
        self.task_info = self.tasks[task_id]
        env_id = self.task_info["env_id"]

        # print("Check connection: ")
        # self.comm.check_connection()
        # print(f"Resetting env: {env_id}")
        self.comm.reset(env_id)

        if init_room is not None:
            self.comm.add_character(initial_room=init_room)
        else:
            self.comm.add_character()

        _, camera_count = self.comm.camera_count()
        self.first_person_camera_id = camera_count - 6

        obs = self.get_observation()
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
        # env_id = self.task_info["env_id"]

        # print("Check connection: ")
        # self.comm.check_connection()
        # print(f"Resetting env: {env_id}")
        self.comm.reset(env_id)

        if init_room is not None:
            self.comm.add_character(initial_room=init_room)
        else:
            self.comm.add_character()

        _, camera_count = self.comm.camera_count()
        self.first_person_camera_id = camera_count - 6

        obs = self.get_observation_CA()
        return obs

    def step(
        self, script: Union[str, List[str]], recording=False
    ) -> Tuple[dict, float, bool, bool, dict]:
        if isinstance(script, str):
            script = [script]
        success, message = self.comm.render_script(
            script,
            recording=recording,
            skip_animation=not recording,
            # camera_mode="FIRST_PERSON",
        )
        if not success:
            print("Not success")
            print(message)
            info = {'msg': message}
        else:
            info = {'msg': None}

        obs = self.get_observation()
        done = self.is_done()
        truncated = self.is_truncated()
        reward = self.calculate_reward(done)
        # info = {'message': message}

        return obs, reward, done, truncated, info

    def step_CA(
        self, script: Union[str, List[str]], save_img_path, recording=False
    ) -> Tuple[dict, float, bool, bool, dict]:
        if isinstance(script, str):
            script = [script]
        try:
            success, message = self.comm.render_script(
                script,
                recording=recording,
                skip_animation=not recording,
                # camera_mode="FIRST_PERSON",
            )
        except:
            success = None
            print("Action not implemented", script)
            return None, None, None, None, None
        # print("success inside step_CA:", success)
        # print("message inside step_CA:", message)
        if not success:
            print("Not success")
            print(message)
            info = {'msg': message}
        else:
            info = {'msg': None}

        obs = self.get_observation_CA(save_img_path)
        # done = self.is_done()
        truncated = self.is_truncated()
        # reward = self.calculate_reward(done)
        # info = {'message': message}

        # return obs, reward, done, truncated, info
        return obs, None, None, truncated, info

    def calculate_reward(self, done):
        return 1.0 if done else 0.0

    def is_done(self):
        _, curr_graph = self.comm.environment_graph()
        curr_graph = VirtualHomeSceneGraph(curr_graph)
        return TaskChecker.is_success(curr_graph, self.task_info["task_conditions"])

    def is_truncated(self):
        return self.steps >= self.max_episode_length

    def close(self):
        self.comm.close()
        if self.unity_process is not None:
            self.unity_process.kill()

class TaskEnv:
    def __init__(
        self,
        env: VirtualHomeNavigationEnv,
        task_complete,
        task_id,
        save_dir,
        use_partial_graph=False,
    ):
        self.env = env
        self.task_complete = task_complete
        self.task_id = task_id
        self.save_dir = save_dir
        self.use_partial_graph = use_partial_graph

        self._prepare_task_info()
        self._prepare_dirs()

        self.memory_scene_graph = MemorySceneGraph({"nodes": [], "edges": []})

        self.step_num = 0
        self.action_history = []
        self.init_room_name = None
        self.init_room_id = None
        self.init_state = None

        self.done = False

        self.to_save = {
            "success": self.done,
            "input": self.task,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "task_complete": self.task_complete,
            "action_history": [],
            "steps": {},
        }

        # self.logger = get_logger(get_main_file_abs_path())

    @property
    def objects_seen(self):
        return self.memory_scene_graph.objects_seen

    @property
    def items_seen(self):
        return self.memory_scene_graph.items_seen

    @property
    def items_remaining(self):
        return self.memory_scene_graph.items_remaining

    @property
    def items_explored(self):
        return self.memory_scene_graph.items_explored

    # @property
    # def containers_seen(self):
    #     return self.memory_scene_graph.containers_seen

    @property
    def rooms_explored(self):
        return self.memory_scene_graph.rooms_explored

    # @property
    # def containers_explored(self):
    #     return self.memory_scene_graph.containers_explored

    @property
    def rooms_remaining(self):
        return self.memory_scene_graph.rooms_remaining

    # @property
    # def containers_remaining(self):
    #     return self.memory_scene_graph.containers_remaining

    def _prepare_task_info(self):
        self.task = self.task_complete["task"]
        self.env_id = self.task_complete["env_id"]
        self.task_completion_criterion = self.task_complete["task_completion_criterion"]

    def _prepare_dirs(self):
        self.save_path = os.path.join(self.save_dir, f"task_{self.task_id}.json")
        self.image_save_dir = os.path.join(self.save_dir, "RGB")
        self.image_save_dir_for_task = os.path.join(
            self.image_save_dir, f"task_{self.task_id}"
        )

    def _prepare_room_info(self):
        # assume agent know all the rooms
        for room_id in self.env.room_ids:
            self.rooms_remaining[room_id] = self.env.get_object_by_id(room_id)
            # self.rooms_seen[room_id] = self.env.get_object_by_id(room_id)

        # self.logger.info("All rooms:%s, %s", self.env.room_names, self.env.room_ids)

    # def update_containers_explored(self, container_id:int):
    #     self.memory_scene_graph.containers_explored[container_id] = (
    #         self.env.get_object_by_id(container_id)
    #     )

    #     if container_id in self.containers_remaining:
    #         del self.containers_remaining[container_id]

    def update_items_explored(self, item_id: int):
        self.memory_scene_graph.items_explored[item_id] = self.env.get_object_by_id(
            item_id
        )

        if item_id in self.items_remaining:
            del self.items_remaining[item_id]

    def update_rooms_explored(self, room_id: int):
        self.memory_scene_graph.rooms_explored[room_id] = self.env.get_object_by_id(
            room_id
        )

        if room_id in self.rooms_remaining:
            del self.rooms_remaining[room_id]

    def reset_env_room(self, room, obs, port_num=3456, env_reset_max_retry=3):
        # self.logger.info("Env reset with port %d", port_num)

        fail_count = 0

        
        while env_reset_max_retry >= fail_count:
            try:
                if fail_count > 0:
                    print("Restarting the env")
                    # self.logger.info("Env reset failed with port %d", port_num)
                    # port_num += random.randint(1, 500)
                    # self.logger.info("port change to %d", port_num)
                    # self.env = VirtualHomeEnv(port=str(port_num))

                # obs = self.env.reset_CA(env_id=self.env_id, room=room)
                if self.init_room_name is None:
                    self.init_room_name = room
                break
            except Exception as e:
                print("reset_env_room failed: %s".format(e))
                fail_count += 1
                if fail_count > env_reset_max_retry:
                    print("env_reset_max_retry(%s) reached!".format(env_reset_max_retry))
                    break

        if obs is None:
            print("Failed to reset env room!")
            raise RuntimeError("Failed to reset env room!")

        # current_room_id = self.env.get_agent_room_id
        # self.rooms_explored[current_room_id] = self.env.get_object_by_id(current_room_id)
        # do not need to remove room from rooms_remaining

        self.update_memory(obs)  # must update rooms_explored first

        self._prepare_room_info()

        if self.init_room_id is None:
            self.init_room_id = self.env.get_agent_room_id()

        character = self.env.get_object_by_id(1)
        self.init_state = {
            key: character[key]
            for key in [
                "prefab_name",
                "obj_transform",
                "bounding_box",
                "properties",
                "states",
            ]
        }

    def _update_to_save_steps(self, step_info):
        self.to_save["steps"].update(step_info)
        # self.logger.info("STEP_%s", str(self.step_num))
        self.step_num += 1

    def save(self):
        # self.logger.info("success: %s", self.done)
        # self.logger.info("previous_generated_steps: %s", self.action_history)

        # save final status
        self.to_save["success"] = self.done
        self.to_save["init_room_name"] = self.init_room_name
        self.to_save["init_room_id"] = self.init_room_id
        self.to_save["init_state"] = self.init_state
        self.to_save["action_history"] = self.action_history

        with open(self.save_path, "w", encoding="utf-8") as outfile:
            json.dump(self.to_save, outfile)

    def _env_step(self, action, save_img=False):
        if save_img:
            save_img_path = os.path.join(
                self.image_save_dir_for_task, str(self.step_num) + "_0.png"
            )
        else:
            save_img_path = None

        success, info = self.env.step(action, recording=False)

        obs = None
        if success:
            obs = self.env.get_observation_360(save_img_path)
            self.update_memory(obs)

        return obs, success, info

    def update_memory(self, obs):
        if self.use_partial_graph:
            graph = obs["partial_graph"]
        else:
            graph = obs["visible_graph"]

        self.memory_scene_graph.add_nodes(graph["nodes"])
        self.memory_scene_graph.add_edges(graph["edges"])

    def get_object_position_2d(self, object_id: int):
        return self.env.get_object_by_id(object_id)["bounding_box"]["center"][:2]

    def get_object_ids_sorted_distance(self, object_ids):
        if isinstance(object_ids, dict):
            object_ids = list(object_ids.keys())

        if len(object_ids) < 2:
            return object_ids

        agent_position = self.get_object_position_2d(1)
        object_postions = [
            self.get_object_position_2d(object_id) for object_id in object_ids
        ]

        distances = distance(agent_position, object_postions)

        object_ids_sorted = [x for _, x in sorted(zip(distances, object_ids))]

        return object_ids_sorted

    def check_relation(self, object_id_1: int, object_id_2: int, relation: str):
        try:
            relation_type = self.memory_scene_graph[object_id_1][object_id_2]
            return relation in [rel["relation_type"] for rel in relation_type.values()]
        except KeyError:
            return None

    @staticmethod
    def get_listed_object_ids_from_name(object_name: str, objects: Union[dict, List]):
        # TODO: add more acceptable types for objects
        object_ids = []

        if isinstance(objects, dict):
            objects = objects.values()

        for object in objects:
            if object_name == object["class_name"]:
                object_ids.append(object["id"])

        return object_ids

    @staticmethod
    def get_unique_value_class_name(object_dict: dict):
        object_names = set()
        for _, object in object_dict.items():
            object_names.add(object["class_name"])

        return list(object_names)

    @staticmethod
    def get_name_ids_from_node_list(node_list: List):
        name_ids = defaultdict(list)
        for node in node_list:
            name_ids[node["class_name"]].append(node["id"])

        return name_ids

@auto_kill_unity(kill_before_return=True)
def test():
    condition = "(CLOSE, character, microwave)"
    env = VirtualHomeNavigationEnv(task_finish_condition=condition)
    task_id = random.randint(0, env.task_num - 1)
    obs = env.reset(task_id=task_id, init_room="bedroom")
    g = obs["full_graph"]

    env.is_done()
    microwave_id = [
        node["id"] for node in g["nodes"] if node["class_name"] == "microwave"
    ][0]
    plt.figure(figsize=(15, 10))
    plt.subplot(131)
    plt.imshow(obs["rgb"][:, :, ::-1])
    plt.subplot(132)
    plt.imshow(obs["seg_inst"])
    plt.subplot(133)
    plt.imshow(obs["depth"])

    action = f"<char0> [walk] <microwave> ({microwave_id})"
    print(f"action: {action}")
    obs, reward, done, trucated, info = env.step(action, recording=False)
    print(f"reward: {reward}, done: {done}, trucated: {trucated}")
    plt.figure(figsize=(15, 10))
    plt.subplot(131)
    plt.imshow(obs["rgb"][:, :, ::-1])
    plt.subplot(132)
    plt.imshow(obs["seg_inst"])
    plt.subplot(133)
    plt.imshow(obs["depth"])

if __name__ == "__main__":
    test()
