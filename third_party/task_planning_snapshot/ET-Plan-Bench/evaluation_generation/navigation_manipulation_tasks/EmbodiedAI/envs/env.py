import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from constant.dirs import UNITY_EXEC_FILE, BASE_DIR
sys.path.append(BASE_DIR)

import json
import math
from collections import defaultdict
from typing import Union, List, Tuple
from PIL import Image
import numpy as np
from functools import reduce

from simulator.virtualhome.virtualhome.simulation import unity_simulator
from simulator.virtualhome.virtualhome.simulation.evolving_graph.utils import (
    get_visible_nodes as get_visible_nodes_in_same_room,
)
from envs.scene_graph import VirtualHomeSceneGraph, SceneGraphInside, MemorySceneGraph
from utils.unity import start_unity
from utils.helper import distance, get_main_file_abs_path
from utils.log import get_logger
from constant.params import STEP_MAX


class VirtualHomeEnv:
    def __init__(
        self,
        port=8088,
        image_width=640,
        image_height=480,
    ):
        self.image_height = image_height
        self.image_width = image_width

        self.logger = get_logger(get_main_file_abs_path())

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

        self.unity_process = None
        if os.environ.get("ET_PLAN_SKIP_START_UNITY", "1") != "1":
            self.unity_process = start_unity()

        no_graphics = os.environ.get("ET_PLAN_NO_GRAPHICS", "1") == "1"
        x_display = None if no_graphics else os.environ.get("ET_PLAN_X_DISPLAY", "0")
        timeout_wait = int(os.environ.get("ET_PLAN_TIMEOUT_WAIT", "60"))
        self.comm = unity_simulator.UnityCommunication(
            file_name=UNITY_EXEC_FILE,
            port=port,
            x_display=x_display,
            no_graphics=no_graphics,
            logging=False,
            timeout_wait=timeout_wait,
        )

        self.full_graph = None
        self.first_person_camera_id = None
        self._scene_graph = None
        self._scene_graph_inside = None

    @property
    def scene_graph(self):
        if self._scene_graph is None:
            self.update_scene_graph()

        return self._scene_graph

    @property
    def scene_graph_inside(self):
        if self._scene_graph_inside is None:
            self.update_scene_graph_inside()

        return self._scene_graph_inside

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

    def update_scene_graph_inside(self):
        self._scene_graph_inside = SceneGraphInside(self.full_graph)

    def get_visible_graph(self, full_graph: dict, visible_ids: list):
        visible_ids = list(map(int, visible_ids))
        nodes = [n for n in full_graph["nodes"] if n["id"] in visible_ids]
        edges = [
            e
            for e in full_graph["edges"]
            if e["from_id"] in visible_ids and e["to_id"] in visible_ids
        ]
        return {"edges": edges, "nodes": nodes}

    def get_visible_objects_dict(self, obs):
        first_person_view_visible_graph = obs["visible_graph"]

        visible_objects_dict = {}

        for item in first_person_view_visible_graph["nodes"]:
            visible_objects_dict.update({item["id"]: item})

        return visible_objects_dict

    def get_first_person_image(self, mode):
        _, images = self.comm.camera_image(
            [self.first_person_camera_id],
            mode=mode,
            image_width=self.image_width,
            image_height=self.image_height,
        )

        images = images[0]

        images = [
            images[:, :, 2],
            images[:, :, 1],
            images[:, :, 0],
        ]
        images = np.array(images)
        images = np.transpose(images, (1, 2, 0))
        images = Image.fromarray(np.uint8(images), "RGB")

        return images

    def get_observation_360(self, save_img_path=None) -> dict:
        _, self.full_graph = self.comm.environment_graph()
        visible_graph_parts = []

        if save_img_path is not None:
            os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
            first_person_view_image = self.get_first_person_image(mode="normal")
            first_person_view_image.save(save_img_path)

            save_img_path_pre = save_img_path.split(".")[0][:-1]

        _, visible_objects = self.comm.get_visible_objects(self.first_person_camera_id)

        for turn_num in range(1, 12):  # 360 degrees rotation in total
            # exe_action_list = [
            #     "<char0> [turnright]",
            #     "<char0> [turnright]",
            #     "<char0> [turnright]",
            # ]  # 90 degrees, 30 degrees for each

            turn_action = "<char0> [turnright]"

            success, _ = self.step(turn_action, recording=False)

            if success:
                if save_img_path is not None:
                    save_img_path = save_img_path_pre + str(turn_num) + ".png"
                    first_person_view_image = self.get_first_person_image(mode="normal")
                    first_person_view_image.save(save_img_path)

                _, visible_objects_after_turn = self.comm.get_visible_objects(
                    self.first_person_camera_id
                )
                visible_objects.update(visible_objects_after_turn)

                visible_graph_part = self.get_visible_graph(
                    self.full_graph, list(visible_objects_after_turn.keys())
                )
                visible_graph_parts.append(visible_graph_part)
            else:
                self.logger.warning("Turn Right Error")

        obs = {
            "full_graph": self.full_graph,
            "partial_graph": get_visible_nodes_in_same_room(
                self.full_graph, agent_id=1
            ),
            "visible_graph": self.get_visible_graph(
                self.full_graph, list(visible_objects.keys())
            ),
            "visible_graph_parts": visible_graph_parts,
            "rgb": save_img_path,
        }

        return obs

    def reset(
        self,
        env_id,
        room=None,
    ) -> dict:
        self.logger.info("Resetting env: %s", env_id)

        self.comm.reset(env_id)

        if room is not None:
            self.comm.add_character(initial_room=room)
        else:
            self.comm.add_character()

        _, camera_count = self.comm.camera_count()
        self.first_person_camera_id = camera_count - 6
        self.logger.info("first_person_camera_id: %s", self.first_person_camera_id)

        obs = self.get_observation_360()

        if self.full_graph is None:
            self.logger.error("env reset failed!")
            raise RuntimeError("env reset failed!")

        return obs

    def step(
        self, script: Union[str, List[str]], recording=False
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
        except Exception:
            success = False
            message = "Time out!"

        if not success:
            self.logger.warning(
                "Action not success! action: %s; message: %s", script, message
            )
            info = {"msg": message}
        else:
            info = {"msg": None}

        success = info["msg"] is None

        return success, info

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
                    self.logger.warning(
                        "Agent in non-room object: %s; id: %s", room_name, room_id
                    )
                else:
                    room_ids.append(room_id)
                    room_names.append(room_name)

        if len(room_ids) > 1:
            self.logger.warning(
                "Agent in more than one room: %s; ids: %s", room_names, room_ids
            )
        elif len(room_ids) == 0:
            self.logger.fatal("Agent not in room!")
            raise RuntimeError("Agent not in room!")

        self.logger.info(
            "Agent is in room: %s; id: %s",
            [self.get_object_name_by_id(room_id) for room_id in room_ids],
            room_ids,
        )

        return room_ids[0]

    def get_object_pos(self, object_id):
        return self.get_object_by_id(object_id)["bounding_box"]["center"]

    def close(self):
        self.comm.close()
        if self.unity_process is not None:
            self.unity_process.kill()


class TaskEnv:
    def __init__(
        self,
        env: VirtualHomeEnv,
        task_complete,
        task_id,
        save_dir,
        use_partial_graph=False,
        use_layout_map=False,
    ):
        self.env = env
        self.task_complete = task_complete
        self.task_id = task_id
        self.save_dir = save_dir
        self.use_partial_graph = use_partial_graph
        self.use_layout_map = use_layout_map

        self._prepare_task_info()
        self._prepare_dirs()

        self.memory_scene_graph = MemorySceneGraph({"nodes": [], "edges": []})

        self.step_num = 0
        self.action_history = []
        self.init_room_name = None
        self.init_room_id = None
        self.init_state = None
        self.target_object_id = None

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

        self.logger = get_logger(get_main_file_abs_path())

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
        # self.task_completion_criterion = self.task_complete["task_completion_criterion"]

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

        self.logger.info("All rooms:%s, %s", self.env.room_names, self.env.room_ids)

    def _get_default_target_object_id(self):
        if "id1" in self.task_complete.keys():
            self.target_object_id = self.task_complete["id1"]

        target_object_name = self.task_complete["object1"]
        self.target_object_id = self.env.object_name_ids_dict[target_object_name][0]

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

    def reset_env_room(self, room, port_num=3456, env_reset_max_retry=3):
        self.logger.info("Env reset with port %d", port_num)

        fail_count = 0

        obs = None
        while env_reset_max_retry >= fail_count:
            try:
                if fail_count > 0:
                    self.logger.info("Restarting the env")
                    # self.logger.info("Env reset failed with port %d", port_num)
                    # port_num += random.randint(1, 500)
                    # self.logger.info("port change to %d", port_num)
                    # self.env = VirtualHomeEnv(port=str(port_num))

                obs = self.env.reset(env_id=self.env_id, room=room)
                if self.init_room_name is None:
                    self.init_room_name = room
                break
            except Exception as e:
                self.logger.error("reset_env_room failed: %s", e)
                fail_count += 1
                if fail_count > env_reset_max_retry:
                    self.logger.warning(
                        "env_reset_max_retry(%s) reached!", env_reset_max_retry
                    )
                    break

        if obs is None:
            self.logger.fatal("Failed to reset env room!")
            raise RuntimeError("Failed to reset env room!")

        # current_room_id = self.env.get_agent_room_id
        # self.rooms_explored[current_room_id] = self.env.get_object_by_id(current_room_id)
        # do not need to remove room from rooms_remaining

        self.update_memory(
            obs, use_layout_map=self.use_layout_map
        )  # must update rooms_explored first

        self._prepare_room_info()
        self._get_default_target_object_id()

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
        self.logger.info("STEP_%s", str(self.step_num))
        self.step_num += 1

    def save(self):
        self.logger.info("success: %s", self.done)
        self.logger.info("previous_generated_steps: %s", self.action_history)

        # save final status
        self.to_save["success"] = self.done
        self.to_save["init_room_name"] = self.init_room_name
        self.to_save["init_room_id"] = self.init_room_id
        self.to_save["init_state"] = self.init_state
        self.to_save["action_history"] = self.action_history
        self.to_save["traverse_distance"] = self.calculate_distance_traversal()
        self.to_save["init_distance"] = self.get_init_distance()
        self.to_save["item_size"] = self.get_item_size()

        with open(self.save_path, "w", encoding="utf-8") as outfile:
            json.dump(self.to_save, outfile)

    def save_abstract(self, task_abstract_complete: dict, found_object_id):
        if not found_object_id:
            self.to_save["execution_success"] = False
            self.to_save["object_correct"] = False
        else:
            self.to_save["execution_success"] = True
            if found_object_id in task_abstract_complete["acceptable_object_ids"]:
                self.to_save["object_correct"] = True
                self.done = True
            else:
                self.to_save["object_correct"] = False

        self.logger.info("execution_success: %s", self.to_save["execution_success"])
        self.logger.info("object_correct: %s", self.to_save["object_correct"])
        self.save()

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

    def llm_step_save(self, action, prompt, response, step_cap=True, save_img=False):
        self.logger.info("Performing action: %s", action)
        obs, success, info = self._env_step(action, save_img)

        if success:
            step_info = {
                self.step_num: {
                    "obs": obs,
                    "prompt": prompt,
                    "gpt_response": response,
                    "action": action,
                    "msg": info["msg"],
                }
            }
            self._update_to_save_steps(step_info)
            self.action_history.append(action)

        if step_cap and self.step_num > STEP_MAX:
            self.logger.info("maximum number of steps reached! step: %s", self.step_num)
            success = False

        return obs, success, info

    def update_memory(self, obs, use_layout_map=False):
        if self.use_partial_graph:
            graph = obs["partial_graph"]
        else:
            graph = obs["visible_graph"]

        self.memory_scene_graph.add_nodes(graph["nodes"])
        self.memory_scene_graph.add_edges(graph["edges"])

        if use_layout_map:
            # may need to add edges
            large_object_nodes = self.get_large_objects_from_room(obs["full_graph"])
            self.memory_scene_graph.add_nodes(large_object_nodes)

    def get_large_objects_from_room(self, full_graph):
        large_item_list = []

        for node in full_graph["nodes"]:
            if "LIEABLE" in node["properties"] or (
                "MOVEABLE" not in node["properties"]
                and "GRABBABLE" not in node["properties"]
            ):
                large_item_list.append(node)

        return large_item_list

    def get_object_position(self, object_id: int):
        return self.env.get_object_by_id(object_id)["obj_transform"]["position"]

    def get_object_ids_sorted_distance(self, object_ids):
        if isinstance(object_ids, dict):
            object_ids = list(object_ids.keys())

        if len(object_ids) < 2:
            return object_ids

        agent_position = self.get_object_position(1)
        object_postions = [
            self.get_object_position(object_id) for object_id in object_ids
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

    def calculate_distance_traversal(self):
        init_pos = self.init_state["obj_transform"]["position"]
        action_history_walk = [
            action.strip() for action in self.action_history if "[walk]" in action
        ]
        dest_ids = []
        for action_walk in action_history_walk:
            dest_id = int(action_walk.split("(")[-1].split(")")[0])
            dest_ids.append(dest_id)

        dest_positions = [self.get_object_position(id) for id in dest_ids]
        traverse_distance = 0
        current_pos = init_pos
        for dest_pos in dest_positions:
            traverse_distance += math.dist(current_pos, dest_pos)
            current_pos = dest_pos

        return traverse_distance

    def get_init_distance(self):
        init_pos = self.init_state["obj_transform"]["position"]
        target_object_pos = self.get_object_position(self.target_object_id)
        init_distance = math.dist(init_pos, target_object_pos)

        return init_distance

    def get_item_size(self):
        item_dim = self.env.get_object_by_id(self.target_object_id)["bounding_box"]["size"]
        item_size = float(reduce(lambda x, y: x*y, item_dim))

        return item_size

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
