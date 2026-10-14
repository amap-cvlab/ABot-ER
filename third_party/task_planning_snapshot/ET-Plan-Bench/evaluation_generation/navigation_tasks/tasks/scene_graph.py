import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

if root_dir not in sys.path:
    sys.path.append(root_dir)
from simulator.virtualhome.virtualhome.simulation import unity_simulator
from simulator.virtualhome.virtualhome.simulation.unity_simulator import comm_unity
import os
from collections import defaultdict
import networkx as nx
from copy import deepcopy
import numpy as np
from typing import Union, List
from itertools import product

class VirtualHomeSceneGraph(nx.DiGraph):
    node_categories = [
        "Rooms",
        "Floor",
        "Walls",
        "Ceiling",
        "Doors",
        "Furniture",
        "Appliances",
        "Lamps",
        "Props",
        "Decor",
        "Electronics",
        "Food",
        "Windows",
        "Floors",
    ]
    node_properties = [
        "SURFACES",
        "CAN_OPEN",
        "CONTAINERS",
        "MOVABLE",
        "GRABBABLE",
        "SITTABLE",
        "RECIPIENT",
        "HAS_SWITCH",
        "HAS_PLUG",
        "EATABLE",
        "CUTTABLE",
        "POURABLE",
        "CREAM",
        "COVER_OBJECT",
        "READABLE",
        "HAS_PAPER",
        "LIEABLE",
        "HANGABLE",
        "CLOTHES",
        "LOOKABLE",
    ]

    def __init__(self, data):
        super().__init__()
        self._add_nodes(data["nodes"])
        self._add_edges(data["edges"])
        self._add_extra_edges()
        self._init_characters()

    def _init_characters(self):
        self._characters = []
        for node in self.full_nodes:
            if node["class_name"] == "character":
                self._characters.append(node)

    @property
    def characters(self):
        return self._characters

    def count_properties(self):
        count = defaultdict(int)
        class_names = defaultdict(set)
        nodes = list(self.nodes(data=True))
        for _, attr in nodes:
            for p in attr["properties"]:
                count[p] += 1
                class_names[p].add(attr["class_name"])
        return count, class_names

    def _add_node(self, node: dict):
        self._class_name_count[node["class_name"]] += 1
        super().add_node(
            node["id"],
            category=node["category"],
            class_name=node["class_name"],
            instance_name=node["class_name"]
            + f"_{self._class_name_count[node['class_name']]}",
            prefab_name=node["prefab_name"],
            obj_transform=node["obj_transform"],
            bounding_box=node["bounding_box"],
            properties=node["properties"],
            states=node["states"],
        )

    def _add_nodes(self, nodes: list):
        self._class_name_count = defaultdict(int)
        for d in nodes:
            self._add_node(d)
        self.full_nodes = [self._get_node(i) for i in self.nodes(data=False)]

    def _add_edge(self, edge: dict):
        super().add_edge(
            edge["from_id"],
            edge["to_id"],
            relation_type=edge["relation_type"],
        )

    def _add_edges(self, edges: list):
        for e in edges:
            self._add_edge(e)

    def _add_extra_edges(self):
        DISTANCE_THRESH_MAX = 0.3
        DISTANCE_THRESH_MIN = 1e-6
        for i in range(len(self.full_nodes)):
            for j in range(i + 1, len(self.full_nodes)):
                node_i, node_j = self.full_nodes[i], self.full_nodes[j]
                if self.has_edge(node_i["id"], node_j["id"]) or self.has_edge(
                    node_j["id"], node_i["id"]
                ):
                    continue
                pos_i = np.array(node_i["obj_transform"]["position"])
                pos_j = np.array(node_j["obj_transform"]["position"])
                if (
                    DISTANCE_THRESH_MIN
                    < np.linalg.norm(pos_i - pos_j)
                    < DISTANCE_THRESH_MAX
                ):
                    self._add_edge(
                        dict(
                            from_id=node_i["id"],
                            to_id=node_j["id"],
                            relation_type="CLOSE",
                        )
                    )
                    self._add_edge(
                        dict(
                            from_id=node_j["id"],
                            to_id=node_i["id"],
                            relation_type="CLOSE",
                        )
                    )

    def _get_node(self, node_id: int):
        node = deepcopy(self.nodes[node_id])
        node["id"] = node_id
        return node

    def filter_nodes(
        self, nodes: list = None, categories=[], class_names=[], properties=[]
    ):
        if nodes is None:
            nodes = self.full_nodes

        if isinstance(categories, str):
            categories = [categories]
        if isinstance(class_names, str):
            class_names = [class_names]
        if isinstance(properties, str):
            properties = [properties]

        selected_nodes = []
        for node in nodes:
            if len(categories) and node["category"] not in categories:
                continue
            if len(class_names) and node["class_name"] not in class_names:
                continue
            if len(properties):
                match = False
                for property in node["properties"]:
                    if property in properties:
                        match = True
                        break
                if not match:
                    continue
            selected_nodes.append(node)
        return selected_nodes  # id list

    def successors(
        self,
        node: dict,
        relation_types: list = [],
        categories=[],
        class_names=[],
        properties=[],
        with_relation=False,
    ):
        successor_node_ids = super().successors(node["id"])
        if len(relation_types) == 0:
            selected_nodes = [self._get_node(id) for id in successor_node_ids]
        else:
            selected_nodes = []
            for id in successor_node_ids:
                relation = self.edges[node["id"], id]["relation_type"]
                if relation in relation_types:
                    selected_nodes.append(self._get_node(id))

        selected_nodes = self.filter_nodes(
            selected_nodes, categories, class_names, properties
        )
        if not with_relation:
            return selected_nodes
        else:
            relation_types = []
            for node_ in selected_nodes:
                relation_types.append(
                    self.edges[node["id"], node_["id"]]["relation_type"]
                )
            return selected_nodes, relation_types

    def predecessors(
        self,
        node: dict,
        relation_types: list = [],
        categories: list = [],
        class_names: list = [],
        properties: list = [],
        with_relation: bool = False,
    ):
        predecessor_node_ids = super().predecessors(node["id"])
        if len(relation_types) == 0:
            selected_nodes = [self._get_node(id) for id in predecessor_node_ids]
        else:
            selected_nodes = []
            for id in predecessor_node_ids:
                relation = self.edges[node["id"], id]["relation_type"]
                if relation in relation_types:
                    selected_nodes.append(self._get_node(id))
        selected_nodes = self.filter_nodes(
            selected_nodes, categories, class_names, properties
        )
        if not with_relation:
            return selected_nodes
        else:
            relation_types = []
            for node_ in selected_nodes:
                relation_types.append(
                    self.edges[node_["id"], node["id"]]["relation_type"]
                )
            return selected_nodes, relation_types

    def _get_candidate_objects(self, obj: Union[int, str, List[int]]) -> list:
        if isinstance(obj, int):  # node id
            candidates = [self._get_node(node_id=obj)]
        elif isinstance(obj, str):  # node class_name
            assert "_" not in obj
            candidates = []
            for node in self.full_nodes:
                if node["class_name"] == obj:
                    candidates.append(node)
        elif isinstance(obj, list):  # a list of node id
            candidates = [self._get_node(x) for x in obj]
        else:
            raise NotImplementedError
        return candidates

    def is_close(
        self,
        obj1: Union[int, str, List[int]],
        obj2: Union[int, str, List[int]],
        DISTANCE_THRESH_MAX=1.0,
    ) -> tuple[bool, list]:
        """
        Args:
            obj1:
                1) int: id of the object.
                2) str: class name of the object.
                3) list: A list of ids
            obj2:
                1) int: id of the object.
                2) str: class name of the object.
                3) list: A list of ids
        Returns:
            result: bool
            success_objects_list: a list of tuples (matching objects).
        """
        DISTANCE_THRESH_MIN = 1e-6
        if obj1 == "character" or obj2 == "character":
            DISTANCE_THRESH_MAX = 1.3

        obj1_candidates = self._get_candidate_objects(obj1)
        obj2_candidates = self._get_candidate_objects(obj2)

        if len(obj1_candidates) == 0:
            print(f"Unable to find a matching object {obj1} !!")
            return False, []
        if len(obj2_candidates) == 0:
            print(f"Unable to find a matching object {obj2} !!")
            return False

        success_objects_list = []
        for candidate_1, candidate_2 in product(obj1_candidates, obj2_candidates):
            pos_1 = np.array(candidate_1["obj_transform"]["position"])[
                :2
            ]  # only consider x, y
            pos_2 = np.array(candidate_2["obj_transform"]["position"])[
                :2
            ]  # only consider x, y
            if (
                DISTANCE_THRESH_MIN
                < np.linalg.norm(pos_1 - pos_2)
                < DISTANCE_THRESH_MAX
            ):
                success_objects_list.append((candidate_1, candidate_2))

            # print(f"distance: {np.linalg.norm(pos_1 - pos_2)}")
        result = len(success_objects_list) > 0
        return result, success_objects_list

    def is_on(
        self,
        source_obj: Union[int, str, List[int]],
        target_obj: Union[int, str, List[int]],
        only_closest=False,
    ):
        # 1. Find the nearest source object to the character
        source_candidates = self._get_candidate_objects(source_obj)
        if only_closest:
            character_pos = self.characters[0]["obj_transform"]["position"]
            source_candidates.sort(
                key=lambda x: np.linalg.norm(
                    np.array(character_pos) - np.array(x["obj_transform"]["position"])
                )
            )
            source_candidates = [source_candidates[0]]

        # 2. Find all target objects
        target_candidates = self._get_candidate_objects(target_obj)

        # 3. if source obj in on a target object
        success_objects_list = []
        for candidate_1, candidate_2 in product(source_candidates, target_candidates):
            if (
                self.has_edge(candidate_1["id"], candidate_2["id"])
                and self.edges[candidate_1["id"], candidate_2["id"]] == "ON"
            ):
                success_objects_list.append((candidate_1, candidate_2))
        result = len(success_objects_list) > 0
        return result, success_objects_list

    def is_inside(
        self,
        source_obj: Union[int, str, List[int]],
        target_obj: Union[int, str, List[int]],
    ):
        source_candidates = self._get_candidate_objects(source_obj)
        target_candidates = self._get_candidate_objects(target_obj)
        success_objects_list = []
        for candidate_1, candidate_2 in product(source_candidates, target_candidates):
            if (
                self.has_edge(candidate_1["id"], candidate_2["id"])
                and self.edges[candidate_1["id"], candidate_2["id"]] == "INSIDE"
            ):
                success_objects_list.append((candidate_1, candidate_2))

        result = len(success_objects_list) > 0
        return result, success_objects_list
