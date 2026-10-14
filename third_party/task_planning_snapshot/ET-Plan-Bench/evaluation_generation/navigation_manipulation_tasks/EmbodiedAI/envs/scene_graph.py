import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from typing import Union, List
from copy import deepcopy
from collections import defaultdict

import networkx as nx

from utils.helper import get_main_file_abs_path
from utils.log import get_logger


class VirtualHomeSceneGraph(nx.MultiDiGraph):
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

        self._class_name_count = defaultdict(int)
        # self.id_object_name_dict = {}
        self.object_name_ids_dict = defaultdict(list)
        self.id_instance_name_dict = {}
        self.instance_name_id_dict = {}
        self.characters = []
        self._full_nodes_list = None
        self.room_names = set()
        self.room_ids = []
        self.object_id_in_room_id_dict = {}
        self.room_id_object_id_dict = {}

        self.logger = get_logger(get_main_file_abs_path())

        self.add_nodes(data["nodes"])
        self.add_edges(data["edges"])
        # self._add_extra_edges()

    @property
    def full_nodes_list(self):
        if self._full_nodes_list is None:
            self._full_nodes_list = [
                self.get_full_node(i) for i in self.nodes(data=False)
            ]

        return self._full_nodes_list

    def _add_node(self, node: dict):
        self._class_name_count[node["class_name"]] += 1
        instance_name = (
            node["class_name"] + f"_{self._class_name_count[node['class_name']]}"
        )

        super().add_node(
            node["id"],
            category=node["category"],
            class_name=node["class_name"],
            instance_name=instance_name,
            prefab_name=node["prefab_name"],
            obj_transform=node["obj_transform"],
            bounding_box=node["bounding_box"],
            properties=node["properties"],
            states=node["states"],
        )

        if node["category"] == "Rooms":
            self.room_names.add(node["class_name"])
            self.room_ids.append(node["id"])
        elif node["category"] == "character":
            self.characters.append(node)
        else:
            # TODO: populate item-related dicts here if needed
            pass

        # self.id_object_name_dict[node["id"]] = node["class_name"]
        self.id_instance_name_dict[node["id"]] = instance_name
        self.object_name_ids_dict[node["class_name"]].append(node["id"])
        self.instance_name_id_dict[instance_name] = node["id"]

    def add_nodes(self, nodes: list):
        for d in nodes:
            self._add_node(d)

    def _add_edge(self, edge: dict):
        from_id = edge["from_id"]
        to_id = edge["to_id"]
        relation_type = edge["relation_type"]

        try:
            if any(
                link["relation_type"] == relation_type
                for link in self[from_id][to_id].values()
            ):
                return
        except KeyError:
            pass

        super().add_edge(
            from_id,
            to_id,
            relation_type=relation_type,
        )

        if relation_type == "INSIDE" and to_id in self.room_ids:
            self.object_id_in_room_id_dict[from_id] = to_id
            self.room_id_object_id_dict[to_id] = from_id

    def add_edges(self, edges: list):
        for e in edges:
            self._add_edge(e)

    def get_full_node(self, node_id: int):
        node = deepcopy(self.nodes[node_id])
        node["id"] = node_id
        return node

    def _get_nodes(self, node_ids: Union[int, List[int]]) -> list:
        if isinstance(node_ids, int):  # node id
            nodes = [self.get_full_node(node_id=node_ids)]
        elif isinstance(node_ids, list):  # a list of node id
            nodes = [self.get_full_node(node_id) for node_id in node_ids]
        else:
            raise ValueError("Type is not accepted: ", type(node_ids))
        return nodes

    # def _add_extra_edges(self):
    #     for node_i, node_j in combinations(self.nodes, 2):
    #         if self.has_edge(node_i["id"], node_j["id"]) or self.has_edge(
    #             node_j["id"], node_i["id"]
    #         ):
    #             continue

    #         pos_i = np.array(node_i["obj_transform"]["position"])
    #         pos_j = np.array(node_j["obj_transform"]["position"])

    #         if (
    #             DISTANCE_THRESH_MIN
    #             < np.linalg.norm(pos_i - pos_j)
    #             < DISTANCE_THRESH_MAX
    #         ):
    #             self._add_edge(
    #                 dict(
    #                     from_id=node_i["id"],
    #                     to_id=node_j["id"],
    #                     relation_type="CLOSE",
    #                 )
    #             )
    #             self._add_edge(
    #                 dict(
    #                     from_id=node_j["id"],
    #                     to_id=node_i["id"],
    #                     relation_type="CLOSE",
    #                 )
    #             )

    def filter_nodes(
        self,
        nodes: list = None,
        categories: Union[str, List[str]] = None,
        class_names: Union[str, List[str]] = None,
        properties: Union[str, List[str]] = None,
    ):
        if nodes is None:
            nodes = self.full_nodes_list

        if isinstance(categories, str):
            categories = [categories]
        if isinstance(class_names, str):
            class_names = [class_names]
        if isinstance(properties, str):
            properties = [properties]

        selected_nodes = []
        for node in nodes:
            if categories is not None and node["category"] not in categories:
                continue
            if class_names is not None and node["class_name"] not in class_names:
                continue
            if properties is not None:
                match = False
                for property in node["properties"]:
                    if property in properties:
                        match = True
                        break
                if not match:
                    continue
            selected_nodes.append(node)
        return selected_nodes  # id list

    def filter_edges(
        self,
        edges: list = None,
        relation_types: Union[str, List[str]] = None,
        from_ids: Union[int, List[int]] = None,
        to_ids: Union[int, List[int]] = None,
    ):
        if edges is None:
            edges = self.full_nodes_list

        if isinstance(relation_types, str):
            relation_types = [relation_types]
        if isinstance(from_ids, int):
            from_ids = [from_ids]
        if isinstance(to_ids, int):
            to_ids = [to_ids]

        selected_edges = []
        for edge in edges:
            if (
                relation_types is not None
                and edge["relation_type"] not in relation_types
            ):
                continue
            if from_ids is not None and edge["from_id"] not in from_ids:
                continue
            if to_ids is not None and edge["to_id"] not in to_ids:
                continue

            selected_edges.append(edge)

        return selected_edges

    def filter_nodes_cat_dict(self, nodes: list = None, categories=None):
        if nodes is None:
            nodes = self.full_nodes_list

        if categories is None:
            return {}

        if isinstance(categories, str):
            categories = [categories]

        selected_nodes = {cat: [] for cat in categories}
        for node in nodes:
            if node["category"] in categories:
                selected_nodes[node["category"]].append(node)

        return selected_nodes  # id list

    def successors(
        self,
        node: dict,
        relation_types: list = None,
        categories=None,
        class_names=None,
        properties=None,
        with_relation=False,
    ):
        successor_node_ids = super().successors(node["id"])
        if relation_types is None:
            selected_nodes = [self.get_full_node(id) for id in successor_node_ids]
        else:
            selected_nodes = []
            for id in successor_node_ids:
                relation = self.edges[node["id"], id]["relation_type"]
                if relation in relation_types:
                    selected_nodes.append(self.get_full_node(id))

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
        relation_types: list = None,
        categories: list = None,
        class_names: list = None,
        properties: list = None,
        with_relation: bool = False,
    ):
        predecessor_node_ids = super().predecessors(node["id"])
        if relation_types is None:
            selected_nodes = [self.get_full_node(id) for id in predecessor_node_ids]
        else:
            selected_nodes = []
            for id in predecessor_node_ids:
                relation = self.edges[node["id"], id]["relation_type"]
                if relation in relation_types:
                    selected_nodes.append(self.get_full_node(id))

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

    def check_relation(
        self, node_id1: int, node_id2: int, target_relations: Union[str, List[str]]
    ):
        if isinstance(target_relations, str):
            target_relations = [target_relations]

        all_relation_types = [
            link["relation_type"] for link in self[node_id1][node_id2].values()
        ]

        check_results = [
            target_relation in all_relation_types
            for target_relation in target_relations
        ]

        return check_results


class SceneGraphInside(VirtualHomeSceneGraph):
    def __init__(self, data):
        data["edges"] = self.filter_edges(data["edges"], relation_types="INSIDE")
        super().__init__(data)


class MemorySceneGraph(VirtualHomeSceneGraph):
    def __init__(self, data):
        self.objects_seen = {}
        # self.rooms_seen = {}
        self.items_seen = {}
        # self.containers_seen = {}
        # self.open_containers_seen = {}
        # self.close_containers_seen = {}

        self.rooms_explored = {}
        self.items_explored = {}
        # self.containers_explored = {}
        # self.open_containers_explored = {}
        # self.close_containers_explored = {}

        self.rooms_remaining = {}
        self.items_remaining = {}
        # self.containers_remaining = {}
        # self.open_containers_remaining = {}
        # self.close_containers_remaining = {}
        super().__init__(data)

    def _add_node(self, node: dict):
        self._class_name_count[node["class_name"]] += 1
        instance_name = (
            node["class_name"] + f"_{self._class_name_count[node['class_name']]}"
        )

        node_id = node["id"]

        super().add_node(
            node_id,
            category=node["category"],
            class_name=node["class_name"],
            instance_name=instance_name,
            prefab_name=node["prefab_name"],
            obj_transform=node["obj_transform"],
            bounding_box=node["bounding_box"],
            properties=node["properties"],
            states=node["states"],
        )

        if node["category"] == "Rooms":
            self.room_names.add(node["class_name"])
            self.room_ids.append(node_id)
        elif node["category"] == "character":
            self.characters.append(node)
        else:
            # TODO: populate item-related dicts here if needed
            pass

        # self.id_object_name_dict[node_id] = node["class_name"]
        self.id_instance_name_dict[node_id] = instance_name
        self.object_name_ids_dict[node["class_name"]].append(node_id)
        self.instance_name_id_dict[instance_name] = node_id

        self.objects_seen[node_id] = node
        if node["category"] == "Rooms":
            pass
            # self.rooms_seen[id] = node
        elif node["category"] != "Characters":
            self.items_seen[node_id] = node

            if node_id not in self.items_explored:
                self.items_remaining[node_id] = node

            # if "CONTAINERS" in node["properties"]:
            #     self.containers_seen[node_id] = node
            #     if node_id not in self.containers_explored:
            #         self.containers_remaining[node_id] = node
                # if "CAN_OPEN" in node["properties"]:
                #     self.close_containers_seen[id] = node
