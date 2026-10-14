import re
import os
import sys
import copy

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from task_generation_codes.scene_graph import VirtualHomeSceneGraph, MemorySceneGraph


class TaskChecker:

    function_mapping = {
        "CLOSE": "is_close",
        "ON": "is_on",
        "INSIDE": "is_inside",
    }

    all_class_names = {
        "character",
        "apple",
        "remotecontrol",
        "amplifier",
        "bathtub",
        "wallshelf",
        "bathroom",
        "doorjamb",
        "painkillers",
        "slippers",
        "bedroom",
        "coffeetable",
        "kitchencounterdrawer",
        "plum",
        "alcohol",
        "bookshelf",
        "stovefan",
        "clock",
        "livingroom",
        "barsoap",
        "mug",
        "deodorant",
        "book",
        "bellpepper",
        "creamybuns",
        "condimentbottle",
        "pudding",
        "papertray",
        "pear",
        "photoframe",
        "washingsponge",
        "sundae",
        "clothesshirt",
        "juice",
        "dishwasher",
        "boardgame",
        "faucet",
        "microwave",
        "cpuscreen",
        "cookingpot",
        "chocolatesyrup",
        "candle",
        "mincedmeat",
        "computer",
        "paper",
        "fridge",
        "powersocket",
        "bottlewater",
        "crayons",
        "walllamp",
        "sofa",
        "condimentshaker",
        "cuttingboard",
        "coffeemaker",
        "chicken",
        "dishbowl",
        "milkshake",
        "tablelamp",
        "bathroomcounter",
        "kitchencounter",
        "tv",
        "speaker",
        "plate",
        "mouse",
        "printer",
        "coffeepot",
        "garbagecan",
        "lime",
        "facecream",
        "ceilingfan",
        "oventray",
        "ceiling",
        "bananas",
        "magazine",
        "kitchen",
        "cellphone",
        "guitar",
        "door",
        "kitchencabinet",
        "closet",
        "toothpaste",
        "waterglass",
        "breadslice",
        "window",
        "stall",
        "cutleryfork",
        "rug",
        "wineglass",
        "cutleryknife",
        "notes",
        "salmon",
        "fryingpan",
        "milk",
        "orchid",
        "wall",
        "washingmachine",
        "peach",
        "chair",
        "desk",
        "radio",
        "salad",
        "poundcake",
        "cutlets",
        "chips",
        "whippedcream",
        "curtains",
        "pancake",
        "knifeblock",
        "hairproduct",
        "toothbrush",
        "clothespile",
        "ceilinglamp",
        "closetdrawer",
        "mousemat",
        "keyboard",
        "nightstand",
        "candybar",
        "toy",
        "wine",
        "sink",
        "bed",
        "stove",
        "towelrack",
        "toaster",
        "cabinet",
        "wallpictureframe",
        "lightswitch",
        "clothespants",
        "towel",
        "floor",
        "box",
        "carrot",
        "wallphone",
        "bench",
        "cupcake",
        "folder",
        "toilet",
        "tvstand",
        "crackers",
        "bathroomcabinet",
        "kitchentable",
        "pie",
        "cereal",
        "dishwashingliquid",
        "toiletpaper",
        "hanger",
        "perfume",
        "pillow",
    }

    @staticmethod
    def parse_task_description(string: str):
        try:
            pattern = r"task:(.*?)\nthink"
            res = re.search(pattern, string, re.DOTALL)
            assert res is not None
            result = res.group(1).strip()
        except Exception as e:
            print(e)
            return None
        return result

    @staticmethod
    def parse_task_completion(string: str):
        try:
            string = copy.deepcopy(string)
            # 如果出现多个task completion criterion，只保留最后一个
            s = "task completion criterion"
            indexes = [substr.start() for substr in re.finditer(s, string)]
            assert len(indexes) > 0, f"length of matched indexes == 0!"
            string = string[indexes[-1] :]
            pattern = r"task completion criterion:(.*)"
            res = re.search(pattern, string)
            assert res is not None, "Fail to get completion criterion!"
            result = res.group(1).strip()
        except Exception as e:
            print(e)
            return None
        return result

    @staticmethod
    def parse_conditions(string: str, pass_preprocess=False):
        try:
            if not pass_preprocess:
                string = TaskChecker.parse_task_completion(string)
                assert string is not None, f"Get None from task completion"
            pattern = r"\((.*?)\)"
            results = re.findall(pattern, string)
            assert (
                len(results) > 0
            ), f"failed to find pattern: {pattern}, string: {string}"
            conditions = []
            for result in results:
                condition = list(map(lambda x: x.strip(), re.split(",", result)))
                condition = list(map(lambda x: x.split("_")[0], condition))
                condition = list(map(lambda x: x.replace(" ", ""), condition))
                condition = list(map(lambda x: re.sub(r"[0-9]+", "", x), condition))
                condition = list(map(lambda x: "character" if x == "robot" else x, condition))
                assert (
                    len(condition) == 3
                ), f"the length of condition is not 3! {condition}"
                assert (
                    condition[1] in TaskChecker.all_class_names
                    and condition[2] in TaskChecker.all_class_names
                ), f"obj not in valid list, obj1: {condition[1]}, obj2: {condition[2]}"
                conditions.append(condition)

            assert len(conditions) > 0, f"Failed to parse any condition!"
        except Exception as e:
            print(e)
            return None
        return conditions

    @staticmethod
    def is_success(scene_graph: VirtualHomeSceneGraph, conditions: list):

        matched_class_name_ids = {}
        for _, obj1_name, obj2_name in conditions:
            matched_class_name_ids[obj1_name] = []
            matched_class_name_ids[obj2_name] = []

        for condition in conditions:
            func_name = TaskChecker.function_mapping[condition[0]]
            obj1_name, obj2_name = condition[1], condition[2]

            # 如果某个object类名在之前的指令中出现过，只遍历此前匹配到的所有instance
            # 实现组合匹配
            obj1_arg = (
                obj1_name
                if len(matched_class_name_ids[obj1_name]) == 0
                else matched_class_name_ids[obj1_name]
            )
            obj2_arg = (
                obj2_name
                if len(matched_class_name_ids[obj2_name]) == 0
                else matched_class_name_ids[obj2_name]
            )
            _is_success, objects_list = getattr(scene_graph, func_name)(
                obj1_arg, obj2_arg
            )
            if not _is_success:
                # print(f"condition: {condition} failed !")
                return False

            obj1_ids = list(set([x[0]["id"] for x in objects_list]))
            obj2_ids = list(set([x[1]["id"] for x in objects_list]))
            matched_class_name_ids[obj1_name] = obj1_ids
            matched_class_name_ids[obj2_name] = obj2_ids

            # print(
            #     f"condition: {condition} success !!, matched obj1 ids number: {len(obj1_ids)}, matched obj2 ids number: {len(obj2_ids)}"
            # )
        return True


if __name__ == "__main__":
    from tasks.utils import load_json

    root_dir = os.path.dirname(os.path.dirname(__file__))
    data = load_json(os.path.join(root_dir, "tests/test_scene_graph.json"))
    graph = VirtualHomeSceneGraph(data)

    string = "task: Locate a radio inside the bookshelf.\nthink: The aim is for the robot to find a radio inside the bookshelf. The robot must be near the radio when the task is considered complete. The first criterion will be (CLOSE, character, radio). A specification of the location of the radio is that it's inside the bookshelf, hence, the second criterion is (INSIDE, radio, bookshelf). Additionally, since the bookshelf is close to the towelrack, and assuming the robot goes to the bookshelf, we should acknowledge that it would also be close to the towelrack when the task is complete. Therefore, the third criterion is (CLOSE, character, towelrack).\ntask completion criterion: (CLOSE, character, radio) (INSIDE, radio, bookshelf) (CLOSE, character, towelrack)"

    conditions = TaskChecker.parse_conditions(string)
    result = TaskChecker.is_success(graph, conditions)
    print(f"result: {result}")
