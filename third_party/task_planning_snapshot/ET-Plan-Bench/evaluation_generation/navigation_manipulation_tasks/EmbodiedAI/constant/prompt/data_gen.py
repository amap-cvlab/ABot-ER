PROMPT_META_TASK = "Given the task: {task}\n\n"


PROMPT_ROOMS_ORDEDED_ABSTRACT = """# ROOM_LIST
{room_names}
# USER_NEED
{task}
# INSTRUCTION
Determine which room is most likely to contain what the user want in USER_NEED. Please re-order the room names in the ROOM_LIST by their probability to contain it.
Please do not output the answer like 'As an AI language model, I don not have the ability to physically determine the location of objects or bring them to you.'
Output your answer inside [], like [room_name1, room_name2, room_name3, room_name4].
The rooms most probable to contain what the user want in USER_NEED are: []
"""


PROMPT_ROOMS_ORDEDED_WITH_OBJECT_NAME = """# ROOM_LIST
{room_names}
# INSTRUCTION
Determine which room is most likely to contain {object_name}. Please re-order the room names in the ROOM_LIST by their probability to contain {object_name}.
Please do not output the answer like 'As an AI language model, I don not have the ability to physically determine the location of objects or bring them to you.'
Output your answer inside [], like [room_name1, room_name2, room_name3, room_name4].
The rooms most probable to contain {object_name} are: []
"""


PROMPT_ROOM_WITH_OBJECT_NAME = """# ROOM_LIST
{room_names}
# INSTRUCTION
Determine which room may contain {object_name}. Please only return ONE room name from the ROOM_LIST.
Please do not output the answer like 'As an AI language model, I don not have the ability to physically determine the location of objects or bring them to you.'
Output your answer inside [], like [room_name].
The room most probable to contain {object_name}: []
"""


# PROMPT_ANOTHER_ROOM_WITH_OBJECT_NAME = "After robot's exploration, The {object_name} seems not be inside the room: {explored_room_name_list}\n\nDetermine which room may contain the object {object_name}, and the room list is {remaining_room_names}. Please only return the room name from the room list.\n\nPlease do not output the answer like 'As an AI language model, I don not have the ability to physically determine the location of objects or bring them to you.'\n\n"


PROMPT_IF_HIDDEN_ORDERED = """# OBJECT_LIST
{blockers}
# INSTRUCTION
{object_name} might be hidden by some other objects. {object_name} might be most likely to be hidden or contained by which objects in the OBJECT_LIST?
From OBJECT_LIST, select only those objects that you think {object_name} might be likely inside or hidden behind, and order these objects by their probabilty to contain or to hide {object_name} (with the most likely object put in the front). Then, output them in a Python-style list, like [object1,object2]. Please output no more than 5 such objects. If you think none of the objects is likely to contain or to hide {object_name}, please output "None".
"""


PROMPT_IF_INSIDE_ORDERED = """# OBJECT_LIST
{containers}
# INSTRUCTION
{object_name} might be hidden by some larger objects. {object_name} is most likely to be hidden by which objects in the OBJECT_LIST?
From OBJECT_LIST, select only those objects that you think {object_name} is likely inside or hidden by, and order these objects by their probabilty to contain or to hide {object_name} (with the most likely object put in the front). Then, output them in a Python-style list, like [object1,object2]; if you think none of the objects is likely to contain or to hide {object_name}, please output "None".
"""


PROMPT_IF_INSIDE = """# OBJECT_LIST
{containers}
# INSTRUCTION
{object_name} might be hidden by some larger objects. {object_name} is most likely to be hidden by which object in the OBJECT_LIST?
If you think {object_name} is likely inside some objects or behind some objects close to it in OBJECT_LIST, please output these objects in a Python-style list, like [object1,object2]; if you think none of the objects is likely to contain or to hide {object_name}, please output None.
"""


PROMPT_IF_EXIST_ITEM_CORRECT = """# USER_NEED
{task}
# OBJECT_LIST
{items}
# INSTRUCTION
Is there any item(s) in the OBJECT_LIST fulfilling the need of the user in USER_NEED? Note that the item needs to DIRECTLY meet the user's need. For example, something likely to contain the target item is NOT acceptable. If yes, output the item(s) as a Python-style list with the first item being the most appropriate one; if not, output "None". Note: fruits are not dessert; do not mix up vegetables and fruits.
"""


PROMPT_IF_HIDDEN_ORDERED_ABSTRACT = """# OBJECT_LIST
{blockers}
# USER_NEED
{task}
# INSTRUCTION
What the user want might be hidden by some other objects. It might be most likely to be hidden or contained by which objects in the OBJECT_LIST?
From OBJECT_LIST, select only those objects that you think what the user want might be likely inside or hidden behind, and order these objects by their probabilty to contain or to hide it (with the most likely object put in the front). Then, output them in a Python-style list, like [object1,object2]. Please output no more than 5 such objects. If you think none of the objects is likely to contain or to hide what the user want, please output "None".
"""
