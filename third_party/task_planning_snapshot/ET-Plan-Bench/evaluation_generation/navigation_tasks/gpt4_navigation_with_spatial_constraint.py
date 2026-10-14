import os
import sys
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import json
import pickle as pkl
from collections import defaultdict 
import random
import requests
from tasks.utils import auto_kill_unity
from envs.env import VirtualHomeNavigationEnv

from evaluation_generation.navigation_tasks.gpt4_navigation import robot_exploration_without_constraint

from openai import OpenAI
import numpy as np

key_file = open("GPT_API_KEY.txt", "r")
api_key_string = key_file.read()

client=OpenAI(api_key=api_key_string)

import time

def retry_fn(fn, max_failures=10, sleep_time=5):
    """A function to handle the interruption issue of GPT API.

    Args:
        fn (function): function.
        max_failures (int, optional): max number of tries. Defaults to 10.
        sleep_time (int, optional): sleep time after each failure. Defaults to 5.

    Raises:
        Exception: max failures are exeeded.

    Returns:
        None
    """
    failures = 0
    while failures < max_failures:
        try:
            return fn()
        except KeyboardInterrupt:
            print('Interrupted')
            sys.exit(0)
        except Exception as e:
            failures += 1
            print('Failed with exception:')
            print(e)
            print(f'Failed {failures} times, waiting {sleep_time}s to retry')
            time.sleep(sleep_time)
            if failures >= max_failures:
                raise Exception('Max failures exceeded.')
            time.sleep(2)

def gpt4v_func(prompt, model="gpt-4o", max_tokens=2048, user_role=True):
    """Function to call GPT.

    Args:
        prompt (str): prompt for GPT.
        model (str, optional): GPT model to use. Defaults to "gpt-4o".
        max_tokens (int, optional): maximum token length. Defaults to 2048.
        user_role (bool, optional): whether to assign the role to be user. Defaults to True.
    """
    def f(prompt, user_role=True):
        if user_role:
            message = [{'role': 'user', 'content': prompt}]
        else:
            message = prompt
        response = client.chat.completions.create(
            model = model,
            messages = message,
            max_tokens = max_tokens,
        )
        return response

    response = retry_fn(lambda: f(prompt, user_role), max_failures=5, sleep_time=5)

    return response

def robot_exploration(task, object_1, object_2, port_num, env, save_dir, task_id, instant_id_1, instant_id_2, two_objects_relation, environment_id, simple_tasks_file):
    """Main function for exploration with spatial constraints

    Args:
        task (str): task description
        object_1 (str): object_1
        object_2 (str): object_2
        port_num (int): port number
        env (object): VirtualHomeNavigationEnv
        save_dir (str): save directory
        task_id (int): task id
        instant_id_1 (int): ID of instance_1
        instant_id_2 (int): ID of instance_2
        two_objects_relation (str): relation between two objects
        environment_id (int): environment ID
        simple_tasks_file (str): task json file path
    """

    constraint = f"({two_objects_relation}, {object_1}, {object_2})"

    def distance_from_agent(graph, large_object_ids, char_id):
        agent_location = [item['obj_transform']['position'] for item in graph['nodes'] if item['id'] == char_id[0]][0]
        id_distance_dict = {}
        for large_object_id in large_object_ids:
            large_object_location = [item['obj_transform']['position'] for item in graph['nodes'] if item['id'] == large_object_id][0]
            id_distance_dict[large_object_id] = np.linalg.norm(np.array(agent_location)-np.array(large_object_location))
        return sorted(id_distance_dict, key=id_distance_dict.get)

    MAX_STEPS = 20

    prompt_0 = f"Given the task: {task}\n\n"

    def calculate_2D_distance(init_position, position):
        init_position = np.array(init_position)
        position = np.array(position)
        return np.linalg.norm(init_position-position)

    def LLM_determine_whether_object_inside_somewhere(gpt4v_find_the_room, object_1, object_2, visible_objects, visible_objects_dict, success_tag, steps_num, previous_generated_steps, to_save, two_objects_relation, instant_id_1, instant_id_2, constraint, whole_graph_edges_list, distance, char_init_position):
        """Use LLM to determine is the target object is in visible object or in the object that needs to be explored further.

        Args:
            gpt4v_find_the_room (str): room to explore
            object_1 (str): object_1
            object_2 (str): object_2
            visible_objects (list[str]): list of visible objects
            visible_objects_dict (dict): dictionary of visible objects indexed by object ID
            success_tag (bool): success tag
            steps_num (int): step number
            previous_generated_steps (list[str]): previously generated steps
            to_save (dict): all info to save
            two_objects_relation (str): relation between two objects
            instant_id_1 (int): ID of instance_1
            instant_id_2 (int): ID of instance_2
            constraint (str): constraint
            whole_graph_edges_list (list): list of whole graph edges
            distance (float): distance from the starting point of robot
            char_init_position (list[float]): coordinates of robot starting point
        """
        if(object_1 in visible_objects and object_2 in visible_objects and constraint in whole_graph_edges_list):
            if(instant_id_1 in visible_objects_dict.keys()):
                find_object_action = f'<char0> [walk] <{object_1}> ({instant_id_1})'
            else:
                the_target_instance_id_list = [the_key for the_key in surround_visible_objects_dict.keys() if surround_visible_objects_dict[the_key] == object_1]
                if(len(the_target_instance_id_list) == 1):
                    the_target_instance_id = the_target_instance_id_list[0]
                else:
                    find_id_tag = False
                    for the_object_id_item in the_target_instance_id_list:
                        if(find_id_tag == True):
                            break
                        for id_object_pair in id_object_edges_list:
                            if(str(the_object_id_item) in id_object_pair and object_2 in id_object_pair):
                                find_id_tag = True
                                the_target_instance_id = the_object_id_item
                                break
                find_object_action = f'<char0> [walk] <{object_1}> ({the_target_instance_id})'

            obs, success, info = env_step(find_object_action, env, steps_num)

            if(success == True):
                to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':find_object_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                print("STEP_"+str(steps_num))
                steps_num += 1

                char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                distance += calculate_2D_distance(char_init_position, char_position)
                char_init_position = char_position

                previous_generated_steps.append(find_object_action)
                print("Find object!")
                success_tag = True
                return True, success_tag, None, steps_num, previous_generated_steps, to_save, distance, char_init_position
            else:
                print("Error:", find_object_action)
                return None, None, None, steps_num, None, to_save, None, None
            
        elif(object_1 in visible_objects):
            gpt4v_response = f"['{object_1}']" # and all visible keys
            return gpt4v_response, success_tag, None, steps_num, previous_generated_steps, to_save, distance, char_init_position
        elif(object_2 in visible_objects):
            gpt4v_response = f"['{object_2}']" # and all visible keys
            return gpt4v_response, success_tag, None, steps_num, previous_generated_steps, to_save, distance, char_init_position
        else:
            prompt_2 = f"Currently, the robot is in the room {gpt4v_find_the_room} and can see objects {visible_objects}. Is this possible the object {object_1} and the object {object_2} are located inside or obscured by one of these visible objects? If yes, please only output these possbile visible objects in a Python-style list, if not, please only output None.\n\nPlease do not output the thinking process.\n\n"
            prompt_for_gpt4v = prompt_0 + prompt_2

            gpt4v_response = gpt4v_func(prompt_for_gpt4v)
            
            return gpt4v_response, success_tag, prompt_for_gpt4v, steps_num, previous_generated_steps, to_save, distance, char_init_position

    save_path = os.path.join(save_dir, f"task_{task_id}.json")

    image_save_path = os.path.join(save_dir, 'RGB')

    if(not os.path.exists(image_save_path)):
        os.mkdir(image_save_path)

    image_save_path_for_task = os.path.join(image_save_path, str(task_id))

    if(not os.path.exists(image_save_path_for_task)):
        os.mkdir(image_save_path_for_task)

    def env_step(action, env, steps_num):
        """Interaction with VirtualHome environment

        Args:
            action (str): VirtualHome action
            env (object): VirtualHomeNavigationEnv
            steps_num (int): step number

        Returns:
            obs (dict): observation
            success (bool): whether the task has been finished successfully
            info (dict): other info
        """
        # Wrapper with env
        if(steps_num == None):
            obs, reward, done, truncated, info = env.step_CA(action, save_img_path=None, recording=False)
        else:
            save_img_path = os.path.join(image_save_path_for_task, str(steps_num)+'.png')
            obs, reward, done, truncated, info = env.step_CA(action, save_img_path, recording=False)
        if info['msg'] is None:
            success = True
        else:
            success = False
        return obs, success, info
    
    def get_visible_objects(obs):
        first_person_view_visible_graph = obs['visible_graph']
        visible_objects_dict = {}
        visible_objects = []
        for the_item in first_person_view_visible_graph['nodes']:
            visible_objects_dict.update({the_item['id']:the_item['class_name']})
            visible_objects.append(the_item['class_name'])

        visible_objects = list(set(visible_objects))
        return visible_objects_dict, visible_objects

    def get_surround_visible_objects(obs):

        visible_objects_dict, visible_objects = get_visible_objects(obs)

        surround_visible_objects_dict = {}
        surround_visible_objects = []

        surround_visible_objects_dict.update(visible_objects_dict)
        surround_visible_objects += visible_objects

        whole_graph_edges_list = []
        id_object_edges_list = []

        whole_graph_edges = obs['visible_graph']['edges']
        for item in whole_graph_edges:
            whole_graph_edges_list.append("("+item['relation_type']+", "+surround_visible_objects_dict[item['from_id']]+", "+surround_visible_objects_dict[item['to_id']]+")")
            id_object_edges_list.append("("+item['relation_type']+", "+str(item['from_id'])+", "+surround_visible_objects_dict[item['to_id']]+")")

        for surround_times in range(11): # 360 degrees rotation in total
            exe_action_list = ['<char0> [turnright]'] # 30 degrees for each

            obs, success, info = env_step(exe_action_list, env, steps_num=None)

            if(success == True):
                visible_objects_dict, visible_objects = get_visible_objects(obs)

                surround_visible_objects_dict.update(visible_objects_dict)
                surround_visible_objects += visible_objects

                whole_graph_edges = obs['visible_graph']['edges']
                for item in whole_graph_edges:
                    whole_graph_edges_list.append("("+item['relation_type']+", "+surround_visible_objects_dict[item['from_id']]+", "+surround_visible_objects_dict[item['to_id']]+")")
                    id_object_edges_list.append("("+item['relation_type']+", "+str(item['from_id'])+", "+surround_visible_objects_dict[item['to_id']]+")")
            else:
                print("Turn Right Error")

        surround_visible_objects = list(set(surround_visible_objects))

        print("surround_visible_objects:", surround_visible_objects)

        whole_graph_edges_list = list(set(whole_graph_edges_list))
        id_object_edges_list = list(set(id_object_edges_list))

        return surround_visible_objects_dict, surround_visible_objects, whole_graph_edges_list, id_object_edges_list

    init_room_pool = ['bathroom', 'bedroom', 'kitchen', 'livingroom']
    random_init_room = random.choice(init_room_pool)
    print("port before try except:", port_num)

    try:
        obs = env.reset_CA(env_id=environment_id,init_room=random_init_room)
    except:
        print('Restarting the env')
        print("port inside except:", port_num)
        port_num += 1
        env = VirtualHomeNavigationEnv(port=str(port_num), input_data_file=simple_tasks_file)
        obs = env.reset_CA(env_id=environment_id,init_room=random_init_room)

    graph = obs['full_graph']

    char_id = [item['id'] for item in graph['nodes'] if item['category'] == 'Characters']

    char_info = [item for item in graph['nodes'] if item['category'] == 'Characters'][0]
    char_init_position = char_info['obj_transform']['position']

    to_save = {'inputs':task, 'object_1':object_1, 'object_2':object_2, 'instant_id_1':instant_id_1, 'instant_id_2':instant_id_2, 'two_objects_relation':two_objects_relation, 'environment_id':environment_id, 'initial_room':random_init_room, 'initial_position':char_init_position, 'distance':0, 'steps':{}}
    distance = 0
    steps_num = 0
    
    success_tag = False
    
    all_rooms_pair = [[item['class_name'],item['id']] for item in graph['nodes'] if item['category'] == 'Rooms']
    all_rooms = []
    all_rooms_dict = {}
    for item in all_rooms_pair:
        all_rooms.append(item[0])
        all_rooms_dict.update({item[1]:item[0]})

    print("all_rooms:", all_rooms)
    print("all_rooms_dict:", all_rooms_dict)
     
    prompt_1 = f"Determine which room may contain the object {object_1} and the object {object_2} in the meantime, and the room list is {all_rooms}. Please ranking these rooms based on the possibility and only output a Python-style list.\n\nThe number of output rooms should be the same as the number of rooms in the original room list.\n\nPlease do not output the answer like 'As an AI language model, I don not have the ability to physically determine the location of objects or bring them to you.'\n\nPlease do not output the thinking process.\n\n"
    
    prompt_for_gpt4v = prompt_0 + prompt_1

    find_room_success_tag = False

    max_num = 0

    while(find_room_success_tag == False):

        if(max_num >=5):
            print("Maximum finding room retry")

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env
        max_num += 1

        gpt4v_response = gpt4v_func(prompt_for_gpt4v)

        print("gpt4v_response:", gpt4v_response) # can be used to check whether the gpt4v key is invalid
        
        ranked_room_list_origin = gpt4v_response.choices[0].message.content
        ranked_room_list = ranked_room_list_origin
        
        if(ranked_room_list[:2] == "['" and ranked_room_list[-2:] == "']"): # valid output
            ranked_room_list = ranked_room_list[2:-2]
        elif(ranked_room_list[:2] == '["' and ranked_room_list[-2:] == '"]'):
            ranked_room_list = ranked_room_list[2:-2]
        elif("['" in ranked_room_list and "']" in ranked_room_list):
            ranked_room_list = ranked_room_list.split("['")[1].split("']")[0]
        elif('["' in ranked_room_list and '"]' in ranked_room_list):
            ranked_room_list = ranked_room_list.split('["')[1].split('"]')[0]
        else:
            print("Error GPT4V format find rooms")

        if("', '" in ranked_room_list):
            ranked_room_list = ranked_room_list.split("', '")
        else:
            ranked_room_list = ranked_room_list.split('", "')

        try:
            if(len(ranked_room_list) == len(all_rooms)):
                find_room_success_tag = True
                pass
            else:
                print("Error GPT4V format find rooms")
        except:
            print("Error GPT4V format find rooms")

    previous_generated_steps = []
    
    print("ranked_room_list:", ranked_room_list)

    for gpt4v_find_the_room in ranked_room_list:

        if(gpt4v_find_the_room not in all_rooms):
            print("Error room name:", gpt4v_find_the_room)

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env

        print("all_rooms_dict:", all_rooms_dict)
        print("gpt4v_find_the_room:", gpt4v_find_the_room)
        explore_room_id = [key for key in all_rooms_dict if all_rooms_dict[key] == gpt4v_find_the_room]
        print("explore_room_id:", explore_room_id)

        if(len(explore_room_id) == 0):
            print("Error room name:", gpt4v_find_the_room)

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env
        
        if(len(explore_room_id) == 1):
            explore_room_id = explore_room_id[0]
        else:
            explore_room_id = distance_from_agent(obs['full_graph'], explore_room_id, char_id)
            print("explore_room_id:", explore_room_id)
            explore_room_id = explore_room_id[0]

        all_rooms_dict.pop(explore_room_id)

        change_room_action = f'<char0> [walk] <{gpt4v_find_the_room}> ({explore_room_id})'

        obs, success, info = env_step(change_room_action, env, steps_num)
        
        if(success == True):
  
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': prompt_for_gpt4v, 'gpt_response':ranked_room_list_origin, 'action':change_room_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
            distance += calculate_2D_distance(char_init_position, char_position)
            char_init_position = char_position

            if(steps_num > MAX_STEPS): # exceed the maximum steps

                to_save['distance'] = distance
                with open(save_path, "w") as outfile:
                    json.dump(to_save, outfile)

                return None, None, port_num, env
            
            previous_generated_steps.append(change_room_action)
        else:
            print("Failed VirtualHome action:", change_room_action)
            print("message:", info['msg'])

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env

        surround_visible_objects_dict, surround_visible_objects, whole_graph_edges_list, id_object_edges_list = get_surround_visible_objects(obs)

        LLM_decision, success_tag, select_large_objects_prompt, steps_num, previous_generated_steps, to_save, distance, char_init_position = LLM_determine_whether_object_inside_somewhere(gpt4v_find_the_room, object_1, object_2, surround_visible_objects, surround_visible_objects_dict, success_tag, steps_num, previous_generated_steps, to_save, two_objects_relation, instant_id_1, instant_id_2, constraint, whole_graph_edges_list, distance, char_init_position)

        if(steps_num > MAX_STEPS): # exceed the maximum steps

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env
        
        if(LLM_decision == None and success_tag == None and select_large_objects_prompt == None):

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env
            
        if(str(LLM_decision).lower() == 'true' or LLM_decision == True): # Find object
            break
        else:
            try:
                LLM_decision = LLM_decision.choices[0].message.content
            except:
                LLM_decision = LLM_decision
        
        print("LLM_decision:", LLM_decision)
        if(LLM_decision == 'None'):
            pass
        else: # a list of large object
            LLM_decision_list = []
            if(LLM_decision[:2] == "['" and LLM_decision[-2:] == "']"): # valid output
                LLM_decision = LLM_decision[2:-2]
            elif(LLM_decision[:2] == '["' and LLM_decision[-2:] == '"]'):
                LLM_decision = LLM_decision[2:-2]
            elif("['" in LLM_decision and "']" in LLM_decision):
                LLM_decision = LLM_decision.split("['")[1].split("']")[0]
            elif('["' in LLM_decision and '"]' in LLM_decision):
                LLM_decision = LLM_decision.split('["')[1].split('"]')[0]
            else:
                print("Error GPT4V format large objects")
                LLM_decision = []
                
            if(LLM_decision == []):
                pass
            else:
                if("', '" in LLM_decision):
                    LLM_decision = LLM_decision.split("', '")
                else:
                    LLM_decision = LLM_decision.split('", "')
                print("processed LLM_decision:", LLM_decision)
            
            for large_object in LLM_decision:
                if large_object in surround_visible_objects and large_object not in all_rooms: # valid large object name
                    large_object_ids = [key for key, name in surround_visible_objects_dict.items() if name == large_object]
                    while(large_object_ids != []):
                        graph = obs['full_graph']
                        large_object_ids = distance_from_agent(graph, large_object_ids, char_id)
                        instance_level_id = large_object_ids[0]
                        large_object_ids = large_object_ids[1:]

                        explore_large_object_step = f"<char0> [walk] <{large_object}> ({instance_level_id})"
                    
                        obs, success, info = env_step(explore_large_object_step, env, steps_num)
                        
                        if(success == True):
                            if(select_large_objects_prompt == None):
                                to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':explore_large_object_step, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                            else:
                                to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': select_large_objects_prompt, 'gpt_response':LLM_decision, 'action':explore_large_object_step, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})

                            print("STEP_"+str(steps_num))
                            steps_num += 1

                            char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                            distance += calculate_2D_distance(char_init_position, char_position)
                            char_init_position = char_position

                            if(steps_num > MAX_STEPS): # exceed the maximum steps

                                to_save['distance'] = distance
                                with open(save_path, "w") as outfile:
                                    json.dump(to_save, outfile)

                                return None, None, port_num, env
            
                            previous_generated_steps.append(explore_large_object_step)

                            if(select_large_objects_prompt == None and large_object == object_1): # object_1没必要打开
                                continue
                        
                            open_large_object_step = f"<char0> [open] <{large_object}> ({instance_level_id})"
                            close_large_object_step = f"<char0> [close] <{large_object}> ({instance_level_id})"
                            
                            obs, success, info = env_step(open_large_object_step, env, steps_num)

                            need_close_tag = False
                            if(success == True):
                                to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':open_large_object_step, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                                print("STEP_"+str(steps_num))
                                steps_num += 1

                                if(steps_num > MAX_STEPS): # exceed the maximum steps

                                    to_save['distance'] = distance
                                    with open(save_path, "w") as outfile:
                                        json.dump(to_save, outfile)

                                    return None, None, port_num, env
            
                                previous_generated_steps.append(open_large_object_step)
                                need_close_tag = True

                            else:
                                print("Error:", explore_large_object_step)
                                # Cannot open the container, it is fine

                            surround_visible_objects_dict, surround_visible_objects, whole_graph_edges_list, id_object_edges_list = get_surround_visible_objects(obs)
                            
                            if(object_1 in surround_visible_objects_dict.values() and object_2 in surround_visible_objects_dict.values() and constraint in whole_graph_edges_list):
                                if(instant_id_1 in surround_visible_objects_dict.keys()):
                                    find_object_action = f'<char0> [walk] <{object_1}> ({instant_id_1})'
                                else:
                                    the_target_instance_id_list = [the_key for the_key in surround_visible_objects_dict.keys() if surround_visible_objects_dict[the_key] == object_1]
                                    if(len(the_target_instance_id_list) == 1):
                                        the_target_instance_id = the_target_instance_id_list[0]
                                    else:
                                        find_id_tag = False
                                        for the_object_id_item in the_target_instance_id_list:
                                            if(find_id_tag == True):
                                                break
                                            for id_object_pair in id_object_edges_list:
                                                if(str(the_object_id_item) in id_object_pair and object_2 in id_object_pair):
                                                    find_id_tag = True
                                                    the_target_instance_id = the_object_id_item
                                                    break
                                    find_object_action = f'<char0> [walk] <{object_1}> ({the_target_instance_id})'

                                obs, success, info = env_step(find_object_action, env, steps_num)

                                if(success == True): # Find object
                                    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':find_object_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                                    print("STEP_"+str(steps_num))
                                    steps_num += 1

                                    char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                                    distance += calculate_2D_distance(char_init_position, char_position)
                                    char_init_position = char_position

                                    if(steps_num > MAX_STEPS): # exceed the maximum steps

                                        to_save['distance'] = distance
                                        with open(save_path, "w") as outfile:
                                            json.dump(to_save, outfile)

                                        return None, None, port_num, env
            
                                    previous_generated_steps.append(find_object_action)
                                    print("Find object!")

                                    success_tag = True

                                    LLM_decision = True

                                    break
                                else:
                                    print("Error:", find_object_action)

                                    to_save['distance'] = distance
                                    with open(save_path, "w") as outfile:
                                        json.dump(to_save, outfile)

                                    return None, None, port_num, env

                            if(need_close_tag == True):
                                obs, success, info = env_step(close_large_object_step, env, steps_num)
                                if(success == True):
                                    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':close_large_object_step, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                                    print("STEP_"+str(steps_num))
                                    steps_num += 1

                                    if(steps_num > MAX_STEPS): # exceed the maximum steps

                                        to_save['distance'] = distance
                                        with open(save_path, "w") as outfile:
                                            json.dump(to_save, outfile)

                                        return None, None, port_num, env
            
                                    previous_generated_steps.append(close_large_object_step)
                                
                        else:
                            print("Error:", explore_large_object_step)
                            # Cannot walk to the container, it is fine

                    if(success_tag == True and LLM_decision == True): # Find object!
                        break
                else:
                    print("invalid large object name")

        if(LLM_decision == True):
            break
    
    print("previous_generated_steps:", previous_generated_steps)
    print("success_tag:", success_tag)

    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':[], 'action_history': previous_generated_steps.copy(), 'msg': success_tag}}) # save final status
    print("STEP_"+str(steps_num))
    steps_num += 1

    if(steps_num > MAX_STEPS): # exceed the maximum steps

        to_save['distance'] = distance
        with open(save_path, "w") as outfile:
            json.dump(to_save, outfile)

        return None, None, port_num, env

    to_save['distance'] = distance
    with open(save_path, "w") as outfile:
        json.dump(to_save, outfile)

    return previous_generated_steps, success_tag, port_num, env

@auto_kill_unity(kill_before_return=True)
def for_loop_data_generation():
    """Loop through all tasks to generate data.
    """
    port_num = 3616
    
    generated_tasks_path_folder = '../../generated_tasks'
    simple_tasks_file = os.path.join(generated_tasks_path_folder, 'navigation_tasks_with_spatial_constraint.json')
    testing_environment_ids = [37, 0, 32, 39, 19, 20, 48, 49, 17, 26]

    save_dir = '../../Output/nav/visible/navigation_tasks_with_spatial_constraint'
    os.makedirs(save_dir, exist_ok=True)
    
    with open(simple_tasks_file, 'r') as json_file:
        json_list = list(json_file)
    
    env = VirtualHomeNavigationEnv(port=str(port_num), input_data_file=simple_tasks_file)
    print("port after VirtualHomeNavigationEnv:", port_num)
    
    for task_id, json_str in enumerate(json_list):

        save_path = os.path.join(save_dir, f"task_{task_id}.json")

        if(task_id < 0):
            continue

        if(os.path.exists(save_path)):
            continue
        
        result = json.loads(json_str)
        print("result:", result)

        object_1 = result['object_1']
        object_2 = result['object_2']
        task = result['task']
        if('\n' in task and task[-1] != '\n'):
            task = task.split('\n')[-1]
        print("object_1:", object_1)
        print("object_2:", object_2)
        print("task:", task)
        environment_id = int(result['env_id'])
        instant_id_1 = int(result['instant_id_1'])
        instant_id_2 = int(result['instant_id_2'])
        print("instant_id_1:", instant_id_1)
        print("instant_id_2:", instant_id_2)
        two_objects_relation = result['task_completion_criterion'].split(')(')[-1].split(',')[0]
        print("two_objects_relation:", two_objects_relation)

        if(environment_id == 9):
            continue

        if(environment_id not in testing_environment_ids):
            continue
        
        all_room_list = ['bathroom', 'livingroom', 'kitchen', 'bedroom']
        if(object_2 in all_room_list):
            only_explore_one_room = [object_2]
            previous_generated_steps, success_tag, port_num, env = robot_exploration_without_constraint(task, object_1, port_num, env, save_dir, task_id, environment_id, simple_tasks_file, only_explore_one_room)
        else:
            previous_generated_steps, success_tag, port_num, env = robot_exploration(task, object_1, object_2, port_num, env, save_dir, task_id, instant_id_1, instant_id_2, two_objects_relation, environment_id, simple_tasks_file)

if __name__ == "__main__":
    for_loop_data_generation()