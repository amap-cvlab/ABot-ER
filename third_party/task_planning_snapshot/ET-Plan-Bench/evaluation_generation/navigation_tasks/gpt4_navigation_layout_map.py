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
            # temperature = temperature,
            max_tokens = max_tokens,
        )
        # response = response.choices[0].message.content
        return response

    response = retry_fn(lambda: f(prompt, user_role), max_failures=5, sleep_time=5)

    return response

def robot_exploration_without_constraint(task, object, port_num, env, save_dir, task_id, environment_id, simple_tasks_file, only_explore_one_room=None):
    """Main function for navigation to object without constraint

    Args:
        task (str): task description
        object (str): target object
        port_num (int): port number
        env (object): VirtualHomeNavigationEnv
        save_dir (str): save directory
        task_id (int): task id
        environment_id (int): environment id
        simple_tasks_file (str): task json file path
        only_explore_one_room (list, optional): the room to explore if not None. Defaults to None.
    """

    def distance_from_agent(graph, large_object_ids, char_id):
        agent_location = [item['obj_transform']['position'] for item in graph['nodes'] if item['id'] == char_id[0]][0]
        # agent_location = [agent_location[0], agent_location[1]]
        id_distance_dict = {}
        for large_object_id in large_object_ids:
            large_object_location = [item['obj_transform']['position'] for item in graph['nodes'] if item['id'] == large_object_id][0]
            # large_object_location = [large_object_location[0], large_object_location[1]]
            id_distance_dict[large_object_id] = np.linalg.norm(np.array(agent_location)-np.array(large_object_location))
        # print("id_distance_dict:", id_distance_dict)
        return sorted(id_distance_dict, key=id_distance_dict.get)
    
    def get_large_objects_from_room(full_graph):
        large_item_list = []
        large_item_dict = {}

        for node in full_graph['nodes']:
            if('LIEABLE' in node['properties']):
                large_item_list.append(node['class_name'])
                large_item_dict.update({node['id']:node['class_name']})
            elif('MOVEABLE' not in node['properties'] and 'GRABBABLE' not in node['properties']):
                large_item_list.append(node['class_name'])
                large_item_dict.update({node['id']:node['class_name']})
        
        large_item_list = list(set(large_item_list))
        return large_item_list, large_item_dict

    MAX_STEPS = 20

    prompt_0 = f"Given the task: {task}\n\n"

    def calculate_2D_distance(init_position, position):
        init_position = np.array(init_position)
        position = np.array(position)
        # return np.linalg.norm(init_position[:2]-position[:2])
        return np.linalg.norm(init_position-position)

    def LLM_determine_whether_object_inside_somewhere(gpt4v_find_the_room, object, visible_objects, visible_objects_dict, success_tag, steps_num, previous_generated_steps, to_save, distance, char_init_position):
        """Use LLM to determine is the target object is in visible object or in the object that needs to be explored further.

        Args:
            gpt4v_find_the_room (str): room to explore
            object (str): target object
            visible_objects (list[str]): list of visible objects
            visible_objects_dict (dict): dictionary of visible objects indexed by object ID
            success_tag (bool): success tag
            steps_num (int): step number 
            previous_generated_steps (list[str]): previously generated steps
            to_save (dict): all info to save
            distance (float): distance from the starting point of robot
            char_init_position (list[float]): coordinates of robot starting point
        """
        if(object in visible_objects):
            object_ids = [key for key, name in visible_objects_dict.items() if name == object]
            find_object_action = f'<char0> [walk] <{object}> ({object_ids[0]})'

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
        else:
            prompt_2 = f"Currently, the robot is in the room {gpt4v_find_the_room} and can see objects {visible_objects}. Since the target object could be obscured by some larger visible objects, the robot needs to explore the room further. Is this possible the object {object} is located inside or nearby one of these visible objects? If yes, please only output these possbile visible objects in a Python-style list and in the order of possibility, if not, please only output None.\n\nPlease do not output the thinking process.\n\n"
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
            obs, reward, done, truncated, info = env.step_CA(action, save_img_path=None, recording=False)
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

    def get_surround_visible_objects(obs, info, to_save, steps_num):

        visible_objects_dict, visible_objects = get_visible_objects(obs)

        surround_visible_objects_dict = {}
        surround_visible_objects = []

        surround_visible_objects_dict.update(visible_objects_dict)
        surround_visible_objects += visible_objects

        to_save['steps'].update({"STEP_"+str(steps_num)+"_0":{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':"'<char0> [turnright]'", 'action_history': '', 'msg': info['msg']}})
        
        for surround_times in range(11): # 360 degrees rotation in total
            exe_action_list = ['<char0> [turnright]'] # 30 degrees for each

            obs, success, info = env_step(exe_action_list, env, steps_num=str(steps_num)+"_"+str(surround_times+1))

            if(success == True):
                visible_objects_dict, visible_objects = get_visible_objects(obs)

                surround_visible_objects_dict.update(visible_objects_dict)
                surround_visible_objects += visible_objects

                to_save['steps'].update({"STEP_"+str(steps_num)+"_"+str(surround_times+1):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':"'<char0> [turnright]'", 'action_history': '', 'msg': info['msg']}})

            else:
                print("Turn Right Error")

        surround_visible_objects = list(set(surround_visible_objects))

        print("surround_visible_objects:", surround_visible_objects)

        return surround_visible_objects_dict, surround_visible_objects

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

    char_info = [item for item in graph['nodes'] if item['category'] == 'Characters'][0]
    char_init_position = char_info['obj_transform']['position']

    to_save = {'inputs':task, 'object_name':object, 'environment_id':environment_id, 'initial_room':random_init_room, 'initial_position':char_init_position, 'distance':0, 'steps':{}}
    distance = 0
    steps_num = 0

    previous_generated_steps = []

    large_item_list, large_item_dict = get_large_objects_from_room(graph)

    if(object in large_item_list):

        target_object_id = [key for key in large_item_dict.keys() if large_item_dict[key] == object][0]
        find_target_action = f'<char0> [walk] <{object}> ({target_object_id})'

        obs, success, info = env_step(find_target_action, env, steps_num)
        
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':find_target_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
            distance += calculate_2D_distance(char_init_position, char_position)
            char_init_position = char_position

            success_tag = True
            
            previous_generated_steps.append(find_target_action)
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':[], 'action_history': previous_generated_steps.copy(), 'msg': success_tag}}) # save final status

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return previous_generated_steps, success_tag, port_num, env

        else:
            print("Failed VirtualHome action:", find_target_action)
            print("message:", info['msg'])

            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env

    char_id = [item['id'] for item in graph['nodes'] if item['category'] == 'Characters']

    success_tag = False
    
    all_rooms_pair = [[item['class_name'],item['id']] for item in graph['nodes'] if item['category'] == 'Rooms']
    all_rooms = []
    all_rooms_dict = {}
    for item in all_rooms_pair:
        all_rooms.append(item[0])
        all_rooms_dict.update({item[1]:item[0]})

    print("all_rooms:", all_rooms)
    print("all_rooms_dict:", all_rooms_dict)
     
    prompt_1 = f"Determine which room may contain the object {object}, and the room list is {all_rooms}. Please ranking these rooms based on the possibility and only output a Python-style list.\n\nThe number of output rooms should be the same as the number of rooms in the original room list.\n\nPlease do not output the answer like 'As an AI language model, I don not have the ability to physically determine the location of objects or bring them to you.'\n\nPlease do not output the thinking process.\n\n"
    
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
        
        if(ranked_room_list[0] == "['" and ranked_room_list[-1] == "']"): # valid output
            ranked_room_list = ranked_room_list[2:-2]
        elif(ranked_room_list[0] == '["' and ranked_room_list[-1] == '"]'):
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

    if(only_explore_one_room != None):
        ranked_room_list = only_explore_one_room

    for gpt4v_find_the_room in ranked_room_list:

        if(gpt4v_find_the_room not in all_rooms):
            print("Error room name:", gpt4v_find_the_room)
            
            to_save['distance'] = distance
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

            return None, None, port_num, env

        print("all_rooms_dict:", all_rooms_dict)
        explore_room_id = [key for key in all_rooms_dict if all_rooms_dict[key] == gpt4v_find_the_room]
        if(len(explore_room_id) == 1):
            explore_room_id = explore_room_id[0]
        else:
            explore_room_id = distance_from_agent(obs['full_graph'], explore_room_id, char_id)
            print("explore_room_id:", explore_room_id)
            explore_room_id = explore_room_id[0]

        large_item_in_current_room_list = []
        large_item_in_current_room_dict = {}

        for the_edge in graph['edges']:
            if(the_edge['to_id'] == explore_room_id and the_edge['relation_type'] == 'INSIDE'):
                if(the_edge['from_id'] in large_item_dict.keys()):
                    large_item_in_current_room_list.append(large_item_dict[the_edge['from_id']])
                    large_item_in_current_room_dict.update({the_edge['from_id']:large_item_dict[the_edge['from_id']]})

        large_item_in_current_room_list = list(set(large_item_in_current_room_list))

        all_rooms_dict.pop(explore_room_id)

        change_room_action = f'<char0> [walk] <{gpt4v_find_the_room}> ({explore_room_id})'

        obs, success, info = env_step(change_room_action, env, steps_num)
        
        if(success == True):
            if(only_explore_one_room != None):
                to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':change_room_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            else:
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

        surround_visible_objects_dict, surround_visible_objects = get_surround_visible_objects(obs, info, to_save, steps_num)

        surround_visible_objects += large_item_in_current_room_list
        surround_visible_objects = list(set(surround_visible_objects))
        surround_visible_objects_dict.update(large_item_in_current_room_dict)

        LLM_decision, success_tag, select_large_objects_prompt, steps_num, previous_generated_steps, to_save, distance, char_init_position = LLM_determine_whether_object_inside_somewhere(gpt4v_find_the_room, object, surround_visible_objects, surround_visible_objects_dict, success_tag, steps_num, previous_generated_steps, to_save, distance, char_init_position)

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
            LLM_decision = LLM_decision.choices[0].message.content
        
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
                    # walk -> find visible object -> open -> find visible object
                    large_object_ids = [key for key, name in surround_visible_objects_dict.items() if name == large_object]
                    while(large_object_ids != []):
                        graph = obs['full_graph']
                        large_object_ids = distance_from_agent(graph, large_object_ids, char_id)
                        instance_level_id = large_object_ids[0]
                        large_object_ids = large_object_ids[1:]

                        explore_large_object_step = f"<char0> [walk] <{large_object}> ({instance_level_id})"
                    
                        obs, success, info = env_step(explore_large_object_step, env, steps_num)
                        
                        if(success == True):
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

                            surround_visible_objects_dict, surround_visible_objects = get_surround_visible_objects(obs, info, to_save, steps_num)
                            
                            if(object in list(surround_visible_objects_dict.values())):
                                object_ids = [key for key, name in surround_visible_objects_dict.items() if name == object]
                                find_object_action = f'<char0> [walk] <{object}> ({object_ids[0]})'

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

    to_save['distance'] = distance
    with open(save_path, "w") as outfile:
        json.dump(to_save, outfile)

    return previous_generated_steps, success_tag, port_num, env

@auto_kill_unity(kill_before_return=True)
def for_loop_data_generation():
    """Loop through all tasks to generate data.
    """
    port_num = 4416
    
    generated_tasks_path_folder = '../../generated_tasks'
    simple_tasks_file = os.path.join(generated_tasks_path_folder, 'navigation_tasks.json')

    testing_environment_ids = [37, 0, 32, 39, 19, 20, 48, 49, 17, 26]

    save_dir = '../../Output/nav/visible_layout/navigation_tasks_with_layout_map'
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
        target_object = result['object']
        task = result['task']
        print("object:", target_object)
        print("task:", task)
        environment_id = int(result['env_id'])

        if(environment_id == 9):
            continue

        if(environment_id not in testing_environment_ids):
            continue

        previous_generated_steps, success_tag, port_num, env = robot_exploration_without_constraint(task, target_object, port_num, env, save_dir, task_id, environment_id, simple_tasks_file)

if __name__ == "__main__":
    for_loop_data_generation()