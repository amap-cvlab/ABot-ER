import os
import sys
import glob
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import json
import pickle as pkl
from collections import defaultdict 
import random
import requests
from tasks.utils import auto_kill_unity
from envs.env import VirtualHomeNavigationEnv, TaskEnv
from simulator.virtualhome.virtualhome.simulation.evolving_graph.utils import get_visible_nodes

from openai import OpenAI
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--mode", type=str, default='train')
parser.add_argument("-skip", "--skip", type=str, default='[9]')
parser.add_argument("-path", type=str)
parser.add_argument("-save_task_folder", "--save_task_folder", type=str)
parser.add_argument("-start_from", "--start_from", type=int)
parser.add_argument("-port_num", "--port_num", type=int)
parser.add_argument("--api_key", "-api_key", type=str, help="api_key to use OPENAI API")
parser = parser.parse_args()

MAX_STEPS = 75

API_key = parser.api_key

def retry_fn(fn, max_failures=10, sleep_time=30):
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

def query_gpt_single(prompt, max_tokens=4096, temperature=0.3, user_role=True):

    client=OpenAI(api_key='<KEY>')
    client.api_key=API_key

    def f(prompt, user_role=True):
        if user_role:
            message = [{'role': 'user', 'content': prompt}]
        else:
            message = prompt
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = message,
            temperature = temperature,
            max_tokens = max_tokens,
        )

        # response = response.choices[0].message.content
        return response

    response = retry_fn(lambda: f(prompt, user_role), max_failures=5, sleep_time=5)

    return response

def env_step(action, env, steps_num, image_save_path_for_task):
    # Wrapper with env
    if(steps_num == None):
        obs, reward, done, truncated, info = env.step_CA(action, save_img_path=None, recording=False)
    else:
        if image_save_path_for_task:
            # save_img_path = os.path.join(image_save_path_for_task, str(steps_num)+'.png')
            save_img_path = None
        else:
            save_img_path = None
        obs, reward, done, truncated, info = env.step_CA(action, save_img_path, recording=False)
    if not info:
        success = False
    elif info['msg'] is None:
        success = True
    else:
        success = False
    return obs, success, info

def robot_find_one_object(task, object, port_num, env, task_env, save_dir, task_id, environment_id, previous_generated_steps, steps_num, to_save, env_obs=None, only_explore_one_room=None, distance=0, cur_room=None):

    generated_tasks_path_folder = parser.save_task_folder
    simple_tasks_file = os.path.join(generated_tasks_path_folder, f'env_{environment_id}.json')
    def distance_from_agent(graph, large_object_ids, char_id):
        agent_location = [item['obj_transform']['position'] for item in graph['nodes'] if item['id'] == char_id[0]][0]
        agent_location = [agent_location[0], agent_location[1]]
        id_distance_dict = {}
        for large_object_id in large_object_ids:
            large_object_location = [item['obj_transform']['position'] for item in graph['nodes'] if item['id'] == large_object_id][0]
            large_object_location = [large_object_location[0], large_object_location[1]]
            id_distance_dict[large_object_id] = np.linalg.norm(np.array(agent_location)-np.array(large_object_location))
        # print("id_distance_dict:", id_distance_dict)
        return sorted(id_distance_dict, key=id_distance_dict.get)
    
    def calculate_2D_distance(init_position, position):
        init_position = np.array(init_position)
        position = np.array(position)
        return np.linalg.norm(init_position[:2]-position[:2])



    prompt_0 = f"Given the task: please find {object}\n\n"

    def LLM_determine_whether_object_inside_somewhere(gpt4v_find_the_room, object, visible_objects, visible_objects_dict, success_tag, steps_num, previous_generated_steps, to_save, distance, char_init_position):

        if(object in visible_objects):
            object_ids = [key for key, name in visible_objects_dict.items() if name == object]
            find_object_action = f'<char0> [walk] <{object}> ({object_ids[0]})'

            obs, success, info = env_step(find_object_action, env, steps_num, image_save_path_for_task)

            if(success == True):
                to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':find_object_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                print("STEP_"+str(steps_num))
                steps_num += 1
                previous_generated_steps.append(find_object_action)
                print("Find object!")
                task_env.update_memory(obs)
                char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                distance += calculate_2D_distance(char_init_position, char_position)
                char_init_position = char_position
                success_tag = True
                return True, success_tag, None, steps_num, previous_generated_steps, to_save, object_ids[0], distance, char_init_position
            else:
                print("Error:", find_object_action)
                # return None, None, None, None, None, None
                return None, None, None, steps_num, previous_generated_steps, to_save, None, distance, char_init_position
        else:
            object_ids = TaskEnv.get_listed_object_ids_from_name(object, task_env.objects_seen)
            if len(object_ids) > 0:
                for object_id in object_ids:
                    find_object_action = f'<char0> [walk] <{object}> ({object_id})'
                    obs, success, info = env_step(find_object_action, env, steps_num, image_save_path_for_task)
                    task_env.update_items_explored(object_id)
                    if success:
                        to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':find_object_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                        print("STEP_"+str(steps_num))
                        steps_num += 1
                        previous_generated_steps.append(find_object_action)
                        print("Find object!")
                        char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                        distance += calculate_2D_distance(char_init_position, char_position)
                        char_init_position = char_position
                        success_tag = True
                        return True, success_tag, None, steps_num, previous_generated_steps, to_save, object_id, distance, char_init_position
            
            prompt_2 = f"Currently, the robot is in the room {gpt4v_find_the_room} and can see objects {visible_objects}. Since the target object could be obscured by some larger visible objects, the robot needs to explore the room further. Is this possible the object {object} is located inside or nearby one of these visible objects? If yes, please only output these possbile visible objects in a Python-style list and in the order of possibility, if not, please only output None.\n\nPlease do not output the thinking process.\n\n"
            prompt_for_gpt4v = prompt_0 + prompt_2
        
            gpt4v_response = query_gpt_single(prompt_for_gpt4v)

            # we should save something here
            
            return gpt4v_response, success_tag, prompt_for_gpt4v, steps_num, previous_generated_steps, to_save, None, distance, char_init_position

    # save_path = os.path.join(save_dir, f"task_{task_id}.pkl")

    image_save_path = os.path.join(save_dir, 'RGB')

    if(not os.path.exists(image_save_path)):
        os.mkdir(image_save_path)

    image_save_path_for_task = os.path.join(image_save_path, str(task_id))

    if(not os.path.exists(image_save_path_for_task)):
        os.mkdir(image_save_path_for_task)
    
    # to_save = {'inputs':task, 'steps':[]}

    
    def get_visible_objects(obs):
        first_person_view_visible_graph = obs['visible_graph']
        visible_objects_dict = {}
        visible_objects = []
        for the_item in first_person_view_visible_graph['nodes']:
            visible_objects_dict.update({the_item['id']:the_item['class_name']})
            visible_objects.append(the_item['class_name'])

        visible_objects = list(set(visible_objects))
        return visible_objects_dict, visible_objects

    def get_surround_visible_objects(obs, image_save_path_for_task):

        visible_objects_dict, visible_objects = get_visible_objects(obs)

        surround_visible_objects_dict = {}
        surround_visible_objects = []

        surround_visible_objects_dict.update(visible_objects_dict)
        surround_visible_objects += visible_objects
        
        # for surround_times in range(3): # 360 degrees rotation in total
        #     exe_action_list = ['<char0> [turnright]','<char0> [turnright]','<char0> [turnright]'] # 90 degrees, 30 degrees for each
        for surround_times in range(11): # 360 degrees rotation in total
            exe_action_list = ['<char0> [turnright]'] # 30 degrees for each

            obs, success, info = env_step(exe_action_list, env, steps_num=None, image_save_path_for_task=image_save_path_for_task)

            # print("success:", success)
            if(success == True):
                visible_objects_dict, visible_objects = get_visible_objects(obs)

                surround_visible_objects_dict.update(visible_objects_dict)
                surround_visible_objects += visible_objects

            else:
                print("Turn Right Error")

        surround_visible_objects = list(set(surround_visible_objects))

        print("surround_visible_objects:", surround_visible_objects)

        return surround_visible_objects_dict, surround_visible_objects

    init_room_pool = ['bathroom', 'bedroom', 'kitchen', 'livingroom']
    random_init_room = random.choice(init_room_pool)
    if not to_save['initial_room']:
        to_save['initial_room'] = random_init_room
    print("port before try except:", port_num)

    if not env_obs:
        try:
            # obs = env.reset(init_room="bedroom")
            # obs = env.reset(init_room=random_init_room)
            obs = env.reset_CA(env_id=environment_id, init_room=random_init_room)
        except:
            print('Restarting the env')
            print("port inside except:", port_num)
            port_num += 1
            env = VirtualHomeNavigationEnv(port=str(port_num), input_data_file=simple_tasks_file)
            # obs = env.reset(init_room="bedroom")
            # obs = env.reset(init_room=random_init_room)
            obs = env.reset_CA(env_id=environment_id,init_room=random_init_room)
            task_env.reset_env_room(random_init_room, obs, port_num)
    else:
        obs = env_obs

    if not obs:
        return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, None
    graph = obs['full_graph']

    char_id = [item['id'] for item in graph['nodes'] if item['category'] == 'Characters']
    char_info = [item for item in graph['nodes'] if item['category'] == 'Characters'][0]
    char_init_position = char_info['obj_transform']['position']
    if not to_save['initial_position']:
        to_save['initial_position'] = char_init_position

    # success, graph = comm.environment_graph()
    
    success_tag = False
    
    # all_rooms = [item['class_name'] for item in graph['nodes'] if item['category'] == 'Rooms']
    all_rooms_pair = [[item['class_name'],item['id']] for item in graph['nodes'] if item['category'] == 'Rooms']
    all_rooms = []
    all_rooms_dict = {}
    for item in all_rooms_pair:
        all_rooms.append(item[0])
        # all_rooms_dict.update({item[0]:item[1]})
        all_rooms_dict.update({item[1]:item[0]})

    print("all_rooms:", all_rooms)
    print("all_rooms_dict:", all_rooms_dict)
    
    # task = 'Please pass me the condiment shaker.'
    # object = 'condimentshaker'
    # task = 'Please pass me the clothesshirt.'
    # object = 'clothesshirt'
     
    prompt_1 = f"Determine which room may contain the object {object}, and the room list is {all_rooms}. Please ranking these rooms based on the possibility and only output a Python-style list.\n\nThe number of output rooms should be the same as the number of rooms in the original room list.\n\nPlease do not output the answer like 'As an AI language model, I don not have the ability to physically determine the location of objects or bring them to you.'\n\nPlease do not output the thinking process.\n\n"
    
    prompt_for_gpt4v = prompt_0 + prompt_1

    find_room_success_tag = False

    max_num = 0

    while(find_room_success_tag == False):

        if(max_num >=5):
            print("Maximum finding room retry")
            to_save['distance'] = distance
            # break
            return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, None
        max_num += 1

        gpt4v_response = query_gpt_single(prompt_for_gpt4v)

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
            # if(len(ranked_room_list) == 4):
            if(len(ranked_room_list) == len(all_rooms)):
                find_room_success_tag = True
                pass
            else:
                print("Error GPT4V format find rooms")
        except:
            print("Error GPT4V format find rooms")



    if(only_explore_one_room != None):
        ranked_room_list = only_explore_one_room


    for k, gpt4v_find_the_room in enumerate(ranked_room_list):

        if(gpt4v_find_the_room not in all_rooms):
            print("Error room name:", gpt4v_find_the_room)
            to_save['distance'] = distance
            return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, None
        print("all_rooms_dict:", all_rooms_dict)
        if (not cur_room) or (not cur_room == ranked_room_list[0]) or (not k == 0):
            explore_room_id = [key for key in all_rooms_dict if all_rooms_dict[key] == gpt4v_find_the_room]
            if(len(explore_room_id) == 1):
                explore_room_id = explore_room_id[0]
            else:
                explore_room_id = distance_from_agent(obs['full_graph'], explore_room_id, char_id)
                print("explore_room_id:", explore_room_id)
                if len(explore_room_id) == 0: 
                    continue
                explore_room_id = explore_room_id[0]

            all_rooms_dict.pop(explore_room_id)

            change_room_action = f'<char0> [walk] <{gpt4v_find_the_room}> ({explore_room_id})'

            obs, success, info = env_step(change_room_action, env, steps_num, image_save_path_for_task)
            if not obs:
                return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, None
            
            if(success == True):
                # to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': prompt_for_gpt4v, 'gpt_response':gpt4v_find_the_room, 'action':change_room_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                if(only_explore_one_room != None):
                    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':change_room_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                else:
                    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': prompt_for_gpt4v, 'gpt_response':ranked_room_list_origin, 'action':change_room_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                print("STEP_"+str(steps_num))
                char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                distance += calculate_2D_distance(char_init_position, char_position)
                char_init_position = char_position
                steps_num += 1
                cur_room = gpt4v_find_the_room

                if(steps_num > MAX_STEPS): # exceed the maximum steps
                    to_save['distance'] = distance
                    return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
                
                previous_generated_steps.append(change_room_action)
            else:
                print("Failed VirtualHome action:", change_room_action)
                if info:
                    print("message:", info['msg'])
                to_save['distance'] = distance
                # break
                return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room

            surround_visible_objects_dict, surround_visible_objects = get_surround_visible_objects(obs, image_save_path_for_task)

            LLM_decision, success_tag, select_large_objects_prompt, steps_num, previous_generated_steps, to_save, object_id, distance, char_init_position = LLM_determine_whether_object_inside_somewhere(gpt4v_find_the_room, object, surround_visible_objects, surround_visible_objects_dict, success_tag, steps_num, previous_generated_steps, to_save, distance, char_init_position)
            if not obs:
                return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, None

        else:
            surround_visible_objects_dict, surround_visible_objects = get_surround_visible_objects(obs, image_save_path_for_task)

            LLM_decision, success_tag, select_large_objects_prompt, steps_num, previous_generated_steps, to_save, object_id, distance, char_init_position = LLM_determine_whether_object_inside_somewhere(cur_room, object, surround_visible_objects, surround_visible_objects_dict, success_tag, steps_num, previous_generated_steps, to_save, distance, char_init_position)
            if not obs:
                return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, None

        if(steps_num > MAX_STEPS): # exceed the maximum steps
            to_save['distance'] = distance
            return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
        
        if(LLM_decision == None and success_tag == None and select_large_objects_prompt == None):
            to_save['distance'] = distance
            return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
            
        if(str(LLM_decision).lower() == 'true' or LLM_decision == True): # Find object
            break
        else:
            # select_large_objects_response = LLM_decision.copy()
            # LLM_decision = LLM_decision['choices'][0]['message']['content']
            # select_large_objects_response = LLM_decision
            LLM_decision = LLM_decision.choices[0].message.content
        
        print("LLM_decision:", LLM_decision)
        # if(LLM_decision == 'None' or " not " in LLM_decision.lower()): # Sometimes LLM will generate response like: The condiment shaker is not listed as one of the visible objects in the room.
        if(LLM_decision == 'None'):
            pass
        else: # a list of large object
            LLM_decision_list = []
            if(LLM_decision[:2] == "['" and LLM_decision[-2:] == "']"): # valid output
                # LLM_decision = LLM_decision[1:-1]
                LLM_decision = LLM_decision[2:-2]
            elif(LLM_decision[:2] == '["' and LLM_decision[-2:] == '"]'):
                LLM_decision = LLM_decision[2:-2]
            elif("['" in LLM_decision and "']" in LLM_decision):
                LLM_decision = LLM_decision.split("['")[1].split("']")[0]
            elif('["' in LLM_decision and '"]' in LLM_decision):
                LLM_decision = LLM_decision.split('["')[1].split('"]')[0]
            else:
                print("Error GPT4V format large objects")
                # return None, None, port_num, env
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
                        if not obs:
                            return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, None
                        graph = obs['full_graph']
                        large_object_ids = distance_from_agent(graph, large_object_ids, char_id)
                        instance_level_id = large_object_ids[0]
                        large_object_ids = large_object_ids[1:]
                    # for instance_level_id in large_object_ids:
                        # explore_large_object_step = f"<char0> [walk] <{large_object}> ({large_object_ids[0]})"
                        explore_large_object_step = f"<char0> [walk] <{large_object}> ({instance_level_id})"
                    
                        obs, success, info = env_step(explore_large_object_step, env, steps_num, image_save_path_for_task)
                        
                        if(success == True):
                            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': select_large_objects_prompt, 'gpt_response':LLM_decision, 'action':explore_large_object_step, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                            print("STEP_"+str(steps_num))
                            steps_num += 1
                            char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                            distance += calculate_2D_distance(char_init_position, char_position)
                            char_init_position = char_position

                            if(steps_num > MAX_STEPS): # exceed the maximum steps
                                to_save['distance'] = distance
                                return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
            
                            previous_generated_steps.append(explore_large_object_step)
                            task_env.update_memory(obs)
                        
                            # open_large_object_step = f"<char0> [open] <{large_object}> ({large_object_ids[0]})"
                            # close_large_object_step = f"<char0> [close] <{large_object}> ({large_object_ids[0]})"
                            open_large_object_step = f"<char0> [open] <{large_object}> ({instance_level_id})"
                            close_large_object_step = f"<char0> [close] <{large_object}> ({instance_level_id})"
                            
                            obs, success, info = env_step(open_large_object_step, env, steps_num, image_save_path_for_task)

                            need_close_tag = False
                            if(success == True):
                                to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':open_large_object_step, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                                print("STEP_"+str(steps_num))
                                steps_num += 1

                                if(steps_num > MAX_STEPS): # exceed the maximum steps
                                    return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
            
                                previous_generated_steps.append(open_large_object_step)
                                need_close_tag = True

                            else:
                                print("Error:", explore_large_object_step)
                                # Cannot open the container, it is fine

                            surround_visible_objects_dict, surround_visible_objects = get_surround_visible_objects(obs, image_save_path_for_task)
                            
                            if(object in list(surround_visible_objects_dict.values())):
                                object_ids = [key for key, name in surround_visible_objects_dict.items() if name == object]
                                find_object_action = f'<char0> [walk] <{object}> ({object_ids[0]})'

                                obs, success, info = env_step(find_object_action, env, steps_num, image_save_path_for_task)

                                if(success == True): # Find object
                                    object_id = object_ids[0]
                                    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':find_object_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                                    print("STEP_"+str(steps_num))
                                    steps_num += 1
                                    task_env.update_memory(obs)
                                    char_position = [item for item in obs['full_graph']['nodes'] if item['category'] == 'Characters'][0]['obj_transform']['position']
                                    distance += calculate_2D_distance(char_init_position, char_position)
                                    char_init_position = char_position
                                    if(steps_num > MAX_STEPS): # exceed the maximum steps
                                        to_save['distance'] = distance
                                        return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
            
                                    previous_generated_steps.append(find_object_action)
                                    print("Find object!")

                                    success_tag = True

                                    LLM_decision = True

                                    break
                                else:
                                    print("Error:", find_object_action)
                                    return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room

                            if(need_close_tag == True):
                                obs, success, info = env_step(close_large_object_step, env, steps_num, image_save_path_for_task)
                                if(success == True):
                                    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':close_large_object_step, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
                                    print("STEP_"+str(steps_num))
                                    steps_num += 1

                                    if(steps_num > MAX_STEPS): # exceed the maximum steps
                                        to_save['distance'] = distance
                                        return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
            
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
    to_save['distance'] = distance

    # to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':[], 'action_history': previous_generated_steps.copy(), 'msg': success_tag}}) # save final status
    # print("STEP_"+str(steps_num))


    if success_tag:
        return previous_generated_steps, success_tag, port_num, env, object_id, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room
    else:
        return previous_generated_steps, success_tag, port_num, env, None, task_env, to_save, steps_num, obs, distance, gpt4v_find_the_room

def robot_pick_one_object(task, object, port_num, env, task_env, save_dir, task_id, environment_id, previous_generated_steps, \
                          steps_num, to_save, only_explore_one_room, image_save_path_for_task, obs, distance, cur_room):
    success_tag = False
    previous_generated_steps, success_tag, port_num, env, object_id, task_env, to_save, steps_num, obs, distance, cur_room = robot_find_one_object(task, object, port_num, env, task_env, save_dir, task_id, \
                                                                                            environment_id, previous_generated_steps, steps_num, to_save, obs, only_explore_one_room, distance, cur_room)
    if not obs:
        return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, cur_room
    if success_tag:
        grab_action = f'<char0> [grab] <{object}> ({object_id})'
        obs, success, info = env_step(grab_action, env, steps_num, image_save_path_for_task)
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':grab_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            if(steps_num > MAX_STEPS): # exceed the maximum steps
                return previous_generated_steps, None, port_num, env, None, task_env, to_save, steps_num, obs, distance, cur_room
            previous_generated_steps.append(grab_action)
        else:
            print("Error:", grab_action)
            previous_generated_steps.append(grab_action)
            success_tag = False
    return previous_generated_steps, success_tag, port_num, env, object_id, task_env, to_save, steps_num, obs, distance, cur_room

def robot_exploration_multi(task, port_num, env, task_env, save_dir, task_id, environment_id, only_explore_one_room=None):
    save_path = os.path.join(save_dir, f"task_{task_id}.json")
    image_save_path = None
    image_save_path_for_task = None
    object1 = task['object1']
    initial_to_save = {'inputs':task, 'environment_id':environment_id, 'initial_room':None, 'initial_position':None, 'distance':0, 'steps':{}}
    previous_generated_steps, success_tag, port_num, env, object_id1, task_env, to_save, steps_num, obs, distance, cur_room = robot_pick_one_object(task, object1, port_num, env, task_env, save_dir, task_id, \
                                                                                            environment_id, [], 0, initial_to_save, only_explore_one_room, image_save_path_for_task, None, 0, None)

    
    if not obs:
        with open(save_path, "w") as outfile:
            json.dump(to_save, outfile)
        return None, None, port_num, env, None, task_env, to_save, steps_num

    if success_tag:
        object3 = task['object3']
        print("Start to find surface")
        previous_generated_steps, success_tag, port_num, env, object_id3, task_env, to_save, steps_num, obs, distance, cur_room = robot_find_one_object(task, object3, port_num, env, task_env, save_dir, task_id, \
                                                                                            environment_id, previous_generated_steps, steps_num, to_save, obs, only_explore_one_room, distance, cur_room)

    need_close_tag = False
    if success_tag:
        print("Successfully find container and grab object 1")
        open_action = f'<char0> [open] <{object3}> ({object_id3})'
        obs, success, info = env_step(open_action, env, steps_num, image_save_path_for_task)
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':open_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            if(steps_num > MAX_STEPS): # exceed the maximum steps
                with open(save_path, "w") as outfile:
                    json.dump(to_save, outfile)
                return None, None, port_num, env, None, task_env, to_save, steps_num
            previous_generated_steps.append(open_action)
            need_close_tag = True
        else:
            print("Error:", open_action)
            previous_generated_steps.append(open_action)
            success_tag = False
    

    if success_tag:
        print("Successfully open container and grab object 1")
        put_on_action = f'<char0> [put] <{object1}> ({object_id1}) <{object3}> ({object_id3})'
        obs, success, info = env_step(put_on_action, env, steps_num, image_save_path_for_task)
        if not success:
            put_on_action = f'<char0> [putin] <{object1}> ({object_id1}) <{object3}> ({object_id3})'
            obs, success, info = env_step(put_on_action, env, steps_num, image_save_path_for_task)
        
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':put_on_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            if(steps_num > MAX_STEPS): # exceed the maximum steps
                with open(save_path, "w") as outfile:
                    json.dump(to_save, outfile)
                return None, None, port_num, env, None, task_env, to_save, steps_num
            previous_generated_steps.append(put_on_action)
        else:
            print("Error:", put_on_action)
            previous_generated_steps.append(put_on_action)
            success_tag = False
            # return previous_generated_steps, success_tag, port_num, env, None, task_env, to_save, steps_num
    
    if success_tag and need_close_tag:
        print("Close container")
        close_action = f'<char0> [close] <{object3}> ({object_id3})'
        obs, success, info = env_step(close_action, env, steps_num, image_save_path_for_task)
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':close_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            if(steps_num > MAX_STEPS): # exceed the maximum steps
                with open(save_path, "w") as outfile:
                    json.dump(to_save, outfile)
                return None, None, port_num, env, None, task_env, to_save, steps_num
            previous_generated_steps.append(close_action)
            need_close_tag = True
        else:
            print("Error:", close_action)
            previous_generated_steps.append(close_action)
            success_tag = False        
    
    if success_tag:
        object2 = task['object2']
        previous_generated_steps, success_tag, port_num, env, object_id2, task_env, to_save, steps_num, obs, distance, cur_room = robot_pick_one_object(task, object2, port_num, env, task_env, save_dir, task_id, \
                                                                                    environment_id, previous_generated_steps, steps_num, to_save, only_explore_one_room, image_save_path_for_task, obs, distance, cur_room)

    if success_tag:
        previous_generated_steps, success_tag, port_num, env, object_id3, task_env, to_save, steps_num, obs, distance, cur_room = robot_find_one_object(task, object3, port_num, env, task_env, save_dir, task_id, \
                                                                                            environment_id, previous_generated_steps, steps_num, to_save, obs, only_explore_one_room, distance, cur_room)
    need_close_tag = False
    if success_tag:
        print("Successfully find container and grab object 2")
        open_action = f'<char0> [open] <{object3}> ({object_id3})'
        obs, success, info = env_step(open_action, env, steps_num, image_save_path_for_task)
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':open_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            if(steps_num > MAX_STEPS): # exceed the maximum steps
                with open(save_path, "w") as outfile:
                    json.dump(to_save, outfile)
                return None, None, port_num, env, None, task_env, to_save, steps_num
            previous_generated_steps.append(open_action)
            need_close_tag = True
        else:
            print("Error:", open_action)
            previous_generated_steps.append(open_action)
            success_tag = False



    if success_tag:
        print("Successfully open container and grab object 2")
        put_on_action = f'<char0> [put] <{object2}> ({object_id2}) <{object3}> ({object_id3})'
        obs, success, info = env_step(put_on_action, env, steps_num, image_save_path_for_task)
        if not success:
            put_on_action = f'<char0> [putin] <{object2}> ({object_id2}) <{object3}> ({object_id3})'
            obs, success, info = env_step(put_on_action, env, steps_num, image_save_path_for_task)
        
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':put_on_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            if(steps_num > MAX_STEPS): # exceed the maximum steps
                with open(save_path, "w") as outfile:
                    json.dump(to_save, outfile)
                return None, None, port_num, env, None, task_env, to_save, steps_num
            previous_generated_steps.append(put_on_action)
        else:
            print("Error:", put_on_action)
            previous_generated_steps.append(put_on_action)
            success_tag = False
    
    if success_tag and need_close_tag:
        print("Close container")
        close_action = f'<char0> [close] <{object3}> ({object_id3})'
        obs, success, info = env_step(close_action, env, steps_num, image_save_path_for_task)
        if(success == True):
            to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':close_action, 'action_history': previous_generated_steps.copy(), 'msg': info['msg']}})
            print("STEP_"+str(steps_num))
            steps_num += 1

            if(steps_num > MAX_STEPS): # exceed the maximum steps
                with open(save_path, "w") as outfile:
                    json.dump(to_save, outfile)
                return None, None, port_num, env, None, task_env, to_save, steps_num
            previous_generated_steps.append(close_action)
            need_close_tag = True
        else:
            print("Error:", close_action)
            previous_generated_steps.append(close_action)
            success_tag = False  
    
    to_save['steps'].update({"STEP_"+str(steps_num):{'obs': obs, 'prompt': '', 'gpt_response':'', 'action':[], 'action_history': previous_generated_steps.copy(), 'msg': success_tag}})
    steps_num += 1


    with open(save_path, "w") as outfile:
        json.dump(to_save, outfile)

    if(steps_num > MAX_STEPS): # exceed the maximum steps
        return None, None, port_num, env, None, task_env, to_save, steps_num      
    return previous_generated_steps, success_tag, port_num, env, None, task_env, to_save, steps_num
        
        
        

    


@auto_kill_unity(kill_before_return=True)
def for_loop_data_generation(env_id=0):
    port_num = parser.port_num
    
    generated_tasks_path_folder = parser.save_task_folder
    simple_tasks_file = os.path.join(generated_tasks_path_folder, f'env_{env_id}.json')
    save_dir = f'{parser.path}/env_{env_id}'
    os.makedirs(save_dir, exist_ok=True)
    
    with open(simple_tasks_file, 'r') as json_file:
        json_list = list(json_file)

    exist_files = glob.glob(f'{save_dir}/task*')
    if len(exist_files) == 0:
        max_idx = 0
    else:
        try:
            max_idx = max([int(i[parser.start_from+1:-5]) for i in exist_files]) + 1
        except:
            max_idx = max([int(i[parser.start_from:-5]) for i in exist_files]) + 1
    print("Start from: ", max_idx)
    env = VirtualHomeNavigationEnv(port=str(port_num), input_data_file=simple_tasks_file)
    print("port after VirtualHomeNavigationEnv:", port_num)
    
    for task_id, json_str in enumerate(json_list[max_idx:60]):
        task_id += max_idx
        result = json.loads(json_str)
        # print("result:", result)
        # target_object = result['object']
        # task = result['task']
        # print("object:", target_object)
        print("task:", result['task'])
        environment_id = int(result['env_id'])

        if(environment_id == 9):
            continue
        task_env = TaskEnv(env, result, task_id, save_dir, use_partial_graph=False)

        previous_generated_steps, success_tag, port_num, env, _, task_env, to_save, steps_num = robot_exploration_multi(result, port_num, env, task_env, save_dir, task_id, environment_id, only_explore_one_room=None)


        # break

if __name__ == "__main__":
    try:
        if parser.mode == 'train':
            indexes = [i for i in range(50) if i not in [0, 17, 19, 20, 26, 32, 37, 39, 48, 49]]
        else:
            indexes = [0, 17, 19, 20, 26, 32, 37, 39, 48, 49]
        for i in indexes:
            if i in eval(parser.skip):
                continue
            for_loop_data_generation(i)
    except KeyboardInterrupt:
        sys.exit(0)