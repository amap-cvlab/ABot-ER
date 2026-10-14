import os
import sys
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import json
import pickle as pkl
from collections import defaultdict 
# import random
from tasks.utils import auto_kill_unity
from envs.env import VirtualHomeNavigationEnv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--mode", type=str, default='train')
parser.add_argument("-skip", "--skip", type=str, default='[9]')
parser.add_argument("-path", type=str)
parser.add_argument("-save_task_folder", "--save_task_folder", type=str)
parser.add_argument("-save_task_file", "--save_task_file", type=str)
parser.add_argument("-start_from", "--start_from", type=int)
parser.add_argument("-port_num", "--port_num", type=int)
parser = parser.parse_args()

class Graph:
    def __init__(self, edges):
        self.graph = defaultdict(list)
        self.vertices = set()
        for (src, dest) in edges:
            self.graph[src].append(dest)
            self.vertices.add(src)
            self.vertices.add(dest)

    def find_longest_path(self, start):
        visited = {v: False for v in self.vertices}
        stack = []
        longest_path = []

        def dfs(node, path):
            nonlocal longest_path
            visited[node] = True
            path.append(node)
            if len(path) > len(longest_path):
                longest_path = path[:]
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor, path)
            path.pop()
            visited[node] = False

        dfs(start, stack)
        return longest_path

def get_relation_dict(selected_results, item_dict):
    relation_dict = []
    for item in selected_results:
        from_id_class_name = item_dict[item['from_id']]
        to_id_class_name = item_dict[item['to_id']]
        relation_dict.append({'from_id': item['from_id'], 'from_name':from_id_class_name, 'to_id':item['to_id'], 'to_name':to_id_class_name, 'relation_type':item['relation_type']})
    return relation_dict

def get_selected_results(object_ids, relation_location_list, graph):
    selected_results = [item for item in graph['edges'] if item['from_id'] in object_ids and item['relation_type'] in relation_location_list]
    return selected_results

def get_relation_list_from_object(object, graph, item_dict):
    relation_dict = []
    
    object_ids = [item['id'] for item in graph['nodes'] if item['class_name'] == object]

    relation_location_list = ['INSIDE']
    
    selected_results = get_selected_results(object_ids, relation_location_list, graph)
    
    while(True):
        relation_dict_results = get_relation_dict(selected_results, item_dict)
        if(relation_dict_results == []):
            break
        relation_dict += relation_dict_results
        object_ids = [item['to_id'] for item in relation_dict_results]
        selected_results = get_selected_results(object_ids, relation_location_list, graph)
    
    relation_list = [(item['from_name']+'_'+str(item['from_id']),item['to_name']+'_'+str(item['to_id'])) for item in relation_dict]
    relation_list = list(set(relation_list))

    return relation_list


def virtualhome_parser(object_sequence, graph): # remove grab action
    virtualhome_steps = []
    # <char0> [grab] <breadslice> (1)
    for object_item_id in range(len(object_sequence)):
        object_item = object_sequence[object_item_id]
        object_item_split = object_item.split('_')
        virtualhome_steps.append(f'<char0> [walk] <{object_item_split[0]}> ({object_item_split[1]})')

    virtualhome_steps.reverse()
    if(len(virtualhome_steps) <= 2):
        small_object_id = int(virtualhome_steps[-1].split("> (")[1].split(")")[0])
        selected_item = [item for item in graph['nodes'] if item['id'] == small_object_id][0]
        # if('GRABBABLE' in selected_item['properties']):
        #     virtualhome_steps.append(virtualhome_steps[-1].replace("walk", "grab"))
        return virtualhome_steps
    else: # add open large object function
        offset = 1
        virtualhome_steps_new = virtualhome_steps.copy()
        for object_item_id in range(1, len(virtualhome_steps)-1):
            large_object = virtualhome_steps[object_item_id]
            small_object = virtualhome_steps[object_item_id+1]
            large_object_name = large_object.split("] <")[1].split("> (")[0]
            large_object_id = int(large_object.split("> (")[1].split(")")[0])
            small_object_id = int(small_object.split("> (")[1].split(")")[0])
            selected_item = [item for item in graph['nodes'] if item['id'] == large_object_id][0]
            if ([item for item in graph['edges'] if item['from_id'] == small_object_id and item['to_id'] == large_object_id][0]['relation_type'] == 'INSIDE' and 'CLOSED' in selected_item['states'] and 'CAN_OPEN' in selected_item['properties']):
                new_step = f'<char0> [open] <{large_object_name}> ({large_object_id})'
                virtualhome_steps_new.insert(object_item_id+offset, new_step)
                offset += 1
        selected_item = [item for item in graph['nodes'] if item['id'] == small_object_id][0]
        # if('GRABBABLE' in selected_item['properties']):
        #     virtualhome_steps_new.append(virtualhome_steps_new[-1].replace("walk", "grab"))
        return virtualhome_steps_new

@auto_kill_unity(kill_before_return=True)
def data_gen_simple_task(save_dir, port, start_from, idx):

    image_save_path = os.path.join(save_dir, 'RGB')

    if(not os.path.exists(image_save_path)):
        os.mkdir(image_save_path)

    error_cases_dict = {}
        
    generated_tasks_path_folder = f'{parser.save_task_folder}/env_{idx}'
    # input_data_file = 'find_and_movein_env0.json'
    # input_data_file = 'find_and_move_inside_env0.json'
    input_data_file = f'{parser.save_task_file}/env_{idx}.json'
    simple_tasks_file = os.path.join(generated_tasks_path_folder, input_data_file)

    with open(simple_tasks_file, 'r') as json_file:
        json_list = list(json_file)

    env = VirtualHomeNavigationEnv(port=str(port), input_data_file=simple_tasks_file)
    for task_id, json_str in enumerate(json_list[:250]):

        image_save_path_for_task = os.path.join(image_save_path, str(task_id))
        if(not os.path.exists(image_save_path_for_task)):
            os.mkdir(image_save_path_for_task)
        
        if task_id < start_from: 
            continue

        try:
            obs = env.reset_CA(env_id=idx, task_id=task_id, init_room="bedroom")
        except:
            print('Restarting the env')
            port += 1
            env = VirtualHomeNavigationEnv(port=str(port), input_data_file=simple_tasks_file)
            obs = env.reset_CA(env_id=idx, task_id=task_id, init_room="bedroom")

        # edge_dict = {}

        graph = obs['full_graph']

        item_dict = {}
        for item in graph['nodes']:
            item_dict[item['id']] = item['class_name']

        result = json.loads(json_str)

        task_name = result['task'].split('\n')[0]

        # try:
        #     reasonable_tag = result['task'].split('\n')[1].split('conclusion: ')[1]
        # except:
        #     # print("*"*20)
        #     # print("result['task']:", result['task'])
        #     # print("*"*20)
        #     # print("result['task'].split('\n')[1]:", result['task'].split('\n')[1])
        #     # print("*"*20)
        #     # return

        #     # "task": "object_1: toaster\nobject_2: toothbrush\nPut the toothbrush into the toaster\nconclusion: Unreasonable"

        #     continue # Unreasonable

        # if(reasonable_tag == 'Unreasonable'):
        #     continue

        # object = result['object']
        movable_object_1 = result['object1']
        movable_object_2 = result['object2']
        container = result['object3']

        def return_vh_actions_based_on_object(the_object):
            # print("object:", object)
            relation_list_return = get_relation_list_from_object(the_object, graph, item_dict)
            # print("relation_list_return:", relation_list_return)
            edges = [[item[0],item[1]]for item in relation_list_return]

            g = Graph(edges)
            
            first_item_list = [item[0] for item in relation_list_return]
            second_item_list = [item[1] for item in relation_list_return]
            all_objects_in_relation_list = list(set(first_item_list + second_item_list))
            
            start_node_list = [item for item in all_objects_in_relation_list if item.split('_')[0] == the_object]
            vh_length = 0
            vh_actions = None
            
            for start_node in start_node_list:
                # start_node = 'condimentshaker_297'
                longest_path = g.find_longest_path(start_node)
                # print("Longest path starting from node", start_node, "is:", longest_path)
                virtualhome_steps = virtualhome_parser(longest_path, graph)
                if len(virtualhome_steps) > vh_length:
                    vh_length = len(virtualhome_steps)
                    vh_actions = virtualhome_steps
            return vh_actions
            # print("VirtualHome Steps", virtualhome_steps)
        
        movable_object_actions_1 = return_vh_actions_based_on_object(movable_object_1)
        movable_object_actions_2 = return_vh_actions_based_on_object(movable_object_2)
        container_actions = return_vh_actions_based_on_object(container)

        print("movable_object_actions_1:", movable_object_actions_1)
        print("movable_object_actions_2:", movable_object_actions_2)
        print("container_actions:", container_actions)

# movable_object_actions: ['<char0> [walk] <bedroom> (73)', '<char0> [walk] <bookshelf> (105)', '<char0> [walk] <folder> (203)']
# container_actions: ['<char0> [walk] <kitchen> (205)', '<char0> [walk] <bellpepper> (325)']
        
        movable_object_id_1 = movable_object_actions_1[-1].split(' ')[-1]
        movable_object_id_2 = movable_object_actions_2[-1].split(' ')[-1]
        container_id = container_actions[-1].split(' ')[-1]

        vh_actions = movable_object_actions_1
        vh_actions.append('<char0> [grab] <'+movable_object_1+'> '+movable_object_id_1)
        vh_actions = vh_actions + container_actions

        selected_item = [item for item in graph['nodes'] if str(item['id']) == container_id[1:-1]][0]
        # print("selected_item['properties']:", selected_item['properties'])
        if('CAN_OPEN' in selected_item['properties']):
            vh_actions.append('<char0> [open] <'+container+'> '+container_id)
        vh_actions.append('<char0> [putin] <'+movable_object_1+'> '+movable_object_id_1+' <'+container+'> '+container_id) # Put an object inside some other object
        if('CAN_OPEN' in selected_item['properties']):
            vh_actions.append('<char0> [close] <'+container+'> '+container_id)

        vh_actions += movable_object_actions_2
        vh_actions.append('<char0> [grab] <'+movable_object_2+'> '+movable_object_id_2)
        vh_actions = vh_actions + container_actions

        if('CAN_OPEN' in selected_item['properties']):
            vh_actions.append('<char0> [open] <'+container+'> '+container_id)
        vh_actions.append('<char0> [putin] <'+movable_object_2+'> '+movable_object_id_2+' <'+container+'> '+container_id) # Put an object inside some other object
        if('CAN_OPEN' in selected_item['properties']):
            vh_actions.append('<char0> [close] <'+container+'> '+container_id)

        # print(result)
        print(vh_actions)
        print(f"task id: {task_id}\tlength: {len(vh_actions)}")

        action_success = True
        err_msg = None
        # to_save = {'inputs':result['task'], 'steps':{}}
        to_save = {'inputs':task_name, 'steps':{}}
        for _i, action in enumerate(vh_actions):
            action_name = action.split('<char0> [')[1].split(']')[0]
            try:
                # save_img_path = os.path.join(image_save_path_for_task, str(_i)+'.png')
                save_img_path = None

                if(action_name == 'putin'):
                    # try:
                    #     obs, reward, done, truncated, info = env.step_CA(action, save_img_path, recording=False)
                    # except:
                    #     action = action.replace('putin', 'put')
                    #     obs, reward, done, truncated, info = env.step_CA(action, save_img_path, recording=False)
                    obs, reward, done, truncated, info = env.step_CA(action, save_img_path, recording=False)
                    if(info['msg'] is not None):
                        action = action.replace('putin', 'put')
                        obs, reward, done, truncated, info = env.step_CA(action, save_img_path, recording=False)
                else:        
                    obs, reward, done, truncated, info = env.step_CA(action, save_img_path, recording=False)
                err_msg = info['msg']

                if err_msg is None:
                    action_success = True
                    to_save['steps']["STEP_"+str(_i)] = {'obs': obs, 'prompt':'', 'gpt_response':'', 'action': action, 'action_history':vh_actions[:_i], 'msg': err_msg}
                else:
                    action_success = False
                    print(f"Task: {result}")
                    print(f"Actions: {vh_actions}")
                    print(f"{action} cannot be executed")
                    # error_cases_dict[task_id] = {'Task':result['task'], 'Actions': vh_actions, 'Error_action': action}
                    error_cases_dict[task_id] = {'Task':task_name, 'Actions': vh_actions, 'Error_action': action}
                    break
                    
            except:
                action_success = False
                print(f"Task: {result}")
                print(f"Actions: {vh_actions}")
                print(f"{action} cannot be executed")
                # error_cases_dict[task_id] = {'Task':result['task'], 'Actions': vh_actions, 'Error_action':action}
                error_cases_dict[task_id] = {'Task':task_name, 'Actions': vh_actions, 'Error_action':action}
                break
        
        if action_success:
            to_save['steps']["STEP_"+str(_i+1)] = {'obs': obs, 'prompt':'', 'gpt_response':'', 'action':[], 'action_history':vh_actions[:_i+1], 'msg': action_success}
            
            save_path = os.path.join(save_dir, f"task_{task_id}.json")
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

    with open(os.path.join(save_dir, 'error_cases.json'), "w") as outfile:
        json.dump(error_cases_dict, outfile)
        
if __name__ == '__main__':
    port = parser.port_num
    try:
        if parser.mode == 'train':
            indexes = [i for i in range(50) if i not in [0, 17, 19, 20, 26, 32, 37, 39, 48, 49]]
        else:
            indexes = [0, 17, 19, 20, 26, 32, 37, 39, 48, 49]
        for idx in indexes:
            if idx in eval(parser.skip):
                continue
            save_dir = f'{parser.path}/env_{idx}'
            os.makedirs(save_dir, exist_ok=True)
            data_gen_simple_task(save_dir, port, start_from=parser.start_from, idx=idx)
    except KeyboardInterrupt:
        sys.exit(0)
