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
        # print("inside while")
        relation_dict_results = get_relation_dict(selected_results, item_dict)
        if(relation_dict_results == []):
            break
        relation_dict += relation_dict_results
        object_ids = [item['to_id'] for item in relation_dict_results]
        selected_results = get_selected_results(object_ids, relation_location_list, graph)
    
    relation_list = [(item['from_name']+'_'+str(item['from_id']),item['to_name']+'_'+str(item['to_id'])) for item in relation_dict]
    relation_list = list(set(relation_list))

    return relation_list

def virtualhome_parser(object_sequence, graph):
    virtualhome_steps = []
    for object_item_id in range(len(object_sequence)):
        object_item = object_sequence[object_item_id]
        object_item_split = object_item.split('_')
        virtualhome_steps.append(f'<char0> [walk] <{object_item_split[0]}> ({object_item_split[1]})')

    virtualhome_steps.reverse()
    if(len(virtualhome_steps) <= 2):
        small_object_id = int(virtualhome_steps[-1].split("> (")[1].split(")")[0])
        selected_item = [item for item in graph['nodes'] if item['id'] == small_object_id][0]
        if('GRABBABLE' in selected_item['properties']):
            virtualhome_steps.append(virtualhome_steps[-1].replace("walk", "grab"))
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
        if('GRABBABLE' in selected_item['properties']):
            virtualhome_steps_new.append(virtualhome_steps_new[-1].replace("walk", "grab"))
        return virtualhome_steps_new

@auto_kill_unity(kill_before_return=True)
def data_gen_simple_task(save_dir, port, start_from, simple_tasks_file):

    image_save_path = os.path.join(save_dir, 'RGB')

    if(not os.path.exists(image_save_path)):
        os.mkdir(image_save_path)

    error_cases_dict = {}

    with open(simple_tasks_file, 'r') as json_file:
        json_list = list(json_file)

    env = VirtualHomeNavigationEnv(port=str(port), input_data_file=simple_tasks_file)
    for task_id, json_str in enumerate(json_list):

        result = json.loads(json_str)
        environment_id = int(result['env_id'])

        image_save_path_for_task = os.path.join(image_save_path, str(task_id))
        if(not os.path.exists(image_save_path_for_task)):
            os.mkdir(image_save_path_for_task)
        
        if task_id < start_from: 
            continue
        
        try:
            obs = env.reset_CA(task_id=task_id, env_id=environment_id, init_room="bedroom")
        except:
            print('Restarting the env')
            port += 1
            env = VirtualHomeNavigationEnv(port=str(port), input_data_file=simple_tasks_file)
            obs = env.reset_CA(task_id=task_id, env_id=environment_id, init_room="bedroom")

        graph = obs['full_graph']

        item_dict = {}
        for item in graph['nodes']:
            item_dict[item['id']] = item['class_name']

        object = result['object']
        relation_list_return = get_relation_list_from_object(object, graph, item_dict)
        edges = [[item[0],item[1]]for item in relation_list_return]

        g = Graph(edges)
        
        first_item_list = [item[0] for item in relation_list_return]
        second_item_list = [item[1] for item in relation_list_return]
        all_objects_in_relation_list = list(set(first_item_list + second_item_list))
        
        start_node_list = [item for item in all_objects_in_relation_list if item.split('_')[0] == object]
        vh_length = 0
        vh_actions = None
        
        for start_node in start_node_list:
            longest_path = g.find_longest_path(start_node)
            virtualhome_steps = virtualhome_parser(longest_path, graph)
            if len(virtualhome_steps) > vh_length:
                vh_length = len(virtualhome_steps)
                vh_actions = virtualhome_steps
        print(f"task id: {task_id}\tlength: {len(vh_actions)}")

        action_success = True
        err_msg = None
        to_save = {'inputs':result['task'], 'steps':{}}
        for _i, action in enumerate(vh_actions):
            try:
                save_img_path = os.path.join(image_save_path_for_task, str(_i)+'.png')
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
                    error_cases_dict[task_id] = {'Task':result['task'], 'Actions': vh_actions, 'Error_action': action}
                    break
                    
            except:
                action_success = False
                print(f"Task: {result}")
                print(f"Actions: {vh_actions}")
                print(f"{action} cannot be executed")
                error_cases_dict[task_id] = {'Task':result['task'], 'Actions': vh_actions, 'Error_action':action}
                break
        
        if action_success:
            to_save['steps']["STEP_"+str(_i+1)] = {'obs': obs, 'prompt':'', 'gpt_response':'', 'action':[], 'action_history':vh_actions[:_i+1], 'msg': action_success}
            
            save_path = os.path.join(save_dir, f"task_{task_id}.json")
            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

    with open(os.path.join(save_dir, 'error_cases.json'), "w") as outfile:
        json.dump(error_cases_dict, outfile)


@auto_kill_unity(kill_before_return=True)
def data_gen_simple_task_with_constraint_filter_invalid_constraint(save_dir, port, start_from, simple_tasks_file):

    image_save_path = os.path.join(save_dir, 'RGB')

    if(not os.path.exists(image_save_path)):
        os.mkdir(image_save_path)

    error_cases_dict = {}

    with open(simple_tasks_file, 'r') as json_file:
        json_list = list(json_file)

    env = VirtualHomeNavigationEnv(port=str(port), input_data_file=simple_tasks_file)

    for task_id, json_str in enumerate(json_list):

        save_path = os.path.join(save_dir, f"task_{task_id}.json")
        if(os.path.exists(save_path)):
            continue
        
        result = json.loads(json_str)
        environment_id = int(result['env_id'])

        image_save_path_for_task = os.path.join(image_save_path, str(task_id))
        if(not os.path.exists(image_save_path_for_task)):
            os.mkdir(image_save_path_for_task)
        
        if task_id < start_from: 
            continue
        
        try:
            obs = env.reset_CA(task_id=task_id, env_id=environment_id, init_room="bedroom")
        except:
            print('Restarting the env')
            port += 1
            env = VirtualHomeNavigationEnv(port=str(port), input_data_file=simple_tasks_file)
            obs = env.reset_CA(task_id=task_id, env_id=environment_id, init_room="bedroom")

        graph = obs['full_graph']

        two_objects_relation = result['task_completion_criterion'].split(')(')[-1].split(',')[0] # new addition
        object_1 = result['object_1']
        object_2 = result['object_2']
        constraint = f"({two_objects_relation}, {object_1}, {object_2})"
        print("constraint:", constraint)

        all_objects_dict = {}
        for the_item in graph['nodes']:
            all_objects_dict.update({the_item['id']:the_item['class_name']})

        whole_graph_edges_list = []
        whole_graph_edges = graph['edges']

        for item in whole_graph_edges:
            whole_graph_edges_list.append("("+item['relation_type']+", "+all_objects_dict[item['from_id']]+", "+all_objects_dict[item['to_id']]+")")

        whole_graph_edges_list = list(set(whole_graph_edges_list))

        if(constraint in whole_graph_edges_list):
            pass
        else:
            continue

        item_dict = {}
        for item in graph['nodes']:
            item_dict[item['id']] = item['class_name']

        object = result['object_1']
        instant_id = int(result['instant_id_1'])

        relation_list_return = get_relation_list_from_object(object, graph, item_dict)
        edges = [[item[0],item[1]]for item in relation_list_return]

        g = Graph(edges)
        
        first_item_list = [item[0] for item in relation_list_return]
        second_item_list = [item[1] for item in relation_list_return]
        all_objects_in_relation_list = list(set(first_item_list + second_item_list))
        
        start_node_list = [item for item in all_objects_in_relation_list if item.split('_')[0] == object]
        vh_length = 0
        vh_actions = None
        
        print("start_node_list:", start_node_list)

        for start_node in start_node_list:

            start_node_id = start_node.split("_")[-1]
            if(start_node_id != str(instant_id)):
                continue
            longest_path = g.find_longest_path(start_node)
            virtualhome_steps = virtualhome_parser(longest_path, graph)
            if len(virtualhome_steps) > vh_length:
                vh_length = len(virtualhome_steps)
                vh_actions = virtualhome_steps
        print(f"task id: {task_id}\tlength: {len(vh_actions)}")

        action_success = True
        err_msg = None

        task = result['task']
        if('\n' in task and task[-1] != '\n'):
            task = task.split('\n')[-1]

        to_save = {'all_inputs_info': result, 'inputs':task, 'steps':{}}
        for _i, action in enumerate(vh_actions):
            try:
                save_img_path = os.path.join(image_save_path_for_task, str(_i)+'.png')
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
                    error_cases_dict[task_id] = {'Task':result['task'], 'Actions': vh_actions, 'Error_action': action}
                    break
                    
            except:
                action_success = False
                print(f"Task: {result}")
                print(f"Actions: {vh_actions}")
                print(f"{action} cannot be executed")
                error_cases_dict[task_id] = {'Task':result['task'], 'Actions': vh_actions, 'Error_action':action}
                break
        
        if action_success:
            to_save['steps']["STEP_"+str(_i+1)] = {'obs': obs, 'prompt':'', 'gpt_response':'', 'action':[], 'action_history':vh_actions[:_i+1], 'msg': action_success}

            with open(save_path, "w") as outfile:
                json.dump(to_save, outfile)

    with open(os.path.join(save_dir, 'error_cases.json'), "w") as outfile:
        json.dump(error_cases_dict, outfile)

def gen_simple_task(port, start_from, simple_tasks_file):
    save_dir = '../../Output/nav/ground_truth/navigation_gt'
    os.makedirs(save_dir, exist_ok=True)
    data_gen_simple_task(save_dir, port, start_from, simple_tasks_file)

def gen_simple_task_with_constraint_filter_invalid_constraint(port, start_from, simple_tasks_file):
    save_dir = '../../Output/nav/ground_truth/navigation_with_spatial_constraint_gt'
    os.makedirs(save_dir, exist_ok=True)
    data_gen_simple_task_with_constraint_filter_invalid_constraint(save_dir, port, start_from, simple_tasks_file)

if __name__ == '__main__':
    port = 2345
    
    generated_tasks_path_folder = '../../generated_tasks'

    simple_tasks_file = os.path.join(generated_tasks_path_folder, 'navigation_tasks.json')
    gen_simple_task(port, 0, simple_tasks_file)

    simple_tasks_with_spatial_constraint_file = os.path.join(generated_tasks_path_folder, 'navigation_tasks_with_spatial_constraint.json')
    gen_simple_task_with_constraint_filter_invalid_constraint(port, 0, simple_tasks_with_spatial_constraint_file)