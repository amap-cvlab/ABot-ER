import os
import subprocess
import time
import torch
import shutil
import requests


def command_exists(cmd: str) -> bool:
    path = shutil.which(cmd)
    return path is not None

def get_group_master_rank(pipeline_parallel_size):
    own_rank = int(os.environ['RANK'])
    group_master_rank = (own_rank // pipeline_parallel_size) * pipeline_parallel_size
    return str(group_master_rank)


def get_group_master_info(pipeline_parallel_size):
    scheduler_prefix = os.environ['SCHEDULER_URL']
    url = f'{scheduler_prefix}get_rank_status'
    print(f"[ray_cluster_init_remote] get_group_master_info api addr: {url}")
   
    retry_cnt = 10
    worker_status = ''
    while retry_cnt:
        response = requests.get(url)
        if response.status_code == 200:
            worker_status = response.json()
            break
        else:
            print(f"[ray_cluster_init_remote] get_group_master_info fail, return code:{response.status_code}, continue...")
            retry_cnt -= 1
            time.sleep(30)

    if retry_cnt == 0:
        raise Exception(f"[ray_cluster_init_remote] get_group_master_info failed after retry")

    group_master_rank = get_group_master_rank(pipeline_parallel_size)

    group_master_info = worker_status.get(group_master_rank, None)
    if group_master_info:
        group_master_addr = group_master_info['pod_id'].split('-')[-1]
        group_master_ret_code = group_master_info['ret_code']
        return {'group_master_addr': group_master_addr,
                'group_master_ret_code': group_master_ret_code}
    else:
        return None

def get_group_master_info_block(pipeline_parallel_size, key_name):
    while True:
        gm_info = get_group_master_info(pipeline_parallel_size)
        if gm_info:
            return gm_info[key_name]


def set_gloo_net_if():
    import psutil
    import socket
    # VLLM-0.6 use gloo backend to create new_group
    net_if_addrs = psutil.net_if_addrs()
    # 去掉环回地址，否则链接不通
    net_if_addrs.pop('lo')
    if 'GLOO_SOCKET_IFNAME' in os.environ:
        if all(
                interface in net_if_addrs
                for interface in os.environ['GLOO_SOCKET_IFNAME'].split(',')
        ) and all(
                any(addr.family == socket.AddressFamily.AF_INET
                    or addr.family == socket.AddressFamily.AF_INET6
                    for addr in net_if_addrs[interface])
                for interface in os.environ['GLOO_SOCKET_IFNAME'].split(',')):
            return
    for interface, addrs in net_if_addrs.items():
        if any(addr.family == socket.AddressFamily.AF_INET
               or addr.family == socket.AddressFamily.AF_INET6
               for addr in addrs):
            if 'GLOO_SOCKET_IFNAME' in os.environ:
                print(
                    f"GLOO_SOCKET_IFNAME={os.environ['GLOO_SOCKET_IFNAME']} is not availble, "
                    f"use {interface} instead")
            os.environ['GLOO_SOCKET_IFNAME'] = interface
            return

def ray_cluster_init_remote(model_path, pipeline_parallel_size, vllm_env_setup):
    import ray
    set_gloo_net_if()
    gloo_net_if = os.getenv("GLOO_SOCKET_IFNAME", "")

    if gloo_net_if:
        os.environ['NCCL_SOCKET_IFNAME'] = gloo_net_if
        os.environ['TP_SOCKET_IFNAME'] = gloo_net_if
    else:
        raise Exception("No GLOO_SOCKET_IFNAME Found!")
    
    print(f"[ray_cluster_init] connection ifname is: {gloo_net_if}")
    
    ray_path = 'ray'
    if not command_exists("ray"):
        ray_path = '/opt/conda/envs/python3.10.13/bin/ray'

    # if os.environ['RANK'] == '0':
    if os.environ['RANK'] == get_group_master_rank(pipeline_parallel_size):
        return_code = subprocess.call([ray_path, 'start', '--head', '--port=6379', '--num-cpus=4'])
        print(f"[ray_cluster_init] ray background thread setup Return code: {return_code}")
    else:
        # master_addr = os.environ['MASTER_ADDR']
        master_addr = get_group_master_info_block(pipeline_parallel_size=pipeline_parallel_size, key_name='group_master_addr')
        print(f"[ray_cluster_init] group_master_addr is: {master_addr}")

        process = subprocess.Popen([ray_path, 'start', '--block', f'--address={master_addr}:6379', '--num-cpus=4'])
        from openlm_hub import repo_download
        repo_download(model_path)
        vllm_env_setup()
        process.wait()

        try:
            master_ret_code = get_group_master_info_block(pipeline_parallel_size=pipeline_parallel_size, key_name='group_master_ret_code')
            assert master_ret_code == 0, f'group master ret code is [{master_ret_code}], not 0 ,reboot'
        except requests.exceptions.ConnectionError:
            print(f"[ray_cluster_init] get_group_master_info from scheduler meet ConnectionError, Maybe scheduler is exit because job finish, pass")
            pass

        exit(0)
      
    ray.init()
    while len(ray.nodes()) != pipeline_parallel_size:
        time.sleep(5)
        print('[ray cluster init] wait for worker node connect')
    print('[ray_cluster_init] all worker online, connect success!')

def ray_cluster_init_local(infer_args):
    set_gloo_net_if()
    import ray
    ray.init(address=None, ignore_reinit_error=True, num_gpus=torch.cuda.device_count(), num_cpus=4)
