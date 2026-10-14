import hashlib
import re
import os
import json

from ..extras.logging import get_logger

logger = get_logger(__name__)

try:
    import oss2
except ImportError:
    oss2 = None


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def get_mdl_config_value():
    """
    读取 Linux 1 号进程的启动命令，提取 --config= 后的值。
    """
    with open("/proc/1/cmdline", "rb") as f:
        cmdline = f.read()

    # /proc/<pid>/cmdline 中参数以 \0 分隔
    cmd_str = cmdline.replace(b"\x00", b" ").decode("utf-8", errors="replace")

    match = re.search(r"--config=(\S+)", cmd_str)
    if match:
        return match.group(1)
    return None


def read_oss_config_info():
    
    config = get_mdl_config_value()

    if os.getenv("NOTEBOOK_CONTAINER", "0") == "1":
        config = os.path.join("/checkpoint/binary/train_package", config)

    with open(config) as f:
        config_dict = json.load(f)
        
    if "oss_access_id" in config_dict and "oss_access_key" in config_dict and "oss_bucket" in config_dict and "oss_endpoint" in config_dict:
        oss_access_id = config_dict["oss_access_id"].split(',')
        oss_access_key = config_dict["oss_access_key"].split(',')
        oss_bucket = config_dict["oss_bucket"].split(',')
        oss_endpoint = config_dict["oss_endpoint"].split(',')
        oss_info_list = list(zip(oss_bucket, oss_access_id, oss_access_key, oss_endpoint))
        oss_info = {item[0]: item for item in oss_info_list}
        return oss_info
    else:
        return None

def create_oss2_downloader():
    if os.environ.get("ABOT_ENABLE_MOS") != "1":
        return None
    oss_info = read_oss_config_info()
    if oss_info:
        if oss2 is None:
            logger.warning("OSS2 is not installed. Please install it using pip install oss2. Could not create OSS2Downloader.")
        else:
            return OSS2Downloader(oss_info)
    return None


class OSS2Downloader(object):
    def __init__(self, oss_info):
        self.oss_info = oss_info
        self.oss_bucket = None

        cache_size = 100
        # 缓存队列：存储 (file_path, local_path) 的顺序
        self.cached_files = []
        # 缓存字典：快速查找 file_path -> local_path
        self.cache_dict = {}
        # 缓存队列大小限制
        self.cache_size = cache_size

        self.oss_buckets = {}
    
    def create_oss_bucket(self, oss_access_id, oss_access_key, oss_bucket, oss_endpoint):
        auth = oss2.Auth(oss_access_id, oss_access_key)
        bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket)
        self.oss_buckets[oss_bucket] = bucket
    
    def get_oss_bucket_name(self, file_path: str):
        assert file_path.startswith("oss://"), f"OSS bucket {file_path} not found in oss_info"

        bucket_name = file_path[len("oss://"):]
        bucket_name = bucket_name.split("/")[0]

        if bucket_name not in self.oss_buckets:
            assert bucket_name in self.oss_info, f"OSS bucket {bucket_name} not found in oss_info"
            oss_bucket = self.oss_info[bucket_name][0]
            oss_access_id = self.oss_info[bucket_name][1]
            oss_access_key = self.oss_info[bucket_name][2]
            oss_endpoint = self.oss_info[bucket_name][3]
            self.create_oss_bucket(oss_access_id, oss_access_key, oss_bucket, oss_endpoint)

        return bucket_name
    
    def get_oss_bucket(self, bucket_name: str):
        return self.oss_buckets[bucket_name]


    def download(self, file_path):
        if not isinstance(file_path, str):
            return file_path
        # 没有video folder配置就是oss2 sdk
        if file_path.startswith("oss://"):
            # 检查缓存
            if file_path in self.cache_dict:
                return self.cache_dict[file_path]

            file_extension = os.path.splitext(file_path)[1]
            local_path = f"/checkpoint/binary/train_package/{md5(file_path)}{file_extension}"

            bucket_name = self.get_oss_bucket_name(file_path)
            oss_sdk_path = file_path[len(f"oss://{bucket_name}/"):]
            self.get_oss_bucket(bucket_name).get_object_to_file(oss_sdk_path, local_path)
            
            # 添加到缓存
            self._add_to_cache(file_path, local_path)
            
            return local_path
        else:
            return file_path

    def _add_to_cache(self, file_path, local_path):
        """将文件路径添加到缓存队列"""
        # 如果缓存已满，移除最旧的项
        if len(self.cached_files) >= self.cache_size:
            oldest_file_path, oldest_local_path = self.cached_files.pop(0)
            del self.cache_dict[oldest_file_path]
            # 删除本地缓存文件
            if os.path.exists(oldest_local_path):
                try:
                    os.remove(oldest_local_path)
                except Exception as e:
                    # 删除失败不影响主流程，记录日志即可
                    logger.warning(f"Failed to remove cached file {oldest_local_path}: {e}")
        
        # 添加新项到队列末尾
        self.cached_files.append((file_path, local_path))
        self.cache_dict[file_path] = local_path      
