import json
import os

MODEL_NAME = ""
ARCHITECTURES = ""

def set_model_name(model_name):
    global MODEL_NAME	
    MODEL_NAME = model_name


def set_architectures(config):
    global ARCHITECTURES	
    _arch = ''
    if hasattr(config, 'architectures') and config.architectures:
        if isinstance(config.architectures, list):
            _arch = config.architectures[0]
        if isinstance(config.architectures, str):
            _arch = config.architectures
    ARCHITECTURES = _arch

def create_mdl_metrics_logger():
    def setup_safe_logger(
        name='MDLMetricLogger',
        log_file='/var/log/mdl_metrics.log',
        max_bytes=100 * 1024 * 1024,  # 100MB
        backup_count=5,
        encoding='utf-8'
    ):
        """
        创建一个进程安全的日志记录器
        :param name: logger 名称
        :param log_file: 日志文件路径
        :param max_bytes: 单个日志文件最大大小（用于轮转）
        :param backup_count: 保留的旧日志文件数量
        :param encoding: 文件编码
        :return: logger 实例
        """
        import logging
        from concurrent_log_handler import ConcurrentRotatingFileHandler
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 创建 logger
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger  # 避免重复添加 handler

        logger.setLevel(logging.DEBUG)

        # 使用 ConcurrentRotatingFileHandler（进程安全 + 自动轮转）
        file_handler = ConcurrentRotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            # delay=True  # 可选：延迟创建文件
        )

        # 设置日志格式
        formatter = logging.Formatter(
            fmt='%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # 添加 handler
        logger.addHandler(file_handler)

        return logger
    return setup_safe_logger(log_file='/var/log/mdl_metrics.log')



class MDLMetricLogger():
    def __init__(self, log_type='TuningFactory', log_func='statistics'):
        self.logger = create_mdl_metrics_logger()
        self.log_type = log_type
        self.log_func = log_func
        self.app_id = os.environ['APP_ID']

    def record_tokens(self, query_cnt, input_tokens, output_tokens):
        if input_tokens is None or output_tokens is None:
            return
        written_token_dict = {"log_type": "TuningFactory", "log_func": "statistics", "model_name": MODEL_NAME, "architectures": ARCHITECTURES, "app_id": self.app_id, "query_cnt":query_cnt, "input_tokens": input_tokens, "output_tokens": output_tokens}
        self.logger.info(json.dumps(written_token_dict, ensure_ascii=False))