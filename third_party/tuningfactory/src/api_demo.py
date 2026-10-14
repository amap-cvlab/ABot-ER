raise RuntimeError("Legacy API publishing is disabled; use abot-er serve-tp on loopback only.")

import os
import sys

import psutil
import uvicorn

from llmtuner import ChatModel, create_app
from llmtuner.extras.logging import get_logger
from llmtuner.webui.publish_chat import PublishManager


logger = get_logger(__name__)


def is_port_in_use(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(("localhost", port))
        return result == 0


def main():
    chat_model = ChatModel()
    publish_manager = PublishManager(timeout_minute=chat_model.app_timeout)
    app = create_app(chat_model, root_path=publish_manager.get_root_path())
    publish_manager.register_timeout_shutdown()
    while True:
        port = publish_manager.get_random_port()
        if is_port_in_use(port):
            logger.info(f"{port} is used,retry other port......")
        else:
            break
    publish_manager.publish(port)
    uvicorn.run(app, host="0.0.0.0", port=int(port), workers=1)
    sys.exit(0)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ENABLE_FLOPS_STATISTICS"] = "-1"
    os.environ["MODEL_PARAMS_THOP"] = "-1"
    ifaddrs = list(psutil.net_if_addrs().keys())
    os.environ["GLOO_SOCKET_IFNAME"] = ifaddrs[0]
    main()
