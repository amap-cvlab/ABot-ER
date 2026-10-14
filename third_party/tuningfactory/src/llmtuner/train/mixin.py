import os
import shutil

import torch.distributed as dist
from ..extras.optional_mos import get_mos_version_uri_in_mdl_task, push_to_mos, register_ckpt, OpenlmHubClientError

from ..extras.logging import get_logger


logger = get_logger(__name__)


def do_nothing(*args, **kwargs):
    pass


try:
    from transformers.nebula_hub_utils import NebulaHubUtils

    NebulaHubUtils().trainer_on_save = do_nothing
except ImportError:
    pass


class OpenlmHubUtils:
    def __init__(self):
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_process_rank = int(os.getenv("LOCAL_PROCESS_RANK", "0"))
        if local_process_rank > 0:
            self.local_rank = local_process_rank

        self.mos_uri = os.environ.get("TP_MOS_OUTPUT_URI") if os.environ.get("ABOT_ENABLE_MOS") == "1" else None
        if self.mos_uri:
            logger.info(f"Use explicit task-planning MOS output URI: {self.mos_uri}")
        else:
            try:
                self.mos_uri = get_mos_version_uri_in_mdl_task()
            except OpenlmHubClientError as e:
                logger.info(f"Didn't set _NEBULA_MODEL, skip upload to MOS! reason: {e}")
                self.mos_uri = None

    def save_local_model_to_mos(
        self, ckpt_id: str, local_state_path: str, keep_local_file: bool = False, metrics: dict = None
    ) -> None:
        if self.mos_uri is None:
            logger.info("Didn't set _NEBULA_MODEL, skip upload to MOS!")
            return

        if dist.is_initialized():
            dist.barrier()

        # Full HF checkpoints live on shared NAS. Upload and delete once globally;
        # using local_rank here would start one duplicate upload per node.
        push_result = None
        if self.rank == 0 and os.path.exists(local_state_path) and len(os.listdir(local_state_path)) != 0:
            push_result = push_to_mos(local_state_path, checkpoint_id=ckpt_id, mos_version_uri=self.mos_uri, metrics=metrics, register_ckpt=False, use_subprocess=True)

        if dist.is_initialized():
            dist.barrier()

        if self.rank == 0 and push_result:
            register_ckpt(push_result)

        if self.rank == 0 and os.path.exists(local_state_path) and not keep_local_file:
            if os.path.isdir(local_state_path):
                shutil.rmtree(local_state_path, ignore_errors=True)
            else:
                os.remove(local_state_path)

        if dist.is_initialized():
            dist.barrier()
