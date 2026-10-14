"""Explicitly opt-in corporate storage; portable runs never import its SDK."""
import os


class OpenlmHubClientError(RuntimeError):
    pass


class MosCkptNotFoundError(OpenlmHubClientError):
    pass


class MosVersionNotFoundError(OpenlmHubClientError):
    pass


def _disabled(*args, **kwargs):
    raise OpenlmHubClientError("MOS is disabled. Use local model/output paths, or explicitly configure the internal integration.")


get_last_ckpt_by_uri = get_version_info = repo_download = push_to_mos = register_ckpt = _disabled


def get_mos_version_uri_in_mdl_task():
    return None


if os.environ.get("ABOT_ENABLE_MOS") == "1":
    from openlm_hub import (
        get_last_ckpt_by_uri, get_mos_version_uri_in_mdl_task,
        get_version_info, repo_download, push_to_mos, register_ckpt,
    )
    from openlm_hub.error import MosCkptNotFoundError, MosVersionNotFoundError, OpenlmHubClientError
