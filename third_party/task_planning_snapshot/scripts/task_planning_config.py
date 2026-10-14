#!/usr/bin/env python
"""Portable path configuration shared by local and distributed evaluation scripts."""

from __future__ import annotations

import os
# Platform sitecustomize writes uncaught Python exceptions to cwd unless this
# flag is exactly '1'. Apply before path/parameter validation can raise, and
# inherit it in every evaluator/model subprocess (including EB and retries).
os.environ["NOTEBOOK_CONTAINER"] = "1"

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARD_WORKERS_PER_GPU = int(os.environ.get("TP_SHARD_WORKERS_PER_GPU", "10"))
DEFAULT_ET_COMMAND_TIMEOUT = int(os.environ.get("TP_ET_COMMAND_TIMEOUT", "900"))
DEFAULT_ALFRED_SHARD_TIMEOUT = int(os.environ.get("TP_ALFRED_SHARD_TIMEOUT", "2400"))
DEFAULT_HABITAT_SHARD_TIMEOUT = int(os.environ.get("TP_HABITAT_SHARD_TIMEOUT", "1200"))


def _absolute_env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path, got: {path}")
    return path


NAS_PROJECT_ROOT = _absolute_env_path("TP_NAS_PROJECT_ROOT", PROJECT_ROOT)
DEPS_ROOT = _absolute_env_path("TP_DEPS_ROOT", NAS_PROJECT_ROOT.parent / "tp-deps")
OUTPUT_ROOT = _absolute_env_path("TP_OUTPUT_ROOT", NAS_PROJECT_ROOT / "outputs")
LOCAL_SCRATCH_ROOT = _absolute_env_path(
    "TP_LOCAL_SCRATCH_ROOT", Path("/tmp/task_planning_eval")
)

ET_SIMULATOR_ROOT = _absolute_env_path(
    "TP_ET_SIMULATOR_ROOT", DEPS_ROOT / "ET-Plan-Bench" / "simulator"
)
AI2THOR_RELEASES_ROOT = _absolute_env_path(
    "TP_AI2THOR_RELEASES_ROOT", DEPS_ROOT / "AI2THOR" / "releases"
)
EB_PYTHON_OVERLAY = _absolute_env_path(
    "TP_EB_PYTHON_OVERLAY", DEPS_ROOT / "python_runtime" / "eb_py310_site"
)
ML_TRACKER_WHEEL = _absolute_env_path(
    "TP_ML_TRACKER_WHEEL",
    DEPS_ROOT / "python_wheels" / "ml_tracker-0.1.37.dev0-py3-none-any.whl",
)

EB_ALFRED_DATA_ROOT = _absolute_env_path(
    "TP_EB_ALFRED_DATA_ROOT",
    NAS_PROJECT_ROOT / "EmbodiedBench" / "embodiedbench" / "envs" / "eb_alfred" / "data",
)
EB_HABITAT_DATA_ROOT = _absolute_env_path(
    "TP_EB_HABITAT_DATA_ROOT",
    NAS_PROJECT_ROOT / "EmbodiedBench" / "embodiedbench" / "envs" / "eb_habitat" / "data",
)
EB_HABITAT_DATASETS_ROOT = _absolute_env_path(
    "TP_EB_HABITAT_DATASETS_ROOT",
    NAS_PROJECT_ROOT / "EmbodiedBench" / "embodiedbench" / "envs" / "eb_habitat" / "datasets",
)
EB_NAVIGATION_DATASETS_ROOT = _absolute_env_path(
    "TP_EB_NAVIGATION_DATASETS_ROOT",
    NAS_PROJECT_ROOT / "EmbodiedBench" / "embodiedbench" / "envs" / "eb_navigation" / "datasets",
)


JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_job_name(job_name: str) -> str:
    """Return a safe single-component global job name."""
    if not JOB_NAME_PATTERN.fullmatch(job_name):
        raise ValueError(
            "job name must match [A-Za-z0-9][A-Za-z0-9._-]* "
            f"and cannot contain path separators, got: {job_name!r}"
        )
    return job_name


def job_output_dir(job_name: str) -> Path:
    return OUTPUT_ROOT / validate_job_name(job_name)


def job_scratch_dir(job_name: str) -> Path:
    """Return node-local, disposable runtime storage for one evaluation."""
    return LOCAL_SCRATCH_ROOT / validate_job_name(job_name)


def resolve_shard_workers(
    gpu_count: int,
    *,
    workers_per_gpu: int = DEFAULT_SHARD_WORKERS_PER_GPU,
    override: int | None = None,
) -> int:
    """Resolve simulator/shard concurrency from GPU count unless explicitly overridden."""
    if gpu_count <= 0:
        raise ValueError(f"gpu_count must be positive, got: {gpu_count}")
    if workers_per_gpu <= 0:
        raise ValueError(f"workers_per_gpu must be positive, got: {workers_per_gpu}")
    if override is not None:
        if override <= 0:
            raise ValueError(f"worker override must be positive, got: {override}")
        return override
    return gpu_count * workers_per_gpu
