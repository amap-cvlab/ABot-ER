"""Atomic result publishing and cross-node run leases for NAS outputs."""

from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Publish a complete file with an atomic rename in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with temp_path.open("x", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


class RunLease:
    """An atomic-mkdir lease that prevents two hosts from writing one run id."""

    def __init__(
        self,
        lock_root: Path,
        run_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        heartbeat_sec: float = 15.0,
        stale_sec: float = 900.0,
    ) -> None:
        if heartbeat_sec <= 0 or stale_sec <= heartbeat_sec:
            raise ValueError("run lease requires 0 < heartbeat_sec < stale_sec")
        self.lock_root = lock_root
        self.run_id = run_id
        self.lock_dir = lock_root / f"{run_id}.lock"
        self.owner_id = uuid.uuid4().hex
        self.owner_path = self.lock_dir / "owner.json"
        self.heartbeat_path = self.lock_dir / f"heartbeat.{self.owner_id}"
        self.heartbeat_sec = heartbeat_sec
        self.stale_sec = stale_sec
        self.metadata = metadata or {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._acquired = False

    def _owner_payload(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "run_id": self.run_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at_epoch": time.time(),
            **self.metadata,
        }

    def _read_owner(self) -> dict[str, Any]:
        try:
            return json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _heartbeat_age(self, owner: dict[str, Any]) -> float:
        owner_id = owner.get("owner_id")
        heartbeat = self.lock_dir / f"heartbeat.{owner_id}" if owner_id else self.lock_dir
        try:
            return max(0.0, time.time() - heartbeat.stat().st_mtime)
        except OSError:
            return float("inf")

    def acquire(self) -> "RunLease":
        self.lock_root.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.lock_dir.mkdir()
            except FileExistsError:
                owner = self._read_owner()
                age = self._heartbeat_age(owner)
                if age <= self.stale_sec:
                    raise RuntimeError(
                        "evaluation run is already active: "
                        f"run_id={self.run_id!r}, owner={owner}, heartbeat_age_sec={age:.1f}"
                    )
                stale_path = self.lock_root / f"{self.lock_dir.name}.stale.{uuid.uuid4().hex}"
                try:
                    self.lock_dir.rename(stale_path)
                except FileNotFoundError:
                    continue
                shutil.rmtree(stale_path, ignore_errors=True)
                continue

            try:
                atomic_write_json(self.owner_path, self._owner_payload())
                self.heartbeat_path.touch()
            except Exception:
                shutil.rmtree(self.lock_dir, ignore_errors=True)
                raise
            self._acquired = True
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"run-lease-{self.run_id}",
                daemon=True,
            )
            self._thread.start()
            return self

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_sec):
            if self._read_owner().get("owner_id") != self.owner_id:
                return
            try:
                self.heartbeat_path.touch()
            except OSError:
                continue

    def release(self) -> None:
        if not self._acquired:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.heartbeat_sec * 2, 1.0))
        self._acquired = False
        if self._read_owner().get("owner_id") != self.owner_id:
            return
        released_path = self.lock_root / f"{self.lock_dir.name}.released.{uuid.uuid4().hex}"
        try:
            self.lock_dir.rename(released_path)
        except FileNotFoundError:
            return
        shutil.rmtree(released_path, ignore_errors=True)

    def __enter__(self) -> "RunLease":
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()
