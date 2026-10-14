#!/usr/bin/env python
"""Collect simulator scratch files inside one job output directory."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from task_planning_config import job_output_dir


UUID_DIR_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def current_job_name(args: argparse.Namespace) -> str:
    return args.run_id


def active_unity_ports() -> set[int]:
    try:
        ps = subprocess.check_output(["ps", "-eo", "args="], text=True, errors="ignore")
    except Exception:
        return set()
    return {int(port) for port in re.findall(r"-http-port=?(\d+)", ps)}


def has_unity_process() -> bool:
    try:
        ps = subprocess.check_output(["ps", "-eo", "stat=,args="], text=True, errors="ignore")
    except subprocess.CalledProcessError:
        return False
    for line in ps.splitlines():
        line = line.strip()
        if not line or "linux_exec" not in line:
            continue
        stat = line.split(maxsplit=1)[0]
        if not stat.startswith("Z"):
            return True
    return False


def active_uuid_dirs(source: Path) -> set[str]:
    active: set[str] = set()
    uuid_names = set(uuid_dir_names(source))
    if not uuid_names:
        return active

    try:
        pid_output = subprocess.check_output(["pgrep", "-f", "linux_exec"], text=True, errors="ignore")
    except subprocess.CalledProcessError:
        return active

    for pid in pid_output.splitlines():
        if not pid.isdigit():
            continue
        proc = Path("/proc") / pid
        cmdline_path = proc / "cmdline"
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"linux_exec" not in cmdline and b"Unity" not in cmdline:
            continue

        paths_to_check: list[Path] = []
        for link in [proc / "cwd", proc / "root"]:
            try:
                paths_to_check.append(Path(os.readlink(link)).resolve())
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                pass
        fd_dir = proc / "fd"
        try:
            fd_entries = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            fd_entries = []
        for fd in fd_entries:
            try:
                paths_to_check.append(Path(os.readlink(fd)).resolve())
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue

        for candidate in paths_to_check:
            for part in candidate.parts:
                if part in uuid_names:
                    active.add(part)
    return active


def uuid_dir_names(source: Path) -> list[str]:
    names: list[str] = []
    if not source.exists():
        return names
    with os.scandir(source) as entries:
        for entry in entries:
            if not UUID_DIR_RE.match(entry.name):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    names.append(entry.name)
            except FileNotFoundError:
                continue
    return names


def move_file(path: Path, dest_dir: Path, dry_run: bool, prepared_dirs: set[Path]) -> bool:
    if dest_dir not in prepared_dirs:
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
        prepared_dirs.add(dest_dir)
    target = dest_dir / path.name
    if dry_run:
        return True
    shutil.move(str(path), str(target))
    return True


def iter_exception_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in source.rglob("*.excp"):
        try:
            path.resolve()
        except FileNotFoundError:
            continue
        if ".git" in path.parts:
            continue
        files.append(path)
    return files


def collect(args: argparse.Namespace) -> None:
    run_id = current_job_name(args)
    root = job_output_dir(run_id)
    source = root / "runtime" / "et_workdir"
    dest = root / "runtime_artifacts"
    active_ports = active_unity_ports()
    active_uuids: set[str] = set()
    prepared_dirs: set[Path] = set()
    moved = {"port_logs": 0, "player_logs": 0, "uuid_dirs": 0, "output_dirs": 0}
    deleted = {"exceptions": 0}
    skipped = {"active_ports": 0, "future_ports": 0, "uuid_dirs": 0, "output_dirs": 0}

    for path in source.glob("port_*.txt"):
        match = re.match(r"port_(\d+)\.txt$", path.name)
        port = int(match.group(1)) if match else None
        if port in active_ports:
            skipped["active_ports"] += 1
            continue
        if args.keep_ports_from is not None and port is not None and port >= args.keep_ports_from:
            skipped["future_ports"] += 1
            continue
        if move_file(path, dest / "port_logs", args.dry_run, prepared_dirs):
            moved["port_logs"] += 1

    for path in source.glob("Player_*.log"):
        match = re.match(r"Player_(\d+)\.log$", path.name)
        port = int(match.group(1)) if match else None
        if port in active_ports:
            skipped["active_ports"] += 1
            continue
        if args.keep_ports_from is not None and port is not None and port >= args.keep_ports_from:
            skipped["future_ports"] += 1
            continue
        if move_file(path, dest / "player_logs", args.dry_run, prepared_dirs):
            moved["player_logs"] += 1

    for path in iter_exception_files(source):
        if not args.dry_run:
            path.unlink(missing_ok=True)
        deleted["exceptions"] += 1

    output_dir = source / "Output"
    if output_dir.exists() and output_dir.is_dir():
        if has_unity_process() and not args.force_output_dir:
            skipped["output_dirs"] += 1
        else:
            output_dest = dest / "root_output_dirs"
            if not args.dry_run:
                output_dest.mkdir(parents=True, exist_ok=True)
                target = output_dest / "Output"
                if target.exists():
                    target = output_dest / f"Output_{int(time.time())}"
                shutil.move(str(output_dir), str(target))
            moved["output_dirs"] += 1

    if args.include_uuid_dirs:
        now = time.time()
        active_uuids = active_uuid_dirs(source)
        uuid_dest = dest / "uuid_dirs"
        if not args.dry_run:
            uuid_dest.mkdir(parents=True, exist_ok=True)
        for name in uuid_dir_names(source):
            if args.max_uuid_dirs is not None and moved["uuid_dirs"] >= args.max_uuid_dirs:
                skipped["uuid_dirs"] += 1
                continue
            path = source / name
            if path.name in active_uuids and not args.force_uuid_dirs:
                skipped["uuid_dirs"] += 1
                continue
            if not args.force_uuid_dirs and args.uuid_min_age_sec > 0:
                try:
                    age_sec = now - path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age_sec < args.uuid_min_age_sec:
                    skipped["uuid_dirs"] += 1
                    continue
            if not args.dry_run:
                shutil.move(str(path), str(uuid_dest / path.name))
            moved["uuid_dirs"] += 1
            if args.progress_every and moved["uuid_dirs"] % args.progress_every == 0:
                print(f"moved uuid_dirs={moved['uuid_dirs']}", flush=True)

    print(
        {
            "job_name": run_id,
            "run_id": run_id,
            "artifact_dir": str(dest),
            "active_ports": sorted(active_ports),
            "active_uuid_dirs": sorted(active_uuids) if args.include_uuid_dirs else [],
            "uuid_min_age_sec": args.uuid_min_age_sec,
            "moved": moved,
            "deleted": deleted,
            "skipped": skipped,
            "dry_run": args.dry_run,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", dest="run_id", required=True)
    parser.add_argument("--keep-ports-from", type=int)
    parser.add_argument("--include-uuid-dirs", action="store_true")
    parser.add_argument("--force-uuid-dirs", action="store_true")
    parser.add_argument("--uuid-min-age-sec", type=int, default=300)
    parser.add_argument("--max-uuid-dirs", type=int)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--force-output-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
