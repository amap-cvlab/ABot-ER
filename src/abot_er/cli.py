"""Explicit commands: verify data, prepare portable paths, and launch sealed SFT."""
import argparse
import json
import os
from pathlib import Path
import sys

from .reproduction import ROOT, prepare_release, read_json, render_recipe, verify_release, write_json


def main():
    parser = argparse.ArgumentParser(prog="abot-er")
    sub = parser.add_subparsers(dest="command", required=True)
    from .task_planning import add_commands
    add_commands(sub)
    download = sub.add_parser("download")
    from .artifacts import add_arguments as artifact_arguments
    artifact_arguments(download)
    verify = sub.add_parser("verify-data")
    verify.add_argument("--data", required=True)
    verify.add_argument("--manifest", required=True)
    public = sub.add_parser("verify-public-data", help="Verify the pinned ModelScope JSONL export")
    public.add_argument("--data", required=True, help="Downloaded repository root, not the stage directory")
    public.add_argument("--stage", required=True, choices=("stage_a_pretrain", "stage_b_posttrain"))
    public.add_argument("--output", required=True, help="New local manifest outside the download directory")
    from .public_data import run as verify_public
    public.set_defaults(handler=verify_public)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--media-roots", required=True)
    prepare.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-eqa")
    from .evaluation import add_arguments
    add_arguments(evaluate)
    train = sub.add_parser("train")
    train.add_argument("--recipe", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "32")))
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--public-data", action="store_true",
                       help="Use the pinned public export, explicitly not a byte-identical historical replay")
    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    elif args.command == "download":
        from .artifacts import download
        download(args)
    elif args.command == "verify-data":
        print(json.dumps(verify_release(args.data, args.manifest), indent=2))
    elif args.command == "prepare-data":
        result = prepare_release(args.data, args.manifest, args.output, args.media_roots)
        print(json.dumps({k: result[k] for k in ("release_id", "physical_records", "ordered_sha256")}, indent=2))
    elif args.command == "evaluate-eqa":
        from .evaluation import run
        run(args)
    else:
        cfg, summary = render_recipe(args.recipe, args.model, args.data, args.output, args.world_size,
                                     public_data=args.public_data)
        if args.dry_run:
            print(json.dumps(summary, indent=2))
            return
        if int(os.environ.get("WORLD_SIZE", "1")) != args.world_size:
            raise ValueError("Launch training with torchrun matching --world-size")
        if os.environ.get("TP_MOS_OUTPUT_URI") or os.environ.get("_NEBULA_MODEL"):
            raise ValueError("Portable entry point requires local checkpoint output, not inherited MOS auto-resume")
        out = Path(args.output).resolve()
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Use a new empty run directory; implicit resume is forbidden")
        # Every rank validates before any rank writes the run record.
        import torch.distributed as dist
        if args.world_size > 1 and not dist.is_initialized():
            import torch
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend="nccl")
        preflight_error = [None]
        if int(os.environ.get("RANK", "0")) == 0:
            try:
                verify_release(Path(args.data) / "shards", Path(args.data) / "manifest.json")
                write_json(out / "reproduction_manifest.json", summary)
            except Exception as exc:
                preflight_error[0] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(preflight_error, src=0)
        if preflight_error[0]:
            raise RuntimeError(f"Data preflight failed on rank 0: {preflight_error[0]}")
        os.environ["ABOT_ENABLE_MOS"] = "0"
        os.environ["ENABLE_C4D_PERF_TRACKER"] = "0"
        sys.path.insert(0, str(ROOT / "third_party/tuningfactory/src"))
        from llmtuner import run_exp
        run_exp(cfg)


if __name__ == "__main__":
    main()
