#!/usr/bin/env python3
"""Launch the reference PhyStrata training recipe on one or more GPU nodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def write_runtime_file(path: Path, content: str, node_rank: int) -> None:
    temporary = path.with_name(f"{path.name}.rank{node_rank}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def existing_path(raw: str, kind: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if kind == "file" and not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if kind == "dir" and not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("2b", "5b"), required=True)
    parser.add_argument("--model-path", required=True, type=lambda value: existing_path(value, "dir"))
    parser.add_argument("--dataset-csv", required=True, type=lambda value: existing_path(value, "file"))
    parser.add_argument("--video-root", required=True, type=lambda value: existing_path(value, "dir"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nnodes < 1 or args.nproc_per_node < 1 or not 0 <= args.node_rank < args.nnodes:
        raise SystemExit("Invalid node count, processes per node, or node rank")
    if args.nnodes > 1 and args.master_addr in {"localhost", "127.0.0.1"}:
        raise SystemExit("Multi-node training requires --master-addr reachable from all nodes")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    if not args.dry_run and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise SystemExit(f"Checkpoint directory is not empty; choose a new --output-dir: {checkpoint_dir}")
    template = ROOT / f"train/configs/phystrata_{args.size}_openvid64k.yaml"
    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    dataset_template = ROOT / f"train/configs/datasets/openvid/openvid64k_{args.size}.template.json"
    dataset = json.loads(dataset_template.read_text(encoding="utf-8"))
    dataset["datasets"][0]["dataset_file"] = str(args.dataset_csv)
    dataset["datasets"][0]["data_root"] = str(args.video_root)
    runtime_dataset = output / "dataset_config.json"
    write_runtime_file(runtime_dataset, json.dumps(dataset, indent=2) + "\n", args.node_rank)

    cfg.update({
        "num_gpus": args.nnodes * args.nproc_per_node,
        "eval_num_gpus": args.nnodes * args.nproc_per_node,
        "pretrained_model_name_or_path": str(args.model_path),
        "pretrained_model_path": str(args.model_path),
        "dataset_config": str(runtime_dataset),
        "output_dir": str(output / "checkpoints"),
        "checkpoint_path": str(output / "checkpoints"),
        "eval_checkpoint_path": str(output / "checkpoints" / f"weight_{cfg['train_steps']}"),
        "result_path": str(output / "eval"),
        "logging_dir": str(output / "logs"),
    })
    if cfg["variant"] != "phystrata" or cfg["model_type"] != "phystrata":
        raise RuntimeError("Training config has an unexpected model type")
    expected_batch_size = 4 if args.size == "2b" else 2
    if cfg["batch_size"] != expected_batch_size or cfg["checkpointing_steps"] != 500:
        raise RuntimeError(
            f"Reference {args.size} recipe must keep batch size {expected_batch_size} "
            "and checkpoint interval 500"
        )
    runtime_config = output / "resolved_config.yaml"
    write_runtime_file(runtime_config, yaml.safe_dump(cfg, sort_keys=False), args.node_rank)

    env = os.environ.copy()
    env["CLUSTER_TRAIN_EVAL_CONFIG"] = json.dumps(cfg)
    env["PHYSTRATA_DRIFT"] = "1"
    env["WANDB_MODE"] = "online" if args.wandb_project else "disabled"
    if args.wandb_project:
        env["WANDB_PROJECT"] = args.wandb_project
        env["WANDB_NAME"] = cfg["name"]
        env["WANDB_DIR"] = str(output / "logs")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nnodes={args.nnodes}", f"--nproc_per_node={args.nproc_per_node}",
        f"--node_rank={args.node_rank}", f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}", "-m", "train.cluster_train_eval",
    ]
    print(f"Config: {runtime_config}")
    print(f"Global batch: {cfg['batch_size'] * args.nnodes * args.nproc_per_node}")
    print("Launch:", " ".join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
