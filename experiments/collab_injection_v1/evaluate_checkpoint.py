#!/usr/bin/env python3
"""Evaluate a saved GraphLoRA checkpoint on a chosen split without training.

The checkpoint's saved config is reused, model.ckpt loads its model weights,
and run.evaluate=True routes the requested split through the existing
evaluation-only runner. Default split is validation, not test.
"""
import argparse
import copy
import subprocess
import sys
from pathlib import Path

import torch
import yaml


DEFAULT_CHECKPOINT = (
    "/home/bbx/user_lx/GraphLoRA/logs/"
    "S2_STRICT_20261017_143611/L31_output/20261017143/"
    "checkpoint_best.pth"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-dir", default="output/S2_L31_HISTORICAL_EVAL")
    parser.add_argument("--config-out", default="tran_configs/collab_v1/eval/S2_L31_HISTORICAL_EVAL.yaml")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("config"), dict):
        raise TypeError("Expected a checkpoint containing a dict at checkpoint['config'].")
    cfg = copy.deepcopy(ckpt["config"])
    if "model" not in cfg or "run" not in cfg:
        raise KeyError("Saved checkpoint config must contain 'model' and 'run'.")

    cfg["model"]["ckpt"] = str(checkpoint_path)
    cfg["model"]["collab_mode"] = cfg["model"].get("collab_mode", "legacy")
    run = cfg["run"]
    run["evaluate"] = True
    run["valid_splits"] = [args.split]
    run["test_splits"] = [args.split]
    run["resume_ckpt_path"] = None
    run["output_dir"] = str((root / args.output_dir).resolve())
    run.pop("mid_epoch_eval_iters", None)
    run.pop("skip_validation", None)

    config_path = root / args.config_out
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[EVAL-CHECKPOINT] checkpoint={checkpoint_path}", flush=True)
    print(f"[EVAL-CHECKPOINT] split={args.split}", flush=True)
    print(f"[EVAL-CHECKPOINT] config={config_path}", flush=True)
    print("[EVAL-CHECKPOINT] training disabled; optimizer state will not be resumed.", flush=True)

    command = [
        sys.executable,
        "train_collm_mf_din.py",
        "--cfg-path",
        str(config_path),
    ]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
