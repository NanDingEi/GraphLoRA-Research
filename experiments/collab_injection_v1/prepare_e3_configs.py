#!/usr/bin/env python3
"""Build matched 3-epoch configs from the actual S2-L31 checkpoint config.

This deliberately copies the checkpoint's saved config rather than using a
reconstructed YAML. Model weights are NOT loaded for training: model.ckpt and
run.resume_ckpt_path are cleared for every experiment.
"""
import argparse
import copy
from pathlib import Path

import torch
import yaml


DEFAULT_CHECKPOINT = (
    "/home/bbx/user_lx/GraphLoRA/logs/"
    "S2_STRICT_20261017_143611/L31_output/20261017143/"
    "checkpoint_best.pth"
)
EXPERIMENTS = (
    ("S2_L31_CONTROL", "legacy", 0.1),
    ("S2_GATED_L31", "gated_l31", 0.1),
    ("S2_CONDITIONAL_LORA", "conditional_lora", 0.0),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        default="tran_configs/collab_v1/e3",
        help="Directory for generated YAML configs",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Champion checkpoint not found: {checkpoint_path}\n"
            "Pass --checkpoint with the actual S2-L31 checkpoint path."
        )

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain a top-level 'config'.")

    saved = checkpoint["config"]
    if not isinstance(saved, dict) or "model" not in saved or "run" not in saved:
        raise TypeError(
            "checkpoint['config'] is not a model/run dictionary; "
            f"type={type(saved).__name__}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path.cwd().resolve()

    for name, mode, scale in EXPERIMENTS:
        cfg = copy.deepcopy(saved)
        model = cfg["model"]
        run = cfg["run"]

        model["ckpt"] = None
        model["collab_mode"] = mode
        model["gnn_scale"] = scale
        model["gcn_layer"] = 31
        model["rel_inject"] = False
        model["rel_rank_mod"] = False

        run["seed"] = 42
        run["max_epoch"] = 3
        run["accum_grad_iters"] = int(run.get("accum_grad_iters", 3))
        run["lr_scale_other"] = float(run.get("lr_scale_other", 1.0))
        run["evaluate"] = False
        run["resume_ckpt_path"] = None
        # Never let a stale smoke-test override truncate a full epoch.
        run.pop("iters_per_epoch", None)
        run.pop("max_iters", None)
        run["output_dir"] = str(
            root / "output" / f"{name}_E3"
        )

        path = out_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        print(
            f"[CREATED] {path}\n"
            f"  mode={mode} gnn_scale={scale} seed=42 epochs=3\n"
            f"  accum_grad_iters={run['accum_grad_iters']} "
            f"warmup_steps={run.get('warmup_steps', '<checkpoint default>')}\n"
            f"  output_dir={run['output_dir']}"
        )

    print("[OK] Configs copied from checkpoint config; no checkpoint weights will be resumed.")


if __name__ == "__main__":
    main()
