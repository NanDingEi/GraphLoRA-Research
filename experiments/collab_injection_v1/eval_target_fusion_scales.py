#!/usr/bin/env python3
"""Evaluate an already-trained target_fusion checkpoint on VALID only.

For the EXACT SAME weights, evaluate scales 0, 0.05, 0.2, 1.0.
This isolates current fusion residual impact from changes in the trained
LoRA, graph and projector weights. All runs reseed prompt sampling.
No training, no test set, no checkpoint writes.
"""
import argparse
import json
import sys
from pathlib import Path

# Launched as python experiments/collab_injection_v1/this_script.py,
# sys.path[0] is the script's directory, not the repository root.
# Insert the root BEFORE importing the project's training entrypoint.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

import train_collm_mf_din as entry
from minigpt4.common.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Original FULL_E1.yaml file")
    parser.add_argument("--checkpoint", required=True, help="FULL_E1 checkpoint_best.pth")
    parser.add_argument("--scales", default="0,0.05,0.2,1")
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    if not ckpt.is_file() or not config.is_file():
        raise FileNotFoundError(f"Missing {ckpt} or {config}")

    # Match the project's original entrypoint exactly up to runner.train(),
    # but load the finished weights via model.ckpt and use runner.eval_epoch.
    # Config expects cfg_path and options, not a plain YAML dict.
    from argparse import Namespace
    cfg = Config(Namespace(cfg_path=str(config), options=None))
    cfg.model_cfg.ckpt = str(ckpt)
    cfg.run_cfg.resume_ckpt_path = None
    cfg.run_cfg.evaluate = False
    cfg.run_cfg.distributed = False  # one local GPU; no collective operations
    # Do not write evaluation artifacts into the trained model output folder.
    cfg.run_cfg.output_dir = str(
        Path(args.result).resolve().parent / "eval_artifacts"
        if args.result else Path.cwd() / "logs/collab_v1/target_fusion_eval_artifacts"
    )

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task = entry.tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    try:
        data_dir = cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_dir = cfg.datasets_cfg.amazon_ood.path
    train_df = pd.read_pickle(data_dir + "train_ood2.pkl")
    valid_df = pd.read_pickle(data_dir + "valid_ood2.pkl")
    test_df = pd.read_pickle(data_dir + "test_ood2.pkl")
    cfg.model_cfg.rec_config.user_num = int(max(
        train_df.uid.max(), valid_df.uid.max(), test_df.uid.max()) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(
        train_df.iid.max(), valid_df.iid.max(), test_df.iid.max()) + 1)

    model = task.build_model(cfg)
    if not hasattr(model, "target_fusion"):
        raise RuntimeError("Expected target_fusion model, missing target_fusion module")
    print("[FUSION-SCALE] loaded checkpoint:", ckpt, flush=True)

    runner = entry.get_runner_class(cfg)(
        cfg=cfg, job_id=entry.now(), task=task, model=model, datasets=datasets
    )
    scales = [float(x) for x in args.scales.split(",")]
    if not scales or any(not 0 <= scale <= 1 for scale in scales):
        raise ValueError("Scales must be within [0,1]")

    results = []
    for scale in scales:
        # Same random prompt choices per sample at every scale.
        entry.setup_seeds(cfg)
        runner.unwrap_dist_model(runner.model).target_fusion.scale = scale
        print(f"[FUSION-SCALE] BEGIN scale={scale}", flush=True)
        record = runner.eval_epoch(split_name="valid", cur_epoch=0, skip_reload=True)
        if record is None:
            raise RuntimeError(f"Validation returned no record for scale {scale}")
        print(f"[FUSION-SCALE] END scale={scale} agg_metrics={record.get('agg_metrics')}", flush=True)
        results.append({"scale": scale, "agg_metrics": record.get("agg_metrics")})

    if args.result:
        out = Path(args.result)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[FUSION-SCALE] finished valid-only evaluations; no model changes.", flush=True)


if __name__ == "__main__":
    main()
