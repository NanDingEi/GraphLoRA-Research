#!/usr/bin/env python3
"""Confirm zero-init K0/K1 L31 Key adapter recovers S2-L31 Valid AUC.

Valid Small only. A fresh model is initialized in the requested VG mode and
warm-started from the historical S2 checkpoint. Validation MUST match champion.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd

import train_collm_mf_din as entry
from minigpt4.common.config import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-auc", type=float, default=0.7296752596287361)
    p.add_argument("--tolerance", type=float, default=0.00002)
    p.add_argument("--result", required=True)
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"{checkpoint} or {cfg_path}")
    cfg = Config(Namespace(cfg_path=str(cfg_path), options=None))
    if cfg.model_cfg.collab_mode not in (
            "target_k_no_candidate", "target_k_candidate"):
        raise RuntimeError("Expected K0 or K1 architecture")
    cfg.model_cfg.ckpt = str(checkpoint)
    cfg.run_cfg.resume_ckpt_path = None
    cfg.run_cfg.evaluate = False
    cfg.run_cfg.distributed = False
    result_path = Path(args.result).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.run_cfg.output_dir = str(result_path.parent / "eval_artifacts")

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task = entry.tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    try:
        data_path = cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_path = cfg.datasets_cfg.amazon_ood.path
    train = pd.read_pickle(data_path + "train_ood2.pkl")
    valid = pd.read_pickle(data_path + "valid_ood2.pkl")
    test = pd.read_pickle(data_path + "test_ood2.pkl")
    cfg.model_cfg.rec_config.user_num = int(max(
        train.uid.max(), valid.uid.max(), test.uid.max()) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(
        train.iid.max(), valid.iid.max(), test.iid.max()) + 1)

    model = task.build_model(cfg)
    if model.target_key_adapter is None:
        raise RuntimeError("K module missing")
    if model.target_key_adapter.network[-1].weight.count_nonzero() != 0:
        raise RuntimeError("K adapter output projection must be zero-initialized")
    runner = entry.get_runner_class(cfg)(
        cfg=cfg, job_id=entry.now(), task=task, model=model, datasets=datasets
    )
    runner.set_model_mode(cfg.run_cfg.mode)
    if runner.unwrap_dist_model(runner.model).run_mode_ != "v2":
        raise RuntimeError("Wrong run mode")
    entry.setup_seeds(cfg)
    result = runner.eval_epoch(split_name="valid", cur_epoch=0, skip_reload=True)
    auc = float(result["agg_metrics"])
    uauc = result["uauc"]
    uauc_value = float(uauc[0]) if isinstance(uauc, (tuple, list)) else None
    record = {"mode": str(cfg.model_cfg.collab_mode), "auc": auc,
              "uauc": uauc_value, "expected_auc": args.expected_auc,
              "abs_delta": abs(auc-args.expected_auc)}
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("[KEY-ANCHOR]", json.dumps(record), flush=True)
    if record["abs_delta"] > args.tolerance:
        raise RuntimeError(
            f"Key zero-init did not match S2: observed={auc:.9f}, "
            f"expected={args.expected_auc:.9f}, tolerance={args.tolerance}"
        )
    print("[KEY-ANCHOR] PASS; full training may proceed", flush=True)


if __name__ == "__main__":
    main()
