#!/usr/bin/env python3
"""Full VALID scale sweep of a *fixed* trained D1 N2/N3 checkpoint.

No training or checkpoint writes. Compare raw Yes logits, AUC and the exact
GraphLoRA non-distributed UAUC (positive-count weighted per-user AUC).
All scales see the same batch and same RNG; ID-V remains unchanged.
DO NOT select a multiplier based on Test.
"""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import minigpt4.tasks as tasks
from audit_D1_logits import rng_states, restore_rng
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.registry import registry
from minigpt4.common.utils import now
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.rec_base_task import uAUC_pandas
from minigpt4.datasets.builders import *  # noqa: F401,F403
from minigpt4.models import *  # noqa: F401,F403
from minigpt4.processors import *  # noqa: F401,F403
from minigpt4.runners import *  # noqa: F401,F403
from minigpt4.tasks import *  # noqa: F401,F403


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Full completed N2/N3 checkpoint, not S2 and not trainable-only")
    parser.add_argument("--cfg-path",
                        default="tran_configs/NIGHT4_N2_DECISION_Q_ID_V.yaml")
    parser.add_argument("--scales", default="0,1,10,20,50,100")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = all Valid batches; positive number = dry-run only")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_batches < 0:
        raise ValueError("Invalid batch size or max-batches")
    scales = [float(x.strip()) for x in args.scales.split(",")]
    if len(set(scales)) != len(scales) or not scales or scales[0] != 0:
        raise ValueError("Scales must be unique and start with zero: 0,1,10,...")
    if any(not np.isfinite(x) or x < 0 for x in scales):
        raise ValueError("Scales must be finite and nonnegative")
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)

    cfg = Config(SimpleNamespace(
        cfg_path=args.cfg_path,
        options=[
            "model.ckpt=" + str(Path(args.ckpt).resolve()),
            "run.evaluate=true",
            "run.batch_size_eval=" + str(args.batch_size),
            "run.output_dir=output/D1_valid_scale_audit",
        ],
    ))
    if cfg.run_cfg.get("distributed", False):
        raise ValueError("This offline single-GPU audit requires distributed=false")
    init_distributed_mode(cfg.run_cfg)
    seed = int(cfg.run_cfg.seed) + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    root = cfg.datasets_cfg.amazon_ood.path
    frames = [pd.read_pickle(str(Path(root) / x)) for x in
              ("train_ood2.pkl", "valid_ood2.pkl", "test_ood2.pkl")]
    cfg.model_cfg.rec_config.user_num = int(max(f.uid.max() for f in frames)) + 1
    cfg.model_cfg.rec_config.item_num = int(max(f.iid.max() for f in frames)) + 1

    model = task.build_model(cfg)
    runner_class = registry.get_runner_class(
        cfg.run_cfg.get("runner", "rec_runner_base"))
    runner = runner_class(
        cfg=cfg, job_id=now(), task=task, model=model, datasets=datasets)
    runner.set_model_mode(cfg.run_cfg.mode)
    model = runner.model
    model.eval()
    if model.d1_mode not in ("decision_q_v", "decision_q_v_gated"):
        raise ValueError("Pass N2 or N3 config, NOT N0/N1")
    qmods = [m for m in model.my_linears if m.module_tag == "q"]
    if len(qmods) != 1 or model.d1_pair_proj is None:
        raise RuntimeError("Expected exactly one decision-Q module and projector")
    orig_mode = qmods[0].d1_mode
    orig_weights = model.d1_pair_proj.weight.detach().clone()
    orig_norm = float(orig_weights.float().norm().item())
    if orig_norm <= 0:
        raise RuntimeError("Loaded decision-Q projector is still zero!")
    print("[VALID-SWEEP] ckpt={} mode={} P_norm={:.9g}".format(
        args.ckpt, model.d1_mode, orig_norm), flush=True)

    multi = runner.dataloaders["valid"]
    if not hasattr(multi, "loaders") or len(multi.loaders) != 1:
        raise RuntimeError("Expected one Valid PrefetchLoader")
    loader = multi.loaders[0]
    n_expected = len(loader.loader.dataset)
    score_lists = {scale: [] for scale in scales}
    users_list, labels_list = [], []
    batches_done = 0

    try:
        with torch.no_grad():
            for batch_idx, raw in enumerate(loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                batch = prepare_sample(raw, cuda_enabled=True)
                users_list.extend(batch["UserID"].detach().cpu().tolist())
                labels_list.extend(batch["label"].detach().cpu().tolist())
                # Each forward starts with exactly the same RNG state,
                # so random prompt selection cannot confound Q comparisons.
                rng = rng_states()
                qmods[0].d1_mode = "v_only"
                restore_rng(rng)
                baseline = (model.generate_for_samples_v2(batch)["logits"]
                            .detach().float().cpu().reshape(-1).numpy().copy())
                score_lists[0.0].extend(baseline.tolist())
                qmods[0].d1_mode = orig_mode

                for scale in scales[1:]:
                    model.d1_pair_proj.weight.copy_(orig_weights * scale)
                    restore_rng(rng)
                    logits = (model.generate_for_samples_v2(batch)["logits"]
                              .detach().float().cpu().reshape(-1).numpy().copy())
                    score_lists[scale].extend(logits.tolist())
                batches_done += 1
                if batches_done % 100 == 0:
                    print("[VALID-SWEEP] {}/{} batches, {} samples".format(
                        batches_done, len(loader), len(labels_list)), flush=True)
    finally:
        qmods[0].d1_mode = orig_mode
        with torch.no_grad():
            model.d1_pair_proj.weight.copy_(orig_weights)

    labels = np.asarray(labels_list, dtype=np.int64)
    users = np.asarray(users_list, dtype=np.int64)
    full_pass = args.max_batches == 0
    if full_pass and len(labels) != n_expected:
        raise RuntimeError("Valid sample count mismatch: expected {}, got {}".
                           format(n_expected, len(labels)))
    if len(labels) < 2 or np.unique(labels).size < 2:
        raise RuntimeError("Valid sample labels insufficient for AUC")
    base = np.asarray(score_lists[0.0], dtype=np.float64)
    if not np.isfinite(base).all():
        raise RuntimeError("Baseline logits contain NaN/Inf")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = Path(args.output_prefix or ("logs/D1_valid_scales_" + stamp))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_npz = {"UserID": users, "label": labels}
    for scale in scales:
        scores = np.asarray(score_lists[scale], dtype=np.float64)
        if len(scores) != len(labels) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid score count or NaN at scale {}".format(scale))
        auc = float(roc_auc_score(labels, scores))
        uauc, computed_users, _ = uAUC_pandas(users, scores, labels)
        delta = scores - base
        row = {
            "scale": scale, "auc": auc, "uauc": float(uauc),
            "n": int(len(labels)), "valid_users": len(computed_users),
            "logits_changed": int(np.count_nonzero(delta)),
            "changed_ratio": float(np.count_nonzero(delta) / len(labels)),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "max_abs_delta": float(np.max(np.abs(delta))),
        }
        rows.append(row)
        prediction_npz["score_scale_" + str(scale).replace(".", "_")] = scores
    auc0, uauc0 = rows[0]["auc"], rows[0]["uauc"]
    for row in rows:
        row["delta_auc_vs_off"] = row["auc"] - auc0
        row["delta_uauc_vs_off"] = row["uauc"] - uauc0
        print("[VALID-SWEEP] scale={:g} AUC={:.9f} UAUC={:.9f} "
              "dAUC={:+.9f} dUAUC={:+.9f} changed={}/{} ({:.2%}) "
              "mean|dlogit|={:.8g} max|dlogit|={:.8g}".format(
                  row["scale"], row["auc"], row["uauc"],
                  row["delta_auc_vs_off"], row["delta_uauc_vs_off"],
                  row["logits_changed"], row["n"], row["changed_ratio"],
                  row["mean_abs_delta"], row["max_abs_delta"]), flush=True)

    with open(str(prefix) + ".csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(str(prefix) + ".json", "w") as file:
        json.dump({
            "checkpoint": str(Path(args.ckpt).resolve()),
            "config": args.cfg_path, "dataset": "valid",
            "complete_valid": full_pass, "expected_n": n_expected,
            "observed_n": int(len(labels)), "batch_size": args.batch_size,
            "num_batches": batches_done, "scales": scales, "results": rows,
            "warning": "Only inference-time sensitivity; scales were NOT trained.",
        }, file, indent=2)
    np.savez_compressed(str(prefix) + ".npz", **prediction_npz)
    print("[VALID-SWEEP] saved {}.csv / .json / .npz".format(prefix), flush=True)
    print("[VALID-SWEEP] FINAL: no checkpoint changed, no training performed, "
          "no Test used.", flush=True)


if __name__ == "__main__":
    main()
