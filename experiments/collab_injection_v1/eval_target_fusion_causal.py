#!/usr/bin/env python3
"""Valid-only causal feature ablations for trained Target-Aware Fusion.

No training, no test split evaluation, no checkpoint mutation.
All conditions reuse exact checkpoint + frozen model, and reset seed to keep
sample order/prompt sampling matched. A shuffled target is changed ONLY in the
decision fusion vector: LLM prompt and historical L31 Q/V injection remain.
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

import numpy as np
import pandas as pd
import torch

import train_collm_mf_din as entry
from minigpt4.common.config import Config


MODES = ("baseline", "original", "shuffle_target", "no_target",
         "no_history", "shuffle_user")


def transform(vec, mode, width=256):
    assert vec.ndim == 2 and vec.shape[1] == 6 * width, vec.shape
    if mode in ("baseline", "original"):
        return vec
    u, t, h, ut, ht, delta = torch.split(vec, width, dim=-1)
    if mode == "shuffle_target":
        # Swap graph features among examples while keeping the original
        # textual target candidate and original L31 GNN injection untouched.
        # 4-sample validation batches permit a deterministic derangement.
        if vec.shape[0] < 2:
            raise RuntimeError("shuffle_target needs batch_size >= 2")
        t = t.roll(1, 0)
        return torch.cat((u, t, h, u * t, h * t, (u - t).abs()), -1)
    if mode == "shuffle_user":
        if vec.shape[0] < 2:
            raise RuntimeError("shuffle_user needs batch_size >= 2")
        u = u.roll(1, 0)
        return torch.cat((u, t, h, u * t, h * t, (u - t).abs()), -1)
    if mode == "no_target":
        zeros = torch.zeros_like(t)
        return torch.cat((u, zeros, h, zeros, zeros, zeros), -1)
    if mode == "no_history":
        zeros = torch.zeros_like(h)
        return torch.cat((u, t, zeros, u * t, zeros, (u - t).abs()), -1)
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scale", type=float, default=0.05,
                   help="Inference-time scale; F1 at 0.05 is near champion AUC")
    p.add_argument("--modes", default="baseline,original,shuffle_target,no_target,no_history,shuffle_user")
    p.add_argument("--result", required=True)
    p.add_argument("--expected-baseline", type=float, default=None,
                   help="Fail early if scale-zero AUC differs by >0.003")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    for file in (checkpoint, config_path):
        if not file.is_file():
            raise FileNotFoundError(file)
    modes = [m.strip() for m in args.modes.split(",")]
    if not modes or len(modes) != len(set(modes)) or any(m not in MODES for m in modes):
        raise ValueError(f"modes must be distinct subset of {MODES}")
    if not 0 <= args.scale <= 1:
        raise ValueError("scale must be in [0,1]")

    outpath = Path(args.result).resolve()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(Namespace(cfg_path=str(config_path), options=None))
    if cfg.model_cfg.get("collab_mode") != "target_fusion":
        raise RuntimeError("Must use target_fusion config")
    cfg.model_cfg.ckpt = str(checkpoint)
    cfg.run_cfg.distributed = False
    cfg.run_cfg.evaluate = False
    cfg.run_cfg.resume_ckpt_path = None
    cfg.run_cfg.output_dir = str(outpath.parent / "eval_artifacts")

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task = entry.tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    try:
        data_path = cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_path = cfg.datasets_cfg.amazon_ood.path
    dfs = [pd.read_pickle(data_path + s + "_ood2.pkl")
           for s in ("train", "valid", "test")]
    cfg.model_cfg.rec_config.user_num = int(max(df.uid.max() for df in dfs) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(df.iid.max() for df in dfs) + 1)
    del dfs

    model = task.build_model(cfg)
    if not hasattr(model, "target_fusion"):
        raise RuntimeError("Fusion module missing")
    runner = entry.get_runner_class(cfg)(
        cfg=cfg, job_id=entry.now(), task=task, model=model, datasets=datasets
    )
    runner.set_model_mode(cfg.run_cfg.mode)
    model = runner.unwrap_dist_model(runner.model)
    assert model.run_mode_ == "v2"
    model.eval()

    original_make = model._make_target_fusion_features
    original_forward = model.target_fusion.forward
    original_generate = model.generate_for_samples
    active = {"mode": None, "capture": None, "norms": None}

    def make_features(fb, samples):
        vec = original_make(fb, samples)
        return transform(vec, active["mode"])

    def wrapped_forward(hidden, vec):
        residual = original_forward(hidden, vec)
        norms = active["norms"]
        if norms is not None:
            h_norm = torch.linalg.vector_norm(hidden.detach().float(), dim=-1)
            r_norm = torch.linalg.vector_norm(residual.detach().float(), dim=-1)
            ratio = (r_norm / h_norm.clamp_min(1e-6)).cpu().numpy()
            cosine = ((hidden.detach().float() * residual.detach().float()).sum(-1)
                      / (h_norm * r_norm).clamp_min(1e-8)).cpu().numpy()
            norms["ratio"].extend(ratio.tolist())
            norms["cosine"].extend(cosine.tolist())
        return residual

    def capture_generate(samples, *a, **kw):
        outputs = original_generate(samples, *a, **kw)
        captures = active["capture"]
        if captures is not None:
            pred = outputs.get("logits")
            if not torch.is_tensor(pred):
                raise RuntimeError("Expected model.generate_for_samples logits tensor")
            captures["score"].extend(pred.detach().float().flatten().cpu().tolist())
            captures["user"].extend(samples["UserID"].detach().flatten().cpu().tolist())
            captures["target"].extend(samples["TargetItemID"].detach().flatten().cpu().tolist())
            captures["label"].extend(samples["label"].detach().flatten().cpu().tolist())
        return outputs

    model._make_target_fusion_features = make_features
    model.target_fusion.forward = wrapped_forward
    model.generate_for_samples = capture_generate

    rows = []
    reference = None
    original_scores = None
    for mode in modes:
        entry.setup_seeds(cfg)
        active["mode"] = mode
        active["capture"] = {"score": [], "user": [], "target": [], "label": []}
        active["norms"] = {"ratio": [], "cosine": []}
        model.target_fusion.scale = 0.0 if mode == "baseline" else args.scale
        print(f"[CAUSAL] BEGIN mode={mode} scale={model.target_fusion.scale}", flush=True)
        result = runner.eval_epoch(split_name="valid", cur_epoch=0, skip_reload=True)
        if result is None or not isinstance(result.get("agg_metrics"), (int, float)):
            raise RuntimeError("Missing AUC from validation")
        cap = active["capture"]
        nrm = active["norms"]
        if len(cap["score"]) != 5200:
            raise RuntimeError(f"Expected 5200 Valid Small samples, got {len(cap['score'])}")
        ids = tuple(tuple(cap[k]) for k in ("user", "target", "label"))
        if reference is None:
            reference = ids
        elif ids != reference:
            raise RuntimeError(
                "Sample order/labels changed between conditions; cannot compare paired logits"
            )
        score = np.asarray(cap["score"], dtype=np.float64)
        if not np.isfinite(score).all():
            raise RuntimeError(f"Non-finite scores in {mode}")
        if mode == "baseline":
            baseline_scores = score
        if mode == "original":
            original_scores = score
        uauc = result.get("uauc")
        uauc_value = float(uauc[0]) if isinstance(uauc, (list, tuple)) else None
        row = {
            "mode": mode, "scale": model.target_fusion.scale,
            "auc": float(result["agg_metrics"]),
            "uauc": uauc_value, "loss": result.get("loss"),
            "mean_relative_residual": float(np.mean(nrm["ratio"])),
            "p95_relative_residual": float(np.percentile(nrm["ratio"], 95)),
            "mean_cosine": float(np.mean(nrm["cosine"])),
        }
        if original_scores is not None:
            row["mean_abs_score_delta_vs_original"] = float(np.mean(np.abs(score - original_scores)))
        if "baseline_scores" in locals():
            row["mean_abs_score_delta_vs_baseline"] = float(np.mean(np.abs(score - baseline_scores)))
        rows.append(row)
        print("[CAUSAL-RESULT] " + json.dumps(row, ensure_ascii=False), flush=True)
        outpath.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        if (mode == "baseline" and args.expected_baseline is not None
                and abs(row["auc"] - args.expected_baseline) > 0.003):
            raise RuntimeError(
                f"Baseline mismatch: observed {row['auc']:.9f}, "
                f"expected {args.expected_baseline:.9f}; stop before further ablations"
            )
        active["capture"] = None
        active["norms"] = None
    if "original" in modes:
        base = next((r["auc"] for r in rows if r["mode"] == "original"), None)
        for row in rows:
            row["auc_delta_vs_original"] = row["auc"] - base
        outpath.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[CAUSAL] DONE valid-only causal ablations; checkpoint unchanged", flush=True)


if __name__ == "__main__":
    main()
