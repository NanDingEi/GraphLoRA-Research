#!/usr/bin/env python3
"""Valid-only VG1 intervention: measure whether learned rank gates reach logits.

Uses trained VG1 checkpoint without training; scales just the trained
delta correction in V (0, 1, 10, 50). Frozen LoRA/GNN/Q stay unchanged.
Prediction order/labels are checked identically across conditions.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--factors", default="0,1,10,50")
    ap.add_argument("--result", required=True)
    ap.add_argument("--expected-baseline", type=float, default=0.7296752596287361)
    ap.add_argument("--auc-tolerance", type=float, default=0.00002)
    args = ap.parse_args()

    factors = [float(x) for x in args.factors.split(",")]
    if not factors or len(factors) != len(set(factors)) or any(x < 0 or x > 100 for x in factors):
        raise ValueError("Factors must be distinct nonnegative numbers <= 100")
    if 0 not in factors or 1 not in factors:
        raise ValueError("Include 0 and 1 to validate anchor and reproduce trained VG1")
    if factors[0] != 0:
        raise ValueError("Baseline factor 0 must be first")

    path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.result).expanduser().resolve()
    for p in (path, checkpoint):
        if not p.is_file():
            raise FileNotFoundError(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(Namespace(cfg_path=str(path), options=None))
    if cfg.model_cfg.collab_mode != "target_v_gate_candidate":
        raise ValueError("Use VG1_CANDIDATE config only")
    cfg.model_cfg.ckpt = str(checkpoint)
    cfg.run_cfg.resume_ckpt_path = None
    cfg.run_cfg.distributed = False
    cfg.run_cfg.evaluate = False
    cfg.run_cfg.output_dir = str(out.parent / "eval_artifacts")

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task = entry.tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    try:
        data_path = cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_path = cfg.datasets_cfg.amazon_ood.path
    tables = [pd.read_pickle(data_path + s + "_ood2.pkl") for s in ("train","valid","test")]
    cfg.model_cfg.rec_config.user_num = int(max(x.uid.max() for x in tables) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(x.iid.max() for x in tables) + 1)
    del tables

    model = task.build_model(cfg)
    gate_modules = [m for m in model.my_linears
                    if getattr(m, "target_v_gate", None) is not None]
    if len(gate_modules) != 1 or gate_modules[0].module_tag != "v":
        raise RuntimeError("Expected exactly one L31 V gate")
    v = gate_modules[0]
    runner = entry.get_runner_class(cfg)(
        cfg=cfg, job_id=entry.now(), task=task, model=model, datasets=datasets
    )
    runner.set_model_mode(cfg.run_cfg.mode)
    model = runner.unwrap_dist_model(runner.model)
    if model.run_mode_ != "v2":
        raise RuntimeError("Expected Yes/No v2 mode")

    active = {"factor": 0.0, "gates": [], "corrections": [], "scores": None}
    original_forward = v.target_v_gate.forward
    original_gen = model.generate_for_samples

    def measured_gate(context):
        delta = original_forward(context)
        factor = active["factor"]
        # Record the raw learned gate and the counterfactual correction.
        # local_node_ids are aligned one-to-one with context rows, as verified
        # by the CPU context/indexing test.
        valid = v.local_node_ids != -1
        if delta.size(0) != int(valid.sum().item()):
            raise RuntimeError("V gate rows do not match valid injected nodes")
        z = v.gcn_output[v.local_node_ids[valid]].detach().float()
        correction = z * (v.gnn_scale * v.warmup_factor) * (delta.detach() * factor)
        active["gates"].append(delta.detach().float().cpu())
        active["corrections"].append(correction.cpu())
        return delta * factor

    def capture_gen(samples, *a, **kw):
        result = original_gen(samples, *a, **kw)
        scores = active["scores"]
        if scores is not None:
            score = result.get("logits")
            if not torch.is_tensor(score):
                raise RuntimeError("Expected logits tensor from Yes/No evaluation")
            scores["score"].extend(score.detach().float().flatten().cpu().tolist())
            for name, key in (("user","UserID"),("target","TargetItemID"),("label","label")):
                scores[name].extend(samples[key].detach().flatten().cpu().tolist())
        return result

    v.target_v_gate.forward = measured_gate
    model.generate_for_samples = capture_gen

    records = []
    anchor = None
    order = None
    baseline_score = None
    trained_score = None
    for factor in factors:
        entry.setup_seeds(cfg)
        active["factor"] = factor
        active["gates"] = []
        active["corrections"] = []
        active["scores"] = {"user":[],"target":[],"label":[],"score":[]}
        print(f"[VG-FACTOR] BEGIN x{factor}", flush=True)
        result = runner.eval_epoch(split_name="valid", cur_epoch=0, skip_reload=True)
        if result is None:
            raise RuntimeError("Evaluation returned no metrics")
        captures = active["scores"]
        if len(captures["score"]) != 5200:
            raise RuntimeError(f"Expected 5200 predictions, got {len(captures['score'])}")
        identity = tuple(tuple(captures[k]) for k in ("user","target","label"))
        if order is None:
            order = identity
        elif identity != order:
            raise RuntimeError("Sample order changed between interventions")
        score = np.asarray(captures["score"], dtype=np.float64)
        if not np.isfinite(score).all():
            raise RuntimeError("Non-finite prediction score")
        gates = torch.cat(active["gates"]).float()
        corrections = torch.cat(active["corrections"]).float()
        if not gates.numel() or not corrections.numel():
            raise RuntimeError("L31 V gate was never invoked")
        uauc = result.get("uauc")
        row = {
            "factor": factor,
            "auc": float(result["agg_metrics"]),
            "uauc": float(uauc[0]) if isinstance(uauc, (tuple,list)) else None,
            "loss": float(result["loss"]),
            "num_gated_nodes": int(gates.shape[0]),
            "raw_gate_mean_abs": float(gates.abs().mean()),
            "raw_gate_std": float(gates.std()),
            "raw_gate_max_abs": float(gates.abs().max()),
            "actual_correction_mean_abs": float(corrections.abs().mean()),
            "actual_correction_max_abs": float(corrections.abs().max()),
        }
        if factor == 0:
            anchor = float(result["agg_metrics"])
            baseline_score = score
            if abs(anchor - args.expected_baseline) > args.auc_tolerance:
                out.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
                raise RuntimeError(
                    f"VG1 gate-off baseline {anchor:.9f} does not match "
                    f"S2 {args.expected_baseline:.9f}"
                )
        if factor == 1:
            trained_score = score
        if baseline_score is not None:
            row["mean_abs_score_delta_vs_gate_off"] = float(
                np.mean(np.abs(score - baseline_score))
            )
            row["fraction_scores_changed_gt_1e-5"] = float(
                np.mean(np.abs(score - baseline_score) > 1e-5)
            )
        if trained_score is not None:
            row["mean_abs_score_delta_vs_trained"] = float(
                np.mean(np.abs(score - trained_score))
            )
        records.append(row)
        print("[VG-FACTOR-RESULT] " + json.dumps(row, ensure_ascii=False), flush=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        active["scores"] = None
    print("[VG-FACTOR] DONE: valid only, no weights updated.", flush=True)


if __name__ == "__main__":
    main()
