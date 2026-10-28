#!/usr/bin/env python3
"""NIGHT23 LightGCN graph conditioned CoRA, five controlled arms.

This runner never retrains an MF or GraphLoRA baseline.  Run pretraining
separately, then run 5 foreground GPU0 validation experiments. No Test.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

BASE = dict(layers="0-31", projections="q_proj,k_proj,v_proj,o_proj",
    cf_input="concat", injection_mode="shared", rank_profile="uniform",
    group_projectors="group8", alpha=48, p_lr=1.0,
    residual_ratio=0.0, depth_profile="uniform",
    projection_scales="q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    granularity_mode="original", fine_residual_mode="off",
    collab_mode="off", seed=42)

EXPERIMENTS = [
    dict(id="G1_LIGHTGCN_BASE", graph_mode="graph_base",
         title="Graph user-item pair, no extra coarse/fine conditions"),
    dict(id="G2_GRAPH_COARSE", graph_mode="graph_coarse",
         title="Graph communities and user/community preference"),
    dict(id="G3_GRAPH_FINE", graph_mode="graph_fine",
         title="Graph residual plus community-specific user positive history"),
    dict(id="G4_GRAPH_DUAL_UNIFORM", graph_mode="graph_uniform",
         title="Graph coarse and fine equal across 8 CoRA groups"),
    dict(id="G5_GRAPH_DUAL_C2F", graph_mode="graph_c2f",
         title="Progressive graph coarse-to-fine conditions across 8 groups"),
]
for r in EXPERIMENTS:
    r.update(BASE)
BY_ID = {r["id"]: r for r in EXPERIMENTS}

def preflight(npz, train):
    if not npz.is_file() or not train.is_file():
        raise FileNotFoundError(f"Missing graph artifact or TRAIN: {npz} {train}")
    if "test" in train.name.lower() or "valid" in train.name.lower():
        raise ValueError("History must derive from TRAIN only")
    with np.load(npz, allow_pickle=False) as f:
        for key in ("user_graph", "item_graph", "item_to_cluster",
                    "prototypes", "item_residual"):
            if key not in f:
                raise KeyError(f"Missing graph artifact key: {key}")
        if f["prototypes"].shape != (16, 256):
            raise ValueError("Graph requires K16, 256-dimensional embeddings")
        if not np.all(np.isfinite(f["user_graph"])) or not np.all(np.isfinite(f["item_graph"])):
            raise ValueError("Graph vectors not finite")

def main():
    p = argparse.ArgumentParser()
    act = p.add_mutually_exclusive_group(required=True)
    act.add_argument("--plan", action="store_true")
    act.add_argument("--run", action="store_true")
    act.add_argument("--only", choices=tuple(BY_ID))
    p.add_argument("--graph-artifact", type=Path, required=True)
    p.add_argument("--train-file", type=Path, default=Path(
        "/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/train_ood2.pkl"))
    a = p.parse_args()
    npz = a.graph_artifact.resolve()
    train = a.train_file.resolve()
    preflight(npz, train)
    common = core.prepared_config()
    rows = [BY_ID[a.only]] if a.only else EXPERIMENTS
    rows = [dict(r, collab_proto_path=str(npz),
                 collab_train_path=str(train)) for r in rows]
    print("[N23] Independent graph CF conditions, MF CTR/KD unchanged, "
          "seed42, rank16 alpha48, QKVO full 0..31, group8, "
          "one epoch Valid only, serial GPU0 foreground", flush=True)
    for r in rows:
        print(f"[N23-PLAN] {r['id']} graph_mode={r['graph_mode']}", flush=True)
    if a.plan:
        return
    subprocess.run([sys.executable, "experiments/no31_cora_original/test_night23_cpu.py"],
                   cwd=ROOT, check=True)
    core.run_trials(rows, common, name_prefix="NO31_CORA_NIGHT23")

if __name__ == "__main__":
    main()
