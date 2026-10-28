#!/usr/bin/env python3
"""NIGHT24 true joint LightGCN -> CoRA -> LLM training.

All runs keep pretrained graph initial state and trainable graph propagation,
original MF prompt and CTR/KD.  Graph group labels remain static.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

BASE = dict(
    layers="0-31", projections="q_proj,k_proj,v_proj,o_proj", cf_input="concat",
    injection_mode="shared", rank_profile="uniform", group_projectors="group8",
    alpha=48, p_lr=1.0, residual_ratio=0.0, depth_profile="uniform",
    projection_scales="q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    granularity_mode="original", fine_residual_mode="off", collab_mode="off",
    graph_joint=True, seed=42,
)
EXPERIMENTS = [
    dict(id="J1_JOINT_GRAPH_BASE", graph_mode="graph_base",
         title="Joint LightGCN baseline graph pair"),
    dict(id="J2_JOINT_DUAL_UNIFORM", graph_mode="graph_uniform",
         title="Joint graph coarse/fine uniform"),
    dict(id="J3_JOINT_DUAL_C2F", graph_mode="graph_c2f",
         title="Joint graph coarse/fine layer schedule"),
]
for row in EXPERIMENTS:
    row.update(BASE)
BY_ID = {r["id"]: r for r in EXPERIMENTS}


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--only", choices=tuple(BY_ID))
    p.add_argument("--graph-artifact", type=Path, required=True)
    p.add_argument("--train-file", type=Path, default=Path(
        "/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/train_ood2.pkl"))
    p.add_argument("--graph-lr-scale", type=float, default=0.1)
    p.add_argument("--graph-aux-weight", type=float, default=0.01)
    a = p.parse_args()
    if not 0 < a.graph_lr_scale <= 1:
        raise ValueError("graph LR scale must be in (0,1]")
    if not 0 <= a.graph_aux_weight <= 0.1:
        raise ValueError("graph aux weight must be in [0,0.1]")
    train = a.train_file.resolve()
    graph = a.graph_artifact.resolve()
    if not train.is_file() or not graph.is_file():
        raise FileNotFoundError("Missing TRAIN or graph artifact")
    if "test" in train.name.lower() or "valid" in train.name.lower():
        raise ValueError("Only TRAIN used for collaborative graph")
    with np.load(graph, allow_pickle=False) as f:
        for key in ("graph_id_table", "user_graph", "item_graph",
                    "item_to_cluster", "prototypes", "item_residual"):
            if key not in f:
                raise KeyError(f"Missing {key}; rerun N24 graph pretraining")
        if f["prototypes"].shape != (16, 256):
            raise ValueError("Expected graph K16 and 256-d features")
    cfg = core.prepared_config()
    rows = [BY_ID[a.only]] if a.only else EXPERIMENTS
    rows = [dict(r, collab_proto_path=str(graph), collab_train_path=str(train),
                 graph_lr_scale=a.graph_lr_scale,
                 graph_aux_weight=a.graph_aux_weight) for r in rows]
    print("[N24] Graph encoder + CoRA/LLM jointly trained; "
          "rank16 alpha48 QKVO 0..31 group8 seed42, CTR=.05 KD=.05, "
          "GPU0 foreground serial, 1 epoch Valid only", flush=True)
    for row in rows:
        print(f"[N24-PLAN] {row['id']} mode={row['graph_mode']} "
              f"graph_lr_scale={a.graph_lr_scale} graph_aux={a.graph_aux_weight}",
              flush=True)
    if a.plan:
        return
    subprocess.run([sys.executable, "experiments/no31_cora_original/test_night24_cpu.py"],
                   cwd=ROOT, check=True)
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT24")


if __name__ == "__main__":
    main()
