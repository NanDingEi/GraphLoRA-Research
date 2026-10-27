#!/usr/bin/env python3
"""NIGHT20 four independent one-epoch GPU0 CoRA training experiments.

T1: coarse-only user+community preference -> QFormer query
T2: fine-only target residual+leave-target-out user community history -> query
T3: both uniformly
T4: coarse-to-fine depth-wise schedule, G0-G7=4 layers each

Shared original group8/alpha48/rank16/QKVO/NO31 CTR-KD/seed42.
Historical NIGHT17 E0 results are referenced, NOT re-trained.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

DEFAULT_PROTOTYPE = Path(
    "/home/bbx/user_lx/GraphLoRA_CF_PROTO_PROBE/logs/"
    "CF_PROTOTYPE_PROBE_20261027_103711/prototypes_k16.npz"
)
DEFAULT_TRAIN = Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/train_ood2.pkl")
BASE = dict(
    layers="0-31",
    projections="q_proj,k_proj,v_proj,o_proj",
    cf_input="concat", injection_mode="shared",
    rank_profile="uniform", group_projectors="group8",
    alpha=48, p_lr=1.0, residual_ratio=0.0, depth_profile="uniform",
    projection_scales="q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    granularity_mode="original", fine_residual_mode="off", seed=42,
)
EXPERIMENTS = [
    dict(id="T1_PROTO_COARSE_ONLY", title="Frozen MF prototype + user-group preference into QFormer",
         collab_mode="coarse_only"),
    dict(id="T2_PROTO_FINE_ONLY", title="Target residual + group-specific positive history into QFormer",
         collab_mode="fine_only"),
    dict(id="T3_PROTO_DUAL_UNIFORM", title="Coarse/fine QFormer adaptation, uniform half-half",
         collab_mode="dual_uniform"),
    dict(id="T4_PROTO_DUAL_C2F", title="Coarse to fine schedule over 8x4-layer groups",
         collab_mode="dual_c2f"),
]
for row in EXPERIMENTS:
    row.update(BASE)
BY_ID = {r["id"]: r for r in EXPERIMENTS}


def preflight_paths(proto_path, train_path):
    import numpy as np
    import pandas as pd
    if not proto_path.is_file():
        raise FileNotFoundError(f"Missing NIGHT20 prototype file: {proto_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing NIGHT20 TRAIN split: {train_path}")
    if "test" in train_path.name.lower() or "valid" in train_path.name.lower():
        raise ValueError("NIGHT20 features must be computed from TRAIN only")
    with np.load(str(proto_path), allow_pickle=False) as data:
        for key in ("item_to_cluster", "prototypes", "mf_residual"):
            if key not in data:
                raise KeyError(f"Missing {key} in {proto_path}")
        if data["prototypes"].shape != (16, 256):
            raise RuntimeError("NIGHT20 requires K=16, MF embedding dim=256")
    first = pd.read_pickle(train_path)
    if not {"uid", "iid", "label"}.issubset(first.columns):
        raise ValueError("Train file needs uid/iid/label")
    print(f"[NIGHT20-PREFLIGHT] prototype={proto_path} train={train_path} "
          f"train_rows={len(first)} K=16", flush=True)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--only", choices=tuple(BY_ID))
    parser.add_argument("--prototype", type=Path, default=DEFAULT_PROTOTYPE)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    args = parser.parse_args()

    proto_path = args.prototype.expanduser().resolve()
    train_path = args.train_file.expanduser().resolve()
    preflight_paths(proto_path, train_path)
    cfg = core.prepared_config()
    rows = [BY_ID[args.only]] if args.only else EXPERIMENTS
    rows = [dict(r, collab_proto_path=str(proto_path),
                 collab_train_path=str(train_path)) for r in rows]
    print("[NIGHT20-HISTORIC-E0] Valid AUC=0.729928474 UAUC=0.686612232 "
          "historical reference only; NOT re-trained.", flush=True)
    print("[NIGHT20-CONFIG] seed42 rank16 alpha48 full32-QKVO group8 "
          "CTR=.05 KD=.05 NO31; fresh-from-same-pretrain; "
          "one-epoch, Valid only, GPU0 serial foreground.", flush=True)
    for row in rows:
        print(f"[NIGHT20-PLAN] {row['id']} mode={row['collab_mode']}",
              flush=True)
    if args.plan:
        return

    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night20_cpu.py"],
        cwd=ROOT, check=True,
    )
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT20")


if __name__ == "__main__":
    main()
