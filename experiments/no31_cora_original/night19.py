#!/usr/bin/env python3
"""NIGHT19: controlled fine-grained residual injection into original CoRA.

All trials retain the original coarse MF pair as the base CoRA condition,
group8 projectors, rank16/alpha48, QKVO in layers 0-31, and NO31 CTR/KD.
Only the fine residual routing/input changes between R1-R8. Valid only, one
trained epoch, serial GPU0 execution; no test-set evaluation.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

BASE = dict(
    layers="0-31",
    projections="q_proj,k_proj,v_proj,o_proj",
    cf_input="concat",
    injection_mode="shared",
    rank_profile="uniform",
    group_projectors="group8",
    alpha=48,
    p_lr=1.0,
    residual_ratio=0.0,
    depth_profile="uniform",
    projection_scales="q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    granularity_mode="original",
    seed=42,
)

EXPERIMENTS = [
    dict(id="R1_FINE_RESIDUAL_R8", title="Fine residual on rank channels 9-16",
         fine_residual_mode="r8"),
    dict(id="R2_FINE_RESIDUAL_R16", title="Fine residual on all 16 rank channels",
         fine_residual_mode="r16"),
    dict(id="R3_FINE_RESIDUAL_LATE8", title="Fine residual only in layers 24-31",
         fine_residual_mode="late8"),
    dict(id="R4_FINE_RESIDUAL_LATE4", title="Fine residual only in layers 28-31",
         fine_residual_mode="late4"),
    dict(id="R5_FINE_RESIDUAL_COARSE_TO_FINE",
         title="Coarse-to-fine residual schedule across group8",
         fine_residual_mode="coarse_to_fine"),
    dict(id="R6_FINE_RESIDUAL_FINE_TO_COARSE",
         title="Reverse fine-to-coarse residual schedule control",
         fine_residual_mode="fine_to_coarse"),
    dict(id="R7_FINE_RESIDUAL_HISTORY",
         title="Target-history interaction only",
         fine_residual_mode="history"),
    dict(id="R8_FINE_RESIDUAL_USER_ITEM",
         title="User-target interaction only",
         fine_residual_mode="user_item"),
]
for row in EXPERIMENTS:
    row.update(BASE)
BY_ID = {row["id"]: row for row in EXPERIMENTS}


def plan(rows):
    cfg = core.prepared_config()
    print(
        "[NIGHT19-BASELINE] Historical NIGHT17 E0 Valid "
        "AUC=0.729928474 UAUC=0.686612232; never retrain baseline.",
        flush=True,
    )
    print(
        "[NIGHT19-COMMON] rank16 alpha48 QKVO L0-31 group8; original coarse MF "
        "condition preserved; seed42 CTR=.05 KD=.05; one epoch; Valid only; "
        "GPU0 serial foreground.",
        flush=True,
    )
    for row in rows:
        print(
            f"[NIGHT19-PLAN] {row['id']} mode={row['fine_residual_mode']} "
            f"title={row['title']}",
            flush=True,
        )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--only", choices=tuple(BY_ID))
    args = parser.parse_args()

    rows = [BY_ID[args.only]] if args.only else EXPERIMENTS
    cfg = plan(rows)
    if args.plan:
        return

    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night19_cpu.py"],
        cwd=ROOT, check=True,
    )
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT19")


if __name__ == "__main__":
    main()
