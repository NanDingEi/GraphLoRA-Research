#!/usr/bin/env python3
"""NIGHT11: diagnose the alpha cliff and test conservative CoRA residual controls.

All runs use seed 42, fresh initialization, one epoch, Valid only, and the
existing NO31+CTR/KD configuration. G2 and alpha52 are historical anchors:
if either drifts by >0.02 AUC or UAUC, the sweep stops before interpreting
the remaining arms.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

ALL = "q_proj,k_proj,v_proj,o_proj"
UNIFORM = "q_proj=1,k_proj=1,v_proj=1,o_proj=1"
QK_HALF_VO_FULL = "q_proj=0.5,k_proj=0.5,v_proj=1,o_proj=1"
FIXED_R0_CHECKPOINT = (
    ROOT / "output/NO31_CORA_ORIGINAL_20261023_234005_110337/"
    "20261023234/checkpoint_0.pth"
)

EXPERIMENTS = [
    dict(id="N1", title="G2 anchor: QKVO alpha48",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=48, p_lr=1.0, residual_ratio=0.0,
         depth_profile="uniform", projection_scales=UNIFORM, seed=42,
         anchor_auc=0.7301218951846701, anchor_uauc=0.6880410112996300,
         anchor_tolerance=0.02),
    dict(id="N2", title="Repeat alpha52 to verify the performance cliff",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=52, p_lr=1.0, residual_ratio=0.0,
         depth_profile="uniform", projection_scales=UNIFORM, seed=42,
         anchor_auc=0.6439879348234456, anchor_uauc=0.5889665782499179,
         anchor_tolerance=0.02),
    dict(id="N3", title="QKVO alpha49",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=49, p_lr=1.0, residual_ratio=0.0,
         depth_profile="uniform", projection_scales=UNIFORM, seed=42),
    dict(id="N4", title="QKVO alpha50",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=50, p_lr=1.0, residual_ratio=0.0,
         depth_profile="uniform", projection_scales=UNIFORM, seed=42),
    dict(id="N5", title="QKVO alpha51",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=51, p_lr=1.0, residual_ratio=0.0,
         depth_profile="uniform", projection_scales=UNIFORM, seed=42),
    dict(id="N6", title="Alpha48 with CoRA-P LR x0.25",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=48, p_lr=0.25, residual_ratio=0.0,
         depth_profile="uniform", projection_scales=UNIFORM, seed=42),
    dict(id="N7", title="Alpha48 with residual/base norm cap 0.25",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=48, p_lr=1.0, residual_ratio=0.25,
         depth_profile="uniform", projection_scales=UNIFORM, seed=42),
    dict(id="N8", title="Alpha48 with half-strength Q/K and full V/O residuals",
         projections=ALL, layers="0-31", cf_input="concat",
         alpha=48, p_lr=1.0, residual_ratio=0.0,
         depth_profile="uniform", projection_scales=QK_HALF_VO_FULL, seed=42),
]
BY_ID = {row["id"]: row for row in EXPERIMENTS}


def validate_plan():
    cfg = core.prepared_config()
    if len(EXPERIMENTS) != 8 or len(BY_ID) != 8:
        raise RuntimeError("NIGHT11 must contain eight unique experiments")
    if int(cfg["run"].get("seed", -1)) != 42:
        raise RuntimeError("Prepared seed changed: all runs must use seed42")
    for row in EXPERIMENTS:
        if row["seed"] != 42:
            raise RuntimeError(f"{row['id']} changes random seed")
        if row["layers"] != "0-31" or row["cf_input"] != "concat":
            raise RuntimeError(f"{row['id']} changes an unapproved architecture axis")
        if row["alpha"] not in (48, 49, 50, 51, 52):
            raise RuntimeError(f"Unexpected alpha in {row['id']}")
        if row["depth_profile"] != "uniform":
            raise RuntimeError(f"Unexpected depth profile in {row['id']}")
        if row["projection_scales"] not in (UNIFORM, QK_HALF_VO_FULL):
            raise RuntimeError(f"Unexpected projection scale in {row['id']}")
    return cfg


def show_plan(rows):
    cfg = validate_plan()
    print("[NIGHT11-COMMON] seed=42; fresh init; serial foreground GPU0; "
          "one epoch; Valid only; train_bs=2 eval_bs=3 accum=4; "
          "init_lr=1e-4; scheduler horizon=3; CTR=.05 KD=.05; "
          "semantic_weight=1e-4 projector_reg=5e-4", flush=True)
    for row in rows:
        print("[NIGHT11-PLAN] %s alpha=%d projections=%s p_lr=%.2f "
              "residual_ratio=%.2f projection_scales=%s seed=42 :: %s"
              % (row["id"], row["alpha"], row["projections"], row["p_lr"],
                 row["residual_ratio"], row["projection_scales"], row["title"]),
              flush=True)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Eight CoRA stability/improvement experiments")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--only", choices=tuple(BY_ID))
    parser.add_argument("--start", choices=tuple(BY_ID))
    parser.add_argument("--first", type=int, choices=range(1, 9), metavar="N")
    parser.add_argument("--skip-fixed-r0-precheck", action="store_true",
                        help="Only if the historical R0 checkpoint is absent and early-layer gradients were already verified")
    args = parser.parse_args()
    if sum(x is not None for x in (args.only, args.start, args.first)) > 1:
        parser.error("--only, --start and --first cannot be combined")
    rows = EXPERIMENTS
    if args.only:
        rows = [BY_ID[args.only]]
    elif args.start:
        rows = EXPERIMENTS[list(BY_ID).index(args.start):]
    elif args.first is not None:
        rows = EXPERIMENTS[:args.first]
    cfg = show_plan(rows)
    if args.plan:
        return
    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night11_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    if not args.skip_fixed_r0_precheck:
        core.precheck(FIXED_R0_CHECKPOINT)
    print("[NIGHT11-PREFLIGHT-PASS] CPU checks and historical early-layer check passed",
          flush=True)
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT11")
    print("[NIGHT11-DONE] Check summary.csv under logs/NO31_CORA_NIGHT11_*",
          flush=True)


if __name__ == "__main__":
    main()
