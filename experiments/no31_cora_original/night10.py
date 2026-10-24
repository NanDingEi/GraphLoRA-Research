#!/usr/bin/env python3
"""Eight seed-42, one-epoch NO31+CoRA experiments targeting stability and AUC.

G1/G2 are exact anchors for the prior strong V/O and QKVO configurations.
G3-G5 refine alpha around the observed QKVO sweet spot.
G6 tests a lower CoRA-P learning rate.
G7 tests per-token residual/base norm clipping.
G8 tests gentle late-layer-weighted injection while retaining all 32 layers.
All arms use fresh, identical seeded initialization and Valid-only evaluation.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.no31_cora_original import night8 as core

ALL="q_proj,k_proj,v_proj,o_proj"
VO="v_proj,o_proj"
FIXED_R0_CHECKPOINT=(
    ROOT/"output/NO31_CORA_ORIGINAL_20261023_234005_110337/"
    "20261023234/checkpoint_0.pth"
)

EXPERIMENTS=[
    dict(id="G1",title="R2 anchor: V/O alpha32",
         projections=VO,layers="0-31",cf_input="concat",
         alpha=32,p_lr=1.0,residual_ratio=0.0,depth_profile="uniform",seed=42,
         anchor_auc=0.7257211273845012,anchor_uauc=0.6829168466943410,
         anchor_tolerance=0.02),
    dict(id="G2",title="F4 anchor: QKVO alpha48",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=48,p_lr=1.0,residual_ratio=0.0,depth_profile="uniform",seed=42,
         anchor_auc=0.7301218951846701,anchor_uauc=0.6880410112996300,
         anchor_tolerance=0.02),
    dict(id="G3",title="QKVO alpha52 local scale refinement",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=52,p_lr=1.0,residual_ratio=0.0,depth_profile="uniform",seed=42),
    dict(id="G4",title="QKVO alpha56 local scale refinement",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=56,p_lr=1.0,residual_ratio=0.0,depth_profile="uniform",seed=42),
    dict(id="G5",title="QKVO alpha60 local scale refinement",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=60,p_lr=1.0,residual_ratio=0.0,depth_profile="uniform",seed=42),
    dict(id="G6",title="QKVO alpha48 CoRA-P LR x0.5",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=48,p_lr=0.5,residual_ratio=0.0,depth_profile="uniform",seed=42),
    dict(id="G7",title="QKVO alpha48 residual/base norm cap 0.5",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=48,p_lr=1.0,residual_ratio=0.5,depth_profile="uniform",seed=42),
    dict(id="G8",title="QKVO alpha48 late-linear layer profile",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=48,p_lr=1.0,residual_ratio=0.0,depth_profile="late_linear",seed=42),
]
BY_ID={row["id"]:row for row in EXPERIMENTS}

def validate_plan():
    cfg=core.prepared_config()
    if len(EXPERIMENTS)!=8 or len(BY_ID)!=8:
        raise RuntimeError("Expected exactly eight unique experiments")
    if int(cfg["run"].get("seed",-1))!=42:
        raise RuntimeError("Prepared seed changed: all runs must keep seed42")
    for row in EXPERIMENTS:
        if row["seed"]!=42:
            raise RuntimeError(f"{row['id']} changes random seed")
        if row["layers"]!="0-31" or row["cf_input"]!="concat":
            raise RuntimeError(f"{row['id']} changes an unapproved architecture axis")
        if row["alpha"] not in (32,48,52,56,60):
            raise RuntimeError(f"Unexpected alpha in {row['id']}")
        if row["depth_profile"] not in ("uniform","late_linear"):
            raise RuntimeError(f"Unexpected depth profile in {row['id']}")
        if row["residual_ratio"] not in (0.0,0.5):
            raise RuntimeError(f"Unexpected residual ratio in {row['id']}")
    return cfg

def show_plan(rows):
    cfg=validate_plan()
    print("[NIGHT10-COMMON] seed=42 for every arm; fresh init; "
          "GPU0 serial foreground; one epoch; Valid only; no Test; "
          "batch2 eval3 accum4; init_lr=1e-4; scheduler horizon=3; "
          "CTR=.05 KD=.05 semantic_weight=1e-4 projector_reg=5e-4",
          flush=True)
    print(f"[NIGHT10-BASELINE] NO31 AUC={core.BASELINE_AUC:.9f} "
          f"UAUC={core.BASELINE_UAUC:.9f}; "
          f"R2 AUC=0.725721127 UAUC=0.682916847; "
          f"F4 AUC=0.730121895 UAUC=0.688041011",flush=True)
    for row in rows:
        print("[NIGHT10-PLAN] %s seed=%d alpha=%d projections=%s p_lr=%.2f "
              "residual_ratio=%.2f depth_profile=%s targets=%d :: %s"
              %(row["id"],row["seed"],row["alpha"],row["projections"],
                row["p_lr"],row["residual_ratio"],row["depth_profile"],
                core.selected_targets(row),row["title"]),flush=True)
    return cfg

def main():
    p=argparse.ArgumentParser(description="Eight one-epoch CoRA stability/improvement experiments")
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan",action="store_true")
    action.add_argument("--run",action="store_true")
    p.add_argument("--only",choices=tuple(BY_ID))
    p.add_argument("--start",choices=tuple(BY_ID))
    p.add_argument("--first",type=int,choices=range(1,9),metavar="N",
                   help="Run only the first N configs, e.g. --first 4")
    p.add_argument("--skip-fixed-r0-precheck",action="store_true",
                   help="Use only if the fixed R0 checkpoint is absent and its early-layer gradients were already verified")
    args=p.parse_args()
    if sum(x is not None for x in (args.only,args.start,args.first))>1:
        p.error("--only, --start and --first cannot be combined")
    chosen=EXPERIMENTS
    if args.only:
        chosen=[BY_ID[args.only]]
    elif args.start:
        chosen=EXPERIMENTS[list(BY_ID).index(args.start):]
    elif args.first is not None:
        chosen=EXPERIMENTS[:args.first]
    cfg=show_plan(chosen)
    if args.plan:
        return
    subprocess.run(
        [sys.executable,"experiments/no31_cora_original/test_night10_cpu.py"],
        cwd=str(ROOT),check=True,
    )
    if not args.skip_fixed_r0_precheck:
        core.precheck(FIXED_R0_CHECKPOINT)
    print("[NIGHT10-PREFLIGHT-PASS] CPU controls and historical early-layer check passed",flush=True)
    core.run_trials(chosen,cfg,name_prefix="NO31_CORA_NIGHT10")
    print("[NIGHT10-DONE] Check summary.csv in logs/NO31_CORA_NIGHT10_*",flush=True)

if __name__=="__main__":
    main()
