#!/usr/bin/env python3
"""NIGHT13: one CoRA model with four depth groups and ranks 4/6/8/10."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

ALL = "q_proj,k_proj,v_proj,o_proj"
UNIFORM = "q_proj=1,k_proj=1,v_proj=1,o_proj=1"

EXPERIMENT = dict(
    id="GROUPED_RANK_4_6_8_10",
    title="Four depth groups with rank 4/6/8/10",
    hypothesis=(
        "Use a narrower collaborative residual in early layers and progressively "
        "more rank capacity in deeper layers, while preserving the original CoRA "
        "user-item input, shared QFormer, Q/K/V/O targets, and residual scale."
    ),
    rank_profile="4-6-8-10",
    alpha=48,
    projections=ALL,
    layers="0-31",
    cf_input="concat",
    injection_mode="shared",
    p_lr=1.0,
    residual_ratio=0.0,
    depth_profile="uniform",
    projection_scales=UNIFORM,
    seed=42,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the grouped-rank CoRA model (L0-7:4, L8-15:6, L16-23:8, L24-31:10)"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()

    rows = [EXPERIMENT]
    cfg = core.plan(rows)
    print(
        "[NIGHT13-PLAN] L0-L7 rank=4; L8-L15 rank=6; "
        "L16-L23 rank=8; L24-L31 rank=10; all Q/K/V/O; "
        "shared QFormer and one rank-10 output head, sliced and renormalized per group; "
        "alpha/r fixed at 3.0 in every group; seed=42; one epoch; Valid-only.",
        flush=True,
    )
    if args.plan:
        return

    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night13_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT13")
    print(
        "[NIGHT13-DONE] See logs/NO31_CORA_NIGHT13_<timestamp>/summary.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
