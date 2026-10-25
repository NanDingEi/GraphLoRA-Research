#!/usr/bin/env python3
"""NIGHT16: eight projector-architecture experiments; G2 is a fixed historical baseline."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

ALL = "q_proj,k_proj,v_proj,o_proj"
UNIFORM = "q_proj=1,k_proj=1,v_proj=1,o_proj=1"

# Eight model-structure variants. P0/G2 is deliberately NOT rerun or counted.
# G2 historical valid metrics: AUC=0.7301218951846701, UAUC=0.68804101129963.
EXPERIMENTS = [
    dict(id="P1_SHARED_PROJECTOR",
         title="One shared residual MLP across all four layer groups",
         group_projectors="shared"),
    dict(id="P2_FOUR_GROUP_PROJECTORS",
         title="Four independent residual MLPs, one per 8-layer group",
         group_projectors="group4"),
    dict(id="P3_TWO_GROUP_PROJECTORS",
         title="Two independent residual MLPs, one per 16-layer group",
         group_projectors="group2"),
    dict(id="P4_EIGHT_GROUP_PROJECTORS",
         title="Eight independent residual MLPs, one per 4-layer group",
         group_projectors="group8"),
    dict(id="P5_SHARED_TRUNK4",
         title="Shared 512-to-32 trunk with four group-specific output heads",
         group_projectors="shared_trunk4"),
    dict(id="P6_SHARED_PLUS_GROUP4",
         title="Shared residual MLP plus four group-specific residual MLPs",
         group_projectors="shared_group4"),
    dict(id="P7_RANK_SPACE_GROUP4",
         title="Four group-specific 16x16 rank-space basis transforms",
         group_projectors="rank_group4"),
    dict(id="P8_LINEAR_BOTTLENECK_GROUP4",
         title="Four group-specific linear 512-to-32-to-512 bottlenecks",
         group_projectors="group4_linear"),
]
for row in EXPERIMENTS:
    row.update(
        rank_profile="uniform",
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
BY_ID = {row["id"]: row for row in EXPERIMENTS}


def validate():
    cfg = core.prepared_config()
    assert len(EXPERIMENTS) == len(BY_ID) == 8
    expected_modes = [
        "shared", "group4", "group2", "group8",
        "shared_trunk4", "shared_group4", "rank_group4", "group4_linear",
    ]
    assert [r["group_projectors"] for r in EXPERIMENTS] == expected_modes
    assert int(cfg["run"]["seed"]) == 42
    for row in EXPERIMENTS:
        assert row["alpha"] == 48 and row["seed"] == 42
        assert row["rank_profile"] == "uniform"
        assert row["projections"] == ALL and row["layers"] == "0-31"
        assert row["cf_input"] == "concat" and row["injection_mode"] == "shared"
        assert row["p_lr"] == 1.0 and row["residual_ratio"] == 0.0
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run eight one-epoch CoRA collaborative-projector architecture experiments"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--only", choices=tuple(BY_ID))
    action.add_argument("--start", choices=tuple(BY_ID))
    args = parser.parse_args()
    if sum(x is not None for x in (args.only, args.start)) > 1:
        parser.error("Use at most one of --only and --start")

    cfg = validate()
    rows = EXPERIMENTS
    if args.only:
        rows = [BY_ID[args.only]]
    elif args.start:
        rows = EXPERIMENTS[list(BY_ID).index(args.start):]

    core.plan(rows)
    print(
        "[NIGHT16-COMMON] Fixed G2 historical baseline AUC=0.730121895 "
        "UAUC=0.688041; NOT rerun. All eight variants use seed42, alpha48, "
        "rank16, QKVO/all32, concat input, P-LR=1, CTR=.05, KD=.05, "
        "NO31 gnn_scale=0, fresh initialization, one epoch, Valid-only, GPU0 serial.",
        flush=True,
    )
    print(
        "[NIGHT16-IDEA] P1 shared; P2 four independent 8-layer groups; "
        "P3 two independent 16-layer groups; P4 eight independent 4-layer groups; "
        "P5 shared trunk/four heads; P6 shared plus group-specific residuals; "
        "P7 rank-space group transforms; P8 parameter-matched linear bottlenecks.",
        flush=True,
    )
    if args.plan:
        return

    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night16_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    # Retain the historical check that early-layer CoRA-P weights were trained.
    fixed_r0 = (
        ROOT / "output/NO31_CORA_ORIGINAL_20261023_234005_110337/"
        "20261023234/checkpoint_0.pth"
    )
    core.precheck(fixed_r0)
    print("[NIGHT16-PREFLIGHT-PASS] architecture tests + early-layer checkpoint check",
          flush=True)
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT16")
    print("[NIGHT16-DONE] Find summary.csv in logs/NO31_CORA_NIGHT16_<timestamp>/",
          flush=True)


if __name__ == "__main__":
    main()
