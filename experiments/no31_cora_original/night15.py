#!/usr/bin/env python3
"""NIGHT15: G2 vs shared vs independent four-group collaborative projectors."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

ALL = "q_proj,k_proj,v_proj,o_proj"
UNIFORM = "q_proj=1,k_proj=1,v_proj=1,o_proj=1"
FIXED_R0_CHECKPOINT = (
    ROOT / "output/NO31_CORA_ORIGINAL_20261023_234005_110337/"
    "20261023234/checkpoint_0.pth"
)

EXPERIMENTS = [
    dict(id="P0_G2_CONTROL", title="Original CoRA G2 alpha48 rank16",
         group_projectors="off", anchor_auc=core.G2_AUC,
         anchor_uauc=core.G2_UAUC, anchor_tolerance=0.02),
    dict(id="P1_SHARED_PROJECTOR", title="One shared 512-32-512 collaborative projector",
         group_projectors="shared"),
    dict(id="P2_FOUR_GROUP_PROJECTORS",
         title="Independent 512-32-512 collaborative projector for each 8-layer group",
         group_projectors="group4"),
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
    assert len(EXPERIMENTS) == len(BY_ID) == 3
    assert [r["group_projectors"] for r in EXPERIMENTS] == ["off", "shared", "group4"]
    assert int(cfg["run"]["seed"]) == 42
    for row in EXPERIMENTS:
        assert row["alpha"] == 48
        assert row["seed"] == 42
        assert row["rank_profile"] == "uniform"
        assert row["projections"] == ALL and row["layers"] == "0-31"
        assert row["cf_input"] == "concat" and row["injection_mode"] == "shared"
        assert row["p_lr"] == 1.0 and row["residual_ratio"] == 0.0
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run G2 and two collaborative projection architecture comparisons"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--only", choices=tuple(BY_ID))
    parser.add_argument("--start", choices=tuple(BY_ID))
    parser.add_argument("--first", type=int, choices=(1, 2, 3))
    args = parser.parse_args()

    cfg = validate()
    if sum(x is not None for x in (args.only, args.start, args.first)) > 1:
        parser.error("Use at most one of --only, --start and --first")
    rows = EXPERIMENTS
    if args.only:
        rows = [BY_ID[args.only]]
    elif args.start:
        rows = EXPERIMENTS[list(BY_ID).index(args.start):]
    elif args.first:
        rows = EXPERIMENTS[:args.first]
    core.plan(rows)
    print(
        "[NIGHT15-COMMON] original QFormer 8-layer/4-query; unchanged rank16/alpha48"
        "/QKVO-all32; input=concat; seed42; P LR=1.0; "
        "P0/P1/P2 in foreground serial GPU0; fresh epoch0; Valid-only",
        flush=True,
    )
    print(
        "[NIGHT15-PROJECTORS] P0=none, P1=one shared 512->32->512 residual MLP, "
        "P2=four independent MLPs serving L0-7/L8-15/L16-23/L24-31. "
        "Every MLP is identity-residual at initialization.",
        flush=True,
    )
    if args.plan:
        return
    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night15_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    # The historical early-layer gradient check is common to all three arms.
    core.precheck(FIXED_R0_CHECKPOINT)
    print("[NIGHT15-PREFLIGHT-PASS] CPU architecture tests + historical CoRA-P checked",
          flush=True)
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT15")
    print("[NIGHT15-DONE] Find summary.csv in logs/NO31_CORA_NIGHT15_<timestamp>/",
          flush=True)


if __name__ == "__main__":
    main()
