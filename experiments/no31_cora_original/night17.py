#!/usr/bin/env python3
"""NIGHT17: test whether CoRA's collaborative mapping should refine with depth."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

ALL = "q_proj,k_proj,v_proj,o_proj"
UNIFORM = "q_proj=1,k_proj=1,v_proj=1,o_proj=1"

# Keep the historical P4 configuration as a same-session anchor. New schedules
# change only the layer-to-projector mapping/architecture, never the seed.
EXPERIMENTS = [
    dict(
        id="E0_UNIFORM_4LAYER_CONTROL",
        title="Control: eight independent projectors, four Transformer layers each",
        group_projectors="group8",
        anchor_auc=0.7299284744316927,
        anchor_uauc=0.6866122315446266,
        anchor_tolerance=0.02,
    ),
    dict(
        id="E1_COARSE_TO_FINE",
        title="Coarse-to-fine: 8L groups, then 4L groups, then 2L groups",
        group_projectors="coarse_to_fine",
    ),
    dict(
        id="E2_FINE_TO_COARSE",
        title="Fine-to-coarse: 2L groups, then 4L groups, then 8L groups",
        group_projectors="fine_to_coarse",
    ),
    dict(
        id="E3_HIERARCHICAL_RESIDUAL",
        title="Shared residual mapping plus progressively stronger group residuals",
        group_projectors="hierarchical_residual",
    ),
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
    if int(cfg["run"]["seed"]) != 42:
        raise RuntimeError("NIGHT17 requires the existing seed=42; do not change it")
    if len(EXPERIMENTS) != 4 or len(BY_ID) != 4:
        raise RuntimeError("NIGHT17 must define exactly four experiment arms")
    expected_modes = [
        "group8", "coarse_to_fine", "fine_to_coarse", "hierarchical_residual"
    ]
    if [row["group_projectors"] for row in EXPERIMENTS] != expected_modes:
        raise RuntimeError("NIGHT17 projector mode order/configuration changed")
    for row in EXPERIMENTS:
        if (
            row["seed"] != 42
            or row["alpha"] != 48
            or row["rank_profile"] != "uniform"
            or row["projections"] != ALL
            or row["layers"] != "0-31"
            or row["cf_input"] != "concat"
            or row["injection_mode"] != "shared"
            or row["p_lr"] != 1.0
            or row["residual_ratio"] != 0.0
            or row["depth_profile"] != "uniform"
        ):
            raise RuntimeError(f"Unexpected non-architectural change in {row['id']}")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="NIGHT17 progressive collaborative-mapping experiments"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--only", choices=tuple(BY_ID))
    action.add_argument("--start", choices=tuple(BY_ID))
    args = parser.parse_args()

    cfg = validate()
    rows = EXPERIMENTS
    if args.only:
        rows = [BY_ID[args.only]]
    elif args.start:
        rows = EXPERIMENTS[list(BY_ID).index(args.start):]

    core.plan(rows)
    print(
        "[NIGHT17-HYPOTHESIS] Compare progressive group granularity, not optimizer "
        "hyperparameters. All arms: seed42, rank16, alpha48, QKVO/all32, "
        "concat CF input, p_lr=1, CTR=.05, KD=.05, fresh init, one epoch, Valid-only.",
        flush=True,
    )
    print(
        "[NIGHT17-SCHEDULE] control=uniform 4-layer groups; "
        "E1=8L->8L->4L->4L->2L->2L->2L->2L; "
        "E2=2L->2L->2L->2L->4L->4L->8L->8L; "
        "E3=shared residual plus depth-scaled group residuals on the E1 schedule.",
        flush=True,
    )
    if args.plan:
        return

    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night17_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    fixed_r0 = (
        ROOT / "output/NO31_CORA_ORIGINAL_20261023_234005_110337/"
        "20261023234/checkpoint_0.pth"
    )
    core.precheck(fixed_r0)
    print("[NIGHT17-PREFLIGHT-PASS] CPU mapping tests + early-layer checkpoint check",
          flush=True)
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT17")
    print("[NIGHT17-DONE] Find summary.csv in logs/NO31_CORA_NIGHT17_<timestamp>/",
          flush=True)


if __name__ == "__main__":
    main()
