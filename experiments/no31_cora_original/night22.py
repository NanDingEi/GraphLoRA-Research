#!/usr/bin/env python3
"""NIGHT22 8-run loss ablation on EXACT NIGHT13 rank4/6/8/10 original CoRA.

R0 reruns all-on baseline on the same branch and pretraining used by R1-R7.
If the independently retrained baseline deviates >0.015 AUC or UAUC from the
user-reported historical Valid (.727748, .683007), STOP before R1.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import night13
from experiments.no31_cora_original.night22_loss_spec import LOSS_SPECS

HISTORIC_AUC = 0.727748  # User-reported Valid, not read from repository logs.
HISTORIC_UAUC = 0.683007
ANCHOR_TOLERANCE = 0.015
ARCH = night13.EXPERIMENT
if ARCH["id"] != "GROUPED_RANK_4_6_8_10":
    raise RuntimeError("NIGHT13 canonical grouped rank CoRA experiment not found")

RECIPES = [
    ("R0_BASE_ALL_ON", "n22_base", "Fresh repeat NIGHT13 all auxiliary losses on"),
    ("R1_NO_CTR", "n22_no_ctr", "KD without CTR; semantic/projector retained"),
    ("R2_NO_KD", "n22_no_kd", "CTR without KD; semantic/projector retained"),
    ("R3_NO_CTR_KD", "n22_no_ctr_kd", "No CTR or KD; semantic/projector retained"),
    ("R4_NO_SEM", "n22_no_sem", "No LLM-to-GNN semantic loss"),
    ("R5_NO_PROJ", "n22_no_proj", "No llama_proj item-similarity preservation"),
    ("R6_NO_SEM_PROJ", "n22_no_sem_proj", "No semantic or projector regularization"),
    ("R7_ALL_OFF", "n22_all_off", "Only L_rec; all 4 auxiliary coefficients zero"),
]
EXPERIMENTS = []
for ident, arm, description in RECIPES:
    weights = LOSS_SPECS[arm]
    row = dict(
        id=ident, title=description, loss_ablation=arm,
        loss_ctr=weights[0], loss_kd=weights[1],
        loss_sem=weights[2], loss_proj=weights[3],
    )
    # Exact NIGHT13 structural settings, including rank4/6/8/10 and
    # group_projectors=off (not NIGHT17's rank16/group8).
    for key in (
        "rank_profile", "alpha", "projections", "layers", "cf_input",
        "injection_mode", "p_lr", "residual_ratio", "depth_profile",
        "projection_scales", "seed",
    ):
        row[key] = ARCH[key]
    row.update(group_projectors="off", granularity_mode="original",
               fine_residual_mode="off", collab_mode="off")
    EXPERIMENTS.append(row)
EXPERIMENTS[0].update(
    anchor_auc=HISTORIC_AUC,
    anchor_uauc=HISTORIC_UAUC,
    anchor_tolerance=ANCHOR_TOLERANCE,
)
BY_ID = {r["id"]: r for r in EXPERIMENTS}


def validate():
    cfg = core.prepared_config()
    # The prepared input must still be the *unchanged* original E0 all-on
    # config, not a previous ablation's train.yaml.
    m, r = cfg["model"], cfg["run"]
    if m.get("sem_adapter_on", False):
        raise RuntimeError(
            "NIGHT22 requires NIGHT13's legacy semantic loss; "
            "sem_adapter_on unexpectedly enabled"
        )
    if m.get("pref_distill", False):
        raise RuntimeError("NIGHT22 requires preference distillation disabled")
    if len(EXPERIMENTS) != len(BY_ID) != 8:
        raise RuntimeError("Expected precisely 8 distinct experiments")
    if list(BY_ID) != [x[0] for x in RECIPES]:
        raise RuntimeError("NIGHT22 experiment order changed")
    for row in EXPERIMENTS:
        for key in (
            "rank_profile", "alpha", "projections", "layers", "cf_input",
            "injection_mode", "p_lr", "residual_ratio", "depth_profile",
            "projection_scales", "seed",
        ):
            if row[key] != ARCH[key]:
                raise RuntimeError(
                    f"{row['id']} differs from NIGHT13 on {key}"
                )
        if (row["rank_profile"] != "4-6-8-10" or row["group_projectors"] != "off"
                or row["alpha"] != 48 or row["seed"] != 42
                or row["collab_mode"] != "off"):
            raise RuntimeError("NIGHT22 must be exactly NIGHT13, seed42")
        actual = (row["loss_ctr"], row["loss_kd"], row["loss_sem"], row["loss_proj"])
        if actual != LOSS_SPECS[row["loss_ablation"]]:
            raise RuntimeError("NIGHT22 per-arm loss weights altered")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="8 one-epoch rank4/6/8/10 CoRA loss ablations, R0 rerun first"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    cfg = validate()
    print(
        "[NIGHT22-REFERENCE] rank4/6/8/10 historical user-reported Valid "
        f"AUC={HISTORIC_AUC:.6f} UAUC={HISTORIC_UAUC:.6f}; "
        f"R0 MUST be retrained, abort if difference exceeds {ANCHOR_TOLERANCE:.3f}",
        flush=True,
    )
    for row in EXPERIMENTS:
        print(
            f"[NIGHT22-PLAN] {row['id']} rank=4/6/8/10 alpha=48 "
            f"CTR={row['loss_ctr']:g} KD={row['loss_kd']:g} "
            f"SEM={row['loss_sem']:g} PROJ={row['loss_proj']:g}",
            flush=True,
        )
    if args.plan:
        return

    for script in ("test_night13_cpu.py", "test_night22_cpu.py"):
        subprocess.run(
            [sys.executable, f"experiments/no31_cora_original/{script}"],
            cwd=str(ROOT), check=True,
        )
    print("[NIGHT22-PREFLIGHT-PASS] rank410 original CoRA + 8-arm loss guards",
          flush=True)
    core.run_trials(
        EXPERIMENTS, cfg, name_prefix="NO31_CORA_NIGHT22"
    )


if __name__ == "__main__":
    main()
