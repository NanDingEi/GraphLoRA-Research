#!/usr/bin/env python3
"""NIGHT21: four one-factor auxiliary loss deletions on healthy NIGHT17 E0 CoRA.

All conditions copy NIGHT17 E0 exactly (K=8 group projectors over 32 layers,
Q/K/V/O, rank16 alpha48, frozen pretrained MF pair, no NIGHT18-20 features).
Only one loss coefficient is zero per fresh single-epoch run. Valid only,
GPU0, seed42, serial foreground, no baseline retraining.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import night17

E0_AUC = 0.7299284744316927
E0_UAUC = 0.6866122315446266

# Borrow exactly the tested healthy CoRA E0 architecture; NEVER use NIGHT20 T1.
E0 = night17.EXPERIMENTS[0]
if E0["id"] != "E0_UNIFORM_4LAYER_CONTROL":
    raise RuntimeError("Historical NIGHT17 E0 anchor is missing")

EXPERIMENTS = [
    dict(id="A1_NO_CTR", title="NIGHT17 E0 minus GNN CTR BCE",
         loss_ablation="no_ctr", loss_ctr=0., loss_kd=.05,
         loss_sem=1e-4, loss_proj=5e-4),
    dict(id="A2_NO_KD", title="NIGHT17 E0 minus LLM-to-GNN KD",
         loss_ablation="no_kd", loss_ctr=.05, loss_kd=0.,
         loss_sem=1e-4, loss_proj=5e-4),
    dict(id="A3_NO_SEM", title="NIGHT17 E0 minus semantic loss (teacher still run)",
         loss_ablation="no_sem", loss_ctr=.05, loss_kd=.05,
         loss_sem=0., loss_proj=5e-4),
    dict(id="A4_NO_PROJ", title="NIGHT17 E0 minus llama_proj structure loss (shadow sampling)",
         loss_ablation="no_proj", loss_ctr=.05, loss_kd=.05,
         loss_sem=1e-4, loss_proj=0.),
]
for row in EXPERIMENTS:
    # Keep only intended E0 architecture/optimizer fields, not E0 anchor and
    # not NIGHT17 extra metadata or historical tolerance.
    for key in ("group_projectors", "rank_profile", "alpha", "projections",
                "layers", "cf_input", "injection_mode", "p_lr",
                "residual_ratio", "depth_profile", "projection_scales", "seed"):
        row[key] = E0[key]
    row.update(
        granularity_mode="original", fine_residual_mode="off",
        collab_mode="off",
    )
BY_ID = {x["id"]: x for x in EXPERIMENTS}


def validate():
    cfg = core.prepared_config()  # Assert ORIGINAL all-on YAML has not drifted.
    model = cfg["model"]
    runner = cfg["run"]
    if model.get("sem_adapter_on", False):
        raise RuntimeError(
            "This NIGHT21 recipe expects NIGHT17 E0 using legacy "
            "gnn_semantic_weight; sem_adapter_on was enabled. "
            "Do not run A3 until sem_corr_weight is correctly controlled."
        )
    if model.get("pref_distill", False):
        raise RuntimeError("NIGHT21 expects preference distillation disabled in E0")
    if int(runner["seed"]) != 42:
        raise RuntimeError("Seed must be fixed at 42")
    if len(EXPERIMENTS) != 4 or len(BY_ID) != 4:
        raise RuntimeError("Require four unique ablation IDs")
    ref = (.05, .05, 1e-4, 5e-4)
    names = ("loss_ctr", "loss_kd", "loss_sem", "loss_proj")
    for index, row in enumerate(EXPERIMENTS):
        got = tuple(float(row[k]) for k in names)
        if any(abs(x) > 1e-12 for x in got[index:index + 1]):
            raise RuntimeError("The ablated loss must be zero")
        for idx, (x, y) in enumerate(zip(got, ref)):
            if idx != index and abs(x - y) > 1e-12:
                raise RuntimeError(f"{row['id']} changed more than one loss")
        if row["group_projectors"] != "group8" or row["alpha"] != 48:
            raise RuntimeError("NIGHT17 E0 architecture mismatch")
        if row["seed"] != 42 or row["collab_mode"] != "off":
            raise RuntimeError("NIGHT21 cannot use NIGHT20 conditions")
    return cfg


def main():
    parser = argparse.ArgumentParser()
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

    print("[NIGHT21-E0] Historical healthy NIGHT17 E0 Valid AUC="
          f"{E0_AUC:.9f}, UAUC={E0_UAUC:.9f}; no E0 retrain.", flush=True)
    print("[NIGHT21-CONFIG] Original E0 group8 rank16 alpha48 32-layer QKVO; "
          "MF concat, NO31, no NIGHT19/20 features; seed42; 1 epoch per arm, "
          "Valid only, GPU0 serial foreground.", flush=True)
    for row in rows:
        print("[NIGHT21-PLAN] %s CTR=%g KD=%g SEM=%g PROJ=%g" % (
            row["id"], row["loss_ctr"], row["loss_kd"],
            row["loss_sem"], row["loss_proj"]
        ), flush=True)
    if args.plan:
        return

    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night21_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    # Already runs historical NO31 CoRA smoke and checks GPU0 idleness.
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT21")


if __name__ == "__main__":
    main()
