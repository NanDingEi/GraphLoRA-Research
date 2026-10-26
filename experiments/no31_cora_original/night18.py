#!/usr/bin/env python3
"""NIGHT18: explicit collaborative granularity, same E0 CoRA architecture.

Runs: fine-only, coarse-only rank gate, dual rank gate.
Historical NIGHT17 E0 is the fixed baseline; it is NOT retrained.
All preserve rank16 / alpha48 / QKVO 0-31 / group8 / seed42 / CTR+KD,
with no Transformer-layer schedule change.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night8 as core

BASE = dict(
    layers="0-31", projections="q_proj,k_proj,v_proj,o_proj",
    cf_input="concat", injection_mode="shared", rank_profile="uniform",
    group_projectors="group8", alpha=48, p_lr=1.0,
    residual_ratio=0.0, depth_profile="uniform",
    projection_scales="q_proj=1,k_proj=1,v_proj=1,o_proj=1", seed=42,
)

E0_VALID_AUC = 0.7299284744316927
E0_VALID_UAUC = 0.6866122315446266

EXPERIMENTS = [
    dict(id="G1_FINE_ONLY", title="Replace CoRA condition with target-history interaction",
         granularity_mode="fine"),
    dict(id="G2_COARSE_GATE", title="Keep E0 condition, learn 8-rank coarse gate",
         granularity_mode="coarse_gate"),
    dict(id="G3_DUAL_GATE", title="Keep E0 condition, fine signal controls second 8 ranks",
         granularity_mode="dual"),
]
for row in EXPERIMENTS:
    row.update(BASE)
BY_ID = {row["id"]: row for row in EXPERIMENTS}


def main():
    p = argparse.ArgumentParser()
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--only", choices=tuple(BY_ID))
    args = p.parse_args()

    cfg = core.plan([BY_ID[args.only]] if args.only else EXPERIMENTS)
    print(
        f"[NIGHT18-FIXED-BASELINE] NIGHT17 E0 historic Valid "
        f"AUC={E0_VALID_AUC:.9f} UAUC={E0_VALID_UAUC:.9f}; "
        "reuse checkpoint/results, never rerun baseline.",
        flush=True,
    )
    if args.plan:
        return
    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night18_cpu.py"],
        cwd=ROOT, check=True,
    )
    # The CPU test validates feature construction and rank routing but is not
    # a substitute for a full checkpoint/logits parity check on the server.
    core.run_trials(
        [BY_ID[args.only]] if args.only else EXPERIMENTS,
        cfg, name_prefix="NO31_CORA_NIGHT18",
    )


if __name__ == "__main__":
    main()
