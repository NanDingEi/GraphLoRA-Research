#!/usr/bin/env python3
"""NIGHT12: four structural CoRA improvements; no hyperparameter sweep."""
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
    dict(
        id="M1_TYPED_TOKENS",
        title="Represent MF user and candidate as separate typed QFormer tokens",
        injection_mode="typed_tokens",
        hypothesis="QFormer cross-attention can distinguish user-side and item-side collaborative evidence rather than receiving one concatenated encoder token.",
    ),
    dict(
        id="M2_INTERACTION_TOKEN",
        title="Add an explicit user-item compatibility token",
        injection_mode="interaction_tokens",
        hypothesis="An elementwise user-item interaction token exposes pair compatibility directly to the dynamic LoRA-A generator.",
    ),
    dict(
        id="M3_LAYER_CONDITIONED",
        title="Condition the dynamic low-rank basis on Transformer layer identity",
        injection_mode="layer_conditioned",
        hypothesis="Different Transformer layers can receive different rank-channel mixtures of the same user-item-conditioned basis.",
    ),
    dict(
        id="M4_TOKEN_GATE",
        title="Adapt the CoRA residual per LLM token",
        injection_mode="token_gate",
        hypothesis="A shared token-wise gate can modulate collaborative injection according to the current hidden state rather than applying the residual uniformly to every token.",
    ),
]
for row in EXPERIMENTS:
    row.update(
        alpha=48,
        projections=ALL,
        layers="0-31",
        cf_input="concat",
        p_lr=1.0,
        residual_ratio=0.0,
        depth_profile="uniform",
        projection_scales=UNIFORM,
        seed=42,
    )
BY_ID = {row["id"]: row for row in EXPERIMENTS}


def validate_plan():
    cfg = core.prepared_config()
    if len(EXPERIMENTS) != 4 or len(BY_ID) != 4:
        raise RuntimeError("NIGHT12 must contain four unique structural experiments")
    if int(cfg["run"].get("seed", -1)) != 42:
        raise RuntimeError("Prepared seed changed; all experiments must use seed42")
    if float(cfg["model"].get("gnn_scale", -1)) != 0:
        raise RuntimeError("NO31 gnn_scale must remain zero")
    for row in EXPERIMENTS:
        if row["seed"] != 42 or row["alpha"] != 48:
            raise RuntimeError(f"{row['id']} changed seed/alpha")
        if row["layers"] != "0-31" or row["projections"] != ALL:
            raise RuntimeError(f"{row['id']} changed the injection targets")
        if row["p_lr"] != 1.0 or row["residual_ratio"] != 0.0:
            raise RuntimeError(f"{row['id']} changes optimization/residual parameters")
    return cfg


def show_plan(rows):
    cfg = validate_plan()
    print(
        "[NIGHT12-COMMON] seed=42; alpha=48; rank=16; QKVO/all32; "
        "P-LR multiplier=1.0; no residual clipping/scaling; "
        "NO31 MF/GNN/Projector, CTR/KD and data split unchanged; "
        "fresh init; one epoch; Valid-only; batch=2 eval=3 accum=4",
        flush=True,
    )
    for row in rows:
        print(
            "[NIGHT12-PLAN] %s mode=%s :: %s\n  hypothesis: %s"
            % (row["id"], row["injection_mode"], row["title"], row["hypothesis"]),
            flush=True,
        )
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Four structural CoRA experiments; not a hyperparameter sweep"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--only", choices=tuple(BY_ID))
    parser.add_argument("--start", choices=tuple(BY_ID))
    parser.add_argument("--first", type=int, choices=range(1, 5), metavar="N")
    parser.add_argument(
        "--skip-fixed-r0-precheck", action="store_true",
        help="Use only if the historical checkpoint is absent and early-layer CoRA-P gradients were already verified",
    )
    args = parser.parse_args()
    if sum(value is not None for value in (args.only, args.start, args.first)) > 1:
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
        [sys.executable, "experiments/no31_cora_original/test_night12_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    if not args.skip_fixed_r0_precheck:
        core.precheck(FIXED_R0_CHECKPOINT)
    print("[NIGHT12-PREFLIGHT-PASS] architecture checks and early-layer CoRA-P check passed",
          flush=True)
    core.run_trials(rows, cfg, name_prefix="NO31_CORA_NIGHT12")
    print("[NIGHT12-DONE] See logs/NO31_CORA_NIGHT12_<timestamp>/summary.csv",
          flush=True)


if __name__ == "__main__":
    main()
