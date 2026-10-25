#!/usr/bin/env python3
"""NIGHT14: parameter-matched depth-wise CoRA ranks 12/14/18/20."""
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
    id="BUDGET_MATCHED_RANK_12_14_18_20",
    title="Depth-redistributed ranks 12/14/18/20 at matched CoRA-P budget",
    rank_profile="12-14-18-20",
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

G2_RANK = 16
RANKS = (12, 14, 18, 20)


def validate():
    cfg = core.prepared_config()
    assert tuple(map(int, EXPERIMENT["rank_profile"].split("-"))) == RANKS
    assert sum(RANKS) == 4 * G2_RANK
    assert int(cfg["run"]["seed"]) == EXPERIMENT["seed"] == 42
    assert float(cfg["model"]["gnn_scale"]) == 0
    assert EXPERIMENT["alpha"] / G2_RANK == 3.0
    assert EXPERIMENT["projections"] == ALL
    assert EXPERIMENT["cf_input"] == "concat"
    assert EXPERIMENT["injection_mode"] == "shared"
    assert EXPERIMENT["p_lr"] == 1.0
    assert EXPERIMENT["residual_ratio"] == 0.0
    return cfg


def print_plan():
    cfg = validate()
    core.plan([EXPERIMENT])
    hidden, cf_dim, extension = 4096, 512, 4
    num_p_g2 = 32 * 4 * hidden * G2_RANK
    num_p_new = sum(8 * 4 * hidden * r for r in RANKS)
    base_head = cf_dim * (hidden * G2_RANK) + hidden * G2_RANK
    # 512->32->(4096*4) with biases.
    extension_head = cf_dim * 32 + 32 + 32 * (hidden * extension) + hidden * extension
    combined_base = num_p_g2 + base_head
    overhead_pct = 100.0 * extension_head / combined_base
    print(
        "[NIGHT14-PARAMS] "
        f"G2_P={num_p_g2:,} new_P={num_p_new:,} "
        f"G2_main_head={base_head:,} "
        f"extra_head={extension_head:,} "
        f"relative_extra_vs_(P+main_head)={overhead_pct:.3f}% "
        f"alpha_over_rank=3.0",
        flush=True,
    )
    print(
        "[NIGHT14-METHOD] L0-L7=12 L8-L15=14 L16-L23=18 L24-L31=20; "
        "four Q/K/V/O projections per layer; original QFormer and "
        "rank-16 head preserved; compact nonlinear 4-channel extension; "
        "one epoch Valid-only, seed42, GPU0 foreground",
        flush=True,
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="CoRA grouped ranks with G2-matched P budget")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()

    cfg = print_plan()
    if args.plan:
        return
    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night14_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    core.run_trials([EXPERIMENT], cfg, name_prefix="NO31_CORA_NIGHT14")
    print("[NIGHT14-DONE] See logs/NO31_CORA_NIGHT14_<timestamp>/summary.csv", flush=True)


if __name__ == "__main__":
    main()
