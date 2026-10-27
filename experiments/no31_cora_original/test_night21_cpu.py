#!/usr/bin/env python3
"""NIGHT21 CPU-only configuration, environment and audit contract check."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import night21
from experiments.no31_cora_original import night17


def main():
    config = night21.validate()
    expected = {
        "A1_NO_CTR": (0., .05, 1e-4, 5e-4),
        "A2_NO_KD": (.05, 0., 1e-4, 5e-4),
        "A3_NO_SEM": (.05, .05, 0., 5e-4),
        "A4_NO_PROJ": (.05, .05, 1e-4, 0.),
    }
    base = night17.EXPERIMENTS[0]
    for row in night21.EXPERIMENTS:
        assert tuple(row[k] for k in (
            "loss_ctr", "loss_kd", "loss_sem", "loss_proj"
        )) == expected[row["id"]]
        for key in (
            "group_projectors", "rank_profile", "alpha", "projections",
            "layers", "cf_input", "injection_mode", "p_lr",
            "residual_ratio", "depth_profile", "projection_scales", "seed",
        ):
            assert row[key] == base[key], (row["id"], key)
        assert core.selected_targets(row) == 128
        env = core.build_environment(row)
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["S2_NIGHT_EPOCHS"] == "1"
        assert float(env["S2_SMALL_KD"]) == row["loss_kd"]
        assert env["S2_INPUT_ALPHA"] == "0"
        assert env["NO31_CORA_LOSS_ABLATION"] == row["loss_ablation"]
        assert env["NO31_CORA_COLLAB_MODE"] == "off"
        assert env["NO31_CORA_FINE_RESIDUAL_MODE"] == "off"
        assert env["NO31_CORA_PROJECTOR_SHADOW"] == (
            "1" if row["loss_ablation"] == "no_proj" else "0"
        )
        print("[NIGHT21-CPU] %s GPU0=YES fresh_E0=YES "
              "disabled_loss=%s PASS" %
              (row["id"], row["loss_ablation"]), flush=True)
    assert config["model"]["proj_reg_weight"] == .0005
    assert config["model"]["gnn_semantic_weight"] == .0001
    assert config["model"]["gnn_conf_loss_weight"] == .05
    print("[NIGHT21-CPU-PASS] 4 loss-ablation arms match historical "
          "NIGHT17 E0 architecture, fixed seed and independent env flags",
          flush=True)


if __name__ == "__main__":
    main()
