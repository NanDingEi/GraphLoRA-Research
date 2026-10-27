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
from experiments.no31_cora_original.night21_loss_guard import (
    BASE_WEIGHTS, LOSS_SPECS, validate_cora_loss_controls,
)
from types import SimpleNamespace


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
    # This exact callable is invoked by CoRA adapter.init_patch after model
    # construction. A plain YAML/env test missed the earlier hard-coded gate.
    e0_runtime = SimpleNamespace(
        group_projector_mode="group8", granularity_mode="original",
        fine_residual_mode="off", collab_mode="off",
        input_mode="concat", injection_mode="shared",
        layers=tuple(range(32)),
        projections=("q_proj", "k_proj", "v_proj", "o_proj"),
        alpha=48, rank=16,
    )
    def mock_model(weights):
        return SimpleNamespace(
            gnn_conf_loss_weight=weights[0], s2_small_kd=weights[1],
            gnn_semantic_weight=weights[2], proj_reg_weight=weights[3],
            s2_input_alpha=0., gnn_semantic_align=True,
        )
    assert validate_cora_loss_controls(
        mock_model(BASE_WEIGHTS), e0_runtime, ""
    ) == BASE_WEIGHTS
    for arm, weights in LOSS_SPECS.items():
        assert validate_cora_loss_controls(
            mock_model(weights), e0_runtime, arm
        ) == weights
        for wrong_arm in LOSS_SPECS:
            if wrong_arm == arm:
                continue
            try:
                validate_cora_loss_controls(
                    mock_model(weights), e0_runtime, wrong_arm
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    f"Disallowed mismatch accepted: {arm} as {wrong_arm}"
                )
    wrong_runtime = SimpleNamespace(
        **{**vars(e0_runtime), "collab_mode": "coarse_only"}
    )
    try:
        validate_cora_loss_controls(
            mock_model(LOSS_SPECS["no_ctr"]), wrong_runtime, "no_ctr"
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("NIGHT20 architecture incorrectly allowed for NIGHT21")
    try:
        validate_cora_loss_controls(
            mock_model(LOSS_SPECS["no_ctr"]), e0_runtime, ""
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Historical E0 strict default unexpectedly relaxed")
    print("[NIGHT21-CPU] shared adapter contract: original E0 + all "
          "4 legal loss arms accepted; mismatched loss and NIGHT20 rejected PASS",
          flush=True)
    assert config["model"]["proj_reg_weight"] == .0005
    assert config["model"]["gnn_semantic_weight"] == .0001
    assert config["model"]["gnn_conf_loss_weight"] == .05
    print("[NIGHT21-CPU-PASS] 4 loss-ablation arms match historical "
          "NIGHT17 E0 architecture, fixed seed and independent env flags",
          flush=True)


if __name__ == "__main__":
    main()
