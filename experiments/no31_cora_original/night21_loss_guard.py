"""Strict NIGHT21 approved one-loss ablation contract.

Shared by CoRA adapter runtime and CPU preflight so historical E0 remains
locked down, while the four one-factor NIGHT21 experiments are actually
accepted by the adapter during model initialization.
"""
from math import isfinite

BASE_WEIGHTS = (0.05, 0.05, 1e-4, 5e-4)
LOSS_SPECS = {
    "no_ctr": (0.0, 0.05, 1e-4, 5e-4),
    "no_kd": (0.05, 0.0, 1e-4, 5e-4),
    "no_sem": (0.05, 0.05, 0.0, 5e-4),
    "no_proj": (0.05, 0.05, 1e-4, 0.0),
}


def validate_cora_loss_controls(model, runtime, loss_arm=""):
    """Verify exact four loss scalars, teacher-path and NIGHT17 E0 architecture."""
    if loss_arm and loss_arm not in LOSS_SPECS:
        raise RuntimeError(f"Unsupported CoRA loss ablation: {loss_arm}")
    expected = LOSS_SPECS[loss_arm] if loss_arm else BASE_WEIGHTS
    actual = (
        float(model.gnn_conf_loss_weight),
        float(model.s2_small_kd),
        float(model.gnn_semantic_weight),
        float(model.proj_reg_weight),
    )
    if (not all(isfinite(w) for w in actual)
            or any(abs(x - y) > 1e-12 for x, y in zip(actual, expected))
            or float(model.s2_input_alpha) != 0.0
            or not bool(model.gnn_semantic_align)):
        raise RuntimeError(
            f"NO31 CoRA loss/semantic controls changed: "
            f"arm={loss_arm or 'E0'} expected={expected} actual={actual} "
            f"input_alpha={model.s2_input_alpha} "
            f"semantic_align={model.gnn_semantic_align}"
        )
    if loss_arm:
        if (runtime.group_projector_mode != "group8"
                or runtime.granularity_mode != "original"
                or runtime.fine_residual_mode != "off"
                or runtime.collab_mode != "off"
                or runtime.input_mode != "concat"
                or runtime.injection_mode != "shared"
                or runtime.layers != tuple(range(32))
                or runtime.projections != ("q_proj", "k_proj", "v_proj", "o_proj")
                or runtime.alpha != 48 or runtime.rank != 16):
            raise RuntimeError(
                "NIGHT21 loss ablation requires exact historical NIGHT17 E0: "
                "group8 rank16 alpha48 QKVO full32, concat input, no NIGHT18-20"
            )
    return actual
