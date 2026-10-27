"""NIGHT22: exactly eight approved loss settings on original NIGHT13 rank 4/6/8/10."""
WEIGHTS_ALL_ON = (0.05, 0.05, 1e-4, 5e-4)
LOSS_SPECS = {
    "n22_base":          WEIGHTS_ALL_ON,
    "n22_no_ctr":        (0.0, 0.05, 1e-4, 5e-4),
    "n22_no_kd":         (0.05, 0.0, 1e-4, 5e-4),
    "n22_no_ctr_kd":     (0.0, 0.0, 1e-4, 5e-4),
    "n22_no_sem":        (0.05, 0.05, 0.0, 5e-4),
    "n22_no_proj":       (0.05, 0.05, 1e-4, 0.0),
    "n22_no_sem_proj":   (0.05, 0.05, 0.0, 0.0),
    "n22_all_off":       (0.0, 0.0, 0.0, 0.0),
}


def verify_rank410(runtime):
    """Refuse CoRA group8/16, NIGHT18/19/20 signals or changed layer scales."""
    required = {
        "rank_profile": (4, 6, 8, 10),
        "group_projector_mode": "off",
        "granularity_mode": "original",
        "fine_residual_mode": "off",
        "collab_mode": "off",
        "input_mode": "concat",
        "injection_mode": "shared",
        "layers": tuple(range(32)),
        "projections": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "alpha": 48,
        "rank": 16,
    }
    for key, expected in required.items():
        got = getattr(runtime, key)
        if got != expected:
            raise RuntimeError(
                f"NIGHT22 must use exact NIGHT13 rank410 CoRA: "
                f"{key}={got!r}, expected={expected!r}"
            )
    if float(runtime.scaling) != 3.0:
        raise RuntimeError("NIGHT22 requires alpha/r=3 for all rank groups")
