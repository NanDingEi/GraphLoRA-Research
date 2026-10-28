#!/usr/bin/env python3
"""NIGHT22 CPU-only: 8 factorial loss recipes + exact NIGHT13 runtime guard."""
import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original import night22, night8
from experiments.no31_cora_original.night22_loss_spec import (
    LOSS_SPECS, verify_rank410,
)
from experiments.no31_cora_original.night21_loss_guard import (
    validate_cora_loss_controls,
)

def fail(action):
    try:
        action()
    except RuntimeError:
        return
    raise AssertionError("Unexpected acceptance of invalid ablation")


def mock_model(w):
    return SimpleNamespace(
        gnn_conf_loss_weight=w[0], s2_small_kd=w[1],
        gnn_semantic_weight=w[2], proj_reg_weight=w[3],
        s2_input_alpha=0., gnn_semantic_align=True,
    )


def mock_rank410():
    return SimpleNamespace(
        rank_profile=(4, 6, 8, 10), group_projector_mode="off",
        granularity_mode="original", fine_residual_mode="off",
        collab_mode="off", input_mode="concat", injection_mode="shared",
        layers=tuple(range(32)),
        projections=("q_proj", "k_proj", "v_proj", "o_proj"),
        alpha=48, rank=16, scaling=3.0,
    )


def main():
    # Reproduce the exact way the GPU runner launches entry.py:
    # execute by file path with no inherited PYTHONPATH.
    # Regression guard for imports attempted before adding repo root to sys.path.
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": "",
        "CUDA_VISIBLE_DEVICES": "0",
        "S2_INPUT_ALPHA": "0",
        "S2_SMALL_KD": "0.05",
        "NO31_CORA_LOSS_ABLATION": "n22_base",
    })
    import_probe = subprocess.run(
        [
            sys.executable, "-I", "experiments/no31_cora_original/entry.py",
            "--cfg-path", "/unused/night22/cpu-import-check.yaml",
            "--check-night22-imports",
        ],
        cwd=str(ROOT), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    if import_probe.returncode or "[NIGHT22-DIRECT-ENTRY-IMPORT-PASS]" not in import_probe.stdout:
        raise RuntimeError(
            "NIGHT22 direct entry.py import preflight failed:\n"
            + import_probe.stdout
        )
    print("[NIGHT22-CPU] isolated direct entry.py imports PASS", flush=True)
    night22.validate()
    expected_ids = (
        "R0_BASE_ALL_ON", "R1_NO_CTR", "R2_NO_KD", "R3_NO_CTR_KD",
        "R4_NO_SEM", "R5_NO_PROJ", "R6_NO_SEM_PROJ", "R7_ALL_OFF",
    )
    assert tuple(r["id"] for r in night22.EXPERIMENTS) == expected_ids
    reference = night22.EXPERIMENTS[0]
    assert reference["anchor_auc"] == .727748
    assert reference["anchor_uauc"] == .683007
    assert reference["anchor_tolerance"] == .015
    runtime = mock_rank410()
    verify_rank410(runtime)
    for row in night22.EXPERIMENTS:
        arm = row["loss_ablation"]
        values = LOSS_SPECS[arm]
        assert values == tuple(row[x] for x in (
            "loss_ctr", "loss_kd", "loss_sem", "loss_proj"
        ))
        assert validate_cora_loss_controls(
            mock_model(values), runtime, arm,
        ) == values
        for other in LOSS_SPECS:
            if other != arm:
                fail(lambda: validate_cora_loss_controls(
                    mock_model(values), runtime, other,
                ))
        env = night8.build_environment(row)
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["S2_NIGHT_EPOCHS"] == "1"
        assert env["S2_INPUT_ALPHA"] == "0"
        assert env["NO31_CORA_LOSS_ABLATION"] == arm
        assert env["NO31_CORA_GROUP_PROJECTORS"] == "off"
        assert env["NO31_CORA_RANK_PROFILE"] == "4-6-8-10"
        assert float(env["S2_SMALL_KD"]) == values[1]
        assert int(env["NO31_CORA_ALPHA"]) == 48
        assert env["NO31_CORA_PROJECTOR_SHADOW"] == (
            "1" if values[3] == 0 else "0"
        )
        assert night8.selected_targets(row) == 128
        print(
            f"[NIGHT22-CPU] {row['id']} CTR={values[0]:g} "
            f"KD={values[1]:g} SEM={values[2]:g} PROJ={values[3]:g} "
            "rank410=PASS model_config=PASS",
            flush=True,
        )

    for key, value in (("rank_profile", None), ("group_projector_mode","group8"),
                       ("collab_mode", "coarse_only"), ("scaling", 4.0)):
        wrong = SimpleNamespace(**vars(runtime))
        setattr(wrong, key, value)
        fail(lambda: verify_rank410(wrong))
        fail(lambda: validate_cora_loss_controls(
            mock_model(LOSS_SPECS["n22_base"]),wrong,"n22_base"
        ))

    fail(lambda: validate_cora_loss_controls(
        mock_model(LOSS_SPECS["n22_no_ctr"]),runtime,""
    ))
    for path in (
        "experiments/no31_cora_original/night22.py",
        "experiments/no31_cora_original/night8.py",
        "experiments/no31_cora_original/entry.py",
        "experiments/no31_cora_original/adapter.py",
        "experiments/no31_cora_original/night22_loss_spec.py",
        "experiments/no31_cora_original/night21_loss_guard.py",
        "minigpt4/models/minigpt4rec_v2.py",
        "minigpt4/models/projector_structure_reg.py",
    ):
        file = ROOT / path
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    print(
        "[NIGHT22-CPU-PASS] Eight distinct loss arms, strict rank410 CoRA, "
        "consistent environment, original E0 guard unaffected; syntax valid.",
        flush=True,
    )


if __name__ == "__main__":
    main()
