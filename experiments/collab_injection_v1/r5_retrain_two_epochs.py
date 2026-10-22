#!/usr/bin/env python3
"""R5: train R4-01/02/03 FROM ORIGINAL S2 for TWO epochs.

This deliberately does NOT resume the R4 one-epoch checkpoints.
It reuses the actual proven R4 FULL_E1 YAML, R4 entrypoint and parameter
isolation machinery.  The LR cosine horizon stays at 3 epochs, matching R4.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from r2_train import S2, sha, run_to_log
from r4_train import ARMS, check_anchor
from r4_freeze import MODES, V31_NAME

R4_ROOT = ROOT / "logs/collab_v1/R4_ISOLATION_20261022"
R5_TAG = "R5_RETRAIN2_20261022"
R5_ROOT = ROOT / "logs/collab_v1" / R5_TAG
OUT_ROOT = ROOT / "output" / R5_TAG

BASELINE = (0.7296752596287361, 0.6841670211619721)
R4_PREVIOUS = {
    "01": (0.7291725439389706, 0.6839711441915867),
    "02": (0.7301846603675677, 0.6842576591246285),
    "03": (0.7302009273202951, 0.6842615129076296),
}
TRAINABLE = {"01": 75968, "02": 133520, "03": 166288}
METRICS_RX = re.compile(
    r"\*\*\*auc:\s*([0-9.eE+-]+).*?\*\*\*uauc:\s*\(\s*([0-9.eE+-]+)"
)


def read_torch(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def parse_valid(log):
    rows = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Averaged stats:" not in line:
            continue
        match = METRICS_RX.search(line)
        if match:
            values = tuple(float(x) for x in match.groups())
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
                raise RuntimeError("Non-finite/out-of-range Valid metrics")
            rows.append(values)
    if len(rows) != 2:
        raise RuntimeError(f"Expected exactly two Valid epochs; found {len(rows)}: {rows}")
    return rows


def validate_source(round_id):
    original_dir = R4_ROOT / ARMS[round_id]
    source_yaml = original_dir / "FULL_E1.yaml"
    manifest = original_dir / "manifest.json"
    if not source_yaml.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Missing original completed R4 source: {original_dir}")
    info = json.loads(manifest.read_text(encoding="utf-8"))
    if info.get("status") != "complete" or info.get("round") != round_id:
        raise RuntimeError(f"R4 manifest not complete: {manifest}")
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "model" not in data or "run" not in data:
        raise RuntimeError(f"Invalid R4 YAML: {source_yaml}")
    model, run = data["model"], data["run"]
    mode, local_v = MODES[round_id]
    if model.get("r3_mode") != mode or model.get("r4_round") != round_id:
        raise RuntimeError("Source R4 method/round mismatch")
    if model.get("collab_mode") != "legacy" or int(model.get("gcn_layer", -1)) != 31:
        raise RuntimeError("R4 must be S2 legacy L31")
    if model.get("freeze_lora") is not False or not model.get("fix_autocast_cache"):
        raise RuntimeError("R4 LoRA forward/cache fix not enabled")
    if Path(model.get("ckpt", "")).resolve() != S2.resolve():
        raise RuntimeError("Wrong initial source: R5 must start from ORIGINAL S2")
    if not S2.is_file():
        raise FileNotFoundError(f"Missing original S2 checkpoint: {S2}")
    expected_s2 = info.get("s2_sha256")
    if expected_s2 != sha(S2):
        raise RuntimeError("S2 SHA256 differs from completed R4 manifest")
    if int(run.get("max_epoch", -1)) != 3:
        raise RuntimeError("R4 scheduler must have 3-epoch horizon")
    if int(run.get("stop_after_n_epochs", -1)) != 1:
        raise RuntimeError("Expected R4 FULL_E1 one-epoch source")
    if run.get("diagnostic_stop_after_epoch0") is not True:
        raise RuntimeError("Expected R4 epoch0 stopping guard")
    if run.get("resume_ckpt_path") is not None or run.get("max_train_steps") is not None:
        raise RuntimeError("Source R4 FULL_E1 must start fresh and be a full epoch")
    if int(run.get("seed", -1)) != 42 or int(run.get("batch_size_train", -1)) != 3:
        raise RuntimeError("R4 training recipe changed from original")
    if abs(float(run.get("init_lr", 0)) - 2e-5) > 1e-12:
        raise RuntimeError("R4 base LR changed from original")
    if abs(float(run.get("lr_scale_other", 1.0)) - 1.0) > 1e-12:
        raise RuntimeError("R4 adapter LR multiplier changed")
    if round_id == "03" and abs(float(run.get("lr_scale_lora", 0)) - 0.1) > 1e-12:
        raise RuntimeError("R4-03 V31 B LR multiplier changed")
    return source_yaml, data


def prepare(round_id):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    source_yaml, cfg = validate_source(round_id)
    # Confirm the original S2-equivalent zero-initialization anchor.
    anchor = check_anchor(round_id)
    arm = ARMS[round_id]
    folder = R5_ROOT / arm
    output_dir = OUT_ROOT / arm / "FULL_E2"
    if folder.exists() or output_dir.exists():
        raise FileExistsError(
            f"R5 new-run path already exists; refuse overwrite: {folder} / {output_dir}"
        )

    run = cfg["run"]
    run["output_dir"] = str(output_dir)
    run["max_epoch"] = 3                       # same 3-epoch LR horizon as R4
    run["stop_after_n_epochs"] = 2              # run exactly epochs 0 and 1
    run["diagnostic_stop_after_epoch0"] = False
    run["resume_ckpt_path"] = None              # DO NOT use R4 checkpoint
    run["skip_validation"] = False
    run["evaluate"] = False
    run["save_checkpoint_every_n_epochs"] = 1  # keep BOTH epochs
    run["test_splits"] = []                    # VALID ONLY
    for stale in ("max_train_steps", "mid_epoch_eval_iters", "iters_per_epoch", "max_iters"):
        run.pop(stale, None)
    if Path(cfg["model"]["ckpt"]).resolve() != S2.resolve():
        raise RuntimeError("A prohibited R4 checkpoint was introduced")

    folder.mkdir(parents=True, exist_ok=False)
    config_file = folder / "FULL_E2.yaml"
    config_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = folder / "manifest.json"
    status = {
        "status": "prepared", "round": round_id, "method": MODES[round_id][0],
        "original_s2": str(S2), "original_s2_sha256": sha(S2),
        "r4_reference_config": str(source_yaml),
        "r4_reference_sha256": sha(source_yaml),
        "r5_config": str(config_file),
        "r5_config_sha256": sha(config_file),
        "source_anchor_status": anchor.get("status"),
        "epochs_to_train": 2, "epoch_indices": [0, 1],
        "scheduler_horizon": 3,
        "lr_adapter": 2e-5, "lr_v31_B": 2e-6 if round_id == "03" else None,
        "seed": 42, "batch_size": 3, "original_s2_auc": BASELINE[0],
        "original_s2_uauc": BASELINE[1], "previous_r4_epoch1": R4_PREVIOUS[round_id],
    }
    manifest.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return folder, config_file, manifest, status


def verify_output(round_id, folder):
    arm = ARMS[round_id]
    txt = (folder / "FULL_E2.log").read_text(encoding="utf-8", errors="replace")
    if "[R4-FROZEN-HASH-OK]" not in txt or "[R4-COUNTERS]" not in txt:
        raise RuntimeError("Missing frozen integrity proof or runtime counters")
    freeze_match = re.search(r"\[R4-FREEZE\]\s*(\{[^\n]+\})", txt)
    counters_match = re.search(r"\[R4-COUNTERS\]\s*(\{[^\n]+\})", txt)
    if not freeze_match or not counters_match:
        raise RuntimeError("Missing R4 freeze manifest or runtime counter JSON")
    freeze = json.loads(freeze_match.group(1))
    counters = json.loads(counters_match.group(1))
    if freeze.get("round") != round_id or freeze.get("trainable_params") != TRAINABLE[round_id]:
        raise RuntimeError(f"Incorrect trainable params: {freeze}")
    allowed = freeze.get("trainable_names", [])
    for name in allowed:
        if not name.startswith("r3_adapter.") and not (round_id == "03" and V31_NAME.search(name)):
            raise RuntimeError(f"Unexpected unfrozen parameter: {name}")
    local = [name for name in allowed if V31_NAME.search(name)]
    if (round_id == "03") != (len(local) == 1):
        raise RuntimeError("Wrong L31 V LoRA-B policy")
    suffix = "cf_forward" if round_id == "01" else "cora_forward"
    for tag in ("q", "v"):
        if counters.get(f"{tag}_{suffix}", 0) < 1:
            raise RuntimeError(f"R3 adapter never executed at {tag}: {counters}")
    valid_rows = parse_valid(folder / "FULL_E2.log")
    checkpoint_root = OUT_ROOT / arm / "FULL_E2"
    for epoch in (0, 1):
        checkpoints = list(checkpoint_root.rglob(f"checkpoint_{epoch}.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Missing/ambiguous checkpoint epoch {epoch}: {checkpoints}")
        data = read_torch(checkpoints[0])
        if int(data.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Checkpoint has incorrect epoch: {checkpoints[0]}")
        keys = [k for k in data["model"] if k.startswith("r3_adapter.")]
        if not keys:
            raise RuntimeError(f"R3 adapter absent from checkpoint epoch {epoch}")
        if round_id == "03" and len([k for k in data["model"] if V31_NAME.search(k)]) != 1:
            raise RuntimeError("L31 V LoRA-B was not saved in R5-03")
        del data
    return valid_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=tuple(MODES), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    folder, config_file, manifest, status = prepare(args.round)
    print(f"[R5-{args.round}-PREPARED] method={status['method']} "
          f"from_original_S2={S2} epochs=2 schedule_horizon=3 "
          f"config={config_file}", flush=True)
    if args.prepare_only:
        return
    try:
        status["status"] = "running"
        manifest.write_text(json.dumps(status, indent=2), encoding="utf-8")
        run_to_log(
            [sys.executable, "-u", "experiments/collab_injection_v1/r4_entry.py",
             "--cfg-path", config_file, "--round", args.round],
            folder / "FULL_E2.log")
        rows = verify_output(args.round, folder)
        status["status"] = "complete"
        status["epochs_valid"] = [{"epoch": i, "auc": row[0], "uauc": row[1]}
                                  for i, row in enumerate(rows)]
        status["delta_epoch2_vs_epoch1"] = {
            "auc": rows[1][0] - rows[0][0], "uauc": rows[1][1] - rows[0][1]}
        status["delta_epoch2_vs_s2"] = {
            "auc": rows[1][0] - BASELINE[0], "uauc": rows[1][1] - BASELINE[1]}
        manifest.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[R5-{args.round}-DONE] E1 AUC={rows[0][0]:.9f} UAUC={rows[0][1]:.9f} "
              f"E2 AUC={rows[1][0]:.9f} UAUC={rows[1][1]:.9f} "
              f"delta_auc={rows[1][0]-rows[0][0]:+.9f} "
              f"delta_uauc={rows[1][1]-rows[0][1]:+.9f}", flush=True)
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
        manifest.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[R5-{args.round}-FAILED] {exc!r}", flush=True)
        raise


if __name__ == "__main__":
    main()
