#!/usr/bin/env python3
"""Read-only TEST evaluation of historical best-Valid-AUC S4_SOFT4.

The exact user-provided checkpoint is:
  CORA_SPLIT_EXPERTS6_20261104/S4_SOFT4/20261104055/checkpoint_best.pth
Historical VALID: AUC ~0.733476, UAUC at best-AUC ~0.697128.

S4 is the original cora_split_experts6.py user-conditioned Soft4 rank-split
router over the four 4-channel experts within original rank16.
Use the S4 router's identical implementation, NOT RR1 or E2's constructor.

Checkpoint is trainable-parameters-only; first construct the pretrained
Vicuna/CoRA + MF model exactly as during S4 training, then strictly audit
the keys and load the trainable state, then evaluate test_ood2.pkl ONCE.
This script never trains or reloads the checkpoint through optimizer state.

From CoRA_official root:
 python experiments/no31_cora_original/cora_s4_best_test.py --check
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_s4_best_test.py --run
"""
import argparse
import csv
import hashlib
import json
import os
import re
import runpy
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
import cora_split_experts6 as split

ROOT = split.base.PROJECT_ROOT
TRAIN_PROTOCOL = "CORA_SPLIT_EXPERTS6_20261104"
TEST_PROTOCOL = "CORA_S4_BEST_TEST_20261110"
NAME = "S4_SOFT4"
RELATIVE_CHECKPOINT = (
    "output/CORA_SPLIT_EXPERTS6_20261104/"
    "S4_SOFT4/20261104055/checkpoint_best.pth"
)
VALID_AUC_REFERENCE = 0.733476
VALID_UAUC_REFERENCE = 0.697128


def _protocol(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe output protocol: {value!r}")
    return value


# GPU0 Test is inference-only (RR1 prior observed peak was ~14 GiB).
# Keep this threshold scoped to test entrypoints; training preflights unchanged.
TEST_MIN_FREE_GPU0_MIB = 16 * 1024


def _test_gpu0_preflight(stage):
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPU0 test preflight failed: {result.stderr.strip()}")
    free = None
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) >= 2 and values[0] == "0":
            free = int(values[1])
            break
    if free is None:
        raise RuntimeError("Physical GPU0 is unavailable")
    print(f"[TEST GPU0 PREFLIGHT] {stage} free={free} MiB "
          f"required>16384 MiB (16 GiB)", flush=True)
    if free <= TEST_MIN_FREE_GPU0_MIB:
        raise RuntimeError(
            f"Physical GPU0 has only {free} MiB free; "
            "requires >16 GiB before inference. Do not stop other jobs."
        )


def _sha256(path):
    dig = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _source():
    source_yaml = HERE / "local_configs" / TRAIN_PROTOCOL / (NAME + ".yaml")
    if not source_yaml.is_file():
        raise FileNotFoundError(
            f"S4 original training YAML not found: {source_yaml}. "
            "Do not recreate it using another trial's config."
        )
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    run = config["run"]
    if (
        int(run["seed"]) != 42
        or run.get("valid_splits") != ["valid"]
        or bool(run.get("test_splits"))
        or bool(run.get("evaluate"))
        or run.get("mode") != "v2"
        or config["model"].get("freeze_rec") is not False
        or config["model"].get("ckpt")
        or run.get("resume_ckpt_path")
    ):
        raise RuntimeError(
            "S4 source configuration mismatched. Need the original seed42, "
            "v2, trainable-MF, Valid-only YAML; no training allowed."
        )
    return source_yaml, config


def _checkpoint(explicit):
    path = Path(explicit).expanduser() if explicit else (HERE / RELATIVE_CHECKPOINT)
    path = path.resolve()
    if not path.is_file() or path.name != "checkpoint_best.pth":
        raise FileNotFoundError(f"Historical S4 checkpoint not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty checkpoint: {path}")
    return path


def _last_best_epoch():
    original_log = HERE / "logs" / TRAIN_PROTOCOL / (NAME + ".log")
    if not original_log.exists():
        print("[S4 TEST CHECK] Original training stdout unavailable; "
              "checkpoint epoch will be recorded, but cannot be cross-checked.")
        return None
    matches = re.findall(
        r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=([-+\d.eE]+)",
        original_log.read_text(encoding="utf-8", errors="replace"),
    )
    if not matches:
        print("[S4 TEST CHECK] No [JOINT8 BEST] marker in training log; "
              "checkpoint epoch cannot be cross-checked.")
        return None
    epoch, auc = max(((int(x), float(y)) for x, y in matches),
                     key=lambda item: item[1])
    print(f"[S4 TEST VALID] historical_best_epoch={epoch} "
          f"historical_best_auc={auc:.9f}")
    if abs(auc - VALID_AUC_REFERENCE) > 0.002:
        raise RuntimeError(
            f"Historical S4 Valid AUC {auc:.9f} does not match "
            f"reference ~{VALID_AUC_REFERENCE}. Wrong training log?"
        )
    return epoch


def _test_config(source, target_protocol):
    run = source["run"]
    test_cfg = json.loads(json.dumps(source))
    # Use evaluate=False so OOD2 builders do not try to load their optional
    # test_warm/test_cold files. We REPLACE RunnerBase.train() with an entirely
    # test-only method; evaluating cannot fall back to training.
    test_cfg["run"]["evaluate"] = False
    test_cfg["run"]["s4_test_only"] = True
    test_cfg["run"]["valid_splits"] = []
    test_cfg["run"]["test_splits"] = ["test"]
    test_cfg["run"]["resume_ckpt_path"] = None
    test_cfg["run"]["output_dir"] = str(HERE / "output" / target_protocol / NAME)
    path = HERE / "local_configs" / target_protocol / (NAME + "_TEST.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(test_cfg, sort_keys=False, allow_unicode=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"Different config exists at {path}; "
                                  "use another --output-protocol")
    else:
        path.write_text(rendered, encoding="utf-8")
    return path


def _install_s4():
    """Rebuild EXACT S4 Rank4+Soft4 model; RR1 and E2 constructors differ."""
    split._setup(TRAIN_PROTOCOL)  # replaces base.Rank4Experts with SplitExperts6
    trial = split.TRIAL_BY_NAME[NAME]
    if trial["expert_mode"] != "soft4":
        raise RuntimeError("S4 must use user-only Soft4, all four rank4 experts")
    core = split.base.late.base.core
    core.patch_runner(
        screen_epochs=0,
        mf_lr_ratio=split.base.MF_LR_RATIO,
        patience=split.base.PATIENCE,
    )
    # Same C6 and D4 patch order as SplitExperts6 original training worker.
    split.base.late.base.combo.install_combo_model(
        split.base.late.base.combo.NAMES["C6_KD_BOTH"]
    )
    split.base.late.base._patch_runtime_classes(trial)
    split.base.install_model(trial)
    split.base.install_loss_audit(trial)
    print("[S4 TEST MODEL] User Soft4 (4xRank4, 32-layer Q/K/V/O), "
          "D4_TOPK_RES025 + C6_KD_BOTH, MF trainable, "
          "original CoRA generator/P, no new training", flush=True)


def _install_eval_runner(checkpoint, original_yaml, report_json, report_csv, valid_epoch):
    from minigpt4.runners.runner_base import RunnerBase

    def test_only(self):
        if self.config.run_cfg.get("s4_test_only") is not True:
            raise RuntimeError("S4 Test-only sentinel missing: refusing to train")
        if self.config.run_cfg.get("evaluate") is not False:
            raise RuntimeError("Wrong dataset-building mode; refusing to continue")
        if list(self.test_splits) != ["test"] or list(self.valid_splits):
            raise RuntimeError("S4 requires test-only evaluation, no Valid selection")
        if report_json.exists() or report_csv.exists():
            raise FileExistsError("Test result already exists, refusing to overwrite.")
        self.set_model_mode(self.config.run_cfg.mode)
        if "test" not in self.dataloaders:
            raise RuntimeError(f"No test split, present splits={list(self.dataloaders)}")
        model = self.unwrap_dist_model(self.model)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            # Older torch versions may not support weights_only; checkpoint
            # must be a trusted file saved by this project's own training job.
            payload = torch.load(checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise RuntimeError("Expected S4 compact model checkpoint dictionary")
        if payload.get("next8_checkpoint_format") != "trainable_model_only_v1":
            raise RuntimeError(
                f"Wrong checkpoint format: {payload.get('next8_checkpoint_format')!r}"
            )
        if payload.get("optimizer_included") is not False:
            raise RuntimeError("Expected model-only best checkpoint, not optimizer resume")
        epoch = int(payload.get("epoch", -1))
        if epoch < 0:
            raise RuntimeError("Checkpoint lacks a valid epoch")
        if valid_epoch is not None and epoch != valid_epoch:
            raise RuntimeError(
                f"Checkpoint epoch {epoch} differs from original Best Valid "
                f"epoch {valid_epoch}; wrong checkpoint"
            )
        saved_run = payload.get("config", {}).get("run", {})
        if int(saved_run.get("seed", -1)) != 42:
            raise RuntimeError("Checkpoint seed is not 42")
        trained = payload["model"]
        current = model.state_dict()
        expected = {name for name, param in model.named_parameters()
                    if param.requires_grad}
        missing_trainable = sorted(expected - set(trained))
        extra = sorted(set(trained) - set(current))
        shape_error = sorted(name for name, val in trained.items()
                             if name in current
                             and tuple(val.shape) != tuple(current[name].shape))
        if missing_trainable or extra or shape_error:
            raise RuntimeError(
                f"S4 checkpoint incompatible: missing trainable="
                f"{missing_trainable[:15]}; extra={extra[:15]}; "
                f"shape mismatch={shape_error[:15]}"
            )
        incompatibility = model.load_state_dict(trained, strict=False)
        if incompatibility.unexpected_keys:
            raise RuntimeError(
                f"Unexpected checkpoint state keys: {incompatibility.unexpected_keys}"
            )
        print(
            f"[S4 TEST CKPT PASS] epoch={epoch}, model_tensors={len(trained)}, "
            f"trainable_tensors={len(expected)}, frozen_keys_restored_from_base="
            f"{len(incompatibility.missing_keys)}", flush=True
        )
        model.eval()
        with torch.no_grad():
            result = self.eval_epoch(
                split_name="test", cur_epoch=epoch, skip_reload=True,
            )
        if not isinstance(result, dict) or "agg_metrics" not in result or "uauc" not in result:
            raise RuntimeError(
                f"S4 unexpected Test output: {str(result)[:300]}"
            )
        auc = float(result["agg_metrics"])
        uauc_val = result["uauc"]
        if isinstance(uauc_val, (tuple, list)):
            uauc_val = uauc_val[0]
        uauc = float(uauc_val)
        if not (0.0 <= auc <= 1.0 and 0.0 <= uauc <= 1.0):
            raise RuntimeError(f"Invalid Test results: auc={auc} uauc={uauc}")
        report = {
            "trial": NAME, "split": "test", "test_auc": auc, "test_uauc": uauc,
            "source_protocol": TRAIN_PROTOCOL,
            "source_valid_auc_reference": VALID_AUC_REFERENCE,
            "source_valid_uauc_reference": VALID_UAUC_REFERENCE,
            "checkpoint_best_epoch": epoch,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_train_yaml": str(original_yaml),
            "seed": 42,
            "test_only": True,
            "route": "S4 user-conditioned Soft4 rank-split, four active rank4 experts",
            "task_result": result,
        }
        report_json.parent.mkdir(parents=True, exist_ok=True)
        with report_json.open("x", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        with report_csv.open("x", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "trial", "test_auc", "test_uauc", "checkpoint_best_epoch",
                "checkpoint_path", "source_protocol", "seed",
            ], extrasaction="ignore")
            writer.writeheader()
            writer.writerow(report)
        print(f"[S4 TEST RESULT] AUC={auc:.9f} UAUC={uauc:.9f} "
              f"checkpoint_epoch={epoch}", flush=True)
        print(f"[S4 TEST SAVED] json={report_json} csv={report_csv}", flush=True)
        return report

    RunnerBase.train = test_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="No GPU or weight loading; check paths/protocols")
    mode.add_argument("--run", action="store_true",
                      help="Evaluate historical S4 on Test with GPU0, no training")
    parser.add_argument("--checkpoint", default="",
                        help="Override exact S4 checkpoint_best.pth path")
    parser.add_argument("--output-protocol", default=TEST_PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    output_protocol = _protocol(args.output_protocol)
    orig_yaml, source = _source()
    checkpoint = _checkpoint(args.checkpoint)
    best_epoch = _last_best_epoch()
    reports = HERE / "out" / output_protocol
    report_json = reports / (NAME + "_test_metrics.json")
    report_csv = reports / (NAME + "_test_metrics.csv")
    print(f"[S4 TEST PLAN] checkpoint={checkpoint}")
    print(f"[S4 TEST PLAN] source_yaml={orig_yaml}")
    print(f"[S4 TEST PLAN] dataset=test_ood2.pkl; selected on Valid AUC; "
          f"report={report_csv}; no training, no overwrite")
    if args.check:
        print("[S4 TEST CHECK PASS] sources found; will load/check weights at --run")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; only physical GPU0 allowed")
    if report_json.exists() or report_csv.exists():
        raise FileExistsError(
            "Existing S4 Test result detected; use --output-protocol with "
            "a unique name rather than overwriting."
        )
    _test_gpu0_preflight("S4-test-before-model")
    test_yaml = _test_config(source, output_protocol)
    _install_s4()
    _install_eval_runner(checkpoint, orig_yaml, report_json, report_csv, best_epoch)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml)]
    print(f"[S4 TEST BEGIN] config={test_yaml}")
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
