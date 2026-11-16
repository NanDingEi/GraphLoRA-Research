#!/usr/bin/env python3
"""Read-only TEST evaluation of RR5_USER_GATE_R31 (best historical Valid AUC).

Checkpoint: CORA_RESIDUAL_ROUTING6_20261103/RR5_USER_GATE_R31/20261103153/
            checkpoint_best.pth  (best valid epoch 21, AUC 0.733768)

RR5 is cora_residual_routing6's user_gate trial:
  fused = z32 + sigmoid(g(u)) * (z31 - z32), gate bias init -1.5,
  D4_TOPK_RES025 input, C6_KD_BOTH losses, trainable MF.
This script reconstructs that exact model, loads the trainable-only best
checkpoint with strict key/shape audit, and evaluates test_ood2.pkl ONCE.
Never trains, never touches the source checkpoint.

From CoRA_official root:
 python experiments/no31_cora_original/cora_rr5_best_test.py --check
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_rr5_best_test.py --run
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
import cora_residual_routing6 as residual

ROOT = residual.PROJECT_ROOT
TRAIN_PROTOCOL = "CORA_RESIDUAL_ROUTING6_20261103"
DEFAULT_OUTPUT_PROTOCOL = "CORA_RR5_BEST_TEST_20261111"
NAME = "RR5_USER_GATE_R31"
RELATIVE_CHECKPOINT = (
    "output/CORA_RESIDUAL_ROUTING6_20261103/"
    "RR5_USER_GATE_R31/20261103153/checkpoint_best.pth"
)
VALID_AUC_REFERENCE = 0.733768
VALID_UAUC_REFERENCE = 0.694790
EXPECTED_BEST_EPOCH = 21


def _protocol(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe output protocol: {value!r}")
    return value


def _sha256(path):
    dig = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _source():
    source_yaml = HERE / "local_configs" / TRAIN_PROTOCOL / (NAME + ".yaml")
    if not source_yaml.is_file():
        raise FileNotFoundError(f"RR5 original training YAML not found: {source_yaml}")
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
        raise RuntimeError("RR5 source configuration mismatched.")
    return source_yaml, config


def _checkpoint(explicit):
    path = Path(explicit).expanduser() if explicit else (HERE / RELATIVE_CHECKPOINT)
    path = path.resolve()
    if not path.is_file() or path.name != "checkpoint_best.pth":
        raise FileNotFoundError(f"RR5 checkpoint not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty checkpoint: {path}")
    return path


def _last_best_epoch():
    original_log = HERE / "logs" / TRAIN_PROTOCOL / (NAME + ".log")
    matches = re.findall(
        r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=([-+\d.eE]+)",
        original_log.read_text(encoding="utf-8", errors="replace"),
    )
    epoch, auc = max(((int(x), float(y)) for x, y in matches), key=lambda item: item[1])
    print(f"[RR5 TEST VALID] historical_best_epoch={epoch} best_auc={auc:.9f}")
    if epoch != EXPECTED_BEST_EPOCH or abs(auc - VALID_AUC_REFERENCE) > 0.002:
        raise RuntimeError("RR5 training log does not match expected references")
    return epoch


def _test_config(source, target_protocol):
    test_cfg = json.loads(json.dumps(source))
    test_cfg["run"]["evaluate"] = False
    test_cfg["run"]["rr5_test_only"] = True
    test_cfg["run"]["valid_splits"] = []
    test_cfg["run"]["test_splits"] = ["test"]
    test_cfg["run"]["resume_ckpt_path"] = None
    test_cfg["run"]["output_dir"] = str(HERE / "output" / target_protocol / NAME)
    path = HERE / "local_configs" / target_protocol / (NAME + "_TEST.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(test_cfg, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"Different config exists at {path}")
    else:
        path.write_text(rendered, encoding="utf-8")
    return path


def _install_rr5():
    trial = residual.TRIAL_BY_NAME[NAME]
    if trial["kind"] != "user_gate":
        raise RuntimeError("RR5 must be the user_gate trial")
    residual.configure()
    core = residual.late.base.core
    core.patch_runner(
        screen_epochs=0, mf_lr_ratio=residual.MF_LR_RATIO,
        patience=residual.PATIENCE,
    )
    residual.late.base.combo.install_combo_model(
        residual.late.base.combo.NAMES["C6_KD_BOTH"]
    )
    residual.late.base._patch_runtime_classes(trial)
    residual.install_trial_module(trial)
    residual.install_optimizer_audit(trial)
    residual.install_trial_forward(trial)
    print("[RR5 TEST MODEL] user_gate kind; D4_TOPK_RES025; C6_KD_BOTH; "
          "gate bias -1.5; no training")


def _install_eval_runner(checkpoint, original_yaml, report_json, report_csv, valid_epoch):
    from minigpt4.runners.runner_base import RunnerBase

    def test_only(self):
        if self.config.run_cfg.get("rr5_test_only") is not True:
            raise RuntimeError("RR5 Test-only sentinel missing; refusing to train")
        if self.config.run_cfg.get("evaluate") is not False:
            raise RuntimeError("Wrong dataset-building mode; refusing to continue")
        if list(self.test_splits) != ["test"] or list(self.valid_splits):
            raise RuntimeError("RR5 requires test-only evaluation")
        if report_json.exists() or report_csv.exists():
            raise FileExistsError("Test result already exists; refusing to overwrite.")
        self.set_model_mode(self.config.run_cfg.mode)
        if "test" not in self.dataloaders:
            raise RuntimeError(f"No test split: {list(self.dataloaders)}")
        model = self.unwrap_dist_model(self.model)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise RuntimeError("Expected RR5 compact model checkpoint dict")
        if payload.get("next8_checkpoint_format") != "trainable_model_only_v1":
            raise RuntimeError(f"Wrong checkpoint format: {payload.get('next8_checkpoint_format')!r}")
        if payload.get("optimizer_included") is not False:
            raise RuntimeError("Expected model-only best checkpoint")
        epoch = int(payload.get("epoch", -1))
        if epoch < 0:
            raise RuntimeError("Checkpoint lacks a valid epoch")
        if valid_epoch is not None and epoch != valid_epoch:
            raise RuntimeError(
                f"Checkpoint epoch {epoch} differs from Best Valid epoch {valid_epoch}")
        if int(payload.get("config", {}).get("run", {}).get("seed", -1)) != 42:
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
                f"RR5 checkpoint incompatible: missing={missing_trainable[:15]}; "
                f"extra={extra[:15]}; shape={shape_error[:15]}")
        fusion_keys = [k for k in trained if k.startswith("cora_residual_fusion.")]
        if not fusion_keys:
            raise RuntimeError("RR5 checkpoint lacks cora_residual_fusion parameters")
        incompatibility = model.load_state_dict(trained, strict=False)
        if incompatibility.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {incompatibility.unexpected_keys}")
        print(f"[RR5 TEST CKPT PASS] epoch={epoch}, tensors={len(trained)}, "
              f"fusion_params={len(fusion_keys)}", flush=True)
        model.eval()
        with torch.no_grad():
            result = self.eval_epoch(split_name="test", cur_epoch=epoch, skip_reload=True)
        if not isinstance(result, dict) or "agg_metrics" not in result or "uauc" not in result:
            raise RuntimeError(f"Unexpected Test output: {str(result)[:300]}")
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
            "seed": 42, "test_only": True,
            "route": "RR5 user_gate R31; s=z32+sigmoid(g(u))*(z31-z32)",
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
        print(f"[RR5 TEST RESULT] AUC={auc:.9f} UAUC={uauc:.9f} "
              f"checkpoint_epoch={epoch}", flush=True)
        print(f"[RR5 TEST SAVED] json={report_json} csv={report_csv}", flush=True)
        return report

    RunnerBase.train = test_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-protocol", default=DEFAULT_OUTPUT_PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    output_protocol = _protocol(args.output_protocol)
    orig_yaml, source = _source()
    checkpoint = _checkpoint(args.checkpoint)
    best_epoch = _last_best_epoch()
    reports = HERE / "out" / output_protocol
    report_json = reports / (NAME + "_test_metrics.json")
    report_csv = reports / (NAME + "_test_metrics.csv")
    print(f"[RR5 TEST PLAN] checkpoint={checkpoint}")
    print(f"[RR5 TEST PLAN] source_yaml={orig_yaml}; dataset=test_ood2.pkl; "
          f"report={report_csv}")
    if args.check:
        print("[RR5 TEST CHECK PASS] sources found")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    if report_json.exists() or report_csv.exists():
        raise FileExistsError("Existing RR5 Test result; use --output-protocol")
    residual.gpu0_preflight("RR5-test-before-model")
    test_yaml = _test_config(source, output_protocol)
    _install_rr5()
    _install_eval_runner(checkpoint, orig_yaml, report_json, report_csv, best_epoch)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml)]
    print(f"[RR5 TEST BEGIN] config={test_yaml}")
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
