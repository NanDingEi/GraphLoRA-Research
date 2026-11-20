#!/usr/bin/env python3
"""Read-only TEST evaluation for the six CORA_HP_MODEL6 trials.

Rebuilds the exact M1-M6 training runtime (D4 grouped inputs + C6_KD_BOTH +
the matching HP_MODEL6 router), loads only the trainable tensors from the
VALID-selected checkpoint, and evaluates test_ood2.pkl once. No training,
checkpoint selection, or test-side tuning is performed.

From CoRA_official root:
  python experiments/no31_cora_original/cora_hp_model6_test.py --check --trial M3_USER_BETA
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp_model6_test.py --run --trial M3_USER_BETA
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
ROOT_DIR = HERE.parent.parent
for _d in (str(ROOT_DIR), str(HERE), str(ROOT_DIR.parent)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import cora_hp_model6 as hf

ROOT = hf.moe.base.PROJECT_ROOT
PROTOCOL = "CORA_HP_MODEL6_GPU_20261119"
OUTPUT_PROTOCOL = "CORA_HP_MODEL6_TEST_20261120"
SUMMARY_CSV = HERE / "out" / PROTOCOL / "hp_model6_valid_summary.csv"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_row(trial_name):
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(
            "Missing VALID summary: %s; run --summary first" % SUMMARY_CSV)
    with SUMMARY_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row.get("trial") == trial_name]
    if len(matches) != 1:
        raise RuntimeError("Expected one VALID summary row for %s" % trial_name)
    row = matches[0]
    if row.get("status") not in ("EARLY_STOP", "CHECK"):
        raise RuntimeError(
            "%s is not eligible for TEST: status=%s" %
            (trial_name, row.get("status")))
    if not row.get("valid_auc") or not row.get("valid_uauc"):
        raise RuntimeError("%s has no complete VALID AUC/UAUC" % trial_name)
    log_path = HERE / "logs" / PROTOCOL / (trial_name + ".log")
    if not log_path.is_file():
        raise FileNotFoundError("Missing training log: %s" % log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "[JOINT8 EARLY STOP]" not in log_text and "Training time" not in log_text:
        raise RuntimeError(
            "%s training completion is not confirmed (no EARLY STOP/Training time marker); "
            "inspect %s before TEST" % (trial_name, log_path))
    return {
        "valid_auc": float(row["valid_auc"]),
        "valid_uauc": float(row["valid_uauc"]),
        "summary_status": row.get("status"),
    }


def _checkpoint(trial_name):
    folder = HERE / "output" / PROTOCOL / trial_name
    candidates = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one checkpoint under %s, found %s" %
            (folder, candidates))
    path = candidates[0].resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Bad checkpoint: %s" % path)
    return path


def _load_payload(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("Expected compact model-only checkpoint")
    if payload.get("next8_checkpoint_format") != "trainable_model_only_v1":
        raise RuntimeError("Unexpected checkpoint format")
    if payload.get("optimizer_included") is not False:
        raise RuntimeError("Checkpoint must not contain optimizer state")
    if int(payload.get("config", {}).get("run", {}).get("seed", -1)) != 42:
        raise RuntimeError("Checkpoint seed is not 42")
    if not payload["model"]:
        raise RuntimeError("Checkpoint contains no trainable tensors")
    return payload


def _source(trial_name):
    source_yaml = HERE / "local_configs" / PROTOCOL / (trial_name + ".yaml")
    if not source_yaml.is_file():
        raise FileNotFoundError("Missing training config: %s" % source_yaml)
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    run = config["run"]
    checks = {
        "seed42": int(run.get("seed", -1)) == 42,
        "valid_only": run.get("valid_splits") == ["valid"] and
                      not run.get("test_splits"),
        "mode_v2": run.get("mode") == "v2",
        "MF_trainable": config["model"].get("freeze_rec") is False,
        "no_model_ckpt": not config["model"].get("ckpt"),
        "no_resume": not run.get("resume_ckpt_path"),
        "not_evaluate": not bool(run.get("evaluate")),
    }
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError("%s source config failed checks: %s" %
                           (trial_name, failed))
    return source_yaml, config


def _test_config(source, trial_name, out_protocol):
    cfg = json.loads(json.dumps(source))
    cfg["run"]["evaluate"] = False
    cfg["run"]["hp_test_only"] = True
    cfg["run"]["valid_splits"] = []
    cfg["run"]["test_splits"] = ["test"]
    cfg["run"]["resume_ckpt_path"] = None
    cfg["run"]["output_dir"] = str(HERE / "output" / out_protocol / trial_name)
    path = HERE / "local_configs" / out_protocol / (trial_name + "_TEST.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError("Different test config already exists: %s" % path)
    else:
        path.write_text(rendered, encoding="utf-8")
    return path


def _install_runtime(trial):
    """Mirror the HP_MODEL6 worker's D4/C6/model-patch order."""
    hf.configure(PROTOCOL)
    core = hf.moe.base.late.base.core
    core.patch_runner(
        screen_epochs=0,
        mf_lr_ratio=hf.moe.base.MF_LR_RATIO,
        patience=hf.moe.base.PATIENCE,
    )
    hf.moe.base.late.base.combo.install_combo_model(
        hf.moe.base.late.base.combo.NAMES["C6_KD_BOTH"])
    hf.moe.base.late.base._patch_runtime_classes(trial)
    hf.install_model(trial)
    print(
        "[HP MODEL6 TEST MODEL] %s variant=%s D4/C6 runtime installed; test only" %
        (trial["name"], trial.get("model_variant")), flush=True)


def _install_eval_runner(trial_name, checkpoint, report_json, report_csv,
                         valid_metrics, payload):
    from minigpt4.runners.runner_base import RunnerBase

    def test_only(self):
        if self.config.run_cfg.get("hp_test_only") is not True:
            raise RuntimeError("Missing hp_test_only sentinel")
        if list(self.test_splits) != ["test"] or list(self.valid_splits):
            raise RuntimeError("Refusing non-test-only split configuration")
        if "test" not in self.dataloaders:
            raise RuntimeError("Test dataloader missing: %s" %
                               list(self.dataloaders))
        if report_json.exists() or report_csv.exists():
            raise FileExistsError("Test report already exists; refusing repeat TEST")
        self.set_model_mode(self.config.run_cfg.mode)
        model = self.unwrap_dist_model(self.model)
        trained = payload["model"]
        current = model.state_dict()
        expected = {name for name, param in model.named_parameters()
                    if param.requires_grad}
        missing = sorted(expected - set(trained))
        extra = sorted(set(trained) - set(current))
        shapes = sorted(
            name for name, value in trained.items()
            if name in current and tuple(value.shape) != tuple(current[name].shape)
        )
        if missing or extra or shapes:
            raise RuntimeError(
                "Checkpoint/model mismatch: missing=%s extra=%s shapes=%s" %
                (missing[:12], extra[:12], shapes[:12]))
        incompat = model.load_state_dict(trained, strict=False)
        if incompat.unexpected_keys:
            raise RuntimeError("Unexpected checkpoint keys: %s" %
                               incompat.unexpected_keys)
        epoch = int(payload.get("epoch", -1))
        if epoch < 0:
            raise RuntimeError("Invalid checkpoint epoch")
        print("[HP MODEL6 TEST CKPT PASS] %s epoch=%d tensors=%d" %
              (trial_name, epoch, len(trained)), flush=True)

        model.eval()
        with torch.no_grad():
            result = self.eval_epoch(
                split_name="test", cur_epoch=epoch, skip_reload=True)
        auc = float(result["agg_metrics"])
        uauc_value = result["uauc"]
        if isinstance(uauc_value, (tuple, list)):
            uauc_value = uauc_value[0]
        if isinstance(uauc_value, dict):
            uauc_value = uauc_value.get("score")
        uauc = float(uauc_value)
        if not (0.0 <= auc <= 1.0 and 0.0 <= uauc <= 1.0):
            raise RuntimeError("Invalid TEST metrics: AUC=%s UAUC=%s" %
                               (auc, uauc))

        report = {
            "trial": trial_name,
            "split": "test",
            "test_auc": auc,
            "test_uauc": uauc,
            "valid_auc_at_selected_checkpoint": valid_metrics["valid_auc"],
            "valid_uauc_at_selected_checkpoint": valid_metrics["valid_uauc"],
            "valid_summary_status": valid_metrics["summary_status"],
            "source_protocol": PROTOCOL,
            "test_protocol": OUTPUT_PROTOCOL,
            "checkpoint_best_epoch": epoch,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "seed": 42,
            "test_only": True,
            "test_used_for_selection": False,
        }
        report_json.parent.mkdir(parents=True, exist_ok=True)
        with report_json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        with report_csv.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "trial", "test_auc", "test_uauc",
                "valid_auc_at_selected_checkpoint",
                "valid_uauc_at_selected_checkpoint",
                "checkpoint_best_epoch", "checkpoint_path",
                "source_protocol", "test_protocol", "seed",
            ], extrasaction="ignore")
            writer.writeheader()
            writer.writerow(report)
        print("[HP MODEL6 TEST RESULT] %s AUC=%.9f UAUC=%.9f epoch=%d" %
              (trial_name, auc, uauc, epoch), flush=True)
        return report

    RunnerBase.train = test_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--trial", required=True,
                        choices=sorted(hf.TRIAL_BY_NAME))
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--output-protocol", default=OUTPUT_PROTOCOL)
    args = parser.parse_args()
    if args.protocol != PROTOCOL:
        raise ValueError("This test script is pinned to source protocol %s" % PROTOCOL)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_protocol):
        raise ValueError("Unsafe output protocol")
    os.chdir(ROOT)

    trial = hf.TRIAL_BY_NAME[args.trial]
    valid_metrics = _valid_row(args.trial)
    checkpoint = _checkpoint(args.trial)
    payload = _load_payload(checkpoint)
    source_yaml, source = _source(args.trial)
    test_yaml = _test_config(source, args.trial, args.output_protocol)
    reports = HERE / "out" / args.output_protocol
    report_json = reports / (args.trial + "_test_metrics.json")
    report_csv = reports / (args.trial + "_test_metrics.csv")

    print("[HP MODEL6 TEST PLAN] trial=%s status=%s valid_auc=%.9f valid_uauc=%.9f" %
          (args.trial, valid_metrics["summary_status"],
           valid_metrics["valid_auc"], valid_metrics["valid_uauc"]))
    print("[HP MODEL6 TEST PLAN] checkpoint=%s epoch=%s bytes=%d" %
          (checkpoint, payload.get("epoch"), checkpoint.stat().st_size))
    print("[HP MODEL6 TEST PLAN] source_config=%s" % source_yaml)
    print("[HP MODEL6 TEST PLAN] test_config=%s" % test_yaml)
    print("[HP MODEL6 TEST PLAN] report=%s" % report_csv)
    if args.check:
        if report_json.exists() or report_csv.exists():
            raise FileExistsError("Existing TEST report: %s" % report_csv)
        print("[HP MODEL6 TEST CHECK PASS] checkpoint/config/VALID summary verified")
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; no other GPU is permitted")
    if report_json.exists() or report_csv.exists():
        raise FileExistsError("Existing TEST report; refusing a second TEST evaluation")
    hf.moe.base._gpu0_preflight("HP-MODEL6-TEST-%s" % args.trial)
    _install_runtime(trial)
    _install_eval_runner(args.trial, checkpoint, report_json, report_csv,
                         valid_metrics, payload)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
