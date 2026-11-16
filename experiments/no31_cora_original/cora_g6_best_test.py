#!/usr/bin/env python3
"""Read-only TEST evaluation for any CORA_GATE_TUNE_20261112 trial.

Reconstructs the exact training runtime (D4_TOPK_RES025 + C6_KD_BOTH +
trial-specific gate bias override / frozen alpha / E2 stacked MoE routing),
loads the trial's best trainable-only checkpoint with strict audit, and
evaluates test_ood2.pkl ONCE. Never trains.

From CoRA_official root:
 python experiments/no31_cora_original/cora_g6_best_test.py --check --trial G1_UGATE_B05
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_g6_best_test.py \
     --run --trial G1_UGATE_B05
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

import cora_residual_routing6 as residual
import cora_gate_tune6 as g6

ROOT = residual.PROJECT_ROOT
TRAIN_PROTOCOL = g6.PROTOCOL
NAME_DEFAULT = "G1_UGATE_B05"
OUTPUT_PROTOCOL = "CORA_G6_BEST_TEST_20261112"
LOG_LINE = re.compile(r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=([-+\d.eE]+)")


def _sha256(path):
    dig = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _checkpoint(trial_name, explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        folder = HERE / "output" / TRAIN_PROTOCOL / trial_name
        candidates = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one checkpoint under {folder}, found "
                f"{len(candidates)}. Pass --checkpoint explicitly.")
        path = candidates[0].resolve()
    if not path.is_file() or path.name != "checkpoint_best.pth":
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty checkpoint: {path}")
    return path


def _best_epoch_from_log(trial_name):
    log = HERE / "logs" / TRAIN_PROTOCOL / f"{trial_name}.log"
    matches = LOG_LINE.findall(log.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        print(f"[G6 TEST] no [JOINT8 BEST] markers in {log}; epoch check skipped")
        return None
    epoch, auc = max(((int(a), float(b)) for a, b in matches), key=lambda t: t[1])
    print(f"[G6 TEST VALID] {trial_name} best_epoch={epoch} best_auc={auc:.9f}")
    return epoch


def _source(trial_name):
    source_yaml = HERE / "local_configs" / TRAIN_PROTOCOL / (trial_name + ".yaml")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    run = config["run"]
    if (int(run["seed"]) != 42 or run.get("valid_splits") != ["valid"]
            or bool(run.get("test_splits")) or bool(run.get("evaluate"))
            or run.get("mode") != "v2"
            or config["model"].get("freeze_rec") is not False
            or config["model"].get("ckpt") or run.get("resume_ckpt_path")):
        raise RuntimeError(f"{trial_name} source configuration mismatched")
    return source_yaml, config


def _test_config(source, trial_name, out_protocol):
    test_cfg = json.loads(json.dumps(source))
    test_cfg["run"]["evaluate"] = False
    test_cfg["run"]["g6_test_only"] = True
    test_cfg["run"]["valid_splits"] = []
    test_cfg["run"]["test_splits"] = ["test"]
    test_cfg["run"]["resume_ckpt_path"] = None
    test_cfg["run"]["output_dir"] = str(HERE / "output" / out_protocol / trial_name)
    path = HERE / "local_configs" / out_protocol / (trial_name + "_TEST.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(test_cfg, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"Different config exists at {path}")
    else:
        path.write_text(rendered, encoding="utf-8")
    return path


def _install_trial(trial):
    g6.bind_globals()
    residual.configure()
    core = residual.late.base.core
    core.patch_runner(screen_epochs=0, mf_lr_ratio=residual.MF_LR_RATIO,
                      patience=residual.PATIENCE)
    residual.late.base.combo.install_combo_model(
        residual.late.base.combo.NAMES["C6_KD_BOTH"])
    residual.late.base._patch_runtime_classes(trial)
    if trial.get("moe_trial"):
        g6.install_moe_route()
    g6.apply_trial_overrides(trial)
    residual.install_trial_module(trial)
    residual.install_optimizer_audit(trial)
    residual.install_trial_forward(trial)
    print(f"[G6 TEST MODEL] {trial['name']} kind={trial['kind']} "
          f"gate_bias={trial.get('gate_bias')} fixed_alpha={trial.get('fixed_alpha')} "
          f"moe={trial.get('moe_trial')}; no training")


def _install_eval_runner(trial_name, checkpoint, report_json, report_csv,
                         valid_epoch, valid_auc_ref):
    from minigpt4.runners.runner_base import RunnerBase

    def test_only(self):
        if self.config.run_cfg.get("g6_test_only") is not True:
            raise RuntimeError("G6 test-only sentinel missing")
        if list(self.test_splits) != ["test"] or list(self.valid_splits):
            raise RuntimeError("Expected test-only evaluation")
        if report_json.exists() or report_csv.exists():
            raise FileExistsError("Test result already exists; refusing overwrite")
        self.set_model_mode(self.config.run_cfg.mode)
        if "test" not in self.dataloaders:
            raise RuntimeError(f"No test split: {list(self.dataloaders)}")
        model = self.unwrap_dist_model(self.model)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise RuntimeError("Expected compact model checkpoint dict")
        if payload.get("next8_checkpoint_format") != "trainable_model_only_v1":
            raise RuntimeError("Wrong checkpoint format")
        if payload.get("optimizer_included") is not False:
            raise RuntimeError("Expected model-only checkpoint")
        epoch = int(payload.get("epoch", -1))
        if valid_epoch is not None and epoch != valid_epoch:
            raise RuntimeError(
                f"Checkpoint epoch {epoch} != log best epoch {valid_epoch}")
        if int(payload.get("config", {}).get("run", {}).get("seed", -1)) != 42:
            raise RuntimeError("Checkpoint seed mismatch")
        trained = payload["model"]
        current = model.state_dict()
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        missing = sorted(expected - set(trained))
        extra = sorted(set(trained) - set(current))
        shapes = sorted(n for n, v in trained.items()
                        if n in current and tuple(v.shape) != tuple(current[n].shape))
        if missing or extra or shapes:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing[:10]} extra={extra[:10]} "
                f"shape={shapes[:10]}")
        incompat = model.load_state_dict(trained, strict=False)
        if incompat.unexpected_keys:
            raise RuntimeError(f"Unexpected keys: {incompat.unexpected_keys}")
        print(f"[G6 TEST CKPT PASS] {trial_name} epoch={epoch} "
              f"tensors={len(trained)}", flush=True)
        model.eval()
        with torch.no_grad():
            result = self.eval_epoch(split_name="test", cur_epoch=epoch,
                                     skip_reload=True)
        auc = float(result["agg_metrics"])
        uauc_val = result["uauc"]
        if isinstance(uauc_val, (tuple, list)):
            uauc_val = uauc_val[0]
        uauc = float(uauc_val)
        if not (0.0 <= auc <= 1.0 and 0.0 <= uauc <= 1.0):
            raise RuntimeError(f"Invalid test metrics: {auc} {uauc}")
        report = {
            "trial": trial_name, "split": "test", "test_auc": auc,
            "test_uauc": uauc, "source_protocol": TRAIN_PROTOCOL,
            "train_valid_auc_best": valid_auc_ref,
            "checkpoint_best_epoch": epoch,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "seed": 42, "test_only": True,
        }
        report_json.parent.mkdir(parents=True, exist_ok=True)
        with report_json.open("x", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        with report_csv.open("x", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "trial", "test_auc", "test_uauc", "checkpoint_best_epoch",
                "checkpoint_path", "source_protocol", "seed"],
                extrasaction="ignore")
            writer.writeheader()
            writer.writerow(report)
        print(f"[G6 TEST RESULT] {trial_name} AUC={auc:.9f} UAUC={uauc:.9f} "
              f"epoch={epoch}", flush=True)
        return report

    RunnerBase.train = test_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--trial", default=NAME_DEFAULT,
                        choices=sorted(g6.TRIAL_BY_NAME))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-protocol", default=OUTPUT_PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_protocol):
        raise ValueError("Unsafe output protocol")
    trial = g6.TRIAL_BY_NAME[args.trial]
    checkpoint = _checkpoint(args.trial, args.checkpoint)
    valid_epoch = _best_epoch_from_log(args.trial)
    orig_yaml, source = _source(args.trial)
    test_yaml = _test_config(source, args.trial, args.output_protocol)
    reports = HERE / "out" / args.output_protocol
    report_json = reports / (args.trial + "_test_metrics.json")
    report_csv = reports / (args.trial + "_test_metrics.csv")
    print(f"[G6 TEST PLAN] trial={args.trial} ckpt={checkpoint}")
    print(f"[G6 TEST PLAN] config={test_yaml} report={report_csv}")
    if args.check:
        print("[G6 TEST CHECK PASS] sources found")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    if report_json.exists() or report_csv.exists():
        raise FileExistsError("Existing result; use a new --output-protocol")
    residual.gpu0_preflight(f"G6-test-{args.trial}")
    _install_trial(trial)
    valid_auc_ref = None
    _install_eval_runner(args.trial, checkpoint, report_json, report_csv,
                         valid_epoch, valid_auc_ref)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
