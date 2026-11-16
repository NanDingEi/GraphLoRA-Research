#!/usr/bin/env python3
"""Read-only TEST evaluation for CORA_HIER_FULLSTACK_20261113 trials.

Reconstructs the exact fullstack training runtime (D4_TOPK_RES025 grouped
inputs + C6_KD_BOTH losses + hier_personal routing installed via install_model),
strictly audits and loads the trial's trainable-only best checkpoint, and
evaluates test_ood2.pkl ONCE. Never trains.

From CoRA_official root:
 python experiments/no31_cora_original/cora_hp_best_test.py --check --trial HP12_FINE_B07
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp_best_test.py \
     --run --trial HP12_FINE_B07
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

import cora_hier_fullstack as hf

ROOT = hf.moe.base.PROJECT_ROOT
PROTOCOL = hf.DEFAULT_PROTOCOL
OUTPUT_PROTOCOL = "CORA_HP_BEST_TEST_20261114"
BEST_LINE = re.compile(r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=([-+\d.eE]+)")


def _sha256(path):
    dig = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _checkpoint(trial_name):
    folder = HERE / "output" / PROTOCOL / trial_name
    candidates = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint under {folder}, found {candidates}")
    path = candidates[0].resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Bad checkpoint: {path}")
    return path


def _best_epoch(trial_name):
    log = HERE / "logs" / PROTOCOL / f"{trial_name}.log"
    matches = BEST_LINE.findall(log.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        print(f"[HP TEST] no [JOINT8 BEST] markers; epoch check skipped")
        return None, None
    epoch, auc = max(((int(a), float(b)) for a, b in matches), key=lambda t: t[1])
    print(f"[HP TEST VALID] {trial_name} best_epoch={epoch} best_auc={auc:.9f}")
    return epoch, auc


def _source(trial_name):
    source_yaml = HERE / "local_configs" / PROTOCOL / (trial_name + ".yaml")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    run = config["run"]
    if (int(run["seed"]) != 42 or bool(run.get("test_splits"))
            or bool(run.get("evaluate")) or run.get("mode") != "v2"
            or config["model"].get("freeze_rec") is not False
            or config["model"].get("ckpt") or run.get("resume_ckpt_path")):
        raise RuntimeError(f"{trial_name} source config mismatched")
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
            raise FileExistsError(f"Different config exists at {path}")
    else:
        path.write_text(rendered, encoding="utf-8")
    return path


def _install_runtime(trial):
    """Same install order as the training worker (proven recipe from
    cora_route_factorial6_test.install_runtime), minus the train-only audit."""
    hf.configure(PROTOCOL)
    core = hf.moe.base.late.base.core
    core.patch_runner(screen_epochs=0, mf_lr_ratio=hf.moe.base.MF_LR_RATIO,
                      patience=hf.moe.base.PATIENCE)
    hf.moe.base.late.base.combo.install_combo_model(
        hf.moe.base.late.base.combo.NAMES["C6_KD_BOTH"])
    hf.moe.base.late.base._patch_runtime_classes(trial)
    hf.install_model(trial)
    print(f"[HP TEST MODEL] {trial['name']} D4/C6 runtime + hier_personal "
          f"routing (lambda={trial['route_lambda']}, beta={trial.get('beta')}, "
          f"fine={trial.get('route_fine')}); no training", flush=True)


def _install_eval_runner(trial_name, checkpoint, report_json, report_csv,
                         valid_epoch, valid_auc):
    from minigpt4.runners.runner_base import RunnerBase

    def test_only(self):
        if self.config.run_cfg.get("hp_test_only") is not True:
            raise RuntimeError("HP test-only sentinel missing")
        if list(self.test_splits) != ["test"] or list(self.valid_splits):
            raise RuntimeError("Expected test-only evaluation")
        if report_json.exists() or report_csv.exists():
            raise FileExistsError("Test result already exists")
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
        print(f"[HP TEST CKPT PASS] {trial_name} epoch={epoch} "
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
            "test_uauc": uauc, "source_protocol": PROTOCOL,
            "train_valid_auc_best": valid_auc,
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
        print(f"[HP TEST RESULT] {trial_name} AUC={auc:.9f} UAUC={uauc:.9f} "
              f"epoch={epoch}", flush=True)
        return report

    RunnerBase.train = test_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--trial", required=True,
                        choices=sorted(hf.TRIAL_BY_NAME))
    parser.add_argument("--output-protocol", default=OUTPUT_PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_protocol):
        raise ValueError("Unsafe output protocol")
    trial = hf.TRIAL_BY_NAME[args.trial]
    checkpoint = _checkpoint(args.trial)
    valid_epoch, valid_auc = _best_epoch(args.trial)
    source_yaml, source = _source(args.trial)
    test_yaml = _test_config(source, args.trial, args.output_protocol)
    reports = HERE / "out" / args.output_protocol
    report_json = reports / (args.trial + "_test_metrics.json")
    report_csv = reports / (args.trial + "_test_metrics.csv")
    print(f"[HP TEST PLAN] trial={args.trial} ckpt={checkpoint}")
    print(f"[HP TEST PLAN] report={report_csv}")
    if args.check:
        print("[HP TEST CHECK PASS]")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    if report_json.exists() or report_csv.exists():
        raise FileExistsError("Existing result; use a new --output-protocol")
    hf.moe.base._gpu0_preflight(f"HP-test-{args.trial}")
    _install_runtime(trial)
    _install_eval_runner(args.trial, checkpoint, report_json, report_csv,
                         valid_epoch, valid_auc)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
