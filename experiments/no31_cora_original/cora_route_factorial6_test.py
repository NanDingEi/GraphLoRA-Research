#!/usr/bin/env python3
"""One-time, read-only TEST evaluation of the best E1-E6 Valid-AUC variant.

Selected BEFORE seeing Test: E2_USER_L10 (Valid AUC=.731639, UAUC=.695967,
checkpoint epoch=27). Recreates the original D4/C6/CoRA and user Soft4 router,
loads its model-only best checkpoint with full trainable-key/shape verification,
and evaluates the original OOD2 test split through RecRunnerBase.eval_epoch.

Never trains, selects on Test, modifies the source checkpoint, or resumes an
optimizer. Single physical GPU0. Existing train/eval metrics remain untouched.

Usage from CoRA_official repository root:
  python experiments/no31_cora_original/cora_route_factorial6_test.py --check
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_route_factorial6_test.py --run

A unique explicit --output-protocol may be provided for repeatability checks.
"""

import argparse
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cora_route_factorial6 as factorial

ROOT = factorial.moe.base.PROJECT_ROOT
TRIAL_NAME = "E2_USER_L10"
SOURCE_PROTOCOL = "CORA_ROUTE_FACTORIAL6_20261110"
OUTPUT_PROTOCOL = "CORA_ROUTE_FACTORIAL6_E2_TEST_20261110"
EXPECTED_EPOCH = 27
EXPECTED_LAMBDA = 1.0
REFERENCE_VALID_AUC = 0.731639
REFERENCE_VALID_UAUC = 0.695967


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe protocol name: {value!r}")
    return value


def locate_checkpoint(source, given):
    if given:
        path = Path(given).expanduser().resolve()
    else:
        root = HERE / "output" / source / TRIAL_NAME
        candidates = sorted(root.rglob("checkpoint_best.pth")) if root.exists() else []
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one {TRIAL_NAME} checkpoint in {root}, "
                f"found {len(candidates)}: {candidates}. "
                "Pass --checkpoint PATH to choose explicitly."
            )
        path = candidates[0].resolve()
    if not path.is_file() or path.name != "checkpoint_best.pth":
        raise FileNotFoundError(f"Required trained checkpoint_best.pth: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty source checkpoint: {path}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_config(protocol):
    path = HERE / "local_configs" / protocol / (TRIAL_NAME + ".yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    run = cfg["run"]
    if (
        int(run["seed"]) != 42
        or run.get("valid_splits") != ["valid"]
        or bool(run.get("test_splits"))
        or bool(run.get("evaluate"))
        or run.get("mode") != "v2"
        or cfg["model"].get("freeze_rec") is not False
        or bool(cfg["model"].get("ckpt"))
        or bool(run.get("resume_ckpt_path"))
    ):
        raise RuntimeError(
            "Source E2 config is not the original seed42, v2, trainable-MF "
            "Valid-only configuration. Refusing to test a mismatched model."
        )
    return path, cfg


def test_config_path(source_path, cfg, protocol):
    target_dir = HERE / "local_configs" / protocol
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (TRIAL_NAME + "_TEST.yaml")
    cfg = json.loads(json.dumps(cfg))  # independent vanilla mapping
    cfg["run"]["evaluate"] = True
    cfg["run"]["valid_splits"] = []
    cfg["run"]["test_splits"] = ["test"]
    cfg["run"]["output_dir"] = str(HERE / "output" / protocol / TRIAL_NAME)
    cfg["run"]["resume_ckpt_path"] = None
    if cfg["run"].get("train_splits") not in ([], ["train"]):
        raise RuntimeError("Unexpected training dataset configuration")
    rendered = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if target.exists():
        previous = target.read_text(encoding="utf-8")
        if previous != rendered:
            raise RuntimeError(
                f"Refusing to overwrite changed test config {target}; choose "
                "--output-protocol with a new unique name."
            )
    else:
        target.write_text(rendered, encoding="utf-8")
    return target


def check_checkpoint_metadata(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("Expected checkpoint with a 'model' state dictionary")
    fmt = payload.get("next8_checkpoint_format")
    if fmt != "trainable_model_only_v1" or payload.get("optimizer_included") is not False:
        raise RuntimeError(
            f"Unexpected checkpoint format={fmt!r}, optimizer_included="
            f"{payload.get('optimizer_included')!r}. Aborting rather than "
            "silently loading the wrong artifact."
        )
    epoch = payload.get("epoch")
    if int(epoch) != EXPECTED_EPOCH:
        raise RuntimeError(
            f"Checkpoint epoch={epoch!r}, but E2's Best Valid epoch is "
            f"{EXPECTED_EPOCH}. Verify source log and checkpoint."
        )
    saved_run = payload.get("config", {}).get("run", {})
    if int(saved_run.get("seed", -1)) != 42:
        raise RuntimeError("Checkpoint seed mismatch")
    return len(payload["model"])


def install_runtime():
    trial = factorial.TRIAL_BY_NAME[TRIAL_NAME]
    if trial["expert_mode"] != "user_soft4" or float(trial["route_lambda"]) != EXPECTED_LAMBDA:
        raise RuntimeError("E2 route definition unexpectedly changed")
    # Same model construction order as the six-trial training worker.
    factorial.configure(SOURCE_PROTOCOL)
    core = factorial.moe.base.late.base.core
    core.patch_runner(
        screen_epochs=0,
        mf_lr_ratio=factorial.moe.base.MF_LR_RATIO,
        patience=factorial.moe.base.PATIENCE,
    )
    core.combo.install_combo_model(core.combo.NAMES["C6_KD_BOTH"]) if False else None
    # The original training worker installs C6 on CoRA before D4 and MoE.
    factorial.moe.base.late.base.combo.install_combo_model(
        factorial.moe.base.late.base.combo.NAMES["C6_KD_BOTH"]
    )
    factorial.moe.base.late.base._patch_runtime_classes(trial)
    factorial.install_model(trial)
    factorial.install_audit(trial)


def install_test_runner(ckpt, report_path, config_source):
    from minigpt4.runners.runner_base import RunnerBase

    def test_only(self):
        if self.config.run_cfg.get("evaluate") is not True:
            raise RuntimeError("Evaluation-only flag missing: refusing training")
        if list(self.test_splits) != ["test"]:
            raise RuntimeError(f"Expected test-only split, got {self.test_splits}")
        if report_path.exists():
            raise FileExistsError(
                f"Test result already exists: {report_path}; "
                "choose another --output-protocol to repeat explicitly."
            )
        if "test" not in self.dataloaders:
            raise RuntimeError(f"Test split not constructed: {list(self.dataloaders)}")
        self.set_model_mode(self.config.run_cfg.mode)
        model = self.unwrap_dist_model(self.model)
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        n_saved = check_checkpoint_metadata(payload)
        sd = payload["model"]
        actual = model.state_dict()
        required = {name for name, param in model.named_parameters() if param.requires_grad}
        missing = sorted(required - set(sd))
        unexpected = sorted(set(sd) - set(actual))
        mismatched = [
            name for name, val in sd.items()
            if name in actual and tuple(val.shape) != tuple(actual[name].shape)
        ]
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "Checkpoint/trainable model mismatch: "
                f"missing_trainables={missing[:15]} "
                f"unexpected={unexpected[:15]} shape_mismatch={mismatched[:15]}"
            )
        incompatible = model.load_state_dict(sd, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
        print(
            f"[E2 TEST CKPT PASS] epoch={payload['epoch']} "
            f"saved_tensors={n_saved} expected_trainable={len(required)} "
            f"missing_frozen_only={len(incompatible.missing_keys)} "
            f"checkpoint={ckpt}", flush=True,
        )
        model.eval()
        with torch.no_grad():
            result = self.eval_epoch(
                split_name="test", cur_epoch=payload["epoch"], skip_reload=True
            )
        if not isinstance(result, dict):
            raise RuntimeError(f"Expected Test result dict, got {type(result)}")
        auc = float(result["agg_metrics"])
        uauc = result["uauc"]
        if isinstance(uauc, (list, tuple)):
            uauc = uauc[0]
        uauc = float(uauc)
        if not (0 <= auc <= 1 and 0 <= uauc <= 1):
            raise RuntimeError(f"Invalid Test metric values: AUC={auc} UAUC={uauc}")
        record = {
            "trial": TRIAL_NAME,
            "split": "test",
            "auc": auc, "uauc": uauc,
            "checkpoint_best_epoch": int(payload["epoch"]),
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "source_train_config": str(config_source),
            "mode": "v2",
            "lambda": EXPECTED_LAMBDA,
            "seed": 42,
            "valid_auc_selected_before_test": REFERENCE_VALID_AUC,
            "valid_uauc_at_best_auc": REFERENCE_VALID_UAUC,
            "test_selection": "None. Checkpoint selected on Valid AUC only",
            "original_task_metrics": result,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        print(f"[E2 TEST RESULT] AUC={auc:.9f} UAUC={uauc:.9f} "
              f"best_valid_epoch={payload['epoch']}", flush=True)
        print(f"[E2 TEST SAVED] {report_path}", flush=True)
        return record

    RunnerBase.train = test_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--check", action="store_true",
                           help="Check input files, no checkpoint load or GPU")
    selection.add_argument("--run", action="store_true",
                           help="Perform one Test evaluation on GPU0, no training")
    parser.add_argument("--source-protocol", default=SOURCE_PROTOCOL)
    parser.add_argument("--output-protocol", default=OUTPUT_PROTOCOL)
    parser.add_argument("--checkpoint", default="",
                        help="Explicit checkpoint_best.pth, if multiple are found")
    args = parser.parse_args()
    os.chdir(ROOT)
    src = safe_name(args.source_protocol)
    out = safe_name(args.output_protocol)
    config_source, cfg = source_config(src)
    ckpt = locate_checkpoint(src, args.checkpoint)
    report = HERE / "out" / out / (TRIAL_NAME + "_test_metrics.json")
    print(f"[E2 TEST PLAN] src={config_source} ckpt={ckpt}")
    print(f"[E2 TEST PLAN] test_dataset=OOD2 test_ood2.pkl "
          f"source_valid_auc={REFERENCE_VALID_AUC} source_epoch={EXPECTED_EPOCH}")
    print(f"[E2 TEST PLAN] report={report}; no train, no checkpoint overwrite")

    if args.check:
        if report.exists():
            print(f"[E2 TEST CHECK] Report already exists: {report}")
        print("[E2 TEST CHECK PASS] Ready for GPU0 Test inference")
        return
    if report.exists():
        raise FileExistsError(f"Already tested: {report}; use a new --output-protocol")
    # User's environment has one physical GPU0. No GPU1 fallback.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 (GPU0 exclusively)")
    factorial.moe.base._gpu0_preflight("E2-TEST-before-model")
    cfg_path = test_config_path(config_source, cfg, out)
    install_runtime()
    install_test_runner(ckpt, report, config_source)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    print(f"[E2 TEST BEGIN] config={cfg_path}; frozen checkpoint evaluation only")
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
