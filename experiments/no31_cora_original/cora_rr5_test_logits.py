#!/usr/bin/env python3
"""Inference-only TEST late-layer logit capture for RR5_USER_GATE_R31.

Same idea as cora_rr1_test_logits.py but for the RR5 user_gate model:
installs the FULL RR5 runtime (D4 + C6 + fusion module + fused eval forward),
loads RR5's best checkpoint, evaluates test once, and captures the RAW
per-layer Yes/No logits at decoder layers 29/30/31/32 (pre-fusion).
The fused score can be reconstructed offline from the captured layers plus
the checkpointed user_gate parameters.

From CoRA_official root:
 python experiments/no31_cora_original/cora_rr5_test_logits.py --check
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_rr5_test_logits.py --run
"""
import argparse
import hashlib
import json
import os
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent.parent
for _d in (str(ROOT_DIR), str(HERE), str(ROOT_DIR.parent)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import cora_residual_routing6 as residual
import cora_late_layer_diag as diag

ROOT = residual.PROJECT_ROOT
TRAIN_PROTOCOL = "CORA_RESIDUAL_ROUTING6_20261103"
OUTPUT_PROTOCOL = "CORA_RR5_TEST_LOGITS_20261112"
NAME = "RR5_USER_GATE_R31"
RELATIVE_CHECKPOINT = (
    "output/CORA_RESIDUAL_ROUTING6_20261103/"
    "RR5_USER_GATE_R31/20261103153/checkpoint_best.pth"
)
TEST_AUC_REFERENCE = 0.739113


def _sha256(path):
    dig = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _checkpoint(explicit):
    path = Path(explicit).expanduser() if explicit else (HERE / RELATIVE_CHECKPOINT)
    path = path.resolve()
    if not path.is_file() or path.name != "checkpoint_best.pth":
        raise FileNotFoundError(f"RR5 checkpoint not found: {path}")
    return path


def _source():
    source_yaml = HERE / "local_configs" / TRAIN_PROTOCOL / (NAME + ".yaml")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    run = config["run"]
    if (int(run["seed"]) != 42 or run.get("valid_splits") != ["valid"]
            or bool(run.get("test_splits")) or bool(run.get("evaluate"))
            or run.get("mode") != "v2"
            or config["model"].get("freeze_rec") is not False
            or config["model"].get("ckpt") or run.get("resume_ckpt_path")):
        raise RuntimeError("RR5 source configuration mismatched")
    return source_yaml, config


def _test_config(source):
    test_cfg = json.loads(json.dumps(source))
    test_cfg["run"]["evaluate"] = False
    test_cfg["run"]["rr5_logits_test_only"] = True
    test_cfg["run"]["valid_splits"] = []
    test_cfg["run"]["test_splits"] = ["test"]
    test_cfg["run"]["resume_ckpt_path"] = None
    test_cfg["run"]["output_dir"] = str(HERE / "output" / OUTPUT_PROTOCOL / NAME)
    path = HERE / "local_configs" / OUTPUT_PROTOCOL / (NAME + "_LOGITS.yaml")
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
    core.patch_runner(screen_epochs=0, mf_lr_ratio=residual.MF_LR_RATIO,
                      patience=residual.PATIENCE)
    residual.late.base.combo.install_combo_model(
        residual.late.base.combo.NAMES["C6_KD_BOTH"])
    residual.late.base._patch_runtime_classes(trial)
    residual.install_trial_module(trial)
    residual.install_optimizer_audit(trial)
    residual.install_trial_forward(trial)
    print("[RR5 LOGITS MODEL] user_gate fusion active; capture hooks wrap "
          "the fused eval; raw per-layer logits recorded pre-fusion")


def _install_capture():
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    if getattr(MiniGPT4Rec_vx, "_rr5_logits_capture_installed", False):
        raise RuntimeError("Capture hook already installed")
    original = MiniGPT4Rec_vx.generate_for_samples_v2
    capture = {"labels": [], "users": [], "items": [],
               "layer_pair": [], "final_pair": []}

    def wrapped(self, samples, *args, **kwargs):
        result, layer_pair, final_pair, max_error = diag._capture_layer_pair(
            self, samples, original, *args, **kwargs)
        capture["labels"].append(diag._to_numpy(samples["label"]).astype(np.int64))
        capture["users"].append(diag._to_numpy(samples["UserID"]).astype(np.int64))
        capture["items"].append(diag._to_numpy(samples["TargetItemID"]).astype(np.int64))
        capture["layer_pair"].append(layer_pair.detach().float().cpu().numpy())
        capture["final_pair"].append(final_pair.detach().float().cpu().numpy())
        if len(capture["labels"]) == 1:
            print(f"[RR5 LOGITS CAPTURE] first_batch={len(capture['labels'][0])} "
                  f"layers={[x + 1 for x in diag.DIAG_LAYERS]} "
                  f"max_error={max_error:.8g}", flush=True)
        return result

    MiniGPT4Rec_vx.generate_for_samples_v2 = wrapped
    MiniGPT4Rec_vx._rr5_logits_capture_installed = True
    return capture


def _uauc(labels, users, scores):
    values = []
    for uid in np.unique(users):
        mask = users == uid
        y = labels[mask]
        if np.any(y == 1) and np.any(y == 0):
            values.append(diag._auc(y, scores[mask]))
    return float(np.mean(values)) if values else float("nan")


def _install_eval_runner(checkpoint, capture, out_dir):
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.runners.runner_base_rec import RecRunnerBase

    def eval_only(self):
        if self.config.run_cfg.get("rr5_logits_test_only") is not True:
            raise RuntimeError("Sentinel missing; refusing to run")
        if list(self.test_splits) != ["test"] or list(self.valid_splits):
            raise RuntimeError("Expected test-only evaluation")
        model = self.unwrap_dist_model(self.model)
        payload = diag._load_compact_trainable_state(model, checkpoint)
        capture["checkpoint_epoch"] = payload.get("epoch")
        if not getattr(model, "pos_ans", None) or model.pos_ans[0] != "Yes":
            raise RuntimeError("Expected v2 Yes/No answer mode")
        if hasattr(model, "set_mode"):
            model.set_mode("v2")
        else:
            model.run_mode_ = "v2"
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.model.eval()
        print("[RR5 LOGITS EVAL START] split=test no_grad=True", flush=True)
        with torch.no_grad():
            result = self.eval_epoch("test", 0, skip_reload=True)
        if not capture["labels"]:
            raise RuntimeError("No test predictions captured")

        labels = np.concatenate(capture["labels"]).astype(np.int64)
        users = np.concatenate(capture["users"]).astype(np.int64)
        items = np.concatenate(capture["items"]).astype(np.int64)
        layer_pair = np.concatenate(capture["layer_pair"], axis=1).astype(np.float32)
        final_pair = np.concatenate(capture["final_pair"], axis=0).astype(np.float32)
        if layer_pair.shape[1] != len(labels):
            raise RuntimeError("Prediction count mismatch")
        max_pair_error = float(np.max(np.abs(layer_pair[-1] - final_pair)))
        if max_pair_error > 0.05:
            raise RuntimeError(f"L32 reconstruction mismatch: {max_pair_error}")
        yes_scores = layer_pair[:, :, 0].astype(np.float64)

        official_auc = float(result["agg_metrics"])
        uauc_official = result["uauc"]
        if isinstance(uauc_official, (list, tuple)):
            uauc_official = uauc_official[0]
        if abs(official_auc - TEST_AUC_REFERENCE) > 0.01:
            raise RuntimeError(
                f"Official test AUC {official_auc:.9f} deviates from the RR5 "
                f"reference {TEST_AUC_REFERENCE:.9f}; runtime install mismatch.")

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "layer_logits_test.npz",
            labels=labels, users=users, items=items,
            yes_logits=yes_scores,
            no_logits=layer_pair[:, :, 1].astype(np.float64),
            yes_minus_no_logits=(layer_pair[:, :, 0] - layer_pair[:, :, 1]).astype(np.float64),
            final_yes_no_logits=final_pair)
        report = {
            "trial": NAME, "split": "test",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_epoch": capture.get("checkpoint_epoch"),
            "n_samples": int(len(labels)),
            "unique_users": int(len(np.unique(users))),
            "official_test_auc": official_auc,
            "official_test_uauc": float(uauc_official),
            "layer32_yes_no_max_abs_error": max_pair_error,
            "layers_one_based": [x + 1 for x in diag.DIAG_LAYERS],
        }
        (out_dir / "test_logits_diagnostics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RR5 LOGITS RESULT] official_test_auc={official_auc:.9f} "
              f"test_uauc={float(uauc_official):.9f} n={len(labels)}", flush=True)
        for i, name in enumerate(diag.LAYER_NAMES):
            z = yes_scores[i]
            print(f"  {name} AUC={diag._auc(labels, z):.6f} "
                  f"UAUC={_uauc(labels, users, z):.6f}", flush=True)
        print(f"[RR5 LOGITS SAVED] {out_dir / 'layer_logits_test.npz'}", flush=True)
        return result

    pending, seen = [RunnerBase], set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        cls.train = eval_only
        pending.extend(cls.__subclasses__())
    if RecRunnerBase not in seen:
        RecRunnerBase.train = eval_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-protocol", default=OUTPUT_PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_protocol):
        raise ValueError("Unsafe output protocol")
    checkpoint = _checkpoint(args.checkpoint)
    source_yaml, source = _source()
    test_yaml = _test_config(source)
    out_dir = HERE / "out" / args.output_protocol
    print(f"[RR5 LOGITS PLAN] checkpoint={checkpoint}")
    print(f"[RR5 LOGITS PLAN] config={test_yaml}; output={out_dir}")
    if args.check:
        print("[RR5 LOGITS CHECK PASS] ready for GPU0 inference")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    if (out_dir / "layer_logits_test.npz").exists():
        raise FileExistsError("Test logits already captured; use new --output-protocol")
    residual.gpu0_preflight("RR5-logits-before-model")
    _install_rr5()
    capture = _install_capture()
    _install_eval_runner(checkpoint, capture, out_dir)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
