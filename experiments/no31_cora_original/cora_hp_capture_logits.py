#!/usr/bin/env python3
"""Inference-only test logit capture for CORA_HIER_FULLSTACK trials.

Installs the exact fullstack runtime (D4 + C6 + hier_personal routing via
cora_hier_fullstack.install_model), loads the trial's best checkpoint with
strict audit, evaluates test ONCE, and captures per-layer Yes/No logits.
The L32 Yes logit equals the model's official masked-forward score
(verified against official AUC within 0.005 before saving).

From CoRA_official root:
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp_capture_logits.py \
     --run --trial HP12_FINE_B07
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

import cora_hier_fullstack as hf
import cora_late_layer_diag as diag

ROOT = hf.moe.base.PROJECT_ROOT
OUTPUT_PROTOCOL = "CORA_HP_TEST_LOGITS_20261115"
BEST_LINE = re.compile(r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=([-+\d.eE]+)")


def _sha256(path):
    dig = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _checkpoint(trial_name):
    folder = HERE / "output" / hf.DEFAULT_PROTOCOL / trial_name
    candidates = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one checkpoint under {folder}: {candidates}")
    return candidates[0].resolve()


def _best_epoch(trial_name):
    log = HERE / "logs" / hf.DEFAULT_PROTOCOL / f"{trial_name}.log"
    matches = BEST_LINE.findall(log.read_text(encoding="utf-8", errors="replace"))
    epoch, auc = max(((int(a), float(b)) for a, b in matches), key=lambda t: t[1])
    return epoch, auc


def _uauc(labels, users, scores):
    values = []
    for uid in np.unique(users):
        mask = users == uid
        y = labels[mask]
        if np.any(y == 1) and np.any(y == 0):
            values.append(diag._auc(y, scores[mask]))
    return float(np.mean(values)) if values else float("nan")


def _install_capture():
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
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
        return result

    MiniGPT4Rec_vx.generate_for_samples_v2 = wrapped
    return capture


def _install_eval_runner(checkpoint, trial_name, capture, out_dir, valid_epoch,
                         capture_split="test", trial=None):
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.runners.runner_base_rec import RecRunnerBase

    def eval_only(self):
        if self.config.run_cfg.get("hp_test_only") is not True:
            raise RuntimeError("Sentinel missing")
        if capture_split == "test":
            if list(self.test_splits) != ["test"] or list(self.valid_splits):
                raise RuntimeError("Expected test-only evaluation")
        else:
            if list(self.valid_splits) != ["valid"] or list(self.test_splits):
                raise RuntimeError("Expected valid-only evaluation")
        if (out_dir / f"layer_logits_{capture_split}.npz").exists():
            raise FileExistsError("Already captured")
        model = self.unwrap_dist_model(self.model)
        payload = diag._load_compact_trainable_state(model, checkpoint)
        epoch = int(payload.get("epoch", -1))
        if valid_epoch is not None and epoch != valid_epoch:
            raise RuntimeError(f"Checkpoint epoch {epoch} != best {valid_epoch}")
        if hasattr(model, "set_mode"):
            model.set_mode("v2")
        else:
            model.run_mode_ = "v2"
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.model.eval()
        with torch.no_grad():
            result = self.eval_epoch(capture_split, 0, skip_reload=True)
        if not capture["labels"]:
            raise RuntimeError("Nothing captured")
        labels = np.concatenate(capture["labels"]).astype(np.int64)
        users = np.concatenate(capture["users"]).astype(np.int64)
        items = np.concatenate(capture["items"]).astype(np.int64)
        layer_pair = np.concatenate(capture["layer_pair"], axis=1).astype(np.float32)
        max_pair_error = float(np.max(np.abs(layer_pair[-1] - np.concatenate(
            capture["final_pair"], axis=0))))
        if max_pair_error > 0.05:
            raise RuntimeError(f"L32 reconstruction mismatch: {max_pair_error}")
        yes = layer_pair[:, :, 0].astype(np.float64)
        official_auc = float(result["agg_metrics"])
        fused_scores = None
        if trial.get("logit_gate"):
            # Dual-gate models score with the FUSED logit; rebuild it from
            # the checkpoint's fusion parameters (RR5-style reconstruction).
            state = payload["model"]
            gk = {k: v for k, v in state.items()
                  if k.startswith("cora_residual_fusion.user_gate.")}
            table = state["rec_encoder.user_embedding.weight"].float()
            user_t = torch.nn.functional.normalize(
                table[torch.as_tensor(users, dtype=torch.long)], dim=-1)
            h = torch.nn.functional.layer_norm(
                user_t, (user_t.shape[1],),
                gk["cora_residual_fusion.user_gate.0.weight"].float(),
                gk["cora_residual_fusion.user_gate.0.bias"].float(), 1e-5)
            h = torch.nn.functional.gelu(torch.nn.functional.linear(
                h, gk["cora_residual_fusion.user_gate.1.weight"].float(),
                gk["cora_residual_fusion.user_gate.1.bias"].float()))
            gate = torch.sigmoid(torch.nn.functional.linear(
                h, gk["cora_residual_fusion.user_gate.3.weight"].float(),
                gk["cora_residual_fusion.user_gate.3.bias"].float())
                .reshape(-1)).numpy()
            fused_scores = yes[3] + gate * (yes[2] - yes[3])
            fused_auc = diag._auc(labels, fused_scores)
            if abs(fused_auc - official_auc) > 0.005:
                raise RuntimeError(
                    f"Fused AUC {fused_auc:.9f} != official {official_auc:.9f}")
        else:
            yes32_auc = diag._auc(labels, yes[3])
            if abs(yes32_auc - official_auc) > 0.005:
                raise RuntimeError(
                    f"L32 AUC {yes32_auc:.9f} != official {official_auc:.9f}")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_kw = dict(labels=labels, users=users, items=items,
                       yes_logits=yes,
                       no_logits=layer_pair[:, :, 1].astype(np.float64))
        if fused_scores is not None:
            save_kw["fused_scores"] = fused_scores
        np.savez_compressed(out_dir / f"layer_logits_{capture_split}.npz",
                            **save_kw)
        report = {"trial": trial_name, "split": capture_split,
                  "official_test_auc": official_auc,
                  "official_test_uauc": float(result["uauc"][0]
                                              if isinstance(result["uauc"], (list, tuple))
                                              else result["uauc"]),
                  "checkpoint": str(checkpoint),
                  "checkpoint_sha256": _sha256(checkpoint),
                  "checkpoint_epoch": epoch}
        (out_dir / "capture_test_diagnostics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[HP CAPTURE RESULT] {trial_name} official_auc={official_auc:.9f} "
              f"n={len(labels)}", flush=True)
        print(f"[HP CAPTURE SAVED] {out_dir / f'layer_logits_{capture_split}.npz'}",
              flush=True)
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
    parser.add_argument("--trial", required=True, choices=sorted(hf.TRIAL_BY_NAME))
    parser.add_argument("--split", default="test", choices=("test", "valid"))
    parser.add_argument("--output-protocol", default=OUTPUT_PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_protocol):
        raise ValueError("Unsafe output protocol")
    trial = hf.TRIAL_BY_NAME[args.trial]
    checkpoint = _checkpoint(args.trial)
    valid_epoch, _ = _best_epoch(args.trial)
    out_dir = HERE / "out" / args.output_protocol / f"{args.trial}_{args.split}"
    print(f"[HP CAPTURE PLAN] trial={args.trial} split={args.split} "
          f"ckpt={checkpoint} out={out_dir}")
    if args.check:
        print("[HP CAPTURE CHECK PASS]")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    if (out_dir / f"layer_logits_{args.split}.npz").exists():
        raise FileExistsError("Already captured")
    hf.moe.base._gpu0_preflight(f"HP-capture-{args.trial}")
    hf.configure(hf.DEFAULT_PROTOCOL)
    core = hf.moe.base.late.base.core
    core.patch_runner(screen_epochs=0, mf_lr_ratio=hf.moe.base.MF_LR_RATIO,
                      patience=hf.moe.base.PATIENCE)
    hf.moe.base.late.base.combo.install_combo_model(
        hf.moe.base.late.base.combo.NAMES["C6_KD_BOTH"])
    hf.moe.base.late.base._patch_runtime_classes(trial)
    hf.install_model(trial)
    source_yaml = HERE / "local_configs" / hf.DEFAULT_PROTOCOL / (args.trial + ".yaml")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))
    config["run"]["evaluate"] = False
    config["run"]["hp_test_only"] = True
    if args.split == "test":
        config["run"]["valid_splits"] = []
        config["run"]["test_splits"] = ["test"]
    else:
        config["run"]["valid_splits"] = ["valid"]
        config["run"]["test_splits"] = []
    config["run"]["resume_ckpt_path"] = None
    config["run"]["output_dir"] = str(HERE / "output" / args.output_protocol / args.trial)
    test_yaml_path = HERE / "local_configs" / args.output_protocol / \
        (args.trial + f"_{args.split.upper()}_CAP.yaml")
    test_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False)
    if test_yaml_path.exists() and test_yaml_path.read_text() != rendered:
        raise FileExistsError("Different capture config exists")
    test_yaml_path.write_text(rendered)
    capture = _install_capture()
    _install_eval_runner(checkpoint, args.trial, capture, out_dir, valid_epoch,
                         capture_split=args.split, trial=trial)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
