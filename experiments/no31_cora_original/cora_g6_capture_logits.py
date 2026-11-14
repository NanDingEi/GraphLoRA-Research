#!/usr/bin/env python3
"""Late-layer logit capture for G3_FIXED_A025 (test) or RR5 (valid).

Parameterized sibling of cora_rr1_test_logits.py / cora_rr5_test_logits.py:
  --trial G3_FIXED_A025 --split test    capture G3's RAW layers; its fused
                                        score is exactly z32+0.25*(z31-z32)
  --trial RR5_USER_GATE_R31 --split valid  capture RR5's RAW layers on VALID
                                        for ensemble weight selection

Sanity: the official eval AUC of this run must match the known reference
(G3 test is being produced here; RR5 valid = 0.733768), and the offline
fused reconstruction must match the official AUC within 0.005.
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
import cora_gate_tune6 as g6
import cora_late_layer_diag as diag

ROOT = residual.PROJECT_ROOT
RR6_PROTOCOL = "CORA_RESIDUAL_ROUTING6_20261103"
G6_PROTOCOL = g6.PROTOCOL
OUTPUT_PROTOCOL = "CORA_G6_CAPTURE_20261112"

RR5_VALID_AUC_REFERENCE = 0.733768


def _sha256(path):
    dig = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _trial_spec(trial_name):
    if trial_name == "RR5_USER_GATE_R31":
        trial = residual.TRIAL_BY_NAME[trial_name]
        return dict(trial=trial, protocol=RR6_PROTOCOL, overrides=False,
                    ref_auc=RR5_VALID_AUC_REFERENCE)
    if trial_name in ("G3_FIXED_A025", "G5_UGATE_B20"):
        trial = g6.TRIAL_BY_NAME[trial_name]
        ref = 0.732170 if trial_name == "G5_UGATE_B20" else None
        return dict(trial=trial, protocol=G6_PROTOCOL, overrides=True,
                    ref_auc=ref)
    raise ValueError(f"Unsupported trial: {trial_name}")


def _checkpoint(spec, trial_name):
    folder = HERE / "output" / spec["protocol"] / trial_name
    candidates = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one checkpoint under {folder}: {candidates}")
    return candidates[0].resolve()


def _source(spec, trial_name):
    source_yaml = HERE / "local_configs" / spec["protocol"] / (trial_name + ".yaml")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    run = config["run"]
    if (int(run["seed"]) != 42 or bool(run.get("test_splits"))
            or bool(run.get("evaluate")) or run.get("mode") != "v2"
            or config["model"].get("freeze_rec") is not False
            or config["model"].get("ckpt") or run.get("resume_ckpt_path")):
        raise RuntimeError(f"{trial_name} source config mismatched")
    return source_yaml, config


def _capture_config(source, trial_name, split, out_protocol):
    cfg = json.loads(json.dumps(source))
    cfg["run"]["evaluate"] = False
    cfg["run"]["g6_logits_only"] = True
    if split == "test":
        cfg["run"]["valid_splits"] = []
        cfg["run"]["test_splits"] = ["test"]
    else:
        cfg["run"]["valid_splits"] = ["valid"]
        cfg["run"]["test_splits"] = []
    cfg["run"]["resume_ckpt_path"] = None
    cfg["run"]["output_dir"] = str(
        HERE / "output" / out_protocol / f"{trial_name}_{split}")
    path = (HERE / "local_configs" / out_protocol /
            f"{trial_name}_{split}_LOGITS.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"Different config exists at {path}")
    else:
        path.write_text(rendered, encoding="utf-8")
    return path


def _install(trial, use_overrides):
    if use_overrides:
        g6.bind_globals()
    residual.configure()
    core = residual.late.base.core
    core.patch_runner(screen_epochs=0, mf_lr_ratio=residual.MF_LR_RATIO,
                      patience=residual.PATIENCE)
    residual.late.base.combo.install_combo_model(
        residual.late.base.combo.NAMES["C6_KD_BOTH"])
    residual.late.base._patch_runtime_classes(trial)
    if use_overrides:
        g6.apply_trial_overrides(trial)
    residual.install_trial_module(trial)
    residual.install_optimizer_audit(trial)
    residual.install_trial_forward(trial)
    print(f"[G6CAP MODEL] {trial['name']} installed; capture hooks next")


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


def _uauc(labels, users, scores):
    values = []
    for uid in np.unique(users):
        mask = users == uid
        y = labels[mask]
        if np.any(y == 1) and np.any(y == 0):
            values.append(diag._auc(y, scores[mask]))
    return float(np.mean(values)) if values else float("nan")


def _install_eval_runner(checkpoint, trial_name, split, spec, capture, out_dir):
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.runners.runner_base_rec import RecRunnerBase

    def eval_only(self):
        if self.config.run_cfg.get("g6_logits_only") is not True:
            raise RuntimeError("Sentinel missing")
        model = self.unwrap_dist_model(self.model)
        payload = diag._load_compact_trainable_state(model, checkpoint)
        if hasattr(model, "set_mode"):
            model.set_mode("v2")
        else:
            model.run_mode_ = "v2"
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.model.eval()
        with torch.no_grad():
            result = self.eval_epoch(split, 0, skip_reload=True)
        if not capture["labels"]:
            raise RuntimeError("Nothing captured")

        labels = np.concatenate(capture["labels"]).astype(np.int64)
        users = np.concatenate(capture["users"]).astype(np.int64)
        items = np.concatenate(capture["items"]).astype(np.int64)
        layer_pair = np.concatenate(capture["layer_pair"], axis=1).astype(np.float32)
        yes = layer_pair[:, :, 0].astype(np.float64)
        official_auc = float(result["agg_metrics"])
        uauc_official = result["uauc"]
        if isinstance(uauc_official, (list, tuple)):
            uauc_official = uauc_official[0]

        # Fused-score sanity: the official eval uses the FUSED logits for
        # user_gate / fixed-alpha trials; rebuild offline and compare.
        trial = spec["trial"]
        if trial_name == "G3_FIXED_A025":
            fused = yes[3] + 0.25 * (yes[2] - yes[3])
        else:
            state = payload["model"]
            user_table = state["rec_encoder.user_embedding.weight"].float()
            gk = {k: v for k, v in state.items()
                  if k.startswith("cora_residual_fusion.user_gate.")}
            user = torch.nn.functional.normalize(
                user_table[torch.as_tensor(users, dtype=torch.long)], dim=-1)
            h = torch.nn.functional.layer_norm(
                user, (user.shape[1],),
                gk["cora_residual_fusion.user_gate.0.weight"].float(),
                gk["cora_residual_fusion.user_gate.0.bias"].float(), 1e-5)
            h = torch.nn.functional.gelu(torch.nn.functional.linear(
                h, gk["cora_residual_fusion.user_gate.1.weight"].float(),
                gk["cora_residual_fusion.user_gate.1.bias"].float()))
            gate = torch.sigmoid(torch.nn.functional.linear(
                h, gk["cora_residual_fusion.user_gate.3.weight"].float(),
                gk["cora_residual_fusion.user_gate.3.bias"].float()).reshape(-1)).numpy()
            fused = yes[3] + gate * (yes[2] - yes[3])
        fused_auc = diag._auc(labels, fused)
        if abs(fused_auc - official_auc) > 0.005:
            raise RuntimeError(
                f"Fused reconstruction AUC {fused_auc:.9f} vs official "
                f"{official_auc:.9f}; install mismatch")
        if spec["ref_auc"] is not None and abs(official_auc - spec["ref_auc"]) > 0.01:
            raise RuntimeError(
                f"Official {official_auc:.9f} deviates from reference "
                f"{spec['ref_auc']:.9f}")

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"layer_logits_{split}.npz",
            labels=labels, users=users, items=items,
            yes_logits=yes,
            no_logits=layer_pair[:, :, 1].astype(np.float64),
            yes_minus_no_logits=(layer_pair[:, :, 0] - layer_pair[:, :, 1]).astype(np.float64))
        report = {
            "trial": trial_name, "split": split,
            "official_auc": official_auc,
            "official_uauc": float(uauc_official),
            "rebuilt_fused_auc": float(fused_auc),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "n_samples": int(len(labels)),
        }
        (out_dir / f"capture_{split}_diagnostics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[G6CAP RESULT] {trial_name}/{split} official_auc={official_auc:.9f} "
              f"uauc={float(uauc_official):.9f} fused_check={fused_auc:.9f}",
              flush=True)
        for i, name in enumerate(diag.LAYER_NAMES):
            print(f"  {name} AUC={diag._auc(labels, yes[i]):.6f} "
                  f"UAUC={_uauc(labels, users, yes[i]):.6f}", flush=True)
        print(f"[G6CAP SAVED] {out_dir / f'layer_logits_{split}.npz'}", flush=True)
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
    parser.add_argument("--trial", required=True,
                        choices=("G3_FIXED_A025", "RR5_USER_GATE_R31",
                                 "G5_UGATE_B20"))
    parser.add_argument("--split", required=True, choices=("test", "valid"))
    parser.add_argument("--output-protocol", default=OUTPUT_PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_protocol):
        raise ValueError("Unsafe output protocol")
    spec = _trial_spec(args.trial)
    checkpoint = _checkpoint(spec, args.trial)
    source_yaml, source = _source(spec, args.trial)
    config_path = _capture_config(source, args.trial, args.split,
                                  args.output_protocol)
    out_dir = HERE / "out" / args.output_protocol / f"{args.trial}_{args.split}"
    print(f"[G6CAP PLAN] trial={args.trial} split={args.split} ckpt={checkpoint}")
    print(f"[G6CAP PLAN] config={config_path} output={out_dir}")
    if args.check:
        print("[G6CAP CHECK PASS]")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    if (out_dir / f"layer_logits_{args.split}.npz").exists():
        raise FileExistsError("Already captured; use a new --output-protocol")
    residual.gpu0_preflight(f"G6CAP-{args.trial}-{args.split}")
    _install(spec["trial"], spec["overrides"])
    capture = _install_capture()
    _install_eval_runner(checkpoint, args.trial, args.split, spec, capture,
                         out_dir)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(config_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
