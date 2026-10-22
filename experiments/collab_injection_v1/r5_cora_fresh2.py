#!/usr/bin/env python3
"""R5-CORA-FRESH: train GraphLoRA + CoRA from epoch 0 (NO S2 checkpoint).

Initialisation:
  pretrained Vicuna-7B = base foundation model;
  randomly initialized MF, GNN, projector, LLM LoRA and CoRA;
  no learned GraphLoRA/S2/R3/R4 weights.
Only one method: CoRA, for TWO COMPLETE epochs with per-epoch Valid AUC/UAUC.
Uses the original R3 adapter runtime (r3_entry.py), NEVER R4 frozen-S2 hooks.
"""
import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from r3_train import smoke_new_params
from r2_train import sha

TAG = "R5_CORA_FRESH2_20261022"
ARM = "CORA_FROM_SCRATCH"
DEFAULT_SOURCE = ROOT / "logs/collab_v1/R4_ISOLATION_20261022/R4_02_CORA_ONLY/FULL_E1.yaml"
ROOT_LOG = ROOT / "logs/collab_v1" / TAG
ROOT_OUT = ROOT / "output" / TAG
METRIC_RX = re.compile(r"\*\*\*auc:\s*([0-9.eE+-]+).*?\*\*\*uauc:\s*\(\s*([0-9.eE+-]+)")


def read_checkpoint(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def metrics(logfile, expected):
    result = []
    for line in logfile.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Averaged stats:" not in line:
            continue
        m = METRIC_RX.search(line)
        if m:
            pair = tuple(float(x) for x in m.groups())
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in pair):
                raise RuntimeError(f"Invalid validation scores {pair}")
            result.append(pair)
    if len(result) != expected:
        raise RuntimeError(f"Expected exactly {expected} valid evaluations, got {result}")
    return result


def run_entry(cfg_path, label, log):
    cmd = [sys.executable, "-u", "experiments/collab_injection_v1/r3_entry.py",
           "--cfg-path", str(cfg_path), "--method", "cora"]
    with log.open("x", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES":"0"},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if any(x in line for x in (
                        "[R3-", "Train: data epoch:", "Averaged stats:",
                        "Saving checkpoint", "Error:", "Traceback", "[FX]")):
                    print(f"[{label}] {line.strip()[:750]}", flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    if code != 0:
        raise RuntimeError(f"{label} exited {code}, see {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\[R3-COUNTERS\]\s*(\{[^\n]+\})", text)
    if match is None:
        raise RuntimeError(f"{label}: R3 runtime counter missing")
    counters = json.loads(match.group(1))
    for key in ("q_cora_forward", "v_cora_forward"):
        if counters.get(key, 0) <= 0:
            raise RuntimeError(f"{label}: CoRA did not execute {key}: {counters}")
    return counters


def unique_checkpoint(output_folder, tag):
    matches = list(output_folder.rglob(f"checkpoint_{tag}.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint_{tag}.pth under {output_folder}, got {matches}")
    return matches[0]


def verify_trainable(ckpt_path, epoch):
    data = read_checkpoint(ckpt_path)
    if int(data.get("epoch", -1)) != epoch:
        raise RuntimeError(f"Wrong checkpoint epoch: {data.get('epoch')}")
    params = data.get("model", {})
    names = tuple(params)
    required = {
        "CoRA": any(k.startswith("r3_adapter.") for k in names),
        "LLM LoRA": any("lora_A." in k or "lora_B." in k for k in names),
        "GNN": any(k.startswith("gcn.") for k in names),
        "MF": any(k.startswith("rec_encoder.") for k in names),
        "Projector": any(k.startswith("llama_proj.") for k in names),
    }
    if not all(required.values()):
        raise RuntimeError(f"Fresh full-model training missing trainable modules: {required}")
    expected_prefixes = ("r3_adapter.", "llama_proj.", "gcn.", "rec_encoder.")
    # Trained checkpoint is intentionally partial: frozen Vicuna base not serialized.
    if not any("lora_B." in k for k in names):
        raise RuntimeError("No trained LLM LoRA B in checkpoint")
    if "optimizer" not in data:
        raise RuntimeError("Missing optimizer state")
    return {"checkpoint": str(ckpt_path), "epoch": epoch, "trainable_present":required,
            "adapter_tensors":sum(k.startswith("r3_adapter.") for k in names)}


def make_yaml(original, steps, dest_output):
    cfg = copy.deepcopy(original)
    m, r = cfg["model"], cfg["run"]
    if m.get("r3_mode") != "cora" or m.get("r4_round") != "02":
        raise RuntimeError("Reference must be exactly R4-02 CoRA")
    if m.get("collab_mode") != "legacy" or int(m.get("gcn_layer", -1)) != 31:
        raise RuntimeError("Original legacy L31 setup required")
    if not bool(m.get("fix_autocast_cache")):
        raise RuntimeError("Original autocast cache fix required")

    # CRITICAL: B means no pretrained recommender and no GraphLoRA/S2 checkpoint.
    m["ckpt"] = ""
    m["rec_config"]["pretrained_path"] = "not_have"
    m["freeze_rec"] = False
    m["freeze_proj"] = False
    m["freeze_lora"] = False
    m.pop("r4_round", None)
    m["r3_mode"] = "cora"
    m.pop("trainable_policy", None)
    m.pop("r2_mode", None)
    if not m.get("llama_model"):
        raise RuntimeError("Missing pretrained Vicuna checkpoint directory")
    # Base Vicuna is pretrained; LLM LoRA, MF, GNN, projector, CoRA are fresh.
    r["resume_ckpt_path"] = None
    r["evaluate"] = False
    r["distributed"] = False
    r["max_epoch"] = 2
    r["diagnostic_stop_after_epoch0"] = False
    r.pop("stop_after_n_epochs", None)
    r["save_checkpoint_every_n_epochs"] = 1
    r["output_dir"] = str(dest_output)
    r["test_splits"] = []  # Never run test during model selection.
    r["valid_splits"] = ["valid"]
    r["seed"] = 42
    # Original GraphLoRA start-from-zero recipe, not S2 low-LR fine-tuning:
    r["init_lr"] = 1e-4
    r["min_lr"] = 8e-5
    r["warmup_lr"] = 1e-5
    r["warmup_steps"] = 200
    for part in ("rec", "proj", "gnn", "lora", "other"):
        r["lr_scale_" + part] = 1.0
    r["skip_validation"] = bool(steps)
    for stale in ("iters_per_epoch", "max_iters", "mid_epoch_eval_iters"):
        r.pop(stale, None)
    if steps is not None:
        r["max_train_steps"] = int(steps)
    else:
        r.pop("max_train_steps", None)
    if m["ckpt"] or m["rec_config"]["pretrained_path"] != "not_have" or r["resume_ckpt_path"]:
        raise RuntimeError("Unexpected pretrained GraphLoRA/recommender warm start")
    if not all(m[k] is False for k in ("freeze_rec", "freeze_proj", "freeze_lora")):
        raise RuntimeError("A trainable branch is frozen")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Must run with CUDA_VISIBLE_DEVICES=0")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Original R4-02 YAML missing: {source}")
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise RuntimeError("Invalid source YAML")

    arm_dir = ROOT_LOG / ARM
    if arm_dir.exists() or (ROOT_OUT/ARM).exists():
        raise FileExistsError(f"Refusing to overwrite prior fresh run: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=False)
    cfg_smoke = arm_dir / "SMOKE.yaml"
    cfg_full = arm_dir / "FULL_E2.yaml"
    cfg_smoke.write_text(yaml.safe_dump(make_yaml(original, 24, ROOT_OUT/ARM/"SMOKE"), sort_keys=False))
    cfg_full.write_text(yaml.safe_dump(make_yaml(original, None, ROOT_OUT/ARM/"FULL_E2"), sort_keys=False))
    manifest = arm_dir/"manifest.json"
    data = {
        "status": "prepared", "method": "cora",
        "definition": "no S2, no rec pretrain; Vicuna pretrain only",
        "source_yaml": str(source), "source_sha256":sha(source),
        "smoke_yaml":str(cfg_smoke), "full_yaml":str(cfg_full),
        "epochs":2, "seed":42, "init_lr":1e-4,
        "base_vicuna":original["model"].get("llama_model"),
        "trainable": ["LLM LoRA", "GNN", "MF", "projector", "CoRA"],
        "valid_only": True,
    }
    def save():
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    print(f"[R5-FRESH-CONFIG] method=CoRA epoch0->epoch1; "
          f"ckpt=NONE pretrained_MF=NONE fresh_LoRA_GNN_projector_CoRA=YES "
          f"source={source}", flush=True)
    if args.prepare_only:
        return
    try:
        data["status"] = "smoke"; save()
        smoke_log = arm_dir/"SMOKE.log"
        smoke_counters = run_entry(cfg_smoke, "SMOKE", smoke_log)
        smoke_ckpt = unique_checkpoint(ROOT_OUT/ARM/"SMOKE", "0")
        smoke_new_params(smoke_ckpt, "cora")
        data["smoke_checkpoint"] = verify_trainable(smoke_ckpt, 0)
        data["smoke_counters"] = smoke_counters

        data["status"] = "full_training"; save()
        full_log = arm_dir/"FULL_E2.log"
        counters = run_entry(cfg_full, "FULL_E2", full_log)
        scores = metrics(full_log, 2)
        checkpoints = [verify_trainable(unique_checkpoint(ROOT_OUT/ARM/"FULL_E2",str(ep)),ep)
                       for ep in (0,1)]
        data["status"] = "complete"
        data["checkpoints"] = checkpoints
        data["counters"] = counters
        data["valid"] = [{"epoch": e, "auc": a, "uauc": u}
                         for e,(a,u) in enumerate(scores)]
        data["delta_second_minus_first"] = {
            "auc":scores[1][0]-scores[0][0],
            "uauc":scores[1][1]-scores[0][1],
        }
        save()
        print(f"[R5-FRESH-COMPLETE] E0 AUC={scores[0][0]:.9f} "
              f"UAUC={scores[0][1]:.9f} E1 AUC={scores[1][0]:.9f} "
              f"UAUC={scores[1][1]:.9f} "
              f"delta_AUC={scores[1][0]-scores[0][0]:+.9f} "
              f"delta_UAUC={scores[1][1]-scores[0][1]:+.9f}", flush=True)
    except BaseException as exc:
        data["status"]="failed"
        data["error"]=repr(exc)
        save()
        print(f"[R5-FRESH-FAILED] {exc!r}",flush=True)
        raise


if __name__=="__main__":
    main()
