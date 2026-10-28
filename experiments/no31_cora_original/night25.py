#!/usr/bin/env python3
"""NIGHT25: THREE foreground-serial MF-initialized / random-long LightGCN arms.

All arms retain MF as CoRA's anchor and train a LightGCN correction,
then jointly train graph ID embeddings, CoRA and LLM on GPU0/seed42.
"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# A previous GraphLoRA worktree may already appear in PYTHONPATH, even
# when this script is launched from NIGHT25. Remove any duplicate local
# entry then FORCE NIGHT25 to index 0 before importing a project module.
sys.path[:] = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() != ROOT
]
sys.path.insert(0, str(ROOT))
# Subprocesses inherit the calling shell's PYTHONPATH. Pin all CPU
# preflights and subsequent entry.py runs to NIGHT25, not NIGHT24.
_path_entries = [
    value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if value and Path(value).resolve() != ROOT
]
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + _path_entries)

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import graph_cf_pretrain as _graph_module
load_mf_init_table = _graph_module.load_mf_init_table
for _module in (core, _graph_module):
    if not Path(_module.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(
            f"N25 worktree mismatch: {_module.__file__} not under {ROOT}"
        )
print(f"[N25-IMPORT-PATH-OK] root={ROOT} graph_module={_graph_module.__file__}",
      flush=True)

ARMS = (
    ("M1_MF_INIT_L2", "mf", 2, "MF initialization, 2 LightGCN propagations"),
    ("M2_MF_INIT_L3", "mf", 3, "MF initialization, 3 LightGCN propagations"),
    ("M3_RANDOM_LONG_L2", "random", 2, "Random initialization, 500-1000 BPR epochs, 2 LightGCN layers"),
)
IDS = tuple(a[0] for a in ARMS)

BASE = dict(
    layers="0-31",
    projections="q_proj,k_proj,v_proj,o_proj",
    cf_input="concat", injection_mode="shared",
    rank_profile="uniform", group_projectors="group8",
    granularity_mode="original", fine_residual_mode="off",
    collab_mode="off", graph_mode="graph_base", graph_joint=True,
    alpha=48, p_lr=1.0, residual_ratio=0.0,
    depth_profile="uniform",
    projection_scales="q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    graph_lr_scale=0.1, graph_aux_weight=0.0,
    seed=42, amp_init_scale=256.0,
)

def expected_graph_artifact(artifact, mf_checkpoint, kind, layers, n_users, n_items):
    with np.load(artifact, allow_pickle=False) as f:
        for field in ("graph_id_table", "user_graph", "item_graph",
                      "item_to_cluster", "prototypes", "item_residual"):
            if field not in f:
                raise KeyError(f"{artifact}: missing {field}")
        raw = f["graph_id_table"]
        if raw.shape != (n_users + n_items, 256):
            raise ValueError("Wrong graph ID table shape")
        if f["prototypes"].shape != (16, 256):
            raise ValueError("Wrong K16 graph prototype shape")
        if not np.isfinite(raw).all():
            raise FloatingPointError("Graph node table contains nonfinite numbers")
        if kind == "mf":
            mf = load_mf_init_table(mf_checkpoint, n_users, n_items, 256).numpy()
            if not np.array_equal(raw, mf):
                raise RuntimeError("MF graph artifact did NOT copy exact pretrained MF table")
            print(f"[N25-MF-WEIGHT-PARITY-PASS] layers={layers} "
                  f"raw_id_embeddings_exact_equal_to_pretrained_MF=True", flush=True)
        else:
            print("[N25-RANDOM-INIT-ASSERT] raw graph embeddings generated "
                  "without any pretrained MF weight initialization", flush=True)
    manifest = json.loads(artifact.with_suffix(".json").read_text())
    if (manifest["layers"] != layers
            or manifest["mf_initialization"] != (kind == "mf")
            or (kind == "mf" and manifest["epochs"] != 0)):
        raise RuntimeError("Graph initialization/pretraining manifest inconsistent")
    return manifest

def stream_pretrain(command, log_file, kind, layers):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[N25-PRETRAIN-START] type={kind} graph_layers={layers} "
          f"log={log_file}", flush=True)
    # This subprocess is foreground; parent blocks and waits for completion.
    with log_file.open("x", encoding="utf-8") as saved:
        proc = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                saved.write(line)
                saved.flush()
                matched = re.search(r"\[N23-GRAPH-EPOCH\]\s+(\d+)/(\d+)", line)
                if (matched and (int(matched.group(1)) == 1
                                 or int(matched.group(1)) % 25 == 0
                                 or matched.group(1) == matched.group(2))):
                    print(f"[N25-PRETRAIN] {line.rstrip()}", flush=True)
                elif any(mark in line for mark in (
                    "[N25-GRAPH-INIT]", "[N23-GRAPH-INPUT]",
                    "[N23-GRAPH-SAVED]", "Traceback", "Error", "Exception",
                )):
                    print(f"[N25-PRETRAIN] {line.rstrip()[:700]}", flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Graph pretraining failed exit={code}: {log_file}")
    print(f"[N25-PRETRAIN-DONE] type={kind} layers={layers}", flush=True)

def prepare_graph(arm, cfg, root, random_epochs):
    ident, kind, layers, _ = arm
    artifact = root / "artifacts" / (ident + ".npz")
    artifact.parent.mkdir(exist_ok=True)
    ckpt = Path(cfg["model"]["rec_config"]["pretrained_path"]).expanduser().resolve()
    train = (Path(cfg["datasets"]["amazon_ood"]["build_info"]["storage"])
             / cfg["datasets"]["amazon_ood"]["train_file"]).resolve()
    if not ckpt.is_file() or not train.is_file():
        raise FileNotFoundError(f"MF checkpoint/train not found: {ckpt}; {train}")
    epochs = 0 if kind == "mf" else random_epochs
    command = [
        sys.executable, "-u", "experiments/no31_cora_original/graph_cf_pretrain.py",
        "--train-file", str(train),
        "--shape-checkpoint", str(ckpt),
        "--output", str(artifact),
        "--device", "cpu" if kind == "mf" else "cuda",
        "--seed", "42",
        "--dim", "256", "--layers", str(layers), "--clusters", "16",
        "--epochs", str(epochs),
    ]
    if kind == "mf":
        command.extend(["--mf-init-checkpoint", str(ckpt)])
    stream_pretrain(command, root / (ident + "_graph_pretrain.log"), kind, layers)
    if not artifact.is_file():
        raise RuntimeError("Expected graph artifact missing after pretraining")
    manifest = expected_graph_artifact(
        artifact, ckpt, kind, layers,
        cfg["model"]["rec_config"]["user_num"],
        cfg["model"]["rec_config"]["item_num"],
    )
    if manifest["epochs"] != epochs:
        raise RuntimeError("Incorrect graph pretraining epoch count")
    return artifact, train, epochs

def main():
    parser = argparse.ArgumentParser(description="NIGHT25 three serial graph-init experiments")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--only", choices=IDS)
    parser.add_argument("--start", choices=IDS, help="Resume from this arm in a NEW run directory")
    parser.add_argument("--random-epochs", type=int, default=500,
                        help="M3 BPR pretraining: 500 through 1000 (default 500)")
    parser.add_argument("--anchor-lambda", type=float, default=0.1,
                        help="Fixed MF-anchored LightGCN correction coefficient (default .1)")
    parser.add_argument("--graph-lr-scale", type=float, default=0.1)
    args = parser.parse_args()
    if args.only and args.start:
        parser.error("--only and --start cannot be combined")
    if args.random_epochs < 500 or args.random_epochs > 1000:
        parser.error("--random-epochs must be between 500 and 1000")
    if not (0 < args.anchor_lambda <= 1 and 0 < args.graph_lr_scale <= 1):
        parser.error("anchor-lambda must be (0,1] and graph LR scale (0,1]")
    cfg = core.prepared_config()
    arms = list(ARMS)
    if args.only:
        arms = [arm for arm in arms if arm[0] == args.only]
    if args.start:
        arms = arms[IDS.index(args.start):]
    print("[N25] foreground serial; GPU0 seed42; MF anchor retained; "
          "LightGCN node table trainable; original MF prompt/NGCF CTR/KD retained; "
          "one epoch Valid only; graph BCE disabled; CoRA FP32; "
          f"random BPR pretrain={args.random_epochs} epochs", flush=True)
    for ident, kind, layers, title in arms:
        print(f"[N25-PLAN] {ident} kind={kind} graph_layers={layers} "
              f"graph_pretrain_epochs={0 if kind == 'mf' else args.random_epochs} "
              f"anchor_lambda={args.anchor_lambda:g} graph_lr_scale={args.graph_lr_scale:g} "
              f"title={title}", flush=True)
    if args.plan:
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("NIGHT25 requires CUDA_VISIBLE_DEVICES=0 (GPU0 only)")
    if core.gpu_is_in_use():
        raise RuntimeError("GPU0 already busy. Stop the existing J series before NIGHT25.")
    subprocess.run([sys.executable,
                    "experiments/no31_cora_original/test_night8_cpu.py"],
                   cwd=str(ROOT), check=True)
    subprocess.run([sys.executable,
                    "experiments/no31_cora_original/test_night24_cpu.py"],
                   cwd=str(ROOT), check=True)
    subprocess.run([sys.executable,
                    "experiments/no31_cora_original/test_night25_cpu.py"],
                   cwd=str(ROOT), check=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = ROOT / "logs" / f"NO31_CORA_NIGHT25_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps({
        "experiment_order": [a[0] for a in arms],
        "random_bpr_pretrain_epochs": args.random_epochs,
        "graph_lr_scale": args.graph_lr_scale,
        "graph_aux_weight": 0.0,
        "mf_anchor_lambda": args.anchor_lambda,
        "seed": 42, "eval_split": "valid", "train_epochs_per_arm": 1,
    }, indent=2))
    print(f"[N25-ROOT] {root}", flush=True)
    records = []
    for arm in arms:
        ident, kind, layers, title = arm
        try:
            artifact, train, epochs = prepare_graph(arm, cfg, root, args.random_epochs)
            row = dict(BASE, id=ident, title=title, graph_init=kind,
                       graph_layers=layers, graph_anchor_lambda=args.anchor_lambda,
                       graph_lr_scale=args.graph_lr_scale,
                       graph_pretrain_epochs=epochs,
                       collab_proto_path=str(artifact), collab_train_path=str(train))
            if core.gpu_is_in_use():
                raise RuntimeError("Unexpected GPU0 compute process before LLM run")
            result = core.run_one(row, cfg, root)
            # Retain complete metadata outside legacy small summary.csv.
            if result["graph_layers"] != layers or result["graph_init"] != kind:
                raise RuntimeError("Graph experiment result metadata mismatch")
            records.append(result)
            with (root / "results.jsonl").open("a", encoding="utf-8") as fout:
                fout.write(json.dumps(result) + "\n")
            cols = ("id", "auc", "uauc", "delta_vs_g2", "delta_uauc_vs_g2",
                    "graph_init", "graph_layers", "graph_pretrain_epochs",
                    "graph_anchor_lambda", "graph_aux_weight", "graph_lr_scale",
                    "seed", "checkpoint", "graph_artifact")
            with (root / "summary.csv").open("w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=cols)
                writer.writeheader()
                for record in records:
                    writer.writerow({field: record.get(field) for field in cols})
            print(f"[N25-ARM-DONE] {ident} valid_auc={result['auc']:.9f} "
                  f"valid_uauc={result['uauc']:.9f}", flush=True)
        except BaseException as exc:
            (root / "failed.json").write_text(json.dumps({
                "id": ident, "error": str(exc),
                "completed": [r["id"] for r in records],
            }, indent=2))
            print(f"[N25-ABORT] {ident} failed; remaining arms NOT launched; "
                  f"previous results remain in {root}", flush=True)
            raise
    print(f"[N25-COMPLETE] three-arm sweep completed: {root / 'summary.csv'}",
          flush=True)

if __name__ == "__main__":
    main()
