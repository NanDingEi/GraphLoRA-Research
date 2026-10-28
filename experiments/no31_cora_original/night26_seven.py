#!/usr/bin/env python3
"""NIGHT26 seven one-epoch GPU0 Valid-only serial trials IN THE SAME WORKTREE.

Experiment order isolates historical AMP -> FP32 CoRA -> AMP scale -> graph
wrapper -> three-layer LightGCN fusion -> adaptive hop aggregation.
Do not run concurrently with an existing NIGHT26 process.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))
old_path = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if p and Path(p).resolve() != ROOT]
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + old_path)

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import night25 as n25

for module in (core, n25):
    if not Path(module.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"Wrong worktree import: {module.__file__}")

# EVERY GPU run starts from a fresh Vicuna/MF checkpoint, seed=42 and a
# one-epoch training window (with unchanged three-epoch LR horizon).
# The previous N26 C0 result is NOT reused as an anchor because its
# graph setup/GradScaler were changed and we need an identical session.
COMMON = dict(n25.BASE, graph_layers=3, graph_init="mf",
              graph_pretrain_epochs=0, graph_aux_weight=0.0,
              graph_lr_scale=0.1, graph_aggregation="uniform")
ARMS = [
    dict(id="S0_HIST_GROUP8_AMP", title="Historical group8 alpha48 MF CoRA; legacy AMP",
         graph_mode="off", graph_joint=False, force_fp32_cora=False,
         graph_anchor_lambda=0.0, amp_init_scale=65536.0),
    dict(id="S1_NO_GRAPH_FP32_65536", title="S0 except FP32 QFormer and CoRA-P",
         graph_mode="off", graph_joint=False, force_fp32_cora=True,
         graph_anchor_lambda=0.0, amp_init_scale=65536.0),
    dict(id="S2_NO_GRAPH_FP32_256", title="S1 except AMP loss initial scale 256",
         graph_mode="off", graph_joint=False, force_fp32_cora=True,
         graph_anchor_lambda=0.0, amp_init_scale=256.0),
    dict(id="S3_GRAPH_WRAPPER_LAMBDA0", title="S2 plus inactive graph wrapper, lambda0",
         graph_mode="graph_base", graph_joint=True, force_fp32_cora=False,
         graph_anchor_lambda=0.0, amp_init_scale=256.0),
    dict(id="S4_MF_L3_LAMBDA010", title="S3 with 3-hop MF LightGCN correction lambda0.1",
         graph_mode="graph_base", graph_joint=True, force_fp32_cora=False,
         graph_anchor_lambda=0.10, amp_init_scale=256.0),
    dict(id="S5_MF_L3_LAMBDA005", title="S4 with graph correction lambda0.05",
         graph_mode="graph_base", graph_joint=True, force_fp32_cora=False,
         graph_anchor_lambda=0.05, amp_init_scale=256.0),
    dict(id="S6_MF_L3_ADAPTIVE005", title="S5 with learned L0-3 graph-hop weights",
         graph_mode="graph_base", graph_joint=True, force_fp32_cora=False,
         graph_anchor_lambda=0.05, graph_aggregation="learned",
         amp_init_scale=256.0),
]
IDS = tuple(x["id"] for x in ARMS)

def resolve_artifact(explicit):
    """Reuse N26 MF-L3 artifact if provided, else latest prior NIGHT26 one.

    This is NOT a new project directory or a new graph pretraining run.
    Every graph arm receives the SAME .npz artifact.
    """
    if explicit:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(
        (ROOT / "logs").glob(
            "NO31_CORA_NIGHT26_*/artifacts/N26_SHARED_MF_L3.npz"
        )
    )
    return candidates[-1].resolve() if candidates else None

def assert_graph_artifact(artifact, cfg):
    import numpy as np
    from experiments.no31_cora_original.graph_cf_pretrain import load_mf_init_table
    if artifact is None:
        return
    manifest = artifact.with_suffix(".json")
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing graph manifest: {manifest}")
    meta = json.loads(manifest.read_text())
    if (meta.get("layers") != 3 or meta.get("mf_initialization") is not True
            or meta.get("epochs") != 0 or meta.get("seed") != 42):
        raise RuntimeError("Shared graph must be 3-hop, actual MF-init, zero BPR, seed42")
    ckpt = Path(cfg["model"]["rec_config"]["pretrained_path"]).expanduser().resolve()
    expect = load_mf_init_table(
        ckpt, int(cfg["model"]["rec_config"]["user_num"]),
        int(cfg["model"]["rec_config"]["item_num"]), 256,
    ).numpy()
    with np.load(artifact, allow_pickle=False) as f:
        if not np.array_equal(f["graph_id_table"], expect):
            raise RuntimeError("Graph artifact raw ID table is NOT identical to active MF checkpoint")
    print(f"[N26-SEVEN-GRAPH-PARITY-PASS] artifact={artifact}", flush=True)

def summarize(root, results):
    names = ("id", "auc", "uauc", "delta_vs_g2", "delta_uauc_vs_g2",
             "graph_mode", "graph_joint", "force_fp32_cora",
             "amp_init_scale", "graph_layers", "graph_aggregation",
             "graph_anchor_lambda", "graph_layer_weights_end",
             "graph_lr_scale", "graph_aux_weight", "amp_summary",
             "seed", "log", "checkpoint", "graph_artifact")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        for record in results:
            writer.writerow({col: record.get(col) for col in names})

def main():
    p = argparse.ArgumentParser(description="Seven serial diagnostic runs, same NIGHT26 worktree")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    p.add_argument("--only", choices=IDS)
    p.add_argument("--start", choices=IDS)
    p.add_argument("--graph-artifact", type=Path,
                   help="Existing NIGHT26 MF-initialized L3 .npz; auto-discover otherwise")
    args = p.parse_args()
    if args.only and args.start:
        p.error("--only and --start cannot both be set")
    chosen = list(ARMS)
    if args.only:
        chosen = [x for x in ARMS if x["id"] == args.only]
    elif args.start:
        chosen = ARMS[IDS.index(args.start):]
    cfg = core.prepared_config()
    print("[N26-SEVEN-PLAN] reuse same NIGHT26 project/worktree; seven *serial* "
          "GPU0 seed42 foreground 1-epoch Valid-only runs, original MF + CTR/KD, "
          "no BPR retraining; outputs in separate subfolders of ONE sweep root",
          flush=True)
    for i, arm in enumerate(chosen):
        print(f"[N26-SEVEN-ARM] {i+1}/{len(chosen)} {arm['id']} "
              f"joint={arm['graph_joint']} fp32={arm['force_fp32_cora']} "
              f"amp_scale={arm['amp_init_scale']:g} "
              f"lambda={arm['graph_anchor_lambda']:g} "
              f"aggregation={arm.get('graph_aggregation', 'uniform')}",
              flush=True)
    artifact = resolve_artifact(args.graph_artifact)
    if artifact is not None:
        assert_graph_artifact(artifact, cfg)
    print(f"[N26-SEVEN-GRAPH] existing={artifact}", flush=True)
    if args.plan:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 explicitly")
    if core.gpu_is_in_use():
        raise RuntimeError("GPU0 busy. Existing NIGHT26 C1/C2 training must FINISH first; "
                           "do not stop or launch any overlapping run.")
    for test in ("test_night8_cpu.py", "test_night24_cpu.py",
                 "test_night25_cpu.py", "test_night26_cpu.py",
                 "test_night26_seven_cpu.py"):
        subprocess.run([sys.executable, f"experiments/no31_cora_original/{test}"],
                       cwd=str(ROOT), check=True)
    root = ROOT / "logs" / ("NO31_CORA_NIGHT26_SEVEN_"
                           + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    root.mkdir(parents=True, exist_ok=False)
    if any(arm["graph_joint"] for arm in chosen) and artifact is None:
        shared = ("N26_SHARED_MF_L3", "mf", 3,
                  "Single MF-initialized L3 graph used by all 7-arm graph trials")
        artifact, train, _ = n25.prepare_graph(shared, cfg, root, random_epochs=500)
    else:
        train = (Path(cfg["datasets"]["amazon_ood"]["build_info"]["storage"])
                 / cfg["datasets"]["amazon_ood"]["train_file"]).resolve()
    if artifact is not None:
        assert_graph_artifact(artifact, cfg)
    (root / "plan.json").write_text(json.dumps(dict(
        experiments=chosen,
        seed=42, split="valid", trained_epochs_per_arm=1, max_epoch_lr_horizon=3,
        mf_checkpoint=cfg["model"]["rec_config"]["pretrained_path"],
        graph_artifact=str(artifact) if artifact else None,
        baseline_reference=dict(G2_auc=core.G2_AUC, G2_uauc=core.G2_UAUC),
    ), indent=2))
    records = []
    for i, arm in enumerate(chosen):
        row = dict(COMMON, **arm)
        if row["graph_joint"]:
            row.update(collab_proto_path=str(artifact),
                       collab_train_path=str(train))
        try:
            if core.gpu_is_in_use():
                raise RuntimeError("GPU0 unexpectedly busy before next arm")
            result = core.run_one(row, cfg, root)
            result["graph_artifact"] = str(artifact) if row["graph_joint"] else ""
            if row.get("graph_aggregation") == "learned":
                weights = result.get("graph_layer_weights_end")
                if not weights or len(weights) != 4:
                    raise RuntimeError("Learned layer coefficients missing")
            records.append(result)
            with (root / "results.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(result) + "\n")
            summarize(root, records)
            print(f"[N26-SEVEN-RESULT] {i+1}/{len(chosen)} {arm['id']} "
                  f"ValidAUC={result['auc']:.9f} ValidUAUC={result['uauc']:.9f} "
                  f"delta_G2={result['delta_vs_g2']:+.9f}", flush=True)
            if arm["id"] == "S0_HIST_GROUP8_AMP":
                drift = abs(result["auc"] - 0.7299284744316927)
                print(f"[N26-SEVEN-ANCHOR] historic_E0_delta_auc="
                      f"{result['auc']-0.7299284744316927:+.7f} "
                      f"abs={drift:.7f} tolerance=0.02", flush=True)
                # A broken historical anchor means the remaining six aren't
                # interpretable; stop before burning the rest of GPU budget.
                if drift > 0.02:
                    raise RuntimeError(
                        "Historical group8 alpha48 anchor drift >0.02 AUC. "
                        "Stop and compare prepared train.yaml / code provenance "
                        "rather than pretending the subsequent six are comparable."
                    )
        except BaseException as exc:
            (root / "failed.json").write_text(json.dumps(dict(
                failed_id=arm["id"], error=str(exc),
                completed=[x["id"] for x in records],
            ), indent=2))
            print(f"[N26-SEVEN-ABORT] at {arm['id']} "
                  f"completed={len(records)} summary={root / 'summary.csv'}",
                  flush=True)
            raise
    print(f"[N26-SEVEN-COMPLETE] {len(records)} records "
          f"summary={root / 'summary.csv'}", flush=True)

if __name__ == "__main__":
    main()
