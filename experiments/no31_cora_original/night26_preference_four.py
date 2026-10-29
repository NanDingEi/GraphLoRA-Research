#!/usr/bin/env python3
"""R1-R4: preference-aware Group8 CoRA residual routing. NO LightGCN/GNN forward.

Uses ONE existing GraphLoRA_NIGHT26 worktree, no new branch/worktree.
One epoch, seed42, strict GPU0 foreground serial, Valid AUC/UAUC only.
"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))
prior = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
         if p and Path(p).resolve() != ROOT]
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + prior)

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original.night20 import BASE as G2_BASE

COMMON = dict(
    G2_BASE,
    group_projectors="group8", rank_profile="uniform",
    graph_mode="off", graph_joint=False, collab_mode="off",
    preference_epsilon=0.1,
    # Opt-in engineering speedup; no change to learned model equations.
    empty_cache_interval=0,
    fast_check_interval=256,
)
ARMS = (
    dict(id="R1_GLOBAL_PREF", title="One sample-wise preference CoRA residual router",
         preference_router="global"),
    dict(id="R2_DEPTH8_PREF", title="Eight depth-group preference residual gates",
         preference_router="depth8"),
    dict(id="R3_QK_VO_PREF", title="Two QK / VO projection preference residual gates",
         preference_router="projection2"),
    dict(id="R4_DEPTH8_QKVO_PREF", title="Eight depth by four projection preference gates",
         preference_router="depth_projection32"),
)
IDS = tuple(x["id"] for x in ARMS)

def write_summary(root, results):
    cols = ("id", "auc", "uauc", "delta_vs_g2", "delta_uauc_vs_g2",
            "preference_router", "preference_epsilon", "preference_router_end",
            "group_projectors", "graph_mode", "graph_joint",
            "empty_cache_interval", "fast_check_interval",
            "wall_seconds", "seed", "log", "checkpoint")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in results:
            w.writerow({k: x.get(k) for k in cols})

def main():
    ap = argparse.ArgumentParser(description="Four lightweight recommendation-aware CoRA routers")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--only", choices=IDS)
    ap.add_argument("--start", choices=IDS)
    ap.add_argument("--legacy-cache", action="store_true",
                    help="restore original per-batch CUDA empty_cache")
    args = ap.parse_args()
    if args.only and args.start:
        ap.error("--only and --start are mutually exclusive")
    cfg = core.prepared_config()
    if cfg["run"]["seed"] != 42:
        raise RuntimeError("Do not change seed42")
    common = dict(COMMON)
    if args.legacy_cache:
        common["empty_cache_interval"] = 1
    arms = list(ARMS)
    if args.only:
        arms = [x for x in arms if x["id"] == args.only]
    elif args.start:
        arms = arms[IDS.index(args.start):]
    print("[N29-FOUR-PLAN] NO graph propagation, no pretraining, original "
          "MF mixed prompt + NO31 CTR/KD, Group8 original CoRA, "
          "seed42 GPU0 only, one-epoch Valid-only, foreground serial", flush=True)
    for j, arm in enumerate(arms, 1):
        print(f"[N29-FOUR-ARM] {j}/{len(arms)} {arm['id']} "
              f"mode={arm['preference_router']} epsilon=0.1 "
              f"empty_cache_interval={common['empty_cache_interval']}", flush=True)
    if args.plan:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 explicitly")
    if core.gpu_is_in_use():
        raise RuntimeError("GPU0 occupied; do not overlap with N3/N4 or other runs")
    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night26_preference_cpu.py"],
        cwd=str(ROOT), check=True)
    root = ROOT / "logs" / ("NO31_CORA_NIGHT26_PREF4_"
                           + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(dict(
        arms=arms, common=common, seed=42, dataset="ml-1m OOD2",
        split="valid", pretrained_MF=True,
        historic_G2_AUC=core.G2_AUC, historic_G2_UAUC=core.G2_UAUC,
        train_epochs=1, no_graph_forward=True,
    ), indent=2))
    results = []
    for j, arm in enumerate(arms, 1):
        row = dict(common, **arm)
        try:
            if core.gpu_is_in_use():
                raise RuntimeError("GPU0 unexpectedly occupied before next experiment")
            start = perf_counter()
            result = core.run_one(row, cfg, root)
            result["wall_seconds"] = round(perf_counter()-start, 3)
            if result["graph_joint"] or result["graph_mode"] != "off":
                raise RuntimeError("R1-R4 accidentally activated an online graph")
            if result["counts"].get("router_projection_forwards", 0) < 128:
                raise RuntimeError("Not all 128 CoRA projections used preference router")
            results.append(result)
            with (root / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result)+"\n")
            write_summary(root, results)
            print(f"[N29-FOUR-DONE] {j}/{len(arms)} {arm['id']} "
                  f"ValidAUC={result['auc']:.9f} ValidUAUC={result['uauc']:.9f} "
                  f"wall_seconds={result['wall_seconds']:.1f}", flush=True)
        except BaseException as exc:
            (root / "failed.json").write_text(json.dumps(dict(
                id=arm["id"], error=str(exc),
                completed=[r["id"] for r in results],
            ), indent=2))
            print(f"[N29-FOUR-ABORT] {arm['id']} summary={root/'summary.csv'}", flush=True)
            raise
    print(f"[N29-FOUR-COMPLETE] {root/'summary.csv'}", flush=True)

if __name__ == "__main__":
    main()
