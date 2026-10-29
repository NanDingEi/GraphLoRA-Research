#!/usr/bin/env python3
"""Four NEW one-epoch graph-CoRA trials in the existing NIGHT26 worktree.

Prioritize fixed, no-new-param improvements near proven E2 lambda=.2.
All four use an opt-in graph_base performance path, preserving differentiable
FULL LightGCN propagation and the original LLM + CoRA + CTR/KD losses.
"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))
old_paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
             if p and Path(p).resolve() != ROOT]
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + old_paths)

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import night25 as n25
from experiments.no31_cora_original.night26_seven import find_artifact, verify_artifact

COMMON = dict(n25.BASE, graph_layers=3, graph_init="mf",
              graph_pretrain_epochs=0, graph_aux_weight=0.0,
              graph_lr_scale=0.1, graph_aggregation="uniform",
              graph_mode="graph_base", graph_joint=True, graph_fusion="fixed",
              graph_anchor_lambda=0.2, amp_init_scale=256.0,
              # All performance controls are opt-in. Previous runs unchanged.
              fast_graph_base=True,
              graph_audit_interval=256,
              fast_check_interval=256,
              empty_cache_interval=0)
ARMS = (
    dict(id="N1_L3_L015", title="Fixed three-hop graph fusion lambda .15",
         graph_anchor_lambda=0.15),
    dict(id="N2_L3_L025", title="Fixed three-hop graph fusion lambda .25",
         graph_anchor_lambda=0.25),
    dict(id="N3_USER01_ITEM03", title="Fixed user .1, item .3 graph coefficients",
         graph_fusion="fixed_pair", graph_user_lambda=0.1,
         graph_item_lambda=0.3),
    dict(id="N4_USER03_ITEM01", title="Fixed user .3, item .1 graph coefficients",
         graph_fusion="fixed_pair", graph_user_lambda=0.3,
         graph_item_lambda=0.1),
)
IDS = tuple(a["id"] for a in ARMS)

def save_summary(root, records):
    keys = ("id", "auc", "uauc", "delta_vs_g2", "delta_uauc_vs_g2",
            "graph_anchor_lambda", "graph_user_lambda", "graph_item_lambda",
            "graph_fusion", "graph_aggregation", "fast_graph_base",
            "graph_audit_interval", "fast_check_interval", "empty_cache_interval",
            "amp_summary", "seed", "log", "checkpoint", "graph_artifact")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in keys})

def main():
    p = argparse.ArgumentParser(description="Four new fixed graph-fusion methods; GPU0 serial")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    p.add_argument("--only", choices=IDS)
    p.add_argument("--start", choices=IDS)
    p.add_argument("--graph-artifact", type=Path)
    # Compare exact same four methods with no performance optimization;
    # using this only for short profiling/debugging (not necessary for science).
    p.add_argument("--legacy-perf", action="store_true",
                   help="disable all optional performance controls for diagnosis")
    args = p.parse_args()
    if args.only and args.start:
        p.error("--only and --start cannot be combined")
    cfg = core.prepared_config()
    chosen = list(ARMS)
    if args.only:
        chosen = [a for a in ARMS if a["id"] == args.only]
    elif args.start:
        chosen = list(ARMS)[IDS.index(args.start):]
    base = dict(COMMON)
    if args.legacy_perf:
        base.update(fast_graph_base=False, graph_audit_interval=1,
                    fast_check_interval=1, empty_cache_interval=1)
    print("[N28-FOUR-PLAN] Four NEW trials only, no historical reruns. "
          "GPU0 seed42 foreground serial, one LLM epoch / Valid-only. "
          "Full-graph LightGCN gradients ON. No graph BCE.", flush=True)
    for i, arm in enumerate(chosen, 1):
        row = dict(base, **arm)
        print(f"[N28-FOUR-ARM] {i}/{len(chosen)} {row['id']} "
              f"fixed_fusion={row['graph_fusion']} "
              f"lambda={row['graph_anchor_lambda']} "
              f"user_lambda={row.get('graph_user_lambda', row['graph_anchor_lambda'])} "
              f"item_lambda={row.get('graph_item_lambda', row['graph_anchor_lambda'])}",
              flush=True)
    artifact = find_artifact(args.graph_artifact)
    if artifact is not None:
        verify_artifact(artifact, cfg)
    if args.plan:
        print(f"[N28-FOUR-ARTIFACT] reuse={artifact or 'create once'} "
              f"fast_graph_base={base['fast_graph_base']} "
              f"empty_cache_interval={base['empty_cache_interval']}", flush=True)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 required")
    if core.gpu_is_in_use():
        raise RuntimeError("GPU0 occupied: refuse parallel training")
    for check in ("test_night8_cpu.py", "test_night24_cpu.py",
                  "test_night25_cpu.py", "test_night26_cpu.py",
                  "test_night26_new7_cpu.py", "test_night26_four_cpu.py"):
        subprocess.run([sys.executable, "experiments/no31_cora_original/" + check],
                       cwd=str(ROOT), check=True)
    root = ROOT / "logs" / ("NO31_CORA_NIGHT26_FOUR_"
                           + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    root.mkdir(parents=True, exist_ok=False)
    if artifact is None:
        artifact, train, _ = n25.prepare_graph(
            ("N28_SHARED_MF_L3", "mf", 3, "Four new fixed fusion experiments"),
            cfg, root, random_epochs=500)
    else:
        train = (Path(cfg["datasets"]["amazon_ood"]["build_info"]["storage"])
                 / cfg["datasets"]["amazon_ood"]["train_file"]).resolve()
    verify_artifact(artifact, cfg)
    (root / "plan.json").write_text(json.dumps(dict(
        experiments=chosen, common=base, seed=42, graph_artifact=str(artifact),
        model="Vicuna-7B, MF+3layer LightGCN, original Group8 CoRA",
        split="valid", trained_epochs_per_arm=1,
        historical_E2_auc=0.7297545888776531,
        historical_E2_uauc=0.6885904780046014), indent=2))
    records = []
    for i, arm in enumerate(chosen, 1):
        row = dict(base, **arm, collab_proto_path=str(artifact),
                   collab_train_path=str(train))
        try:
            if core.gpu_is_in_use():
                raise RuntimeError("GPU0 unexpectedly occupied between arms")
            result = core.run_one(row, cfg, root)
            result["graph_artifact"] = str(artifact)
            if result["graph_fusion"] != row["graph_fusion"]:
                raise RuntimeError("Graph fusion mode mismatch")
            if row["fast_graph_base"] and result["counts"].get("fast_graph_forwards", 0) <= 0:
                raise RuntimeError("Fast graph path not actually called")
            records.append(result)
            with (root / "results.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(result) + "\n")
            save_summary(root, records)
            print(f"[N28-FOUR-DONE] {i}/{len(chosen)} {row['id']} "
                  f"ValidAUC={result['auc']:.9f} ValidUAUC={result['uauc']:.9f}",
                  flush=True)
        except BaseException as e:
            (root / "failed.json").write_text(json.dumps(dict(
                failed_id=row["id"], error=str(e),
                completed=[a["id"] for a in records]), indent=2))
            print(f"[N28-FOUR-ABORT] {row['id']} "
                  f"completed={len(records)} results={root/'summary.csv'}",
                  flush=True)
            raise
    print(f"[N28-FOUR-COMPLETE] summary={root/'summary.csv'}", flush=True)

if __name__ == "__main__":
    main()
