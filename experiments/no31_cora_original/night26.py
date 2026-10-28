#!/usr/bin/env python3
"""NIGHT26: matched MF-only FP32 control, lambda ablation, learned graph-depth mixing.

All 3 arms are run strictly serial on GPU0, one LLM epoch, valid AUC/UAUC.
They reuse one exact MF-initialized 3-layer graph artifact.
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
original = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if p and Path(p).resolve() != ROOT]
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + original)

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import night25 as n25

for module in (core, n25):
    if not Path(module.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"NIGHT26 imported wrong worktree: {module.__file__}")

# In particular C1 and C2 differ ONLY in the graph layer aggregation.
ARMS = (
    dict(id="C0_MF_ONLY_FP32", title="Matched FP32 CoRA MF-only baseline",
         graph_anchor_lambda=0.0, graph_aggregation="uniform"),
    dict(id="C1_L3_LAMBDA005", title="Three-layer MF LightGCN, lambda=0.05, uniform",
         graph_anchor_lambda=0.05, graph_aggregation="uniform"),
    dict(id="C2_L3_ADAPTIVE_LAMBDA005", title="Three-layer MF LightGCN, lambda=0.05, learned L0..3",
         graph_anchor_lambda=0.05, graph_aggregation="learned"),
)
IDS = tuple(arm["id"] for arm in ARMS)

BASE = dict(n25.BASE, graph_layers=3, graph_init="mf",
            graph_pretrain_epochs=0, graph_aux_weight=0.0, graph_lr_scale=0.1)

def write_summary(root, records):
    fields = ("id", "auc", "uauc", "delta_vs_g2", "delta_uauc_vs_g2",
              "graph_layers", "graph_aggregation", "graph_anchor_lambda",
              "graph_layer_weights_end", "graph_aux_weight", "graph_lr_scale",
              "amp_summary", "seed", "checkpoint", "graph_artifact")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fields})

def main():
    parser = argparse.ArgumentParser(description="N26 matched CoRA + LightGCN experiments")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--only", choices=IDS)
    parser.add_argument("--start", choices=IDS)
    args = parser.parse_args()
    if args.only and args.start:
        parser.error("--only and --start are mutually exclusive")
    cfg = core.prepared_config()
    arms = list(ARMS)
    if args.only:
        arms = [arm for arm in arms if arm["id"] == args.only]
    elif args.start:
        arms = arms[IDS.index(args.start):]
    print("[N26-PLAN] shared MF-initialized LightGCN raw table, "
          "propagation depth=3, graph BCE=0, FP32 CoRA, CTR=0.05 KD=0.05, "
          "GPU0 seed42 1 LLM epoch each, Valid only, foreground serial",
          flush=True)
    for arm in arms:
        print(f"[N26-ARM] {arm['id']} lambda={arm['graph_anchor_lambda']} "
              f"aggregation={arm['graph_aggregation']}", flush=True)
    if args.plan:
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("NIGHT26 requires CUDA_VISIBLE_DEVICES=0")
    if core.gpu_is_in_use():
        raise RuntimeError("GPU0 already busy; NIGHT26 refuses concurrent runs")
    for test in ("test_night8_cpu.py", "test_night24_cpu.py",
                 "test_night25_cpu.py", "test_night26_cpu.py"):
        subprocess.run([sys.executable, f"experiments/no31_cora_original/{test}"],
                       cwd=str(ROOT), check=True)

    root = ROOT / "logs" / ("NO31_CORA_NIGHT26_"
                           + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(dict(
        arms=arms, seed=42, graph_layers=3, graph_init="mf",
        graph_pretrain_epochs=0, graph_aux_weight=0.0,
        matched_precision="FP32_CoRA", split="valid", train_epochs=1,
        baseline_reference=dict(g2_auc=core.G2_AUC,
                                g2_uauc=core.G2_UAUC)), indent=2))
    print(f"[N26-ROOT] {root}", flush=True)
    # All three start from the identical MF+3-layer graph artifact. This
    # also fixes graph community assignments and the two MF/graph RMS scales.
    shared = ("N26_SHARED_MF_L3", "mf", 3, "Shared untrained MF-initialized graph")
    artifact, train, _ = n25.prepare_graph(shared, cfg, root, random_epochs=500)
    records = []
    for arm in arms:
        ident = arm["id"]
        row = dict(BASE, **arm, collab_proto_path=str(artifact),
                   collab_train_path=str(train))
        try:
            if core.gpu_is_in_use():
                raise RuntimeError("GPU0 occupied before next serial arm")
            result = core.run_one(row, cfg, root)
            result["graph_artifact"] = str(artifact)
            if result["graph_aggregation"] != arm["graph_aggregation"]:
                raise RuntimeError("Wrong graph aggregation recorded")
            if arm["graph_aggregation"] == "learned":
                weights = result["graph_layer_weights_end"]
                if weights is None or len(weights) != 4:
                    raise RuntimeError("N26 failed to save four learned layer weights")
            records.append(result)
            with (root / "results.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(result) + "\n")
            write_summary(root, records)
            print(f"[N26-ARM-DONE] {ident} ValidAUC={result['auc']:.9f} "
                  f"ValidUAUC={result['uauc']:.9f}", flush=True)
        except BaseException as exc:
            (root / "failed.json").write_text(json.dumps(dict(
                id=ident, error=str(exc), completed=[r["id"] for r in records]
            ), indent=2))
            print(f"[N26-ABORT] failed={ident}; later arms not started; "
                  f"previous results preserved at {root}", flush=True)
            raise
    print(f"[N26-COMPLETE] summary={root/'summary.csv'}", flush=True)

if __name__ == "__main__":
    main()
