#!/usr/bin/env python3
"""Seven NEW LightGCN+CoRA experiments. Same NIGHT26 worktree; foreground serial.

E1/E2: change fixed fusion strength.
E3/E4: learned separate user/item lambdas vs training-degree conditioned gates.
E5/E6: shallow-prior fixed hop weighting vs adaptive hop weighting.
E7: 3-hop graph coarse-to-fine Group8 QFormer conditioning.

No legacy baseline reruns. Single shared MF-L3 graph artifact, seed42.
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
prior_path = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
              if p and Path(p).resolve() != ROOT]
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + prior_path)

from experiments.no31_cora_original import night8 as core
from experiments.no31_cora_original import night25 as n25

# The seven experiments use NO31/CoRA existing settings exactly and modify
# only the named graph-side variables.
COMMON = dict(n25.BASE,
              graph_layers=3, graph_init="mf", graph_pretrain_epochs=0,
              graph_aux_weight=0.0, graph_lr_scale=0.1,
              graph_aggregation="uniform", graph_fusion="fixed",
              graph_mode="graph_base", graph_joint=True,
              amp_init_scale=256.0)
ARMS = (
    dict(id="E1_L3_LAMBDA0025", title="MF 3-hop uniform LightGCN, lambda .025",
         graph_anchor_lambda=0.025),
    dict(id="E2_L3_LAMBDA0200", title="MF 3-hop uniform LightGCN, lambda .2",
         graph_anchor_lambda=0.2),
    dict(id="E3_DUAL_UI_GATE", title="Separate learned user/item fusion lambdas, init .1",
         graph_anchor_lambda=0.1, graph_fusion="dual"),
    dict(id="E4_DEGREE_UI_GATE", title="Degree-aware learned user/item fusion, init .1",
         graph_anchor_lambda=0.1, graph_fusion="degree"),
    dict(id="E5_SHALLOW_FIXED", title="Fixed shallow-prior hops [.4,.3,.2,.1], lambda .1",
         graph_anchor_lambda=0.1, graph_aggregation="shallow_fixed"),
    dict(id="E6_SHALLOW_LEARNED", title="Learn shallow-prior hops initialized [.4,.3,.2,.1]",
         graph_anchor_lambda=0.1, graph_aggregation="shallow_learned"),
    dict(id="E7_GRAPH_C2F_GROUP8", title="Graph K16 coarse/fine Group8 schedule, lambda .1",
         graph_anchor_lambda=0.1, graph_mode="graph_c2f"),
)
IDS = tuple(a["id"] for a in ARMS)

def find_artifact(optional_path):
    if optional_path is not None:
        path = optional_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    locations = sorted((ROOT / "logs").glob(
        "NO31_CORA_NIGHT26_*/artifacts/N26_SHARED_MF_L3.npz"
    ))
    return locations[-1].resolve() if locations else None

def verify_artifact(path, cfg):
    import numpy as np
    from experiments.no31_cora_original.graph_cf_pretrain import load_mf_init_table
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        raise FileNotFoundError("Graph artifact manifest missing: " + str(manifest))
    m = json.loads(manifest.read_text())
    if not (m.get("mf_initialization") is True and m.get("layers") == 3
            and m.get("epochs") == 0 and m.get("seed") == 42):
        raise RuntimeError("Require MF-initialized L3 graph artifact without BPR pretrain")
    ckpt = Path(cfg["model"]["rec_config"]["pretrained_path"]).expanduser().resolve()
    mf = load_mf_init_table(
        ckpt, int(cfg["model"]["rec_config"]["user_num"]),
        int(cfg["model"]["rec_config"]["item_num"]), 256).numpy()
    with np.load(path, allow_pickle=False) as arr:
        if not np.array_equal(mf, arr["graph_id_table"]):
            raise RuntimeError("Graph ID table differs from current MF checkpoint")
    print(f"[N27-ARTIFACT-PARITY-PASS] {path}", flush=True)

def save_summary(root, results):
    fields = (
        "id", "auc", "uauc", "delta_vs_g2", "delta_uauc_vs_g2",
        "graph_anchor_lambda", "graph_fusion", "graph_gate_end",
        "graph_layers", "graph_aggregation", "graph_layer_weights_end",
        "graph_mode", "graph_head_end", "graph_aux_weight", "graph_lr_scale",
        "amp_summary", "seed", "log", "checkpoint", "graph_artifact"
    )
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for rec in results:
            writer.writerow({key: rec.get(key) for key in fields})

def main():
    parser = argparse.ArgumentParser(description="Seven NEW graph+CoRA trials in NIGHT26 project")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--only", choices=IDS)
    parser.add_argument("--start", choices=IDS)
    parser.add_argument("--graph-artifact", type=Path, default=None)
    args = parser.parse_args()
    if args.only and args.start:
        parser.error("--only and --start cannot both be used")
    cfg = core.prepared_config()
    rows = list(ARMS)
    if args.only:
        rows = [next(a for a in rows if a["id"] == args.only)]
    elif args.start:
        rows = list(ARMS)[IDS.index(args.start):]
    print("[N27-NEW7-PLAN] 7 NEW methods; no baseline reruns; "
          "fresh Vicuna+MF seeds=42 each; train=1 epoch valid only; "
          "LightGCN MF init 3 hops; CTR=.05 KD=.05 graphBCE=0; "
          "GPU0 foreground strict serial", flush=True)
    for idx, arm in enumerate(rows):
        row = dict(COMMON, **arm)
        print(f"[N27-NEW7-ARM] {idx+1}/{len(rows)} {row['id']} "
              f"lambda={row['graph_anchor_lambda']:.4g} "
              f"fusion={row['graph_fusion']} aggregation={row['graph_aggregation']} "
              f"mode={row['graph_mode']}", flush=True)
    graph_artifact = find_artifact(args.graph_artifact)
    if graph_artifact is not None:
        verify_artifact(graph_artifact, cfg)
    if args.plan:
        print(f"[N27-NEW7-ARTIFACT] reuse={graph_artifact or 'generate once'}", flush=True)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Must set CUDA_VISIBLE_DEVICES=0")
    if core.gpu_is_in_use():
        raise RuntimeError("GPU0 already occupied; do NOT run concurrent experiments")
    for test in ("test_night8_cpu.py", "test_night24_cpu.py",
                 "test_night25_cpu.py", "test_night26_cpu.py",
                 "test_night26_new7_cpu.py"):
        subprocess.run([sys.executable, "experiments/no31_cora_original/" + test],
                       cwd=str(ROOT), check=True)
    root = ROOT / "logs" / ("NO31_CORA_NIGHT26_NEW7_"
                           + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    root.mkdir(parents=True, exist_ok=False)
    if graph_artifact is None:
        graph_artifact, train_file, _ = n25.prepare_graph(
            ("N27_SHARED_MF_L3", "mf", 3, "one shared MF-initialized L3 graph"),
            cfg, root, random_epochs=500)
    else:
        train_file = (
            Path(cfg["datasets"]["amazon_ood"]["build_info"]["storage"])
            / cfg["datasets"]["amazon_ood"]["train_file"]
        ).resolve()
    verify_artifact(graph_artifact, cfg)
    (root / "plan.json").write_text(json.dumps(dict(
        experiments=rows, seed=42, graph_artifact=str(graph_artifact),
        trained_epoch_per_arm=1, dataset="ml-1m OOD2", split="valid",
        historical_reference=dict(
            G2_AUC=core.G2_AUC, G2_UAUC=core.G2_UAUC,
            M2_AUC=0.7269020041628543, M2_UAUC=0.6849193742344957,
        ),
    ), indent=2))
    records = []
    for index, arm in enumerate(rows, start=1):
        row = dict(COMMON, **arm,
                   collab_proto_path=str(graph_artifact),
                   collab_train_path=str(train_file))
        try:
            if core.gpu_is_in_use():
                raise RuntimeError("GPU0 busy before next serial arm")
            result = core.run_one(row, cfg, root)
            result["graph_artifact"] = str(graph_artifact)
            if result["graph_fusion"] != row["graph_fusion"]:
                raise RuntimeError("Wrong fusion mode saved")
            if row["graph_fusion"] != "fixed" and not result["graph_gate_end"]:
                raise RuntimeError("Learned graph gate summary missing")
            if row["graph_aggregation"] == "shallow_learned":
                if not result["graph_layer_weights_end"]:
                    raise RuntimeError("Learned shallow hop weights missing")
            if row["graph_mode"] == "graph_c2f" and not result["graph_head_end"]:
                raise RuntimeError("Coarse-to-fine update report missing")
            records.append(result)
            with (root / "results.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(result) + "\n")
            save_summary(root, records)
            print(f"[N27-NEW7-DONE] {index}/{len(rows)} id={row['id']} "
                  f"AUC={result['auc']:.9f} UAUC={result['uauc']:.9f}",
                  flush=True)
        except BaseException as exc:
            (root / "failed.json").write_text(json.dumps(dict(
                id=row["id"], error=str(exc),
                completed=[x["id"] for x in records]), indent=2))
            print(f"[N27-NEW7-ABORT] failed={row['id']} remaining_not_run=True "
                  f"results={root / 'summary.csv'}", flush=True)
            raise
    print(f"[N27-NEW7-COMPLETE] {len(records)} experiments; "
          f"summary={root / 'summary.csv'}", flush=True)

if __name__ == "__main__":
    main()
