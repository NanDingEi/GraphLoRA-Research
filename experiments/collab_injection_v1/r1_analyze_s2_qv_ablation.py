#!/usr/bin/env python3
"""R1-2: CPU only, LABELS FOR DIAGNOSTICS (never construct a gate).

Consumes R1-1 five-arm native S2 Yes logits on the SAME Valid Small 5200
examples. Computes positive-weighted UAUC, paired AUC rescue/damage
including half-credit ties, paired within-user summary, and train-only
item-popularity strata. Produces one machine-readable JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_s2_layer_complementarity import (
    auc_metrics, paired_complementarity,
)
# Keep CPU analysis independent of the 7B/Vicuna training imports.
# The NPZ stores the exact arm names and Q/V coefficients and these are
# verified against this predeclared protocol (no scores used to select arms).
ARMS = (
    ("R1_A_S2_QV", 0.1, 0.1),
    ("R1_B_OFF", 0.0, 0.0),
    ("R1_C_V_ONLY", 0.0, 0.1),
    ("R1_D_Q_ONLY", 0.1, 0.0),
    ("R1_E_V_HALF", 0.0, 0.05),
)
EXPECTED_S2_AUC = 0.7296752596287361
EXPECTED_S2_UAUC = 0.6841670211619721

def popularity_strata(train, items):
    """Train-only frequency per candidate item (ALL train rows; no labels)."""
    if "iid" not in train.columns:
        raise ValueError("train_ood2.pkl lacks iid column")
    frequency=train["iid"].value_counts()
    if not frequency.index.is_unique:
        raise RuntimeError("Duplicate frequency keys")
    # CDF percentile among unique items with at least one training row;
    # ties share the same rank (no threshold chosen by valid labels).
    pct=frequency.rank(method="average",pct=True)
    counts=frequency.reindex(items).fillna(0).to_numpy(dtype=np.int64)
    percentile=pct.reindex(items).fillna(0).to_numpy(dtype=np.float64)
    strata={
        "unseen_in_train":counts==0,
        "train_pop_q1":(counts>0)&(percentile<=0.25),
        "train_pop_q2":(counts>0)&(percentile>0.25)&(percentile<=0.5),
        "train_pop_q3":(counts>0)&(percentile>0.5)&(percentile<=0.75),
        "train_pop_q4":(counts>0)&(percentile>0.75),
    }
    if not np.array_equal(
        np.stack(list(strata.values())).astype(int).sum(axis=0),
        np.ones(len(items),dtype=int)):
        raise RuntimeError("Candidate item popularity bins not exhaustive")
    return strata,counts,percentile


def analyze(data, train, expected_n=5200):
    names=[a[0] for a in ARMS]
    required={"user","item","label","arm_names","q_scales","v_scales"}|set(names)
    missing=required-set(data)
    if missing:
        raise RuntimeError(f"Missing required prediction archive keys: {sorted(missing)}")
    user=np.asarray(data["user"]).astype(np.int64).ravel()
    item=np.asarray(data["item"]).astype(np.int64).ravel()
    labels=np.asarray(data["label"]).astype(np.int64).ravel()
    if len(user)!=expected_n or len(item)!=expected_n or len(labels)!=expected_n:
        raise RuntimeError("R1 unexpected sample count or alignment")
    if set(np.unique(labels))!={0,1}:
        raise RuntimeError("R1 expected both binary labels")
    if list(data["arm_names"])!=names or any(
        abs(float(data["q_scales"][i])-q)>1e-12 or
        abs(float(data["v_scales"][i])-v)>1e-12
        for i,(_,q,v) in enumerate(ARMS)):
        raise RuntimeError("The R1 file does not match predeclared five arms")
    scores={}
    metrics={}
    for name in names:
        x=np.asarray(data[name],dtype=np.float64).ravel()
        if len(x)!=expected_n or not np.isfinite(x).all():
            raise RuntimeError(f"{name} has invalid Yes score")
        scores[name]=x
        metrics[name]=auc_metrics(user,labels,x)
    base=metrics["R1_A_S2_QV"]
    if (abs(base["auc"]-EXPECTED_S2_AUC)>2e-5 or
            abs(base["uauc"]-EXPECTED_S2_UAUC)>2e-5):
        raise RuntimeError("Historical S2 AUC/UAUC anchor failed")
    for name in names:
        metrics[name]["delta_auc_vs_S2"]=metrics[name]["auc"]-base["auc"]
        metrics[name]["delta_uauc_vs_S2"]=metrics[name]["uauc"]-base["uauc"]
    pair_results={}
    # Compare original S2 ON against OFF and the three path/intensity
    # ablations: all pair-counts use ON as the *baseline*.
    for name in names[1:]:
        report=paired_complementarity(
            user,labels,scores["R1_A_S2_QV"],scores[name],chunk=128)
        if (abs(report["global_auc_pairs"]["net_delta_auc"]-
                metrics[name]["delta_auc_vs_S2"])>1e-10 or
            abs(report["positive_weighted_uauc"]["delta_uauc"]-
                metrics[name]["delta_uauc_vs_S2"])>1e-10):
            raise RuntimeError(f"{name} pair ledger does not match AUC/UAUC")
        pair_results[name]=report

    strata,counts,percentiles=popularity_strata(train,item)
    pop_groups=[]
    for name,mask in strata.items():
        n=int(mask.sum())
        entry={
            "group":name,"n_samples":n,
            "n_positive":int(np.sum(labels[mask]==1)),
            "n_negative":int(np.sum(labels[mask]==0)),
            "mean_train_item_count":float(counts[mask].mean()) if n else None,
        }
        entry["metrics"]={
            arm:auc_metrics(user[mask],labels[mask],scores[arm][mask])
            if n>0 and len(set(labels[mask]))==2 else {
                "n":n,"auc":None,"uauc":None,"eligible_users":0}
            for arm in names
        }
        a=entry["metrics"]["R1_A_S2_QV"]
        b=entry["metrics"]["R1_B_OFF"]
        entry["S2_minus_OFF_auc"]=a["auc"]-b["auc"] \
            if a["auc"] is not None and b["auc"] is not None else None
        entry["S2_minus_OFF_uauc"]=a["uauc"]-b["uauc"] \
            if a["uauc"] is not None and b["uauc"] is not None else None
        pop_groups.append(entry)

    # ON/OFF Yes-logit differences: diagnostic score shifts by true label.
    delta=scores["R1_A_S2_QV"]-scores["R1_B_OFF"]
    score_effect={
        "mean_signed_logit_shift_ON_minus_OFF":float(delta.mean()),
        "mean_abs_logit_shift":float(np.abs(delta).mean()),
        "fraction_nonzero_shifts":float(np.count_nonzero(delta)/len(delta)),
        "mean_logit_shift_when_label_1":float(delta[labels==1].mean()),
        "mean_logit_shift_when_label_0":float(delta[labels==0].mean()),
        "pearson_ON_OFF":float(np.corrcoef(
            scores["R1_A_S2_QV"],scores["R1_B_OFF"])[0,1]),
    }
    return {
        "status":"complete","dataset_split":"valid",
        "n_samples":expected_n,
        "metrics":metrics,
        "ON_baseline_paired_comparisons":pair_results,
        "train_item_frequency_groups":pop_groups,
        "ON_OFF_score_effect":score_effect,
        "disclaimers":[
            "No parameters were trained: OFF ablates trained S2 injection",
            "Train-only item frequency counts every train row regardless of label",
            "Group AUC/UAUC on filtered subsets is not additive to global AUC",
            "Pooled within-user pair proportions are not positive-weighted UAUC",
            "Labels are used for descriptive diagnosis, not to select an R2 gate",
            "Do not claim causal effect of GNN data itself from injection OFF",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",
        default="logs/collab_v1/S2_R1_QV_ABLATION.npz")
    p.add_argument("--manifest",
        default="logs/collab_v1/S2_R1_QV_ABLATION.json")
    p.add_argument("--train-pkl",default=None,
        help="Exact train_ood2.pkl; if omitted derive from the R1 model YAML")
    p.add_argument("--out",
        default="logs/collab_v1/S2_R1_QV_ANALYSIS.json")
    p.add_argument("--expected-samples",type=int,default=5200)
    args=p.parse_args()
    pred=Path(args.predictions).resolve()
    manifest=Path(args.manifest).resolve()
    out=Path(args.out).resolve()
    for path in (pred,manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing R1 report: {out}")
    manifest_data=json.loads(manifest.read_text(encoding="utf-8"))
    if (manifest_data.get("status")!="complete" or
        Path(manifest_data.get("prediction_archive","")).resolve()!=pred):
        raise RuntimeError("Prediction archive does not match completed R1 manifest")
    if args.train_pkl is None:
        import yaml
        source=Path(manifest_data["config"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing R1 source config: {source}")
        cfg=yaml.safe_load(source.read_text(encoding="utf-8"))
        datasets=cfg.get("datasets",{})
        ds=datasets.get("movie_ood",datasets.get("amazon_ood",{}))
        train_path=Path(str(ds.get("path","")))/"train_ood2.pkl"
    else:
        train_path=Path(args.train_pkl).expanduser()
    train_path=train_path.resolve()
    if not train_path.is_file():
        raise FileNotFoundError(
            f"train_ood2.pkl not found: {train_path}; "
            "pass --train-pkl with the actual OOD2 training table")
    with np.load(pred,allow_pickle=False) as npz:
        data={key:npz[key] for key in npz.files}
    train=pd.read_pickle(train_path)
    report=analyze(data,train,expected_n=args.expected_samples)
    for name,metrics in report["metrics"].items():
        print(f"[R1-SCORE] {name:16s} "
              f"AUC={metrics['auc']:.9f} UAUC={metrics['uauc']:.9f} "
              f"dAUC={metrics['delta_auc_vs_S2']:+.9f} "
              f"dUAUC={metrics['delta_uauc_vs_S2']:+.9f}",flush=True)
    for name,diag in report["ON_baseline_paired_comparisons"].items():
        g=diag["global_auc_pairs"]
        u=diag["positive_weighted_uauc"]
        print(f"[R1-PAIRS] ON_vs_{name} "
              f"ON_to_alt_rescue={g['strict_rescue']} "
              f"ON_to_alt_damage={g['strict_damage']} "
              f"pairs={g['pairs']} delta_AUC={g['net_delta_auc']:+.9f} "
              f"delta_UAUC={u['delta_uauc']:+.9f}",flush=True)
    for g in report["train_item_frequency_groups"]:
        print(f"[R1-POP] {g['group']} n={g['n_samples']} "
              f"S2_minus_OFF_AUC={g['S2_minus_OFF_auc']} "
              f"UAUC={g['S2_minus_OFF_uauc']}",flush=True)
    report["prediction_archive"]=str(pred)
    report["r1_manifest"]=str(manifest)
    report["train_data"]=str(train_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),
                   encoding="utf-8")
    print(f"[R1-ANALYSIS-COMPLETE] CPU only; {out}",flush=True)


if __name__=="__main__":
    main()
