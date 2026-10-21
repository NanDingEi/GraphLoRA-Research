#!/usr/bin/env python3
"""R2-01 CPU-only native Yes-logit ON/OFF perturbation audit on R1 NPZ."""
import argparse
import json
from pathlib import Path
import numpy as np


def numbers(x):
    x=np.asarray(x,dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Invalid R2-01 numeric data")
    return {"mean":float(x.mean()),"p50":float(np.quantile(x,.5)),
            "p90":float(np.quantile(x,.9)),"p99":float(np.quantile(x,.99)),
            "max":float(x.max())}


def audit(data):
    n=len(data["user"])
    if n!=5200:
        raise ValueError(f"R2-01 expects historical 5200 examples, got {n}")
    on=np.asarray(data["R1_A_S2_QV"],dtype=np.float64).reshape(-1)
    off=np.asarray(data["R1_B_OFF"],dtype=np.float64).reshape(-1)
    y=np.asarray(data["label"],dtype=np.int64).reshape(-1)
    if on.shape!=off.shape or len(y)!=n or np.unique(y).tolist()!=[0,1]:
        raise RuntimeError("Prediction/sample archive mismatch")
    dif=on-off
    pos=y==1
    neg=y==0
    report={
        "sample_count":n,"absolute_ON_OFF_shift":numbers(np.abs(dif)),
        "signed_shift_for_positive":numbers(dif[pos]),
        "signed_shift_for_negative":numbers(dif[neg]),
        "fraction_unchanged_fp32":float((dif==0).mean()),
        "fraction_abs_shift_lt_1e_4":float((np.abs(dif)<1e-4).mean()),
        "corr_ON_OFF":float(np.corrcoef(on,off)[0,1]),
        "ON_scores":numbers(on),"OFF_scores":numbers(off),
        "note":"This probes inference score sensitivity; no labels used for gates.",
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="logs/collab_v1/S2_R1_QV_ABLATION.npz")
    p.add_argument("--output",default="logs/collab_v1/R2_01_SCORE_AUDIT.json")
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    with np.load(a.input,allow_pickle=False) as f:
        report=audit(f)
    report["input"]=str(Path(a.input).resolve())
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("[R2-01]",json.dumps(report,ensure_ascii=False),flush=True)
    print("[R2-01-COMPLETE]",out,flush=True)


if __name__=="__main__":
    main()
