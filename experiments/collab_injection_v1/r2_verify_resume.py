#!/usr/bin/env python3
"""Check whether an R2 round really completed, without starting GPU work.

Exit 0: completed AND basic artifact consistency verified.
Exit 3: never completed; safe for the launcher to start.
Exit 2: partial/corrupt/mismatched; abort instead of overwriting data.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PROBES={"02":("audit",),"03":("off","decision_q"),
        "04":("off","history_k"),
        "05":("off","relation_v","relation_shuffle")}


def verify(round_id):
    root=ROOT/"logs/collab_v1"
    if round_id=="01":
        f=root/"R2_01_SCORE_AUDIT.json"
        if not f.exists():
            return False
        j=json.loads(f.read_text(encoding="utf-8"))
        if j.get("sample_count")!=5200 or (
                j.get("absolute_ON_OFF_shift",{}).get("p90") is None):
            raise RuntimeError("R2-01 existing JSON is not a valid completed audit")
        return True
    if round_id in PROBES:
        file=root/f"R2_{round_id}_PROBE.json"
        archive=file.with_suffix(".npz")
        if not file.exists() and not archive.exists():
            return False
        if not file.is_file() or not archive.is_file():
            raise RuntimeError(f"Partial R2-{round_id} probe results: {file}, {archive}")
        j=json.loads(file.read_text(encoding="utf-8"))
        if (j.get("status")!="complete" or j.get("round")!=round_id or
                j.get("n_samples")!=512 or
                tuple(j.get("variants",()))!=PROBES[round_id] or
                Path(j["prediction_archive"]).resolve()!=archive.resolve()):
            raise RuntimeError(f"Invalid R2-{round_id} probe manifest")
        with np.load(archive,allow_pickle=False) as p:
            for key in ("user","item","label",*PROBES[round_id]):
                if key not in p or len(p[key])!=512:
                    raise RuntimeError(f"Invalid R2-{round_id} archive field {key}")
        return True
    if round_id in ("06","07","08"):
        run=root/"R2_EIGHT_20261022"/f"R2_{round_id}"
        manifest=run/"manifest.json"
        if not run.exists():
            return False
        if not manifest.is_file():
            raise RuntimeError(f"R2-{round_id} directory exists without manifest")
        j=json.loads(manifest.read_text(encoding="utf-8"))
        if j.get("status")!="complete":
            raise RuntimeError(
                f"R2-{round_id} already started status={j.get('status')} "
                "and cannot be overwritten; investigate before resuming")
        if j.get("round")!=round_id or j.get("mode") not in (
                "relation_v","relation_v_q","relation_v_raw","relation_shuffle"):
            raise RuntimeError(f"R2-{round_id} completed manifest invalid")
        ckpt=Path(j.get("best_checkpoint",""))
        if not ckpt.is_file() or ckpt.name!="checkpoint_best.pth":
            raise RuntimeError(f"R2-{round_id} missing best checkpoint {ckpt}")
        if not all(isinstance(j.get(k),(int,float)) for k in ("auc","uauc")):
            raise RuntimeError(f"R2-{round_id} completed manifest lacks AUC/UAUC")
        return True
    raise ValueError(round_id)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--round",required=True,choices=(
        "01","02","03","04","05","06","07","08"))
    args=p.parse_args()
    try:
        completed=verify(args.round)
    except Exception as e:
        print(f"[R2-RESUME-STOP] round={args.round}: {e}",flush=True)
        sys.exit(2)
    if completed:
        print(f"[R2-RESUME-SKIP] round={args.round} verified complete",flush=True)
        return
    print(f"[R2-RESUME-RUN] round={args.round} no completed archive found",
          flush=True)
    sys.exit(3)


if __name__=="__main__":
    main()
