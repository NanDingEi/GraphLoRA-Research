#!/usr/bin/env python3
"""Diagnose A2/A3 collapse without retraining. AUC/UAUC on VALID only.

Compare identical trained checkpoints with gate intensity 0, 0.25, 1.
Gate 0 means 1 + 0*(gate-1) = identity (not zeroing GNN).
Saves compact out/ report; does not save evaluation console logs.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = [
    ("A2", "A2_HIST_GATE", "tran_configs/S2_candidate_history_A2.yaml"),
    ("A3", "A3_LEARNED_ATTN_GATE", "tran_configs/S2_candidate_history_A3.yaml"),
]
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RESULT = re.compile(r"\*\*\*auc:\s*(" + NUM + r")\s+\*\*\*uauc:\s*\(\s*(" + NUM + r")")
CORR = re.compile(r"\[cand\]\s+mode=gate\s+step=(\d+)\s+active=(\d+)/(\d+)\s+correction_norm=(" + NUM + r")")


def find_checkpoint(label, specified):
    if specified:
        path = Path(specified).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path
    paths = list((ROOT / "output").glob(label + "_*/**/checkpoint_best.pth"))
    if not paths:
        raise FileNotFoundError("No checkpoint for {} under output/; pass --a2-ckpt / --a3-ckpt".format(label))
    return max(paths, key=lambda p: p.stat().st_mtime)


def get_valid_epochs(ckpt):
    filename = ckpt.parent / "valid_log.txt"
    rows = []
    if not filename.is_file():
        return rows
    for line in filename.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
            auc = data.get("valid_agg_metrics")
            if auc is None:
                continue
            u = data.get("valid_uauc")
            uauc = u[0] if isinstance(u, (list, tuple)) and u else None
            rows.append((float(auc), None if uauc is None else float(uauc)))
        except (ValueError, TypeError):
            continue
    return rows


def get_correction_norm(label, ckpt):
    # Match the same experiment stamp to avoid accidentally reading old logs.
    run_name = ckpt.parent.parent.name
    tag = run_name[len(label) + 1:] if run_name.startswith(label + "_") else ""
    log = ROOT / "log" / (label + "_train_" + tag + ".log")
    if not log.is_file():
        hits = sorted((ROOT / "log").glob(label + "_train_*.log"))
        log = hits[-1] if hits else None
    matches = []
    if log and log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            found = CORR.search(line)
            if found:
                matches.append((int(found.group(1)),
                                int(found.group(2)), int(found.group(3)),
                                float(found.group(4))))
    return log, matches


def evaluate(name, config, ckpt, scale, gpu, stamp):
    out_dir = ROOT / "output" / "A2_A3_gate_diag" / stamp / name / ("scale_" + str(scale))
    cmd = [
        sys.executable, "-u", "train_collm_mf_din.py",
        "--cfg-path", config, "--options",
        "model.ckpt=" + str(ckpt),
        "model.candidate_hist_eval_multiplier=" + str(scale),
        "run.evaluate=true", "run.test_splits=[valid]",
        "run.output_dir=" + str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    results = None
    tail = deque(maxlen=12)
    with subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
        for line in proc.stdout:
            tail.append(line.rstrip()[:350])
            if "Averaged stats:" in line:
                found = RESULT.search(line)
                if found:
                    results = float(found.group(1)), float(found.group(2))
        ret = proc.wait()
    if ret != 0 or results is None:
        raise RuntimeError("{} scale={} evaluation failed (exit={}):\n{}".format(
            name, scale, ret, "\n".join(tail)))
    return results


def write_report(rows, stamp, annotations):
    folder = ROOT / "out"
    folder.mkdir(parents=True, exist_ok=True)
    stem = "A2_A3_gate_diagnosis_" + stamp
    ptxt = folder / (stem + ".txt")
    pcsv = folder / (stem + ".csv")
    ref = {r["experiment"]: r for r in rows if r["scale"] == 0 and r["status"] == "OK"}
    lines = [
        "A2/A3 post-training gate diagnosis (VALID only; no retraining)",
        "Scale 0: identity gate; 0.25: quarter strength; 1: original trained gate",
        "",
        "{:<9} {:>7} {:>12} {:>12} {:>15} {:>9}".format(
            "Experiment", "Scale", "AUC", "UAUC", "AUC vs Off", "Status"),
        "-" * 75,
    ]
    for r in rows:
        off = ref.get(r["experiment"])
        diff = None if (off is None or r["auc"] is None) else r["auc"] - off["auc"]
        r["auc_vs_off"] = diff
        fmt = lambda x: "-" if x is None else "{:.8f}".format(x)
        lines.append("{:<9} {:>7} {:>12} {:>12} {:>15} {:>9}".format(
            r["experiment"], r["scale"], fmt(r["auc"]), fmt(r["uauc"]),
            "-" if diff is None else "{:+.8f}".format(diff), r["status"]))
    lines += ["", "TRAINING DIAGNOSTICS"]
    lines += annotations
    ptxt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cols = ["experiment", "scale", "auc", "uauc", "auc_vs_off",
            "status", "checkpoint", "error"]
    with pcsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return ptxt, pcsv, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--a2-ckpt", default=None)
    parser.add_argument("--a3-ckpt", default=None)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, annotations = [], []
    for name, label, config in CASES:
        ckpt = find_checkpoint(label, args.a2_ckpt if name == "A2" else args.a3_ckpt)
        val = get_valid_epochs(ckpt)
        training_log, corrections = get_correction_norm(label, ckpt)
        annotations.append("{} checkpoint: {}".format(name, ckpt))
        annotations.append("  Logged Valid epochs: " + (
            "; ".join("ep{} AUC={:.8f} UAUC={}".format(
                i, values[0], values[1]) for i, values in enumerate(val))
            if val else "not available"))
        annotations.append("  Training log: {}".format(training_log))
        if corrections:
            first, last = corrections[0], corrections[-1]
            annotations.append(
                "  Gate correction norm: first(step {}) {:.6f}; last(step {}) {:.6f}; max {:.6f}".format(
                    first[0], first[3], last[0], last[3],
                    max(x[3] for x in corrections)))
            annotations.append("  Gate active last batch: {}/{}".format(last[1], last[2]))
        else:
            annotations.append("  No [cand] correction_norm lines found in training log")
        for scale in (0, 0.25, 1):
            print("{} gate multiplier={} | checkpoint={}".format(name, scale, ckpt), flush=True)
            entry = {
                "experiment": name, "scale": scale, "auc": None, "uauc": None,
                "auc_vs_off": None, "status": "FAILED", "checkpoint": str(ckpt), "error": "",
            }
            rows.append(entry)
            try:
                entry["auc"], entry["uauc"] = evaluate(name, config, ckpt, scale, args.gpu, stamp)
                entry["status"] = "OK"
                print("  AUC={:.8f}  UAUC={:.8f}".format(entry["auc"], entry["uauc"]), flush=True)
            except Exception as exc:
                entry["error"] = str(exc)
                print("  Error: {}".format(str(exc)[:700]), flush=True)
            write_report(rows, stamp, annotations)
    txt, csvpath, result = write_report(rows, stamp, annotations)
    print("\nFINAL GATE DIAGNOSIS\n" + result)
    print("Saved:", txt, "\nSaved:", csvpath)
    return 1 if any(r["status"] != "OK" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
