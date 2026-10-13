#!/usr/bin/env python3
"""A1 inference-only ablation: display/save summary without console log files."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import os
import re
import subprocess
import sys
from collections import deque

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [("A1-Off", 0), ("A1-Original", 1),
               ("A1-Scale2", 2), ("A1-Scale5", 5)]
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PATTERN = re.compile(
    r"\*\*\*auc:\s*(" + NUM + r")\s+\*\*\*uauc:\s*\(\s*(" + NUM + r")"
)


def summarize(rows, ckpt, split):
    base = next((r for r in rows if r["scale"] == 1 and r["auc"] is not None), None)
    lines = [
        "A1 inference ablation | " + split,
        "Checkpoint: " + str(ckpt),
        "",
        "{:<16} {:>6} {:>12} {:>12} {:>14} {:>9}".format(
            "Experiment", "Scale", "AUC", "UAUC", "Delta AUC", "Status"),
        "-" * 78,
    ]
    for r in rows:
        a = "-" if r["auc"] is None else "{:.8f}".format(r["auc"])
        u = "-" if r["uauc"] is None else "{:.8f}".format(r["uauc"])
        d = "-" if r["auc"] is None or base is None else "{:+.8f}".format(
            r["auc"] - base["auc"])
        lines.append("{:<16} {:>6} {:>12} {:>12} {:>14} {:>9}".format(
            r["name"], r["scale"], a, u, d, r["status"]))
    return "\n".join(lines) + "\n"


def save(rows, ckpt, split, stem):
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    txt = out / (stem + ".txt")
    csv_file = out / (stem + ".csv")
    txt.write_text(summarize(rows, ckpt, split), encoding="utf-8")
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        cols = ["name", "scale", "auc", "uauc", "status", "checkpoint"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return txt, csv_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser().resolve()
    else:
        found = list((ROOT / "output" / "S2_CAND_A1").rglob("checkpoint_best.pth"))
        if not found:
            parser.error("No A1 checkpoint found in output/S2_CAND_A1")
        ckpt = max(found, key=lambda p: p.stat().st_mtime)
    if not ckpt.is_file():
        parser.error("Checkpoint not found: " + str(ckpt))

    stem = "A1_ablation_{}_{}".format(
        args.split, datetime.now().strftime("%Y%m%d_%H%M%S"))
    rows = []
    print("Checkpoint:", ckpt, flush=True)
    for name, scale in EXPERIMENTS:
        print("\n[{}] Evaluating {} (scale={})".format(args.split, name, scale), flush=True)
        cmd = [
            sys.executable, "train_collm_mf_din.py",
            "--cfg-path", "tran_configs/S2_candidate_history_A1.yaml",
            "--options", "model.ckpt=" + str(ckpt),
            "model.candidate_hist_eval_multiplier=" + str(scale),
            "run.evaluate=true",
            "run.test_splits=[" + args.split + "]",
            "run.output_dir=" + str(ROOT / "output" / "S2_CAND_A1_ablation" / ("scale_" + str(scale))),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        metrics = None
        tail = deque(maxlen=12)
        with subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1
        ) as proc:
            for line in proc.stdout:
                tail.append(line.rstrip()[:500])
                if "Averaged stats:" in line:
                    match = PATTERN.search(line)
                    if match:
                        metrics = (float(match.group(1)), float(match.group(2)))
            exit_code = proc.wait()

        ok = exit_code == 0 and metrics is not None
        row = dict(name=name, scale=scale,
                   auc=metrics[0] if ok else None,
                   uauc=metrics[1] if ok else None,
                   status="OK" if ok else "FAILED",
                   checkpoint=str(ckpt))
        rows.append(row)
        if ok:
            print("  AUC={:.8f}  UAUC={:.8f}".format(*metrics), flush=True)
        else:
            print("  FAILED (exit code {})".format(exit_code), flush=True)
            print("\n".join(tail), flush=True)
        save(rows, ckpt, args.split, stem)

    txt, csv_file = save(rows, ckpt, args.split, stem)
    print("\n" + "=" * 78 + "\nFINAL SUMMARY\n" + "=" * 78)
    print(summarize(rows, ckpt, args.split), end="")
    print("Saved:", txt)
    print("Saved:", csv_file)
    return 1 if any(r["status"] != "OK" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
