#!/usr/bin/env python3
"""Run A2 then A3 sequentially, each from scratch under the same S2 recipe.

Train logs: log/A2_*.log, log/A3_*.log
Artifacts/checkpoints: output/A2_*/ and output/A3_*/
Compact metrics: out/A2_A3_serial_*.txt and .csv
Testing prints metrics only; it does not save a full test console log.
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
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
METRICS = re.compile(
    r"\*\*\*auc:\s*(" + NUMBER + r")\s+\*\*\*uauc:\s*\(\s*(" + NUMBER + r")"
)
EXPERIMENTS = [
    ("A2", "tran_configs/S2_candidate_history_A2.yaml", "A2_HIST_GATE"),
    ("A3", "tran_configs/S2_candidate_history_A3.yaml", "A3_LEARNED_ATTN_GATE"),
]


def get_best_valid(run_dir):
    runs = list(run_dir.rglob("checkpoint_best.pth"))
    if not runs:
        raise RuntimeError("No checkpoint_best.pth found under {}".format(run_dir))
    ckpt = max(runs, key=lambda p: p.stat().st_mtime)
    valid_log = ckpt.parent / "valid_log.txt"
    if not valid_log.is_file():
        raise RuntimeError("Missing valid_log.txt alongside " + str(ckpt))
    vals = []
    for line in valid_log.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "valid_agg_metrics" not in row:
            continue
        uauc = row.get("valid_uauc")
        score = uauc[0] if isinstance(uauc, (tuple, list)) and uauc else None
        vals.append((float(row["valid_agg_metrics"]),
                     float(score) if score is not None else None,
                     row.get("valid_best_epoch", "")))
    if not vals:
        raise RuntimeError("No valid_agg_metrics found in " + str(valid_log))
    best = max(vals, key=lambda v: v[0])
    return ckpt, best


def call_training(config, output_root, log_file, gpu):
    cmd = [
        sys.executable, "-u", "train_collm_mf_din.py",
        "--cfg-path", config,
        "--options", "run.output_dir=" + str(output_root),
        "model.ckpt=null", "run.resume_ckpt_path=null", "run.evaluate=false",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    latest_lines = deque(maxlen=20)
    with log_file.open("w", encoding="utf-8") as file:
        with subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
        ) as process:
            for line in process.stdout:
                file.write(line)
                file.flush()
                latest_lines.append(line.strip())
                if ("Start training epoch" in line or "Saving checkpoint at epoch" in line
                        or "[cand] mode=" in line or "Traceback" in line):
                    print("  " + line.strip()[:250], flush=True)
                if "Averaged stats:" in line:
                    match = METRICS.search(line)
                    if match:
                        print("  VALID: AUC={} UAUC={}".format(
                            match.group(1), match.group(2)), flush=True)
            return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("Training failed with code {}. Tail:\n{}".format(
            return_code, "\n".join(latest_lines)))
    return return_code


def call_test(config, checkpoint, output_root, gpu):
    cmd = [
        sys.executable, "-u", "train_collm_mf_din.py",
        "--cfg-path", config,
        "--options", "model.ckpt=" + str(checkpoint),
        "run.evaluate=true", "run.test_splits=[test]",
        "run.output_dir=" + str(output_root),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    metrics = None
    latest_lines = deque(maxlen=16)
    with subprocess.Popen(
        cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
    ) as process:
        for line in process.stdout:
            latest_lines.append(line.strip()[:400])
            match = METRICS.search(line) if "Averaged stats:" in line else None
            if match:
                metrics = (float(match.group(1)), float(match.group(2)))
        return_code = process.wait()
    if return_code != 0 or metrics is None:
        raise RuntimeError("Test failed or lacks metrics, code {}. Tail:\n{}".format(
            return_code, "\n".join(latest_lines)))
    return metrics


def report(rows, report_name):
    (ROOT / "out").mkdir(parents=True, exist_ok=True)
    path = ROOT / "out" / (report_name + ".txt")
    csv_path = ROOT / "out" / (report_name + ".csv")
    lines = [
        "S2-matched sequential experiments: A2 -> A3",
        "Experiment   Valid AUC    Valid UAUC    Test AUC     Test UAUC   Status",
        "-" * 74,
    ]
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return "-" if value is None else "{:.8f}".format(value)
        lines.append("{:<12} {:<12} {:<13} {:<12} {:<12} {}".format(
            row["experiment"], fmt("valid_auc"), fmt("valid_uauc"),
            fmt("test_auc"), fmt("test_uauc"), row["status"]))
        lines.append("  checkpoint: " + str(row.get("checkpoint", "-")))
        lines.append("  train log:  " + str(row.get("train_log", "-")))
        if row.get("error"):
            lines.append("  error: " + row["error"].replace("\n", " | ")[:800])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ("experiment", "valid_auc", "valid_uauc", "test_auc",
              "test_uauc", "status", "checkpoint", "train_log", "error")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path, csv_path, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip test-set evaluation, collect Valid only")
    args = parser.parse_args()
    print("Preflight: CPU unit tests (A1/A2/A3 fusion, masks, gradients)", flush=True)
    subprocess.run(
        [sys.executable, "-m", "unittest",
         "tests/test_candidate_history_fusion.py", "-v"],
        cwd=str(ROOT), check=True,
    )
    (ROOT / "log").mkdir(parents=True, exist_ok=True)
    (ROOT / "out").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = "A2_A3_serial_" + stamp
    rows = []
    for name, config, label in EXPERIMENTS:
        log_path = ROOT / "log" / (label + "_train_" + stamp + ".log")
        output_root = ROOT / "output" / (label + "_" + stamp)
        row = dict(experiment=name, valid_auc=None, valid_uauc=None,
                   test_auc=None, test_uauc=None, status="RUNNING",
                   checkpoint="", train_log=str(log_path), error="")
        rows.append(row)
        print("\n" + "=" * 76, flush=True)
        print("{} training, independent fresh initialization".format(name), flush=True)
        print("Train log: {}".format(log_path), flush=True)
        try:
            call_training(config, output_root, log_path, args.gpu)
            ckpt, best = get_best_valid(output_root)
            row["checkpoint"] = str(ckpt)
            row["valid_auc"], row["valid_uauc"] = best[:2]
            row["status"] = "TRAINED"
            print("{} best Valid AUC={:.8f} UAUC={:.8f} checkpoint={}".format(
                name, best[0], best[1], ckpt), flush=True)

            if not args.train_only:
                print("{} evaluating selected best checkpoint on Test...".format(name),
                      flush=True)
                test_metrics = call_test(
                    config, ckpt, ROOT / "output" / (label + "_test_" + stamp),
                    args.gpu,
                )
                row["test_auc"], row["test_uauc"] = test_metrics
                print("{} Test AUC={:.8f} UAUC={:.8f}".format(
                    name, *test_metrics), flush=True)
            row["status"] = "OK"
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = str(exc)
            print("{} FAILED: {}".format(name, str(exc)[:1000]), flush=True)
        # Always persist partial report before starting the next experiment.
        report(rows, report_name)

    path, csv_path, summary = report(rows, report_name)
    print("\n" + "=" * 76 + "\nFINAL A2/A3 SUMMARY\n" + "=" * 76)
    print(summary)
    print("TXT saved:", path)
    print("CSV saved:", csv_path)
    return 1 if any(row["status"] == "FAILED" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
