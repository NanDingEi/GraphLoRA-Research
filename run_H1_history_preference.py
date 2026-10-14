#!/usr/bin/env python3
"""H1 historical collaborative preference BCE: foreground training and test.

GraphLoRA/S2 binary CTR labels, original Yes-logit scoring. Extra history-
candidate GNN relation supervision is training-only, never injected at eval.
Train log in log/, model in output/, compact final metrics in out/.
No nohup/background process and no verbose test console log.
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
CONFIG = "tran_configs/S2_H1_history_preference.yaml"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
AUC_RE = re.compile(r"\*\*\*auc:\s*(" + NUMBER + r")\s+\*\*\*uauc:\s*\(\s*(" + NUMBER + r")")
LOSS_RE = re.compile(r"Averaged stats:\s*lr:\s*" + NUMBER + r"\s+loss:\s*(" + NUMBER + r")")
STEP_RE = re.compile(r"Train: data epoch:\s*\[(\d+)\]\s*\[\s*(\d+)/(\d+)\]")


def check_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / CONFIG)
    if cfg.model.answer_score_mode != "yes":
        raise RuntimeError("H1 must preserve S2 Yes-only binary score")
    if float(cfg.model.history_pref_weight) <= 0:
        raise RuntimeError("H1 history preference loss must be enabled")
    if cfg.model.candidate_history_fusion:
        raise RuntimeError("H1 must not contain A1/A2 candidate fusion")
    if cfg.model.ckpt is not None or cfg.run.resume_ckpt_path is not None:
        raise RuntimeError("H1 must train from scratch, not warm-start/resume")
    if cfg.model.gnn_semantic_align is not True:
        raise RuntimeError("S2 semantic alignment must remain enabled")


def run(cmd, gpu, logfile=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tail = deque(maxlen=15)
    values, train_losses = [], []
    fp = logfile.open("w", encoding="utf-8") if logfile else None
    try:
        with subprocess.Popen(
            cmd, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as p:
            for line in p.stdout:
                tail.append(line.strip()[:550])
                if fp is not None:
                    fp.write(line)
                    fp.flush()
                if "***auc:" in line:
                    m = AUC_RE.search(line)
                    if m:
                        values.append((float(m.group(1)), float(m.group(2))))
                        print("  AUC={:.8f} UAUC={:.8f}".format(*values[-1]), flush=True)
                elif "Averaged stats: lr:" in line:
                    m = LOSS_RE.search(line)
                    if m:
                        train_losses.append(float(m.group(1)))
                        print("  Train loss={:.6f}".format(train_losses[-1]), flush=True)
                elif fp is not None and ("[H1]" in line or "Saving checkpoint at epoch" in line):
                    print("  " + line.strip()[:250], flush=True)
                elif fp is not None:
                    m = STEP_RE.search(line)
                    if m and int(m.group(2)) % 3000 == 0:
                        print("  Epoch {} step {}/{}".format(*m.groups()), flush=True)
            exitcode = p.wait()
    finally:
        if fp is not None:
            fp.close()
    if exitcode != 0:
        raise RuntimeError("Command failed (exit={}):\n{}".format(
            exitcode, "\n".join(tail)))
    return values, train_losses


def best_from_log(outdir):
    checkpoints = sorted(outdir.rglob("checkpoint_best.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected exactly one best checkpoint, found {} in {}".format(
            len(checkpoints), outdir))
    ckpt = checkpoints[0]
    valid_log = ckpt.parent / "valid_log.txt"
    if not valid_log.is_file():
        raise RuntimeError("Missing " + str(valid_log))
    results = []
    for line in valid_log.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            auc = row.get("valid_agg_metrics")
            if auc is None:
                continue
            uauc = row.get("valid_uauc")
            uauc = uauc[0] if isinstance(uauc, (list, tuple)) and uauc else None
            results.append((float(auc), None if uauc is None else float(uauc)))
        except (ValueError, TypeError):
            continue
    if not results:
        raise RuntimeError("Valid AUC/UAUC missing in {}".format(valid_log))
    best_idx = max(range(len(results)), key=lambda i: results[i][0])
    return ckpt, best_idx, results


def write_report(stamp, info):
    dest = ROOT / "out"
    dest.mkdir(parents=True, exist_ok=True)
    stem = "H1_history_preference_" + stamp
    txtfile, csvfile = dest / (stem + ".txt"), dest / (stem + ".csv")
    fmt = lambda x: "-" if x is None else "{:.8f}".format(x)
    report = [
        "H1: original S2 Yes-logit binary CTR, extra history-candidate GNN BCE; no ranking",
        "Status: " + info["status"],
        "Best epoch (0-index): " + str(info["best_epoch"]),
        "Valid AUC: " + fmt(info["valid_auc"]),
        "Valid UAUC: " + fmt(info["valid_uauc"]),
        "Test AUC: " + fmt(info["test_auc"]),
        "Test UAUC: " + fmt(info["test_uauc"]),
        "Train loss per epoch: " + info["train_loss_per_epoch"],
        "Valid AUC per epoch: " + info["valid_auc_per_epoch"],
        "Checkpoint: " + info["checkpoint"],
        "Training log: " + info["train_log"],
        "Error: " + info["error"],
    ]
    txtfile.write_text("\n".join(report) + "\n", encoding="utf-8")
    with csvfile.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(info.keys()))
        writer.writeheader()
        writer.writerow(info)
    return txtfile, csvfile, "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--skip-test", action="store_true",
                        help="Stop after best validation checkpoint")
    args = parser.parse_args()
    check_config()
    subprocess.run([sys.executable, "-m", "unittest",
                    "tests/test_history_preference_aux.py", "-v"],
                   cwd=str(ROOT), check=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "output" / ("H1_HISTORY_PREFERENCE_" + stamp)
    logdir = ROOT / "log"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / ("H1_HISTORY_PREFERENCE_train_" + stamp + ".log")
    info = dict(
        status="RUNNING", best_epoch="", valid_auc=None, valid_uauc=None,
        test_auc=None, test_uauc=None, train_loss_per_epoch="",
        valid_auc_per_epoch="", checkpoint="", train_log=str(logfile), error="",
    )
    print("H1 history-preference auxiliary (CTR/BCE, no rank loss). Starting foreground training.", flush=True)
    try:
        cmd = [sys.executable, "-u", "train_collm_mf_din.py", "--cfg-path", CONFIG,
               "--options", "model.ckpt=null", "run.evaluate=false",
               "run.resume_ckpt_path=null", "run.output_dir=" + str(outdir)]
        _, losses = run(cmd, args.gpu, logfile)
        ckpt, epoch, valid = best_from_log(outdir)
        info.update(
            status="TRAINED", best_epoch=epoch,
            valid_auc=valid[epoch][0], valid_uauc=valid[epoch][1],
            train_loss_per_epoch=";".join("{:.6f}".format(x) for x in losses),
            valid_auc_per_epoch=";".join("{:.8f}".format(x[0]) for x in valid),
            checkpoint=str(ckpt),
        )
        write_report(stamp, info)
        print("Best Valid epoch={} AUC={:.8f} UAUC={}".format(
            epoch, valid[epoch][0], valid[epoch][1]), flush=True)
        if not args.skip_test:
            test_cmd = [
                sys.executable, "-u", "train_collm_mf_din.py",
                "--cfg-path", CONFIG, "--options",
                "model.ckpt=" + str(ckpt), "run.evaluate=true",
                "run.test_splits=[test]",
                "run.output_dir=" + str(ROOT / "output" / ("H1_TEST_" + stamp)),
            ]
            print("Testing best-Valid checkpoint on Test...", flush=True)
            metrics, _ = run(test_cmd, args.gpu)
            if not metrics:
                raise RuntimeError("No final Test AUC/UAUC found")
            info["test_auc"], info["test_uauc"] = metrics[-1]
        info["status"] = "OK"
    except Exception as exc:
        info["status"] = "FAILED"
        info["error"] = str(exc)
        print("H1 experiment failed:", str(exc)[:1500], flush=True)
    txt, csvf, summary = write_report(stamp, info)
    print("\n" + "="*65 + "\nH1 FINAL METRICS\n" + "="*65, flush=True)
    print(summary, flush=True)
    print("Saved:", txt, csvf, flush=True)
    return int(info["status"] != "OK")


if __name__ == "__main__":
    sys.exit(main())
