#!/usr/bin/env python3
"""Diagnose A2 training collapse; fail closed if S2 warm-start cannot be reproduced.

Sequence:
  1. CPU fusion tests and config sanity check.
  2. Evaluate S2 checkpoint through A2 with an exact identity gate on Valid.
  3. Train A2-TrainOff from scratch (2 epochs).
  4. Train A2-S2Init from S2 model weights with a fresh low-LR optimizer
     (2 epochs; non-frozen LoRA/GNN/projector and independent gate LR).
Every stage saves a compact report to out/, training console logs to log/.
No full evaluation console .log is saved. Test split is not used for model choice.
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
EXPERIMENTS = (
    ("A2-TrainOff", "tran_configs/S2_candidate_history_A2_trainoff.yaml", "A2_TRAIN_OFF"),
    ("A2-S2Init", "tran_configs/S2_candidate_history_A2_s2init.yaml", "A2_S2_INIT"),
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
EVAL = re.compile(r"\*\*\*auc:\s*(" + NUMBER + r")\s+\*\*\*uauc:\s*\(\s*(" + NUMBER + r")")
TRAIN = re.compile(r"Averaged stats:\s*lr:\s*(" + NUMBER + r")\s+loss:\s*(" + NUMBER + r")")


def command_stream(command, gpu, logfile=None, selected_only=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    last_lines = deque(maxlen=20)
    metrics = None
    train_losses = []
    restored = []
    if logfile:
        outfile = logfile.open("w", encoding="utf-8")
    else:
        outfile = None
    try:
        with subprocess.Popen(
            command, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as proc:
            for line in proc.stdout:
                last_lines.append(line.rstrip()[:500])
                if outfile:
                    outfile.write(line)
                    outfile.flush()
                if "[S2-init]" in line:
                    restored.append(line.strip())
                    print("  " + line.strip(), flush=True)
                if "Averaged stats:" in line:
                    found = EVAL.search(line)
                    if found:
                        metrics = float(found.group(1)), float(found.group(2))
                        print("  Valid AUC={:.8f} UAUC={:.8f}".format(*metrics), flush=True)
                    loss = TRAIN.search(line)
                    if loss:
                        train_losses.append(float(loss.group(2)))
                        print("  Train loss={:.6f}".format(float(loss.group(2))), flush=True)
                if outfile and ("Start training epoch" in line or "Saving checkpoint at epoch" in line):
                    print("  " + line.strip()[:180], flush=True)
            code = proc.wait()
    finally:
        if outfile:
            outfile.close()
    if code != 0:
        raise RuntimeError("Subprocess failed exit={}:\n{}".format(
            code, "\n".join(last_lines)))
    return metrics, train_losses, restored


def get_best_epoch(output_root):
    checkpoints = list(output_root.rglob("checkpoint_best.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected exactly one best checkpoint in {}, found {}"
                           .format(output_root, len(checkpoints)))
    checkpoint = checkpoints[0]
    log = checkpoint.parent / "valid_log.txt"
    if not log.is_file():
        raise RuntimeError("Missing valid_log.txt: {}".format(log))
    by_epoch = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if "valid_agg_metrics" in item:
            u = item.get("valid_uauc")
            uauc = u[0] if isinstance(u, (list, tuple)) and u else None
            by_epoch.append((
                float(item["valid_agg_metrics"]),
                None if uauc is None else float(uauc),
            ))
    if not by_epoch:
        raise RuntimeError("No valid metrics in {}".format(log))
    best = max(range(len(by_epoch)), key=lambda i: by_epoch[i][0])
    return checkpoint, best, by_epoch


def save_report(rows, stamp, s2):
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    txtpath = out / ("A2_training_recovery_" + stamp + ".txt")
    csvpath = out / ("A2_training_recovery_" + stamp + ".csv")
    fmt = lambda v: "-" if v is None else "{:.8f}".format(v)
    lines = [
        "A2 training recovery: TrainOff vs S2Init | VALID only",
        "S2 checkpoint: {}".format(s2),
        "",
        "{:<13} {:>12} {:>12} {:>10} {:>12}".format(
            "Experiment", "Valid AUC", "Valid UAUC", "Best ep", "Status"),
        "-" * 68,
    ]
    for r in rows:
        lines.append("{:<13} {:>12} {:>12} {:>10} {:>12}".format(
            r["experiment"], fmt(r["valid_auc"]), fmt(r["valid_uauc"]),
            str(r["best_epoch"]), r["status"]))
        lines.append("  checkpoint: " + r["checkpoint"])
        lines.append("  training log: " + r["train_log"])
        lines.append("  train loss/epoch: " + r["train_loss_by_epoch"])
        lines.append("  valid AUC/epoch: " + r["valid_auc_by_epoch"])
        if r["error"]:
            lines.append("  error: " + r["error"][:800].replace("\n", " | "))
    txtpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cols = list(rows[0].keys())
    with csvpath.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return txtpath, csvpath, "\n".join(lines)


def default_s2():
    exact = ROOT / "output" / "S2_groups_acc3" / "20261010223" / "checkpoint_best.pth"
    if exact.is_file():
        return exact
    options = list((ROOT / "output" / "S2_groups_acc3").rglob("checkpoint_best.pth"))
    if len(options) != 1:
        raise RuntimeError(
            "S2 best checkpoint unclear: found {} options. Supply --s2-ckpt PATH."
            .format(len(options)))
    return options[0]


def validate_configs():
    # Fail before GPU allocation if critical CLI/config flags are absent.
    from omegaconf import OmegaConf
    off = OmegaConf.load(ROOT / EXPERIMENTS[0][1])
    init = OmegaConf.load(ROOT / EXPERIMENTS[1][1])
    assert off.model.candidate_hist_train_off is True
    assert off.model.candidate_history_mode == "gate"
    assert not off.model.s2_init_verify
    assert off.model.ckpt is None
    assert init.model.candidate_hist_train_off is False
    assert init.model.candidate_history_mode == "gate"
    assert init.model.s2_init_verify is True
    assert float(init.run.lr_scale_candidate) > 0
    assert float(init.run.init_lr) < float(off.run.init_lr)
    assert init.model.ckpt is None
    assert init.run.resume_ckpt_path is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--s2-ckpt", type=Path, default=None)
    parser.add_argument("--min-s2-valid-auc", type=float, default=0.7279,
                        help="Fail fast if preflight AUC does not reproduce S2 0.732884 (default tolerance ~0.005)")
    args = parser.parse_args()
    checkpoint = (
        args.s2_ckpt.expanduser().resolve()
        if args.s2_ckpt is not None else default_s2()
    )
    if not checkpoint.is_file():
        parser.error("S2 checkpoint missing: {}".format(checkpoint))
    print("S2 source checkpoint:", checkpoint, flush=True)
    validate_configs()
    subprocess.run(
        [sys.executable, "-m", "unittest",
         "tests/test_candidate_history_fusion.py", "-v"],
        cwd=str(ROOT), check=True,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (ROOT / "log").mkdir(parents=True, exist_ok=True)
    (ROOT / "out").mkdir(parents=True, exist_ok=True)

    # Evaluate with identity GNN gate, checkpoint model weights must load.
    print("\nPREFLIGHT: S2 checkpoint -> A2 identity gate -> VALID", flush=True)
    verify_cmd = [
        sys.executable, "-u", "train_collm_mf_din.py",
        "--cfg-path", EXPERIMENTS[1][1],
        "--options", "model.ckpt=" + str(checkpoint),
        "model.candidate_hist_eval_multiplier=0",
        "run.evaluate=true", "run.test_splits=[valid]",
        "run.output_dir=" + str(ROOT / "output" / ("A2_S2_PREFLIGHT_" + stamp)),
    ]
    metrics, _, restore = command_stream(verify_cmd, args.gpu)
    if metrics is None:
        raise RuntimeError("S2 preflight failed to print Valid AUC/UAUC")
    if not restore or "[S2-init] Model weights restored" not in "\n".join(restore):
        raise RuntimeError("S2 preflight did not verify checkpoint weights")
    if metrics[0] < args.min_s2_valid_auc:
        raise RuntimeError(
            "S2 preflight AUC {:.8f} below required {:.8f}; "
            "STOP before training; inspect checkpoint/protocol".format(
                metrics[0], args.min_s2_valid_auc))
    print("PREFLIGHT PASSED: AUC={:.8f} UAUC={:.8f}".format(*metrics), flush=True)

    rows = []
    for label, config, name in EXPERIMENTS:
        row = dict(
            experiment=label, valid_auc=None, valid_uauc=None,
            best_epoch="-", status="RUNNING", checkpoint="", train_log="",
            train_loss_by_epoch="", valid_auc_by_epoch="", error="",
        )
        rows.append(row)
        run_dir = ROOT / "output" / (name + "_" + stamp)
        log_path = ROOT / "log" / (name + "_train_" + stamp + ".log")
        row["train_log"] = str(log_path)
        cmd = [
            sys.executable, "-u", "train_collm_mf_din.py",
            "--cfg-path", config,
            "--options", "run.evaluate=false",
            "run.output_dir=" + str(run_dir),
            "run.resume_ckpt_path=null",
        ]
        if label == "A2-S2Init":
            # Initialize model weights from checkpoint; do NOT resume optimizer.
            cmd.append("model.ckpt=" + str(checkpoint))
        else:
            cmd.append("model.ckpt=null")
        print("\n{}: training".format(label), flush=True)
        try:
            _, losses, restored = command_stream(cmd, args.gpu, log_path)
            if label == "A2-S2Init" and not any(
                "[S2-init] Model weights restored" in line for line in restored
            ):
                raise RuntimeError("S2Init training process did not confirm weight restoration")
            best_ckpt, epoch, scores = get_best_epoch(run_dir)
            row.update(
                status="OK", checkpoint=str(best_ckpt),
                best_epoch=epoch, valid_auc=scores[epoch][0],
                valid_uauc=scores[epoch][1],
                train_loss_by_epoch="; ".join("{:.6f}".format(x) for x in losses),
                valid_auc_by_epoch="; ".join("{:.8f}".format(x[0]) for x in scores),
            )
            print("{} best Valid AUC={:.8f} UAUC={} epoch={}".format(
                label, row["valid_auc"], row["valid_uauc"], epoch), flush=True)
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = str(exc)
            print(label + " FAILED:\n" + str(exc)[:1000], flush=True)
        save_report(rows, stamp, checkpoint)

    txt, csvfile, table = save_report(rows, stamp, checkpoint)
    print("\nFINAL RECOVERY SUMMARY\n" + table, flush=True)
    print("Saved TXT:", txt, "\nSaved CSV:", csvfile, flush=True)
    return int(any(r["status"] != "OK" for r in rows))


if __name__ == "__main__":
    sys.exit(main())
