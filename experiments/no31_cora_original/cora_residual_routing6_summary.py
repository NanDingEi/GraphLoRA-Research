#!/usr/bin/env python3
"""Read-only VALID result summary for the six CoRA residual routing trials.

Reads completed/partial trial stdout logs and the original RunnerBase
valid_log.txt JSONL files. Does not import torch, start GPU work, load models,
modify checkpoints, or invoke evaluation. UAUC is paired with best AUC epoch.
"""
import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "CORA_RESIDUAL_ROUTING6_20261103"
NAMES = [
    "RR1_L32_BASE",
    "RR2_GLOBAL_R31",
    "RR3_FIXED_MULTI",
    "RR4_GLOBAL_ROUTER",
    "RR5_USER_GATE_R31",
    "RR6_USER_ROUTER_MULTI",
]
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
EVAL_PAIR = re.compile(
    r"\*\*\*auc:\s*(" + NUMBER + r")\s+\*\*\*uauc:\s*\(?\s*(" + NUMBER + r")",
    flags=re.IGNORECASE,
)
EPOCH_START = re.compile(r"Start training epoch\s+(\d+)")
EPOCH_BEST = re.compile(r"\[JOINT8 BEST\]\s+epoch=(\d+)\s+valid_auc=(" + NUMBER + r")")
FAIL = re.compile(
    r"Traceback \(most recent call last\)|^\s*(?:RuntimeError|"
    r"torch\.OutOfMemoryError|FloatingPointError|ValueError|"
    r"AssertionError|OSError|TypeError):|\[R6 STOP ON FAILURE\]",
    flags=re.MULTILINE,
)
FINISH = re.compile(r"\bTraining time\b|\[JOINT8 EARLY STOP\]")


def as_float(value):
    """Handle RunnerBase valid_uauc as float, tuple/list or {score: ...}."""
    try:
        if isinstance(value, dict):
            value = value.get("score")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return float(value) if value is not None else None
    except (ValueError, TypeError, IndexError):
        return None


def read_valid_jsonl(path):
    records = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, raw in enumerate(stream, 1):
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                auc = as_float(obj.get("valid_agg_metrics"))
                uauc = as_float(obj.get("valid_uauc"))
                if auc is None:
                    continue
                records.append(
                    {
                        "auc": auc,
                        "uauc": uauc,
                        "epoch": obj.get("valid_best_epoch"),
                        "source": f"{path}:{line_number}",
                        "job_dir": path.parent,
                    }
                )
    except OSError:
        pass
    return records


def parse_log(path):
    if not path.is_file():
        return "NO_LOG", []
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "UNREADABLE", []
    if FAIL.search(data):
        status = "FAILED"
    elif FINISH.search(data):
        status = "COMPLETE"
    else:
        status = "INCOMPLETE"

    records = []
    # Log points are a fallback when no valid_log.txt is available.
    epoch = None
    within_valid = False
    for line in data.splitlines():
        match = EPOCH_START.search(line)
        if match:
            epoch = int(match.group(1))
            within_valid = False
        if "Evaluating on valid." in line:
            within_valid = True
        if within_valid:
            match = EVAL_PAIR.search(line)
            if match:
                records.append({
                    "auc": float(match.group(1)),
                    "uauc": float(match.group(2)),
                    "epoch": epoch,
                    "source": str(path),
                    "job_dir": None,
                })
                within_valid = False
    markers = [
        (int(match.group(1)), float(match.group(2)))
        for match in EPOCH_BEST.finditer(data)
    ]
    if records:
        selected = max(records, key=lambda v: v["auc"])
        if selected["epoch"] is None:
            for marker_epoch, marker_auc in reversed(markers):
                if abs(marker_auc - selected["auc"]) <= 5e-7:
                    selected["epoch"] = marker_epoch
                    break
    return status, records


def checkpoint_status(folder):
    if folder is None or not folder.is_dir():
        return "MISSING", ""
    path = folder / "checkpoint_best.pth"
    if not path.exists():
        return "MISSING", str(path)
    try:
        if path.stat().st_size > 0 and zipfile.is_zipfile(path):
            return "FOUND", str(path)
        return "INVALID", str(path)
    except OSError:
        return "UNREADABLE", str(path)


def collect():
    output_root = HERE / "output" / PROTOCOL
    log_root = HERE / "logs" / PROTOCOL
    rows = []
    for name in NAMES:
        trial_folder = output_root / name
        stdout = log_root / f"{name}.log"
        status, log_records = parse_log(stdout)
        jsonl_files = sorted(trial_folder.rglob("valid_log.txt")) if trial_folder.is_dir() else []
        records = []
        for p in jsonl_files:
            records.extend(read_valid_jsonl(p))
        if records:
            best = max(records, key=lambda v: v["auc"])
        elif log_records:
            best = max(log_records, key=lambda v: v["auc"])
        else:
            best = None
        checkpoint_dir = best["job_dir"] if best else None
        ckpt, checkpoint_path = checkpoint_status(checkpoint_dir)

        # The log-only fallback cannot establish the precise job folder.
        if best is not None and checkpoint_dir is None and trial_folder.is_dir():
            job_paths = sorted(trial_folder.rglob("checkpoint_best.pth"))
            if len(job_paths) == 1:
                ckpt, checkpoint_path = checkpoint_status(job_paths[0].parent)

        if status == "NO_LOG" and not records:
            status = "NOT_STARTED"
        elif status == "NO_LOG" and records:
            status = "NO_LOG_HAS_VALID"
        rows.append({
            "trial": name,
            "auc": best["auc"] if best else None,
            "uauc_at_best_auc": best["uauc"] if best else None,
            "best_epoch": best["epoch"] if best else None,
            "status": status,
            "checkpoint": ckpt,
            "valid_records": len(records) if records else len(log_records),
            "source": best["source"] if best else "",
            "checkpoint_path": checkpoint_path,
            "stdout_log": str(stdout),
        })
    baseline = rows[0]
    for row in rows:
        row["delta_auc"] = (
            row["auc"] - baseline["auc"]
            if row["auc"] is not None and baseline["auc"] is not None
            else None
        )
        row["delta_uauc"] = (
            row["uauc_at_best_auc"] - baseline["uauc_at_best_auc"]
            if row["uauc_at_best_auc"] is not None
            and baseline["uauc_at_best_auc"] is not None
            else None
        )
    return rows


def render_float(value, delta=False):
    if value is None:
        return "—"
    return f"{value:+.6f}" if delta else f"{value:.6f}"


def show(rows):
    print("\nCoRA RESIDUAL ROUTING 6 — BEST VALIDATION AUC (NOT TEST)")
    header = (
        f"{'Trial':29} {'AUC':>10} {'UAUC@best':>11} "
        f"{'dAUC':>10} {'dUAUC':>10} {'Epoch':>6} "
        f"{'Status':>12} {'CKPT':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['trial']:29} {render_float(r['auc']):>10} "
            f"{render_float(r['uauc_at_best_auc']):>11} "
            f"{render_float(r['delta_auc'], True):>10} "
            f"{render_float(r['delta_uauc'], True):>10} "
            f"{str(r['best_epoch']) if r['best_epoch'] is not None else '—':>6} "
            f"{r['status']:>12} {r['checkpoint']:>9}"
        )
    eligible = [
        r for r in rows
        if r["status"] == "COMPLETE"
        and r["checkpoint"] == "FOUND"
        and r["auc"] is not None
    ]
    if eligible:
        best = max(eligible, key=lambda r: r["auc"])
        print(
            f"\nBest completed run with checkpoint: {best['trial']}"
            f" — AUC={best['auc']:.6f}, "
            f"UAUC@bestAUC={render_float(best['uauc_at_best_auc'])}, "
            f"epoch={best['best_epoch']}"
        )
    else:
        print("\nNo completed run with a structurally present checkpoint found.")
    print(
        "UAUC is from the SAME validation record as best AUC; "
        "not independently maximized."
    )
    print(
        "Checkpoint FOUND checks only file/ZIP structure; it does not "
        "test whether model weights can be loaded."
    )
    print("Brier/LogLoss are not reconstructed without saved per-sample predictions.\n")


def write(rows):
    out = HERE / "out" / PROTOCOL
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "six_trial_valid_summary.csv"
    fields = [
        "trial", "auc", "uauc_at_best_auc", "delta_auc", "delta_uauc",
        "best_epoch", "status", "checkpoint", "valid_records",
        "source", "checkpoint_path", "stdout_log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path = out / "six_trial_valid_summary.md"
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write("# Six CoRA residual routing trials — Valid only\n\n")
        stream.write(
            "| Trial | AUC | UAUC@bestAUC | ΔAUC | ΔUAUC | Epoch | Status | CKPT |\n"
            "|---|---:|---:|---:|---:|---:|---|---|\n"
        )
        for r in rows:
            stream.write(
                f"| {r['trial']} | {render_float(r['auc'])} | "
                f"{render_float(r['uauc_at_best_auc'])} | "
                f"{render_float(r['delta_auc'], True)} | "
                f"{render_float(r['delta_uauc'], True)} | "
                f"{r['best_epoch'] if r['best_epoch'] is not None else '—'} | "
                f"{r['status']} | {r['checkpoint']} |\n"
            )
        stream.write(
            "\nUAUC is paired with the best AUC epoch, not independently maximized. "
            "Failed/incomplete attempts are not successful training runs. "
            "Checkpoint file checks are structural only. No Test results.\n"
        )
    print(f"CSV saved: {csv_path}")
    print(f"Markdown saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Print only; don't write summary files")
    args = parser.parse_args()
    rows = collect()
    show(rows)
    if not args.no_write:
        write(rows)


if __name__ == "__main__":
    main()
