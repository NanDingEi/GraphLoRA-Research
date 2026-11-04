#!/usr/bin/env python3
"""Recover six Split-LoRA-Expert VALID metrics from EXISTING logs/checkpoints.

Read-only for training artifacts. Uses no GPU, torch, model import or training.
Sources, in order: RunnerBase valid_log.txt; stdout ***auc/***uauc; JOINT8
best-epoch markers. Independently finds checkpoint_best.pth under actual
run.output_dir and the original protocol's output directory. Does not
manufacture metrics. Handles finished logs with no validation separately.
"""
import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = "CORA_SPLIT_EXPERTS6_20261104"
NAMES = [
    "S1_SPLIT_ALL",
    "S2_FIXED_TOP2",
    "S3_SIM_TOP2",
    "S4_SOFT4",
    "S5_WARMUP_TOP2",
    "S6_SHARED_C2F",
]
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
EVAL = re.compile(
    r"\*\*\*auc:\s*(" + NUM + r").*?\*\*\*uauc:\s*[\(\[]?\s*(" + NUM + r")",
    re.I,
)
EPOCH = re.compile(r"Start training epoch\s+(\d+)", re.I)
BEST = re.compile(r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=(" + NUM + r")", re.I)
TRAIN_EPOCH = re.compile(r"(?:data epoch:)\s*\[(\d+)\]", re.I)
FINISH = re.compile(r"Training time\b|\[JOINT8 EARLY STOP\]|\[R4E COMPLETE\]", re.I)
FAIL = re.compile(
    r"Traceback \(most recent call last\)|"
    r"^\s*(?:RuntimeError|ValueError|TypeError|"
    r"torch\.OutOfMemoryError|FloatingPointError|OSError):",
    re.I,
)
WORKER_OUTPUT = re.compile(r"\[R4E WORKER\]\s+\S+\s+output=(\S+)")
CKPT_MESSAGE = re.compile(r"Saving checkpoint .*?to\s+(\S+\.pth)", re.I)
RUNNER_CKPT = re.compile(r"\[NEXT8 CKPT\].*?epoch=(\d+)")


def numeric(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("score")
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def parse_stdout(path):
    points, markers, output_dirs = [], [], []
    complete = failed = False
    valid_blocks = 0
    epoch = None
    training_steps = 0
    if not path.is_file():
        return dict(points=points, markers=markers, output_dirs=output_dirs,
                    status="NOT_STARTED", valid_blocks=0, training_steps=0)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return dict(points=[], markers=[], output_dirs=[], status="UNREADABLE",
                    valid_blocks=0, training_steps=0)
    valid_context = False
    for index, line in enumerate(lines, 1):
        if "Evaluating on valid." in line:
            valid_blocks += 1
            valid_context = True
        if "Start training epoch" in line:
            match = EPOCH.search(line)
            if match:
                epoch = int(match.group(1))
                valid_context = False
        if "data epoch:" in line and epoch is None:
            match = TRAIN_EPOCH.search(line)
            if match:
                epoch = int(match.group(1))
        if "[R4E TRAIN]" in line:
            training_steps += 1
        if FINISH.search(line):
            complete = True
        if FAIL.search(line):
            failed = True
        match = WORKER_OUTPUT.search(line)
        if match:
            output_dirs.append(Path(match.group(1)))
        match = BEST.search(line)
        if match:
            markers.append((int(match.group(1)), float(match.group(2))))
        # This protocol has valid-only evaluation; preserve results even when
        # logger output lost the "Evaluating on valid." line.
        match = EVAL.search(line)
        if match:
            points.append({
                "auc": float(match.group(1)),
                "uauc": float(match.group(2)),
                "epoch": epoch,
                "source": f"{path}:{index}",
                "kind": "training_log",
                "job_dir": None,
                "within_valid_block": valid_context,
            })
            valid_context = False

    # If epoch association is missing or ambiguous, match the explicit
    # best-AUC marker. Do NOT claim this resolves every epoch in general.
    for point in points:
        if point["epoch"] is None:
            for ep, auc in reversed(markers):
                if abs(auc - point["auc"]) <= 5e-7:
                    point["epoch"] = ep
                    break
    status = "FAILED" if failed else "COMPLETE" if complete else "INCOMPLETE"
    return dict(points=points, markers=markers, output_dirs=output_dirs,
                status=status, valid_blocks=valid_blocks,
                training_steps=training_steps)


def output_dirs_for(name, protocol, stdout):
    # Read the absolute output_dir from the actual YAML when possible.
    candidates = list(stdout["output_dirs"])
    cfg = HERE / "local_configs" / protocol / (name + ".yaml")
    if cfg.is_file():
        try:
            import yaml
            run = yaml.safe_load(cfg.read_text(encoding="utf-8"))["run"]
            path = Path(str(run["output_dir"])).expanduser()
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.append(ROOT / path)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    candidates.extend([
        HERE / "output" / protocol / name,
        ROOT / "output" / protocol / name,
        ROOT / "experiments" / "no31_cora_original" / "output" / protocol / name,
    ])
    resolved = []
    seen = set()
    for p in candidates:
        target = p.resolve()
        if target not in seen:
            seen.add(target)
            resolved.append(target)
    return resolved


def read_valid_files(folders):
    points = []
    files = set()
    for folder in folders:
        if folder.is_dir():
            files.update(folder.rglob("valid_log.txt"))
    for path in sorted(files):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    auc = numeric(obj.get("valid_agg_metrics"))
                    ua = numeric(obj.get("valid_uauc"))
                    if auc is None:
                        continue
                    points.append({
                        "auc": auc,
                        "uauc": ua,
                        "epoch": obj.get("valid_best_epoch"),
                        "source": f"{path}:{line_number}",
                        "kind": "valid_log.txt",
                        "job_dir": path.parent,
                    })
        except OSError:
            continue
    return points, sorted(files)


def find_checkpoints(folders):
    paths = set()
    for folder in folders:
        if folder.is_dir():
            paths.update(folder.rglob("checkpoint_best.pth"))
    results = []
    for path in sorted(paths):
        try:
            ok = path.stat().st_size > 0 and zipfile.is_zipfile(path)
        except OSError:
            ok = False
        results.append((path, ok))
    return results


def recover(protocol):
    rows = []
    for name in NAMES:
        stdout_file = HERE / "logs" / protocol / (name + ".log")
        stdout = parse_stdout(stdout_file)
        folders = output_dirs_for(name, protocol, stdout)
        file_points, valid_files = read_valid_files(folders)
        # Use saved validation records first; fallback to logged epoch-paired
        # AUC/UAUC if RunnerBase wrote to a different folder.
        available = file_points if file_points else stdout["points"]
        best = max(available, key=lambda p: p["auc"]) if available else None
        checkpoints = find_checkpoints(folders)
        good_ckpts = [str(path) for path, good in checkpoints if good]
        if good_ckpts:
            ckpt_status = "FOUND_UNVERIFIED"
        elif checkpoints:
            ckpt_status = "INVALID"
        else:
            ckpt_status = "NOT_FOUND"
        state = stdout["status"]
        if best is None and state == "COMPLETE":
            state = "COMPLETE_NO_VALID"
        rows.append({
            "trial": name,
            "auc": best["auc"] if best else None,
            "uauc_at_best_auc": best["uauc"] if best else None,
            "epoch": best["epoch"] if best else None,
            "delta_auc_vs_RR1": best["auc"] - 0.732387 if best else None,
            "delta_uauc_vs_RR1": (
                best["uauc"] - 0.701999
                if best and best["uauc"] is not None else None
            ),
            "status": state,
            "checkpoint": ckpt_status,
            "source": best["source"] if best else "",
            "stdout_eval_points": len(stdout["points"]),
            "valid_blocks": stdout["valid_blocks"],
            "valid_files": len(valid_files),
            "training_log_steps": stdout["training_steps"],
            "checkpoints_found": len(checkpoints),
            "checkpoint_paths": ";".join(good_ckpts),
            "search_dirs": ";".join(str(p) for p in folders),
        })
    return rows


def fmt(x, sign=False):
    if x is None:
        return "—"
    return f"{x:+.6f}" if sign else f"{x:.6f}"


def display(rows, debug):
    print("\nSPLIT-LORA EXPERTS S1-S6 — VALID RESULTS (NOT TEST)")
    print(f"{'Trial':23} {'AUC':>10} {'UAUC@best':>11} {'dAUC':>10} "
          f"{'dUAUC':>10} {'Epoch':>7} {'Status':>20} {'CKPT':>18}")
    print("-" * 120)
    for row in rows:
        print(
            f"{row['trial']:23} {fmt(row['auc']):>10} "
            f"{fmt(row['uauc_at_best_auc']):>11} "
            f"{fmt(row['delta_auc_vs_RR1'], True):>10} "
            f"{fmt(row['delta_uauc_vs_RR1'], True):>10} "
            f"{str(row['epoch']) if row['epoch'] is not None else '—':>7} "
            f"{row['status']:>20} {row['checkpoint']:>18}"
        )
    print("\nDiagnostics: eval=number of AUC/UAUC pairs in stdout; "
          "valid_file=number of discovered valid_log.txt files")
    for row in rows:
        print(
            f"  {row['trial']}: eval={row['stdout_eval_points']}, "
            f"valid_blocks={row['valid_blocks']}, "
            f"valid_file={row['valid_files']}, "
            f"ckpt_files={row['checkpoints_found']}"
        )
        if debug:
            print("    source:", row["source"] or "none")
            print("    output candidates:", row["search_dirs"])
            print("    checkpoints:", row["checkpoint_paths"] or "none")
    usable = [
        r for r in rows if r["status"] == "COMPLETE"
        and r["auc"] is not None and r["uauc_at_best_auc"] is not None
    ]
    if usable:
        top = max(usable, key=lambda r: r["auc"])
        print(
            f"\nBest completed logged Valid AUC: {top['trial']} "
            f"{top['auc']:.6f}, paired UAUC={top['uauc_at_best_auc']:.6f}"
        )
    print(
        "No GPU or Test used; checkpoint FOUND_UNVERIFIED confirms only "
        "file existence and ZIP format, NOT model loadability.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    rows = recover(args.protocol)
    display(rows, args.debug)
    if not args.no_write:
        out = HERE / "out" / args.protocol
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "split_experts6_recovered_valid.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("[RECOVER] CSV:", csv_path)


if __name__ == "__main__":
    main()
