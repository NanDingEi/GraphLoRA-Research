#!/usr/bin/env python3
"""Read-only recovery of M1-M6 collaborative CoRA LoRA-MoE VALID results.

No model instantiation, GPU, checkpoint modification or TEST evaluation.
Parse trial stdout AUC+UAUC first, discover checkpoint_best.pth independently
from the YAML output_dir and stdout [R4E WORKER] directory, then write CSV.
"""
import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = "CORA_COLLAB_MOE6_20261104"
NAMES = [
    "M1_GLOBAL",
    "M2_USER_CLUSTER",
    "M3_ITEM_CLUSTER",
    "M4_PAIR",
    "M5_MULTI_INTEREST",
    "M6_LAYER_PAIR",
]
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
EVAL = re.compile(
    r"\*\*\*auc:\s*(" + NUMBER + r")\s+\*\*\*uauc:\s*"
    r"[\(\[]?\s*(?:np\.float(?:32|64)\s*\()?\s*(" + NUMBER + r")",
    re.IGNORECASE,
)
EPOCH = re.compile(r"Start training epoch\s+(\d+)")
BEST = re.compile(r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=(" + NUMBER + r")")
FINISHED = re.compile(r"\bTraining time\b|\[JOINT8 EARLY STOP\]")
FAILED = re.compile(
    r"Traceback \(most recent call last\)|"
    r"^\s*(?:RuntimeError|ValueError|TypeError|FloatingPointError|"
    r"torch\.OutOfMemoryError|AssertionError|OSError):",
    re.MULTILINE,
)
WORKER_OUT = re.compile(r"\[R4E WORKER\]\s+\S+\s+output=(\S+)")


def parse_log(path):
    if not path.is_file():
        return "NOT_STARTED", [], [], [], 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "UNREADABLE", [], [], [], 0
    points = []
    markers = []
    output_dirs = []
    epoch = None
    valid_count = 0
    for line_no, line in enumerate(lines, 1):
        m = EPOCH.search(line)
        if m:
            epoch = int(m.group(1))
        if "Evaluating on valid." in line:
            valid_count += 1
        m = BEST.search(line)
        if m:
            markers.append((int(m.group(1)), float(m.group(2))))
        m = WORKER_OUT.search(line)
        if m:
            output_dirs.append(Path(m.group(1)))
        m = EVAL.search(line)
        if m:
            points.append({
                "auc": float(m.group(1)),
                "uauc": float(m.group(2)),
                "epoch": epoch,
                "source": f"{path}:{line_no}",
            })
    # Do not misclassify failed/incomplete runs based on a stale completed
    # validation. All data are still returned, but status is explicit.
    content = "\n".join(lines)
    status = (
        "FAILED" if FAILED.search(content)
        else "COMPLETE" if FINISHED.search(content)
        else "INCOMPLETE"
    )
    for point in points:
        if point["epoch"] is None:
            for ep, auc in reversed(markers):
                if abs(point["auc"] - auc) < 5e-7:
                    point["epoch"] = ep
                    break
    return status, points, markers, output_dirs, valid_count


def numeric(value):
    if isinstance(value, dict):
        value = value.get("score")
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def candidate_dirs(protocol, name, from_log):
    base_dir = HERE / "output" / protocol / name
    candidates = [base_dir, *from_log]
    cfg_path = HERE / "local_configs" / protocol / (name + ".yaml")
    if cfg_path.is_file():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            path = Path(str(cfg["run"]["output_dir"])).expanduser()
            candidates.extend([path, HERE.parents[1] / path])
        except (ValueError, OSError, TypeError, KeyError):
            pass
    unique = {}
    for folder in candidates:
        key = str(folder.resolve())
        unique[key] = folder.resolve()
    return list(unique.values())


def jsonl_points(folders):
    files = set()
    for folder in folders:
        if folder.is_dir():
            files.update(folder.rglob("valid_log.txt"))
    points = []
    for path in sorted(files):
        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        for n, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            auc = numeric(obj.get("valid_agg_metrics"))
            ua = numeric(obj.get("valid_uauc"))
            if auc is not None:
                points.append({
                    "auc": auc, "uauc": ua,
                    "epoch": obj.get("valid_best_epoch"),
                    "source": f"{path}:{n}",
                })
    return points, len(files)


def checkpoint_files(folders):
    found = set()
    for folder in folders:
        if folder.is_dir():
            found.update(folder.rglob("checkpoint_best.pth"))
    paths = sorted(found)
    valid = []
    for p in paths:
        try:
            if p.stat().st_size > 0 and zipfile.is_zipfile(p):
                valid.append(str(p))
        except OSError:
            pass
    return len(paths), valid


def collect(protocol):
    rows = []
    for name in NAMES:
        logfile = HERE / "logs" / protocol / f"{name}.log"
        status, points_log, markers, output_dirs, valid_blocks = parse_log(logfile)
        folders = candidate_dirs(protocol, name, output_dirs)
        points_file, nfiles = jsonl_points(folders)
        # Prefer structured validation records; if none, use logger.
        points = points_file if points_file else points_log
        best = max(points, key=lambda x: x["auc"]) if points else None
        count_ckpt, valid_ckpt = checkpoint_files(folders)
        if valid_ckpt:
            ckpt_status = "FOUND_UNVERIFIED"
        elif count_ckpt:
            ckpt_status = "CORRUPT_OR_PARTIAL"
        else:
            ckpt_status = "NOT_FOUND"
        if status == "COMPLETE" and best is None:
            status = "COMPLETE_NO_VALID"
        rows.append({
            "trial": name,
            "valid_auc": best["auc"] if best else None,
            "valid_uauc_at_best_auc": best["uauc"] if best else None,
            "best_epoch": best["epoch"] if best else None,
            "delta_auc_vs_S1": None,
            "delta_uauc_vs_S1": None,
            "status": status,
            "checkpoint": ckpt_status,
            "stdout_valid_points": len(points_log),
            "valid_blocks": valid_blocks,
            "valid_log_files": nfiles,
            "checkpoint_files": count_ckpt,
            "source": best["source"] if best else "",
            "checkpoint_paths": ";".join(valid_ckpt),
            "search_dirs": ";".join(str(p) for p in folders),
        })
    baseline = rows[0]
    for row in rows:
        if baseline["valid_auc"] is not None and row["valid_auc"] is not None:
            row["delta_auc_vs_S1"] = row["valid_auc"] - baseline["valid_auc"]
        if (baseline["valid_uauc_at_best_auc"] is not None
                and row["valid_uauc_at_best_auc"] is not None):
            row["delta_uauc_vs_S1"] = (
                row["valid_uauc_at_best_auc"] - baseline["valid_uauc_at_best_auc"]
            )
    return rows


def fval(value, signed=False):
    if value is None:
        return "—"
    return (f"{value:+.6f}" if signed else f"{value:.6f}")


def run_summary(protocol=DEFAULT_PROTOCOL, debug=False):
    rows = collect(protocol)
    print("\nCOLLABORATIVE CoRA LoRA-MoE 6 — BEST VALID (NOT TEST)")
    print(f"{'trial':23} {'AUC':>10} {'UAUC@best':>12} "
          f"{'dAUC':>10} {'dUAUC':>10} {'Epoch':>7} {'Status':>18} {'CKPT':>18}")
    print("-" * 120)
    for row in rows:
        ep = str(row["best_epoch"]) if row["best_epoch"] is not None else "—"
        print(
            f"{row['trial']:23} {fval(row['valid_auc']):>10} "
            f"{fval(row['valid_uauc_at_best_auc']):>12} "
            f"{fval(row['delta_auc_vs_S1'], True):>10} "
            f"{fval(row['delta_uauc_vs_S1'], True):>10} "
            f"{ep:>7} {row['status']:>18} {row['checkpoint']:>18}"
        )
        print(
            f"  valid_stdout={row['stdout_valid_points']} "
            f"valid_blocks={row['valid_blocks']} "
            f"valid_files={row['valid_log_files']} "
            f"checkpoint_files={row['checkpoint_files']}"
        )
        if debug:
            print("  source=", row["source"])
            print("  checkpoints=", row["checkpoint_paths"])
            print("  search_dirs=", row["search_dirs"])
    out = HERE / "out" / protocol
    out.mkdir(parents=True, exist_ok=True)
    target = out / "collab_moe6_valid_summary.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[MOE6 SUMMARY] {target}")
    print(
        "AUC/UAUC paired by the best-AUC record; checkpoint existence/ZIP "
        "does NOT prove it can be loaded. No GPU or TEST used."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_summary(args.protocol, debug=args.debug)


if __name__ == "__main__":
    main()
