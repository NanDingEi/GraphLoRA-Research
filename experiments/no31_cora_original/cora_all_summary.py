#!/usr/bin/env python3
"""Aggregate all rounds of CoRA/MF VALID best metrics from local logs.

No GPU, imports of CoRA, checkpoint loading, training, or re-evaluation.
For each trial, show max logged Valid AUC, its paired UAUC, epoch, run status,
and checkpoint presence. Preserve failed/incomplete attempts and their metrics
without presenting them as successful, recoverable best models.
"""
import argparse
import csv
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = (
    "R0_FROZEN", "R1_MF_MAIN", "R2_MF_CTR", "R3_MF_KD",
    "R4_CTR_KD", "R5_CTR_KD_SEM", "R6_CTR_KD_ANC", "R7_ALL",
    "N1_CTR_001", "N2_CTR_010", "N3_KD_001", "N4_KD_010",
    "N5_CTR5_KD1", "N6_CTR1_KD5", "N7_KD_RAMP10", "N8_MF_DECOUPLE",
    "C1_WEAK_BOTH", "C2_CTR_DOM", "C3_KD_DOM", "C4_KD_AGREE",
    "C5_KD_CONF", "C6_KD_BOTH", "C7_BIDIR", "C8_MAIN_HALF",
)
NUMBER = re.compile(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")
EVAL = re.compile(
    r"\*\*\*auc:\s*(" + NUMBER.pattern + r")\s+\*\*\*uauc:\s*(" +
    NUMBER.pattern + r")", re.IGNORECASE)
EPOCH = re.compile(r"Start training epoch\s+(\d+)", re.IGNORECASE)
BEST = re.compile(r"\[JOINT8 BEST\]\s+epoch=(\d+)\s+valid_auc=(" +
                  NUMBER.pattern + r")")
FINISHED = re.compile(r"\[JOINT8 EARLY STOP\]|Training time", re.IGNORECASE)
FAILED = re.compile(
    r"Traceback \(most recent call last\):|^\s*RuntimeError:|"
    r"^\s*OSError:|^\s*UnboundLocalError:|"
    r"\[NEXT8 CKPT FAIL\]", re.MULTILINE)
RECOGNIZED = set(ORDER)

def candidate_logs(root):
    """Discover known CoRA trial names only; never mix in unrelated GraphLoRA logs."""
    base = root / "logs"
    collected = {key: [] for key in ORDER}
    if not base.is_dir():
        return collected
    for path in sorted(base.glob("CORA_*/*.log")):
        if not path.is_file():
            continue
        # Distinguish failed-attempt archives from canonical trial names.
        name = path.name
        for key in ORDER:
            if name == key + ".log" or name.startswith(key + ".failed"):
                collected[key].append(path)
                break
    return collected

def parse_log(path):
    epoch = None
    points = []
    latest_valid = False
    best_markers = []
    failed = False
    finished = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = EPOCH.search(line)
                if m:
                    epoch = int(m.group(1))
                    latest_valid = False
                if "Evaluating on valid." in line:
                    latest_valid = True
                if latest_valid:
                    m = EVAL.search(line)
                    if m:
                        points.append({
                            "auc": float(m.group(1)),
                            "uauc": float(m.group(2)),
                            "epoch": epoch,
                            "index": len(points),
                        })
                        latest_valid = False
                m = BEST.search(line)
                if m:
                    best_markers.append((int(m.group(1)), float(m.group(2))))
                if FINISHED.search(line):
                    finished = True
                if FAILED.search(line):
                    failed = True
    except OSError as exc:
        return {"path": path, "points": [], "status": "UNREADABLE",
                "error": str(exc)}
    status = "FAILED" if failed else "COMPLETE" if finished else "INCOMPLETE"
    if not points:
        status = "NO_VALID_" + status
    if points:
        best = max(points, key=lambda x: x["auc"])
        # If epoch is not exposed in a log, use the explicit best marker only
        # when it refers to exactly the same reported best AUC.
        if best["epoch"] is None:
            for e, a in reversed(best_markers):
                if abs(a - best["auc"]) <= 5e-7:
                    best["epoch"] = e
                    break
    else:
        best = None
    return {"path": path, "points": points, "best": best,
            "status": status, "error": ""}

def checkpoints(root, trial, protocol):
    folder = root / "output" / protocol / trial
    if not folder.is_dir():
        return []
    found = []
    # Look only for the best checkpoint, not the potentially corrupt
    # temporary files. ZIP integrity here is structural only, not a full
    # loading/validation of model weights or proof of matching best epoch.
    for p in sorted(folder.rglob("checkpoint_best.pth")):
        try:
            size = p.stat().st_size
            good = size > 0 and zipfile.is_zipfile(p)
        except OSError:
            size, good = 0, False
        found.append((p, good, size))
    return found

def pick(attempts):
    """Select successful attempt first, then best AUC; never discard failures."""
    if not attempts:
        return None
    with_points = [a for a in attempts if a.get("best")]
    if not with_points:
        return max(attempts, key=lambda a: a["path"].stat().st_mtime)
    return max(with_points, key=lambda a:
               (a["status"] == "COMPLETE",
                a["best"]["auc"],
                a["path"].stat().st_mtime))

def collect(root):
    rows = []
    discovered = candidate_logs(root)
    for name in ORDER:
        attempts = [parse_log(p) for p in discovered[name]]
        chosen = pick(attempts)
        if chosen is None:
            rows.append(dict(trial=name, round="Round1" if name.startswith("R") else "Round2" if name.startswith("N") else "Round3",
                             auc=None, uauc=None, epoch=None, status="NOT_FOUND",
                             ckpt="—", attempts=0, logfile="", protocol=""))
            continue
        best = chosen.get("best")
        proto = chosen["path"].parent.name
        available = checkpoints(root, name, proto)
        valid_zip = [p for p, ok, n in available if ok]
        present = [p for p, ok, n in available if not ok]
        # For failed/incomplete training, even a valid ZIP may belong to an
        # earlier best epoch. Do not claim it matches the selected AUC.
        if valid_zip:
            ckpt = "FOUND_UNVERIFIED" if chosen["status"] != "COMPLETE" else "FOUND"
        elif present:
            ckpt = "CORRUPT_OR_PARTIAL"
        else:
            ckpt = "MISSING"
        rows.append(dict(
            trial=name, round="Round1" if name.startswith("R") else "Round2" if name.startswith("N") else "Round3",
            auc=best["auc"] if best else None,
            uauc=best["uauc"] if best else None,
            epoch=best["epoch"] if best else None,
            status=chosen["status"], ckpt=ckpt,
            attempts=len(attempts), logfile=str(chosen["path"]), protocol=proto))
    return rows

def metric(v):
    return "—" if v is None else f"{v:.6f}"

def status(v):
    return {"COMPLETE": "DONE", "INCOMPLETE": "RUNNING/STOPPED",
            "FAILED": "FAILED", "NOT_FOUND": "NO LOG"}.get(v, v)

def display(rows, sort_by):
    scored = [r for r in rows if r["auc"] is not None]
    if sort_by == "auc":
        rows = sorted(rows, key=lambda x:
                      (x["auc"] is None, -(x["auc"] or 0)))
    elif sort_by == "uauc":
        rows = sorted(rows, key=lambda x:
                      (x["uauc"] is None, -(x["uauc"] or 0)))
    print("\nALL CoRA/MF VALID EXPERIMENTS — NOT TEST")
    print(f"{'Trial':20} {'AUC':>10} {'UAUC@bestAUC':>13} {'Epoch':>6} "
          f"{'Status':>16} {'CKPT':>18} {'Attempts':>8}")
    print("-" * 100)
    for r in rows:
        ep = "—" if r["epoch"] is None else str(r["epoch"])
        print(f"{r['trial']:20} {metric(r['auc']):>10} "
              f"{metric(r['uauc']):>13} {ep:>6} "
              f"{status(r['status']):>16} {r['ckpt']:>18} "
              f"{r['attempts']:>8}")
    # Rank completed experiments; checkpoint availability has separate status.
    ranked = [r for r in scored if r["status"] == "COMPLETE"]
    if ranked:
        a = max(ranked, key=lambda x: x["auc"])
        u = max(ranked, key=lambda x: x["uauc"])
        print(f"\nBEST COMPLETED VALID AUC : {a['trial']} {a['auc']:.6f}, "
              f"UAUC@bestAUC={a['uauc']:.6f}, epoch={a['epoch']}")
        print(f"BEST UAUC@AUC-best      : {u['trial']} {u['uauc']:.6f}, "
              f"AUC={u['auc']:.6f}, epoch={u['epoch']}")
    else:
        print("\nNo completed Valid runs found in recognized log folders.")
    if any(r["status"] == "FAILED" for r in rows):
        print("\nNOTE: FAILED rows show best *logged* Valid results only, "
              "NOT a guaranteed loadable best checkpoint.")
    if any(r["ckpt"] in ("MISSING", "CORRUPT_OR_PARTIAL", "FOUND_UNVERIFIED")
           for r in scored):
        print("NOTE: CKPT FOUND checks ZIP structure, not model contents "
              "or exact best-epoch correspondence.")
    if any(r["status"] == "NOT_FOUND" for r in rows):
        print("NOTE: NO LOG means no matching local file under logs/CORA_*; "
              "it does not prove an experiment was never run.")
    print("")

def write_reports(rows, folder):
    folder.mkdir(parents=True, exist_ok=True)
    csv_file = folder / "cora_all_valid_best.csv"
    fields = ["round", "trial", "auc", "uauc", "epoch", "status", "ckpt",
              "attempts", "protocol", "logfile"]
    with csv_file.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    md_file = folder / "cora_all_valid_best.md"
    with md_file.open("w", encoding="utf-8") as fp:
        fp.write("# CoRA / MF: all trial best validation metrics\n\n")
        fp.write("Valid only. UAUC is paired with the checkpoint-selection "
                 "best AUC, not independently maximized. "
                 "For failed runs the metrics may not have loadable checkpoints.\n\n")
        fp.write("| Trial | Valid AUC | UAUC@bestAUC | Epoch | Status | CKPT |\n")
        fp.write("|---|---:|---:|---:|---|---|\n")
        for r in rows:
            fp.write(f"| {r['trial']} | {metric(r['auc'])} | "
                     f"{metric(r['uauc'])} | "
                     f"{r['epoch'] if r['epoch'] is not None else '—'} | "
                     f"{status(r['status'])} | {r['ckpt']} |\n")
        fp.write("\nNotes: COMPLETE requires a training-finished/early-stop "
                 "marker without a traceback. ZIP structural checks do not "
                 "prove that a checkpoint corresponds to the reported best "
                 "epoch or can load successfully.\n")
    return csv_file, md_file

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="CoRA_official directory (default: script directory)")
    parser.add_argument("--sort", choices=("protocol", "auc", "uauc"),
                        default="auc")
    parser.add_argument("--no-write", action="store_true",
                        help="Print only; do not write CSV/Markdown files")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = collect(root)
    display(rows, args.sort)
    if not args.no_write:
        a, b = write_reports(rows, root / "out")
        print("CSV: ", a)
        print("Markdown:", b)

if __name__ == "__main__":
    main()
