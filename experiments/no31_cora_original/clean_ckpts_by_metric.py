#!/usr/bin/env python3
"""Metrics-based checkpoint cleanup: delete checkpoints of trials whose best
Valid AUC is below a keep-threshold. Never touches running trials, and never
deletes anything whose metrics cannot be determined.

Usage:
  python experiments/no31_cora_original/clean_ckpts_by_metric.py --dry-run
  python experiments/no31_cora_original/clean_ckpts_by_metric.py --run
"""
import argparse
import json
import re
import shutil
from pathlib import Path

EXP = Path(__file__).resolve().parent
OUT = EXP / "output"
LOGS = EXP / "logs"
BEST_LINE = re.compile(r"\*\*\*auc:\s*([\d.]+)")
EARLY_STOP = "[JOINT8 EARLY STOP]"
THRESHOLD = 0.7250


def best_auc_for_log(log_path):
    pairs = [float(m) for m in BEST_LINE.findall(
        log_path.read_text(errors="replace"))]
    return max(pairs) if pairs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    freed = 0
    deleted, kept = [], []
    for ck in sorted(OUT.rglob("checkpoint_best.pth")):
        trial_dir = ck.parent.parent          # output/<PROTO>/<trial>/<stamp>/
        trial_name = trial_dir.parent.name
        proto = trial_dir.parent.parent.name
        log = LOGS / proto / f"{trial_name}.log"
        if not (LOGS / proto).is_dir():
            log = LOGS / proto
        auc = None
        if log.is_file():
            auc = best_auc_for_log(log)
        else:
            for cand in LOGS.rglob(f"{trial_name}.log"):
                auc = best_auc_for_log(cand)
                break
        if auc is None:
            kept.append((trial_name, str(ck), "NO_METRICS"))
            continue
        if auc < args.threshold:
            if args.run:
                shutil.rmtree(trial_dir)
            freed += ck.stat().st_size
            deleted.append((f"{proto}/{trial_name}", f"{auc:.4f}"))
        else:
            kept.append((trial_name, f"{auc:.4f}", "KEEP"))
    action = "DELETED" if args.run else "WOULD DELETE"
    print(f"threshold={args.threshold:.4f} {action}: {len(deleted)} trial dirs, "
          f"~{freed/1e9:.2f} GB")
    for name, auc in deleted:
        print(f"  {action}: {name} (best {auc})")
    print(f"kept {len(kept)} checkpoints:")
    for name, auc, _ in kept[:20]:
        print(f"  KEEP: {name} ({auc})")


if __name__ == "__main__":
    main()
