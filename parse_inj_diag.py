"""Parse [inj-diag] window summaries from a training log into an R/C
trajectory table and the A0 verdict statistics (early vs stable window)."""
import argparse
import re

import numpy as np

PAT = re.compile(
    r"\[inj-diag\]\[(?P<tag>\w)\] steps=(?P<s0>\d+)-(?P<s1>\d+) "
    r"R_med=(?P<rmed>[-\d.]+) R_p90=(?P<rp90>[-\d.]+) "
    r"C_med=(?P<cmed>[-\d.]+) C_p10=(?P<cp10>[-\d.]+) "
    r"\|a\|=(?P<na>[-\d.]+) \|g\|=(?P<ng>[-\d.]+)"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--early-frac", type=float, default=0.1)
    args = p.parse_args()

    rows = {t: [] for t in ("q", "v")}
    for line in open(args.log):
        m = PAT.search(line)
        if m:
            d = m.groupdict()
            rows[d["tag"]].append(
                (int(d["s1"]), float(d["rmed"]), float(d["rp90"]),
                 float(d["cmed"]), float(d["cp10"]), float(d["na"]), float(d["ng"]))
            )

    for tag, rs in rows.items():
        if not rs:
            continue
        arr = np.array(rs)
        steps, rmed, rp90, cmed, cp10 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        total = steps.max()
        early = steps <= total * args.early_frac
        late = steps >= total * 0.5
        print(f"\n=== [{tag}]  windows={len(rs)}  last_step={total} ===")
        print(f"{'step':>7s} {'R_med':>7s} {'R_p90':>7s} {'C_med':>7s} {'C_p10':>7s}")
        idx = np.linspace(0, len(rs) - 1, min(15, len(rs))).astype(int)
        for i in idx:
            print(f"{int(steps[i]):7d} {rmed[i]:7.4f} {rp90[i]:7.4f} {cmed[i]:7.4f} {cp10[i]:7.4f}")
        print(f"EARLY  (<= {args.early_frac:.0%}): R_med mean={rmed[early].mean():.4f}  "
              f"C_med mean={cmed[early].mean():.4f}  R_p90 max={rp90[early].max():.4f}")
        if late.any():
            print(f"LATE   (>= 50%):    R_med mean={rmed[late].mean():.4f}  "
                  f"C_med mean={cmed[late].mean():.4f}  R_p90 max={rp90[late].max():.4f}")
        # verdict heuristics
        ratio = rmed[early].mean() / max(rmed[late].mean(), 1e-6) if late.any() else float("nan")
        print(f"early/late R_med ratio = {ratio:.2f}  "
              f"({'EARLY SPIKE — warm-up justified' if ratio > 2 else 'NO EARLY SPIKE — warm-up not motivated' if ratio < 1.3 else 'MILD'}); "
              f"q-branch C_med early mean sign: {np.sign(cmed[early].mean()):+.0f}")


if __name__ == "__main__":
    main()
