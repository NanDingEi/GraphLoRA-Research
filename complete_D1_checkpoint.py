#!/usr/bin/env python3
"""Attach frozen S2 MF/NGCF weights to a D1 fine-tune checkpoint.

D1 training snapshots contain trainable parameters only. This tool adds
exactly the original frozen collaborative weights for standalone evaluation.
"""
import argparse
from pathlib import Path
import torch


def complete_checkpoint(s2_path, d1_path, output_path):
    s2 = torch.load(str(s2_path), map_location="cpu")
    d1 = torch.load(str(d1_path), map_location="cpu")
    a, b = s2["model"], d1["model"]
    frozen = ("rec_encoder.", "gcn.")
    keys = [k for k in a if k.startswith(frozen)]
    if not any(k.startswith("rec_encoder.") for k in keys):
        raise ValueError("Original S2 checkpoint has no rec_encoder weights")
    if not any(k.startswith("gcn.") for k in keys):
        raise ValueError("Original S2 checkpoint has no gcn weights")
    conflicts = [k for k in keys if k in b and
                 (a[k].shape != b[k].shape or not torch.equal(a[k], b[k]))]
    if conflicts:
        raise ValueError("D1 checkpoint changed frozen parameters: " + str(conflicts[:5]))
    combined = dict(b)
    combined.update({k: a[k] for k in keys})
    d1["model"] = combined
    output_path = Path(output_path)
    if output_path.resolve() == Path(d1_path).resolve():
        raise ValueError("Do not overwrite original train checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d1, str(output_path))
    check = torch.load(str(output_path), map_location="cpu")
    for prefix in frozen:
        assert any(k.startswith(prefix) for k in check["model"])
    print("[D1] self-contained test checkpoint:", str(output_path), flush=True)
    print("[D1] appended unchanged frozen S2 tensors:", len(keys), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s2", required=True)
    p.add_argument("--d1", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    complete_checkpoint(args.s2, args.d1, args.out)


if __name__ == "__main__":
    main()
