#!/usr/bin/env python3
"""Compare S2 champion and target-v-gate checkpoints: CPU-only, no training.

Use with equal model architectures (GraphLoRA warm start plus optional new head).
This reports actual saved trainable tensor drift, not prediction quality.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import torch


def read_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model"), dict):
        raise ValueError(f"Missing model state in {path}")
    return ckpt["model"]


def group(key):
    if ".target_v_gate." in key:
        return "new_target_v_gate"
    if "lora_A." in key or "lora_B." in key:
        return "llm_lora"
    if key.startswith("gcn.") or ".gcn." in key:
        return "gnn"
    if "llama_proj." in key:
        return "projector"
    if "rec_encoder." in key:
        return "small_recommender"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champion", required=True)
    p.add_argument("--trained", required=True)
    args = p.parse_args()
    old = read_checkpoint(args.champion)
    new = read_checkpoint(args.trained)
    common = sorted(k for k in (old.keys() & new.keys())
                    if torch.is_tensor(old[k]) and torch.is_tensor(new[k])
                    and old[k].shape == new[k].shape)
    old_only = sorted(old.keys() - new.keys())
    new_only = sorted(new.keys() - old.keys())
    print("[DRIFT] matched saved tensor keys:", len(common), flush=True)
    print("[DRIFT] original-only:", len(old_only),
          "trained-only:", len(new_only), flush=True)
    if not common:
        raise RuntimeError("No comparable saved parameters; inspect checkpoint key naming")
    groups = defaultdict(list)
    for key in common:
        before, after = old[key].float(), new[key].float()
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise RuntimeError(f"Non-finite parameter in {key}")
        delta = (after - before).norm().item()
        ratio = delta / max(before.norm().item(), 1e-8)
        groups[group(key)].append((key, ratio, delta))
    for name in ("llm_lora", "gnn", "projector", "small_recommender", "other"):
        rows = groups.get(name, [])
        if not rows:
            print(f"[DRIFT] {name}: no matched saved tensors")
            continue
        changed = sum(1 for _, _, delta in rows if delta != 0)
        print(f"[DRIFT] {name}: {changed}/{len(rows)} tensors changed")
        for key, ratio, delta in sorted(rows, key=lambda r: r[1], reverse=True)[:3]:
            print(f"    rel_change={ratio:.6g} abs_norm={delta:.6g} key={key}")
    extra_fusion = [k for k in new if ".target_v_gate." in k]
    print("[DRIFT] target-v-gate saved parameters:", len(extra_fusion))
    for key in sorted(extra_fusion):
        value = new[key]
        if torch.is_tensor(value):
            print(f"    {key}: norm={value.float().norm().item():.6g}")
    if len(extra_fusion) == 0:
        raise RuntimeError("No V gate parameters in trained checkpoint")
    print("[DRIFT] Comparisons completed; no models were modified.")


if __name__ == "__main__":
    main()
