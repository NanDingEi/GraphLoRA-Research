#!/usr/bin/env python3
"""Read prior/night4 D1 checkpoint tensors on CPU. NO GPU and NO training.

Usage:
  python audit_D1_checkpoints.py --root minigpt4/output --stamp 20261016_010634

Checkpoints are trusted files made by the user's own training jobs.
"""
import argparse
import glob
from pathlib import Path

import torch

RUNS = ["N0_S2_CONTINUE", "N1_ID_V_ONLY",
        "N2_DECISION_Q_ID_V", "N3_GATED_DECISION_Q_ID_V"]


def find_run(root, stamp, group):
    pat = str(Path(root) / ("NIGHT4_" + stamp + "_" + group + "_train")
              / "*" / "checkpoint_best_complete.pth")
    items = sorted(glob.glob(pat))
    if not items:
        pat = pat.replace("checkpoint_best_complete.pth", "checkpoint_best.pth")
        items = sorted(glob.glob(pat))
    return items[-1] if items else None


def load_trainable(path):
    # PyTorch checkpoint was produced by our own trainer (includes optimizer
    # and configuration); NEVER load untrusted pickle files.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model"]


def tensor_summary(name, state):
    tensor = state.get(name)
    if tensor is None:
        return "MISSING"
    x = tensor.detach().float().cpu()
    return ("shape={} l2={:.9g} absmax={:.9g} nonzero={}/{}"
            .format(tuple(x.shape), x.norm().item(), x.abs().max().item(),
                    torch.count_nonzero(x).item(), x.numel()))


def choose_one(state, endings):
    for suffix in endings:
        matches = [k for k in state if k.endswith(suffix)]
        if matches:
            return sorted(matches)[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="minigpt4/output")
    parser.add_argument("--stamp", default="20261016_010634")
    args = parser.parse_args()
    picked = {}
    for group in RUNS:
        ckpt = find_run(args.root, args.stamp, group)
        if not ckpt:
            print("[AUDIT]", group, "checkpoint NOT FOUND")
            continue
        sd = load_trainable(ckpt)
        picked[group] = sd
        kq = choose_one(sd, (
            "layers.31.self_attn.q_proj.lora_B.default.weight",))
        kv = choose_one(sd, (
            "layers.31.self_attn.v_proj.lora_B.default.weight",))
        print("\n[AUDIT] {} {}".format(group, ckpt), flush=True)
        print("D1 pair: " + tensor_summary("d1_pair_proj.weight", sd))
        print("Layer31 Q-B: " + (tensor_summary(kq, sd) if kq else "MISSING"))
        print("Layer31 V-B: " + (tensor_summary(kv, sd) if kv else "MISSING"))
        print("Projector: " + (
            tensor_summary(choose_one(sd, ("llama_proj.0.weight",)), sd)
            if choose_one(sd, ("llama_proj.0.weight",)) else "MISSING"))
    base = picked.get(RUNS[0])
    if base is not None:
        for g in RUNS[1:]:
            v = picked.get(g)
            if v is None:
                continue
            keys = [
                k for k in base.keys() if
                ("lora_" in k or k.startswith("llama_proj.")) and k in v
            ]
            ndiff = sum(
                1 for k in keys if not torch.equal(base[k], v[k])
            )
            print("[AUDIT] {} vs N0: {}/{} common trained tensors DIFFER"
                  .format(g, ndiff, len(keys)))
    for g in RUNS[2:]:
        v = picked.get(g)
        if v is not None:
            proj = v.get("d1_pair_proj.weight")
            if proj is None:
                print("[AUDIT][FAIL]", g, "missing trained D1 projector")
            elif not torch.count_nonzero(proj):
                print("[AUDIT][FAIL]", g, "D1 projector is EXACTLY ZERO: "
                      "the decision-Q path has never become active")
            else:
                print("[AUDIT]", g, "nonzero D1 projector; further "
                      "prediction-level activation test is required")


if __name__ == "__main__":
    main()
