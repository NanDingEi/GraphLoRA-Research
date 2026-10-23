#!/usr/bin/env python3
"""Exact historical NO31 CTR/KD training, with OPTIONAL added CoRA.

Control: entirely original NO31 code path. CoRA: only an initially-zero
ID-token residual on L31 Q/V; original L31 graph injection remains OFF.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def shared_fingerprint(model):
    groups = {key: [] for key in ("MF", "GNN", "Projector", "LoRA")}
    for name, param in model.named_parameters():
        if name.startswith("rec_encoder."):
            group = "MF"
        elif name.startswith("gcn."):
            group = "GNN"
        elif name.startswith("llama_proj."):
            group = "Projector"
        elif ".lora_A." in name or ".lora_B." in name:
            group = "LoRA"
        else:
            continue
        groups[group].append((name, param))
    digests = {}
    for group, named_params in groups.items():
        if not named_params:
            raise RuntimeError(f"No {group} parameters; cannot verify fairness")
        sha = hashlib.sha256()
        for name, param in sorted(named_params):
            tensor = param.detach().cpu().contiguous()
            sha.update(name.encode())
            sha.update(str(tuple(tensor.shape)).encode())
            sha.update(str(tensor.dtype).encode())
            sha.update(tensor.numpy().tobytes())
        digests[group] = {"sha256": sha.hexdigest(),
                          "trainable": sum(p.numel() for _, p in named_params
                                           if p.requires_grad)}
    return digests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--mode", choices=("no31", "cora"), required=True)
    parser.add_argument("--init-hash", required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--cora-gain", type=float, default=0.1)
    parser.add_argument("--cora-active", choices=("both", "q", "v"), default="both")
    parser.add_argument("--cora-detach-gnn", choices=("true", "false"), default="true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("NO31-CORA experiment must use CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("S2_INPUT_ALPHA", "0") not in ("0", "0.0", "0.00"):
        raise RuntimeError("Input fusion must be disabled (alpha=0)")
    if float(os.environ.get("S2_SMALL_KD", ".05")) != .05:
        raise RuntimeError("Historical NO31 requires KD=0.05")
    if int(os.environ.get("S2_NIGHT_EPOCHS", "1")) not in (1, 2):
        raise RuntimeError("Supported historical run length: 1 or 2 epochs")
    os.environ["S2_INPUT_ALPHA"] = "0.00"
    os.environ["S2_SMALL_KD"] = "0.05"
    os.environ.setdefault("S2_NIGHT_EPOCHS", "1")
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(HERE)]
    from minigpt4.models import minigpt4rec_v2 as graph

    runtime = None
    if args.mode == "cora":
        from adapter import install_cora
        runtime = install_cora(gain=args.cora_gain,
                               detach_gnn=args.cora_detach_gnn == "true",
                               active=args.cora_active)

    original = graph.MiniGPT4Rec_v2.from_config

    def inspect_model(cls, cfg):
        m = original(cfg)
        is_cora = getattr(m, "r3_adapter", None) is not None
        if is_cora != (args.mode == "cora"):
            raise RuntimeError("Model/CoRA variant mismatch")
        if m.gnn_conf_loss_weight != .05 or m.s2_small_kd != .05 or m.s2_input_alpha != 0:
            raise RuntimeError("NO31 historical supervision changed")
        if any(mod.gnn_scale != 0 for mod in m.my_linears):
            raise RuntimeError("L31 graph injection must be disabled")
        result = shared_fingerprint(m)
        fp = Path(args.init_hash).resolve()
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("x", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "groups": result}, f, indent=2)
        print("[NO31-INIT-HASH] " + json.dumps(result, sort_keys=True), flush=True)
        if args.probe_only:
            print("[NO31-PROBE-OK] mode="+args.mode, flush=True)
            raise SystemExit(0)
        return m

    graph.MiniGPT4Rec_v2.from_config = classmethod(inspect_model)
    import train_collm_mf_din as train
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(Path(args.cfg_path).resolve())]
    train.main()
    # The @record training wrapper can return from the model-init-only probe.
    # In probe mode we intentionally do NOT run any L31 forward passes:
    # Q/V execution counters must be checked ONLY after real training.
    if args.probe_only:
        return
    if runtime is not None:
        stats = dict(runtime.counters)
        print("[NO31-CORA-COUNTERS] " + json.dumps(stats), flush=True)
        for tag in (("q", "v") if args.cora_active == "both" else (args.cora_active,)):
            if stats.get(tag+"_id_forward", 0) <= 0:
                raise RuntimeError(f"CoRA branch {tag} never executed")


if __name__ == "__main__":
    main()
