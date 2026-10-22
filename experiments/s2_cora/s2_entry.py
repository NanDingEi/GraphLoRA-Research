#!/usr/bin/env python3
"""Train the archived S2 GraphLoRA baseline or minimal CoRA on the SAME runtime.

S2 base commit d818fa2943ede4e9a810b032a3422e94d0c77ed0.
This entrypoint does NOT replace optimizer, scheduler, datasets or S2 model.
The baseline installs no CoRA hooks whatsoever.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def fingerprint(model):
    groups = {k: [] for k in ("MF", "GNN", "Projector", "LoRA")}
    for name, p in model.named_parameters():
        if name.startswith("rec_encoder."):
            key = "MF"
        elif name.startswith("gcn."):
            key = "GNN"
        elif name.startswith("llama_proj."):
            key = "Projector"
        elif ".lora_A." in name or ".lora_B." in name:
            key = "LoRA"
        else:
            continue
        groups[key].append((name, p))
    result = {}
    for group, pairs in groups.items():
        if not pairs:
            raise RuntimeError(f"Missing {group} model parameters")
        h = hashlib.sha256()
        for name, p in sorted(pairs):
            tensor = p.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(str(tensor.dtype).encode())
            h.update(tensor.numpy().tobytes())
        result[group] = {"sha256": h.hexdigest(),
                         "tensor_count": len(pairs),
                         "trainable": sum(p.numel() for _, p in pairs if p.requires_grad)}
    return result


def main():
    ap = argparse.ArgumentParser(description="Archived S2 baseline and CoRA")
    ap.add_argument("--cfg-path", required=True)
    ap.add_argument("--mode", choices=("baseline", "cora"), required=True)
    ap.add_argument("--fingerprint", required=True)
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(HERE)]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Use CUDA_VISIBLE_DEVICES=0")
    from minigpt4.models import minigpt4rec_v2 as graph

    runtime = None
    if args.mode == "cora":
        from s2_cora_adapter import install_cora
        runtime = install_cora()

    original = graph.MiniGPT4Rec_v2.from_config

    def instrumented(cls, cfg):
        model = original(cfg)
        expected = args.mode == "cora"
        if (getattr(model, "r3_adapter", None) is not None) != expected:
            raise RuntimeError("CoRA attachment mismatch")
        if expected and not all(
                not bool(torch.any(param.detach() != 0))
                for param in (model.r3_adapter.q_up.weight,
                              model.r3_adapter.v_up.weight)):
            raise RuntimeError("CoRA must be exactly zero-output at init")
        state = fingerprint(model)
        dest = Path(args.fingerprint).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(f"Won't overwrite init fingerprints: {dest}")
        dest.write_text(json.dumps({"mode": args.mode, "source": "S2 d818fa2943",
                                    "groups": state}, indent=2), encoding="utf-8")
        print("[S2-INIT-HASH] "+json.dumps(state, sort_keys=True), flush=True)
        return model

    import torch
    graph.MiniGPT4Rec_v2.from_config = classmethod(instrumented)
    import train_collm_mf_din as train
    if args.probe_only:
        class ProbeRunner:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]
            def train(self):
                print("[S2-PROBE-OK] "+args.mode, flush=True)
        train.get_runner_class = lambda cfg: ProbeRunner

    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(Path(args.cfg_path).resolve())]
    try:
        train.main()
        if runtime is not None:
            count = dict(runtime.counters)
            print("[S2-CORA-COUNTERS] "+json.dumps(count, sort_keys=True), flush=True)
            if not args.probe_only:
                for key in ("q_cora_forward", "v_cora_forward"):
                    if count.get(key, 0) <= 0:
                        raise RuntimeError(f"CoRA forward never executed: {key}")
    finally:
        if runtime is not None and not args.probe_only:
            print("[S2-CORA-END-COUNTERS] "+
                  json.dumps(dict(runtime.counters), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
