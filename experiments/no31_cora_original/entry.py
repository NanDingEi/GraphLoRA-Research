#!/usr/bin/env python3
"""Original-CoRA full dynamic-LoRA-A trained under archived NO31+CTR/KD."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cfg-path",required=True)
    p.add_argument("--mode",choices=["cora_original"],default="cora_original")
    p.add_argument("--probe-only",action="store_true")
    args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("Run on GPU0 with CUDA_VISIBLE_DEVICES=0")
    if float(os.environ.get("S2_INPUT_ALPHA","0")) != 0:
        raise RuntimeError("NO31 alpha must remain zero")
    if float(os.environ.get("S2_SMALL_KD","0.05")) != 0.05:
        raise RuntimeError("NO31 KD must remain 0.05")
    os.environ["S2_INPUT_ALPHA"]="0"
    os.environ["S2_SMALL_KD"]="0.05"
    os.environ.setdefault("S2_NIGHT_EPOCHS","1")
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    from experiments.no31_cora_original.adapter import install_original_cora
    from minigpt4.models import minigpt4rec_v2 as graph
    runtime=install_original_cora()
    original_from_config=graph.MiniGPT4Rec_v2.from_config
    def build_and_verify(cls,cfg):
        model=original_from_config(cfg)
        if len(runtime.targets)!=128:
            raise RuntimeError("Original CoRA 128 projection modules missing")
        if any(float(m.gnn_scale)!=0 for m in model.my_linears):
            raise RuntimeError("NO31 L31 GNN injection accidentally enabled")
        if model.gnn_conf_loss_weight!=0.05 or model.s2_small_kd!=0.05:
            raise RuntimeError("NO31 CTR/KD changed")
        static=[n for n,p in model.named_parameters()
                if (".lora_A." in n or ".lora_B." in n) and p.requires_grad]
        if static:
            raise RuntimeError(f"Static LoRA A/B still trainable: {static[:4]}")
        pweights=[(n,p) for n,p in model.named_parameters()
                  if n.endswith("cora_lora_P.weight")]
        if len(pweights)!=128:
            raise RuntimeError(f"Found only {len(pweights)} dynamic CoRA-P projections")
        if any(p.shape!=(4096,16) or p.detach().count_nonzero()!=0 for _,p in pweights):
            raise RuntimeError("CoRA P not zero initialized in all attention projections")
        gen=model.cora_weight_generator
        if (gen.num_queries,gen.r,gen.hidden_size,gen.embedding_size)!=(4,16,4096,256):
            raise RuntimeError("Original CoRA generator architecture mismatch")
        qformer=gen.qformer.encoder.layer
        if len(qformer)!=8:
            raise RuntimeError("Expected original 8-layer QFormer")
        print(
            "[CORA-ORIGINAL-PROBE-OK] QFormer=8x8 queries=4 "
            "dynamic_A=[B,4096,16] normalized=True static_P=128 x [4096,16] "
            "zero_init=True all_tokens=True base_LoRA_AB_frozen=True NO31_CTRKD=ON",
            flush=True,
        )
        if args.probe_only:
            raise SystemExit(0)
        return model
    graph.MiniGPT4Rec_v2.from_config=classmethod(build_and_verify)
    import train_collm_mf_din as training
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(Path(args.cfg_path).resolve())]
    training.main()
    if args.probe_only:
        return
    print("[CORA-ORIGINAL-COUNTERS] "+json.dumps(dict(runtime.counters)),flush=True)
    if runtime.counters["conditioned_batches"]<=0 or runtime.counters["projection_forwards"]<=0:
        raise RuntimeError("CoRA full-matrix PLoRA never actually executed")


if __name__=="__main__":
    main()
