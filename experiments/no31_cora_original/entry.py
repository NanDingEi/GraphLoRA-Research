#!/usr/bin/env python3
"""Original-CoRA full dynamic-LoRA-A trained under archived NO31+CTR/KD."""
import argparse
import json
import os
import re
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
        if len(runtime.targets)!=runtime.expected_targets:
            raise RuntimeError(
                f"CoRA targets mismatch: {len(runtime.targets)} "
                f"expected {runtime.expected_targets}"
            )
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
        if len(pweights)!=runtime.expected_targets:
            raise RuntimeError(
                f"Found {len(pweights)} dynamic CoRA-P projections, "
                f"expected {runtime.expected_targets}"
            )
        for name, param in pweights:
            match = re.search(r"layers\.(\d+)\.self_attn", name)
            if match is None:
                raise RuntimeError(f"Cannot identify Transformer layer for CoRA-P: {name}")
            layer = int(match.group(1))
            expected_rank = runtime.rank_for_layer(layer)
            if param.shape != (4096, expected_rank) or param.detach().count_nonzero() != 0:
                raise RuntimeError(
                    f"CoRA P shape/init mismatch at layer {layer}: "
                    f"got {tuple(param.shape)}, expected {(4096, expected_rank)} zero-init"
                )
        gen=model.cora_weight_generator
        expected_generator_rank = max(runtime.rank_profile) if runtime.rank_profile else 16
        if (gen.num_queries,gen.r,gen.hidden_size,gen.embedding_size)!=(4,expected_generator_rank,4096,256):
            raise RuntimeError("Original CoRA generator architecture mismatch")
        if gen.rank_profile != runtime.rank_profile:
            raise RuntimeError("CoRA generator rank profile mismatch")
        qformer=gen.qformer.encoder.layer
        if len(qformer)!=8:
            raise RuntimeError("Expected original 8-layer QFormer")
        if gen.input_mode != runtime.input_mode:
            raise RuntimeError("CoRA input mode mismatch")
        if gen.injection_mode != runtime.injection_mode:
            raise RuntimeError("CoRA injection mode mismatch")
        # Every selected target receives original CoRA's exact P(x @ A)
        # forward; unselected projections do not receive dynamic weights.
        if any(getattr(mod, "_cora_weight", "missing") != None
               for mod in runtime.targets):
            raise RuntimeError("Unexpected pre-active CoRA weights")
        if runtime.rank_profile:
            dynamic_desc = "groupwise [B,4096,r_g], r_g=4/6/8/10"
            p_desc = f"static_P={len(pweights)} x [4096,grouped ranks 4/6/8/10]"
            rank_desc = "L0-7:4,L8-15:6,L16-23:8,L24-31:10"
        else:
            dynamic_desc = "[B,4096,16]"
            p_desc = f"static_P={len(pweights)} x [4096,16]"
            rank_desc = "uniform16"
        print(
            "[CORA-ORIGINAL-PROBE-OK] QFormer=8x8 queries=4 "
            f"dynamic_A={dynamic_desc} normalized=True "
            f"{p_desc} rank_profile={rank_desc} group_scaling={runtime.scaling:g} "
            f"zero_init=True all_tokens=True base_LoRA_AB_frozen=True "
            f"NO31_CTRKD=ON alpha={runtime.alpha} "
            f"layers={runtime.layers[0]}-{runtime.layers[-1]} "
            f"projections={','.join(runtime.projections)} input={runtime.input_mode} "
            f"injection_mode={runtime.injection_mode} "
            f"residual_ratio={runtime.residual_ratio:g} depth_profile={runtime.depth_profile}",
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
