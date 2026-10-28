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
    import torch
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
        if gen.group_projector_mode != runtime.group_projector_mode:
            raise RuntimeError("CoRA group projector mode mismatch")
        expected_projector_modules = {
            "off": 0, "shared": 1, "group2": 2, "group4": 4, "group8": 8,
            "shared_group4": 5, "shared_trunk4": 0, "rank_group4": 0,
            "group4_linear": 4, "coarse_to_fine": 8, "fine_to_coarse": 8,
            "hierarchical_residual": 8,
        }[runtime.group_projector_mode]
        projectors = getattr(gen, "group_projectors", [])
        if len(projectors) != expected_projector_modules:
            raise RuntimeError(
                f"CoRA projector module count mismatch: mode={runtime.group_projector_mode} "
                f"got={len(projectors)} expected={expected_projector_modules}"
            )
        if runtime.group_projector_mode != "off":
            if runtime.rank_profile is not None or gen.r != 16:
                raise RuntimeError("Group projector mode must preserve rank16")
            zero_output_layers = []
            if projectors:
                zero_output_layers.extend(projector[-1] for projector in projectors)
            if runtime.group_projector_mode == "shared_trunk4":
                zero_output_layers.extend(gen.group_projector_heads)
            if runtime.group_projector_mode == "rank_group4":
                if torch.count_nonzero(gen.group_rank_delta.detach()).item() != 0:
                    raise RuntimeError("Rank-space group deltas must start at zero")
            for layer in zero_output_layers:
                if any(torch.count_nonzero(p).item() != 0 for p in (
                    layer.weight.detach(), layer.bias.detach()
                )):
                    raise RuntimeError("Group projector output must be identity-initialized")
        if runtime.graph_mode != "off":
            if (not hasattr(gen, "n20_coarse_head")
                    or not hasattr(gen, "n20_fine_head")
                    or len(gen.n20_schedule) != 8):
                raise RuntimeError("N24 graph CoRA conditioning heads missing")
            if runtime.graph_joint:
                if not hasattr(model, "graph_joint"):
                    raise RuntimeError("N24 joint graph encoder missing")
                p = model.graph_joint.id_embedding
                if p.ndim != 2 or p.size(1) != 256 or not p.requires_grad:
                    raise RuntimeError("N24 joint graph table is not trainable")
            print(f"[N24-GRAPH-PROBE-OK] mode={runtime.graph_mode} "
                  f"joint={int(runtime.graph_joint)} "
                  f"graph_aux_weight={runtime.graph_aux_weight:g}", flush=True)
        if runtime.collab_mode != "off":
            # Strict NIGHT20 fail-closed adapter validation before expensive
            # Vicuna training. CPU smoke checks initial full-generator parity.
            if runtime.group_projector_mode != "group8":
                raise RuntimeError("NIGHT20 requires group8")
            if not hasattr(gen, "n20_coarse_head") or not hasattr(gen, "n20_fine_head"):
                raise RuntimeError("NIGHT20 missing trainable QFormer conditioning heads")
            if tuple(gen.n20_coarse_head[0].weight.shape) != (64, 514):
                raise RuntimeError("NIGHT20 coarse head input mismatch")
            if tuple(gen.n20_fine_head[0].weight.shape) != (64, 513):
                raise RuntimeError("NIGHT20 fine head input mismatch")
            if len(gen.n20_schedule) != 8 or abs(gen.n20_query_cap - 0.1) > 1e-12:
                raise RuntimeError("NIGHT20 schedule/cap mismatch")
            for head in (gen.n20_coarse_head, gen.n20_fine_head):
                final = head[-1]
                if not final.weight.requires_grad or not final.bias.requires_grad:
                    raise RuntimeError("NIGHT20 heads must remain trainable")
                if final.weight.detach().count_nonzero() or final.bias.detach().count_nonzero():
                    raise RuntimeError("NIGHT20 heads must be zero-output initialized")
            print(
                f"[CORA-N20-PROBE-OK] mode={runtime.collab_mode} "
                f"schedule={gen.n20_schedule} query_cap={gen.n20_query_cap} "
                "heads=coarse[514,64,512]+fine[513,64,512] "
                "zero_output=True target_pair_leave_out=True",
                flush=True,
            )
        # Every selected target receives original CoRA's exact P(x @ A)
        # forward; unselected projections do not receive dynamic weights.
        if any(getattr(mod, "_cora_weight", "missing") != None
               for mod in runtime.targets):
            raise RuntimeError("Unexpected pre-active CoRA weights")
        if runtime.rank_profile:
            rank_text = "/".join(map(str, runtime.rank_profile))
            dynamic_desc = f"groupwise [B,4096,r_g], r_g={rank_text}"
            p_desc = f"static_P={len(pweights)} x [4096,grouped ranks {rank_text}]"
            rank_desc = ",".join(
                f"L{j*8}-{j*8+7}:{rank}"
                for j, rank in enumerate(runtime.rank_profile)
            )
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
            f"group_projectors={runtime.group_projector_mode} "
            f"group_schedule={runtime.group_schedule_description()} "
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
    if runtime.graph_joint:
        graph = runtime.model.graph_joint
        if graph.aggregation == "learned":
            weights = graph.layer_weight_summary()
            expected = 1.0 / len(weights)
            changed = max(abs(w - expected) for w in weights)
            print("[N26-LAYER-WEIGHTS-END] " + json.dumps(dict(
                layers=graph.layers,
                weights=weights,
                max_deviation_from_uniform=changed,
                positive_gradient_events=graph._n26_layer_grad_events,
            )), flush=True)
            if graph._n26_layer_grad_events == 0 or changed <= 1e-9:
                raise RuntimeError(
                    "N26 adaptive aggregation did not receive usable gradients "
                    "or softmax layer weights stayed at initialization."
                )
        amp_stats = dict(
            skipped_updates=graph._n24_amp_skip_total,
            successful_updates=graph._n24_amp_success_total,
            nonfinite_backward_events=graph._n24_bad_gradient_events,
        )
        print("[N24-AMP-SUMMARY] " + json.dumps(amp_stats), flush=True)
        if amp_stats["successful_updates"] == 0:
            raise RuntimeError("N24 graph completed training with zero successful optimizer updates")
    print("[CORA-ORIGINAL-COUNTERS] "+json.dumps(dict(runtime.counters)),flush=True)
    if runtime.collab_mode != "off":
        gen = runtime.model.cora_weight_generator
        heads = {
            name: {
                "last_weight_norm": float(head[-1].weight.detach().float().norm()),
                "last_bias_norm": float(head[-1].bias.detach().float().norm()),
            }
            for name, head in (
                ("coarse", gen.n20_coarse_head),
                ("fine", gen.n20_fine_head),
            )
        }
        print("[CORA-N20-HEAD-NORMS] " + json.dumps(heads), flush=True)
    if runtime.group_projector_mode != "off":
        gen = runtime.model.cora_weight_generator
        projector_norms = []
        for idx, projector in enumerate(getattr(gen, "group_projectors", [])):
            layer = projector[-1]
            projector_norms.append({
                "module": idx,
                "last_weight_norm": float(layer.weight.detach().float().norm()),
                "last_bias_norm": float(layer.bias.detach().float().norm()),
            })
        if runtime.group_projector_mode == "shared_trunk4":
            for idx, layer in enumerate(gen.group_projector_heads):
                projector_norms.append({
                    "module": f"head{idx}",
                    "last_weight_norm": float(layer.weight.detach().float().norm()),
                    "last_bias_norm": float(layer.bias.detach().float().norm()),
                })
        if runtime.group_projector_mode == "rank_group4":
            projector_norms.append({
                "module": "rank_delta",
                "norm": float(gen.group_rank_delta.detach().float().norm()),
            })
        print("[CORA-GROUP-PROJECTORS] "+json.dumps(projector_norms), flush=True)
    if runtime.counters["conditioned_batches"]<=0 or runtime.counters["projection_forwards"]<=0:
        raise RuntimeError("CoRA full-matrix PLoRA never actually executed")


if __name__=="__main__":
    main()
