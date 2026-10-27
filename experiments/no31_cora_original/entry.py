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
    ablation = os.environ.get("NO31_CORA_LOSS_ABLATION", "")
    loss_specs = {
        "no_ctr": (0., .05, 1e-4, 5e-4),
        "no_kd": (.05, 0., 1e-4, 5e-4),
        "no_sem": (.05, .05, 0., 5e-4),
        "no_proj": (.05, .05, 1e-4, 0.),
    }
    from experiments.no31_cora_original.night22_loss_spec import (
        LOSS_SPECS as NIGHT22_LOSS_SPECS, verify_rank410,
    )
    loss_specs.update(NIGHT22_LOSS_SPECS)
    if ablation and ablation not in loss_specs:
        raise RuntimeError(f"Unsupported NIGHT21 loss ablation: {ablation}")
    expected_kd = loss_specs[ablation][1] if ablation else .05
    if float(os.environ.get("S2_SMALL_KD", "0.05")) != expected_kd:
        raise RuntimeError(f"KD environment mismatch: expected {expected_kd:g}")
    os.environ["S2_INPUT_ALPHA"]="0"
    os.environ["S2_SMALL_KD"]=str(expected_kd)
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
        if ablation:
            expected = loss_specs[ablation]
            actual = (
                float(model.gnn_conf_loss_weight),
                float(model.s2_small_kd),
                float(model.gnn_semantic_weight),
                float(model.proj_reg_weight),
            )
            if any(abs(x - y) > 1e-12 for x,y in zip(actual, expected)):
                raise RuntimeError(
                    f"NIGHT21 {ablation} auxiliary weights do not match: "
                    f"actual={actual}, expected={expected}"
                )
            if not model.gnn_semantic_align:
                raise RuntimeError("NIGHT21 requires semantic teacher computation; "
                                   "NO_SEM must zero its loss weight only")
            if not hasattr(model, "gnn_conf_kappa"):
                raise RuntimeError("NIGHT21 kappa missing; CTR-off changed model architecture")
            if ablation.startswith("n22_"):
                verify_rank410(runtime)
                print(
                    f"[NIGHT22-LOSS-CONFIG] arm={ablation} "
                    f"CTR={actual[0]:g} KD={actual[1]:g} "
                    f"SEM={actual[2]:g} PROJ={actual[3]:g} "
                    "semantic_teacher=True grouped_rank=4/6/8/10",
                    flush=True,
                )
            else:
                if runtime.collab_mode != "off" or runtime.fine_residual_mode != "off":
                    raise RuntimeError("NIGHT21 must use historical E0, not NIGHT19/20")
                if runtime.group_projector_mode != "group8":
                    raise RuntimeError("NIGHT21 requires NIGHT17 E0 group8")
            # Presence of a loss in the forward does not prove that its
            # target module is trainable; print a minimal parameter-path audit.
            projector_trainable = sum(
                p.numel() for p in model.llama_proj.parameters() if p.requires_grad
            ) if model.llama_proj is not None else 0
            gnn_trainable = sum(
                p.numel() for p in model.gcn.parameters() if p.requires_grad
            ) if model.gcn is not None else 0
            cora_trainable = sum(
                p.numel() for p in model.cora_weight_generator.parameters()
                if p.requires_grad
            )
            print(
                f"[NIGHT21-GRAD-PATH] arm={ablation} "
                f"llama_proj_trainable={projector_trainable} "
                f"gnn_trainable={gnn_trainable} "
                f"cora_generator_trainable={cora_trainable} "
                f"kappa_trainable={model.gnn_conf_kappa.requires_grad} "
                "gnn_scale=0 input_alpha=0",
                flush=True,
            )
            print(
                f"[NIGHT21-LOSS-CONFIG] arm={ablation} "
                f"CTR={actual[0]:g} KD={actual[1]:g} "
                f"SEM={actual[2]:g} PROJ={actual[3]:g} "
                "semantic_teacher=True kappa=present collab_mode=off",
                flush=True,
            )
        elif model.gnn_conf_loss_weight!=0.05 or model.s2_small_kd!=0.05:
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
            f"NO31_CTRKD={'ABLATION' if ablation else 'ON'} "
            f"alpha={runtime.alpha} "
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
