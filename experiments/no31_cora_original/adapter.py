"""Original-CoRA dynamic low-rank injection atop UNCHANGED NO31 CTR/KD.

Source mechanism:
  w_(u,i) = QFormer([frozen CF user ; frozen CF target])       [B,4096,16]
  q/k/v/o: y = W x + (alpha/r) P_(layer,projection) (x @ w)  ALL tokens
  Each P is initialized to zero. No static LoRA-A/B residual or rank gate.

Original public CoRA: VanillaCreamer/CoRA, plora.py:682-709, pllama.py.
We hook the matching projection forward of the installed GraphLoRA
HuggingFace LLaMA; no dependency on a forked custom pllama.py.
"""
from collections import Counter
import os
import re
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator
from experiments.no31_cora_original.granularity_signals import build_coarse_fine_signals
from experiments.no31_cora_original.night20_features import (
    install_frozen_history, make_collaborative_features,
)


class OriginalCoRARuntime:
    def __init__(self):
        self.model = None
        self.counters = Counter()
        self.targets = []
        self.active = False
        # Optional experiment controls default to the original CoRA equation.
        self.granularity_mode = os.environ.get("NO31_CORA_GRANULARITY_MODE", "original")
        if self.granularity_mode not in ("original", "fine", "coarse_gate", "dual"):
            raise ValueError(f"Unsupported granularity mode: {self.granularity_mode}")
        self.fine_residual_mode = os.environ.get("NO31_CORA_FINE_RESIDUAL_MODE", "off")
        if self.fine_residual_mode not in (
            "off", "r8", "r16", "late8", "late4",
            "coarse_to_fine", "fine_to_coarse", "history", "user_item",
        ):
            raise ValueError(f"Unsupported fine residual mode: {self.fine_residual_mode}")
        self.collab_mode = os.environ.get("NO31_CORA_COLLAB_MODE", "off")
        if self.collab_mode not in (
            "off", "coarse_only", "fine_only", "dual_uniform", "dual_c2f",
        ):
            raise ValueError(f"Unsupported NIGHT20 collaborative mode: {self.collab_mode}")
        if self.collab_mode != "off" and (
            self.granularity_mode != "original" or self.fine_residual_mode != "off"
        ):
            raise ValueError("NIGHT20 must not activate NIGHT18/19 feature paths")
        self.rank = 16
        self.alpha = int(os.environ.get("NO31_CORA_ALPHA", "32"))
        if self.alpha not in (16, 32, 48, 49, 50, 51, 52, 56, 60, 64, 96):
            raise ValueError("Unsupported NO31_CORA_ALPHA")
        self.residual_ratio = float(os.environ.get("NO31_CORA_RESIDUAL_RATIO", "0"))
        if not 0.0 <= self.residual_ratio <= 1.0:
            raise ValueError("NO31_CORA_RESIDUAL_RATIO must be in [0, 1]")
        self.depth_profile = os.environ.get("NO31_CORA_DEPTH_PROFILE", "uniform")
        if self.depth_profile not in ("uniform", "late_linear"):
            raise ValueError("NO31_CORA_DEPTH_PROFILE must be uniform or late_linear")
        # Optional per-projection residual multipliers, e.g.
        # q_proj=0.5,k_proj=0.5,v_proj=1,o_proj=1. Default is exact parity.
        scale_text = os.environ.get(
            "NO31_CORA_PROJECTION_SCALES",
            "q_proj=1,k_proj=1,v_proj=1,o_proj=1",
        )
        self.projection_scales = {}
        for field in scale_text.split(","):
            key, sep, value = field.strip().partition("=")
            if not sep or key not in ("q_proj", "k_proj", "v_proj", "o_proj"):
                raise ValueError(f"Invalid NO31_CORA_PROJECTION_SCALES entry: {field}")
            if key in self.projection_scales:
                raise ValueError(f"Duplicate projection scale: {key}")
            scale = float(value)
            if not 0.0 <= scale <= 2.0:
                raise ValueError(f"Projection scale must be in [0,2]: {key}={scale}")
            self.projection_scales[key] = scale
        if set(self.projection_scales) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
            raise ValueError("Projection scales must define Q/K/V/O exactly once")
        self.scaling = self.alpha / self.rank
        projection_string = os.environ.get("NO31_CORA_PROJECTIONS", "q_proj,k_proj,v_proj,o_proj")
        self.projections = tuple(projection_string.split(","))
        if not self.projections or len(set(self.projections)) != len(self.projections):
            raise ValueError("Duplicate/empty CoRA target projections")
        if any(p not in ("q_proj","k_proj","v_proj","o_proj") for p in self.projections):
            raise ValueError("Unexpected CoRA target projection")
        layer_span = os.environ.get("NO31_CORA_LAYERS", "0-31")
        if layer_span not in ("0-31", "16-31"):
            raise ValueError("NO31_CORA_LAYERS must be 0-31 or 16-31")
        self.layers = tuple(range(16, 32)) if layer_span=="16-31" else tuple(range(32))
        self.input_mode = os.environ.get("NO31_CORA_INPUT", "concat")
        if self.input_mode not in ("concat", "two_tokens"):
            raise ValueError("NO31_CORA_INPUT must be concat or two_tokens")
        self.injection_mode = os.environ.get("NO31_CORA_INJECTION_MODE", "shared")
        if self.injection_mode not in (
            "shared", "typed_tokens", "interaction_tokens",
            "layer_conditioned", "token_gate",
        ):
            raise ValueError(f"Unsupported NO31_CORA_INJECTION_MODE: {self.injection_mode}")
        rank_profile_text = os.environ.get("NO31_CORA_RANK_PROFILE", "uniform")
        if rank_profile_text == "uniform":
            self.rank_profile = None
        elif rank_profile_text in ("4-6-8-10", "12-14-18-20"):
            self.rank_profile = tuple(map(int, rank_profile_text.split("-")))
            if self.layers != tuple(range(32)):
                raise ValueError("Grouped-rank CoRA requires all 32 Transformer layers")
            if self.projections != ("q_proj", "k_proj", "v_proj", "o_proj"):
                raise ValueError("Grouped-rank CoRA requires Q/K/V/O in every layer")
            if self.input_mode != "concat" or self.injection_mode != "shared":
                raise ValueError("Grouped-rank CoRA requires the original shared/concat path")
        else:
            raise ValueError(
                "NO31_CORA_RANK_PROFILE must be uniform, 4-6-8-10 or 12-14-18-20"
            )
        self.group_projector_mode = os.environ.get("NO31_CORA_GROUP_PROJECTORS", "off")
        self.group_count_by_mode = {
            "off": 0, "shared": 4, "group2": 2, "group4": 4, "group8": 8,
            "shared_group4": 4, "shared_trunk4": 4, "rank_group4": 4,
            "group4_linear": 4, "coarse_to_fine": 8, "fine_to_coarse": 8,
            "hierarchical_residual": 8,
        }
        if self.group_projector_mode not in self.group_count_by_mode:
            raise ValueError(
                f"Unsupported NO31_CORA_GROUP_PROJECTORS={self.group_projector_mode}"
            )
        self.group_count = self.group_count_by_mode[self.group_projector_mode]
        if self.collab_mode != "off" and self.group_projector_mode != "group8":
            raise ValueError("NIGHT20 requires original group8 projector architecture")
        if self.fine_residual_mode != "off" and self.group_projector_mode != "group8":
            raise ValueError("Fine residual experiments require group8 projectors")
        if self.group_projector_mode != "off" and (
            self.rank_profile is not None or self.input_mode != "concat"
            or self.injection_mode != "shared"
            or self.layers != tuple(range(32))
            or self.projections != ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            raise ValueError("Group projectors require G2 rank16, full32 QKVO, concat/shared")
        self.expected_targets = len(self.projections) * len(self.layers)

    def group_for_layer(self, layer_index):
        if self.group_projector_mode == "off":
            raise ValueError("group_for_layer is only used in group-projector modes")
        layer = int(layer_index)
        if not 0 <= layer < 32:
            raise ValueError(f"Invalid Transformer layer index: {layer}")

        mode = self.group_projector_mode
        if mode in ("coarse_to_fine", "hierarchical_residual"):
            # 2 x 8-layer groups, then 2 x 4-layer groups, then 4 x 2-layer groups.
            if layer < 8:
                return 0
            if layer < 16:
                return 1
            if layer < 20:
                return 2
            if layer < 24:
                return 3
            return 4 + (layer - 24) // 2

        if mode == "fine_to_coarse":
            # 4 x 2-layer groups, then 2 x 4-layer groups, then 2 x 8-layer groups.
            if layer < 8:
                return layer // 2
            if layer < 16:
                return 4 + (layer - 8) // 4
            if layer < 24:
                return 6
            return 7

        return min(layer // (32 // self.group_count), self.group_count - 1)

    def group_schedule_description(self):
        mode = self.group_projector_mode
        if mode in ("coarse_to_fine", "hierarchical_residual"):
            return (
                "L0-7:g0,L8-15:g1,L16-19:g2,L20-23:g3,"
                "L24-25:g4,L26-27:g5,L28-29:g6,L30-31:g7"
            )
        if mode == "fine_to_coarse":
            return (
                "L0-1:g0,L2-3:g1,L4-5:g2,L6-7:g3,"
                "L8-11:g4,L12-15:g5,L16-23:g6,L24-31:g7"
            )
        if mode == "off":
            return "off"
        group_width = 32 // self.group_count
        return ",".join(
            f"L{idx * group_width}-{(idx + 1) * group_width - 1}:g{idx}"
            for idx in range(self.group_count)
        )

    def rank_for_layer(self, layer_index):
        if self.rank_profile is None:
            return self.rank
        if not 0 <= int(layer_index) < 32:
            raise ValueError(f"Invalid Transformer layer index: {layer_index}")
        return self.rank_profile[int(layer_index) // 8]

    def fine_residual_group_scale(self, group_idx):
        """Fixed layer-group coefficient for the NIGHT19 fine residual branch."""
        if not 0 <= int(group_idx) < 8:
            raise ValueError(f"Invalid group index for fine residual: {group_idx}")
        mode = self.fine_residual_mode
        if mode in ("off", "r8", "r16", "history", "user_item"):
            return 1.0
        if mode == "late8":
            return 1.0 if group_idx >= 6 else 0.0
        if mode == "late4":
            return 1.0 if group_idx >= 7 else 0.0
        if mode == "coarse_to_fine":
            return (0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0)[group_idx]
        if mode == "fine_to_coarse":
            return (1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0)[group_idx]
        raise RuntimeError(f"Unsupported fine residual mode: {mode}")

    def apply_fine_residual(self, base, residual, group_idx):
        """Add a rank-/depth-controlled residual capped at 10% of base-A norm."""
        mode = self.fine_residual_mode
        if mode == "off":
            return base
        if mode == "r8":
            # Restrict fine updates to the latter 8 of 16 rank channels.
            residual = torch.cat(
                (torch.zeros_like(residual[..., :8]), residual[..., 8:]), dim=-1
            )
        scale = self.fine_residual_group_scale(group_idx)
        if scale == 0.0:
            return base
        # Clip the unscaled branch first, then apply the layer coefficient.
        # Otherwise clipping could cancel the intended coarse-to-fine schedule.
        base_norm = base.detach().float().norm(dim=(1, 2), keepdim=True)
        residual_norm = residual.detach().float().norm(dim=(1, 2), keepdim=True)
        factor = (0.1 * base_norm / residual_norm.clamp_min(1e-8)).clamp(max=1.0)
        residual = residual * factor.to(dtype=residual.dtype) * scale
        return base + residual

    def install(self):
        from minigpt4.models import minigpt4rec_v2 as graph
        if getattr(graph, "_original_cora_installed", False):
            raise RuntimeError("Original CoRA installed twice")
        old_init = graph.MiniGPT4Rec_v2.__init__
        old_feedback = graph.MiniGPT4Rec_v2.compute_feedback_gcn
        old_forward_v2 = graph.MiniGPT4Rec_v2.forward_v2
        old_generate_v2 = graph.MiniGPT4Rec_v2.generate_for_samples_v2
        state = self

        def init_patch(model, *args, **kwargs):
            old_init(model, *args, **kwargs)
            if not model.use_lora:
                raise RuntimeError("NO31+CoRA requires LLM PEFT wrapper")
            if any(float(m.gnn_scale) != 0 for m in model.my_linears):
                raise RuntimeError("NO31 forbids layer31 GNN feature injection")
            # Historical CoRA/E0 remains strict by default. Only NIGHT21
            # explicitly permits its four approved one-factor loss ablations;
            # never relax this check for NIGHT17/18/19/20 or general runs.
            loss_arm = os.environ.get("NO31_CORA_LOSS_ABLATION", "")
            loss_specs = {
                "no_ctr": (0.0, 0.05, 1e-4, 5e-4),
                "no_kd": (0.05, 0.0, 1e-4, 5e-4),
                "no_sem": (0.05, 0.05, 0.0, 5e-4),
                "no_proj": (0.05, 0.05, 1e-4, 0.0),
            }
            if loss_arm and loss_arm not in loss_specs:
                raise RuntimeError(
                    f"Unsupported CoRA auxiliary loss ablation: {loss_arm}"
                )
            expected = loss_specs.get(loss_arm, (0.05, 0.05, 1e-4, 5e-4))
            actual = (
                float(model.gnn_conf_loss_weight),
                float(model.s2_small_kd),
                float(model.gnn_semantic_weight),
                float(model.proj_reg_weight),
            )
            if (any(abs(x - y) > 1e-12 for x, y in zip(actual, expected))
                    or model.s2_input_alpha != 0
                    or not model.gnn_semantic_align):
                raise RuntimeError(
                    f"NO31 CoRA loss/semantic controls changed: "
                    f"arm={loss_arm or 'E0'} expected={expected} actual={actual} "
                    f"input_alpha={model.s2_input_alpha} "
                    f"semantic_align={model.gnn_semantic_align}"
                )
            if loss_arm:
                if (self.group_projector_mode != "group8" or
                        self.granularity_mode != "original" or
                        self.fine_residual_mode != "off" or
                        self.collab_mode != "off"):
                    raise RuntimeError(
                        "NIGHT21 loss ablations require historical NIGHT17 E0 "
                        "group8, without NIGHT18/19/20 feature variants"
                    )
                print(
                    f"[CORA-NIGHT21-ADAPTER-CONFIG] arm={loss_arm} "
                    f"loss_weights={actual} original_E0=True",
                    flush=True,
                )
            if model.rec_model_type != "MF":
                raise RuntimeError("Strict original-CoRA MF experiment requires MF")
            e = int(model.rec_encoder.config.embedding_size)
            hidden = int(model.llama_model.config.hidden_size)
            if e != 256 or hidden != 4096:
                raise RuntimeError(f"Unverified original CoRA input dimensions {e}, {hidden}")

            # Original CoRA freezes the pre-trained CF model. Make a frozen
            # snapshot for *only the CoRA condition*. NO31's own MF remains
            # trainable through its original recommendation/CTR/KD pathways.
            model.register_buffer(
                "cora_fixed_user",
                model.rec_encoder.user_embedding.weight.detach().float().clone(),
                persistent=False,
            )
            model.register_buffer(
                "cora_fixed_item",
                model.rec_encoder.item_embedding.weight.detach().float().clone(),
                persistent=False,
            )
            model.cora_weight_generator = PerceptualWeightsGenerator(
                num_queries=4,
                user_embedding_size=e,
                hidden_size=hidden,
                r=state.rank,
                input_mode=state.input_mode,
                injection_mode=state.injection_mode,
                rank_profile=state.rank_profile,
                group_projector_mode=state.group_projector_mode,
            )
            if state.collab_mode != "off":
                # All four NIGHT20 arms have identical extra parameter counts.
                # Save and restore the global RNG stream so base CoRA P-init
                # and training sample order remain consistent with NIGHT17 E0.
                _n20_rng = torch.get_rng_state()
                try:
                    generator = model.cora_weight_generator
                    generator.n20_coarse_head = nn.Sequential(
                        nn.Linear(2 * e + 2, 64),
                        nn.GELU(),
                        nn.Linear(64, 2 * e),
                    )
                    generator.n20_fine_head = nn.Sequential(
                        nn.Linear(2 * e + 1, 64),
                        nn.GELU(),
                        nn.Linear(64, 2 * e),
                    )
                    for head in (generator.n20_coarse_head, generator.n20_fine_head):
                        nn.init.zeros_(head[-1].weight)
                        nn.init.zeros_(head[-1].bias)
                    generator.n20_query_cap = 0.1
                    if state.collab_mode == "coarse_only":
                        generator.n20_schedule = [(1., 0.)] * 8
                    elif state.collab_mode == "fine_only":
                        generator.n20_schedule = [(0., 1.)] * 8
                    elif state.collab_mode == "dual_uniform":
                        generator.n20_schedule = [(0.5, 0.5)] * 8
                    else:
                        a = (1., 1., 0.75, 0.75, 0.25, 0.25, 0., 0.)
                        generator.n20_schedule = [(x, 1. - x) for x in a]
                finally:
                    torch.set_rng_state(_n20_rng)
                proto_path = os.environ.get("NO31_CORA_COLLAB_PROTO_PATH", "")
                train_path = os.environ.get("NO31_CORA_COLLAB_TRAIN_PATH", "")
                if not proto_path or not train_path:
                    raise RuntimeError(
                        "NIGHT20 requires both NO31_CORA_COLLAB_PROTO_PATH "
                        "and NO31_CORA_COLLAB_TRAIN_PATH"
                    )
                install_frozen_history(model, proto_path, train_path)
            if state.fine_residual_mode != "off":
                # Compact residual head: <=2.2M parameters, rather than a
                # second dense 512 -> (4096*16) projection. Zero-init makes
                # the new branch an exact no-op at initialization.
                _fine_head_rng = torch.get_rng_state()
                try:
                    model.cora_weight_generator.fine_residual_head = nn.Sequential(
                        nn.Linear(2 * e, 32),
                        nn.GELU(),
                        nn.Linear(32, hidden * state.rank),
                    )
                    nn.init.zeros_(model.cora_weight_generator.fine_residual_head[-1].weight)
                    nn.init.zeros_(model.cora_weight_generator.fine_residual_head[-1].bias)
                finally:
                    torch.set_rng_state(_fine_head_rng)
            if state.granularity_mode in ("dual", "coarse_gate"):
                # Gate 8 of 16 CoRA rank channels. Exact identity at init.
                # New params belong to the generator's existing optimizer group.
                saved_rng = torch.get_rng_state()
                try:
                    model.cora_weight_generator.fine_rank_gate = nn.Sequential(
                        nn.Linear(2 * e, 64),
                        nn.GELU(),
                        nn.Linear(64, 8),
                    )
                    nn.init.zeros_(model.cora_weight_generator.fine_rank_gate[-1].weight)
                    nn.init.zeros_(model.cora_weight_generator.fine_rank_gate[-1].bias)
                finally:
                    torch.set_rng_state(saved_rng)
            if state.injection_mode == "token_gate":
                # Keep the sampler/P-init RNG stream identical to shared CoRA.
                # The gate is zero-initialized anyway; its constructor must not
                # shift all later random draws for this architecture ablation.
                _global_cpu_rng = torch.get_rng_state()
                try:
                    model.cora_token_gate = nn.Linear(hidden, 1)
                    nn.init.zeros_(model.cora_token_gate.weight)
                    nn.init.zeros_(model.cora_token_gate.bias)
                finally:
                    torch.set_rng_state(_global_cpu_rng)
            state.model = model

            # All 32 Transformer layers, all attention Q/K/V/O modules,
            # exactly as in the original CoRA Movie config. Replace the
            # baseline static LoRA, do not add CoRA on top of it.
            named = []
            seen = set()
            for name, mod in model.llama_model_lora.named_modules():
                match = re.search(
                    r"(?:^|\.)layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
                    name,
                )
                if match is None:
                    continue
                layer, projection = int(match.group(1)), match.group(2)
                if (layer, projection) in seen:
                    raise RuntimeError(f"Duplicate CoRA attention module {layer},{projection}")
                if not hasattr(mod, "weight") or mod.weight.shape != (hidden, hidden):
                    raise RuntimeError(f"Unexpected projection shape {name}")
                seen.add((layer, projection))
                named.append((name, mod, layer, projection))
            if len(named) != 128 or len(seen) != 128:
                raise RuntimeError(f"Expected 32 layers x 4 attention projections; found {len(named)}")
            selected = [(name, mod, layer, proj) for name, mod, layer, proj in named
                        if layer in state.layers and proj in state.projections]
            if len(selected) != state.expected_targets:
                raise RuntimeError(f"Invalid CoRA target selection: {len(selected)} / {state.expected_targets}")

            # Freeze the original standard LoRA-A/B weights. They are not
            # used by the original CoRA forward (P @ dynamic A instead).
            for pname, param in model.llama_model_lora.named_parameters():
                if ".lora_A." in pname or ".lora_B." in pname:
                    param.requires_grad_(False)

            # IMPORTANT: reproduce the R0 P-initialization RNG sequence
            # even when an ablation selects fewer than 128 projections.
            # This keeps downstream projector initialization identical
            # rather than confounding architecture with a new RNG stream.
            selected_ids = {(layer, proj) for _, _, layer, proj in selected}
            for name, mod, layer, proj in named:
                p_rank = state.rank_for_layer(layer)
                if state.rank_profile is None:
                    p = nn.Linear(p_rank, hidden, bias=False)
                    nn.init.zeros_(p.weight)
                else:
                    # P is zero-initialized; do not let its rank-dependent
                    # temporary Linear initialization perturb later RNG draws.
                    _p_rng = torch.get_rng_state()
                    try:
                        p = nn.Linear(p_rank, hidden, bias=False)
                        nn.init.zeros_(p.weight)
                    finally:
                        torch.set_rng_state(_p_rng)
                if (layer, proj) not in selected_ids:
                    continue  # consume baseline RNG, but don't register P
                mod.cora_lora_P = p
                mod.cora_dropout = nn.Dropout(p=0.05)
                mod._cora_weight = None
                mod._cora_condition = None
                mod.cora_layer_index = layer
                mod.cora_projection_name = proj
                mod.cora_rank = p_rank
                mod.forward = MethodType(state.projection_forward, mod)
                state.targets.append(mod)
            model.cora_fixed_user.requires_grad_(False)
            model.cora_fixed_item.requires_grad_(False)
            model.cora_weight_generator.to(model.cora_fixed_user.device)
            if hasattr(model, "cora_token_gate"):
                model.cora_token_gate.to(model.cora_fixed_user.device)
            projection_scale_summary = ",".join(
                f"{key}={state.projection_scales[key]:g}"
                for key in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            if state.rank_profile is None:
                rank_summary = "uniform16"
                generator_shape_summary = f"[B,{hidden},16]"
            else:
                rank_text = "-".join(str(rank) for rank in state.rank_profile)
                layer_text = ",".join(
                    f"L{group*8}-{group*8+7}:{rank}"
                    for group, rank in enumerate(state.rank_profile)
                )
                rank_summary = f"{rank_text} ({layer_text})"
                generator_shape_summary = (
                    f"groupwise [B,{hidden},r_g], "
                    f"r_g={'/'.join(map(str,state.rank_profile))}"
                )
            print(
                f"[CORA-ORIGINAL-INIT] rank={state.rank} alpha={state.alpha} "
                f"qformer_layers=8 qformer_queries=4 dynamic_A={generator_shape_summary} "
                f"rank_profile={rank_summary} group_projectors={state.group_projector_mode} "
                f"granularity_mode={state.granularity_mode} "
                f"fine_residual_mode={state.fine_residual_mode} "
                f"collab_mode={state.collab_mode} "
                f"group_scaling={state.scaling:g} "
                f"target_modules={len(named)} target=QKVO/all32/all_tokens "
                f"frozen_CF=MF original_GNN_losses=ON "
                f"selected_targets={len(selected)} layers={state.layers[0]}-{state.layers[-1]} "
                f"projections={','.join(state.projections)} input_mode={state.input_mode} "
                f"injection_mode={state.injection_mode} "
                f"residual_ratio={state.residual_ratio:g} depth_profile={state.depth_profile} "
                f"projection_scales={projection_scale_summary}",
                flush=True,
            )

        def feedback_patch(model, samples, *args, **kwargs):
            if model is not state.model:
                raise RuntimeError("Original-CoRA attached to wrong model")
            if state.active:
                raise RuntimeError("Nested original-CoRA conditioning forward")
            users = samples["UserID"].long().reshape(-1)
            items = samples["TargetItemID"].long().reshape(-1)
            if (users.min() < 0 or items.min() < 0 or
                users.max() >= model.cora_fixed_user.shape[0] or
                items.max() >= model.cora_fixed_item.shape[0]):
                raise RuntimeError("Invalid CF user/item index for CoRA generator")
            pair = torch.cat(
                (model.cora_fixed_user[users], model.cora_fixed_item[items]), dim=-1
            )
            fine = None
            fine_user_item = None
            fine_target_history = None
            if (state.granularity_mode != "original"
                    or state.fine_residual_mode != "off"):
                history = samples.get("InteractedItemIDs_pad")
                if history is None:
                    raise RuntimeError("Granularity mode requires padded history item IDs")
                if state.fine_residual_mode != "off":
                    coarse, fine, fine_user_item, fine_target_history = build_coarse_fine_signals(
                        users, items, history, model.cora_fixed_user,
                        model.cora_fixed_item,
                        padding_idx=int(model.rec_encoder.padding_index),
                        return_components=True,
                    )
                else:
                    coarse, fine = build_coarse_fine_signals(
                        users, items, history, model.cora_fixed_user,
                        model.cora_fixed_item,
                        padding_idx=int(model.rec_encoder.padding_index),
                    )
                if state.granularity_mode == "fine":
                    pair = fine
                else:
                    pair = coarse

            def route_rank_channels(weight):
                if state.granularity_mode not in ("dual", "coarse_gate"):
                    return weight
                signal = fine if state.granularity_mode == "dual" else pair
                gate = 0.1 * torch.tanh(
                    model.cora_weight_generator.fine_rank_gate(signal)
                )
                factor = torch.cat(
                    (torch.ones_like(gate), 1.0 + gate), dim=-1
                )
                return weight * factor.unsqueeze(1)

            # Original implementation uses MF [u, i] 256+256, not NGCF
            # penultimate states and not a learned per-rank gate.
            if state.group_projector_mode != "off":
                collab_features = None
                if state.collab_mode != "off":
                    collab_features = make_collaborative_features(model, users, items)
                weights_by_group = model.cora_weight_generator(
                    pair, collab_features=collab_features,
                )
                if state.fine_residual_mode != "off":
                    if state.fine_residual_mode == "history":
                        residual_input = fine_target_history
                    elif state.fine_residual_mode == "user_item":
                        residual_input = fine_user_item
                    else:
                        residual_input = fine
                    if residual_input is None:
                        raise RuntimeError("Fine residual signal was not constructed")
                    fine_residual = model.cora_weight_generator.fine_residual_head(
                        residual_input
                    ).view(users.numel(), 4096, state.rank)
                    weights_by_group = {
                        group_idx: state.apply_fine_residual(weight, fine_residual, group_idx)
                        for group_idx, weight in weights_by_group.items()
                    }
                if state.granularity_mode in ("dual", "coarse_gate"):
                    weights_by_group = {
                        group: route_rank_channels(weight)
                        for group, weight in weights_by_group.items()
                    }
                if set(weights_by_group) != set(range(state.group_count)):
                    raise RuntimeError(
                        f"Group projector output ids mismatch: {sorted(weights_by_group)} "
                        f"expected={list(range(state.group_count))}"
                    )
                for group_idx, matrix in weights_by_group.items():
                    if matrix.shape != (users.numel(), 4096, 16):
                        raise RuntimeError(
                            f"Invalid group {group_idx} CoRA A shape: {matrix.shape}"
                        )
                    if not torch.isfinite(matrix).all():
                        raise RuntimeError(f"Group {group_idx} CoRA A is nonfinite")
                condition = None
                weights_by_rank = None
            elif state.rank_profile is not None:
                weights_by_rank = model.cora_weight_generator(pair)
                condition = None
            elif state.injection_mode == "layer_conditioned":
                w, condition = model.cora_weight_generator(pair, return_condition=True)
                weights_by_rank = {state.rank: w}
            else:
                w = model.cora_weight_generator(pair)
                w = route_rank_channels(w)
                condition = None
                weights_by_rank = {state.rank: w}
            if weights_by_rank is not None:
                for rank, w in weights_by_rank.items():
                    if w.shape != (users.numel(), 4096, int(rank)):
                        raise RuntimeError(
                            f"CoRA dynamic LoRA-A shape invalid for rank {rank}: {w.shape}"
                        )
                    if not torch.isfinite(w).all():
                        raise RuntimeError(f"CoRA generated non-finite weights for rank {rank}")
            if state.targets[0]._cora_weight is not None:
                raise RuntimeError("Stale CoRA weights were not cleared")
            for mod in state.targets:
                if state.group_projector_mode != "off":
                    group_idx = state.group_for_layer(mod.cora_layer_index)
                    mod._cora_weight = weights_by_group[group_idx]
                else:
                    rank = state.rank_for_layer(mod.cora_layer_index)
                    if rank not in weights_by_rank:
                        raise RuntimeError(f"Missing dynamic CoRA basis for rank {rank}")
                    mod._cora_weight = weights_by_rank[rank]
                mod._cora_condition = condition
            state.active = True
            state.counters["conditioned_batches"] += 1
            # Keep untouched NO31 GNN semantic/CTR/KD data flow. Its
            # internal early LLM teacher pass now sees the same CoRA weights.
            return old_feedback(model, samples, *args, **kwargs)

        def wrapped_forward(model, samples, *args, **kwargs):
            try:
                return old_forward_v2(model, samples, *args, **kwargs)
            finally:
                state.clear()

        def wrapped_eval(model, samples, *args, **kwargs):
            try:
                return old_generate_v2(model, samples, *args, **kwargs)
            finally:
                state.clear()

        graph.MiniGPT4Rec_v2.__init__ = init_patch
        graph.MiniGPT4Rec_v2.compute_feedback_gcn = feedback_patch
        graph.MiniGPT4Rec_v2.forward_v2 = wrapped_forward
        graph.MiniGPT4Rec_v2.generate_for_samples_v2 = wrapped_eval
        graph._original_cora_installed = True
        return self

    def clear(self):
        if self.model is None:
            return
        for mod in self.targets:
            mod._cora_weight = None
            mod._cora_condition = None
        self.active = False

    def layer_scale(self, layer_index):
        """Optional fixed depth profile; uniform is the original CoRA behavior."""
        if self.depth_profile == "uniform":
            return 1.0
        if self.depth_profile == "late_linear":
            return 0.5 + 0.5 * float(layer_index) / 31.0
        raise RuntimeError(f"Unsupported depth profile {self.depth_profile}")

    def apply_residual_controls(self, base, residual, layer_index, projection=None):
        """Apply opt-in projection/depth scaling and residual/base norm clipping."""
        scale = self.layer_scale(layer_index)
        projection_scale = self.projection_scales.get(projection, 1.0)
        scale *= projection_scale
        if scale != 1.0:
            residual = residual * scale
        if self.residual_ratio > 0.0:
            # Detach norms so clipping caps the branch magnitude without adding
            # a gradient path through the norm-based scale itself.
            base_norm = base.float().norm(dim=-1, keepdim=True).detach()
            residual_norm = residual.float().norm(dim=-1, keepdim=True).detach()
            max_norm = self.residual_ratio * base_norm
            factor = (max_norm / residual_norm.clamp_min(1e-6)).clamp(max=1.0)
            residual = residual * factor.to(dtype=residual.dtype)
        return residual

    def projection_forward(self, module, x):
        """Exactly PLora's P(bmm(x, w)) + base W x, across ALL tokens.

        Optional experimental controls act only on the CoRA residual branch;
        uniform/no clipping preserves the original equation.
        """
        base = F.linear(x, module.weight, bias=module.bias)
        w = module._cora_weight
        if w is None:
            if self.active:
                raise RuntimeError("CoRA conditioning missing inside LLM")
            # Other model calls (if any) outside a sampled user/item batch
            # are disallowed: strict reproduction cannot use base-only LLM.
            raise RuntimeError("Original-CoRA requires a user/item-conditioned LLM forward")
        if (x.ndim != 3 or x.shape[0] != w.shape[0] or x.shape[-1] != 4096
                or w.shape[-1] != module.cora_rank):
            raise RuntimeError(f"CoRA mismatch x={x.shape}, w={w.shape}, rank={module.cora_rank}")
        x_fp = module.cora_dropout(x).to(dtype=module.cora_lora_P.weight.dtype)
        if self.injection_mode == "layer_conditioned":
            condition = module._cora_condition
            if condition is None:
                raise RuntimeError("Layer-conditioned CoRA is missing its pair condition")
            layer_ids = torch.full(
                (condition.shape[0],), int(module.cora_layer_index),
                dtype=torch.long, device=condition.device,
            )
            layer_embedding = self.model.cora_weight_generator.layer_embedding(layer_ids)
            layer_input = torch.cat((condition, layer_embedding), dim=-1)
            rank_modulation = torch.tanh(
                self.model.cora_weight_generator.layer_modulator(layer_input)
            )
            w = w * (1.0 + rank_modulation.unsqueeze(1).to(dtype=w.dtype))
        weighted = torch.bmm(x_fp, w.to(dtype=x_fp.dtype))
        residual = module.cora_lora_P(weighted) * self.scaling
        if self.injection_mode == "token_gate":
            token_gate = 2.0 * torch.sigmoid(self.model.cora_token_gate(x.float()))
            residual = residual * token_gate.to(dtype=residual.dtype)
        residual = self.apply_residual_controls(
            base, residual, module.cora_layer_index, module.cora_projection_name
        )
        self.counters["projection_forwards"] += 1
        return (base + residual.to(dtype=base.dtype)).to(x.dtype)


def install_original_cora():
    return OriginalCoRARuntime().install()
