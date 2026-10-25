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


class OriginalCoRARuntime:
    def __init__(self):
        self.model = None
        self.counters = Counter()
        self.targets = []
        self.active = False
        # Optional experiment controls default to the original CoRA equation.
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
        self.expected_targets = len(self.projections) * len(self.layers)

    def rank_for_layer(self, layer_index):
        if self.rank_profile is None:
            return self.rank
        if not 0 <= int(layer_index) < 32:
            raise ValueError(f"Invalid Transformer layer index: {layer_index}")
        return self.rank_profile[int(layer_index) // 8]

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
            if not (model.gnn_conf_loss_weight == 0.05 and
                    model.s2_small_kd == 0.05 and model.s2_input_alpha == 0 and
                    model.gnn_semantic_align):
                raise RuntimeError("NO31 CTR=.05 KD=.05 alpha=0 semantic controls changed")
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
            )
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
                f"rank_profile={rank_summary} group_scaling={state.scaling:g} "
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
            # Original implementation uses MF [u, i] 256+256, not NGCF
            # penultimate states and not a learned per-rank gate.
            if state.rank_profile is not None:
                weights_by_rank = model.cora_weight_generator(pair)
                condition = None
            elif state.injection_mode == "layer_conditioned":
                w, condition = model.cora_weight_generator(pair, return_condition=True)
                weights_by_rank = {state.rank: w}
            else:
                w = model.cora_weight_generator(pair)
                condition = None
                weights_by_rank = {state.rank: w}
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
