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
        self.rank = 16
        self.alpha = 32
        self.scaling = self.alpha / self.rank

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
            )
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

            # Freeze the original standard LoRA-A/B weights. They are not
            # used by the original CoRA forward (P @ dynamic A instead).
            for pname, param in model.llama_model_lora.named_parameters():
                if ".lora_A." in pname or ".lora_B." in pname:
                    param.requires_grad_(False)

            for name, mod, layer, proj in named:
                mod.cora_lora_P = nn.Linear(state.rank, hidden, bias=False)
                nn.init.zeros_(mod.cora_lora_P.weight)
                mod.cora_dropout = nn.Dropout(p=0.05)
                mod._cora_weight = None
                mod.forward = MethodType(state.projection_forward, mod)
                state.targets.append(mod)
            model.cora_fixed_user.requires_grad_(False)
            model.cora_fixed_item.requires_grad_(False)
            model.cora_weight_generator.to(model.cora_fixed_user.device)
            print(
                f"[CORA-ORIGINAL-INIT] rank={state.rank} alpha={state.alpha} "
                f"qformer_layers=8 qformer_queries=4 full_matrix=[B,{hidden},{state.rank}] "
                f"target_modules={len(named)} target=QKVO/all32/all_tokens "
                f"frozen_CF=MF original_GNN_losses=ON",
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
            w = model.cora_weight_generator(pair)
            if w.shape != (users.numel(), 4096, state.rank):
                raise RuntimeError(f"CoRA full LoRA-A shape invalid: {w.shape}")
            if not torch.isfinite(w).all():
                raise RuntimeError("CoRA generated non-finite weights")
            if state.targets[0]._cora_weight is not None:
                raise RuntimeError("Stale CoRA weights were not cleared")
            for mod in state.targets:
                mod._cora_weight = w
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
        self.active = False

    def projection_forward(self, module, x):
        """Exactly PLora's P(bmm(x, w)) + base W x, across ALL tokens.

        This function replaces PEFT's ordinary A/B and MyLinear forward.
        Hence GraphLoRA NO31's inactive L31 injector cannot run here.
        """
        base = F.linear(x, module.weight, bias=module.bias)
        w = module._cora_weight
        if w is None:
            if self.active:
                raise RuntimeError("CoRA conditioning missing inside LLM")
            # Other model calls (if any) outside a sampled user/item batch
            # are disallowed: strict reproduction cannot use base-only LLM.
            raise RuntimeError("Original-CoRA requires a user/item-conditioned LLM forward")
        if x.ndim != 3 or x.shape[0] != w.shape[0] or x.shape[-1] != 4096:
            raise RuntimeError(f"CoRA mismatch x={x.shape}, w={w.shape}")
        x_fp = module.cora_dropout(x).to(dtype=module.cora_lora_P.weight.dtype)
        weighted = torch.bmm(x_fp, w.to(dtype=x_fp.dtype))
        residual = module.cora_lora_P(weighted) * self.scaling
        self.counters["projection_forwards"] += 1
        return (base + residual.to(dtype=base.dtype)).to(x.dtype)


def install_original_cora():
    return OriginalCoRARuntime().install()
