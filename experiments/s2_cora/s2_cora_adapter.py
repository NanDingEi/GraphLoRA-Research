#!/usr/bin/env python3
"""Minimal CoRA-on-S2 adapter. No R4/R5 hooks, no changes to S2 optimizer.

Source S2: d818fa2943ede4e9a810b032a3422e94d0c77ed0.
CoRA formulation is the R3 CollaborativeQueryGenerator:
a second initially-zero B_c diag(s(u,i)) A low-rank delta on L31 Q and V.
The original S2 BA + NGCF rank injection remains trainable and intact.
"""
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F


class CollaborativeQueryGenerator(nn.Module):
    def __init__(self, hidden_size, graph_dim=256, rank=8,
                 query_dim=64, queries=4):
        super().__init__()
        self.rank = int(rank)
        self.query = nn.Parameter(torch.randn(queries, query_dim) * 0.02)
        self.cf_project = nn.Linear(graph_dim, query_dim)
        self.self_attn = nn.MultiheadAttention(query_dim, 4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(query_dim, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(nn.Linear(query_dim, 2*query_dim),
                                nn.GELU(), nn.Linear(2*query_dim, query_dim))
        self.rank_generator = nn.Linear(query_dim, 2*self.rank)
        self.q_up = nn.Linear(self.rank, hidden_size, bias=False)
        self.v_up = nn.Linear(self.rank, hidden_size, bias=False)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(self, user, item):
        tokens = torch.stack((user.float(), item.float()), dim=1)
        kv = self.cf_project(tokens)
        queries = self.query.unsqueeze(0).expand(user.size(0), -1, -1)
        q, _ = self.self_attn(queries, queries, queries, need_weights=False)
        q = self.norm1(q + queries)
        z, _ = self.cross_attn(q, kv, kv, need_weights=False)
        z = self.norm2(z + q)
        z = self.norm3(z + self.ff(z))
        return (1.0 + 0.5 * torch.tanh(self.rank_generator(z.mean(dim=1)))).view(
            -1, 2, self.rank)


def pick_pair_features(fb, samples, model):
    for key in ("sub_node_ids", "gnn_penult"):
        if key not in fb:
            raise RuntimeError(f"Missing S2 CoRA GNN feature: {key}")
    nodes = fb["sub_node_ids"].long()
    penult = fb["gnn_penult"].float()
    if nodes.ndim != 1 or nodes.numel() == 0 or penult.shape != (nodes.numel(), 256):
        raise RuntimeError("S2 CoRA graph features have wrong dimensions")
    if not bool((nodes[1:] > nodes[:-1]).all()):
        raise RuntimeError("Subgraph nodes must be sorted and unique")
    if model.training and not model.train_gnn_with_rec_loss:
        penult = penult.detach()

    def pick(global_ids):
        ids = global_ids.long().contiguous()
        idx = torch.searchsorted(nodes, ids)
        safe = idx.clamp(max=nodes.numel()-1)
        if not bool(((idx < nodes.numel()) & (nodes[safe] == ids)).all()):
            raise RuntimeError("Missing CoRA user/target item in sampled graph")
        return penult[safe]

    user = pick(samples["UserID"])
    item = pick(samples["TargetItemID"] + int(model.num_users))
    if user.shape != item.shape or user.ndim != 2:
        raise RuntimeError("Invalid S2 CoRA user/target features")
    return user, item


class S2CoRARuntime:
    def __init__(self, active="both", gain=1.0):
        if active not in ("both", "q", "v"):
            raise ValueError(f"Invalid CoRA branch {active!r}")
        if not 0 < float(gain) <= 5:
            raise ValueError(f"Invalid CoRA gain: {gain}")
        self.active = active
        self.gain = float(gain)
        self.counters = Counter()
        self.model = None

    def install(self):
        from minigpt4.models import minigpt4rec_v2 as graph
        if hasattr(graph, "_s2_cora_runtime_installed"):
            raise RuntimeError("S2 CoRA installed twice")
        original_init = graph.MiniGPT4Rec_v2.__init__
        original_feedback = graph.MiniGPT4Rec_v2.compute_feedback_gcn
        original_forward = graph.MyLinear.forward
        runtime = self

        def init_patch(model, *args, **kwargs):
            original_init(model, *args, **kwargs)
            modules = {m.module_tag: m for m in model.my_linears}
            if set(modules) != {"q", "v"} or len(model.my_linears) != 2:
                raise RuntimeError("CoRA expects exactly L31 Q/V MyLinear modules")
            ranks = []
            for tag, mod in modules.items():
                if mod.layer_idx != 31 or mod.gcn_layer != 31:
                    raise RuntimeError(f"CoRA requires L31 only, got {tag} L{mod.layer_idx}")
                if mod.rel_pair_gate is not None:
                    raise RuntimeError("Unexpected S2 relation gate active")
                ranks.append(mod.lora_A[mod.active_adapter].out_features)
            if len(set(ranks)) != 1 or ranks[0] != 8:
                raise RuntimeError(f"CoRA requires LoRA rank 8, got {ranks}")
            # Existing S2 torch RNG stream is preserved exactly.
            with torch.random.fork_rng(devices=[]):
                model.r3_adapter = CollaborativeQueryGenerator(
                    hidden_size=modules["q"].out_features, rank=ranks[0])
            model.r3_adapter.to(modules["q"].weight.device)
            model._s2_cora_need_view = False
            runtime.model = model
            runtime.counters["model_created"] += 1
            print("[S2-CORA-MODEL] trainable=%d" %
                  sum(p.numel() for p in model.r3_adapter.parameters()), flush=True)

        def feedback_patch(model, samples, samples_encode, replaced_idx,
                           inputs_embeds, attention_mask):
            if model is not runtime.model:
                raise RuntimeError("S2 CoRA is bound to another model instance")
            model._s2_cora_need_view = True
            try:
                fb = original_feedback(model, samples, samples_encode,
                                       replaced_idx, inputs_embeds, attention_mask)
            finally:
                model._s2_cora_need_view = False
            u, item = pick_pair_features(fb, samples, model)
            with torch.cuda.amp.autocast(enabled=False):
                strengths = model.r3_adapter(u.float(), item.float())
            if strengths.shape != (u.shape[0], 2, 8):
                raise RuntimeError(f"Wrong CoRA pair-conditioned shape {strengths.shape}")
            modules = {m.module_tag: m for m in model.my_linears}
            modules["q"]._s2_cora_payload = strengths[:, 0]
            modules["v"]._s2_cora_payload = strengths[:, 1]
            runtime.counters["feedback"] += 1
            runtime.counters["pairs"] += int(u.shape[0])
            return fb

        def forward_patch(mod, x):
            if mod.layer_idx != 31 or mod.module_tag not in ("q", "v"):
                return original_forward(mod, x)
            if mod.gcn_output is None:
                return original_forward(mod, x)
            if runtime.active != "both" and mod.module_tag != runtime.active:
                return original_forward(mod, x)
            if not hasattr(mod, "_s2_cora_payload"):
                raise RuntimeError("CoRA payload not installed before L31 forward")
            # Reproduce original S2 MyLinear BA+GNN path, then add the
            # independent zero-initialized B_c diag(s_ui) A correction.
            from minigpt4.models.minigpt4rec_v2 import transpose
            prev_dtype = x.dtype
            result = F.linear(x, transpose(mod.weight, mod.fan_in_fan_out),
                              bias=mod.bias)
            name = mod.active_adapter
            if (name not in mod.lora_A or mod.disable_adapters or mod.merged or
                    mod.r[name] <= 0 or mod.rel_pair_gate is not None):
                raise RuntimeError("CoRA expects active, unmerged S2 LoRA")
            x_low = mod.lora_dropout[name](x.to(mod.lora_A[name].weight.dtype))
            low = mod.lora_A[name](x_low)
            valid = mod.local_node_ids != -1
            if bool(valid.any()):
                rows = mod.batch_indices[valid]
                positions = mod.token_positions[valid]
                local = mod.local_node_ids[valid]
                if mod.warmup_factor == 1.0:
                    low[rows, positions] = (
                        low[rows, positions] * mod.lora_scale
                        + mod.gcn_output[local] * mod.gnn_scale)
                else:
                    low[rows, positions] = (
                        low[rows, positions] * mod.lora_scale
                        + mod.gcn_output[local] * (mod.gnn_scale * mod.warmup_factor))
            payload = mod._s2_cora_payload
            if payload.shape != (low.shape[0], low.shape[-1]):
                raise RuntimeError(f"CoRA payload mismatch {payload.shape}, low {low.shape}")
            s2_delta = mod.lora_B[name](low)
            condition = low.float() * payload.float().unsqueeze(1)
            up = getattr(runtime.model.r3_adapter, mod.module_tag + "_up")
            delta = F.linear(condition, up.weight.float()).to(s2_delta.dtype)
            result += s2_delta + (runtime.gain * delta)
            runtime.counters[mod.module_tag+"_cora_forward"] += 1
            return result.to(prev_dtype)

        graph.MiniGPT4Rec_v2.__init__ = init_patch
        graph.MiniGPT4Rec_v2.compute_feedback_gcn = feedback_patch
        graph.MyLinear.forward = forward_patch
        graph._s2_cora_runtime_installed = True
        return self


def install_cora(active="both", gain=1.0):
    return S2CoRARuntime(active=active, gain=gain).install()
