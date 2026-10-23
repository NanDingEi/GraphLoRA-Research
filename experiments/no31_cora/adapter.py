"""Minimal collaborative Q/V LoRA on top of the ARCHIVED NO31+CTR/KD code.

Only difference from the historical NO31 model is a zero-initialized
pair-conditioned low-rank *ID-token* residual at L31. No L31 GNN feature
addition (gnn_scale=0), no change to original CTR/KD/semantic supervision.
"""
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


class PairConditionGenerator(nn.Module):
    """Same 4-query collaborative generator as the previous S2-CoRA trial."""
    def __init__(self, hidden_size, rank=8, graph_dim=256, query_dim=64, num_queries=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.cf_project = nn.Linear(graph_dim, query_dim)
        self.self_attn = nn.MultiheadAttention(query_dim, 4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(query_dim, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(nn.Linear(query_dim, 2 * query_dim), nn.GELU(),
                                nn.Linear(2 * query_dim, query_dim))
        self.rank_generator = nn.Linear(query_dim, 2 * rank)
        self.q_up = nn.Linear(rank, hidden_size, bias=False)
        self.v_up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(self, user, target_item):
        tokens = torch.stack((user.float(), target_item.float()), dim=1)
        kv = self.cf_project(tokens)
        queries = self.query.unsqueeze(0).expand(user.shape[0], -1, -1)
        q, _ = self.self_attn(queries, queries, queries, need_weights=False)
        q = self.norm1(q + queries)
        z, _ = self.cross_attn(q, kv, kv, need_weights=False)
        z = self.norm2(z + q)
        z = self.norm3(z + self.ff(z))
        return (1.0 + 0.5 * torch.tanh(self.rank_generator(z.mean(1)))).reshape(
            user.shape[0], 2, -1)


class No31CoRARuntime:
    def __init__(self, gain=0.1, detach_gnn=True, active="both"):
        if not 0 < float(gain) <= 2:
            raise ValueError(f"CoRA gain outside (0,2]: {gain}")
        if active not in ("both", "q", "v"):
            raise ValueError(f"Unknown CoRA target {active!r}")
        self.gain = float(gain)
        self.detach_gnn = bool(detach_gnn)
        self.active = active
        self.model = None
        self.counters = Counter()

    def install(self):
        from minigpt4.models import minigpt4rec_v2 as graph
        if getattr(graph, "_no31_cora_installed", False):
            raise RuntimeError("NO31 CoRA runtime installed twice")

        original_init = graph.MiniGPT4Rec_v2.__init__
        original_feedback = graph.MiniGPT4Rec_v2.compute_feedback_gcn
        state = self

        def init_patch(model, *args, **kwargs):
            original_init(model, *args, **kwargs)
            if any(float(m.gnn_scale) != 0 for m in model.my_linears):
                raise RuntimeError("NO31 CoRA forbids original L31 feature injection")
            if not (model.gnn_semantic_align and model.gnn_conf_loss_weight == 0.05
                    and model.s2_small_kd == 0.05 and model.s2_input_alpha == 0):
                raise RuntimeError("NO31 CoRA requires historical CTR=.05 KD=.05 alpha=0")
            modules = {m.module_tag: m for m in model.my_linears}
            if set(modules) != {"q", "v"} or len(model.my_linears) != 2:
                raise RuntimeError("Expected exactly two L31 LoRA Q/V modules")
            for tag, mod in modules.items():
                if mod.layer_idx != 31 or mod.gcn_layer != 31:
                    raise RuntimeError(f"{tag} was not installed at L31")
                if mod.r[mod.active_adapter] != 8:
                    raise RuntimeError("Expected LoRA rank 8")
            with torch.random.fork_rng(devices=[]):
                model.r3_adapter = PairConditionGenerator(
                    hidden_size=modules["q"].out_features, rank=8)
            model.r3_adapter.to(modules["q"].weight.device)
            for tag, mod in modules.items():
                mod._no31_cora_hook = lambda out, low, tag=tag: state.apply(tag, out, low)
            state.model = model
            state.counters["initialized"] += 1
            print("[NO31-CORA-INIT] active=%s gain=%.5f detach_gnn=%s params=%d" %
                  (state.active,state.gain,state.detach_gnn,
                   sum(p.numel() for p in model.r3_adapter.parameters())),flush=True)
            for up in (model.r3_adapter.q_up, model.r3_adapter.v_up):
                if torch.count_nonzero(up.weight.detach()) != 0:
                    raise RuntimeError("CoRA output must be identically zero initially")

        def fb_patch(model, samples, samples_encode, replaced_idx, inputs_embeds,
                     attention_mask):
            fb = original_feedback(model, samples, samples_encode, replaced_idx,
                                   inputs_embeds, attention_mask)
            if model is not state.model:
                raise RuntimeError("NO31 CoRA attached to unexpected model")
            if "token_gnn" not in fb or "token_node_ids" not in fb:
                raise RuntimeError("NO31 auxiliary CTR/KD GNN features missing")
            rows = fb["batch_indices"].long()
            positions = fb["token_positions"].long()
            ids = fb["token_node_ids"].long()
            z = fb["token_gnn"].float()
            if rows.shape != positions.shape or rows.shape != ids.shape:
                raise RuntimeError("Invalid NO31 GNN/token index alignment")
            if z.shape != (rows.numel(), 256):
                raise RuntimeError(f"Unexpected NGCF penult {z.shape}")

            batch = int(samples["UserID"].numel())
            selected_users=[]
            selected_targets=[]
            for b in range(batch):
                u = ((rows == b) & (ids == samples["UserID"][b])).nonzero(as_tuple=True)[0]
                target_id = samples["TargetItemID"][b] + int(model.num_users)
                it = ((rows == b) & (ids == target_id)).nonzero(as_tuple=True)[0]
                if u.numel() < 1 or it.numel() < 1:
                    raise RuntimeError(f"Missing ID tokens for sample {b}")
                selected_users.append(u[0])
                # Candidate is the final matching item token, not history.
                selected_targets.append(it[-1])
            idx_u = torch.stack(selected_users)
            idx_t = torch.stack(selected_targets)
            u = z[idx_u]
            item = z[idx_t]
            if state.detach_gnn:
                u = u.detach()
                item = item.detach()
            with torch.cuda.amp.autocast(enabled=False):
                strength = model.r3_adapter(u, item)
            if strength.shape != (batch, 2, 8):
                raise RuntimeError(f"Invalid CoRA conditioning shape {strength.shape}")
            by_tag = {m.module_tag:m for m in model.my_linears}
            for j, tag in enumerate(("q", "v")):
                mod = by_tag[tag]
                mod._no31_cora_payload = strength[:, j]
                mod._no31_cora_rows = rows
                mod._no31_cora_positions = positions
            state.counters["feedback"] += 1
            return fb

        graph.MiniGPT4Rec_v2.__init__ = init_patch
        graph.MiniGPT4Rec_v2.compute_feedback_gcn = fb_patch
        graph._no31_cora_installed = True
        return self

    def apply(self, tag, original, low):
        """Add only at ID-token positions; never rewrite prompt text tokens."""
        mod = next(m for m in self.model.my_linears if m.module_tag == tag)
        if mod.gcn_output is None or (self.active != "both" and tag != self.active):
            return original
        if float(mod.gnn_scale) != 0:
            raise RuntimeError("NO31 injection unexpectedly became nonzero")
        payload = getattr(mod, "_no31_cora_payload", None)
        rows = getattr(mod, "_no31_cora_rows", None)
        positions = getattr(mod, "_no31_cora_positions", None)
        if payload is None or rows is None or positions is None:
            raise RuntimeError("Missing CoRA conditioning for L31")
        if low.ndim != 3 or original.ndim != 3:
            raise RuntimeError("Unexpected L31 hidden shapes")
        if payload.shape != (low.shape[0], low.shape[-1]):
            raise RuntimeError(f"Bad CoRA rank payload shape {payload.shape}")
        if rows.numel() != positions.numel():
            raise RuntimeError("ID-token positions misaligned")
        if not rows.numel():
            return original
        # Rank-conditioned ID-token LoRA update only, sharing original LoRA-A.
        condition = low[rows, positions, :].float() * payload[rows].float()
        up = getattr(self.model.r3_adapter, tag + "_up")
        correction = F.linear(condition, up.weight.float())
        correction = (self.gain * correction).to(dtype=original.dtype)
        if not torch.isfinite(correction).all():
            raise RuntimeError("Nonfinite CoRA correction")
        offset = rows * original.shape[1] + positions
        if int(torch.unique(offset).numel()) != int(offset.numel()):
            raise RuntimeError("Duplicate CoRA ID-token positions")
        flat = original.reshape(-1, original.shape[-1])
        result = flat.index_add(0, offset, correction).reshape_as(original)
        self.counters[tag + "_id_forward"] += 1
        return result


def install_cora(gain=0.1, detach_gnn=True, active="both"):
    return No31CoRARuntime(gain=gain, detach_gnn=detach_gnn, active=active).install()
