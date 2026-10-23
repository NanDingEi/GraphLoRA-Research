"""CoRA on the EXACT historical S2-NO31 CTR/KD training model.

This module installs only one optional *LLM-side* conditional low-rank
adapter. Historical NO31 source + CTR/KD/semantic alignment stay unchanged.

Default configuration:
  - no original GNN LoRA injection (gnn_scale=0, enforced by historical model)
  - no GNN->input fusion (S2_INPUT_ALPHA=0)
  - GNN trained by original CTR=.05, KD=.05 and semantic alignment
  - conditional adapter operates on matched ID-token positions, not all tokens
  - conditioning source: user and target-item 256-D NGCF penultimate embeddings
  - CoRA->NGCF gradient OFF by default: only historical CTR/KD/semantic train GNN
  - separate zero-initialized Q/V up projections; residual gain=.1
  - original S2 LoRA forward is called VERBATIM before adding the residual
"""
from collections import Counter
import json

import torch
from torch import nn
from torch.nn import functional as F


class NO31ConditionalAdapter(nn.Module):
    def __init__(self, hidden_dim, rank=8, graph_dim=256, query_dim=64):
        super().__init__()
        self.rank = int(rank)
        self.queries = nn.Parameter(torch.randn(4, query_dim) * 0.02)
        self.graph_proj = nn.Linear(graph_dim, query_dim)
        self.self_attn = nn.MultiheadAttention(query_dim, 4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(query_dim, 4, batch_first=True)
        self.ln1 = nn.LayerNorm(query_dim)
        self.ln2 = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(nn.Linear(query_dim, 2 * query_dim),
                                nn.GELU(), nn.Linear(2 * query_dim, query_dim))
        self.ln3 = nn.LayerNorm(query_dim)
        self.to_rank = nn.Linear(query_dim, 2 * self.rank)
        self.q_up = nn.Linear(self.rank, hidden_dim, bias=False)
        self.v_up = nn.Linear(self.rank, hidden_dim, bias=False)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(self, user, target):
        pair = torch.stack((user.float(), target.float()), dim=1)
        projected = self.graph_proj(pair)
        queries = self.queries.unsqueeze(0).expand(pair.size(0), -1, -1)
        self_out, _ = self.self_attn(queries, queries, queries, need_weights=False)
        q = self.ln1(queries + self_out)
        cross, _ = self.cross_attn(q, projected, projected, need_weights=False)
        h = self.ln2(q + cross)
        h = self.ln3(h + self.ff(h))
        return (1.0 + 0.5 * torch.tanh(self.to_rank(h.mean(dim=1)))).reshape(
            pair.size(0), 2, self.rank)


def exact_pair_features(samples, fb, num_users, *, detach=True):
    """Select ACTUAL target item, not the last historical item."""
    rows = fb["batch_indices"].long()
    ids = fb["token_node_ids"].long()
    local = fb["local_node_ids"].long()
    features = fb["token_gnn"].float()
    bsz = int(samples["UserID"].numel())
    if rows.ndim != 1 or ids.shape != rows.shape or local.shape != rows.shape:
        raise RuntimeError("NO31 graph token alignment failed")
    if features.shape != (rows.numel(), 256):
        raise RuntimeError("NO31 expected 256-D NGCF penultimate token features")
    users = samples["UserID"].long()
    targets = samples["TargetItemID"].long() + int(num_users)
    uemb, temb = [], []
    for b in range(bsz):
        valid = (rows == b) & (local >= 0)
        user_indices = torch.nonzero(valid & (ids == users[b]), as_tuple=True)[0]
        target_indices = torch.nonzero(valid & (ids == targets[b]), as_tuple=True)[0]
        if user_indices.numel() < 1 or target_indices.numel() < 1:
            raise RuntimeError(f"NO31 CoRA user/target item missing for sample {b}")
        uemb.append(features[user_indices[0]])
        temb.append(features[target_indices[-1]])
    user_emb=torch.stack(uemb,dim=0)
    target_emb=torch.stack(temb,dim=0)
    if detach:
        user_emb=user_emb.detach()
        target_emb=target_emb.detach()
    return user_emb,target_emb


class NO31CoRARuntime:
    def __init__(self, gain=.1, detach_gnn=True):
        self.gain = float(gain)
        if not 0 < self.gain <= 1:
            raise ValueError("CoRA gain must be in (0,1]")
        self.detach_gnn=bool(detach_gnn)
        self.model=None
        self.counters=Counter()

    def install(self):
        from minigpt4.models import minigpt4rec_v2 as g
        if getattr(g,"_no31_cora_installed",False):
            raise RuntimeError("NO31 CoRA runtime installed twice")
        init0=g.MiniGPT4Rec_v2.__init__
        feedback0=g.MiniGPT4Rec_v2.compute_feedback_gcn
        forward0=g.MyLinear.forward
        train0=g.MiniGPT4Rec_v2.forward_v2
        eval0=g.MiniGPT4Rec_v2.generate_for_samples_v2
        rt=self

        def patched_init(model,*args,**kwargs):
            init0(model,*args,**kwargs)
            if model.s2_input_alpha != 0 or model.s2_small_kd != .05:
                raise RuntimeError("Expected historical NO31 alpha=0 KD=.05")
            if abs(float(model.gnn_conf_loss_weight)-.05)>1e-9:
                raise RuntimeError("Expected historical NO31 CTR=.05")
            modules={m.module_tag:m for m in model.my_linears}
            if set(modules)!={"q","v"} or len(model.my_linears)!=2:
                raise RuntimeError("CoRA needs original S2 L31 Q/V MyLinear")
            for tag,mod in modules.items():
                if mod.layer_idx!=31 or mod.gcn_layer!=31 or float(mod.gnn_scale)!=0:
                    raise RuntimeError(f"NO31 forbids original L31 GNN injection: {tag}")
                if int(mod.r[mod.active_adapter])!=8:
                    raise RuntimeError("Expected S2 LoRA rank 8")
            with torch.random.fork_rng(devices=[]):
                model.no31_cora=NO31ConditionalAdapter(modules["q"].out_features,rank=8)
            model.no31_cora.to(modules["q"].weight.device)
            rt.model=model
            rt.counters["models"]=1
            assert torch.count_nonzero(model.no31_cora.q_up.weight)==0
            assert torch.count_nonzero(model.no31_cora.v_up.weight)==0
            print(f"[NO31-CORA-MODEL] CTR=.05 KD=.05 alpha=0 gnn_scale=0 "
                  f"gain={rt.gain} detach_gnn={rt.detach_gnn} "
                  f"adapter_trainable={sum(p.numel() for p in model.no31_cora.parameters())}",
                  flush=True)

        def feedback_patch(model,samples,samples_encode,replaced_idx,
                           inputs_embeds,attention_mask):
            fb=feedback0(model,samples,samples_encode,replaced_idx,
                         inputs_embeds,attention_mask)
            if model is not rt.model:
                raise RuntimeError("Wrong NO31 CoRA model instance")
            if "token_gnn" not in fb or "token_node_ids" not in fb:
                raise RuntimeError("NO31 CTR/KD must expose NGCF penultimate features")
            u,i=exact_pair_features(samples,fb,model.num_users,detach=rt.detach_gnn)
            with torch.cuda.amp.autocast(enabled=False):
                factors=model.no31_cora(u.float(),i.float())
            if factors.shape!=(u.shape[0],2,8):
                raise RuntimeError("Wrong CoRA rank factor dimensions")
            for idx,tag in enumerate(("q","v")):
                mod=next(m for m in model.my_linears if m.module_tag==tag)
                mod._no31_cora_rank=factors[:,idx,:]
            rt.counters["feedback"]+=1
            rt.counters["pairs"]+=u.shape[0]
            return fb

        def lora_forward_patch(mod,x):
            # The historical S2+NO31 LoRA output is computed UNCHANGED.
            result=forward0(mod,x)
            if mod.layer_idx!=31 or mod.module_tag not in ("q","v"):
                return result
            if mod.gcn_output is None:
                return result  # detached early semantic-teacher pass
            if mod.batch_indices is None or mod.local_node_ids is None:
                raise RuntimeError("Missing original S2 ID-token selection")
            payload=getattr(mod,"_no31_cora_rank",None)
            if payload is None:
                raise RuntimeError("Missing NO31 CoRA user/item context")
            mask=mod.local_node_ids>=0
            if not bool(mask.any()):
                return result
            row=mod.batch_indices[mask].long()
            pos=mod.token_positions[mask].long()
            adapter=mod.active_adapter
            if payload.size(0)!=result.size(0):
                raise RuntimeError("CoRA batch mismatch")
            # No second dropout call (which would perturb S2 RNG state).
            # Additional LoRA-A projection is computed ONLY at ID tokens.
            a=mod.lora_A[adapter]
            positions=x[row,pos].to(dtype=a.weight.dtype)
            with torch.cuda.amp.autocast(enabled=False):
                low=F.linear(positions.float(),a.weight.float())
                weighted=low*payload[row].float()
                up=getattr(rt.model.no31_cora,mod.module_tag+"_up")
                delta=F.linear(weighted,up.weight.float())
            res=result.clone()
            res[row,pos]=res[row,pos]+(delta*rt.gain).to(res.dtype)
            rt.counters[mod.module_tag+"_cora_forward"]+=1
            return res

        def cleanup(model):
            for mod in model.my_linears:
                mod._no31_cora_rank=None

        def train_patch(model,*args,**kwargs):
            try: return train0(model,*args,**kwargs)
            finally: cleanup(model)

        def eval_patch(model,*args,**kwargs):
            try: return eval0(model,*args,**kwargs)
            finally: cleanup(model)

        g.MiniGPT4Rec_v2.__init__=patched_init
        g.MiniGPT4Rec_v2.compute_feedback_gcn=feedback_patch
        g.MyLinear.forward=lora_forward_patch
        g.MiniGPT4Rec_v2.forward_v2=train_patch
        g.MiniGPT4Rec_v2.generate_for_samples_v2=eval_patch
        g._no31_cora_installed=True
        return self


def install(gain=.1,detach_gnn=True):
    return NO31CoRARuntime(gain,detach_gnn).install()
