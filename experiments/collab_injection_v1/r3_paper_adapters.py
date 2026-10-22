#!/usr/bin/env python3
"""GraphLoRA/S2 controlled parameter-injection trials inspired by two papers.

R3-CF: ReLLaX (CFLoRA), candidate-conditioned full interaction B(I+M)A
       at L31 Q and V, keeping the original GraphLoRA injected rank vector.
R3-CO: CoRA-inspired collaborative query generator feeding a second
       candidate-conditioned low-rank delta B_c diag(s_ui) A at L31 Q and V.

These are deliberately *not* complete paper reproductions:
ReLLaX SUBR/SPA and CoRA's frozen-LLM training protocol are not used.
LLM LoRA remains trainable; original Yes-BCE and S2 GNN remain unchanged.

Install BEFORE MiniGPT4Rec_v2 is instantiated. The runtime must be
installed for both training and later evaluation of R3 checkpoints.
"""
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F

R3_MODES=("off","cflora","cora")


class CFLoRAMixer(nn.Module):
    """Per-example, Q/V independent rank x rank residual on L31 LoRA.

    The entire last projection is initialized to exactly zero:
    B(I+M)A == BA before adaptation; context features may have nontrivial
    random initial values but are multiplied by a zero matrix.
    """
    def __init__(self, graph_dim=256, rank=8, hidden=64):
        super().__init__()
        self.rank=int(rank)
        self.norm=nn.LayerNorm(4*graph_dim)
        self.encoder=nn.Sequential(
            nn.Linear(4*graph_dim,hidden),
            nn.GELU(),
            nn.Linear(hidden,2*self.rank*self.rank))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, user, item):
        x=torch.cat((user,item,user*item,torch.abs(user-item)),dim=-1).float()
        logits=self.encoder(self.norm(x))
        return (0.1*torch.tanh(logits)).view(-1,2,self.rank,self.rank)


class CollaborativeQueryGenerator(nn.Module):
    """CoRA-inspired queries: self-attn -> cross-attn(user,item) -> FFN.

    The conditioned, per-example low-rank weight is
        DeltaW(ui) = B_c * diag(s_ui) * A_S2,
    where B_c is independent, trainable and initially all-zero.
    Keeps the original S2 LoRA/GNN pathway unchanged at warm start.
    """
    def __init__(self, hidden_size, graph_dim=256, rank=8,
                 query_dim=64, queries=4):
        super().__init__()
        self.rank=int(rank)
        self.query=nn.Parameter(torch.randn(queries,query_dim)*0.02)
        self.cf_project=nn.Linear(graph_dim,query_dim)
        self.self_attn=nn.MultiheadAttention(query_dim,4,batch_first=True)
        self.cross_attn=nn.MultiheadAttention(query_dim,4,batch_first=True)
        self.norm1=nn.LayerNorm(query_dim)
        self.norm2=nn.LayerNorm(query_dim)
        self.norm3=nn.LayerNorm(query_dim)
        self.ff=nn.Sequential(nn.Linear(query_dim,2*query_dim),nn.GELU(),
                              nn.Linear(2*query_dim,query_dim))
        self.rank_generator=nn.Linear(query_dim,2*self.rank)
        self.q_up=nn.Linear(self.rank,hidden_size,bias=False)
        self.v_up=nn.Linear(self.rank,hidden_size,bias=False)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(self,user,item):
        graph_tokens=torch.stack((user.float(),item.float()),dim=1)
        kv=self.cf_project(graph_tokens)
        queries=self.query.unsqueeze(0).expand(len(user),-1,-1)
        q,_=self.self_attn(queries,queries,queries,need_weights=False)
        q=self.norm1(q+queries)
        z,_=self.cross_attn(q,kv,kv,need_weights=False)
        z=self.norm2(z+q)
        z=self.norm3(z+self.ff(z))
        summary=z.mean(dim=1)
        return 1.0+0.5*torch.tanh(self.rank_generator(summary)).view(
            -1,2,self.rank)


def _pair_features(fb, samples, model):
    required=("gnn_penult","sub_node_ids")
    if any(k not in fb for k in required):
        raise RuntimeError("R3 cannot find GNN penultimate graph representation")
    nodes=fb["sub_node_ids"].long()
    features=fb["gnn_penult"].float()
    if nodes.ndim!=1 or features.ndim!=2 or features.shape!=(
            nodes.numel(),256):
        raise RuntimeError("R3 GNN graph mapping has invalid shape")
    if not bool((nodes[1:]>nodes[:-1]).all()):
        raise RuntimeError("R3 GNN node IDs must be distinct and sorted")
    if model.training and not model.train_gnn_with_rec_loss:
        features=features.detach()

    def pick(ids):
        ids=ids.long().contiguous()
        indices=torch.searchsorted(nodes,ids)
        safe=indices.clamp(max=nodes.numel()-1)
        if not bool(((indices<nodes.numel()) & (nodes[safe]==ids)).all()):
            raise RuntimeError("R3 missing user or candidate graph node")
        return features[safe]

    users=pick(samples["UserID"])
    items=pick(samples["TargetItemID"]+int(model.num_users))
    if users.shape!=items.shape or users.shape[1]!=256:
        raise RuntimeError("R3 pair features mismatched")
    return users,items


class R3Runtime:
    def __init__(self,mode):
        if mode not in R3_MODES:
            raise ValueError(f"Invalid R3 mode: {mode}")
        self.mode=mode
        self.counters=Counter()
        self.original_init=None
        self.original_forward=None
        self.original_feedback=None
        self._injected_model=None

    def switch(self,mode):
        if mode not in R3_MODES:
            raise ValueError(mode)
        self.mode=mode
        self.counters["switches"]+=1

    def _validate(self,model):
        if model.collab_mode!="legacy":
            raise RuntimeError("R3 only supports legacy S2")
        mods={m.module_tag:m for m in model.my_linears}
        if set(mods)!={"q","v"} or len(model.my_linears)!=2:
            raise RuntimeError("R3 expects precisely one L31 Q and V adapter")
        a_rank=None
        for tag,m in mods.items():
            if m.layer_idx!=31 or m.gcn_layer!=31:
                raise RuntimeError("R3 must operate at L31")
            if m.collab_mode!="legacy" or m.rel_pair_gate is not None:
                raise RuntimeError("R3 expects unmodified legacy MyLinear")
            rank=m.lora_A[m.active_adapter].out_features
            if a_rank is not None and a_rank!=rank:
                raise RuntimeError("R3 Q/V LoRA rank mismatch")
            a_rank=rank
        return mods,a_rank

    def init_model(self,model,*args,**kwargs):
        self.original_init(model,*args,**kwargs)
        _,rank=self._validate(model)
        # Preserve S2 RNG for dataset order, LoRA dropout and warm-start
        # training, despite the random initialization of new modules.
        with torch.random.fork_rng(devices=[]):
            if self.mode=="cflora":
                model.r3_adapter=CFLoRAMixer(rank=rank)
            elif self.mode=="cora":
                hidden=model.my_linears[0].out_features
                model.r3_adapter=CollaborativeQueryGenerator(
                    hidden_size=hidden,rank=rank)
            else:
                raise RuntimeError("R3 model creation requires cflora or cora")
        model.r3_adapter.to(model.my_linears[0].weight.device)
        for p in model.r3_adapter.parameters():
            p.requires_grad_(True)
        self._injected_model=model
        self.counters["model_created"]+=1
        print(f"[R3-MODEL] mode={self.mode} rank={rank} "
              f"adapter_trainable={sum(p.numel() for p in model.r3_adapter.parameters())}",
              flush=True)

    def feedback(self,model,samples,samples_encode,replaced_idx,
                 inputs_embeds,attention_mask):
        saved=model.collab_mode
        try:
            # Only ask existing GraphLoRA GNN for its 256-d penultimate view.
            # No new graph propagation, data/labels or LLM teacher calls.
            model.collab_mode="target_k_candidate"
            fb=self.original_feedback(model,samples,samples_encode,replaced_idx,
                                      inputs_embeds,attention_mask)
        finally:
            model.collab_mode=saved
        mods,_=self._validate(model)
        if self._injected_model is not model:
            raise RuntimeError("R3 adapter bound to a different model")
        for m in mods.values():
            m.r3_payload=None
        user,item=_pair_features(fb,samples,model)
        # Graph node context stays float32 to preserve gradient precision.
        # No added logits head; outputs are consumed by LoRA projections only.
        with torch.cuda.amp.autocast(enabled=False):
            if self.mode=="cflora":
                mats=model.r3_adapter(user.float(),item.float())
                if mats.ndim!=4:
                    raise RuntimeError("R3 CFLoRA requires [B,2,r,r] matrices")
                mods["q"].r3_payload=mats[:,0]
                mods["v"].r3_payload=mats[:,1]
            elif self.mode=="cora":
                ranks=model.r3_adapter(user.float(),item.float())
                if ranks.ndim!=3:
                    raise RuntimeError("R3 CoRA requires [B,2,r] rank strengths")
                mods["q"].r3_payload=ranks[:,0]
                mods["v"].r3_payload=ranks[:,1]
            else:
                raise RuntimeError("R3 forward called while runtime is OFF")
        self.counters["feedback"]+=1
        self.counters["pairs"]+=int(user.shape[0])
        return fb

    def forward(self,module,x):
        if module.layer_idx!=31 or module.module_tag not in ("q","v"):
            return self.original_forward(module,x)
        if self.mode=="off" or module.gcn_output is None:
            return self.original_forward(module,x)
        if module.r3_payload is None:
            raise RuntimeError(f"R3 {module.module_tag}: missing collaborative payload")
        # The following matches original S2 MyLinear forward for legacy L31:
        # one dropout, one LoRA-A, unchanged per-ID graph addition, one B.
        from minigpt4.models.minigpt4rec_v2 import transpose
        orig_dtype=x.dtype
        result=F.linear(x,transpose(module.weight,module.fan_in_fan_out),
                        bias=module.bias)
        name=module.active_adapter
        if name not in module.lora_A or module.disable_adapters or module.merged:
            raise RuntimeError("R3 expects enabled unmerged L31 LoRA")
        if module.r[name]<=0 or module.rel_pair_gate is not None:
            raise RuntimeError("R3 bad LoRA rank or interfering rank gate")
        x_lora=module.lora_dropout[name](
            x.to(module.lora_A[name].weight.dtype))
        low=module.lora_A[name](x_lora)
        valid=module.local_node_ids!=-1
        if bool(valid.any()):
            rows=module.batch_indices[valid].long()
            pos=module.token_positions[valid].long()
            local=module.local_node_ids[valid].long()
            graph=module.gcn_output[local]
            if graph.shape!=(rows.numel(),low.shape[-1]):
                raise RuntimeError("R3 injected GNN/LoRA rank mismatch")
            low=low.clone()
            low[rows,pos]=(low[rows,pos]*module.lora_scale
                           + graph.to(low.dtype)*
                           (module.gnn_scale*module.warmup_factor))
        # CFLoRA B(I+M)A in the SAME post-injection rank bottleneck.
        # CoRA adds an independent low-rank delta weight B_c diag(s) A.
        payload=module.r3_payload
        if payload.shape[0]!=low.shape[0]:
            raise RuntimeError("R3 collaborative sample batch mismatch")
        if self.mode=="cflora":
            if payload.shape!=(low.shape[0],low.shape[-1],low.shape[-1]):
                raise RuntimeError("R3 CFLoRA matrix dimensions invalid")
            diff=torch.bmm(low.float(),payload.float().transpose(1,2))
            low=low+diff.to(low.dtype)
            result+=module.lora_B[name](low)
            self.counters[module.module_tag+"_cf_forward"]+=1
        elif self.mode=="cora":
            if payload.shape!=(low.shape[0],low.shape[-1]):
                raise RuntimeError("R3 CoRA rank strength dimensions invalid")
            main=module.lora_B[name](low)
            collab=low.float()*payload.float().unsqueeze(1)
            up=getattr(self._injected_model.r3_adapter,
                       module.module_tag+"_up")
            # Use FP32 only for the small rank->hidden collaborative delta,
            # then cast back to native LLM tensor type.
            delta=F.linear(collab,up.weight.float())
            result+=main+delta.to(main.dtype)
            self.counters[module.module_tag+"_cora_forward"]+=1
        else:
            raise RuntimeError(self.mode)
        return result.to(orig_dtype)


_RUNTIME=None


def install_runtime(mode):
    """Install the R3 adapter before constructing any S2 GraphLoRA model."""
    global _RUNTIME
    if _RUNTIME is not None:
        raise RuntimeError("R3 runtime installed twice")
    if mode not in ("cflora","cora"):
        raise ValueError("Installation requires cflora or cora")
    from minigpt4.models import minigpt4rec_v2 as graph
    rt=R3Runtime(mode)
    rt.original_init=graph.MiniGPT4Rec_v2.__init__
    rt.original_forward=graph.MyLinear.forward
    rt.original_feedback=graph.MiniGPT4Rec_v2.compute_feedback_gcn

    def init_patch(model,*args,**kwargs):
        return rt.init_model(model,*args,**kwargs)

    def forward_patch(module,x):
        return rt.forward(module,x)

    def feedback_patch(model,*args,**kwargs):
        return rt.feedback(model,*args,**kwargs)

    graph.MiniGPT4Rec_v2.__init__=init_patch
    graph.MiniGPT4Rec_v2.compute_feedback_gcn=feedback_patch
    graph.MyLinear.forward=forward_patch
    _RUNTIME=rt
    return rt
