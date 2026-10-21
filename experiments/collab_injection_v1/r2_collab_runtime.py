#!/usr/bin/env python3
"""R2 GraphLoRA runtime hooks: candidate-aware history-V / decision-Q / K probe.

Installed BEFORE S2 model construction by r2_entry.py / r2_probe.py.
Patches no checkpoint tensor or optimizer code. Original legacy Q/V forward
is used except for history-V relation reweighting. Model's GCN view path is
temporarily selected to expose penultimate node embeddings, then reverted
BEFORE the actual LLM forward. All probes use real LLM hidden-state paths,
NEVER alter the Yes head or output logits directly.

WARNING: R2 checkpoints must be evaluated through this same runtime with
the recorded mode, not with the vanilla training entrypoint.
"""
from collections import defaultdict
import torch
import torch.nn.functional as F

MODES = ("off","audit","decision_q","history_k","relation_v",
         "relation_shuffle","relation_v_q","relation_v_raw")
_ACTIVE = None


def _gather(node_ids, features, query):
    if node_ids.ndim != 1 or features.shape[0] != node_ids.numel():
        raise RuntimeError("R2 invalid graph IDs/features shape")
    if node_ids.numel() == 0 or not bool((node_ids[1:] > node_ids[:-1]).all()):
        raise RuntimeError("R2 needs sorted distinct nonempty node IDs")
    index = torch.searchsorted(node_ids, query.long().contiguous())
    safe = index.clamp(max=node_ids.numel()-1)
    if not bool(((index < node_ids.numel()) & (node_ids[safe]==query)).all()):
        raise RuntimeError("R2 candidate/user node missing in subgraph")
    return features[safe]


def make_relation_weights(fb, samples, num_users, mode, normalize=True):
    """Compute per-token relation from candidate GNN penult and history GNN.

    Role=1 is history. User and target tokens get an exact unit weight.
    For WRONG-candidate contrast, only candidate graph context is swapped.
    No label, user or prompt information is changed.
    """
    if "gnn_penult" not in fb or "sub_node_ids" not in fb or "token_roles" not in fb:
        raise RuntimeError("R2 GCN metadata missing penult/node_ids/roles")
    ids=fb["sub_node_ids"].long()
    penult=fb["gnn_penult"].float()
    rows=fb["batch_indices"].long()
    local=fb["local_node_ids"].long()
    roles=fb["token_roles"].long()
    if rows.shape!=local.shape or rows.shape!=roles.shape:
        raise RuntimeError("R2 GCN token map is not aligned")
    if not bool(((local>=0)&(local<ids.numel())).all()):
        raise RuntimeError("R2 invalid local node ID among active tokens")
    bsz=int(samples["TargetItemID"].numel())
    cand=_gather(ids,penult,samples["TargetItemID"].long()+int(num_users))
    mismatch=0
    if mode=="relation_shuffle":
        from minigpt4.models.minigpt4rec_v2 import mismatch_candidate_features
        cand,mismatch=mismatch_candidate_features(
            samples["TargetItemID"],cand,ids,penult,num_users)
    chosen=cand[rows]
    source=penult[local]
    sim=F.cosine_similarity(source.float(),chosen.float(),dim=-1,eps=1e-8)
    raw=1.0+0.5*torch.tanh(sim)
    hist=roles==1
    weights=torch.ones_like(raw)
    if normalize:
        for b in range(bsz):
            sel=hist&(rows==b)
            if bool(sel.any()):
                weights[sel]=(raw[sel]/raw[sel].mean().clamp_min(1e-8)).clamp(0.5,1.5)
    else:
        weights=torch.where(hist,raw,weights)
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("R2 nonfinite relation weights")
    if not bool((weights[~hist]==1).all()):
        raise RuntimeError("R2 nonhistory weights changed")
    return weights,hist,sim,int(mismatch)


def _make_context(fb, samples, num_users):
    ids=fb["sub_node_ids"].long()
    z=fb["gcn_output"].float()
    u=_gather(ids,z,samples["UserID"].long())
    t=_gather(ids,z,samples["TargetItemID"].long()+int(num_users))
    ctx=(u+t)/2.0
    ctx=F.normalize(ctx,dim=-1,eps=1e-8)
    if not bool(torch.isfinite(ctx).all()):
        raise RuntimeError("R2 invalid decision Q context")
    return ctx


def _safe_quant(values):
    if not values:
        return {"n":0}
    x=torch.cat([v.reshape(-1) for v in values],dim=0).float()
    return {"n":int(x.numel()),"mean":float(x.mean()),
            "p50":float(x.quantile(0.5)),"p90":float(x.quantile(0.9)),
            "p99":float(x.quantile(0.99)),"max":float(x.max())}


class R2Runtime:
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown R2 mode {mode}; valid={MODES}")
        self.mode=mode
        self.stats=defaultdict(list)
        self.counters=defaultdict(int)
        self.original_forward=None
        self.original_feedback=None

    def record(self,key,tensor):
        with torch.no_grad():
            val=tensor.detach().float().flatten().cpu()
            if val.numel():
                # Limit probe and training memory overhead: store first
                # 2000 batches only, then aggregate via explicit count.
                if self.counters[key+"_records"]<2000:
                    self.stats[key].append(val)
                    self.counters[key+"_records"]+=1

    def state(self):
        return {
            "mode":self.mode,"counters":dict(self.counters),
            "stats":{k:_safe_quant(v) for k,v in self.stats.items()},
        }

    def activate(self,mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode=mode
        self.counters["mode_switch"]+=1

    def _feedback(self, model, samples, samples_encode, replaced_idx,
                  inputs_embeds, attention_mask):
        # Store original collab_mode while requesting graph penult.
        # Reuses ORIGINAL model GNN, no new parameter and no second graph pass.
        original_mode=model.collab_mode
        try:
            model.collab_mode="target_k_candidate"
            fb=self.original_feedback(
                model,samples,samples_encode,replaced_idx,
                inputs_embeds,attention_mask)
        finally:
            model.collab_mode=original_mode
        if original_mode!="legacy":
            raise RuntimeError("R2 requires legacy GraphLoRA model")
        if model.gcn_layer!=31:
            raise RuntimeError("R2 only supports original L31 injection")
        if "gnn_penult" not in fb:
            raise RuntimeError("R2 missing penultimate graph output")
        self.counters["feedback_calls"]+=1
        mods={m.module_tag:m for m in model.my_linears}
        if set(mods)!={"q","v"}:
            raise RuntimeError("R2 needs Q/V MyLinear only")
        # Clear previous batch state here. Original forward clears gcn_output.
        for m in mods.values():
            m.r2_weights=None
            m.r2_decision=None
            m.r2_key=None

        if self.mode in ("relation_v","relation_shuffle","relation_v_q","relation_v_raw"):
            mode="relation_shuffle" if self.mode=="relation_shuffle" else "relation_v"
            weights,hist,similarity,mismatched=make_relation_weights(
                fb,samples,model.num_users,mode,
                normalize=(self.mode!="relation_v_raw"))
            mods["v"].r2_weights=weights
            self.counters["history_tokens"]+=int(hist.sum().item())
            self.counters["candidate_mismatch_fallbacks"]+=mismatched
            self.record("history_weights",weights[hist])
            self.record("history_graph_similarity",similarity[hist])
        if self.mode in ("decision_q","relation_v_q"):
            tokenized=model.llama_tokenizer(
                [model.pos_ans[0] if int(y)==1 else model.neg_ans[0]
                 for y in samples["label"]],
                return_tensors="pt",padding="longest",truncation=True,
                max_length=model.max_txt_len,add_special_tokens=False)
            pos=int(inputs_embeds.shape[1]-tokenized.input_ids.shape[-1]-1)
            if pos<=0 or pos>=inputs_embeds.shape[1]:
                raise RuntimeError(f"R2 invalid decision Q position {pos}")
            context=_make_context(fb,samples,model.num_users)
            mods["q"].r2_decision=(context,pos)
            self.counters["decision_q_prepared"]+=int(context.shape[0])
        if self.mode=="history_k":
            roles=fb["token_roles"]
            hist=roles==1
            context=F.normalize(
                fb["gcn_output"][fb["local_node_ids"].long()].float(),
                dim=-1,eps=1e-8)
            mods["q"].r2_key=(
                context[hist],fb["batch_indices"][hist].long(),
                fb["token_positions"][hist].long())
            self.counters["history_k_prepared"]+=int(hist.sum().item())
        return fb

    def _v_forward(self, module, x):
        if module.module_tag!="v" or module.layer_idx!=31 or (
            self.mode not in ("relation_v","relation_shuffle",
                              "relation_v_q","relation_v_raw")):
            output=self.original_forward(module,x)
            if (module.module_tag=="v" and module.layer_idx==31
                    and module.gcn_output is not None and self.mode=="audit"):
                with torch.no_grad():
                    mask=module.local_node_ids!=-1
                    if bool(mask.any()):
                        local=module.local_node_ids[mask].long()
                        z=module.gcn_output[local].float()*float(module.gnn_scale)
                        A=module.lora_A[module.active_adapter]
                        B=module.lora_B[module.active_adapter]
                        x_at=x[module.batch_indices[mask],module.token_positions[mask]]
                        a=A(x_at.to(A.weight.dtype)).float()*float(module.lora_scale)
                        self.record("rank_graph_norm",z.norm(dim=-1))
                        self.record("rank_lora_norm",a.norm(dim=-1))
                        self.record("proj_graph_norm",F.linear(z,B.weight.float()).norm(dim=-1))
                        self.record("proj_lora_norm",F.linear(a,B.weight.float()).norm(dim=-1))
                        self.record("native_v_norm",output[
                            module.batch_indices[mask],module.token_positions[mask]
                        ].float().norm(dim=-1))
                        self.counters["audited_tokens"]+=int(mask.sum().item())
            return output

        # Match the ORIGINAL legacy MyLinear V expression, modifying ONLY
        # per-token GNN z contribution by candidate-history scalar weights.
        from minigpt4.models.minigpt4rec_v2 import transpose
        prev_dtype=x.dtype
        result=F.linear(x,transpose(module.weight,module.fan_in_fan_out),
                        bias=module.bias)
        name=module.active_adapter
        if name not in module.lora_A:
            return result.to(prev_dtype)
        if module.disable_adapters or module.merged or module.r[name]<=0:
            raise RuntimeError("R2 V requires unmerged enabled r>0 LoRA")
        xl=module.lora_dropout[name](x.to(module.lora_A[name].weight.dtype))
        a=module.lora_A[name](xl)
        if module.gcn_output is not None:
            if module.r2_weights is None:
                raise RuntimeError("R2 V weights missing (feedback not called)")
            valid=module.local_node_ids!=-1
            if bool(valid.any()):
                row=module.batch_indices[valid].long()
                pos=module.token_positions[valid].long()
                local=module.local_node_ids[valid].long()
                w=module.r2_weights[valid]
                if w.shape!=local.shape or a.shape[-1]!=module.gcn_output.shape[-1]:
                    raise RuntimeError("R2 V dimension/mapping mismatch")
                # Same scaling order as the original legacy branch, with
                # w==1 exactly reproducing original injection.
                graph=module.gcn_output[local]*w.to(module.gcn_output.dtype).unsqueeze(-1)
                a=a.clone()
                a[row,pos]=a[row,pos]*module.lora_scale + graph*(
                    module.gnn_scale*module.warmup_factor)
                self.counters["weighted_v_tokens"]+=int(valid.sum().item())
                self.record("weighted_v_graph_norm",(graph.float()*module.gnn_scale).norm(dim=-1))
        result+=module.lora_B[name](a)
        return result.to(prev_dtype)

    def _q_hook(self,module,inputs,output):
        if self.mode not in ("decision_q","relation_v_q"):
            return output
        if module.gcn_output is None or module.r2_decision is None:
            return output
        ctx,pos=module.r2_decision
        if output.ndim!=3 or len(ctx)!=output.shape[0] or not 0<=pos<output.shape[1]:
            raise RuntimeError("R2 decision Q shape/position failure")
        B=module.lora_B[module.active_adapter]
        delta=F.linear(ctx.to(B.weight.dtype),B.weight)
        if delta.shape!=(len(ctx),output.shape[-1]):
            raise RuntimeError("R2 decision Q projection shape mismatch")
        patched=output.clone()
        patched[:,pos,:]=output[:,pos,:]+(0.1*delta).to(output.dtype)
        self.record("decision_q_delta_norm",(0.1*delta).float().norm(dim=-1))
        self.counters["decision_q_hook"]+=1
        return patched

    def _k_hook(self,module,inputs,output):
        if self.mode!="history_k":
            return output
        # q.gcn_output is non-None only in the real full-model pass.
        q=self._q_mod
        if q.gcn_output is None or q.r2_key is None:
            return output
        context,rows,pos=q.r2_key
        if not len(rows):
            return output
        B=q.lora_B[q.active_adapter]
        delta=F.linear(context.to(B.weight.dtype),B.weight)
        if delta.shape!=(len(rows),output.shape[-1]):
            raise RuntimeError("R2 K probe only supports matching Q/K dimensions")
        patched=output.clone()
        patched[rows,pos]=output[rows,pos]+(0.1*delta).to(output.dtype)
        self.record("history_k_delta_norm",(0.1*delta).float().norm(dim=-1))
        self.counters["history_k_hook"]+=1
        return patched

    def attach_hooks(self,model):
        selected={m.module_tag:m for m in model.my_linears}
        self._q_mod=selected["q"]
        q_layer=model.llama_model.model.layers[31].self_attn
        if q_layer.q_proj is not self._q_mod:
            raise RuntimeError("R2 decision Q module identity mismatch")
        self.q_handle=q_layer.q_proj.register_forward_hook(self._q_hook)
        self.k_handle=q_layer.k_proj.register_forward_hook(self._k_hook)


def install_runtime(mode):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("R2 runtime already installed")
    from minigpt4.models import minigpt4rec_v2 as graph
    runtime=R2Runtime(mode)
    runtime.original_forward=graph.MyLinear.forward
    runtime.original_feedback=graph.MiniGPT4Rec_v2.compute_feedback_gcn
    def forward_patch(module,x):
        return runtime._v_forward(module,x)
    def feedback_patch(model,samples,samples_encode,replaced_idx,inputs_embeds,attention_mask):
        return runtime._feedback(model,samples,samples_encode,replaced_idx,
                                 inputs_embeds,attention_mask)
    graph.MyLinear.forward=forward_patch
    graph.MiniGPT4Rec_v2.compute_feedback_gcn=feedback_patch
    # Register Q/K hooks at model creation; attach only after all modules
    # have been replaced and checkpoint weights loaded.
    original_build=graph.MiniGPT4Rec_v2.from_config
    # Classmethod from_config is bound; wrap as classmethod.
    def build_wrapper(cls,cfg):
        model=original_build(cfg)
        if (str(model.collab_mode)!="legacy" or len(model.my_linears)!=2
                or int(model.gcn_layer)!=31):
            raise RuntimeError("R2 requires original S2 legacy 31 Q/V")
        runtime.attach_hooks(model)
        print(f"[R2-RUNTIME] attached={runtime.mode} legacy L31",flush=True)
        return model
    graph.MiniGPT4Rec_v2.from_config=classmethod(build_wrapper)
    _ACTIVE=runtime
    return runtime
