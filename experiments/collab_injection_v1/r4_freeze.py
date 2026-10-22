#!/usr/bin/env python3
"""R4 parameter isolation policy applied AFTER from_config loads the S2 state.

R4-01: CFLoRA adapter only.
R4-02: CoRA adapter only.
R4-03: CoRA adapter + exactly L31 V LoRA-B.default.weight.

Keep original S2 GNN->LoRA forward path unchanged. Do NOT use
model.freeze_lora = True: that can disable the LoRA forward itself.
Instead use named_parameters().requires_grad_ flags before Runner
initializes the AdamW optimizer.
"""
import hashlib
import json
import re
import torch

MODES={
    "01":("cflora",False),
    "02":("cora",False),
    "03":("cora",True),
}
V31_NAME=re.compile(
    r"(?:^|\.)layers\.31\.self_attn\.v_proj\.lora_B\.default\.weight$"
)
_WATCH_CLASSES=(
    "r3_adapter.", "gcn.", "rec_encoder.","llama_proj.",
    "llama_model_lora."
)


def allowed_params(model,round_id):
    if round_id not in MODES:
        raise ValueError(f"Unknown R4 round {round_id}")
    expected_method,local_v=MODES[round_id]
    names=[n for n,_ in model.named_parameters()]
    if not any(n.startswith("r3_adapter.") for n in names):
        raise RuntimeError("R4 adapter missing from model named_parameters")
    extra=[n for n in names if V31_NAME.search(n)]
    if len(extra)!=1:
        raise RuntimeError(
            f"R4 expected exactly ONE named L31 V-B default tensor, got {extra}")
    allow={n for n in names if n.startswith("r3_adapter.")}
    if local_v:
        allow.add(extra[0])
    if not all(n.startswith("r3_adapter.") or V31_NAME.search(n)
               for n in allow):
        raise RuntimeError("R4 unexpected parameter selection")
    return expected_method,allow


def freeze_after_s2_load(model,round_id):
    method,allowed=allowed_params(model,round_id)
    if model.collab_mode!="legacy":
        raise RuntimeError("R4 requires original S2 legacy injection")
    if model.my_linears[0].gcn_layer!=31:
        raise RuntimeError("R4 requires L31 S2")
    count=0
    for name,param in model.named_parameters():
        train=name in allowed
        param.requires_grad_(train)
        if train:
            count+=param.numel()
    actual={n for n,p in model.named_parameters() if p.requires_grad}
    if actual!=allowed:
        raise RuntimeError(
            f"R4 trainable mismatch expected={sorted(allowed)}, "
            f"actual={sorted(actual)}")
    names=model.state_dict().keys()
    if not any(k.startswith("r3_adapter.") for k in names):
        raise RuntimeError("R4 parameters are not registered in checkpoint state")
    model.r4_policy=round_id
    model.r4_trainable_names=tuple(sorted(allowed))
    model.r4_trainable_count=count
    print("[R4-FREEZE] "+json.dumps({
        "round":round_id,"method":method,
        "trainable_tensors":len(allowed),"trainable_params":count,
        "trainable_names":sorted(allowed)},ensure_ascii=False),flush=True)
    return allowed


def install_freeze(round_id):
    """Install on top of R3 installed runtime, before building the model."""
    if round_id not in MODES:
        raise ValueError(round_id)
    from minigpt4.models import minigpt4rec_v2 as graph
    original_build=graph.MiniGPT4Rec_v2.from_config
    def patched_build(cls,cfg):
        model=original_build(cfg)
        freeze_after_s2_load(model,round_id)
        # Snapshot before the optimizer ever sees the model.
        model.r4_frozen_baseline_hash=frozen_hash(model)
        print(f"[R4-FROZEN-SNAPSHOT] frozen_tensors="
              f"{len(model.r4_frozen_baseline_hash)}",flush=True)
        return model
    graph.MiniGPT4Rec_v2.from_config=classmethod(patched_build)


def _selected_tensors(model):
    """Small but broad frozen invariant set; no 7B-param CPU clone.

    Q/V LoRA (all 32 layers), GNN, MF, projector and any L31 injected
    biases are ~small enough for GPU SHA snapshotting.
    """
    for n,p in model.named_parameters():
        if n.startswith("r3_adapter."):
            continue
        if ("lora_" in n or n.startswith(("gcn.","rec_encoder.","llama_proj."))
            or ("layers.31.self_attn" in n and n.endswith(".bias"))):
            yield n,p


@torch.no_grad()
def frozen_hash(model):
    """Fingerprint frozen optimization-sensitive S2 tensors.

    Excludes the one explicitly-unfrozen L31 V-B in R4-03.
    """
    result={}
    for n,p in _selected_tensors(model):
        if p.requires_grad:
            continue
        value=p.detach().contiguous().cpu().numpy().tobytes()
        result[n]=hashlib.sha256(value).hexdigest()
    return result


@torch.no_grad()
def assert_frozen_unchanged(model,baseline):
    after=frozen_hash(model)
    if after!=baseline:
        changes=sorted(n for n in set(after)|set(baseline)
                       if after.get(n)!=baseline.get(n))
        raise RuntimeError(f"R4 FROZEN PARAMETER DRIFT: {changes[:20]}")
    print(f"[R4-FROZEN-HASH-OK] verified={len(baseline)} "
          "frozen LLM LoRA/GNN/MF/projector/bias tensors",flush=True)
    return True
