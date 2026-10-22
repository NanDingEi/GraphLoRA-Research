#!/usr/bin/env python3
"""CPU-only regression tests for frozen-S2 parameter isolation.

Do not import Vicuna or dataset. Validate exact trainable-name whitelist,
frozen hashes, and that R4-03 has precisely one original trainable tensor.
"""
import torch
from torch import nn
from types import SimpleNamespace
from r4_freeze import (
    MODES,V31_NAME,freeze_after_s2_load,
    frozen_hash,assert_frozen_unchanged,
)


def insert(root,full_name):
    parts=full_name.split(".")
    m=root
    for part in parts[:-1]:
        if part not in m._modules:
            m.add_module(part,nn.Module())
        m=m._modules[part]
    m.register_parameter(parts[-1],nn.Parameter(torch.randn(2,2)))


def model_fixture():
    torch.manual_seed(8)
    root=nn.Module()
    root.collab_mode="legacy"
    root.my_linears=[SimpleNamespace(gcn_layer=31)]
    for name in (
        "r3_adapter.encoder.2.weight",
        "r3_adapter.encoder.2.bias",
        "r3_adapter.q_up.weight",
        "r3_adapter.v_up.weight",
        "gcn.gnn_layer.weight",
        "rec_encoder.user_embedding.weight",
        "llama_proj.0.weight",
        "llama_model_lora.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "llama_model_lora.base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
        "llama_model_lora.base_model.model.model.layers.31.self_attn.v_proj.lora_B.default.weight",
        "llama_model_lora.base_model.model.model.layers.31.self_attn.q_proj.lora_B.default.weight",
        "llama_model_lora.base_model.model.model.layers.31.self_attn.k_proj.weight",
    ):
        insert(root,name)
    return root


def test_policy(round_id):
    root=model_fixture()
    selected=freeze_after_s2_load(root,round_id)
    expected_v=(round_id=="03")
    actual_v=[n for n in selected if V31_NAME.search(n)]
    assert len(actual_v)==int(expected_v)
    assert len(selected)==4+int(expected_v)
    baseline=frozen_hash(root)
    assert len(baseline)>=4
    before={n:p.detach().clone() for n,p in root.named_parameters()}
    for n,p in root.named_parameters():
        if p.requires_grad:
            with torch.no_grad():
                p.add_(1.0)
    assert_frozen_unchanged(root,baseline)
    for n,p in root.named_parameters():
        if n in selected:
            assert not torch.equal(p,before[n]),n
        else:
            assert torch.equal(p,before[n]),n
    frozen=next(p for n,p in root.named_parameters()
                if n.startswith("gcn."))
    with torch.no_grad():
        frozen.add_(0.01)
    try:
        assert_frozen_unchanged(root,baseline)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failed to detect GNN frozen parameter drift")
    print(f"[R4-CPU] R4-{round_id} whitelist and SHA integrity PASS",flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    for k in MODES:
        test_policy(k)
    print("[R4-CPU] ALL PASS (no Vicuna and no CUDA)",flush=True)
