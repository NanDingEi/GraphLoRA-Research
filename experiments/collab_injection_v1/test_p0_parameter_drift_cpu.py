#!/usr/bin/env python3
"""CPU-only synthetic tests for P1 S2/P0 parameter drift analyzer."""
import math

import torch

from analyze_p0_parameter_drift import (
    analyze_states, finalize, tensor_category, tensor_groups,
    tensor_update_summary,
)

b = torch.tensor([1.0,2.0],dtype=torch.float32)
c = torch.tensor([2.0,2.0],dtype=torch.float32)
a = torch.tensor([1.0,3.0],dtype=torch.float32)
v=tensor_update_summary(b,c,a,chunk=1)
assert v["n_elements"]==2
assert abs(v["base_sq"]-5.0)<1e-12
assert abs(v["control_update_sq"]-1.0)<1e-12
assert abs(v["aux_update_sq"]-1.0)<1e-12
assert abs(v["between_sq"]-2.0)<1e-12
assert abs(v["update_dot"])<1e-12
print("[P1-CPU] exact norm and orthogonal-update dot products PASS",flush=True)

d=tensor_update_summary(b, b+torch.tensor([1.,0.]),
                        b-torch.tensor([1.,0.]))
stats=finalize({
    "n_tensors":1,"n_elements":2,
    "changed_control":1,"changed_aux":1,"changed_between":1,
    **d,
})
assert abs(stats["update_direction_cosine"]+1.0)<1e-12
assert abs(stats["aux_minus_control_norm"]-2.0)<1e-12
print("[P1-CPU] opposing update directions and distance PASS",flush=True)

assert tensor_category("llama_model_lora.base_model.model.model.layers.30.self_attn.q_proj.lora_A.default.weight")=="llm_lora"
groups=tensor_groups("llama_model_lora.base_model.model.model.layers.30.self_attn.q_proj.lora_A.default.weight")
assert "lora_layer/L30" in groups
assert "lora_layer_proj/L30/q_proj" in groups
assert "lora_layer_ab/L30/lora_A" in groups
assert tensor_category("gcn.layers.0.weight")=="gnn"
assert tensor_category("llama_proj.0.weight")=="projector"
print("[P1-CPU] real P0-style module/layer/adapter grouping PASS",flush=True)

base, ctrl, aux = {}, {}, {}
# 12 LoRA tensors, 4 distinct layers and both Q/V projections.
for layer in (0,10,24,30):
    for proj,ab in (("q_proj","lora_A"),("q_proj","lora_B"),
                    ("v_proj","lora_A")):
        name=(f"llama_model_lora.base_model.model.model.layers.{layer}."
              f"self_attn.{proj}.{ab}.default.weight")
        base[name]=torch.tensor([2.,3.,4.])
        ctrl[name]=torch.tensor([3.,3.,4.])
        aux[name]=torch.tensor([3.,4.,4.])
base["gcn.embedding.weight"]=torch.tensor([1.,1.])
ctrl["gcn.embedding.weight"]=torch.tensor([2.,1.])
aux["gcn.embedding.weight"]=torch.tensor([2.,2.])
base["llama_proj.0.weight"]=torch.tensor([10.,10.])
ctrl["llama_proj.0.weight"]=torch.tensor([11.,10.])
aux["llama_proj.0.weight"]=torch.tensor([10.,11.])
report=analyze_states(base,ctrl,aux)
assert report["matched_floating_tensors"]==14
assert report["per_group"]["module/llm_lora"]["n_tensors"]==12
assert report["per_group"]["lora_layer/L30"]["n_tensors"]==3
assert report["per_group"]["module/gnn"]["n_tensors"]==1
assert report["per_group"]["module/projector"]["n_tensors"]==1
assert report["per_group"]["module/llm_lora"]["changed_between"]==12
assert len(report["top_20_between"])==14
print("[P1-CPU] whole-state grouping, matched tensor coverage PASS",flush=True)

try:
    analyze_states(base,{k:v for k,v in ctrl.items()
                         if ".layers.30." not in k},aux)
except RuntimeError as err:
    assert "LoRA state-key mismatch" in str(err)
else:
    raise AssertionError("Missing LoRA must be detected, not silently ignored")
print("[P1-CPU] unmatched LoRA checkpoint tensors rejected PASS",flush=True)

print("[P1-CPU] ALL PASS; no checkpoint, Vicuna, GPU or training",flush=True)
