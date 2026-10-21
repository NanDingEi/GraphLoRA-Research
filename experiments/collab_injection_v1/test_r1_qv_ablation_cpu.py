#!/usr/bin/env python3
"""R1 CPU synthetic tests: scalar Q/V ablation wiring and pairwise diagnosis.

No Vicuna, no CUDA, no dataset/checkpoint files. The model module helper
functions are AST-loaded to avoid model construction or the training stack.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import r1_analyze_s2_qv_ablation as r1
from analyze_s2_layer_complementarity import weighted_uauc

model_file=Path(__file__).resolve().parents[2]/"experiments/collab_injection_v1/r1_eval_s2_qv_ablation.py"
ast_root=ast.parse(model_file.read_text(encoding="utf-8"))
names={"select_l31_modules","set_scales"}
funcs=[n for n in ast_root.body if isinstance(n,ast.FunctionDef)
       and n.name in names]
assert {n.name for n in funcs}==names
scoped=ast.Module(body=funcs,type_ignores=[])
ns={"np":np}
exec(compile(scoped,str(model_file),"exec"),ns)
q=SimpleNamespace(layer_idx=31,gcn_layer=31,
                  module_tag="q",collab_mode="legacy",
                  gnn_scale=0.1,warmup_factor=1.)
v=SimpleNamespace(layer_idx=31,gcn_layer=31,
                  module_tag="v",collab_mode="legacy",
                  gnn_scale=0.1,warmup_factor=1.)
model=SimpleNamespace(collab_mode="legacy",my_linears=[v,q])
mods=ns["select_l31_modules"](model)
for name,bq,bv in r1.ARMS:
    ns["set_scales"](mods,bq,bv)
    assert q.gnn_scale==bq and v.gnn_scale==bv
ns["set_scales"](mods,0.1,0.1)
assert q.gnn_scale==0.1 and v.gnn_scale==0.1
try:
    ns["set_scales"](mods,0.1,0.2)
except ValueError:
    pass
else:
    raise AssertionError("Out of bounds beta must fail")
print("[R1-CPU] five Q/V arms set exactly expected gnn_scale PASS",flush=True)

user=np.array([10]*4+[20]*4,dtype=np.int64)
item=np.arange(1,9)
label=np.array([1,1,0,0,1,0,1,0],dtype=np.int64)
off=np.array([0.8,0.2,0.6,0.5,0.1,0.7,0.4,0.3])
on =np.array([0.9,0.8,0.5,0.4,0.2,0.6,0.7,0.1])
half=(off+on)*0.5
q_only=off.copy()
v_only=on.copy()
test_data={
    "user":user,"item":item,"label":label,
    "arm_names":np.asarray([n for n,_,_ in r1.ARMS]),
    "q_scales":np.asarray([q for _,q,_ in r1.ARMS]),
    "v_scales":np.asarray([v for _,_,v in r1.ARMS]),
    "R1_A_S2_QV":on,"R1_B_OFF":off,
    "R1_C_V_ONLY":v_only,"R1_D_Q_ONLY":q_only,
    "R1_E_V_HALF":half,
}
old_a,old_u=r1.EXPECTED_S2_AUC,r1.EXPECTED_S2_UAUC
r1.EXPECTED_S2_AUC=float(roc_auc_score(label,on))
r1.EXPECTED_S2_UAUC=weighted_uauc(user,label,on)[0]
try:
    # train iid appears with varying frequencies. Test labels never used
    # for ranking item popularity; frequencies derive from train rows.
    train=pd.DataFrame({"iid":[1]*2+[2]*4+[3]*8+[4]+[5]*3+[6]*3+[7]*6})
    report=r1.analyze(test_data,train,expected_n=8)
    assert len(report["metrics"])==5
    assert report["ON_baseline_paired_comparisons"]["R1_B_OFF"][
        "global_auc_pairs"]["net_delta_auc"]==(
        report["metrics"]["R1_B_OFF"]["auc"]-
        report["metrics"]["R1_A_S2_QV"]["auc"])
    assert sum(g["n_samples"] for g in report["train_item_frequency_groups"])==8
    assert any(g["group"]=="unseen_in_train" and g["n_samples"]==1
               for g in report["train_item_frequency_groups"])
    assert report["metrics"]["R1_C_V_ONLY"]["auc"]==(
        report["metrics"]["R1_A_S2_QV"]["auc"])
    assert report["metrics"]["R1_D_Q_ONLY"]["auc"]==(
        report["metrics"]["R1_B_OFF"]["auc"])
    print("[R1-CPU] ON/OFF pair ledger and n_positive-weighted UAUC PASS",flush=True)
    print("[R1-CPU] train-only popularity bins, unseen bin and totals PASS",flush=True)
    bad=dict(test_data)
    bad["item"]=item[:-1]
    try:
        r1.analyze(bad,train,expected_n=8)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Sample alignment mismatch should fail")
    print("[R1-CPU] archive sample alignment guard PASS",flush=True)
finally:
    r1.EXPECTED_S2_AUC=old_a
    r1.EXPECTED_S2_UAUC=old_u

print("[R1-CPU] ALL PASS; no GPU or model loading",flush=True)
