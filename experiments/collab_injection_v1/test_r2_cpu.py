#!/usr/bin/env python3
"""Lightweight CPU tests for R2 graph/role alignment and runtime isolation."""
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from r2_collab_runtime import (
    R2Runtime, make_relation_weights, _gather, _validate_l31_adapters, MODES,
)


def fixture():
    # user 1,2; history item global 101,102,103,104; target 105,106
    num_users=100
    ids=torch.tensor([1,2,101,102,103,104,105,106])
    feats=F.normalize(torch.stack([
        torch.tensor([0.7,0.2]),torch.tensor([0.2,0.8]),
        torch.tensor([1.0,0.0]),torch.tensor([0.0,1.0]),
        torch.tensor([-1.0,0.0]),torch.tensor([0.1,0.9]),
        torch.tensor([0.9,0.1]),torch.tensor([0.0,1.0]),
    ]).float(),dim=-1)
    role=torch.tensor([0,1,1,2, 0,1,1,2])
    batch=torch.tensor([0,0,0,0, 1,1,1,1])
    local=torch.tensor([0,2,3,6, 1,4,5,7])
    fb={
        "sub_node_ids":ids,"gnn_penult":feats,
        "gcn_output":torch.ones(8,8,dtype=torch.float32),
        "batch_indices":batch,"token_positions":torch.tensor(
            [4,5,6,7, 10,11,12,13]),
        "local_node_ids":local,"token_roles":role,
    }
    samples={
        "UserID":torch.tensor([1,2]),
        "TargetItemID":torch.tensor([5,6]),
        "label":torch.tensor([1,0]),
    }
    return fb,samples,num_users


def test_weights():
    fb,samples,n=fixture()
    w,h,sim,wrong=make_relation_weights(fb,samples,n,"relation_v")
    assert w.shape==(8,) and int(h.sum())==4 and wrong==0
    assert torch.equal(w[~h],torch.ones_like(w[~h]))
    assert torch.isfinite(w).all()
    for b in (0,1):
        m=h&(fb["batch_indices"]==b)
        assert abs(float(w[m].mean())-1)<1e-6
    assert torch.any((w[h]-1).abs()>1e-5)
    raw,_,_,_=make_relation_weights(fb,samples,n,"relation_v",normalize=False)
    assert torch.any(torch.abs(raw[h]-w[h])>1e-5)
    try:
        altered=dict(fb)
        altered["token_roles"]=torch.tensor([0,1,1])
        make_relation_weights(altered,samples,n,"relation_v")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Misaligned token roles should fail")
    print("[R2-CPU] masked per-user historical mean-one weights PASS",flush=True)


def test_model_context():
    fb,samples,n=fixture()
    runtime=R2Runtime("relation_v")
    runtime.original_feedback=lambda model,*a: fb
    modq=SimpleNamespace(module_tag="q",layer_idx=31,gcn_layer=31,
                         collab_mode="legacy",r2_weights=None,
                         r2_decision=None,r2_key=None)
    modv=SimpleNamespace(module_tag="v",layer_idx=31,gcn_layer=31,
                         collab_mode="legacy",r2_weights=None,
                         r2_decision=None,r2_key=None)
    model=SimpleNamespace(
        collab_mode="legacy",num_users=n,
        my_linears=[modq,modv],
    )
    # Regression: real MiniGPT4Rec_v2 does not expose model.gcn_layer.
    assert not hasattr(model,"gcn_layer")
    assert set(_validate_l31_adapters(model))=={"q","v"}
    try:
        modv.gcn_layer=30
        _validate_l31_adapters(model)
    except RuntimeError:
        pass
    else:
        raise AssertionError("R2 should reject wrong adapter injection depth")
    finally:
        modv.gcn_layer=31
    got=runtime._feedback(
        model,samples,None,None,torch.zeros(2,20,8),None)
    assert got is fb
    assert model.collab_mode=="legacy"
    assert modv.r2_weights.shape==(8,)
    assert modq.r2_weights is None
    assert runtime.counters["history_tokens"]==4
    assert runtime.stats["history_weights"]
    print("[R2-CPU] model mode restoration, payload row alignment PASS",flush=True)


def test_graph_gather():
    fb,samples,n=fixture()
    out=_gather(fb["sub_node_ids"],fb["gnn_penult"],torch.tensor([105,106]))
    assert out.shape==(2,2)
    try:
        _gather(fb["sub_node_ids"],fb["gnn_penult"],torch.tensor([909]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing node not rejected")
    print("[R2-CPU] graph user/target ID bounds PASS",flush=True)


def test_probe_loader_contract():
    """Regression: MultiIterLoader has __next__ but NOT __iter__.

    Extract probe's tiny loader-unwrapping function by AST so a CPU unit
    test does NOT import train_collm_mf_din or construct Vicuna.
    """
    import ast
    from pathlib import Path
    source=Path(__file__).with_name("r2_probe.py")
    root=ast.parse(source.read_text(encoding="utf-8"))
    functions=[n for n in root.body if isinstance(n,ast.FunctionDef)
               and n.name=="get_valid_probe_loader"]
    if len(functions)!=1:
        raise AssertionError("R2 probe concrete loader helper missing")
    ns={}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),"exec"),ns)
    get=ns["get_valid_probe_loader"]

    class ConcreteValid:
        def __len__(self):
            return 1300
        def __iter__(self):
            yield {"UserID":torch.tensor([10,20,30])}
            yield {"UserID":torch.tensor([40,50,60])}

    class MultiIterOnly:
        def __init__(self):
            self.loaders=[ConcreteValid()]
        def __next__(self):
            return next(iter(self.loaders[0]))
        # Deliberately NO __iter__. This matches GraphLoRA MultiIterLoader.

    wrapper=MultiIterOnly()
    assert not hasattr(wrapper,"__iter__")
    runner=SimpleNamespace(dataloaders={"valid":wrapper})
    loader=get(runner)
    assert [b["UserID"].tolist() for b in loader]==[
        [10,20,30],[40,50,60]]
    # Distinct fresh iteration restarts from the first Valid example.
    assert next(iter(get(runner)))["UserID"][0].item()==10
    wrapper.loaders.append(ConcreteValid())
    try:
        get(runner)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Multiple Valid loaders should be rejected")
    print("[R2-CPU] MultiIterLoader unwrap + fresh Valid iteration PASS",flush=True)


def main():
    test_probe_loader_contract()
    test_weights()
    test_graph_gather()
    test_model_context()
    assert len(MODES)==8
    print("[R2-CPU] ALL PASS (no Vicuna and no CUDA)",flush=True)


if __name__=="__main__":
    main()
