#!/usr/bin/env python3
"""R3 paper adapter CPU invariants: zero-start, per-example, gradients, RNG.

No pretrained LLM, ML-1M, or GPU necessary. This only checks the small
modules and the mathematical parameter-injection construction; the GPU0
32-sample S2-native anchor is a separate required check.
"""
import torch
from torch import nn
from r3_paper_adapters import CFLoRAMixer, CollaborativeQueryGenerator


def test_cflora():
    torch.manual_seed(33)
    module=CFLoRAMixer(graph_dim=4,rank=3,hidden=8)
    u=torch.randn(2,4)
    i=torch.randn(2,4)
    output=module(u,i)
    assert output.shape==(2,2,3,3)
    assert bool((output==0).all()),"CFLoRA must start as identity"
    low=torch.randn(2,6,3)
    B=nn.Linear(3,7,bias=False)
    baseline=B(low)
    adapted=B(low+torch.bmm(low,output[:,1].transpose(1,2)))
    assert torch.equal(baseline,adapted)
    # Exactly-zero residual must still receive nonzero update gradient.
    loss=adapted.square().sum()
    loss.backward()
    p=module.encoder[-1].weight
    assert p.grad is not None and p.grad.norm()>0
    with torch.no_grad():
        p.add_(1e-3*p.grad)
    changed=module(u,i)
    assert not torch.allclose(changed[0],changed[1])
    print("[R3-CPU] CFLoRA identity, B(I+M)A and learnable sample matrices PASS")


def test_cora():
    torch.manual_seed(34)
    module=CollaborativeQueryGenerator(hidden_size=7,graph_dim=4,
                                       rank=3,query_dim=16,queries=3)
    u=torch.randn(2,4)
    i=torch.randn(2,4)
    strengths=module(u,i)
    assert strengths.shape==(2,2,3) and bool(torch.isfinite(strengths).all())
    low=torch.randn(2,6,3)
    baseline=torch.randn(2,6,7)
    delta=module.v_up(low*strengths[:,1,None,:])
    assert torch.count_nonzero(delta)==0
    assert torch.equal(baseline+delta,baseline)
    loss=(baseline+delta).square().mean()
    loss.backward()
    assert module.v_up.weight.grad is not None
    assert module.v_up.weight.grad.norm()>0
    with torch.no_grad():
        module.v_up.weight.add_(module.v_up.weight.grad)
    module.zero_grad(set_to_none=True)
    strengths=module(u,i)
    new_delta=module.v_up(low*strengths[:,1,None,:])
    assert torch.count_nonzero(new_delta)>0
    new_delta.square().mean().backward()
    assert module.rank_generator.weight.grad is not None
    assert module.rank_generator.weight.grad.norm()>0
    assert module.query.grad is not None
    assert module.query.grad.norm()>0
    print("[R3-CPU] CoRA queries, conditional low-rank, zero start, query grads PASS")


def test_modes():
    import ast
    from pathlib import Path
    path=Path(__file__).with_name("r3_paper_adapters.py")
    ast.parse(path.read_text(encoding="utf-8"))
    assert "cflora" in path.read_text()
    assert "cora" in path.read_text()
    print("[R3-CPU] runtime AST and two explicit modes PASS")


if __name__=="__main__":
    torch.set_num_threads(2)
    test_cflora()
    test_cora()
    test_modes()
    print("[R3-CPU] ALL PASS (no Vicuna and no CUDA)")
