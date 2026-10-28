#!/usr/bin/env python3
"""CPU-only synthetic preflight for E1-E7 graph representation and gradient paths."""
import json
import math
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from experiments.no31_cora_original.night24_joint_graph import (
    JointLightGCN, _sparse_normalize,
)
from experiments.no31_cora_original.night23_graph_features import install_graph_history
from experiments.no31_cora_original.night26_seven import ARMS

def setup(td):
    nu, ni, k, d = 3, 18, 16, 256
    df = pd.DataFrame({"uid": [0, 0, 1, 1, 2, 2],
                       "iid": [1, 2, 3, 4, 5, 6],
                       "label": [1, 1, 1, 0, 1, 0]})
    train = td / "train_ood2.pkl"
    df.to_pickle(train)
    u = np.array([0, 0, 1, 2], dtype=np.int64)
    i = np.array([1, 2, 3, 5], dtype=np.int64)
    torch.manual_seed(42)
    mf = torch.randn(nu + ni, d).float() * 0.1
    mat = _sparse_normalize(u, i, nu, ni)
    h = mf
    states = [h]
    for _ in range(3):
        h = torch.sparse.mm(mat, h)
        states.append(h)
    avg = sum(states) / 4.
    user, item = avg[:nu].numpy().copy(), avg[nu:].numpy().copy()
    item[0] = 0
    group = np.full(ni, -1, dtype=np.int64)
    group[1:] = np.arange(ni - 1) % k
    proto = np.stack([item[1:][group[1:] == j].mean(0) for j in range(k)])
    residual = np.zeros_like(item)
    residual[1:] = item[1:] - proto[group[1:]]
    artifact = td / "graph.npz"
    np.savez(artifact, graph_id_table=mf.numpy(), user_graph=user,
             item_graph=item, item_to_cluster=group, prototypes=proto,
             item_residual=residual)
    return artifact, train, mf[:nu], mf[nu:]

def model(artifact, train, mf_u, mf_i, aggregation="uniform",
          fusion="fixed", lam=0.1):
    return JointLightGCN(artifact, train, mf_u.shape[0], mf_i.shape[0],
                         mf_u, mf_i, layers=3, k=16, aggregation=aggregation,
                         fusion=fusion, initial_lambda=lam)

def check_gate(graph, mf_u, mf_i, mode):
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([1, 3, 5], dtype=torch.long)
    u, i = graph._propagate()
    lu, li = graph.fusion_weights(users, items)
    assert lu.shape == li.shape == (3, 1)
    assert torch.allclose(lu, torch.full_like(lu, 0.1), atol=1e-6)
    assert torch.allclose(li, torch.full_like(li, 0.1), atol=1e-6)
    assert (lu >= 0).all() and (li <= 0.5).all()
    # A proxy downstream loss directly through each user's/item's graph
    # residual verifies both trainable lambda and trainable LightGCN table.
    mixed_u = mf_u[users] + lu * (u[users] - mf_u[users])
    mixed_i = mf_i[items] + li * (i[items] - mf_i[items])
    loss = mixed_u.square().sum() + 0.13 * mixed_i.square().sum()
    loss.backward()
    assert graph.id_embedding.grad is not None
    assert graph.id_embedding.grad.abs().max().item() > 0
    if mode == "dual":
        grad = graph.dual_gate_logits.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().min().item() > 0
        torch.optim.AdamW([graph.dual_gate_logits], lr=1e-2).step()
        after = graph.fusion_summary()
        assert (abs(after["user_lambda"] - 0.1) > 1e-6
                or abs(after["item_lambda"] - 0.1) > 1e-6)
    else:
        for p in (graph.degree_gate_intercepts, graph.degree_gate_slopes):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().max() > 0
        # Degree conditioning must produce distinct coefficients after slope
        # changes; initial equal .1 is deliberate for a matched ablation.
        with torch.no_grad():
            graph.degree_gate_slopes[0] = 0.4
            graph.degree_gate_slopes[1] = -0.4
        lu2, li2 = graph.fusion_weights(users, items)
        assert torch.std(lu2).item() > 1e-6
        # Item degrees differ between interacted (item1) and cold (item7).
        p1, pi1 = graph.fusion_weights(torch.tensor([0, 0]),
                                       torch.tensor([1, 7]))
        assert abs(pi1[0].item() - pi1[1].item()) > 1e-6
        assert torch.isfinite(p1).all()
    print(f"[N27-CPU-GATE-PASS] mode={mode} graph_and_gate_grads=True", flush=True)

def main():
    assert len(ARMS) == 7 and len({a["id"] for a in ARMS}) == 7
    assert [round(a["graph_anchor_lambda"], 3) for a in ARMS] == [
        .025, .2, .1, .1, .1, .1, .1,
    ]
    assert ARMS[2]["graph_fusion"] == "dual"
    assert ARMS[3]["graph_fusion"] == "degree"
    assert ARMS[4]["graph_aggregation"] == "shallow_fixed"
    assert ARMS[5]["graph_aggregation"] == "shallow_learned"
    assert ARMS[6]["graph_mode"] == "graph_c2f"
    with tempfile.TemporaryDirectory(prefix="n27_new7_") as temp:
        artifact, train, mf_u, mf_i = setup(Path(temp))
        fixed = model(artifact, train, mf_u, mf_i,
                      aggregation="shallow_fixed")
        learned = model(artifact, train, mf_u, mf_i,
                        aggregation="shallow_learned")
        expected = [0.4, 0.3, 0.2, 0.1]
        assert np.allclose(fixed.layer_weight_summary(), expected, atol=1e-7)
        assert np.allclose(learned.layer_weight_summary(), expected, atol=1e-7)
        fu, fi = fixed._propagate()
        lu, li = learned._propagate()
        assert torch.allclose(fu, lu, atol=1e-6)
        assert torch.allclose(fi, li, atol=1e-6)
        opt = torch.optim.AdamW(learned.parameters(), lr=1e-2)
        opt.zero_grad()
        proxy = lu[0].square().sum() + 0.3 * li[1].square().sum()
        proxy.backward()
        assert learned.layer_logits.grad is not None
        assert torch.isfinite(learned.layer_logits.grad).all()
        assert learned.layer_logits.grad.abs().max() > 0
        opt.step()
        now = learned.layer_weight_summary()
        assert max(abs(a-b) for a,b in zip(now, expected)) > 1e-6
        assert abs(sum(now) - 1) < 1e-5
        print("[N27-CPU-SHALLOW-PASS] E5/E6 same init; learned weights updated", flush=True)

        for mode in ("dual", "degree"):
            g = model(artifact, train, mf_u, mf_i, fusion=mode)
            check_gate(g, mf_u, mf_i, mode)

        # E7: graph forward provides actual community and residual signals,
        # while the original Group8 schedule alternates coarse -> fine.
        graph = model(artifact, train, mf_u, mf_i)
        mock = nn.Module()
        mock.register_buffer("cora_fixed_user", mf_u.clone())
        mock.register_buffer("cora_fixed_item", mf_i.clone())
        install_graph_history(mock, artifact, train)
        counts = {name: getattr(mock, "n20_" + name) for name in (
            "pair_keys", "pair_total", "pair_pos", "user_total",
            "user_pos", "group_prior"
        )}
        users = torch.tensor([0, 1])
        items = torch.tensor([1, 3])
        pair, (coarse, fine), aux = graph(users, items, counts)
        assert pair.shape == (2, 512)
        assert coarse.shape == (2, 514)
        assert fine.shape == (2, 513)
        assert aux is None
        schedule = ((1., 0.), (1., 0.), (.75, .25), (.75, .25),
                    (.25, .75), (.25, .75), (0., 1.), (0., 1.))
        assert all(abs(a+b-1.) < 1e-6 for a,b in schedule)
        ch = nn.Sequential(nn.Linear(514, 64), nn.GELU(), nn.Linear(64, 512))
        fh = nn.Sequential(nn.Linear(513, 64), nn.GELU(), nn.Linear(64, 512))
        for head in (ch, fh):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        token = torch.randn(2, 512)
        loss = sum((token + a*ch(coarse) + b*fh(fine)).square().mean()
                   for a,b in schedule)
        loss.backward()
        assert ch[-1].weight.grad is not None and ch[-1].weight.grad.abs().max() > 0
        assert fh[-1].weight.grad is not None and fh[-1].weight.grad.abs().max() > 0
        print("[N27-CPU-C2F-PASS] graph coarse/fine signal widths and "
              "both group8 heads gradient=True", flush=True)
    print("[N27-CPU-PASS] seven unique experiments, shallow aggregation, "
          "user/item and degree gates, real coarse/fine features", flush=True)

if __name__ == "__main__":
    main()
