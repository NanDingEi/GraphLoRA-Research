#!/usr/bin/env python3
"""CPU parity tests: full graph-base vs fast graph-base, and fixed asymmetric fusion.

LightGCN remains full-graph and differentiable. Synthetic train-only adjacency
with no GPU, Vicuna, or user checkpoint dependency.
"""
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn

from experiments.no31_cora_original.night26_seven import ARMS as OLD_SEVEN
from experiments.no31_cora_original.night26_four import ARMS, COMMON
from experiments.no31_cora_original.test_night26_new7_cpu import setup, model
from experiments.no31_cora_original.night23_graph_features import install_graph_history
from experiments.no31_cora_original.night24_joint_graph import JointLightGCN
from experiments.no31_cora_original import night8 as core

def main():
    assert len(ARMS) == 4 and len({r["id"] for r in ARMS}) == 4
    assert set([r["id"] for r in ARMS]).isdisjoint({r["id"] for r in OLD_SEVEN})
    assert [r["graph_anchor_lambda"] for r in ARMS] == [.15, .25, .2, .2]
    assert [r.get("graph_fusion", "fixed") for r in ARMS] == [
        "fixed", "fixed", "fixed_pair", "fixed_pair"]
    assert [r.get("graph_user_lambda", 0) for r in ARMS[2:]] == [.1, .3]
    assert [r.get("graph_item_lambda", 0) for r in ARMS[2:]] == [.3, .1]
    assert COMMON["fast_graph_base"] is True
    assert COMMON["empty_cache_interval"] == 0
    assert COMMON["graph_aux_weight"] == 0.0
    # Check that all four environment variables for fast path actually reach
    # the subprocess; this is a cheap protection against miswired CLI rows.
    env = core.build_environment(dict(COMMON, **ARMS[2]))
    assert env["NO31_CORA_GRAPH_FUSION"] == "fixed_pair"
    assert env["NO31_CORA_GRAPH_USER_LAMBDA"] == ".1" or float(
        env["NO31_CORA_GRAPH_USER_LAMBDA"]) == .1
    assert float(env["NO31_CORA_GRAPH_ITEM_LAMBDA"]) == .3
    assert env["NO31_CORA_FAST_GRAPH_BASE"] == "1"
    assert env["NO31_CORA_EMPTY_CACHE_INTERVAL"] == "0"
    with tempfile.TemporaryDirectory(prefix="n28_four_cpu_") as tmp:
        art, train, mf_user, mf_item = setup(Path(tmp))
        graph = model(art, train, mf_user, mf_item, lam=.2)
        mock = nn.Module()
        mock.register_buffer("cora_fixed_user", mf_user.clone())
        mock.register_buffer("cora_fixed_item", mf_item.clone())
        install_graph_history(mock, art, train)
        counts = {key: getattr(mock, "n20_" + key) for key in (
            "pair_keys", "pair_total", "pair_pos", "user_total",
            "user_pos", "group_prior")}
        users = torch.tensor([0, 1, 2])
        targets = torch.tensor([1, 3, 5])

        old_pair, features, old_aux = graph(users, targets, counts,
                                           labels=None)
        assert features is not None and old_aux is None
        old_loss = old_pair.square().sum()
        old_loss.backward()
        old_grad = graph.id_embedding.grad.detach().clone()
        graph.zero_grad(set_to_none=True)

        fast_pair, fast_features, fast_aux = graph(
            users, targets, counts, labels=None, use_conditions=False,
            compute_aux=False,
        )
        assert fast_features is None and fast_aux is None
        assert torch.equal(old_pair, fast_pair)
        fast_pair.square().sum().backward()
        fast_grad = graph.id_embedding.grad.detach().clone()
        assert torch.equal(old_grad, fast_grad)
        assert fast_grad.abs().max() > 0
        print("[N28-CPU-FAST-PARITY-PASS] graph-base pair and full-graph "
              "ID-table gradients bitwise equal, community/history skipped",
              flush=True)

        for l_u, l_i in ((.1, .3), (.3, .1)):
            pair_graph = JointLightGCN(
                art, train, len(mf_user), len(mf_item),
                mf_user, mf_item, layers=3, k=16,
                aggregation="uniform", fusion="fixed_pair",
                initial_lambda=.2, user_lambda=l_u, item_lambda=l_i,
                sparse_audit_interval=256,
            )
            out_pair, _, _ = pair_graph(
                users, targets, counts, use_conditions=False,
                compute_aux=False,
            )
            u, i = out_pair.split(256, dim=-1)
            lam_u, lam_i = pair_graph.fusion_weights(users, targets)
            assert lam_u.shape == lam_i.shape == (3, 1)
            assert torch.allclose(lam_u, torch.full_like(lam_u, l_u))
            assert torch.allclose(lam_i, torch.full_like(lam_i, l_i))
            mf_pair = torch.cat((mf_user[users], mf_item[targets]), -1)
            mf_u, mf_i = mf_pair.split(256, dim=-1)
            mixed = torch.cat((
                mf_u + lam_u * (u - mf_u),
                mf_i + lam_i * (i - mf_i),
            ), dim=-1)
            assert mixed.shape == (3, 512) and torch.isfinite(mixed).all()
            loss = mixed.square().sum()
            loss.backward()
            assert pair_graph.id_embedding.grad is not None
            assert torch.isfinite(pair_graph.id_embedding.grad).all()
            assert pair_graph.id_embedding.grad.abs().max() > 0
            print(f"[N28-CPU-ASYMMETRIC-PASS] user={l_u} item={l_i} "
                  "full-graph gradient=True", flush=True)
    print("[N28-CPU-PASS] exactly four new arms and opt-in cache controls, "
          "legacy/fast graph-base forward+gradient parity, asymmetric fusion",
          flush=True)

if __name__ == "__main__":
    main()
