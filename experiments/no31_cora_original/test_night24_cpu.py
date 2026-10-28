#!/usr/bin/env python3
"""CPU-only smoke tests for N24 joint propagation and gradient paths."""
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Force the currently executing worktree ahead of stale PYTHONPATH entries.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from experiments.no31_cora_original.night23_graph_features import install_graph_history
from experiments.no31_cora_original.night24_joint_graph import (
    JointLightGCN, _sparse_normalize,
)


class MockModel(torch.nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()
        self.register_buffer("cora_fixed_user", torch.ones(n_users, 256) * 0.1)
        self.register_buffer("cora_fixed_item", torch.ones(n_items, 256) * 0.1)


def main():
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="n24_cpu_") as td:
        home = Path(td)
        nu, ni, k, d = 3, 18, 16, 256
        # All communities have >=1 item; target of user0 is the only
        # positive in its own community.
        df = pd.DataFrame(dict(uid=[0, 0, 1, 1, 2, 2],
                               iid=[1, 2, 3, 4, 5, 6],
                               label=[1, 1, 1, 0, 1, 0]))
        train = home / "train.pkl"
        df.to_pickle(train)
        u = df.loc[df.label.eq(1), "uid"].to_numpy(np.int64)
        i = df.loc[df.label.eq(1), "iid"].to_numpy(np.int64)
        rng = np.random.default_rng(42)
        table = rng.normal(0, .1, size=(nu + ni, d)).astype(np.float32)
        adj = _sparse_normalize(u, i, nu, ni)
        h = torch.from_numpy(table)
        hs = [h]
        for _ in range(2):
            h = torch.sparse.mm(adj, h)
            hs.append(h)
        total = torch.stack(hs).mean(0).numpy()
        ug = total[:nu].copy()
        ig = total[nu:].copy()
        ig[0] = 0
        groups = np.full(ni, -1, dtype=np.int64)
        groups[1:] = (np.arange(ni - 1) % k)
        proto = np.stack([ig[1:][groups[1:] == idx].mean(0)
                          for idx in range(k)])
        residual = np.zeros_like(ig)
        residual[1:] = ig[1:] - proto[groups[1:]]
        path = home / "graph.npz"
        np.savez(path, graph_id_table=table, user_graph=ug, item_graph=ig,
                 item_to_cluster=groups, prototypes=proto,
                 item_residual=residual)
        mock = MockModel(nu, ni)
        install_graph_history(mock, path, train)
        graph = JointLightGCN(path, train, nu, ni, mock.cora_fixed_user,
                              mock.cora_fixed_item, layers=2, k=k)
        counts = {key: getattr(mock, "n20_" + key) for key in (
            "pair_keys", "pair_total", "pair_pos", "user_total",
            "user_pos", "group_prior",
        )}
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([1, 3, 5])
        pair, (coarse, fine), aux = graph(users, items, counts,
                                         torch.tensor([1, 1, 1]))
        assert pair.shape == (3, 512)
        assert coarse.shape == (3, 514)
        assert fine.shape == (3, 513)
        assert aux.ndim == 0 and torch.isfinite(aux).all()
        assert torch.equal(fine[0, 256:512], torch.zeros(d))
        # LLM proxy gradients: use no graph BCE auxiliary and assert
        # graph propagation ID table still receives nonzero gradients.
        proxy_loss = pair.square().mean() + coarse.square().mean() + fine.square().mean()
        proxy_loss.backward()
        assert graph.id_embedding.grad is not None
        assert torch.isfinite(graph.id_embedding.grad).all()
        assert graph.id_embedding.grad.norm() > 0
        graph.zero_grad(set_to_none=True)
        _, _, aux2 = graph(users, items, counts, torch.tensor([1, 1, 1]))
        aux2.backward()
        assert graph.id_embedding.grad is not None
        assert graph.id_embedding.grad.norm() > 0
        # Explicitly simulate the outer AMP context: both sparse operations
        # must stay in FP32 and remain differentiable.
        graph.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp_pair, (amp_coarse, amp_fine), _ = graph(users, items, counts)
            amp_loss = amp_pair.square().mean() + amp_coarse.square().mean() + amp_fine.square().mean()
        amp_loss.backward()
        assert amp_pair.dtype == torch.float32
        assert torch.isfinite(graph.id_embedding.grad).all()
        assert graph.id_embedding.grad.norm() > 0
        # The AMP GradScaler must see the original nonfinite gradient to
        # skip the optimizer step. The hook is NOT allowed to raise or
        # silently mask Inf/NaN before GradScaler's unscale_ / found_inf.
        graph._grad_audit_calls = 3
        prior = graph._n24_bad_gradient_events
        bad = torch.zeros_like(graph.id_embedding)
        bad.view(-1)[0] = float("inf")
        returned = graph._audit_gradient(bad)
        assert returned is bad and torch.isinf(returned).any()
        assert graph._n24_bad_gradient_events == prior + 1
        # Miniature QFormer + group8 dynamic-A + NONZERO CoRA-P.
        # Reproduce the FP32 precision islands under an outer AMP scope
        # and verify the generated condition receives finite gradients.
        from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator
        generator = PerceptualWeightsGenerator(
            num_queries=4, user_embedding_size=16, hidden_size=32, r=16,
            input_mode="concat", injection_mode="shared",
            group_projector_mode="group8",
        )
        condition = torch.randn(2, 32, requires_grad=True)
        p_matrix = torch.nn.Parameter(torch.randn(32, 16) * 0.001)
        hidden = torch.randn(2, 5, 32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with torch.autocast(device_type="cpu", enabled=False):
                generated = generator(condition.float())
            assert len(generated) == 8
            assert all(w.dtype == torch.float32 for w in generated.values())
            with torch.autocast(device_type="cpu", enabled=False):
                weighted = torch.bmm(hidden.float(), generated[0].float())
                out = torch.nn.functional.linear(weighted, p_matrix.float())
            check_loss = out.square().mean()
        check_loss.backward()
        assert condition.grad is not None and condition.grad.norm() > 0
        assert torch.isfinite(condition.grad).all()
        assert p_matrix.grad is not None and torch.isfinite(p_matrix.grad).all()
    print("[N24-CPU-PASS] graph artifact parity, shape, target-exclusion, "
          "LLM-proxy graph gradients, auxiliary graph gradients, "
          "AMP-safe sparse propagation/history, FP32 CoRA generator/P, "
          "and nonfinite GradScaler handoff", flush=True)


if __name__ == "__main__":
    main()
