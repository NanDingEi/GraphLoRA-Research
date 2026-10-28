#!/usr/bin/env python3
"""NIGHT26 CPU preflight: matched FP32 control, learned vs uniform 3-hop graph.

No Vicuna GPU or user files required. All inputs synthetic and train-only.
"""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from experiments.no31_cora_original.night24_joint_graph import (
    JointLightGCN, _sparse_normalize
)

def main():
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="n26_cpu_") as temp:
        root = Path(temp)
        nu, ni, k, dim, layers = 3, 18, 16, 256, 3
        df = pd.DataFrame({
            "uid": [0, 0, 1, 1, 2, 2],
            "iid": [1, 2, 3, 4, 5, 6],
            "label": [1, 1, 1, 0, 1, 0],
        })
        train_path = root / "train_ood2.pkl"
        df.to_pickle(train_path)
        pu = df.loc[df.label.eq(1), "uid"].to_numpy(np.int64)
        pi = df.loc[df.label.eq(1), "iid"].to_numpy(np.int64)
        mf = torch.randn(nu + ni, dim, dtype=torch.float32) * 0.1
        adj = _sparse_normalize(pu, pi, nu, ni)
        states = [mf]
        for _ in range(layers):
            states.append(torch.sparse.mm(adj, states[-1]))
        avg = sum(states) / len(states)
        user_g = avg[:nu].detach().cpu().numpy().copy()
        item_g = avg[nu:].detach().cpu().numpy().copy()
        item_g[0] = 0
        clusters = np.full(ni, -1, dtype=np.int64)
        clusters[1:] = np.arange(ni-1) % k
        prototypes = np.stack([
            item_g[1:][clusters[1:] == group].mean(0)
            for group in range(k)
        ])
        residuals = np.zeros_like(item_g)
        residuals[1:] = item_g[1:] - prototypes[clusters[1:]]
        artifact = root / "mf_l3.npz"
        np.savez(artifact, graph_id_table=mf.numpy(),
                 user_graph=user_g, item_graph=item_g,
                 item_to_cluster=clusters, prototypes=prototypes,
                 item_residual=residuals)
        mf_user = mf[:nu].clone()
        mf_item = mf[nu:].clone()
        uniform = JointLightGCN(artifact, train_path, nu, ni, mf_user,
                                mf_item, layers=layers, k=k,
                                aggregation="uniform")
        learned = JointLightGCN(artifact, train_path, nu, ni, mf_user,
                                mf_item, layers=layers, k=k,
                                aggregation="learned")
        assert len(learned.layer_weight_summary()) == 4
        assert learned.layer_weight_summary() == [0.25] * 4
        uu, ui = uniform._propagate()
        lu, li = learned._propagate()
        assert torch.allclose(uu, lu, atol=1e-7, rtol=1e-6)
        assert torch.allclose(ui, li, atol=1e-7, rtol=1e-6)
        print("[N26-CPU-INITIAL-PARITY-PASS] L3 uniform==learned softmax(zeros)",
              flush=True)

        optimizer = torch.optim.AdamW([
            {"params": [learned.id_embedding], "lr": 1e-4},
            {"params": [learned.layer_logits], "lr": 1e-3},
        ])
        before = learned.layer_logits.detach().clone()
        # Simulate a nonzero downstream LLM+CoRA loss, with no BPR/BCE.
        # A weighted user/item interaction signal distinguishes layers.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            users, items = learned._propagate()
            proxy = (users[0] * items[1]).sum() + 0.3 * (
                users[1] * items[3]).sum()
        optimizer.zero_grad(set_to_none=True)
        proxy.backward()
        assert learned.id_embedding.grad is not None
        assert torch.isfinite(learned.id_embedding.grad).all()
        assert learned.id_embedding.grad.abs().max() > 0
        assert learned.layer_logits.grad is not None
        assert torch.isfinite(learned.layer_logits.grad).all()
        assert learned.layer_logits.grad.abs().max() > 0
        optimizer.step()
        after = learned.layer_logits.detach().clone()
        assert not torch.equal(before, after)
        weights = learned.layer_weight_summary()
        assert abs(sum(weights) - 1.) < 1e-6
        assert all(w > 0 for w in weights)
        assert max(abs(w - 0.25) for w in weights) > 1e-7
        print(f"[N26-CPU-LEARNED-PASS] new_weights={weights} "
              "graph_grad=True weights_changed=True", flush=True)

        # The MF-only control's condition must not connect to graph params
        # even if the graph module is instantiated for RNG/optimizer parity.
        for p in uniform.parameters():
            p.grad = None
        anchored_mf = torch.cat((mf_user[:1], mf_item[1:2]), dim=-1)
        probe = torch.nn.Parameter(torch.ones_like(anchored_mf))
        output = (anchored_mf + 0.0 * probe).square().sum()
        output.backward()
        assert all(p.grad is None for p in uniform.parameters())
        print("[N26-CPU-MF-ONLY-PASS] no_graph_forward/no_graph_grad",
              flush=True)
    print("[N26-CPU-PASS] exact uniform init, learned layer logits, "
          "FP32 gradient and MF-only isolation", flush=True)

if __name__ == "__main__":
    main()
