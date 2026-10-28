#!/usr/bin/env python3
"""CPU preflight: ensure NIGHT25 uses exact MF initialization and true graph depth."""
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Force the currently executing worktree ahead of stale PYTHONPATH entries.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.no31_cora_original import graph_cf_pretrain as _graph_module
if not Path(_graph_module.__file__).resolve().is_relative_to(ROOT):
    raise RuntimeError(
        f"N25 imported the wrong graph_cf_pretrain: {_graph_module.__file__}"
    )
graph_train = _graph_module.graph_train
load_mf_init_table = _graph_module.load_mf_init_table
normalized_adj = _graph_module.normalized_adj
print(f"[N25-IMPORT-PATH-OK] graph_module={_graph_module.__file__}", flush=True)

def main():
    torch.manual_seed(42)
    n_users, n_items, dim = 3, 18, 256
    pos_u = np.array([0, 0, 1, 2], dtype=np.int64)
    pos_i = np.array([1, 2, 3, 4], dtype=np.int64)
    pos_keys = pos_u * n_items + pos_i
    with tempfile.TemporaryDirectory(prefix="n25_cpu_") as tmp:
        ckpt = Path(tmp) / "mf.pth"
        user = torch.randn(n_users, dim) * 0.1
        item = torch.randn(n_items, dim) * 0.1
        item[0] = 0
        torch.save({"user_embedding.weight": user,
                    "item_embedding.weight": item}, ckpt)
        mf = load_mf_init_table(ckpt, n_users, n_items, dim)
        assert torch.equal(mf, torch.cat((user, item)))
        adj, _ = normalized_adj(
            pos_u, pos_i, n_users, n_items, torch.device("cpu")
        )
        final = {}
        for layers in (2, 3):
            zu, zi, _, raw = graph_train(
                pos_u, pos_i, pos_keys, n_users, n_items, dim,
                layers, epochs=0, max_pairs=8,
                lr=0.01, reg=1e-4, seed=42,
                device=torch.device("cpu"), init_table=mf,
            )
            assert np.array_equal(raw, mf.numpy()), "MF initial embedding changed"
            x = mf
            states = [x]
            for _ in range(layers):
                x = torch.sparse.mm(adj, x)
                states.append(x)
            exact = torch.stack(states).mean(0)
            assert np.allclose(zu, exact[:n_users].numpy(), atol=1e-7)
            assert np.allclose(zi[1:], exact[n_users+1:].numpy(), atol=1e-7)
            final[layers] = exact.detach().clone()
            print(f"[N25-CPU-LAYER-PASS] layers={layers} raw_MF_exact=True "
                  f"propagated_shape={tuple(exact.shape)}", flush=True)
        assert not torch.allclose(final[2], final[3])
        # The fixed MF anchor is *not* a detach of trainable graph state.
        raw_param = torch.nn.Parameter(mf.clone())
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with torch.autocast(device_type="cpu", enabled=False):
                states = [raw_param.float()]
                for _ in range(2):
                    states.append(torch.sparse.mm(adj.float(), states[-1]))
                propagated = sum(states) / len(states)
                anchored = mf + 0.1 * (propagated - mf)
                loss = anchored.square().sum()
        loss.backward()
        assert raw_param.grad is not None
        assert torch.isfinite(raw_param.grad).all()
        assert raw_param.grad.norm().item() > 0
        # Random BPR pretraining must not accidentally load MF weights.
        _, _, _, random_raw = graph_train(
            pos_u, pos_i, pos_keys, n_users, n_items, dim, 2,
            epochs=1, max_pairs=8, lr=0.01, reg=1e-4,
            seed=42, device=torch.device("cpu"), init_table=None,
        )
        assert not np.array_equal(random_raw, mf.numpy())
    print("[N25-CPU-PASS] exact pretrained MF copy, 2/3 layer parity, "
          "FP32 graph->anchor gradient, random-vs-MF distinction", flush=True)

if __name__ == "__main__":
    main()
