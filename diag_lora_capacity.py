"""
Per-layer LoRA capacity diagnostic on a GraphLoRA checkpoint.

For every decoder layer's q/v LoRA adapter, computes
    dW_l = B_l A_l,
    ||dW_l||_F,  ||dW_l||_2,  stable_rank = ||dW_l||_F^2 / ||dW_l||_2^2
to show where adaptation capacity is actually being used. Zero GPU cost.

Usage: python diag_lora_capacity.py --ckpt output/.../checkpoint_best.pth
"""
import argparse
import re

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]

    pat = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.default\.weight$")
    tensors = {}
    for k, v in sd.items():
        m = pat.search(k)
        if m is not None:
            layer, proj, ab = int(m.group(1)), m.group(2)[0], m.group(3)
            tensors.setdefault((layer, proj), {})[ab] = v.float()

    print(f"{'layer':>5s} {'q|dW|F':>9s} {'q|dW|2':>9s} {'q srank':>8s} "
          f"{'v|dW|F':>9s} {'v|dW|2':>9s} {'v srank':>8s}")
    q_fs, v_fs = [], []
    for layer in sorted({l for l, _ in tensors}):
        row = []
        for proj, fs in (("q", q_fs), ("v", v_fs)):
            t = tensors.get((layer, proj))
            if not t or "A" not in t or "B" not in t:
                row += [float("nan")] * 3
                continue
            dW = t["B"] @ t["A"]
            f = dW.norm().item()
            s2 = torch.linalg.matrix_norm(dW, ord=2).item()
            srank = f * f / max(s2 * s2, 1e-12)
            fs.append((layer, f, srank))
            row += [f, s2, srank]
        print(f"{layer:5d} " + " ".join(f"{x:9.4f}" if i % 3 == 0 else f"{x:9.4f}" if i % 3 == 1 else f"{x:8.3f}"
                                         for i, x in enumerate(row)))

    for proj, fs in (("q", q_fs), ("v", v_fs)):
        arr = torch.tensor([f for _, f, _ in fs])
        # thirds of the network
        n = len(arr) // 3
        thirds = [arr[: n].mean().item(), arr[n: 2 * n].mean().item(), arr[2 * n:].mean().item()]
        mx = max(fs, key=lambda x: x[1])
        print(f"\n[{proj}] ||BA||_F thirds (L0-10/L11-21/L22-31): "
              + " / ".join(f"{x:.3f}" for x in thirds)
              + f"   argmax layer = {mx[0]} (|F|={mx[1]:.4f})")


if __name__ == "__main__":
    main()
