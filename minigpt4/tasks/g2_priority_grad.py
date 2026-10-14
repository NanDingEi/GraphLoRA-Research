"""G2 conflict projection for accumulated GNN parameter gradients.

Rec-gradient has priority. Project only the weighted semantic gradient if
its dot product with the recommendation gradient is negative. Calculations
are in FP32 and are performed once per optimizer update, after gradient
accumulation. No trainable modules, prediction changes or new objectives.
"""
import torch


def project_auxiliary(rec_grad, aux_grad, eps=1e-12):
    if len(rec_grad) != len(aux_grad):
        raise ValueError("G2 gradient list lengths disagree")
    if not rec_grad:
        raise ValueError("G2 requires at least one GNN parameter")
    dot = torch.zeros((), dtype=torch.float32, device=rec_grad[0].device)
    norm2 = torch.zeros_like(dot)
    for r, a in zip(rec_grad, aux_grad):
        if r.shape != a.shape:
            raise ValueError("G2 gradient dimensions differ")
        if not torch.isfinite(r).all() or not torch.isfinite(a).all():
            raise FloatingPointError("G2 received non-finite gradient")
        dot = dot + (r.float() * a.float()).sum()
        norm2 = norm2 + r.float().square().sum()
    # The negative dot is the only condition for destructive interference.
    conflict = bool((dot < 0).item()) and bool((norm2 > eps).item())
    if conflict:
        coef = dot / norm2.clamp_min(eps)
        corrected = [r.float() + a.float() - coef * r.float()
                     for r, a in zip(rec_grad, aux_grad)]
    else:
        corrected = [r.float() + a.float() for r, a in zip(rec_grad, aux_grad)]
    return corrected, conflict, float(dot.item())


def add_grads(accumulator, gradients):
    if accumulator is None:
        return [(torch.zeros_like(g, dtype=torch.float32) if g is not None
                 else None) for g in gradients]
    return accumulator


def accumulate_(accumulator, gradients, params):
    if accumulator is None:
        accumulator = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    for slot, g in zip(accumulator, gradients):
        if g is not None:
            slot.add_(g.detach().float())
    return accumulator
