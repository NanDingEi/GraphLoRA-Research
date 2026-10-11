"""Cosine-structure regularization for GraphLoRA's input projector (llama_proj).

Auxiliary loss deliberately detaches source collaborative embeddings, thereby
regularizing only the projector. No projector pretraining and no modifications
of the Layer31 GNN -> LoRA injection path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def projector_cosine_reg_loss(
    projector,
    item_embedding_weight: torch.Tensor,
    num_items: int = 64,
    padding_index: int | None = 0,
) -> torch.Tensor:
    """Match item-item cosine similarity before and after llama_proj.

    Randomly sample distinct item IDs without replacement each call. Padding
    index is excluded. The source is treated as a stop-gradient teacher, so
    this auxiliary loss does not modify rec_encoder or GNN parameters.

    Return: scalar differentiable MSE over off-diagonal pairs (not diagonals).
    """
    if item_embedding_weight.ndim != 2:
        raise ValueError("item_embedding_weight must be [num_items, embedding_dim]")
    n_total = item_embedding_weight.size(0)
    num_items = int(num_items)
    if num_items < 2:
        raise ValueError("num_items must be >= 2")
    if n_total < 2:
        raise ValueError("Need at least two non-padding item embeddings")

    # On-device, no CPU sync for sampling. Without-replacement avoids duplicate
    # item IDs masquerading as off-diagonal positive pairs.
    ids = torch.randperm(n_total, device=item_embedding_weight.device)
    if padding_index is not None:
        ids = ids[ids != int(padding_index)]
    ids = ids[:num_items]
    if ids.numel() < 2:
        raise ValueError("Need at least two non-padding item embeddings")

    source = item_embedding_weight.detach().index_select(0, ids)
    # Use the actual parameter dtype and same device, while allowing CUDA
    # autocast to handle the dense projector as in GraphLoRA's main forward.
    param = next(projector.parameters(), None)
    if param is not None:
        source = source.to(device=param.device, dtype=param.dtype)
    projection = projector(source)
    if projection.ndim < 2 or projection.shape[0] != source.shape[0]:
        raise ValueError("Projector output must be [N, ...] and preserve batch axis")
    projection = projection.reshape(projection.shape[0], -1)

    # Similarities in fp32 avoid fp16 numerical instability. Teacher detached.
    with torch.no_grad():
        source_unit = F.normalize(source.float(), dim=-1, eps=1e-8)
        sim_original = source_unit @ source_unit.T
    projected_unit = F.normalize(projection.float(), dim=-1, eps=1e-8)
    sim_projected = projected_unit @ projected_unit.T
    n = sim_original.size(0)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=sim_original.device)
    return (sim_projected[offdiag] - sim_original[offdiag]).square().mean()


from functools import wraps


def projector_regularized_forward(forward_fn):
    """Add optional SP loss to forward_v2's final training loss.

    This is deliberately a method decorator, not an edit to the method body:
    locally modified GraphLoRA versions can return different dictionaries in
    different branches. The wrapper preserves those results and all existing
    auxiliary losses, and only adds to the existing result['loss'].

    With weight=0, evaluation mode, or no-grad, it calls and returns the
    original forward method without computing the auxiliary loss.
    """
    @wraps(forward_fn)
    def wrapped(self, *args, **kwargs):
        result = forward_fn(self, *args, **kwargs)
        weight = float(getattr(self, 'proj_reg_weight', 0.0))
        if not (self.training and torch.is_grad_enabled() and weight > 0.0):
            return result

        if not isinstance(result, dict) or 'loss' not in result:
            raise RuntimeError(
                'Projector regularization is enabled, but forward_v2 returned '
                'a non-dict or a dict without "loss" during training; refusing '
                'to silently skip the auxiliary loss.'
            )
        if not isinstance(result['loss'], torch.Tensor):
            raise TypeError('forward_v2 output["loss"] must be a torch.Tensor')
        rec_encoder = getattr(self, 'rec_encoder', None)
        proj = getattr(self, 'llama_proj', None)
        emb = getattr(rec_encoder, 'item_embedding', None)
        if proj is None or emb is None or not hasattr(emb, 'weight'):
            raise RuntimeError(
                'proj_reg requires llama_proj and rec_encoder.item_embedding.weight '
                '(MF-style GraphLoRA baseline).'
            )

        # Run in the same autocast context as the original model (if present).
        context = getattr(self, 'maybe_autocast', None)
        if context is None:
            reg_loss = projector_cosine_reg_loss(
                proj, emb.weight,
                num_items=int(getattr(self, 'proj_reg_items', 64)),
                padding_index=getattr(rec_encoder, 'padding_index', None),
            )
        else:
            with context():
                reg_loss = projector_cosine_reg_loss(
                    proj, emb.weight,
                    num_items=int(getattr(self, 'proj_reg_items', 64)),
                    padding_index=getattr(rec_encoder, 'padding_index', None),
                )

        # Copy to avoid unexpectedly mutating dictionaries referenced by custom
        # training instrumentation. No original loss component is overwritten.
        out = dict(result)
        out['loss'] = result['loss'] + weight * reg_loss
        self._last_proj_reg_loss = reg_loss.detach()
        self._proj_reg_step = int(getattr(self, '_proj_reg_step', 0)) + 1
        _glora_maybe_probe_gradients(self.llama_proj, result['loss'], reg_loss, weight, self._proj_reg_step)
        if self._proj_reg_step == 1 or self._proj_reg_step % 500 == 0:
            print('[projector-reg] step=%d cosine_mse=%.6f weighted=%.6f' % (
                self._proj_reg_step,
                reg_loss.detach().float().item(),
                weight * reg_loss.detach().float().item(),
            ))
        return out

    return wrapped


# GLORA_GRAD_PROBE_V1
# This diagnostic never calls .backward() nor mutates optimizer gradients or loss.
def _glora_maybe_probe_gradients(projector, base_loss, reg_loss, weight, step):
    """Compare gradients of existing forward loss vs weighted projector loss.

    'base' is the full original forward_v2 loss (rec + any semantic/N1 losses),
    NOT an isolated recommendation-only loss.
    """
    import os
    import math

    if os.environ.get('GLORA_GRAD_PROBE', '0') != '1':
        return
    try:
        first = max(0, int(os.environ.get('GLORA_GRAD_FIRST', '3')))
        every = max(1, int(os.environ.get('GLORA_GRAD_EVERY', '500')))
    except ValueError:
        raise ValueError('GLORA_GRAD_FIRST and GLORA_GRAD_EVERY must be integers')
    if step > first and step % every != 0:
        return

    scope = os.environ.get('GLORA_GRAD_SCOPE', 'first').strip().lower()
    named = [(n, p) for n, p in projector.named_parameters() if p.requires_grad]
    weights = [(n, p) for n, p in named if n.endswith('weight')]
    if scope == 'first':
        named = weights[:1]
    elif scope == 'last':
        named = weights[-1:]
    elif scope != 'all':
        raise ValueError('GLORA_GRAD_SCOPE must be first, last or all')
    if not named:
        print('[grad-probe] step=%d SKIP no trainable projector parameters' % step, flush=True)
        return
    params = tuple(p for _, p in named)
    if not base_loss.requires_grad or not reg_loss.requires_grad:
        print('[grad-probe] step=%d SKIP loss not differentiable' % step, flush=True)
        return

    # Both calls retain the computation graph for the *normal* training backward.
    # autograd.grad doesn't accumulate anything into parameter .grad.
    gbase = torch.autograd.grad(base_loss, params, retain_graph=True,
                                create_graph=False, allow_unused=True)
    greg = torch.autograd.grad(reg_loss, params, retain_graph=True,
                               create_graph=False, allow_unused=True)

    # Aggregate one cosine over the selected projector parameters, not a
    # misleading average of per-layer cosine values.
    dev = params[0].device
    base_sq = torch.zeros((), dtype=torch.float32, device=dev)
    reg_sq = torch.zeros((), dtype=torch.float32, device=dev)
    dot = torch.zeros((), dtype=torch.float32, device=dev)
    active_base = 0
    active_reg = 0
    for a, b in zip(gbase, greg):
        if a is not None:
            active_base += 1
            af = a.detach().float().reshape(-1)
            base_sq = base_sq + torch.dot(af, af)
        if b is not None:
            active_reg += 1
            bf = b.detach().float().reshape(-1)
            reg_sq = reg_sq + torch.dot(bf, bf)
        if a is not None and b is not None:
            dot = dot + torch.dot(af, bf)
    bs, rs, dp = torch.stack((base_sq, reg_sq, dot)).detach().cpu().tolist()
    nb = math.sqrt(max(0., bs))
    nr = math.sqrt(max(0., rs))
    cos = dp / (nb * nr) if nb > 0 and nr > 0 else float('nan')
    ratio = (abs(weight) * nr / nb) if nb > 0 else float('nan')
    print('[grad-probe] step=%d scope=%s params=%s base_norm=%.5g reg_norm=%.5g '
          'weighted_reg_norm=%.5g ratio=%.5g cosine=%.6f conflict=%s '
          'active_base=%d/%d active_reg=%d/%d' % (
              step, scope, ','.join(n for n, _ in named), nb, nr, abs(weight)*nr,
              ratio, cos, ('YES' if cos < 0 else 'NO') if math.isfinite(cos) else 'NA',
              active_base, len(params), active_reg, len(params)
          ), flush=True)

