#!/usr/bin/env python3
"""Non-destructive installer for gradient-only diagnostic in GraphLoRA projector v2.

Patches only minigpt4/models/projector_structure_reg.py, never model source/YAML.
Without GLORA_GRAD_PROBE=1 the original projector regularizer behaves as before.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil

TARGET = 'minigpt4/models/projector_structure_reg.py'
MARKER = '# GLORA_GRAD_PROBE_V1'
ANCHOR = "        self._proj_reg_step = int(getattr(self, '_proj_reg_step', 0)) + 1\n"
ADDITION = "        _glora_maybe_probe_gradients(self.llama_proj, result['loss'], reg_loss, weight, self._proj_reg_step)\n"

HELPER = '''
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
'''


def patch(text):
    if MARKER in text:
        raise RuntimeError('Gradient probe is already installed; not modifying twice')
    if 'def projector_regularized_forward(' not in text or 'def projector_cosine_reg_loss(' not in text:
        raise RuntimeError('Target is not the v2 GraphLoRA projector regularization module')
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Could not uniquely identify regularizer step; unchanged source required')
    modified = text.replace(ANCHOR, ANCHOR + ADDITION, 1) + '\n' + HELPER + '\n'
    ast.parse(modified)
    return modified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path.cwd())
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    target = args.project.resolve() / TARGET
    if not target.exists():
        parser.error('Missing %s; run from GraphLoRA root' % target)
    original = target.read_text(encoding='utf-8')
    updated = patch(original)
    print('[OK] source parsed; injector located; gradient probe prepared')
    if args.check:
        print('[DRY RUN] No files modified')
        return
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = target.with_name(target.name + '.before_grad_probe_' + suffix)
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(target, backup)
    try:
        target.write_text(updated, encoding='utf-8')
        compile(target.read_text(encoding='utf-8'), str(target), 'exec')
    except Exception:
        shutil.copy2(backup, target)
        raise
    print('[backup]', backup)
    print('[done] Gradient diagnostics installed (OFF unless GLORA_GRAD_PROBE=1)')


if __name__ == '__main__':
    main()
