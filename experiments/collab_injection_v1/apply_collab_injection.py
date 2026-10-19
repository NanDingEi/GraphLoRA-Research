#!/usr/bin/env python3
"""Safe, idempotent patch for GraphLoRA's MiniGPT4Rec_v2. Run at repo root.

Modes:
  legacy: exact original path
  gated_l31: sample/token-conditioned scalar gate on additive GNN contribution;
             2*sigmoid(0)=1 initializes to original L31 behavior.
  conditional_lora: GNN-conditioned per-rank modulation of LoRA-A result;
             starts identity, no additive GNN contribution (gnn_scale should be 0).

Fails closed on unfamiliar source, backs up the file, does not touch checkpoints.
"""
from pathlib import Path
import argparse
import ast
import shutil

MARK = '# COLLAB_INJECTION_V1'

def once(src, old, new, tag):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f'{tag}: expected exactly one anchor, found {n}; source is different, no changes made')
    return src.replace(old, new, 1)

def patch(src):
    if MARK in src:
        raise RuntimeError('Already patched; refusing to apply twice')
    src = once(src,
       'def __init__(self, original_linear, layer_idx=None, lora_scale=1.0, gnn_scale=0, gcn_layer=0):',
       'def __init__(self, original_linear, layer_idx=None, lora_scale=1.0, gnn_scale=0, gcn_layer=0, collab_mode="legacy"):', 'MyLinear signature')
    src = once(src, '        self.gcn_layer = gcn_layer\n', '''        self.gcn_layer = gcn_layer
        # COLLAB_INJECTION_V1: injected only at the existing L31 ID positions.
        self.collab_mode = collab_mode
        if collab_mode not in ("legacy", "gated_l31", "conditional_lora"):
            raise ValueError(f"Unknown collab_mode: {collab_mode}")
        rank = self.lora_A[self.active_adapter].out_features
        if collab_mode == "gated_l31":
            self.collab_gate = nn.Linear(2 * rank, 1, bias=True)
            nn.init.zeros_(self.collab_gate.weight)
            nn.init.zeros_(self.collab_gate.bias)
        elif collab_mode == "conditional_lora":
            self.collab_conditioner = nn.Linear(rank, rank, bias=True)
            nn.init.zeros_(self.collab_conditioner.weight)
            nn.init.zeros_(self.collab_conditioner.bias)
''', 'GNN property')
    start = src.index('                    # Original GraphLoRA injection:')
    end = src.index('\n            if self.rel_pair_gate is not None:', start)
    old = src[start:end]
    assert 'self.gcn_output[local_node_ids] * self.gnn_scale' in old, 'Unknown original injection block'
    extra = '''                    if self.collab_mode == "legacy":
''' + ''.join('    '+line+'\n' for line in old.splitlines()) + '''                    else:
                        a = lora_a_output[batch_indices, token_positions]
                        z = self.gcn_output[local_node_ids].to(dtype=a.dtype)
                        if z.shape != a.shape:
                            raise RuntimeError(f"GNN/LoRA rank mismatch: {z.shape} vs {a.shape}")
                        if self.collab_mode == "gated_l31":
                            # Baseline-preserving init: gate=2*sigmoid(0)=1.
                            gate_in = torch.cat((a.float(), z.float()), dim=-1)
                            gate = (2.0 * torch.sigmoid(self.collab_gate(gate_in)))
                            gate = gate.to(dtype=a.dtype)
                            updated = (a * self.lora_scale +
                                       z * (self.gnn_scale * self.warmup_factor) * gate)
                        else:
                            # Condition the LoRA rank activations; no direct GNN add.
                            modulation = (2.0 * torch.sigmoid(
                                self.collab_conditioner(z.float()))).to(dtype=a.dtype)
                            updated = a * self.lora_scale * modulation
                        lora_a_output[batch_indices, token_positions] = updated
'''
    src = src[:start] + extra.rstrip('\n') + src[end:]
    src = once(src, '            gcn_layer=0,\n', '            gcn_layer=0,\n            collab_mode="legacy",\n', 'Model signature')
    src = once(src, '        self.inj_diag = bool(inj_diag)\n', '''        if collab_mode not in ("legacy", "gated_l31", "conditional_lora"):
            raise ValueError(f"Unknown collab_mode: {collab_mode}")
        self.collab_mode = collab_mode
        if collab_mode == "conditional_lora" and float(gnn_scale) != 0:
            raise ValueError("conditional_lora requires gnn_scale=0 to avoid confounded additive injection")
        if collab_mode != "legacy" and (rel_rank_mod or rel_inject):
            raise ValueError("Disable rel_rank_mod/rel_inject for isolated collaborative injection experiments")
        self.inj_diag = bool(inj_diag)
''', 'model mode check')
    src = once(src, '''                            gcn_layer=gcn_layer
                        )''', '''                            gcn_layer=gcn_layer,
                            collab_mode=self.collab_mode
                        )''', 'MyLinear construction')
    src = once(src, '        gcn_layer = cfg.get("gcn_layer", 0)\n',
               '        gcn_layer = cfg.get("gcn_layer", 0)\n        collab_mode = cfg.get("collab_mode", "legacy")\n', 'cfg read')
    src = once(src, '            gcn_layer=gcn_layer,\n            gnn_semantic_align=gnn_semantic_align,', '            gcn_layer=gcn_layer,\n            collab_mode=collab_mode,\n            gnn_semantic_align=gnn_semantic_align,', 'cfg forwarding')
    ast.parse(src)
    return src

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--file', default='minigpt4/models/minigpt4rec_v2.py')
    ap.add_argument('--check', action='store_true')
    args=ap.parse_args()
    p=Path(args.file)
    src=p.read_text()
    out=patch(src)
    if args.check:
        print('PASS: patch anchors and Python syntax verified; no file modified')
        return
    backup=p.with_suffix(p.suffix+'.before_collab_v1.bak')
    if backup.exists():
        raise RuntimeError(f'Backup exists: {backup}; refusing to overwrite')
    shutil.copy2(p,backup)
    p.write_text(out)
    print(f'PATCHED {p}; BACKUP {backup}')

if __name__=='__main__':
    main()
