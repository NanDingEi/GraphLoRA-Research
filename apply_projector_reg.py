#!/usr/bin/env python3
"""Safely insert optional projector cosine regularization into GraphLoRA.

Designed for wgj15965/GraphLoRA's minigpt4rec_v2.py, including locally
modified versions with multiple custom return branches in forward_v2.
Uses a decorator to preserve all original forward_v2 code and return paths.
Fails closed if expected class/constructor/config constructs are absent.

Usage (from repo root):
  python apply_patch.py --config tran_configs/collm_pretrain_mf_ood.yaml --weight 0.001 --apply
  python apply_patch.py --config tran_configs/collm_pretrain_mf_ood.yaml --weight 0.001 --check
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
import re
import shutil

IMPORT_LINE = 'from minigpt4.models.projector_structure_reg import projector_regularized_forward\n'
INIT_ATTRS = '''        # Optional item-item structure regularization on llama_proj only.
        self.proj_reg_weight = float(proj_reg_weight)
        self.proj_reg_items = int(proj_reg_items)
        self._proj_reg_step = 0
        self._last_proj_reg_loss = None
'''


def get_class(tree):
    matches = [c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'MiniGPT4Rec_v2']
    if len(matches) != 1:
        raise RuntimeError('Cannot uniquely locate class MiniGPT4Rec_v2; no changes made')
    return matches[0]


def get_method(cls, name):
    matches = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError(f'Cannot locate MiniGPT4Rec_v2.{name}; no changes made')
    return matches[0]


def _insert_signature_args(lines, func):
    if any(x.arg == 'proj_reg_weight' for x in func.args.args + func.args.kwonlyargs):
        raise RuntimeError('proj_reg_weight already exists. Refusing to reapply patch')
    beg = func.lineno - 1
    end = func.body[0].lineno - 1
    close = next((i for i in range(beg, end) if re.match(r'^\s*\)\s*(?:->.*)?:\s*$', lines[i])), None)
    if close is None:
        raise RuntimeError('Expected multiline __init__ signature not found')
    # If **kwargs is present, insert before it rather than after.
    kwarg_idx = next((i for i in range(beg, close) if '**' in lines[i] and not lines[i].lstrip().startswith('#')), None)
    pos = kwarg_idx if kwarg_idx is not None else close
    prev = pos - 1
    while prev > beg and not lines[prev].strip():
        prev -= 1
    if not lines[prev].rstrip().endswith(','):
        lines[prev] = lines[prev].rstrip() + ',\n'
    indent = re.match(r'^(\s*)', lines[prev]).group(1)
    lines[pos:pos] = [
        f'{indent}proj_reg_weight=0.0,\n',
        f'{indent}proj_reg_items=64,\n',
    ]


def _insert_attrs(lines, func):
    # Textual search scoped to the method, not the entire source.
    start, stop = func.body[0].lineno - 1, func.end_lineno
    matches = [i for i in range(start, stop) if 'self.proj_token_num = proj_token_num' in lines[i]]
    if len(matches) != 1:
        raise RuntimeError('Could not find unique proj_token_num assignment in __init__')
    lines[matches[0]+1:matches[0]+1] = INIT_ATTRS.splitlines(keepends=True)


def _insert_from_config(lines, func):
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'cls']
    if len(calls) != 1:
        raise RuntimeError('Expected one model = cls(...) in from_config')
    c = calls[0]
    if any(k.arg in ('proj_reg_weight', 'proj_reg_items') for k in c.keywords):
        raise RuntimeError('proj_reg arguments already in from_config')
    anchors = [i for i in range(c.lineno - 1, c.end_lineno)
               if re.search(r'^\s*proj_mid\s*=\s*proj_mid\s*,?', lines[i])]
    if len(anchors) != 1:
        raise RuntimeError('Cannot find exact proj_mid=proj_mid argument in from_config')
    indent = re.match(r'^(\s*)', lines[anchors[0]]).group(1)
    if not lines[anchors[0]].rstrip().endswith(','):
        lines[anchors[0]] = lines[anchors[0]].rstrip() + ',\n'
    lines[anchors[0]+1:anchors[0]+1] = [
        f'{indent}proj_reg_weight=cfg.get("proj_reg_weight", 0.0),\n',
        f'{indent}proj_reg_items=cfg.get("proj_reg_items", 64),\n',
    ]


def _insert_forward_decorator(lines, func):
    """Wrap forward_v2 without touching ANY return statement or control flow.

    Preserves custom N1/RLM return dictionaries, early returns, and additional
    loss terms. Runs the regularizer after forward_v2 computes its result.
    """
    if not isinstance(func, ast.FunctionDef):
        raise RuntimeError('forward_v2 must be a normal instance method')
    if func.decorator_list and any(isinstance(d, ast.Name) and d.id == 'projector_regularized_forward'
                                   for d in func.decorator_list):
        raise RuntimeError('forward_v2 already wrapped by projector_regularized_forward')
    # If forward_v2 has existing decorators, put this at the top, so our wrapper
    # receives the final result of any other wrappers (outermost decorator).
    insert_at = (min(d.lineno for d in func.decorator_list) if func.decorator_list else func.lineno) - 1
    indent = re.match(r'^(\s*)', lines[func.lineno-1]).group(1)
    lines.insert(insert_at, f'{indent}@projector_regularized_forward\n')


def patch_source(src):
    if 'projector_cosine_reg_loss' in src or 'proj_reg_weight' in src:
        raise RuntimeError('Projector regularization appears already installed; refusing to double-patch')
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    cls = get_class(tree)
    # Reverse-source line order edits to preserve AST original line positions.
    _insert_from_config(lines, get_method(cls, 'from_config'))
    _insert_forward_decorator(lines, get_method(cls, 'forward_v2'))
    init = get_method(cls, '__init__')
    _insert_attrs(lines, init)
    _insert_signature_args(lines, init)
    # Add import after existing torch imports, no package name collisions.
    imp_idx = next((i for i,l in enumerate(lines) if l.startswith('import torch')), None)
    if imp_idx is None:
        raise RuntimeError('torch import not found')
    lines.insert(imp_idx, IMPORT_LINE)
    patched = ''.join(lines)
    ast.parse(patched)
    return patched


def patch_config(src, weight, items):
    if re.search(r'^\s*proj_reg_weight\s*:', src, flags=re.M):
        raise RuntimeError('proj_reg_weight already appears in YAML; refusing duplicate')
    lines = src.splitlines(keepends=True)
    model_start = next((i for i,l in enumerate(lines) if re.fullmatch(r'model:\s*',l.strip())), None)
    if model_start is None:
        raise RuntimeError('YAML model: section not found')
    model_stop = next((i for i in range(model_start+1,len(lines)) if lines[i].strip() and not lines[i].startswith((' ', '#', '\t'))), len(lines))
    anchor = next((i for i in range(model_start+1,model_stop) if re.match(r'^\s+proj_mid_times\s*:',lines[i])), None)
    if anchor is None:
        raise RuntimeError('YAML model.proj_mid_times not found')
    indent = re.match(r'^(\s*)',lines[anchor]).group(1)
    lines[anchor+1:anchor+1] = [
        f'{indent}proj_reg_weight: {weight}  # 0 disables; start with 0.001\n',
        f'{indent}proj_reg_items: {items}  # distinct non-padding item IDs per step\n',
    ]
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='minigpt4/models/minigpt4rec_v2.py')
    parser.add_argument('--config', default='tran_configs/collm_pretrain_mf_ood.yaml')
    parser.add_argument('--weight', type=float, default=0.001)
    parser.add_argument('--items', type=int, default=64)
    parser.add_argument('--check', action='store_true', help='validate, but do not write')
    parser.add_argument('--apply', action='store_true', help='backup and write changes')
    a = parser.parse_args()
    if a.apply == a.check:
        parser.error('Specify exactly one of --check or --apply')
    if a.weight < 0.0 or a.items < 2:
        parser.error('Require weight >= 0 and items >= 2')
    model_path, config_path = Path(a.model), Path(a.config)
    if not model_path.exists() or not config_path.exists():
        parser.error('Missing model/config path; run from GraphLoRA project root')
    if not Path('minigpt4/models/projector_structure_reg.py').is_file():
        parser.error('Missing minigpt4/models/projector_structure_reg.py; unzip patch bundle first')
    src = model_path.read_text(encoding='utf-8')
    yaml_src = config_path.read_text(encoding='utf-8')
    patched_model = patch_source(src)
    patched_config = patch_config(yaml_src, a.weight, a.items)
    print('[OK] model AST parses; forward_v2 decorator, __init__, from_config patch points found')
    print('[OK] YAML patch point found; weight=%g, items=%d' % (a.weight,a.items))
    if a.check:
        print('DRY RUN ONLY: no files modified')
        return
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for f in (model_path,config_path):
        backup = f.with_name(f.name+'.before_proj_reg_'+stamp)
        shutil.copy2(f,backup)
        print('[backup]',backup)
    model_path.write_text(patched_model,encoding='utf-8')
    config_path.write_text(patched_config,encoding='utf-8')
    print('[done] projector regularization installed; existing model logic preserved')


if __name__ == '__main__':
    main()
