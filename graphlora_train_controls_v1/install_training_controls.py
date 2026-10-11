#!/usr/bin/env python3
"""Install safe optimizer LR groups, scheduler multipliers and gradient accumulation.

Targets the runner/task/scheduler files only. Does not touch any customized model code.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import pathlib
import re
import sys

FILES = (
    'minigpt4/runners/runner_base.py',
    'minigpt4/common/optims.py',
    'minigpt4/tasks/base_task.py',
)
MARKER = 'graphlora-train-controls-v1'

OPTIMIZER_BLOCK = '''            # graphlora-train-controls-v1: per-module LR multipliers.
            lr_scales = {
                "rec": float(self.config.run_cfg.get("lr_scale_rec", 1.0)),
                "proj": float(self.config.run_cfg.get("lr_scale_proj", 1.0)),
                "gnn": float(self.config.run_cfg.get("lr_scale_gnn", 1.0)),
                "lora": float(self.config.run_cfg.get("lr_scale_lora", 1.0)),
                "other": float(self.config.run_cfg.get("lr_scale_other", 1.0)),
            }
            import math
            for key, scale in lr_scales.items():
                if not math.isfinite(scale) or scale < 0:
                    raise ValueError("Invalid LR multiplier for {}: {}".format(key, scale))

            wd_value = float(self.config.run_cfg.weight_decay)
            layer_id = str(self.config.model_cfg.get("gcn_layer", 31))
            groups = {}
            num_parameters = 0
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                print(n)
                if n.startswith("rec_encoder."):
                    part = "rec"
                elif n.startswith("llama_proj."):
                    part = "proj"
                elif n.startswith("gcn.") or n.startswith("gnn.") or \
                     "Wneck" in n or "gnn_proj" in n or \
                     (".layers." + layer_id + "." in n and
                      (n.endswith(".q_proj.bias") or n.endswith(".v_proj.bias"))):
                    part = "gnn"
                elif "lora_" in n or ".lora_" in n:
                    part = "lora"
                else:
                    part = "other"
                no_decay = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
                key = (part, no_decay)
                if key not in groups:
                    groups[key] = {
                        "params": [],
                        "weight_decay": 0.0 if no_decay else wd_value,
                        "lr_scale": lr_scales[part],
                        "group_name": part + ("_no_wd" if no_decay else "_wd"),
                    }
                groups[key]["params"].append(p)
                num_parameters += p.numel()
            logging.info("number of trainable parameters: %d", num_parameters)
            self._num_trainable_para = num_parameters > 0
            optim_params = list(groups.values())
            for group in optim_params:
                logging.info("[lr-group] %s tensors=%d scale=%.4g wd=%.4g",
                             group["group_name"], len(group["params"]),
                             group["lr_scale"], group["weight_decay"])
'''

def patch_runner(src):
    if MARKER in src:
        raise RuntimeError('Runner already patched. Do not run --apply twice.')
    method = src.find('    def optimizer(self):')
    if method < 0:
        raise RuntimeError('Cannot find optimizer(self) method')
    start = src.find('            num_parameters = 0', method)
    end = src.find('            beta2 = self.config.run_cfg.get("beta2", 0.999)', start)
    # Stop if we passed into another method.
    next_method = src.find('\n    def ', method + 10)
    next_property = src.find('\n    @property', method + 10)
    bounds = [i for i in (next_method, next_property) if i >= 0]
    limit = min(bounds) if bounds else len(src)
    if not(method < start < end < limit):
        raise RuntimeError('Optimizer build shape differs: cannot safely patch; upload runner_base.py')
    original = src[start:end]
    if ('p_wd, p_non_wd' not in original or 'optim_params = [' not in original
            or 'self.model.named_parameters()' not in original):
        raise RuntimeError('Optimizer source is customized: refusing an unsafe replacement')
    result = src[:start] + OPTIMIZER_BLOCK + src[end:]
    ast.parse(result)
    return result


def patch_schedule(src):
    if MARKER in src:
        raise RuntimeError('Scheduler already patched.')
    needle = '        param_group["lr"] = lr'
    count = src.count(needle)
    if count != 3:
        raise RuntimeError('Expected 3 scheduler assignments, found {}. Upload optims.py'.format(count))
    res = src.replace(needle, '        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)  # ' + MARKER)
    ast.parse(res)
    return res


def patch_task(src):
    if MARKER in src:
        raise RuntimeError('Task already patched.')
    use_amp = '        use_amp = scaler is not None\n'
    if src.count(use_amp) != 1:
        raise RuntimeError('Could not find unique scaler setup in BaseTask')
    src = src.replace(use_amp, use_amp + '''        # graphlora-train-controls-v1: valid accumulation factor.
        accum_grad_iters = int(accum_grad_iters)
        if accum_grad_iters < 1:
            raise ValueError("accum_grad_iters must be >= 1")
''')
    before = '''            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
'''
    after = '''            # Average each accumulation window, including the final shorter one.
            # graphlora-train-controls-v1: no extra optimizer step is introduced.
            window_size = min(
                accum_grad_iters,
                iters_per_epoch - (i // accum_grad_iters) * accum_grad_iters,
            )
            backward_loss = loss / window_size
            if use_amp:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
'''
    if src.count(before) != 1:
        raise RuntimeError('Could not find expected AMP/FP32 backward block. Upload base_task.py')
    src = src.replace(before, after)
    cond = '            if (i + 1) % accum_grad_iters == 0:'
    if src.count(cond) != 1:
        raise RuntimeError('Could not find expected optimizer update condition')
    src = src.replace(cond, '            if (i + 1) % accum_grad_iters == 0 or (i + 1) == iters_per_epoch:')
    ast.parse(src)
    return src

PATCHERS = (patch_runner, patch_schedule, patch_task)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='GraphLoRA project root')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    targets = [root / name for name in FILES]
    transformed = []
    for name, path, fn in zip(FILES, targets, PATCHERS):
        if not path.is_file():
            parser.error('File not found: ' + str(path))
        old = path.read_text(encoding='utf-8')
        try:
            new = fn(old)
        except (RuntimeError, SyntaxError) as e:
            parser.error(f'{name}: {e}')
        transformed.append((old, new))
        print('[OK] AST check & safe patch point:', name)
    if args.check:
        print('DRY RUN: files unchanged')
        return 0
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backups = []
    for path, (old, new) in zip(targets, transformed):
        backup = path.with_name(path.name + '.before_train_controls_' + stamp)
        if backup.exists():
            parser.error('Backup already exists: ' + str(backup))
        backup.write_text(old, encoding='utf-8')
        backups.append(backup)
        print('[backup]', backup)
    written = []
    try:
        for path, (_, new) in zip(targets, transformed):
            path.write_text(new, encoding='utf-8')
            written.append(path)
            print('[patched]', path)
    except Exception:
        for path, (old, _) in zip(targets, transformed):
            if path in written:
                path.write_text(old, encoding='utf-8')
        raise
    print('[DONE] optimizer LR multipliers + scheduler scales + normalized accumulation installed')
    return 0

if __name__ == '__main__':
    sys.exit(main())
