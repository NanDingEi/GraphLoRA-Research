#!/usr/bin/env python3
"""Bridge R4 checkpoint r3_mode into the local GraphLoRA R5 model constructor.

Never fabricate an adapter: abort if the actual R4-enabled model is absent.
Use with the existing r5_extend_one_epoch.py in GraphLoRA_collab_v1.
"""
import argparse
import importlib
import inspect
import os
from pathlib import Path
import runpy
import sys

ROOT = Path('/home/bbx/user_lx/GraphLoRA_collab_v1')
SOURCE = {
    '01': 'R4_01_CFLORA_ONLY/FULL_E1/20261022134/checkpoint_best.pth',
    '02': 'R4_02_CORA_ONLY/FULL_E1/20261022141/checkpoint_best.pth',
    '03': 'R4_03_CORA_V31B/FULL_E1/20261022144/checkpoint_best.pth',
}
EXPECTED_MODE = {'01': 'cflora', '02': 'cora', '03': 'cora'}
EXPECTED_TENSORS = {'02': 25, '03': 25}
CF_REQUIRED = {
    'r3_adapter.encoder.0.weight', 'r3_adapter.encoder.0.bias',
    'r3_adapter.encoder.2.weight', 'r3_adapter.encoder.2.bias',
    'r3_adapter.norm.weight', 'r3_adapter.norm.bias',
}


def load_checkpoint_metadata(round_id):
    import torch
    path = ROOT / 'output' / 'R4_ISOLATION_20261022' / SOURCE[round_id]
    if not path.is_file():
        raise FileNotFoundError(f'R4 checkpoint not found: {path}')
    try:
        data = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(str(path), map_location='cpu')
    if data.get('epoch') != 0:
        raise RuntimeError(f'Expected R4 epoch 0, got {data.get("epoch")}')
    cfg = data.get('config')
    if not isinstance(cfg, dict) or not isinstance(cfg.get('model'), dict):
        raise RuntimeError('R4 checkpoint missing model config')
    model_cfg = cfg['model']
    mode = model_cfg.get('r3_mode')
    if mode != EXPECTED_MODE[round_id]:
        raise RuntimeError(f'R4 model.r3_mode={mode!r}, expected {EXPECTED_MODE[round_id]!r}')
    keys = [k for k in data.get('model', {}) if k.startswith('r3_adapter.')]
    expected = EXPECTED_TENSORS.get(round_id)
    if expected is not None and len(keys) != expected:
        raise RuntimeError(f'R4 adapter checkpoint has {len(keys)} tensors, expected {expected}')
    if round_id == '01' and not CF_REQUIRED.issubset(set(keys)):
        raise RuntimeError(f'R4 CFLoRA missing required weights: {sorted(CF_REQUIRED - set(keys))}')
    if not data.get('optimizer'):
        raise RuntimeError('R4 checkpoint has no optimizer state; exact resume not possible')
    print(f'[R5-R3-META] round={round_id} mode={mode} adapter_tensors={len(keys)}', flush=True)
    del data
    return model_cfg


def install_constructor_bridge(model_cfg):
    import minigpt4.models
    import minigpt4.models.minigpt4rec_v2
    from minigpt4.common.registry import registry

    cls = registry.get_model_class('mini_gpt4rec_v2')
    if cls is None:
        raise RuntimeError('mini_gpt4rec_v2 model not registered; check local repo imports')
    original_init = cls.__init__
    parameters = inspect.signature(original_init).parameters
    if 'r3_mode' not in parameters:
        raise RuntimeError(
            'Current MiniGPT4Rec_v2.__init__ does not support r3_mode; '
            f'loaded from {inspect.getfile(cls)}. You are not using the R4 R3-enabled '
            'model implementation. Restore the local R4 model code and '
            'experiments/collab_injection_v1/r3_paper_adapters.py before resuming. '
            'Refusing to create a fake adapter.'
        )

    source = ROOT / 'experiments' / 'collab_injection_v1' / 'r3_paper_adapters.py'
    if not source.is_file():
        raise FileNotFoundError(f'R4 adapter implementation missing: {source}')
    importlib.import_module('experiments.collab_injection_v1.r3_paper_adapters')

    extra = {k: v for k, v in model_cfg.items() if k.startswith('r3_') and k in parameters}
    extra['r3_mode'] = model_cfg['r3_mode']
    print(f'[R5-R3-BRIDGE] model_source={inspect.getfile(cls)} '
          f'constructor_injected={extra}', flush=True)

    def r4_compatible_init(self, *args, **kwargs):
        for key, value in extra.items():
            if key in kwargs and key == 'r3_mode' and kwargs[key] != value:
                raise RuntimeError(f'Conflicting r3_mode from model factory: {kwargs[key]!r} vs R4 {value!r}')
            kwargs.setdefault(key, value)
        return original_init(self, *args, **kwargs)

    cls.__init__ = r4_compatible_init
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True, choices=SOURCE.keys())
    ap.add_argument('--probe-only', action='store_true', help='Build model, do not train')
    args = ap.parse_args()
    os.chdir(ROOT)
    model_cfg = load_checkpoint_metadata(args.round)
    cls = install_constructor_bridge(model_cfg)

    if args.probe_only:
        original_from_config = cls.from_config

        def probed_from_config(cfg):
            model = original_from_config(cfg)
            adapter = getattr(model, 'r3_adapter', None)
            if adapter is None:
                raise RuntimeError('R3 constructor ran but did not create r3_adapter; check R4 source')
            num = sum(p.numel() for p in adapter.parameters())
            print(f'[R5-R3-PROBE-PASS] round={args.round} adapter_params={num}', flush=True)
            raise SystemExit(0)

        cls.from_config = probed_from_config

    script = ROOT / 'r5_extend_one_epoch.py'
    if not script.is_file():
        raise FileNotFoundError(f'Existing R5 training script missing: {script}')
    sys.argv = [str(script), '--round', args.round]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
