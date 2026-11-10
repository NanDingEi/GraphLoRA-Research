#!/usr/bin/env python3
"""Six controlled full-information four-expert CoRA-MoE experiments from RR1 best.

Every expert sees the COMPLETE 16-channel CoRA feature z = x @ W_ui.
Each expert independently maps 16 -> 4 -> output dimension. No splitting
of the input z, no additive residual branch, no extra Q-Former copies.
An orthogonal, dense initialization ensures that the sum of all four experts
with uniform router weights exactly reconstructs the original RR1 P(z).

We restore RR1 and continue joint training its MF and dynamic weight generator;
only superseded original P matrices are frozen. Train new expert B/P with controlled routing:
uniform, user-only, pair-shared, or pair-layer routing; lambda=0.2 / 1.0.
All variants use dense full-rank16 input to every expert. Expert-only warmup
precedes learned-router training, and the RR1 checkpoint is not overwritten.
The original C6/D4 pipeline, valid-only checkpoints, and seed42 are reused.
"""

import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import types
import weakref
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if not (ROOT / 'train_collm_mf_din.py').is_file():
    raise RuntimeError('Place under CoRA_official/experiments/no31_cora_original/')
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cora_rank4_experts5 as base

SEED = 42
NAME = 'FULLINFO4_RR1_JOINT'
DEFAULT_PROTOCOL = 'CORA_FULLINFO_MOE6_MFJOINT_FIX1_20261110'
DEFAULT_RR1_PROTOCOL = 'CORA_RESIDUAL_ROUTING6_20261103'
DEFAULT_CHECKPOINT = (HERE / 'output' / DEFAULT_RR1_PROTOCOL /
                      'RR1_L32_BASE' / '20261103092' / 'checkpoint_best.pth')
DEFAULT_SOURCE_YAML = (HERE / 'local_configs' / DEFAULT_RR1_PROTOCOL /
                       'RR1_L32_BASE.yaml')
RANK = 16
EXPERTS = 4
EXPERT_RANK = 4
LAYERS = 32
PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
N_SLOTS = LAYERS * len(PROJECTIONS)
OUT_DIM = 4096
BALANCE_WEIGHT = 1e-3
TRAIN_MICROSTEPS_PER_EPOCH = 400


def _safe_load(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


class FullInfoCoRAMoE(nn.Module):
    """Four full-input rank4 experts per Q/K/V/O projection.

    z [B,T,16] -> B_e [16,4] -> P_e [4096,4] -> [B,T,4096].
    Each B_e consumes ALL 16 latent coordinates (dense orthogonal init).
    Summing all four is rank16 capacity with only 128*4*4*16 extra small
    mixer parameters beyond a rank16 P projection bank.
    """

    def __init__(self, user_dim, warmup_microsteps=200, route_mode='pair', route_strength=0.2, balance_weight=1e-3):
        super().__init__()
        self.user_dim = int(user_dim)
        self.warmup_microsteps = int(warmup_microsteps)
        self.route_mode = str(route_mode)
        self.route_strength = float(route_strength)
        self.balance_weight = float(balance_weight)
        if self.route_mode not in ('uniform', 'user', 'pair', 'layer_pair'):
            raise ValueError('Unknown route mode: ' + self.route_mode)
        if not (0.0 <= self.route_strength <= 1.0):
            raise ValueError('Route strength must be in [0,1]')
        self.mix = nn.Parameter(torch.empty(N_SLOTS, EXPERTS, RANK, EXPERT_RANK))
        self.proj = nn.Parameter(torch.empty(N_SLOTS, EXPERTS, OUT_DIM, EXPERT_RANK))
        self.router = nn.Sequential(
            nn.LayerNorm(3 * self.user_dim),
            nn.Linear(3 * self.user_dim, 64),
            nn.GELU(),
            nn.Linear(64, EXPERTS * (4 if self.route_mode == 'layer_pair' else 1)),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self._route = None
        self._steps = 0
        self._last_probs = None
        self._last_balance = None
        self._projection_calls = 0
        self._initialized = False

    @torch.no_grad()
    def initialize_from_rr1(self, module_by_slot):
        """Rotate P and simultaneously factor it into FULL-input experts.

        Q is an orthogonal dense 16x16 matrix. B_e=Q[:,4e:4e+4],
        P_e=P_rr1 @ Q[:,4e:4e+4]. Sum_e (z @ B_e) @ P_e.T = z @ P_rr1.T.
        Unlike a coordinate split, EACH expert's B_e has dense support
        over the 16 original channels, from the first forward onward.
        """
        if set(module_by_slot) != set(range(N_SLOTS)):
            raise RuntimeError('Expected exactly 128 projection slots')
        minimum_mixing = 1.0
        for slot in range(N_SLOTS):
            mod = module_by_slot[slot]
            p = mod.lora_P[mod.active_adapter].weight
            if tuple(p.shape) != (OUT_DIM, RANK):
                raise RuntimeError(f'RR1 P shape mismatch slot={slot}: {tuple(p.shape)}')
            generator = torch.Generator(device='cpu').manual_seed(SEED + 901 + slot)
            q, _ = torch.linalg.qr(torch.randn(RANK, RANK, generator=generator))
            # Q has no deliberate coordinate exclusivity: all four blocks are
            # dense functions of the complete z. Numerical zeros are unlikely.
            minimum_mixing = min(minimum_mixing, float(q.abs().min()))
            q = q.to(device=p.device, dtype=torch.float32)
            p_f = p.detach().float()
            self.mix[slot].copy_(q.T.reshape(EXPERTS, EXPERT_RANK, RANK).transpose(-1, -2))
            self.proj[slot].copy_(
                (p_f @ q).T.reshape(EXPERTS, EXPERT_RANK, OUT_DIM).transpose(-1, -2))
        self._initialized = True
        print(f'[FULL INIT RR1] 128 projection banks; four independent full-rank16-input '
              f'rank4 experts each; expert_params={self.mix.numel()+self.proj.numel()}; '
              f'min_init_abs_mixer_entry={minimum_mixing:.8g}', flush=True)

    def begin_batch(self, rec, samples, training):
        if not self._initialized:
            raise RuntimeError('FullInfo experts not initialized from RR1')
        if self._route is not None:
            raise RuntimeError('Nested FullInfo MoE forward context')
        users = samples['UserID'].long().reshape(-1)
        items = samples['TargetItemID'].long().reshape(-1)
        u = rec.user_embedding.weight.index_select(0, users.to(rec.user_embedding.weight.device))
        v = rec.item_embedding.weight.index_select(0, items.to(rec.item_embedding.weight.device))
        u = F.normalize(u.detach().float(), dim=-1)
        v = F.normalize(v.detach().float(), dim=-1)
        if self.route_mode == 'user':
            # Same MLP shape as pair; only its first user block carries data.
            feats = torch.cat((u, torch.zeros_like(v), torch.zeros_like(u)), dim=-1)
        else:
            feats = torch.cat((u, v, u * v), dim=-1)

        warmup = bool(training and self._steps < self.warmup_microsteps)
        if self.route_mode == 'uniform' or warmup:
            probs = feats.new_full((feats.shape[0], 4, EXPERTS), 1.0 / EXPERTS)
        else:
            logits = self.router(feats).float()
            if self.route_mode == 'layer_pair':
                probs = F.softmax(logits.reshape(-1, 4, EXPERTS), dim=-1)
            else:
                probs = F.softmax(logits, dim=-1).unsqueeze(1).expand(-1, 4, -1)
        # Mean-preserving soft scaling: p=1/4 -> m=1 identically for all λ.
        self._route = 1.0 + self.route_strength * (EXPERTS * probs - 1.0)
        self._last_probs = probs.detach()
        balance = (probs.mean(dim=(0, 1)) - 1.0 / EXPERTS).square().sum()
        self._last_balance = self.balance_weight * balance
        self._projection_calls = 0
        if training:
            self._steps += 1
        return self._last_balance

    def end_batch(self):
        self._route = None

    def forward_slot(self, slot, z):
        if self._route is None:
            raise RuntimeError('Missing full-information routing context')
        if z.ndim != 3 or z.shape[-1] != RANK:
            raise RuntimeError(f'Expected z=[B,T,16], got {tuple(z.shape)}')
        if z.shape[0] != self._route.shape[0]:
            raise RuntimeError('Router batch differs from CoRA projection batch')
        # Fused computational form: four *independent* 16->4 transforms,
        # concatenated to 16 channels for efficient GEMMs, with gate repeated
        # across each four-channel expert. No input z channel slicing.
        b = self.mix[slot].permute(1, 0, 2).reshape(RANK, RANK).T.contiguous()
        p = self.proj[slot].permute(1, 0, 2).reshape(OUT_DIM, RANK).contiguous()
        z = z.to(dtype=b.dtype)
        mid = F.linear(z, b)
        weights = self._route[:, slot // (8 * len(PROJECTIONS)), :].repeat_interleave(EXPERT_RANK, dim=-1)
        mid = mid * weights.unsqueeze(1).to(dtype=mid.dtype)
        result = F.linear(mid, p)
        self._projection_calls += 1
        return result


def _attach_model():
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.models.ppeft.plora import Linear as PLoraLinear

    if getattr(MiniGPT4Rec_vx, '_fullinfo_moe_patched', False):
        raise RuntimeError('FullInfo MoE already patched in this process')
    previous_init = MiniGPT4Rec_vx.__init__
    previous_train = MiniGPT4Rec_vx.forward_v2
    previous_eval = MiniGPT4Rec_vx.generate_for_samples_v2
    rank_re = base.late.base.RANK_RE
    warmup = _OPTIONS['warmup_microsteps']
    route_mode = _OPTIONS['route_mode']
    route_strength = _OPTIONS['route_strength']
    balance_weight = _OPTIONS['balance_weight']

    def init_with_fullinfo(self, *args, **kwargs):
        previous_init(self, *args, **kwargs)
        rr1_names = {n for n, p in self.named_parameters() if p.requires_grad}
        if not rr1_names:
            raise RuntimeError('No RR1 trainable parameters before freezing')
        modules = {}
        for name, mod in self.llama_model_lora.named_modules():
            match = rank_re.search(name)
            if match and isinstance(mod, PLoraLinear):
                slot = int(match.group(1)) * 4 + PROJECTIONS.index(match.group(2))
                if slot in modules:
                    raise RuntimeError(f'Duplicate projection slot {slot}')
                if int(mod.r[mod.active_adapter]) != RANK:
                    raise RuntimeError('Original RR1 Q/K/V/O must all be rank16')
                original_p = mod.lora_P[mod.active_adapter]
                if tuple(original_p.weight.shape) != (OUT_DIM, RANK):
                    raise RuntimeError(f'Unexpected RR1 projection matrix: {tuple(original_p.weight.shape)}')
                modules[slot] = mod
        if set(modules) != set(range(N_SLOTS)):
            raise RuntimeError(f'Expected 128 projections, found {len(modules)}')
        user_dim = int(self.rec_encoder.user_embedding.weight.shape[1])
        if int(self.rec_encoder.item_embedding.weight.shape[1]) != user_dim:
            raise RuntimeError('User/item MF dimensions differ')
        # Parameter values will be loaded from RR1 before any train forward;
        # new experts will then be initialized by a dense orthogonal rotation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 310)
            self.cora_fullinfo_moe = FullInfoCoRAMoE(
                user_dim, warmup, route_mode=route_mode,
                route_strength=route_strength, balance_weight=balance_weight)
        # Keep RR1's existing MF, weight-generator and other trainable
        # parameters trainable. Only the 128 superseded P weights are bypassed
        # below, so freeze those dead weights to avoid wasting optimizer state.
        replaced_p_ids = {
            id(mod.lora_P[mod.active_adapter].weight)
            for mod in modules.values()
        }
        for _name, param in self.named_parameters():
            if id(param) in replaced_p_ids:
                param.requires_grad_(False)
        mf_trainable = sum(p.numel() for p in self.rec_encoder.parameters()
                           if p.requires_grad)
        generator_trainable = sum(
            p.numel() for p in
            self.llama_model_lora.base_model.weight_generator.parameters()
            if p.requires_grad
        )
        if mf_trainable == 0 or generator_trainable == 0:
            raise RuntimeError(
                f'RR1 joint training requires MF and generator trainable: '
                f'MF={mf_trainable} generator={generator_trainable}'
            )
        self._fullinfo_original_modules = modules  # Python dict; no new module registration.
        ref = weakref.ref(self.cora_fullinfo_moe)
        for slot, mod in modules.items():
            old_p = mod.lora_P[mod.active_adapter]
            def projection_forward(p_self, z, _slot=slot, _ref=ref):
                moe = _ref()
                if moe is None:
                    raise RuntimeError('FullInfo MoE instance not available')
                return moe.forward_slot(_slot, z)
            old_p.forward = types.MethodType(projection_forward, old_p)
        self._fullinfo_rr1_names = rr1_names
        self._fullinfo_rr1_loaded = False
        print(f'[FULL MODEL] RR1 joint: trainable_MF={mf_trainable} '
              f'trainable_generator={generator_trainable}; '
              f'128 bypassed original P weights frozen; four full-input '
              f'Rank4 experts; original_RR1_trainable_keys={len(rr1_names)}',
              flush=True)

    def wrap(fn, training_method):
        def routed(self, samples, *args, **kwargs):
            moe = self.cora_fullinfo_moe
            balance = moe.begin_batch(self.rec_encoder, samples,
                                      bool(training_method and self.training))
            try:
                result = fn(self, samples, *args, **kwargs)
                if moe._projection_calls < N_SLOTS:
                    # If the actual forward sequence only routes e.g. q/v,
                    # this check catches unexpected missing projections.
                    raise RuntimeError(f'Only {moe._projection_calls} / {N_SLOTS} '
                                       'CoRA projections called')
                if training_method and self.training:
                    if not isinstance(result, dict) or 'loss' not in result:
                        raise RuntimeError('C6 train forward must return loss')
                    result['loss'] = result['loss'] + balance
                    if moe._steps == 1 or moe._steps % 100 == 0:
                        stage = 'expert_only' if moe._steps <= moe.warmup_microsteps else 'expert_router'
                        print(f'[FULL TRAIN] microstep={moe._steps} stage={stage} '
                              f'proj_calls={moe._projection_calls} '
                              f'route={moe.route_mode} lambda={moe.route_strength} '
                              f'prob_mean={moe._last_probs.mean((0,1)).cpu().tolist()} '
                              f'balance={float(balance.detach()):.8g}', flush=True)
                return result
            finally:
                moe.end_batch()
        return routed

    MiniGPT4Rec_vx.__init__ = init_with_fullinfo
    MiniGPT4Rec_vx.forward_v2 = wrap(previous_train, True)
    MiniGPT4Rec_vx.generate_for_samples_v2 = wrap(previous_eval, False)
    MiniGPT4Rec_vx._fullinfo_moe_patched = True


def _install_rr1_loader_and_saver(checkpoint):
    from minigpt4.runners.runner_base import RunnerBase
    prev_train = RunnerBase.train
    source = Path(checkpoint).resolve()

    def train_from_rr1(self):
        model = self.unwrap_dist_model(self.model)
        if getattr(model, '_fullinfo_rr1_loaded', False):
            raise RuntimeError('RR1 loaded twice; this must be a fresh run')
        if not source.is_file() or not zipfile.is_zipfile(str(source)):
            raise FileNotFoundError('Missing or damaged RR1 checkpoint: ' + str(source))
        payload = _safe_load(source)
        if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
            raise RuntimeError('RR1 checkpoint must be trainable-only {model: state_dict}')
        original = model._fullinfo_rr1_names
        supplied = set(payload['model'])
        state = model.state_dict()
        missing = original - supplied
        unexpected = supplied - original
        wrong_shape = [n for n, w in payload['model'].items()
                       if n in state and tuple(w.shape) != tuple(state[n].shape)]
        if missing or unexpected or wrong_shape:
            raise RuntimeError('RR1 weight-key mismatch: missing={} unexpected={} shapes={}'.format(
                sorted(missing)[:12], sorted(unexpected)[:12], wrong_shape[:12]))
        if self.config.run_cfg.get('resume_ckpt_path'):
            raise RuntimeError('Do not use ordinary resume_ckpt_path with model-only RR1')
        loaded = model.load_state_dict(payload['model'], strict=False)
        if loaded.unexpected_keys:
            raise RuntimeError('RR1 state has unexpected keys: ' + str(loaded.unexpected_keys[:10]))
        model.cora_fullinfo_moe.initialize_from_rr1(model._fullinfo_original_modules)
        model._fullinfo_rr1_loaded = True
        self._fullinfo_source_checkpoint = str(source)
        self._fullinfo_rr1_epoch = int(payload.get('epoch', -1))
        remaining_rr1 = {
            n for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith('cora_fullinfo_moe.')
        }
        if not remaining_rr1:
            raise RuntimeError('RR1 MF/generator were accidentally frozen')
        print('[FULL RR1 LOADED] path={} epoch={} tensors={} '
              'MF_and_generator_trainable=True new_expert_params={}'.format(
                  source, self._fullinfo_rr1_epoch, len(supplied),
                  sum(p.numel() for p in model.cora_fullinfo_moe.parameters())
              ), flush=True)
        optimizer = self.optimizer
        in_optimizer = {
            id(p) for group in optimizer.param_groups for p in group['params']
        }
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        mf_params = {id(p) for p in model.rec_encoder.parameters() if p.requires_grad}
        stale_p_params = {
            id(mod.lora_P[mod.active_adapter].weight)
            for mod in model._fullinfo_original_modules.values()
        }
        if in_optimizer != expected:
            raise RuntimeError(
                'Optimizer must include exactly the current trainable MF, '
                'CoRA generator, experts, and router; '
                f'missing={len(expected-in_optimizer)} '
                f'extra={len(in_optimizer-expected)}'
            )
        if not mf_params or not mf_params.issubset(in_optimizer):
            raise RuntimeError('MF parameters missing from joint optimizer')
        if not any(group.get('joint8_mf') for group in optimizer.param_groups):
            raise RuntimeError('Expected Joint8 0.1x MF learning-rate group')
        if stale_p_params & in_optimizer:
            raise RuntimeError('Superseded original P weights must remain frozen')
        print('[FULL OPT PASS] joint MF + original CoRA generator + new '
              'experts/router; replaced original P frozen; fresh AdamW, '
              'no RR1 optimizer resume', flush=True)
        return prev_train(self)

    def save_self_contained(self, cur_epoch, is_best=False, **kwargs):
        model = self.unwrap_dist_model(self.model)
        if not getattr(model, '_fullinfo_rr1_loaded', False):
            raise RuntimeError('Refusing to save before RR1 checkpoint was loaded')
        if not is_best:
            # This experiment only needs validation-selected best checkpoints.
            return
        target = Path(self.output_dir) / 'checkpoint_best.pth'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # The normal training loop can improve its previous best; replace
            # only this experiment's own file, never the source RR1 file.
            pass
        state = model.state_dict()
        selected = set(model._fullinfo_rr1_names) | {
            n for n, p in model.named_parameters()
            if n.startswith('cora_fullinfo_moe.') and p.requires_grad
        }
        # Stage warmup counter is not needed for inference; it uses learned
        # routing, initialized uniformly. Include all necessary tensor weights.
        if not selected.issubset(set(state)):
            raise RuntimeError('Checkpoint missing selected model keys')
        payload = {
            'model': {k: state[k].detach().cpu() for k in sorted(selected)},
            'epoch': int(cur_epoch),
            'config': self.config.to_dict(),
            'fullinfo_moe': {
                'source_rr1_checkpoint': self._fullinfo_source_checkpoint,
                'source_rr1_epoch': self._fullinfo_rr1_epoch,
                'rank': RANK, 'experts': EXPERTS, 'expert_rank': EXPERT_RANK,
                'architecture': 'full-rank16-input four independent rank4 experts',
                'training_microsteps': model.cora_fullinfo_moe._steps,
                'backbone_frozen': False,
                'original_P_replaced_and_frozen': True,
                'mf_and_generator_joint_training': True,
                'route_mode': model.cora_fullinfo_moe.route_mode,
                'route_strength': model.cora_fullinfo_moe.route_strength,
            },
        }
        tmp = target.with_name(target.name + '.writing')
        if tmp.exists():
            raise FileExistsError('Unfinished temporary checkpoint: ' + str(tmp))
        est_bytes = sum(t.numel() * t.element_size() for t in payload['model'].values())
        if shutil.disk_usage(target.parent).free < int(est_bytes * 1.5) + 128 * 1024**2:
            raise OSError('Insufficient disk space to safely save MoE best checkpoint')
        try:
            torch.save(payload, str(tmp))
            with zipfile.ZipFile(tmp) as archive:
                if archive.testzip() is not None:
                    raise IOError('MoE checkpoint ZIP CRC error')
            os.replace(tmp, target)
        except BaseException:
            print('[FULL SAVE FAILED] preserving temporary file ' + str(tmp), flush=True)
            raise
        print('[FULL BEST SAVED] epoch={} path={} keys={} includes_rr1_weights=True'.format(
            cur_epoch, target, len(selected)), flush=True)

    RunnerBase.train = train_from_rr1
    RunnerBase._save_checkpoint = save_self_contained


def _paths(protocol):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', protocol):
        raise ValueError('Unsafe protocol name')
    return (HERE / 'local_configs' / protocol / (NAME + '.yaml'),
            HERE / 'logs' / protocol / (NAME + '.log'),
            HERE / 'output' / protocol / NAME)


def plan(args):
    source_yaml = Path(args.source_yaml).resolve()
    source_ckpt = Path(args.rr1_checkpoint).resolve()
    if not source_yaml.is_file() or not source_ckpt.is_file():
        raise FileNotFoundError(f'Missing RR1 config or checkpoint: {source_yaml}, {source_ckpt}')
    if not zipfile.is_zipfile(str(source_ckpt)):
        raise RuntimeError('RR1 checkpoint is not a valid PyTorch ZIP file')
    cfg = yaml.safe_load(source_yaml.read_text(encoding='utf-8'))
    run = cfg['run']
    if (int(run['seed']) != SEED or run.get('valid_splits') != ['valid'] or
            int(run['batch_size_train']) != 2 or
            int(run['accum_grad_iters']) != 4 or
            int(run['iters_per_epoch']) != TRAIN_MICROSTEPS_PER_EPOCH or
            cfg['model']['freeze_rec'] is not False):
        raise RuntimeError('RR1 config controls differ from expected seed42/b2/accum4/400/valid/MF')
    cfg_path, log_path, output_dir = _paths(args.protocol)
    if cfg_path.exists() or log_path.exists() or output_dir.exists():
        raise FileExistsError('Protocol already has artifacts; choose another --protocol')
    run['init_lr'] = float(args.lr)
    if float(run.get('min_lr', 0)) > float(args.lr):
        run['min_lr'] = float(args.lr) * 0.1
    run['lr_scale_other'] = 1.0
    run['output_dir'] = str(output_dir)
    run['seed'] = SEED
    run['evaluate'] = False
    run['valid_splits'] = ['valid']
    run['test_splits'] = []
    run['resume_ckpt_path'] = None
    cfg['model']['ckpt'] = None
    # RR1 pretraining rec and CoRA trainables must be constructed first;
    # we freeze them only AFTER loading the selected RR1 best state.
    cfg['model']['freeze_rec'] = False
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    manifest = {
        'protocol': args.protocol, 'model': NAME, 'seed': SEED,
        'source_rr1_best_checkpoint': str(source_ckpt),
        'source_rr1_yaml': str(source_yaml), 'rr1_valid_auc_reference': 0.732387,
        'new_expert_parameters': N_SLOTS * (EXPERTS*RANK*EXPERT_RANK + EXPERTS*OUT_DIM*EXPERT_RANK),
        'expert_init': 'dense orthogonal RR1-equivalent full-input experts',
        'router': args.route_mode,
        'route_strength': args.route_strength,
        'balance_weight': args.balance_weight,
        'warmup_microsteps': args.warmup_microsteps,
        'lr': float(args.lr),
        'epochs_cap': args.epochs, 'test': False,
        'loss': 'original RR1 C6 + mean-route balance * balance_weight',
        'train': 'RR1 warm start; joint MF, dynamic generator, full-input experts and router; replaced old P frozen; fresh optimizer',
        'checkpoint': 'trainable-only original RR1 keys plus MoE new parameters',
    }
    (log_path.parent / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[FULL PLAN PASS] YAML=' + str(cfg_path), flush=True)
    print('[FULL PLAN PASS] RR1=' + str(source_ckpt), flush=True)
    print('[FULL PLAN PASS] LOG=' + str(log_path), flush=True)
    print('[FULL PLAN PASS] OUT=' + str(output_dir), flush=True)


def worker(args):
    cfg_path, _, _ = _paths(args.protocol)
    if not cfg_path.is_file():
        raise FileNotFoundError('Run --plan first: ' + str(cfg_path))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Only physical GPU0 is permitted: CUDA_VISIBLE_DEVICES=0')
    base.configure(args.protocol)
    core = base.late.base.core
    core.patch_runner(screen_epochs=args.epochs,
                      mf_lr_ratio=base.MF_LR_RATIO, patience=args.patience)
    base.late.base.combo.install_combo_model(base.late.base.combo.NAMES['C6_KD_BOTH'])
    trial = {'name': NAME, 'mode': 'neighbor_mix', 'personal_share': 0.25}
    base.late.base._patch_runtime_classes(trial)
    _attach_model()
    _install_rr1_loader_and_saver(args.rr1_checkpoint)
    print('[FULL WORKER] RR1 joint MF+CoRA; replaced original P frozen; D4+C6 kept; independent 128 projection full-input expert banks; '
          f'route={args.route_mode} lambda={args.route_strength}; Valid-only; no TEST', flush=True)
    sys.argv = ['train_collm_mf_din.py', '--cfg-path', str(cfg_path)]
    runpy.run_path(str(ROOT / 'train_collm_mf_din.py'), run_name='__main__')


def preflight(min_free):
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError('nvidia-smi error: ' + p.stderr)
    for line in p.stdout.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if fields[0] == '0':
            free = int(fields[1])
            print(f'[FULL GPU] physical_GPU0_free={free}MiB minimum={min_free}MiB', flush=True)
            if free < min_free:
                raise RuntimeError('GPU0 insufficient free memory; will not kill other jobs')
            return
    raise RuntimeError('GPU0 not found')


def cpu_test():
    torch.manual_seed(SEED)
    batch, seq, user_dim = 3, 5, 8
    moe = FullInfoCoRAMoE(user_dim=user_dim, warmup_microsteps=1, route_mode='pair', route_strength=0.2)
    class DummyP(nn.Module):
        def __init__(self):
            super().__init__()
            self.active_adapter = 'default'
            self.lora_P = nn.ModuleDict({'default': nn.Linear(RANK, OUT_DIM, bias=False)})
    modules = {i: DummyP() for i in range(N_SLOTS)}
    original = modules[0].lora_P['default'].weight.detach().clone()
    moe.initialize_from_rr1(modules)
    B = moe.mix[0]
    assert all(torch.count_nonzero(B[e].abs() > 1e-7) == RANK*EXPERT_RANK
               for e in range(EXPERTS)), 'Each expert must read every input coordinate'
    z = torch.randn(batch, seq, RANK)
    rec = type('Rec', (), {'user_embedding': nn.Embedding(15, user_dim),
                           'item_embedding': nn.Embedding(15, user_dim)})()
    samples = {'UserID': torch.tensor([1, 2, 3]),
               'TargetItemID': torch.tensor([4, 5, 6])}
    moe.train()
    moe.begin_batch(rec, samples, True)
    y = moe.forward_slot(0, z)
    reference = F.linear(z, original)
    torch.testing.assert_close(y, reference, atol=1.5e-5, rtol=1.5e-5)
    loss = y.square().mean() + moe._last_balance
    loss.backward()
    assert moe.mix.grad is not None and moe.mix.grad[0].abs().sum() > 0
    assert moe.proj.grad is not None and moe.proj.grad[0].abs().sum() > 0
    assert all(p.grad is None for p in moe.router.parameters())
    moe.end_batch()
    moe.zero_grad(set_to_none=True)
    with torch.no_grad():
        moe.proj[0,0].add_(0.01)
    moe.begin_batch(rec, samples, True)
    second = moe.forward_slot(0, z)
    (second.square().mean() + moe._last_balance).backward()
    assert (second-reference).abs().max() > 0
    assert all(p.grad is not None for p in moe.router.parameters())
    moe.end_batch()
    expert_count = moe.mix.numel() + moe.proj.numel()
    expected = N_SLOTS * EXPERTS * EXPERT_RANK * (RANK+OUT_DIM)
    assert expert_count == expected, (expert_count, expected)
    # Controlled routing tests: every variant starts as RR1 under uniform logits,
    # reads every rank16 input coordinate, and receives nonzero expert gradients.
    for mode in ('uniform', 'user', 'pair', 'layer_pair'):
        for strength in (0.2, 1.0):
            t = FullInfoCoRAMoE(user_dim, warmup_microsteps=1,
                                route_mode=mode, route_strength=strength)
            t.initialize_from_rr1(modules)
            t.train()
            t.begin_batch(rec, samples, True)
            y0 = t.forward_slot(0, z)
            torch.testing.assert_close(y0, reference, atol=1.5e-5, rtol=1.5e-5)
            (y0.square().mean() + t._last_balance).backward()
            assert t.mix.grad is not None and t.proj.grad is not None
            t.end_batch()
            t.zero_grad(set_to_none=True)
            with torch.no_grad():
                t.proj[0, 0].add_(0.01)
            t.begin_batch(rec, samples, True)
            y1 = t.forward_slot(0, z)
            (y1.square().mean() + t._last_balance).backward()
            if mode == 'uniform':
                assert all(p.grad is None for p in t.router.parameters())
            else:
                assert all(p.grad is not None for p in t.router.parameters())
                assert torch.isfinite(torch.stack([p.grad.abs().sum()
                        for p in t.router.parameters()])).all()
            t.end_batch()
    print(f'[FULL6 CPU PASS] 4 full-input route modes, 2 strengths; RR1 exact identity; '
          f'expert gradients and learned router gradients; '
          f'expert_params={expert_count}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-test', action='store_true')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--rr1-checkpoint', default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--source-yaml', default=str(DEFAULT_SOURCE_YAML))
    p.add_argument('--protocol', default=DEFAULT_PROTOCOL)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup-microsteps', type=int, default=200)
    p.add_argument('--route-mode', choices=['uniform','user','pair','layer_pair'], default='pair')
    p.add_argument('--route-strength', type=float, default=0.2)
    p.add_argument('--balance-weight', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--min-free-mib', type=int, default=22000)
    args = p.parse_args()
    if args.cpu_test:
        cpu_test()
        return
    if not (0 < args.lr <= 1e-3 and
            args.epochs >= 2 and args.patience >= 1 and args.warmup_microsteps >= 1 and
            0 <= args.route_strength <= 1 and args.balance_weight >= 0):
        p.error('Require lr in (0,1e-3], epochs>=2, warmup>=1')
    if args.plan:
        plan(args)
        return
    if args._worker:
        worker(args)
        return
    if not args.run:
        p.error('Use --cpu-test, --plan, or --run')
    cfg_path, log, output_dir = _paths(args.protocol)
    if not cfg_path.exists():
        plan(args)
    if log.exists() or output_dir.exists():
        raise FileExistsError('Existing results, choose a NEW --protocol: ' + str(log))
    preflight(args.min_free_mib)
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--_worker',
           '--protocol', args.protocol, '--rr1-checkpoint', args.rr1_checkpoint,
           '--source-yaml', args.source_yaml, '--lr', str(args.lr),
           '--warmup-microsteps', str(args.warmup_microsteps),
           '--route-mode', args.route_mode, '--route-strength', str(args.route_strength),
           '--balance-weight', str(args.balance_weight),
           '--epochs', str(args.epochs), '--patience', str(args.patience)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    print('[FULL RUN] ' + ' '.join(cmd), flush=True)
    with log.open('x', encoding='utf-8') as fp:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            fp.write(line)
        code = proc.wait()
    if code:
        raise SystemExit(f'[FULL STOP] exit={code}; log={log}; source RR1 unmodified')
    print(f'[FULL COMPLETE] log={log}; valid-only output={output_dir}', flush=True)


_OPTIONS = {}
if __name__ == '__main__':
    # Hook installation only happens in the isolated training worker.
    args_ = sys.argv[1:]
    def option(name, default, cast):
        pos = ('--' + name)
        return cast(args_[args_.index(pos)+1]) if pos in args_ else default
    _OPTIONS = {'warmup_microsteps': option('warmup-microsteps', 200, int),
                'route_mode': option('route-mode','pair',str),
                'route_strength': option('route-strength',0.2,float),
                'balance_weight': option('balance-weight',1e-3,float)}
    main()
