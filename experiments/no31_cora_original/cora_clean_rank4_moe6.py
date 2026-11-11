#!/usr/bin/env python3
"""Clean CoRA-MoE: four INDEPENDENT rank-4 experts, fresh initialization.

Every expert is conditioned on the SAME complete concatenated MF user/item
embedding. A single shared official CoRA Q-Former is evaluated once; four
independent output heads each generate [B, 4096, 4] dynamic A_e.
Every Q/K/V/O projection has four distinct [4096, 4] trainable P_e.

There is NO rank-16 intermediate A, NO rank-axis slicing, NO D4 multi-granularity
neighbors, NO RR1 checkpoint or resume, NO C6 CTR/KD or residual branch.
Original pretrained MF initialization is retained, then MF is trained jointly
with Q-Former, all rank-4 heads/projections, and the router.
The Vicuna backbone remains frozen. Only train/valid are used.

Six variants: uniform, user soft-0.2, pair soft-0.2, layer-pair soft-0.2,
              pair soft-1.0, layer-pair soft-1.0.

Run from CoRA_official root:
 python experiments/no31_cora_original/cora_clean_rank4_moe6.py --cpu-test
 python experiments/no31_cora_original/cora_clean_rank4_moe6.py --plan --protocol CLEAN_R4_F1
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_clean_rank4_moe6.py --run --protocol CLEAN_R4_F1 --route-mode uniform
"""

import argparse
import inspect
import json
import math
import os
import re
import runpy
import subprocess
import sys
import textwrap
import types
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if not (ROOT / "train_collm_mf_din.py").is_file():
    raise RuntimeError("Expected CoRA_official root above experiments/no31_cora_original")
for d in (str(ROOT), str(HERE)):
    if d not in sys.path:
        sys.path.insert(0, d)

import cora_joint8 as joint

SEED = 42
N_EXPERTS = 4
RANK_PER_EXPERT = 4
N_LAYERS = 32
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
SLOTS = N_LAYERS * len(PROJECTIONS)
STEPS = 400
GROUPS = 4
ROUTER_HIDDEN = 64
DEFAULT_PROTOCOL = "CORA_CLEAN_RANK4_MOE6_20261111"
RANK_RE = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def _heads(input_dim, hidden, device=None, dtype=None):
    """Independent rank-4 heads, never one rank-16 head that is split."""
    heads = nn.ModuleList([
        nn.Linear(input_dim, hidden * RANK_PER_EXPERT) for _ in range(N_EXPERTS)
    ])
    if device is not None:
        heads.to(device=device, dtype=dtype)
    return heads


def _dynamic_rank4_heads(heads, q_states, hidden):
    pooled = q_states.mean(dim=1)  # [B, full collaborative QFormer width]
    result = []
    for head in heads:
        a = F.normalize(head(pooled), p=2, dim=-1, eps=1e-12)
        # Four unit-norm rank4 experts together match the norm of a single
        # official CoRA generator with total rank budget 16.
        result.append(a.reshape(pooled.shape[0], hidden, RANK_PER_EXPERT) / math.sqrt(N_EXPERTS))
    return torch.stack(result, dim=1)  # [B, E=4, H, r=4]


class CleanRank4MoE(nn.Module):
    def __init__(self, n_slots, hidden, user_dim, mode, strength, warmup):
        super().__init__()
        self.n_slots = int(n_slots)
        self.hidden = int(hidden)
        self.user_dim = int(user_dim)
        self.mode = str(mode)
        self.strength = float(strength)
        self.warmup = int(warmup)
        if mode not in ("uniform", "user", "pair", "layer_pair"):
            raise ValueError("Unknown mode " + mode)
        if not 0 <= strength <= 1:
            raise ValueError("Router strength outside [0,1]")
        # Each expert owns an independent rank4 output projection per target.
        self.proj = nn.Parameter(torch.zeros(n_slots, N_EXPERTS, hidden, RANK_PER_EXPERT))
        self.router = nn.Sequential(
            nn.LayerNorm(3 * user_dim),
            nn.Linear(3 * user_dim, ROUTER_HIDDEN),
            nn.GELU(),
            nn.Linear(ROUTER_HIDDEN, N_EXPERTS * (4 if mode == "layer_pair" else 1)),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self._route = None
        self._steps = 0
        self._calls = 0
        self._route_mean = None

    def begin(self, rec, samples, is_training):
        if self._route is not None:
            raise RuntimeError("Nested CoRA MoE forward")
        user_table = rec.user_embedding.weight
        item_table = rec.item_embedding.weight
        uids = samples["UserID"].long().view(-1).to(user_table.device)
        iids = samples["TargetItemID"].long().view(-1).to(item_table.device)
        user = F.normalize(user_table.index_select(0, uids).detach().float(), dim=-1)
        item = F.normalize(item_table.index_select(0, iids).detach().float(), dim=-1)
        if self.mode == "user":
            ctx = torch.cat((user, torch.zeros_like(item), torch.zeros_like(user)), dim=-1)
        else:
            ctx = torch.cat((user, item, user * item), dim=-1)
        if ctx.shape[1] != 3 * self.user_dim:
            raise RuntimeError("Unexpected MF router context dimension")
        stage_warmup = bool(is_training and self._steps < self.warmup)
        if self.mode == "uniform" or stage_warmup:
            prob = ctx.new_full((ctx.shape[0], GROUPS, N_EXPERTS), 0.25)
        else:
            logits = self.router(ctx).float()
            if self.mode == "layer_pair":
                prob = F.softmax(logits.reshape(-1, GROUPS, N_EXPERTS), dim=-1)
            else:
                prob = F.softmax(logits, dim=-1).unsqueeze(1).expand(-1, GROUPS, -1)
        route = 1.0 + self.strength * (N_EXPERTS * prob - 1.0)
        self._route = route
        self._calls = 0
        self._route_mean = prob.detach().mean(dim=(0, 1)).cpu()
        balance = 1e-3 * (prob.mean(dim=(0, 1)) - 0.25).square().sum()
        if is_training:
            self._steps += 1
        return balance

    def end(self):
        self._route = None

    def project(self, slot, x, dynamic, dropout, scaling, base_output):
        if self._route is None:
            raise RuntimeError("MoE routing context missing")
        if dynamic.ndim != 4 or dynamic.shape[1] != N_EXPERTS or dynamic.shape[-1] != RANK_PER_EXPERT:
            raise RuntimeError("Generator must provide four independent rank4 weights")
        if x.ndim != 3 or dynamic.shape[0] != x.shape[0] or dynamic.shape[2] != x.shape[2]:
            raise RuntimeError("CoRA weight / token shape mismatch")
        p = self.proj[slot]
        # BMM over four separately generated rank-4 experts; NO rank-16
        # generator, NO sliced feature channels, NO residual branch.
        xx = dropout(x).to(dtype=p.dtype)
        a = dynamic.to(dtype=p.dtype)
        z = torch.einsum("bth,behr->bter", xx, a)
        mask = self._route[:, slot // (8 * len(PROJECTIONS)), :].to(dtype=p.dtype)
        delta = torch.einsum("bter,eor->bto", z * mask[:, None, :, None], p)
        self._calls += 1
        return base_output + (delta * scaling).to(base_output.dtype)


def _patch_model(args):
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.models.ppeft.plora import Linear as PLoraLinear

    prior_init = MiniGPT4Rec_vx.__init__
    prior_train = MiniGPT4Rec_vx.forward_v2
    prior_eval = MiniGPT4Rec_vx.generate_for_samples_v2
    prior_linear = PLoraLinear.forward

    def init_clean(self, *a, **kw):
        prior_init(self, *a, **kw)
        gen = self.llama_model_lora.base_model.weight_generator
        if int(gen.r) != RANK_PER_EXPERT:
            raise RuntimeError("Clean CoRA config MUST initialize r=4")
        old_final = gen.final
        if not isinstance(old_final, nn.Linear):
            raise RuntimeError("Expected original CoRA single final head")
        hidden = int(gen.hidden_size)
        if hidden != 4096:
            raise RuntimeError("Expected Vicuna-7B hidden=4096")
        if gen.query_tokens.shape[-1] != old_final.in_features:
            raise RuntimeError("QFormer/head input dimension mismatch")
        heads = _heads(old_final.in_features, hidden, old_final.weight.device, old_final.weight.dtype)
        gen.final = heads
        # Share the original QFormer; compute all four rank4 heads independently.
        def four_rank4_forward(g, ui_embedding):
            if ui_embedding.ndim != 2 or ui_embedding.shape[-1] != g.embedding_size * 2:
                raise RuntimeError("Generator needs full concatenated MF user+item")
            encoder = ui_embedding.unsqueeze(1)
            attn = torch.ones(encoder.shape[:2], dtype=torch.long, device=encoder.device)
            queries = g.query_tokens.expand(encoder.shape[0], -1, -1)
            states = g.qformer(
                query_embeds=queries,
                encoder_hidden_states=encoder,
                encoder_attention_mask=attn,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
            return _dynamic_rank4_heads(g.final, states, hidden)
        gen.forward = types.MethodType(four_rank4_forward, gen)

        modules = {}
        for name, mod in self.llama_model_lora.named_modules():
            m = RANK_RE.search(name)
            if m and isinstance(mod, PLoraLinear):
                slot = int(m.group(1)) * len(PROJECTIONS) + PROJECTIONS.index(m.group(2))
                if slot in modules:
                    raise RuntimeError("Duplicate Q/K/V/O slot")
                if int(mod.r[mod.active_adapter]) != RANK_PER_EXPERT:
                    raise RuntimeError("Each original projection must initialize rank4")
                if tuple(mod.lora_P[mod.active_adapter].weight.shape) != (hidden, RANK_PER_EXPERT):
                    raise RuntimeError("Unexpected rank4 P shape")
                modules[slot] = mod
        if len(modules) != SLOTS:
            raise RuntimeError(f"Expected 128 Q/K/V/O targets, found {len(modules)}")
        rec = self.rec_encoder
        d = int(rec.user_embedding.weight.shape[-1])
        if int(rec.item_embedding.weight.shape[-1]) != d:
            raise RuntimeError("User and item MF embedding dimensions differ")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 304)
            moe = CleanRank4MoE(SLOTS, hidden, d, args.route_mode, args.route_strength,
                                args.warmup_microsteps)
        first_module = next(iter(modules.values()))
        original_p = first_module.lora_P[first_module.active_adapter].weight
        # New P parameters follow the original LoRA P device and dtype.
        moe = moe.to(device=original_p.device, dtype=original_p.dtype)
        self.cora_clean_rank4_moe = moe
        # Superseded old single-expert P weights do not participate in forward.
        # Leave MF and QFormer fully trainable; freeze only unused old P.
        for slot, mod in modules.items():
            mod.lora_P[mod.active_adapter].weight.requires_grad_(False)
            mod._clean_rank4_slot = slot
            mod._clean_rank4_ref = weakref.ref(self.cora_clean_rank4_moe)
        train_mf = sum(p.numel() for p in rec.parameters() if p.requires_grad)
        train_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        if not train_mf or not train_gen:
            raise RuntimeError("MF and CoRA QFormer/head must both be trainable")
        if len(gen.final) != 4 or any(h.out_features != hidden * 4 for h in gen.final):
            raise RuntimeError("Expected exactly four independent rank4 generator heads")
        print(f"[CLEAN R4 INIT] MF_trainable={train_mf} generator_trainable={train_gen} "
              f"expert_P_trainable={moe.proj.numel()} mode={args.route_mode} "
              "four independent rank4 A heads; NO D4/C6/RR1", flush=True)

    def routed_linear(self, x, w=None):
        slot = getattr(self, "_clean_rank4_slot", None)
        if slot is None:
            return prior_linear(self, x, w)
        if w is None:
            raise RuntimeError("Missing four-rank4 CoRA weights")
        weight = self.weight.T if self.fan_in_fan_out else self.weight
        result = F.linear(x, weight, self.bias)
        if self.disable_adapters or self.merged:
            raise RuntimeError("Clean CoRA MoE cannot disable or merge its adapter")
        moe = self._clean_rank4_ref()
        if moe is None:
            raise RuntimeError("Missing clean CoRA MoE module")
        return moe.project(slot, x, w, self.lora_dropout[self.active_adapter],
                           self.scaling[self.active_adapter], result)

    def wrapped(fn, train_method):
        def f(self, samples, *a, **kw):
            moe = self.cora_clean_rank4_moe
            balance = moe.begin(self.rec_encoder, samples, train_method and self.training)
            try:
                out = fn(self, samples, *a, **kw)
                if moe._calls < SLOTS:
                    raise RuntimeError(f"Only {moe._calls}/{SLOTS} rank4 projections used")
                if train_method and self.training:
                    if not isinstance(out, dict) or "loss" not in out:
                        raise RuntimeError("Expected original CoRA BCE loss dictionary")
                    out["loss"] = out["loss"] + balance
                    if moe._steps == 1 or moe._steps % 100 == 0:
                        print(f"[CLEAN R4 TRAIN] step={moe._steps} "
                              f"route={moe.mode} lambda={moe.strength} "
                              f"balance={float(balance.detach()):.8g} "
                              f"prob_mean={moe._route_mean.tolist()}", flush=True)
                return out
            finally:
                moe.end()
        return f

    MiniGPT4Rec_vx.__init__ = init_clean
    PLoraLinear.forward = routed_linear
    MiniGPT4Rec_vx.forward_v2 = wrapped(prior_train, True)
    MiniGPT4Rec_vx.generate_for_samples_v2 = wrapped(prior_eval, False)


def _paths(protocol):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol):
        raise ValueError("Unsafe protocol name")
    cfg = HERE / "local_configs" / protocol / "CLEAN_R4_MOE.yaml"
    log = HERE / "logs" / protocol / "CLEAN_R4_MOE.log"
    out = HERE / "output" / protocol / "CLEAN_R4_MOE"
    return cfg, log, out


def plan(args):
    source = Path(args.source_yaml).resolve()
    if not source.is_file():
        raise FileNotFoundError("Missing original clean CoRA base config: " + str(source))
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    run, model = cfg["run"], cfg["model"]
    if int(run["seed"]) != SEED:
        raise RuntimeError("Original CoRA config must have seed42")
    if not model["lora_config"].get("use_lora", False):
        raise RuntimeError("Expected CoRA LoRA")
    cfg_path, log, out = _paths(args.protocol)
    if cfg_path.exists() or log.exists() or out.exists():
        raise FileExistsError("Existing experiment artifacts; choose a NEW protocol")
    # Fresh single-rank4 bootstrap, then four independent rank4 experts.
    model["lora_config"]["r"] = RANK_PER_EXPERT
    model["lora_config"]["alpha"] = 2 * RANK_PER_EXPERT
    model["lora_config"]["target_modules"] = list(PROJECTIONS)
    model["freeze_rec"] = False
    model["freeze_lora"] = False
    model["ckpt"] = None
    run["seed"] = SEED
    run["init_lr"] = float(args.lr)
    if float(run.get("min_lr", 0)) > args.lr:
        run["min_lr"] = args.lr / 10
    run["batch_size_train"] = 2
    run["accum_grad_iters"] = 4
    run["iters_per_epoch"] = STEPS
    run["evaluate"] = False
    run["valid_splits"] = ["valid"]
    run["test_splits"] = []
    run["resume_ckpt_path"] = None
    run["output_dir"] = str(out)
    # The trainer's epoch iterator is uncapped at runtime, and will stop
    # only after 5 consecutive Valid AUC non-improvements. Keep 1000 merely
    # as the unchanged cosine LR horizon, not a hard training cutoff.
    run["max_epoch"] = max(int(run.get("max_epoch", 1000)), 1000)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (log.parent / "clean_rank4_manifest.json").write_text(
        json.dumps({
            "protocol": args.protocol,
            "seed": SEED, "clean_input": "raw MF user/item concat (full 512)",
            "start": "fresh CoRA QFormer and 4 independent rank4 heads",
            "mf": "pretrained baseline MF init then jointly trainable",
            "expert_rank": RANK_PER_EXPERT,
            "experts": N_EXPERTS, "combined_rank_budget": N_EXPERTS * RANK_PER_EXPERT,
            "no_rank16_intermediate": True, "no_D4": True,
            "no_RR1_checkpoint": True, "no_CTR_or_KD": True,
            "loss": "original CoRA Yes-token BCE + 1e-3 router balance",
            "mode": args.route_mode, "strength": args.route_strength,
            "warmup_microsteps": args.warmup_microsteps,
            "lr": args.lr, "epoch_cap": None,
            "stop": "5 consecutive Valid AUC evaluations without improvement",
            "global_warmup_fixed": True,
            "batch_size": 2, "gradient_accumulation": 4,
            "valid_only": True, "test_disabled": True,
            "gpu0_only": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CLEAN R4 PLAN PASS] cfg={cfg_path} log={log} output={out} "
          "epochs=uncapped patience=5", flush=True)



def _remove_training_ceiling_and_fix_warmup():
    """Experiment-local runner compatibility patch; source files unchanged.

    Some CoRA_official RunnerBase versions hardcode S2_NIGHT_EPOCHS=1/2
    in train(). That forcibly stops regardless of the externally requested
    patience. Rewrite only that trusted method in memory, refusing unknown
    implementations. The main training loop becomes itertools.count and
    stops exclusively by the existing Valid-AUC early-stop hook (patience=5).

    Also correct the original scheduler bug: warmup uses cur_step, resetting
    the LR on each short-epoch boundary. Use the accumulated global step.
    No other scheduler hyperparameter is altered.
    """
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.common.optims import (
        LinearWarmupCosineLRScheduler,
        warmup_lr_schedule,
        cosine_lr_schedule,
    )

    train_src = textwrap.dedent(inspect.getsource(RunnerBase.train))
    lines = train_src.splitlines(keepends=True)
    old_loop = "for cur_epoch in range(self.start_epoch, self.max_epoch):"
    old_night = '_night_epochs=int(__import__("os").environ["S2_NIGHT_EPOCHS"])'
    loop_idxs = [i for i, line in enumerate(lines) if line.strip() == old_loop]
    night_idxs = [i for i, line in enumerate(lines) if line.strip() == old_night]
    if len(loop_idxs) != 1 or len(night_idxs) > 1:
        raise RuntimeError(
            "Unexpected RunnerBase.train structure: refuse unsafe uncapping "
            f"loop={loop_idxs} stop={night_idxs}"
        )
    loop_i = loop_idxs[0]
    indentation = lines[loop_i][:len(lines[loop_i]) - len(lines[loop_i].lstrip())]
    lines[loop_i] = (
        indentation + 'for cur_epoch in __import__("itertools").count(self.start_epoch):\n'
    )
    # Older servers hardcode S2_NIGHT_EPOCHS=1/2; other installed RunnerBase
    # versions don't contain that block at all. Both are valid. Remove the
    # block ONLY when present, after verifying its exact six-line structure.
    if night_idxs:
        night_i = night_idxs[0]
        expected_night_lines = [
            old_night,
            "assert _night_epochs in (1,2)",
            "if cur_epoch+1 >= _night_epochs:",
            'logging.info("[NIGHT8-STOP] trained_epochs=%d max_epoch=%d "',
            '"Valid and checkpoint completed",cur_epoch+1,self.max_epoch)',
            "break",
        ]
        actual_night_lines = [v.strip() for v in lines[night_i:night_i + 6]]
        if actual_night_lines != expected_night_lines:
            raise RuntimeError(
                "Unknown RunnerBase.train forced-stop block; refusing to modify: "
                + repr(actual_night_lines)
            )
        del lines[night_i:night_i + 6]
        runner_mode = "hardcoded_1_or_2_epoch_cap_removed"
    else:
        runner_mode = "no_hardcoded_cap_present"
    train_src = "".join(lines)
    if "S2_NIGHT_EPOCHS" in train_src or old_loop in train_src:
        raise RuntimeError("Remaining hardcoded ceiling in RunnerBase.train")

    patched_globals = RunnerBase.train.__globals__
    patched_local = {}
    exec(
        compile(
            train_src,
            inspect.getsourcefile(RunnerBase.train) or "<clean-rank4-runner>",
            "exec",
        ),
        patched_globals,
        patched_local,
    )
    RunnerBase.train = patched_local["train"]
    print(
        f"[CLEAN R4 UNCAPPED PASS] runner_mode={runner_mode}; "
        "epoch loop unbounded; Valid-AUC patience=5 early stopping active",
        flush=True,
    )

    def continuous_warmup_step(self, cur_epoch, cur_step):
        total = cur_epoch * self.iters_per_epoch + cur_step
        if total < self.warmup_steps:
            warmup_lr_schedule(
                step=total,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
                total_step=total,
            )
        else:
            # Keep the original long-horizon cosine schedule unchanged.
            cosine_lr_schedule(
                epoch=total,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )

    # Install BEFORE joint.patch_runner(), which wraps step with the
    # original 0.1x MF LR multiplier. The MF split is therefore preserved.
    LinearWarmupCosineLRScheduler.step = continuous_warmup_step
    print(
        "[CLEAN R4 LR PASS] warmup uses global microstep across epochs; "
        "MF=0.1x, original LR/cosine schedule preserved",
        flush=True,
    )


def _audit_optimizer():
    from minigpt4.runners.runner_base import RunnerBase
    old = RunnerBase.optimizer.fget
    def checked(self):
        optim = old(self)
        if not getattr(self, "_clean_r4_optim_checked", False):
            model = self.unwrap_dist_model(self.model)
            actual = {id(p) for group in optim.param_groups for p in group["params"]}
            wanted = {id(p) for p in model.parameters() if p.requires_grad}
            if actual != wanted:
                raise RuntimeError(f"Optimizer mismatch: missing={len(wanted-actual)} extra={len(actual-wanted)}")
            mf = {id(p) for p in model.rec_encoder.parameters() if p.requires_grad}
            gen = model.llama_model_lora.base_model.weight_generator
            heads = {id(p) for p in gen.final.parameters() if p.requires_grad}
            moe = model.cora_clean_rank4_moe
            projectors = {id(moe.proj)}
            if not (mf and heads and projectors and (mf | heads | projectors).issubset(actual)):
                raise RuntimeError("MF, four independent rank4 heads, or expert P missing from optimizer")
            if not any(group.get("joint8_mf") for group in optim.param_groups):
                raise RuntimeError("MF LR 0.1x group missing")
            self._clean_r4_optim_checked = True
            print("[CLEAN R4 OPT PASS] MF + QFormer + 4 rank4 dynamic heads "
                  "+ 128x4 rank4 P + router; unused bootstrap P frozen", flush=True)
        return optim
    RunnerBase.optimizer = property(checked)


def worker(args):
    cfg_path, _, _ = _paths(args.protocol)
    if not cfg_path.is_file():
        raise FileNotFoundError("Run --plan first")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    # Reuse the validated MF LR split, AMP monitoring and early stopping,
    # but NOT any C6/D4 forward or multi-granularity patches.

    if args.epochs != 0 or args.patience != 5:
        raise RuntimeError("Uncapped Clean R4 protocol requires --epochs 0 --patience 5")
    _remove_training_ceiling_and_fix_warmup()
    joint.patch_runner(screen_epochs=0, mf_lr_ratio=0.1, patience=5)
    joint.install_compact_atomic_best_checkpoint()
    _patch_model(args)
    _audit_optimizer()
    print(f"[CLEAN R4 WORKER] protocol={args.protocol} from_scratch=True "
          f"MF_joint=True expert_rank=4 num_experts=4 total_rank=16 "
          f"route={args.route_mode} lambda={args.route_strength} "
          "no_RR1=True no_D4=True no_C6=True valid_only=True "
          "no_epoch_cap=True early_stop_patience=5 global_warmup_fixed=True",
          flush=True)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def gpu_check(min_free):
    check = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(check.stderr)
    for line in check.stdout.splitlines():
        vals = [v.strip() for v in line.split(",")]
        if vals[0] == "0":
            free = int(vals[1])
            print(f"[CLEAN R4 GPU] gpu0_free={free}MiB required={min_free}MiB", flush=True)
            if free < min_free:
                raise RuntimeError("Insufficient free GPU0 memory; other jobs untouched")
            return
    raise RuntimeError("Physical GPU0 unavailable")


def cpu_test():
    torch.manual_seed(SEED)
    hidden, d, batch, seq = 24, 8, 3, 5
    heads = _heads(2*d, hidden)
    queries = torch.randn(batch, 4, 2*d, requires_grad=True)
    weights = _dynamic_rank4_heads(heads, queries, hidden)
    assert tuple(weights.shape) == (batch, 4, hidden, 4)
    assert len(heads) == 4 and all(h.out_features == hidden*4 for h in heads)
    # Changing any user/item coordinate can influence every expert.
    for e in range(4):
        g = torch.autograd.grad(weights[:,e].sum(), queries, retain_graph=True)[0]
        assert (g.abs().sum(dim=(0,1)) > 0).all()
    rec = type("MF", (), {
        "user_embedding": nn.Embedding(12, d),
        "item_embedding": nn.Embedding(12, d)})()
    ids = {"UserID": torch.tensor([1,2,3]), "TargetItemID": torch.tensor([4,5,6])}
    x = torch.randn(batch, seq, hidden, requires_grad=True)
    for mode in ("uniform", "user", "pair", "layer_pair"):
        for strength in (0.2, 1.0):
            moe = CleanRank4MoE(2, hidden, d, mode, strength, warmup=0)
            with torch.no_grad():
                moe.proj.normal_(0, .01)
            loss_reg = moe.begin(rec, ids, True)
            y = moe.project(0, x, weights, nn.Identity(), 2.0,
                            torch.zeros_like(x))
            (y.square().mean()+loss_reg).backward(retain_graph=True)
            assert moe.proj.grad is not None and moe.proj.grad.abs().sum() > 0
            if mode == "uniform":
                assert all(p.grad is None for p in moe.router.parameters())
            else:
                assert all(p.grad is not None for p in moe.router.parameters())
            moe.end()
    # Preflight the ACTUAL uncapped runner patch and corrected global-step LR.
    # This prevents launching six GPU trials when the local RunnerBase differs
    # from the GitHub version or the warmup still restarts at Epoch 1.
    _remove_training_ceiling_and_fix_warmup()
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.common.optims import LinearWarmupCosineLRScheduler
    if "S2_NIGHT_EPOCHS" in str(RunnerBase.train.__code__.co_consts):
        raise RuntimeError("Forced S2_NIGHT_EPOCHS cap remained in compiled runner")
    toy = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1e-5)
    sched = LinearWarmupCosineLRScheduler(
        toy, max_epoch=1000, iters_per_epoch=400,
        min_lr=8e-5, init_lr=3e-4, warmup_steps=800, warmup_start_lr=1e-5,
    )
    sched.step(0, 399)
    prev_lr = toy.param_groups[0]["lr"]
    sched.step(1, 0)
    current_lr = toy.param_groups[0]["lr"]
    assert current_lr > prev_lr and current_lr > 1.5e-4, (
        f"Warmup discontinuity: epoch0_end={prev_lr} epoch1_start={current_lr}"
    )
    print("[CLEAN R4 TRAIN-CTRL CPU PASS] uncapped runner patch; "
          f"warmup continuous across epoch boundary {prev_lr:.8g}->{current_lr:.8g}",
          flush=True)
    print("[CLEAN R4 CPU PASS] each expert sees complete MF condition, "
          "4 independent rank4 dynamic heads, rank4 P, routing gradients", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-test", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    p.add_argument("--source-yaml", default=str(joint.BASE))
    p.add_argument("--route-mode", choices=("uniform","user","pair","layer_pair"), default="pair")
    p.add_argument("--route-strength", type=float, default=0.2)
    p.add_argument("--warmup-microsteps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=0, help="0 means no epoch ceiling")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-free-mib", type=int, default=22000)
    args = p.parse_args()
    if args.cpu_test:
        cpu_test()
        return
    if not (0 < args.lr <= 0.001 and args.epochs == 0 and args.patience == 5
            and args.warmup_microsteps >= 0 and 0 <= args.route_strength <= 1):
        p.error("Require --epochs 0 (unlimited), --patience 5, valid LR/warmup/route")
    if args.plan:
        plan(args)
        return
    if args._worker:
        worker(args)
        return
    if not args.run:
        p.error("Specify --cpu-test, --plan or --run")
    cfg_path, log, out = _paths(args.protocol)
    if not cfg_path.is_file():
        plan(args)
    if log.exists() or out.exists():
        raise FileExistsError("Refusing to overwrite existing results; use a new protocol")
    gpu_check(args.min_free_mib)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
           "--protocol", args.protocol, "--route-mode", args.route_mode,
           "--route-strength", str(args.route_strength),
           "--warmup-microsteps", str(args.warmup_microsteps),
           "--lr", str(args.lr), "--epochs", str(args.epochs),
           "--patience", str(args.patience)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    print("[CLEAN R4 RUN] " + " ".join(cmd), flush=True)
    with log.open("x", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        code = proc.wait()
    if code:
        raise SystemExit(f"[CLEAN R4 STOP] code={code} log={log}")
    print(f"[CLEAN R4 DONE] output={out} VALID only; TEST not used", flush=True)


if __name__ == "__main__":
    main()
