#!/usr/bin/env python3
"""CoRA official-source joint MF eight-run experiment suite.

- All eight trials start independently from the SAME Vicuna / pretrained MF.
- Train objective remains original CoRA LLM Yes-logit BCE.
- Auxiliary CTR: BCE(MF dot score, ground-truth label).
- Auxiliary KD: BCE(MF dot score, stopgrad(sigmoid(LLM Yes logit))).
- SEM_TEXT: MF item-item cosine matching detached mean Vicuna token
  EMBEDDING similarities of target item titles. This is a lightweight
  semantic proxy, NOT prior GraphLoRA deep-hidden-state semantic loss.
- MF_ANCHOR: squared deviation of visited MF rows from pretrained MF.
  Original GraphLoRA projector regularization is inapplicable: official
  title-only CoRA does NOT use its input projector.
- MF optimizer learning rate is 0.1x LoRA LR, with per-step scheduler patch.
- Foreground, serial, only GPU 0, VALID only; original files untouched.
"""
import argparse
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "local_configs/cora_mf_ood2_official.yaml"
RUNS = [
    ("R0_FROZEN",   True,  0,    0,    0,    0),
    ("R1_MF_MAIN",  False, 0,    0,    0,    0),
    ("R2_MF_CTR",   False, .05,  0,    0,    0),
    ("R3_MF_KD",    False, 0,    .05,  0,    0),
    ("R4_CTR_KD",   False, .05,  .05,  0,    0),
    ("R5_CTR_KD_SEM",False,.05,  .05,  1e-4, 0),
    ("R6_CTR_KD_ANC",False,.05,  .05,  0,    5e-4),
    ("R7_ALL",      False,.05,  .05,  1e-4, 5e-4),
]
PROTOCOL = "CORA_JOINT8_LR3E4_AMPFIX_20261030"
AMP_INIT_SCALE = 256.0
MAX_CONSECUTIVE_AMP_SKIPS = 12
CORA_LR = 3e-4
MF_LR_RATIO = 0.1
CONFIG_DIR = ROOT / "local_configs" / PROTOCOL
OUT_DIR = ROOT / "output" / PROTOCOL
LOG_DIR = ROOT / "logs" / PROTOCOL


def get_trial(name):
    for t in RUNS:
        if t[0] == name:
            return t
    raise ValueError(f"Unknown trial: {name}")


def check_sources():
    if not BASE.is_file():
        raise FileNotFoundError(f"Missing base config: {BASE}")
    source = (ROOT / "minigpt4/tasks/base_task.py").read_text()
    if "backward_loss = loss / accum_grad_iters" not in source:
        raise RuntimeError("Missing prior gradient accumulation averaging patch in base_task.py")
    source = (ROOT / "minigpt4/tasks/rec_base_task.py").read_text()
    if "if 'perplexity' in eval_output.keys():" not in source:
        raise RuntimeError("Missing evaluation .keys() fix in rec_base_task.py")


def prepare():
    check_sources()
    cfg0 = yaml.safe_load(BASE.read_text())
    run = cfg0["run"]
    if int(run["seed"]) != 42:
        raise RuntimeError("Do not change seed=42")
    if (int(run["batch_size_train"]), int(run["accum_grad_iters"]), int(run["iters_per_epoch"])) != (2,4,400):
        raise RuntimeError("Expected fixed BS=2/accum=4/400 microsteps; audit base config before running")
    if cfg0["model"].get("ckpt"):
        raise RuntimeError("This suite requires no stage-1/adaptor checkpoint: start all trials from same MF and fresh CoRA")
    mf_path = Path(cfg0["model"]["rec_config"]["pretrained_path"])
    if not mf_path.is_file():
        raise FileNotFoundError(f"MF checkpoint missing: {mf_path}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name,freeze,ctr,kd,sem,anc in RUNS:
        cfg = yaml.safe_load(BASE.read_text())
        # Build the updated eight-trial set without modifying the original source config.
        cfg["run"]["init_lr"] = CORA_LR
        # Keep the same warmup_lr/min_lr as the successful 3e-4 LR comparison.
        cfg["model"]["freeze_rec"] = freeze
        cfg["model"]["ckpt"] = None
        cfg["run"]["evaluate"] = False
        cfg["run"]["valid_splits"] = ["valid"]
        cfg["run"]["test_splits"] = []
        cfg["run"]["resume_ckpt_path"] = None
        cfg["run"]["output_dir"] = str(OUT_DIR / name)
        cfg["run"]["seed"] = 42
        # Intentionally retain max_epoch=1000 for ORIGINAL cosine scheduler.
        # Epoch cap is applied at runtime to RunnerBase.max_epoch only.
        assert int(cfg["run"]["max_epoch"]) == 1000
        target = CONFIG_DIR / f"{name}.yaml"
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = {
        "source_config":str(BASE), "purpose":"eight independent valid-only screens",
        "seed":42, "gpu":"0",
        "original_source_config_lr":float(run["init_lr"]),
        "cora_lr":CORA_LR, "base_lr":CORA_LR,
        "mf_lr_ratio":MF_LR_RATIO, "mf_lr_effective_peak":CORA_LR*MF_LR_RATIO,
        "warmup_lr":float(run["warmup_lr"]), "min_lr":float(run["min_lr"]),
        "batch_size_train":2, "accum_grad_iters":4,
        "iters_per_epoch":400, "scheduler_max_epoch":1000,
        "default_early_stop_patience":10, "default_epoch_cap":0,
        "amp_init_scale":AMP_INIT_SCALE,
        "max_consecutive_amp_skips":MAX_CONSECUTIVE_AMP_SKIPS,
        "sem_note":"mean frozen Vicuna INPUT token embeddings (not deep hidden states)",
        "anchor_note":"pretrained MF embedding drift; not projector regularization",
        "variants":[{"id":n,"freeze_rec":f,"ctr":c,"kd":k,"sem_text":s,"mf_anchor":a}
                    for n,f,c,k,s,a in RUNS]
    }
    (LOG_DIR / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for t in RUNS:
        print(f"{t[0]:18} frozen={t[1]} ctr={t[2]} kd={t[3]} sem={t[4]} anchor={t[5]}",flush=True)
    print(f"[JOINT8 LR] CoRA={CORA_LR:.8f}; MF={CORA_LR*MF_LR_RATIO:.8f} "
          f"(MF={MF_LR_RATIO:g}x). Seed=42, patience=10.", flush=True)
    print(f"[CONFIGS] {CONFIG_DIR} [OUTPUT] {OUT_DIR} [LOGS] {LOG_DIR}",flush=True)


def patch_runner(screen_epochs, mf_lr_ratio, patience):
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.runners.runner_base_rec import RecRunnerBase
    from minigpt4.common.optims import LinearWarmupCosineLRScheduler

    if patience < 1:
        raise ValueError("Patience must be a positive number of full validation epochs")

    # Previously the MF parameter hook raised on the *scaled* gradient
    # during backward(), before GradScaler.step/update could detect overflow
    # and safely skip the optimizer step.  Use a modest initial loss scale,
    # track actual skipped updates, and abort only if scaling never stabilizes.
    original_scaler = RunnerBase.scaler.fget

    def monitored_scaler(self):
        if self.config.run_cfg.get("amp", False) and self._scaler is None:
            import torch
            self._scaler = torch.cuda.amp.GradScaler(
                init_scale=AMP_INIT_SCALE, growth_interval=2000)
            self._joint8_amp_skip_streak = 0
            self._joint8_amp_skips_total = 0
            original_update = self._scaler.update

            def monitored_update(*args, **kwargs):
                before = float(self._scaler.get_scale())
                result = original_update(*args, **kwargs)
                after = float(self._scaler.get_scale())
                if after < before:
                    self._joint8_amp_skip_streak += 1
                    self._joint8_amp_skips_total += 1
                    print(f"[JOINT8 AMP SKIP] before_scale={before:g} "
                          f"after_scale={after:g} "
                          f"consecutive={self._joint8_amp_skip_streak} "
                          f"total={self._joint8_amp_skips_total}", flush=True)
                    if self._joint8_amp_skip_streak >= MAX_CONSECUTIVE_AMP_SKIPS:
                        raise RuntimeError(
                            "AMP consistently found nonfinite gradients and skipped "
                            f"{MAX_CONSECUTIVE_AMP_SKIPS} optimizer updates. "
                            "This is a real numerical instability, not a gradient-hook error. "
                            "Check the MF->weight_generator->dynamic LoRA backward path.")
                else:
                    if self._joint8_amp_skip_streak:
                        print(f"[JOINT8 AMP RECOVERED] loss_scale={after:g}; "
                              "optimizer update completed", flush=True)
                    self._joint8_amp_skip_streak = 0
                return result

            self._scaler.update = monitored_update
            print(f"[JOINT8 AMP INIT] loss_scale={AMP_INIT_SCALE:g}; "
                  "overflow recovery enabled; skipped optimizer updates monitored",
                  flush=True)
        return original_scaler(self)

    RunnerBase.scaler = property(monitored_scaler)
    original_eval_epoch = RecRunnerBase.eval_epoch
    original_model_to_be_trained = RunnerBase.model_to_betrained

    def early_stopping_eval(self, split_name, cur_epoch, skip_reload=False):
        result = original_eval_epoch(self, split_name, cur_epoch, skip_reload)
        if split_name != "valid" or result is None or not isinstance(cur_epoch, int):
            return result
        auc = float(result["agg_metrics"])
        if not (-float("inf") < auc < float("inf")):
            raise RuntimeError(f"Invalid Valid AUC at epoch {cur_epoch}: {auc}")
        best = getattr(self, "_joint8_best_valid_auc", -float("inf"))
        if auc > best:
            self._joint8_best_valid_auc = auc
            self._joint8_best_epoch = cur_epoch
            self._joint8_no_improve = 0
            print(f"[JOINT8 BEST] epoch={cur_epoch} valid_auc={auc:.6f}; patience reset", flush=True)
        else:
            self._joint8_no_improve = getattr(self, "_joint8_no_improve", 0) + 1
            print(f"[JOINT8 PATIENCE] epoch={cur_epoch} valid_auc={auc:.6f} "
                  f"best={best:.6f} at epoch={self._joint8_best_epoch} "
                  f"no_improve={self._joint8_no_improve}/{patience}", flush=True)
            if self._joint8_no_improve >= patience:
                self._joint8_should_stop = True
                print(f"[JOINT8 EARLY STOP] {patience} consecutive epochs without "
                      f"Valid AUC improvement; best_epoch={self._joint8_best_epoch} "
                      f"best_auc={best:.6f}. Best checkpoint preserved.", flush=True)
        return result

    def stop_if_no_improvement(self):
        if getattr(self, "_joint8_should_stop", False):
            return False
        return original_model_to_be_trained(self)

    # The original runner checks model_to_betrained() AFTER saving its best
    # checkpoint, so this stops cleanly with no exception and no source edit.
    RecRunnerBase.eval_epoch = early_stopping_eval
    RunnerBase.model_to_betrained = stop_if_no_improvement

    original_max_epoch = RunnerBase.max_epoch.fget
    if screen_epochs > 0:
        RunnerBase.max_epoch = property(
            lambda self: min(original_max_epoch(self), screen_epochs))
    # RunnerBase.lr_scheduler uses self.max_epoch. The screening cap above
    # must NOT shorten the original cosine horizon; restore 1000 on scheduler.
    original_scheduler = RunnerBase.lr_scheduler.fget
    def joint_scheduler(self):
        sched = original_scheduler(self)
        sched.max_epoch = int(self.config.run_cfg.max_epoch)
        return sched
    RunnerBase.lr_scheduler = property(joint_scheduler)
    original_optimizer = RunnerBase.optimizer.fget

    def joint_optimizer(self):
        optimizer = original_optimizer(self)
        if getattr(self, "_joint8_optimizer_split", False):
            return optimizer
        self._joint8_optimizer_split = True
        target = self.unwrap_dist_model(self.model).rec_encoder
        trainable_mf = {id(p) for p in target.parameters() if p.requires_grad}
        if not trainable_mf:
            print("[JOINT8 OPT] MF frozen; original optimizer unchanged", flush=True)
            return optimizer
        total = 0
        for group in list(optimizer.param_groups):
            params = list(group["params"])
            selected = [p for p in params if id(p) in trainable_mf]
            if not selected:
                continue
            group["params"] = [p for p in params if id(p) not in trainable_mf]
            optimizer.add_param_group({
                "params":selected, "weight_decay":group["weight_decay"],
                "joint8_mf":True, "lr":group["lr"] * mf_lr_ratio
            })
            total += sum(p.numel() for p in selected)
        if total == 0:
            raise RuntimeError("MF unfreezing requested but optimizer contains no trainable MF parameters")
        print(f"[JOINT8 OPT] trainable MF parameters={total}, LR multiplier={mf_lr_ratio}", flush=True)
        return optimizer

    RunnerBase.optimizer = property(joint_optimizer)
    old_step = LinearWarmupCosineLRScheduler.step

    def lr_step(self, cur_epoch, cur_step):
        old_step(self, cur_epoch, cur_step)
        for group in self.optimizer.param_groups:
            if group.get("joint8_mf"):
                group["lr"] *= mf_lr_ratio

    LinearWarmupCosineLRScheduler.step = lr_step
    print(f"[JOINT8 SCHED] original 1000-epoch cosine horizon, "
          f"epoch_cap={screen_epochs or 'none'}, early_stop_patience={patience}", flush=True)


def patch_model(spec):
    import torch
    import torch.nn.functional as F
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    name, freeze, ctr_w, kd_w, sem_w, anc_w = spec
    original = MiniGPT4Rec_vx.forward_v2

    def joint_forward(self, samples):
        if not hasattr(self, "_joint8_checked"):
            self._joint8_checked = True
            rec = self.rec_encoder
            flags = [p.requires_grad for p in rec.parameters()]
            if bool(any(flags)) == bool(freeze):
                raise RuntimeError(f"MF freeze mismatch: trial {name}, flags={flags}")
            print(f"[JOINT8 MODEL] {name} MF-trainable={any(flags)} "
                  f"CoRA trainable={sum(p.numel() for p in self.llama_model_lora.parameters() if p.requires_grad)}",
                  flush=True)
            if not freeze:
                # torch comes from patch_model() closure; do NOT import it here.
                # A conditional import marks it function-local for *all* batches,
                # causing UnboundLocalError once _joint8_checked is already set.
                if rec.user_embedding.weight.dtype != torch.float32:
                    raise RuntimeError("MF embeddings must be float32 parameters for safe mixed-precision updates")
                self._joint8_grad_checks = 0
                def audit(gradient):
                    if self._joint8_grad_checks < 4:
                        finite = bool(torch.isfinite(gradient).all())
                        maxabs = (gradient.detach().float().abs().max().item()
                                  if finite else float("nan"))
                        print(f"[JOINT8 MF SCALED GRAD] {name} "
                              f"backward={self._joint8_grad_checks + 1} "
                              f"finite={finite} maxabs={maxabs:.6g} "
                              "(observed BEFORE GradScaler unscale; not an update result)",
                              flush=True)
                    self._joint8_grad_checks += 1
                    return gradient
                rec.user_embedding.weight.register_hook(audit)
            if anc_w:
                self._joint8_u0 = rec.user_embedding.weight.detach().float().clone()
                self._joint8_i0 = rec.item_embedding.weight.detach().float().clone()
            self._joint8_step = 0

        captured = {}
        hook = None
        if kd_w:
            answers = [self.pos_ans[0] if int(y) == 1 else self.neg_ans[0]
                       for y in samples["label"]]
            tokens = self.llama_tokenizer(
                answers, return_tensors="pt", padding="longest", truncation=True,
                max_length=self.max_txt_len, add_special_tokens=False)
            offset = tokens.input_ids.shape[-1] + 1
            yes_id = self.llama_tokenizer(
                self.pos_ans[0], add_special_tokens=False).input_ids[0]
            def save_yes(module, inputs, output):
                captured["yes"] = output.logits[:, -offset, yes_id]
            hook = self.llama_model_lora.register_forward_hook(save_yes)
        try:
            result = original(self, samples)
        finally:
            if hook is not None:
                hook.remove()
        if freeze:
            return result

        user_ids = samples["UserID"].long()
        item_ids = samples["TargetItemID"].long()
        labels = samples["label"].float()
        mf = self.rec_encoder
        mf_logits = mf(user_ids, item_ids).float()
        extra = mf_logits.sum() * 0.0
        details = {}
        if ctr_w:
            loss_ctr = F.binary_cross_entropy_with_logits(mf_logits, labels)
            extra = extra + ctr_w * loss_ctr
            details["ctr"] = float(loss_ctr.detach())
        if kd_w:
            if "yes" not in captured:
                raise RuntimeError("LLM forward hook did not capture the Yes logits")
            teacher_prob = torch.sigmoid(
                torch.nan_to_num(captured["yes"].detach().float(), nan=0.0,
                                 posinf=1.0, neginf=-1.0))
            loss_kd = F.binary_cross_entropy_with_logits(mf_logits, teacher_prob)
            extra = extra + kd_w * loss_kd
            details["kd"] = float(loss_kd.detach())
        if sem_w:
            titles = samples["TargetItemTitle"]
            tk = self.llama_tokenizer(
                list(titles), return_tensors="pt", padding=True,
                truncation=True, max_length=64, add_special_tokens=False)
            embed_dev = self.llama_model.model.embed_tokens.weight.device
            with torch.no_grad():
                token_ids = tk.input_ids.to(embed_dev)
                mask = tk.attention_mask.to(embed_dev).unsqueeze(-1)
                title_embed = self.llama_model.model.embed_tokens(token_ids).float()
                title_embed = (title_embed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                s_llm = F.normalize(title_embed, dim=-1) @ F.normalize(title_embed, dim=-1).T
            item_emb = mf.item_encoder(item_ids).float()
            s_mf = F.normalize(item_emb, dim=-1) @ F.normalize(item_emb, dim=-1).T
            mask_pair = torch.triu(
                item_ids[:,None] != item_ids[None,:],diagonal=1)
            if mask_pair.any():
                loss_sem = ((s_mf - s_llm.to(s_mf.device).detach()) ** 2)[mask_pair].mean()
                extra = extra + sem_w * loss_sem
                details["sem_text"] = float(loss_sem.detach())
            else:
                details["sem_text"] = 0.0
        if anc_w:
            uf = mf.user_encoder(user_ids).float()
            it = mf.item_encoder(item_ids).float()
            u_ref = self._joint8_u0[user_ids].to(uf.device)
            i_ref = self._joint8_i0[item_ids].to(it.device)
            anchor = (uf-u_ref).square().mean() + (it-i_ref).square().mean()
            extra = extra + anc_w * anchor
            details["mf_anchor"] = float(anchor.detach())
        result["loss"] = result["loss"] + extra
        self._joint8_step += 1
        if self._joint8_step == 1 or self._joint8_step % 100 == 0:
            print(f"[JOINT8 LOSS] {name} step={self._joint8_step} "
                  f"main={float((result['loss']-extra).detach()):.6g} "
                  f"extra={float(extra.detach()):.6g} terms={details}",flush=True)
        return result

    MiniGPT4Rec_vx.forward_v2 = joint_forward


def worker(name, screen_epochs, patience):
    spec = get_trial(name)
    patch_runner(screen_epochs=screen_epochs, mf_lr_ratio=MF_LR_RATIO, patience=patience)
    patch_model(spec)
    cfg = CONFIG_DIR / (name+".yaml")
    saved_cfg = yaml.safe_load(cfg.read_text())
    if abs(float(saved_cfg["run"]["init_lr"]) - CORA_LR) > 1e-12:
        raise RuntimeError(f"Wrong CoRA LR for {name}: {saved_cfg['run']['init_lr']}")
    sys.argv = ["train_collm_mf_din.py", "--cfg-path",str(cfg)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def summarize():
    regex = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*([0-9.]+)")
    print("Trial                 Best Valid AUC   UAUC(at AUC best)   Evaluations")
    for name,*_ in RUNS:
        log=LOG_DIR / (name+".log")
        if not log.exists():
            print(f"{name:20} pending")
            continue
        points=[(float(x),float(y)) for x,y in regex.findall(log.read_text(errors="replace"))]
        if not points:
            print(f"{name:20} no valid result")
            continue
        idx=max(range(len(points)),key=lambda i:points[i][0])
        print(f"{name:20} {points[idx][0]:.6f}         {points[idx][1]:.6f}           {len(points)} (best index {idx})")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="Run all eight trials sequentially in foreground on GPU0")
    parser.add_argument("--screen-epochs",type=int,default=0,
                        help="Optional epoch cap; default 0 = run until Valid AUC early stopping")
    parser.add_argument("--patience",type=int,default=10,
                        help="Stop after N consecutive epochs without improving Valid AUC (default 10)")
    parser.add_argument("--only",default="",help="Optional comma-separated subset e.g. R1_MF_MAIN,R4_CTR_KD")
    parser.add_argument("--summary",action="store_true")
    parser.add_argument("--_worker",default="",help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(ROOT)
    if args._worker:
        worker(args._worker,args.screen_epochs,args.patience)
        return
    if args.summary:
        summarize()
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("--patience must be >= 1 and --screen-epochs >= 0")
    prepare()
    if not args.run:
        print("PLAN only. To start eight foreground trials: python cora_joint8.py --run",flush=True)
        return
    selected=[x.strip() for x in args.only.split(",") if x.strip()] if args.only else [t[0] for t in RUNS]
    for name in selected:
        get_trial(name)
    for name in selected:
        log=LOG_DIR / (name+".log")
        if log.exists() and log.stat().st_size:
            raise RuntimeError(f"Existing log protects previous experiment: {log}. Do not overwrite; rename it explicitly.")
        print(f"\n[JOINT8 START] {name} -- serial GPU0; log={log}",flush=True)
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--_worker",name,
             "--screen-epochs",str(args.screen_epochs),"--patience",str(args.patience)]
        with log.open("w",encoding="utf-8") as fh:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                print(line,end="",flush=True)
                fh.write(line)
            rc=proc.wait()
        if rc:
            raise SystemExit(f"[STOP ON FAILURE] {name}, exit={rc}; review {log}")
        summarize()
    print("\n[JOINT8 COMPLETE] 8 trials VALID-only; no TEST metrics.",flush=True)


if __name__ == "__main__":
    main()
