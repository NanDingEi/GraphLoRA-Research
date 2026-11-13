#!/usr/bin/env python3
"""Capture RR1/RR5 TRAIN teacher logits, one 7B model at a time.

Training-only data. No optimizer, no Test access, no environment changes.
RR1 teacher: L32 raw Yes logit.
RR5 teacher: L32 + trained user gate * (L31-L32).
"""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PROTOCOL = "CORA_STUDENT_JOINT_20261113"
CHECKPOINTS = {
    "RR1": ROOT / "output/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE/20261103092/checkpoint_best.pth",
    "RR5": ROOT / "output/CORA_RESIDUAL_ROUTING6_20261103/RR5_USER_GATE_R31/20261103153/checkpoint_best.pth",
}
TRIAL_IDS = {"RR1": "RR1_L32_BASE", "RR5": "RR5_USER_GATE_R31"}


def _data_from_runner(runner):
    # Before reorg_datasets_by_split(): {"movie_ood": {"train": dataset,...}}
    matches = []
    for group in runner.datasets.values():
        if isinstance(group, dict) and "train" in group:
            matches.append(group["train"])
    if len(matches) != 1 or not hasattr(matches[0], "__len__"):
        raise RuntimeError("Expected exactly one original TRAIN dataset; no mixing VALID/TEST")
    return matches[0]


def _load_checkpoint(model, checkpoint):
    import torch
    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("Expected trainable-only CoRA checkpoint with model state")
    saved = payload["model"]
    required = {n for n,p in model.named_parameters() if p.requires_grad}
    available = set(model.state_dict())
    if required - set(saved):
        raise RuntimeError("Checkpoint missing trainable keys: " + repr(sorted(required - set(saved))[:12]))
    if set(saved) - available:
        raise RuntimeError("Checkpoint contains unexpected keys: " + repr(sorted(set(saved)-available)[:12]))
    incompatible = model.load_state_dict(saved, strict=False)
    if set(incompatible.missing_keys) & required or incompatible.unexpected_keys:
        raise RuntimeError("Checkpoint failed strict trainable-key restoration")
    print(f"[TEACHER CKPT PASS] epoch={payload.get('epoch')} tensors={len(saved)} "
          f"trainable_keys={len(required)}", flush=True)
    return payload


def _install_capture(name, checkpoint, output):
    import torch
    from torch.utils.data import DataLoader
    from minigpt4.datasets.data_utils import prepare_sample
    from minigpt4.runners.runner_base import RunnerBase

    import cora_residual_routing6 as r6

    def capture(self):
        if output.exists():
            raise FileExistsError("Refusing overwrite of teacher predictions: " + str(output))
        dataset = _data_from_runner(self)
        model = self.unwrap_dist_model(self.model)
        _load_checkpoint(model, checkpoint)
        model.set_mode("v2")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                            collate_fn=getattr(dataset, "collater", None),
                            pin_memory=True, drop_last=False)
        records = {"labels": [], "users": [], "items": [], "scores": []}
        with torch.no_grad():
            for step, samples in enumerate(loader):
                samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
                result, final, layers, ui, err = r6._capture_live(
                    model, samples, model.generate_for_samples_v2,
                    name == "RR5")
                if name == "RR1":
                    score = layers[3].float()
                else:
                    if not hasattr(model, "cora_residual_fusion"):
                        raise RuntimeError("RR5 user gate module missing: refuse plain L32 teacher")
                    score = model.cora_residual_fusion(ui,layers,final)[0].float()
                if not torch.isfinite(score).all():
                    raise RuntimeError("Nonfinite teacher logits")
                if score.numel() != samples["label"].numel():
                    raise RuntimeError("Teacher score/sample count mismatch")
                for key, source in [("labels","label"), ("users","UserID"), ("items","TargetItemID")]:
                    records[key].append(samples[source].detach().cpu().numpy().reshape(-1))
                records["scores"].append(score.detach().cpu().numpy().reshape(-1))
                if step == 0 or (step + 1) % 2000 == 0:
                    print(f"[TEACHER CAPTURE] {name} batch={step+1} "
                          f"processed={sum(len(x) for x in records['scores'])} "
                          f"l32_error={err:.6g}", flush=True)

        data={k:np.concatenate(v) for k,v in records.items()}
        n=len(dataset)
        if any(len(v)!=n for v in data.values()):
            raise RuntimeError(f"Teacher output lengths do not match TRAIN dataset n={n}")
        ann=getattr(dataset,"annotation",None)
        if ann is None or not all(k in ann.columns for k in ("UserID","TargetItemID","label")):
            raise RuntimeError("Expected Movielens TRAIN annotation frame")
        for a,b in [("labels","label"),("users","UserID"),("items","TargetItemID")]:
            if not np.array_equal(data[a],ann[b].to_numpy().reshape(-1)):
                raise RuntimeError(f"Teacher dataset identity mismatch in {a}; abort")
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_name(output.name+".writing.npz")
        if temp.exists():
            raise RuntimeError("Incomplete temporary file already exists: " + str(temp))
        np.savez_compressed(temp, sample_index=np.arange(n,dtype=np.int64),
                            labels=data["labels"].astype(np.int64),
                            users=data["users"].astype(np.int64),
                            items=data["items"].astype(np.int64),
                            scores=data["scores"].astype(np.float32))
        os.replace(temp,output)
        print(f"[TEACHER DONE] name={name} TRAIN_only={n} output={output} "
              f"score_mean={float(data['scores'].mean()):.5g} "
              f"score_std={float(data['scores'].std()):.5g} no_test=True",
              flush=True)

    pending=[RunnerBase]
    seen=set()
    while pending:
        cls=pending.pop()
        if cls in seen: continue
        seen.add(cls)
        cls.train=capture
        pending.extend(cls.__subclasses__())


def _launch(name,checkpoint,output):
    import torch
    import cora_residual_routing6 as r6
    r6.configure()
    r6.gpu0_preflight(f"teacher_{name}")
    trial=r6.TRIAL_BY_NAME[TRIAL_IDS[name]]
    config=ROOT / "local_configs/CORA_RESIDUAL_ROUTING6_20261103" / (trial["name"]+".yaml")
    if not config.is_file():
        raise FileNotFoundError(f"Missing existing R6 config: {config}; run cora_residual_routing6.py --plan")
    r6.late.base.combo.install_combo_model(r6.late.base.combo.NAMES["C6_KD_BOTH"])
    r6.late.base._patch_runtime_classes(trial)
    if name=="RR5":
        r6.install_trial_module(trial)
    _install_capture(name,checkpoint,output)
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(config)]
    runpy.run_path(str(PROJECT/"train_collm_mf_din.py"),run_name="__main__")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member",choices=["RR1","RR5"])
    p.add_argument("--plan",action="store_true")
    p.add_argument("--run",action="store_true")
    args=p.parse_args()
    if not args.member or (args.plan==args.run):
        p.error("Specify --member RR1|RR5 and exactly one of --plan/--run")
    os.chdir(PROJECT)
    ckpt=CHECKPOINTS[args.member]
    out=ROOT/"out"/PROTOCOL/f"teacher_{args.member}_train.npz"
    config=ROOT/"local_configs/CORA_RESIDUAL_ROUTING6_20261103"/(TRIAL_IDS[args.member]+".yaml")
    print(f"[TEACHER PLAN] member={args.member} checkpoint={ckpt} exists={ckpt.is_file()} "
          f"config={config} exists={config.is_file()} output={out}",flush=True)
    if not ckpt.is_file() or not config.is_file():
        raise FileNotFoundError("Missing source checkpoint or R6 config; do not run")
    if out.exists():
        raise FileExistsError("Refusing overwrite: "+str(out))
    if args.run:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        _launch(args.member,ckpt,out)

if __name__=="__main__":
    main()
