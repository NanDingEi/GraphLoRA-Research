#!/usr/bin/env python3
"""No-training R3-0 probe: same 32 Valid examples, S2 vs zero-init module.

It is a correctness check, NOT model selection and NOT an AUC claim.
This uses the original S2 checkpoint; it does not evaluate the Test set.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from r3_paper_adapters import install_runtime


def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def predict(cfg,runner,model,rt,mode,n=32):
    from minigpt4.datasets.data_utils import prepare_sample
    from r2_probe import get_valid_probe_loader
    import train_collm_mf_din as entry
    rt.switch(mode)
    entry.setup_seeds(cfg)
    model.eval()
    result={k:[] for k in ("user","item","label","score")}
    for samples in get_valid_probe_loader(runner):
        if len(result["score"])>=n:
            break
        samples=prepare_sample(samples,cuda_enabled=True)
        scores=model.generate_for_samples(samples)["logits"]
        k=min(n-len(result["score"]),scores.numel())
        result["score"].extend(scores.detach().float().cpu().reshape(-1)[:k].tolist())
        for key,dst in (("UserID","user"),("TargetItemID","item"),
                        ("label","label")):
            result[dst].extend(samples[key].detach().cpu().reshape(-1)[:k].tolist())
    return {key:np.asarray(vals,dtype=np.float64 if key=="score" else np.int64)
            for key,vals in result.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=("cflora","cora"),required=True)
    p.add_argument("--config",default="tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml")
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--r1",default="logs/collab_v1/S2_R1_QV_ABLATION.npz")
    p.add_argument("--n",type=int,default=32)
    args=p.parse_args()
    if args.n!=32:
        raise RuntimeError("R3 probe must use predeclared n=32, not tune n to Valid")
    output=ROOT/"logs/collab_v1"/f"R3_{args.method.upper()}_ANCHOR.json"
    if output.exists():
        raise FileExistsError(f"R3 anchor already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU0 unavailable")
    rt=install_runtime(args.method)
    # Reuse existing S2 dataset, exact checkpoint and Valid loader, but DO NOT
    # install the earlier R2 runtime. No R2 modification is active here.
    from r2_probe import build_model
    a=argparse.Namespace(config=args.config,
                         output=str(output),checkpoint=args.checkpoint)
    cfg,runner,model,ckpt=build_model(a)
    baseline=predict(cfg,runner,model,rt,"off")
    adapted=predict(cfg,runner,model,rt,args.method)
    for k in ("user","item","label"):
        if not np.array_equal(baseline[k],adapted[k]):
            raise RuntimeError(f"R3 {args.method} Valid sample misalignment {k}")

    r1=Path(args.r1)
    if not r1.is_file():
        raise FileNotFoundError(r1)
    with np.load(r1,allow_pickle=False) as prev:
        for k in ("user","item","label"):
            if not np.array_equal(baseline[k],prev[k][:args.n]):
                raise RuntimeError(f"R3 {k} differs from original R1 sample order")
        anchor=float(np.max(np.abs(
            baseline["score"]-prev["R1_A_S2_QV"][:args.n])))
    diff=float(np.max(np.abs(adapted["score"]-baseline["score"])))
    print(f"[R3-ANCHOR] mode={args.method} native_vs_R1={anchor:.9g} "
          f"zero_init_vs_S2={diff:.9g}",flush=True)
    if anchor>2e-4 or diff>2e-4:
        raise RuntimeError("R3 initial predictions differ from strict S2 control")
    expected=("q_cf_forward","v_cf_forward") if args.method=="cflora" else (
        "q_cora_forward","v_cora_forward")
    for key in expected:
        if rt.counters.get(key,0)<1:
            raise RuntimeError(f"R3 {args.method} module did not execute: {key}")
    report={
        "status":"complete","method":args.method,"samples":args.n,
        "checkpoint":str(ckpt),"checkpoint_sha256":file_hash(ckpt),
        "max_abs_native_vs_R1":anchor,"max_abs_zero_init_vs_S2":diff,
        "counters":dict(rt.counters),
        "note":"32 samples, zero-init parity only, no AUC or Test evaluation"}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"[R3-{args.method.upper()}-ANCHOR-PASS] {output}",flush=True)


if __name__=="__main__":
    main()
