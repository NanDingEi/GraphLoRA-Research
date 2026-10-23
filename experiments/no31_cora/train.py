#!/usr/bin/env python3
"""Use original NO31 CTR/KD model, optionally attach the CoRA module.

Example (foreground on GPU0):
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora/train.py \
      --mode cora --cfg-path logs/NO31_CORA_20261023/CORA.yaml
"""
import argparse
import os
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    args=argparse.ArgumentParser()
    args.add_argument("--cfg-path",required=True)
    args.add_argument("--mode",choices=("cora","control"),default="cora")
    args.add_argument("--cora-gain",type=float,default=.1)
    args.add_argument("--allow-cora-to-update-gnn",action="store_true")
    opt=args.parse_args()
    cfg=Path(opt.cfg_path).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(cfg)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("NO31 CoRA must use CUDA_VISIBLE_DEVICES=0")
    os.environ["S2_INPUT_ALPHA"]="0"
    os.environ["S2_SMALL_KD"]="0.05"
    os.environ["PYTHONUNBUFFERED"]="1"
    os.chdir(ROOT)
    sys.path[:0]=[str(HERE),str(ROOT)]

    import yaml
    c=yaml.safe_load(cfg.read_text())
    m,r=c["model"],c["run"]
    if (float(m["gnn_scale"])!=0 or int(m["gcn_layer"])!=31 or
        float(m["gnn_conf_loss_weight"])!=.05 or
        not m["gnn_semantic_align"] or
        not m["train_gnn_with_rec_loss"] or
        m["rec_model"]!="MF" or m["lora_gnn"]!="ngcf"):
        raise RuntimeError("Invalid original NO31 CTR/KD model controls")
    if (r["seed"]!=42 or r["max_epoch"]!=3 or
        int(r.get("no31_stop_after_epochs",0)) not in (1,2) or
        int(r["accum_grad_iters"])!=3 or int(r["batch_size_train"])!=3):
        raise RuntimeError("Unexpected original NO31 seed/schedule/batch")
    if m.get("ckpt") or r.get("resume_ckpt_path"):
        raise RuntimeError("NO31 starts fresh with pretrained MF, not S2/R4 checkpoints")
    if not Path(m["rec_config"]["pretrained_path"]).is_file():
        raise FileNotFoundError("Original pretrained MF checkpoint missing")
    if float(m.get("proj_reg_weight",0))!=.0005 or float(m["gnn_semantic_weight"])!=.0001:
        raise RuntimeError("Original S2 semantic/projector losses not preserved")
    runtime=None
    if opt.mode=="cora":
        from runtime import install
        runtime=install(gain=opt.cora_gain,
                        detach_gnn=not opt.allow_cora_to_update_gnn)
    # Before the optimizer is built, verify that adding CoRA did not perturb
    # the pretrained MF, seeded NGCF/Projector, or original LoRA initialization.
    # The fingerprint covers all shared model tensors, not the new CoRA branch.
    import hashlib
    import torch
    from minigpt4.models import minigpt4rec_v2 as graph
    from collections import OrderedDict
    original_from_config=graph.MiniGPT4Rec_v2.from_config
    fingerprint_dir=cfg.parent
    fingerprint_file=fingerprint_dir/(opt.mode.upper()+"_init.json")
    peer_file=fingerprint_dir/(("CONTROL" if opt.mode=="cora" else "CORA")+"_init.json")

    def source_model_with_audit(cls, model_cfg):
        model=original_from_config(model_cfg)
        groups=OrderedDict((key,[]) for key in ("MF","NGCF","Projector","LoRA"))
        for name,p in model.named_parameters():
            if name.startswith("rec_encoder."):group="MF"
            elif name.startswith("gcn."):group="NGCF"
            elif name.startswith("llama_proj."):group="Projector"
            elif ".lora_A." in name or ".lora_B." in name:group="LoRA"
            else:continue
            groups[group].append((name,p))
        fingerprints={}
        for name,ps in groups.items():
            if not ps:raise RuntimeError(f"Missing original NO31 {name} weights")
            h=hashlib.sha256()
            for param_name,p in sorted(ps):
                data=p.detach().contiguous().cpu()
                h.update(param_name.encode())
                h.update(str(tuple(data.shape)).encode())
                h.update(str(data.dtype).encode())
                h.update(data.numpy().tobytes())
            fingerprints[name]=h.hexdigest()
        if fingerprint_file.exists():
            raise FileExistsError(f"NO31 init snapshot already exists: {fingerprint_file}")
        if peer_file.is_file():
            peer=json.loads(peer_file.read_text())["sha256"]
            if peer!=fingerprints:
                raise RuntimeError("NO31 Control and CoRA shared initial MF/NGCF/Projector/LoRA weights differ")
            print("[NO31-INIT-PARITY-PASS] Control and CoRA initial shared hashes identical",flush=True)
        fingerprint_file.write_text(json.dumps(
            {"mode":opt.mode,"sha256":fingerprints},indent=2),encoding="utf-8")
        print("[NO31-INIT-SHA256] "+json.dumps(fingerprints,sort_keys=True),flush=True)
        return model

    graph.MiniGPT4Rec_v2.from_config=classmethod(source_model_with_audit)

    print(f"[NO31-CORA-CONFIG] mode={opt.mode} seed=42 scale=0 "
          f"alpha=0 CTR=.05 KD=.05 epochs={r['no31_stop_after_epochs']} "
          f"cora_gain={opt.cora_gain} detach_gnn={not opt.allow_cora_to_update_gnn} "
          f"source={cfg}",flush=True)
    import train_collm_mf_din
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(cfg)]
    try:
        train_collm_mf_din.main()
    finally:
        if runtime is not None:
            data=dict(runtime.counters)
            print("[NO31-CORA-COUNTERS] "+json.dumps(data,sort_keys=True),flush=True)
            if (data.get("q_cora_forward",0)==0 or
                data.get("v_cora_forward",0)==0):
                raise RuntimeError("NO31 CoRA Q/V forward paths were not used")


if __name__=="__main__":
    main()
