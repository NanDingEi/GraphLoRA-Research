#!/usr/bin/env python3
"""Prepare standalone NO31 CTR/KD control + CoRA using historical E8 recipe.

The archived S2 source/NO31 patch is committed in this GitHub branch.
Never load a trained S2, R4, or R5 checkpoint.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[2]
HISTORICAL_NO31=Path("/home/bbx/user_lx/GraphLoRA/logs/S2_NO31_NIGHT8_20261018_012408/E6_INPUT_CTR_KD/recipe.yaml")
HISTORICAL_STRICT=Path("/home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/L31.yaml")
ARCHIVED_FALLBACK=ROOT/"tran_configs/S2_groups_acc3_recovered.yaml"
LOG=ROOT/"logs/NO31_CTR_KD_CORA_20261023"
OUT=ROOT/"output/NO31_CTR_KD_CORA_20261023"


def prepare(epochs,source_override=None):
    if epochs not in (1,2):
        raise ValueError("Only one or two complete epochs supported")
    if LOG.exists() or OUT.exists():
        raise FileExistsError(f"Refuse to overwrite previous runs: {LOG} or {OUT}")
    candidates=([Path(source_override).expanduser().resolve()]
                if source_override else
                [HISTORICAL_NO31,HISTORICAL_STRICT,ARCHIVED_FALLBACK])
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:
        raise FileNotFoundError(f"Missing original recipe among {candidates}")
    cfg=yaml.safe_load(path.read_text())
    m,r=cfg["model"],cfg["run"]
    if int(m["gcn_layer"])!=31 or m["lora_gnn"]!="ngcf" or m["rec_model"]!="MF":
        raise RuntimeError("Historical MF/NGCF + L31 expected")
    if not m["gnn_semantic_align"] or not m["train_gnn_with_rec_loss"]:
        raise RuntimeError("Historical GNN training switches missing")
    if r["seed"]!=42 or int(r["max_epoch"])!=3 or int(r["accum_grad_iters"])!=3:
        raise RuntimeError("Original seed and 3-epoch scheduler must be retained")
    if int(r["batch_size_train"])!=3 or float(r["init_lr"])!=1e-4:
        raise RuntimeError("Original batch size / LR not retained")
    for group,expected in (("rec",.5),("gnn",.25),("proj",.5),("lora",1.)):
        if abs(float(r["lr_scale_"+group])-expected)>1e-9:
            raise RuntimeError(f"Unexpected historical learning rate scale {group}")
    if m.get("ckpt") or r.get("resume_ckpt_path"):
        raise RuntimeError("Unexpected jointly trained GraphLoRA/S2 checkpoint")
    if not Path(m["rec_config"]["pretrained_path"]).is_file():
        raise FileNotFoundError("Pretrained MF checkpoint required by historical E8")
    if not Path(m["llama_model"]).is_dir():
        raise FileNotFoundError("Pretrained Vicuna base directory missing")
    for term,expected in (("proj_reg_weight",.0005),("gnn_semantic_weight",.0001)):
        if abs(float(m.get(term,0))-expected)>1e-12:
            raise RuntimeError(f"Unexpected historical model setting {term}")
    if m.get("freeze_rec") is not False or m.get("freeze_lora") is not False or m.get("freeze_proj") is not False:
        raise RuntimeError("Historical CTR/KD requires trainable MF/LoRA/projector")
    # Keep original graph NPZ exact: do not regenerate the graph.
    for split in ("train","valid","test"):
        target=Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m")/f"movie_graph_{split}.npz"
        dest=ROOT/"dataset/ml-1m"/f"movie_graph_{split}.npz"
        if not target.is_file():
            raise FileNotFoundError(target)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(target)
        if not dest.is_file():
            raise FileNotFoundError(dest)
    LOG.mkdir(parents=True,exist_ok=False)
    for arm in ("CONTROL","CORA"):
        c=copy.deepcopy(cfg)
        c["model"]["gnn_scale"]=0.0
        c["model"]["gnn_conf_loss_weight"]=.05
        c["model"]["ckpt"]=None
        c["run"]["resume_ckpt_path"]=None
        c["run"]["no31_stop_after_epochs"]=epochs
        c["run"]["max_epoch"]=3
        c["run"]["output_dir"]=str(OUT/arm)
        c["run"]["evaluate"]=False
        c["run"]["valid_splits"]=["valid"]
        c["run"]["test_splits"]=[]
        c["run"]["save_checkpoint_every_n_epochs"]=1
        (LOG/f"{arm}.yaml").write_text(yaml.safe_dump(c,sort_keys=False),encoding="utf-8")
    d1=yaml.safe_load((LOG/"CONTROL.yaml").read_text())
    d2=yaml.safe_load((LOG/"CORA.yaml").read_text())
    d1["run"].pop("output_dir")
    d2["run"].pop("output_dir")
    if d1 != d2:
        raise RuntimeError("Control and CoRA configs differ outside output directory")
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    (LOG/"manifest.json").write_text(json.dumps({
        "status":"prepared","source":str(path),"source_sha256":sha,
        "original_source_commit":"d818fa2943ede4e9a810b032a3422e94d0c77ed0",
        "historical_experiment_branch":"exp/layer-injection-study-20261016",
        "gnn_scale":0,"S2_INPUT_ALPHA":0,"S2_SMALL_KD":.05,
        "gnn_ctr_weight":.05,"seed":42,"epochs":epochs,"max_epoch":3,
        "CoRA_default":"Q/V ID-token-only, gain=.1, GNN detached",
        "uses_historical_E8_recipe":path==HISTORICAL_NO31,
    },indent=2),encoding="utf-8")
    if path!=HISTORICAL_NO31:
        print("[NO31-CORA-WARN] Actual E8 YAML unavailable: using alternative S2 recipe; "
              "cannot claim exact historical E8 reproduction",flush=True)
    print(f"[NO31-CORA-PREPARED] {epochs} epoch(s); original source={path} "
          "CTR=.05 KD=.05 alpha=0 gnn_scale=0 seed=42",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--source-historical-yaml")
    args=ap.parse_args()
    prepare(args.epochs,args.source_historical_yaml)
