#!/usr/bin/env python3
"""R1-1: S2 L31 Q/V ablation, inference only, native Yes logit.

S2 checkpoint loaded once. Change only MyLinear.gnn_scale per Q and V in
five deterministic sequential conditions; no optimizer or parameter change.
Use identical unshuffled Valid Small and compare rowwise sample identities.
Follow the established eval_s2_multilayer_vote.py runner protocol.

Outputs .npz and a small .json manifest; R1-2 analyzes ranking on CPU.
NOTE: OFF is a post-hoc ablation of a trained S2, not a retrained no-GNN
model. GNN may still influence upstream graph-related features.
"""
import argparse
import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import train_collm_mf_din as entry
from minigpt4.common.config import Config
from minigpt4.tasks.rec_base_task import uAUC_pandas

EXPECTED_S2_AUC = 0.7296752596287361
EXPECTED_S2_UAUC = 0.6841670211619721
# Defined before looking at any R1 measurements.
# Ordered: native S2 anchor, no injection, V-only, Q-only, V half-strength.
ARMS = (
    ("R1_A_S2_QV", 0.1, 0.1),
    ("R1_B_OFF", 0.0, 0.0),
    ("R1_C_V_ONLY", 0.0, 0.1),
    ("R1_D_Q_ONLY", 0.1, 0.0),
    ("R1_E_V_HALF", 0.0, 0.05),
)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def select_l31_modules(model):
    """Require exactly one legacy MyLinear for L31 Q and one for L31 V."""
    if getattr(model, "collab_mode", None) != "legacy":
        raise RuntimeError("R1 requires S2 collab_mode=legacy")
    selected = {}
    for mod in model.my_linears:
        if int(mod.layer_idx) != 31 or int(mod.gcn_layer) != 31:
            raise RuntimeError("R1 only supports L31 Q/V injection")
        tag = mod.module_tag
        if tag in selected or tag not in ("q", "v"):
            raise RuntimeError(f"Duplicate/unexpected L31 adapter tag: {tag}")
        if mod.collab_mode != "legacy":
            raise RuntimeError(f"Non-legacy MyLinear {tag}")
        selected[tag] = mod
    if set(selected) != {"q", "v"}:
        raise RuntimeError(f"Expected exactly L31 q/v adapters; got {list(selected)}")
    for tag in ("q", "v"):
        if abs(float(selected[tag].gnn_scale)-0.1)>1e-10:
            raise RuntimeError(f"Initial S2 {tag} gnn_scale != 0.1")
    return selected


def set_scales(modules, beta_q, beta_v):
    """Modify ONLY scalar injection coefficients; no new graph or LLM path."""
    if not all(np.isfinite([beta_q, beta_v])):
        raise ValueError("Nonfinite injection coefficient")
    for tag, beta in (("q", beta_q), ("v", beta_v)):
        if not 0 <= beta <= 0.1:
            raise ValueError(f"Unsupported {tag} beta={beta}")
        modules[tag].gnn_scale = float(beta)
        modules[tag].warmup_factor = 1.0
        if abs(float(modules[tag].gnn_scale)-beta) > 1e-12:
            raise RuntimeError("Scale assignment failed")


def metric(user, label, scores):
    if not all(len(x)==len(user) for x in (label,scores)):
        raise RuntimeError("Metric arrays are not aligned")
    auc=float(roc_auc_score(label,scores))
    uauc, valid, _=uAUC_pandas(user,scores,label)
    return {"auc":auc,"uauc":float(uauc),"eligible_users":len(valid)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml")
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--result",default="logs/collab_v1/S2_R1_QV_ABLATION.json")
    p.add_argument("--expected-samples",type=int,default=5200)
    p.add_argument("--auc-tolerance",type=float,default=2e-5)
    p.add_argument("--uauc-tolerance",type=float,default=2e-5)
    args=p.parse_args()

    cfgpath=Path(args.config).resolve()
    result=Path(args.result).resolve()
    npz=result.with_suffix(".npz")
    if not cfgpath.is_file():
        raise FileNotFoundError(
            f"{cfgpath}; run experiments/collab_injection_v1/prepare_e3_configs.py")
    if result.exists() or npz.exists():
        raise FileExistsError("R1 will not overwrite a previous experiment; "
                              "choose a new --result or archive old files")
    result.parent.mkdir(parents=True,exist_ok=True)
    cfg=Config(Namespace(cfg_path=str(cfgpath),options=None))
    if str(cfg.model_cfg.collab_mode)!="legacy" or (
        int(cfg.model_cfg.gcn_layer)!=31 or
        abs(float(cfg.model_cfg.gnn_scale)-0.1)>1e-10
    ):
        raise RuntimeError("R1 must load original S2 L31 legacy beta=0.1")
    if float(cfg.model_cfg.get("l30_aux_weight",0))!=0:
        raise RuntimeError("Do not use P0 L30 auxiliary configuration for R1")
    checkpoint=Path(args.checkpoint or cfg.model_cfg.ckpt).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg.model_cfg.ckpt=str(checkpoint)
    cfg.run_cfg.distributed=False
    cfg.run_cfg.resume_ckpt_path=None
    cfg.run_cfg.evaluate=False
    cfg.run_cfg.output_dir=str(result.parent/"R1_EVAL_ARTIFACTS")

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task=entry.tasks.setup_task(cfg)
    datasets=task.build_datasets(cfg)
    try:
        datapath=cfg.datasets_cfg.movie_ood.path
    except Exception:
        datapath=cfg.datasets_cfg.amazon_ood.path
    # Test metadata is used ONLY to size embedding vocab, as in original S2
    # evaluation; no Test inference, Test label metrics or Test selection.
    tables=[pd.read_pickle(datapath+t+"_ood2.pkl")
            for t in ("train","valid","test")]
    cfg.model_cfg.rec_config.user_num=int(max(
        df.uid.max() for df in tables)+1)
    cfg.model_cfg.rec_config.item_num=int(max(
        df.iid.max() for df in tables)+1)
    del tables

    raw=task.build_model(cfg)
    runner=entry.get_runner_class(cfg)(
        cfg=cfg,job_id=entry.now(),task=task,model=raw,datasets=datasets)
    runner.set_model_mode(cfg.run_cfg.mode)
    model=runner.unwrap_dist_model(runner.model)
    model.eval()
    if model.run_mode_!="v2":
        raise RuntimeError("S2 must be evaluated with native Yes/No forward v2")
    modules=select_l31_modules(model)
    if any(p.requires_grad and p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected pre-existing parameter gradients")
    original_generate=model.generate_for_samples
    scores={}
    base_ids=None
    metrics={}
    signature=None

    try:
        for name,beta_q,beta_v in ARMS:
            set_scales(modules,beta_q,beta_v)
            ids={"user":[],"item":[],"label":[]}
            score=[]
            call_counter=[0]

            def capture(samples,*pos_args,**kw_args):
                if pos_args or kw_args:
                    raise RuntimeError("Nonstandard generate_for_samples call")
                if bool(samples.get("is_test",False)):
                    raise RuntimeError("R1 must not score Test split")
                with torch.no_grad():
                    output=original_generate(samples)
                v=output.get("logits")
                if not torch.is_tensor(v):
                    raise RuntimeError("Native L31 Yes-logit missing")
                y=samples["label"].detach().cpu().reshape(-1)
                u=samples["UserID"].detach().cpu().reshape(-1)
                t=samples["TargetItemID"].detach().cpu().reshape(-1)
                x=v.detach().float().cpu().reshape(-1)
                if not (len(u)==len(t)==len(y)==len(x)):
                    raise RuntimeError("Prediction/sample alignment failed")
                ids["user"].extend(u.tolist())
                ids["item"].extend(t.tolist())
                ids["label"].extend(y.tolist())
                score.extend(x.tolist())
                call_counter[0]+=1
                if call_counter[0]%260==0:
                    print(f"[R1-PROGRESS] {name} batches={call_counter[0]} "
                          f"examples={len(score)}",flush=True)
                return output

            model.generate_for_samples=capture
            # Same seed before every arm; standard Valid dataloader and native
            # runner evaluation. No optimizer steps or model re-instantiation.
            entry.setup_seeds(cfg)
            print(f"[R1-START] {name} Q={beta_q} V={beta_v}",flush=True)
            result_dict=runner.eval_epoch(
                split_name="valid",cur_epoch=0,skip_reload=True)
            model.generate_for_samples=original_generate
            if len(score)!=args.expected_samples:
                raise RuntimeError(
                    f"{name}: expected {args.expected_samples}, got {len(score)}")
            user=np.asarray(ids["user"],dtype=np.int64)
            item=np.asarray(ids["item"],dtype=np.int64)
            label=np.asarray(ids["label"],dtype=np.int64)
            logits=np.asarray(score,dtype=np.float64)
            if not np.isfinite(logits).all():
                raise RuntimeError(f"{name} contains NaN/Inf logits")
            if set(np.unique(label).tolist())!={0,1}:
                raise RuntimeError("Expected both labels 0 and 1")
            if base_ids is None:
                base_ids={"user":user,"item":item,"label":label}
            elif any(not np.array_equal(base_ids[k],v)
                     for k,v in (("user",user),("item",item),("label",label))):
                raise RuntimeError(
                    f"{name} changed Valid order or sample identity; "
                    "refusing to compare predictions")
            row=metric(user,label,logits)
            if abs(float(result_dict["agg_metrics"])-row["auc"])>1e-8:
                raise RuntimeError(f"{name} native runner AUC mismatch")
            if name=="R1_A_S2_QV" and (
                abs(row["auc"]-EXPECTED_S2_AUC)>args.auc_tolerance or
                abs(row["uauc"]-EXPECTED_S2_UAUC)>args.uauc_tolerance
            ):
                raise RuntimeError(f"S2 anchor failed {row}; stop all arms")
            scores[name]=logits
            row.update({"beta_q":beta_q,"beta_v":beta_v})
            metrics[name]=row
            print(f"[R1-RESULT] {name} AUC={row['auc']:.9f} "
                  f"UAUC={row['uauc']:.9f} Q={beta_q} V={beta_v}",flush=True)
    finally:
        model.generate_for_samples=original_generate
        set_scales(modules,0.1,0.1)

    if len(scores)!=len(ARMS):
        raise RuntimeError("Not all R1 arms completed")
    if np.array_equal(scores["R1_A_S2_QV"],scores["R1_B_OFF"]):
        raise RuntimeError(
            "Turning both L31 Q/V GNN injections OFF left all Yes logits "
            "identical: verify MyLinear scale wiring before analyzing R1")
    if np.array_equal(scores["R1_C_V_ONLY"],scores["R1_E_V_HALF"]):
        print("[R1-WARNING] V=0.1 and V=0.05 produced identical scores; "
              "investigate scale effect",flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Evaluation unexpectedly produced parameter gradients")
    for name,_,_ in ARMS:
        metrics[name]["delta_auc_vs_s2"]=metrics[name]["auc"]-metrics[
            "R1_A_S2_QV"]["auc"]
        metrics[name]["delta_uauc_vs_s2"]=metrics[name]["uauc"]-metrics[
            "R1_A_S2_QV"]["uauc"]
    if set(scores)!={x[0] for x in ARMS}:
        raise RuntimeError("Output arms mismatch")
    np.savez_compressed(
        npz,**base_ids,**scores,
        arm_names=np.asarray([a[0] for a in ARMS]),
        q_scales=np.asarray([a[1] for a in ARMS]),
        v_scales=np.asarray([a[2] for a in ARMS]))
    info={
        "status":"complete","split":"valid",
        "checkpoint":str(checkpoint),
        "checkpoint_sha256":file_sha256(checkpoint),
        "config":str(cfgpath),"config_sha256":file_sha256(cfgpath),
        "n_samples":args.expected_samples,
        "protocol":"one S2 checkpoint, eval only, native Yes logit, "
                   "same sample order, legacy MyLinear Q/V beta only",
        "arms":[{"name":a,"beta_q":q,"beta_v":v} for a,q,v in ARMS],
        "metrics":metrics,"prediction_archive":str(npz),
        "warning":"OFF is trained-model injection ablation, "
                  "not retrained no-GNN. Validation labels are for "
                  "posthoc diagnostics only.",
    }
    result.write_text(json.dumps(info,ensure_ascii=False,indent=2),
                      encoding="utf-8")
    print(f"[R1-COMPLETE] no training; report={result} predictions={npz}",
          flush=True)


if __name__=="__main__":
    main()
