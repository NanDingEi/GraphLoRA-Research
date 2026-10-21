#!/usr/bin/env python3
"""P0 pretrain parity: L30 auxiliary supervision MUST NOT affect evaluation.

Validate the same unmodified S2 champion checkpoint with the auxiliary
training hook configured ON, but eval() inactive. Valid Small only.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
import train_collm_mf_din as entry
from minigpt4.common.config import Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--expected-auc",type=float,default=0.7296752596287361)
    p.add_argument("--expected-uauc",type=float,default=0.6841670211619721)
    p.add_argument("--tolerance",type=float,default=2e-5)
    args=p.parse_args()
    cp=Path(args.config).resolve()
    ck=Path(args.checkpoint).resolve()
    out=Path(args.result).resolve()
    if not cp.is_file() or not ck.is_file():
        raise FileNotFoundError("Missing P0 config or S2 checkpoint")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg=Config(Namespace(cfg_path=str(cp),options=None))
    if (cfg.model_cfg.collab_mode!="legacy" or
        int(cfg.model_cfg.gcn_layer)!=31 or
        float(cfg.model_cfg.l30_aux_weight)<=0):
        raise RuntimeError("P0 anchor expects L30 auxiliary branch enabled")
    cfg.model_cfg.ckpt=str(ck)
    cfg.run_cfg.distributed=False
    cfg.run_cfg.resume_ckpt_path=None
    cfg.run_cfg.evaluate=False
    cfg.run_cfg.output_dir=str(out.parent/"P0_ANCHOR_ARTIFACTS")
    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task=entry.tasks.setup_task(cfg)
    datasets=task.build_datasets(cfg)
    try:
        root=cfg.datasets_cfg.movie_ood.path
    except Exception:
        root=cfg.datasets_cfg.amazon_ood.path
    tables=[pd.read_pickle(root+x+"_ood2.pkl") for x in ("train","valid","test")]
    cfg.model_cfg.rec_config.user_num=max(int(df.uid.max()) for df in tables)+1
    cfg.model_cfg.rec_config.item_num=max(int(df.iid.max()) for df in tables)+1
    del tables
    model=task.build_model(cfg)
    runner=entry.get_runner_class(cfg)(
        cfg=cfg,job_id=entry.now(),task=task,model=model,datasets=datasets)
    runner.set_model_mode(cfg.run_cfg.mode)
    active=runner.unwrap_dist_model(runner.model)
    if active.run_mode_!="v2":
        raise RuntimeError("Unexpected S2 v2 prediction protocol")
    result=runner.eval_epoch(split_name="valid",cur_epoch=0,skip_reload=True)
    auc=float(result["agg_metrics"])
    raw=result["uauc"]
    uauc=float(raw[0] if isinstance(raw,(tuple,list)) else raw)
    if active._l30_aux_forward_calls:
        raise RuntimeError("L30 auxiliary hook fired during evaluation")
    info={"auc":auc,"uauc":uauc,"expected_auc":args.expected_auc,
          "expected_uauc":args.expected_uauc,"aux_captures":0}
    out.write_text(json.dumps(info,indent=2),encoding="utf-8")
    if abs(auc-args.expected_auc)>args.tolerance or (
        abs(uauc-args.expected_uauc)>args.tolerance
    ):
        raise RuntimeError(f"P0 S2 eval parity failed: {info}")
    print(f"[L30-ANCHOR] PASS AUC={auc:.9f} UAUC={uauc:.9f} "
          "aux_inactive_in_eval=1",flush=True)

if __name__=="__main__":
    main()
