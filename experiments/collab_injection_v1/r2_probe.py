#!/usr/bin/env python3
"""R2-02..05: foreground GPU0 no-training L31 collaborative pathway probes.

Uses same S2 Valid Small order and original checkpoint. Each --round
evaluates a specific fixed set of variants on the first 512 examples.
Native baseline is anchored sample-by-sample to R1's original native Yes
logit. Subset AUC is NOT a model selection metric.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import train_collm_mf_din as entry
from minigpt4.common.config import Config
from minigpt4.datasets.data_utils import prepare_sample
from r2_collab_runtime import install_runtime

VARIANTS={
    "02":("audit",),
    "03":("off","decision_q"),
    "04":("off","history_k"),
    "05":("off","relation_v","relation_shuffle"),
}


def build_model(args):
    cfg=Config(Namespace(cfg_path=str(Path(args.config).resolve()),options=None))
    if (str(cfg.model_cfg.collab_mode)!="legacy" or
        int(cfg.model_cfg.gcn_layer)!=31 or
        abs(float(cfg.model_cfg.gnn_scale)-.1)>1e-10):
        raise RuntimeError("R2 requires original legacy L31 S2")
    ckpt=Path(args.checkpoint or cfg.model_cfg.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    cfg.model_cfg.ckpt=str(ckpt)
    cfg.run_cfg.distributed=False
    cfg.run_cfg.resume_ckpt_path=None
    cfg.run_cfg.evaluate=False
    cfg.run_cfg.output_dir=str(Path(args.output).resolve().parent/"R2_PROBE_ARTIFACTS")
    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task=entry.tasks.setup_task(cfg)
    datasets=task.build_datasets(cfg)
    try:
        data_dir=cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_dir=cfg.datasets_cfg.amazon_ood.path
    frames=[pd.read_pickle(data_dir+split+"_ood2.pkl") for split in (
        "train","valid","test")]
    # Test tables ONLY determine vocabulary max for the original model;
    # NO Test inference or metrics.
    cfg.model_cfg.rec_config.user_num=int(max(x.uid.max() for x in frames)+1)
    cfg.model_cfg.rec_config.item_num=int(max(x.iid.max() for x in frames)+1)
    del frames
    model=task.build_model(cfg)
    runner=entry.get_runner_class(cfg)(
        cfg=cfg,job_id=entry.now(),task=task,model=model,datasets=datasets)
    runner.set_model_mode(cfg.run_cfg.mode)
    return cfg,runner,runner.unwrap_dist_model(runner.model),ckpt


def get_valid_probe_loader(runner):
    """Use the same concrete Valid DataLoader as RecBaseTask.evaluation.

    RunnerBase wraps each split in MultiIterLoader for multi-dataset
    sampling. That wrapper implements __next__ but NOT __iter__.
    Its .loaders entries are PrefetchLoader instances and ARE iterable.
    Accessing exactly one loader is intentional for ML-1M OOD2 so every
    variant restarts at Valid sample 0 without exhausting a shared iterator.
    """
    wrapped=runner.dataloaders.get("valid")
    loaders=getattr(wrapped,"loaders",None)
    if not isinstance(loaders,(list,tuple)) or len(loaders)!=1:
        raise RuntimeError(
            "R2 requires one concrete Valid iterator inside MultiIterLoader; "
            f"wrapper={type(wrapped).__name__} "
            f"num_loaders={len(loaders) if loaders is not None else 'N/A'}")
    loader=loaders[0]
    if not callable(getattr(loader,"__iter__",None)):
        raise RuntimeError(f"R2 concrete Valid loader is not iterable: {type(loader)}")
    if len(loader)*3 < 512:
        raise RuntimeError("R2 Valid loader unexpectedly too small")
    return loader


@torch.no_grad()
def infer_512(cfg,runner,model,runtime,mode,n_samples):
    runtime.stats.clear()
    runtime.counters.clear()
    runtime.activate(mode)
    model.eval()
    entry.setup_seeds(cfg)
    arrays={k:[] for k in ("user","item","label","score")}
    seen=0
    # Concrete PrefetchLoader, as used in RecBaseTask.evaluation.
    # A new iter(loader) starts from batch zero for EVERY mode.
    for samples in get_valid_probe_loader(runner):
        if seen>=n_samples:
            break
        samples=prepare_sample(samples,cuda_enabled=True)
        scores=model.generate_for_samples(samples)["logits"].detach().float().cpu().reshape(-1)
        take=min(len(scores),n_samples-seen)
        arrays["score"].extend(scores[:take].tolist())
        for src,dst in (("UserID","user"),("TargetItemID","item"),("label","label")):
            arrays[dst].extend(samples[src].detach().cpu().reshape(-1)[:take].tolist())
        seen+=take
        if seen%128==0 or seen==n_samples:
            print(f"[R2-PROBE-PROGRESS] mode={mode} examples={seen}",flush=True)
    if seen!=n_samples:
        raise RuntimeError(f"Expected {n_samples} samples, got {seen}")
    for k in arrays:
        arrays[k]=np.asarray(arrays[k],dtype=np.float64 if k=="score" else np.int64)
    if not np.isfinite(arrays["score"]).all():
        raise RuntimeError("Nonfinite native Yes logits")
    info=runtime.state()
    info["n_samples"]=seen
    if len(np.unique(arrays["label"]))==2:
        info["subset_auc_descriptive"]=float(roc_auc_score(
            arrays["label"],arrays["score"]))
    print(f"[R2-PROBE-RESULT] mode={mode} "
          f"subset_AUC={info.get('subset_auc_descriptive','NA')} "
          f"runtime_counters={info['counters']}",flush=True)
    return arrays,info


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--round",choices=VARIANTS,required=True)
    p.add_argument("--config",default="tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml")
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--n-samples",type=int,default=512)
    p.add_argument("--r1-predictions",
        default="logs/collab_v1/S2_R1_QV_ABLATION.npz")
    p.add_argument("--output",default=None)
    args=p.parse_args()
    if args.n_samples!=512:
        raise ValueError("R2 probes use fixed 512 samples; never tune n to Valid scores")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("R2 only runs on the server GPU0")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU0 CUDA unavailable")
    args.output=args.output or f"logs/collab_v1/R2_{args.round}_PROBE.json"
    output=Path(args.output).resolve()
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Refuse to overwrite R2 probe {output}")
    output.parent.mkdir(parents=True,exist_ok=True)

    runtime=install_runtime("off")
    cfg,runner,model,checkpoint=build_model(args)
    variant_arrays={}
    variant_info={}
    canonical=None
    for mode in VARIANTS[args.round]:
        a,info=infer_512(cfg,runner,model,runtime,mode,args.n_samples)
        if canonical is None:
            canonical={k:a[k] for k in ("user","item","label")}
        else:
            for key in canonical:
                if not np.array_equal(canonical[key],a[key]):
                    raise RuntimeError(f"R2 {mode} samples not aligned {key}")
        variant_arrays[mode]=a["score"]
        variant_info[mode]=info

    if args.round!="02":
        r1=Path(args.r1_predictions)
        if not r1.is_file():
            raise FileNotFoundError(r1)
        with np.load(r1,allow_pickle=False) as original:
            for key in canonical:
                if not np.array_equal(canonical[key],original[key][:args.n_samples]):
                    raise RuntimeError(f"R2 Valid sample keys mismatch R1 for {key}")
            anchor=np.asarray(original["R1_A_S2_QV"][:args.n_samples],dtype=np.float64)
        jitter=float(np.max(np.abs(variant_arrays["off"]-anchor)))
        print(f"[R2-ANCHOR] max_abs_native_vs_R1={jitter:.9g}",flush=True)
        if jitter>2e-4:
            raise RuntimeError("R2 no-modification S2 anchor does not match R1")
        differences={}
        for mode in VARIANTS[args.round][1:]:
            d=variant_arrays[mode]-variant_arrays["off"]
            differences[mode]={
                "max_abs":float(np.abs(d).max()),
                "mean_abs":float(np.abs(d).mean()),
                "fraction_nonzero":float(np.count_nonzero(d)/len(d))}
            print(f"[R2-DELTA] round={args.round} variant={mode} "
                  f"{differences[mode]}",flush=True)
        if args.round=="05":
            ww=variant_info["relation_v"]["stats"].get("history_weights",{})
            wrong=variant_info["relation_shuffle"]["stats"].get("history_weights",{})
            if ww.get("n",0)<1 or wrong.get("n",0)<1:
                raise RuntimeError("R2-05 no valid history weights")
    else:
        if variant_info["audit"]["counters"].get("audited_tokens",0)<1:
            raise RuntimeError("R2-02 audit failed to observe L31 V graph injection")
        differences={}

    record={
        "status":"complete","round":args.round,
        "n_samples":args.n_samples,"checkpoint":str(checkpoint),
        "variants":list(VARIANTS[args.round]),
        "results":variant_info,
        "score_differences_vs_native":differences,
        "warning":"512 samples: descriptive pathway probe, NOT a tuned Valid AUC",
    }
    archive=output.with_suffix(".npz")
    np.savez_compressed(archive,**canonical,**variant_arrays)
    record["prediction_archive"]=str(archive)
    output.write_text(json.dumps(record,indent=2,ensure_ascii=False),
                      encoding="utf-8")
    print(f"[R2-{args.round}-COMPLETE] {output} {archive}",flush=True)


if __name__=="__main__":
    main()
