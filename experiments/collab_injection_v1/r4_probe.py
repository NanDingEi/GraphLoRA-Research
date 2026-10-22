#!/usr/bin/env python3
"""32-example frozen-S2 native Yes-logit anchor for R4 parameter policy."""
import argparse
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
from r4_freeze import MODES,install_freeze,assert_frozen_unchanged
from r3_probe import predict,file_hash
from r2_probe import build_model
from r2_train import S2


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--round",required=True,choices=tuple(MODES))
    p.add_argument("--config",default="tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml")
    p.add_argument("--r1",default="logs/collab_v1/S2_R1_QV_ABLATION.npz")
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R4 requires GPU0")
    mode,_=MODES[args.round]
    outfile=ROOT/"logs/collab_v1"/f"R4_{args.round}_FROZEN_ANCHOR.json"
    if outfile.exists():
        raise FileExistsError(outfile)
    rt=install_runtime(mode)
    install_freeze(args.round)
    import argparse as ar
    cfg,runner,model,ckpt=build_model(ar.Namespace(
        config=args.config,checkpoint=str(S2),output=str(outfile)))
    if Path(ckpt).resolve()!=S2.resolve():
        raise RuntimeError("R4 probe did not load strict original S2")
    no_adapter=predict(cfg,runner,model,rt,"off")
    zero_adapt=predict(cfg,runner,model,rt,mode)
    for key in ("user","item","label"):
        if not np.array_equal(no_adapter[key],zero_adapt[key]):
            raise RuntimeError(f"R4 sample mismatch: {key}")
    r1=Path(args.r1)
    with np.load(r1,allow_pickle=False) as archive:
        for key in ("user","item","label"):
            if not np.array_equal(no_adapter[key],archive[key][:32]):
                raise RuntimeError(f"R4 vs R1 first 32 IDs mismatch: {key}")
        deviation=float(np.max(np.abs(
            no_adapter["score"]-archive["R1_A_S2_QV"][:32])))
    parity=float(np.max(np.abs(
        zero_adapt["score"]-no_adapter["score"])))
    if deviation>2e-4 or parity>2e-4:
        raise RuntimeError(
            f"R4 does not anchor S2: vs_R1={deviation} extra_adapter={parity}")
    assert_frozen_unchanged(model,model.r4_frozen_baseline_hash)
    required={
        "cflora":("q_cf_forward","v_cf_forward"),
        "cora":("q_cora_forward","v_cora_forward"),
    }[mode]
    if any(rt.counters.get(k,0)<=0 for k in required):
        raise RuntimeError("R4 main Q/V adapter did not execute")
    report={
        "status":"complete","round":args.round,"method":mode,
        "original_s2_checkpoint":str(ckpt),"original_s2_sha256":file_hash(ckpt),
        "samples":32,"max_abs_S2_R1":deviation,
        "max_abs_zero_init":parity,
        "trainable_tensors":list(model.r4_trainable_names),
        "trainable_params":model.r4_trainable_count,
        "frozen_hash_count":len(model.r4_frozen_baseline_hash),
        "runtime_counters":dict(rt.counters),
    }
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(report,indent=2,ensure_ascii=False),
                       encoding="utf-8")
    print(f"[R4-{args.round}-ANCHOR-PASS] "
          f"initial_diff={parity:.9g} {outfile}",flush=True)


if __name__=="__main__":
    main()
