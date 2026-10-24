#!/usr/bin/env python3
"""Follow-up 8 controlled NO31-CoRA runs after R2(V/O) and R7(alpha64).

Single existing GraphLoRA_NO31_CORA worktree, GPU0, foreground and serial.
No checkpoint resume, no Test, no edits to historical run recipes.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.no31_cora_original import night8 as core

# Completed Valid references: exact logs from 2026-10-24 (seed42).
R2_AUC=0.7257211273845012
R2_UAUC=0.682916846694341
R7_AUC=0.7288642146203805
R7_UAUC=0.6855345718193175
R8_AUC=0.6162779607748061
R8_UAUC=0.560654410909415

# Fixed R0 checkpoint: required preflight to verify L0-L7 got gradients
# after activating the semantic-teacher autocast-cache fix.
FIXED_R0_CHECKPOINT=(
    ROOT/"output/NO31_CORA_ORIGINAL_20261023_234005_110337/"
    "20261023234/checkpoint_0.pth"
)

ALL_PROJECTIONS="q_proj,k_proj,v_proj,o_proj"

FOLLOWUP8=[
    dict(id="F1",title="V/O alpha64: cross R2 x R7",
         projections="v_proj,o_proj",layers="0-31",
         cf_input="concat",alpha=64,p_lr=1.0,seed=42),
    dict(id="F2",title="V/O alpha96: amplitude ceiling",
         projections="v_proj,o_proj",layers="0-31",
         cf_input="concat",alpha=96,p_lr=1.0,seed=42),
    dict(id="F3",title="QKVO alpha96: high-alpha extension",
         projections=ALL_PROJECTIONS,layers="0-31",
         cf_input="concat",alpha=96,p_lr=1.0,seed=42),
    dict(id="F4",title="QKVO alpha48: bracket alpha32/64",
         projections=ALL_PROJECTIONS,layers="0-31",
         cf_input="concat",alpha=48,p_lr=1.0,seed=42),
    dict(id="F5",title="Only V alpha32: marginal V effect",
         projections="v_proj",layers="0-31",
         cf_input="concat",alpha=32,p_lr=1.0,seed=42),
    dict(id="F6",title="Only O alpha32: marginal O effect",
         projections="o_proj",layers="0-31",
         cf_input="concat",alpha=32,p_lr=1.0,seed=42),
    dict(id="F7",title="V/O alpha32 second seed (R2 vs F7)",
         projections="v_proj,o_proj",layers="0-31",
         cf_input="concat",alpha=32,p_lr=1.0,seed=43),
    dict(id="F8",title="QKVO alpha64 second seed (R7 vs F8)",
         projections=ALL_PROJECTIONS,layers="0-31",
         cf_input="concat",alpha=64,p_lr=1.0,seed=43),
]
LOOKUP={row["id"]:row for row in FOLLOWUP8}

def check_config():
    cfg=core.prepared_config()
    if cfg["model"]["fix_autocast_cache"] is not True:
        raise RuntimeError("Semantic teacher autocast fix must be enabled")
    assert len(FOLLOWUP8)==len(LOOKUP)==8, "Eight unique experiments required"
    for row in FOLLOWUP8:
        if row["cf_input"] != "concat" or row["layers"] != "0-31":
            raise RuntimeError("Unapproved non-R2/R7 injection pathway")
        if row["p_lr"] != 1.0:
            raise RuntimeError("P learning rate must stay at control scale")
        if row["seed"] not in (42,43):
            raise RuntimeError("Unverified follow-up seed")
    return cfg

def plan(rows):
    cfg=check_config()
    print("[FOLLOWUP8-CONTROL] single GPU0, independent 1-epoch "
          "Valid-only, MF pretrained frozen CoRA condition, rank16 "
          "CTR=KD=.05 gnn_scale=0 alpha_input=0 "
          "fix_autocast_cache=True batch2 eval3 accum4 "
          "init_lr=.0001 p_lr1x scheduler=3epoch; no checkpoint resume",
          flush=True)
    print(f"[FOLLOWUP8-BASELINE] NO31 AUC={core.BASELINE_AUC:.9f} "
          f"UAUC={core.BASELINE_UAUC:.9f}; "
          f"R2(V/O,a32) AUC={R2_AUC:.9f} UAUC={R2_UAUC:.9f}; "
          f"R7(QKVO,a64) AUC={R7_AUC:.9f} UAUC={R7_UAUC:.9f}",
          flush=True)
    for r in rows:
        print("[FOLLOWUP8-PLAN] %s seed=%d %-42s "
              "layers=%s projections=%s alpha=%d P=%d"
              % (r["id"],r["seed"],r["title"],r["layers"],
                 r["projections"],r["alpha"],core.selected_targets(r)),
              flush=True)
    return cfg

def main():
    parser=argparse.ArgumentParser(
        description="8 controlled follow-ups to R2 (V/O) and R7 (QKVO alpha64)"
    )
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan",action="store_true")
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--check-fixed-r0",action="store_true")
    parser.add_argument("--only",choices=list(LOOKUP))
    parser.add_argument("--start",choices=list(LOOKUP))
    parser.add_argument("--first",type=int,choices=range(1,9),metavar="N",
                        help="Run or preview only the first N experiments (e.g. --first 4 means F1-F4)")
    parser.add_argument(
        "--skip-check-fixed-r0",action="store_true",
        help="Only if the historical fixed R0 checkpoint has been removed "
             "AND early-layer gradient health was independently verified."
    )
    args=parser.parse_args()
    if args.check_fixed_r0:
        core.precheck(FIXED_R0_CHECKPOINT)
        return
    if sum(x is not None for x in (args.only,args.start,args.first)) > 1:
        parser.error("--only, --start and --first cannot be combined")
    if (args.only or args.start) and not args.run:
        parser.error("--only and --start require --run")
    chosen=FOLLOWUP8
    if args.only:
        chosen=[LOOKUP[args.only]]
    elif args.start:
        ids=list(LOOKUP)
        chosen=FOLLOWUP8[ids.index(args.start):]
    elif args.first is not None:
        chosen=FOLLOWUP8[:args.first]
    cfg=plan(chosen)
    if args.plan:
        return
    # CPU validation is run before GPU training, not only in the --plan view.
    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_followup8_cpu.py"],
        cwd=str(ROOT),check=True,
    )
    if not args.skip_check_fixed_r0:
        core.precheck(FIXED_R0_CHECKPOINT)
    print("[FOLLOWUP8-PREFLIGHT-PASS]",flush=True)
    core.run_trials(chosen,cfg,name_prefix="NO31_CORA_FOLLOWUP8")
    print("[FOLLOWUP8-DONE] Summary saved under logs/NO31_CORA_FOLLOWUP8_*",
          flush=True)


if __name__=="__main__":
    main()
