#!/usr/bin/env python3
"""Synthetic CPU-only P2A regression, no datasets, model, CUDA or training."""
import json
import tempfile
from pathlib import Path

import torch
import yaml

from run_p2a_autocast_cache_probe import (
    prepare, compare, sha, paths,
)

with tempfile.TemporaryDirectory(prefix="p2a_") as tmp:
    root=Path(tmp)
    run=root/"logs"
    run.mkdir()
    out=root/"outputs"
    champion=root/"S2.pth"
    base={}
    unfixed={}
    fixed={}
    for i in range(32):
        for suffix in ("q_proj.lora_A.default.weight",
                       "q_proj.lora_B.default.weight",
                       "v_proj.lora_A.default.weight",
                       "v_proj.lora_B.default.weight"):
            k=f"llama_model_lora.base_model.model.model.layers.{i}.self_attn.{suffix}"
            base[k]=torch.tensor([1.,-1.])
            unfixed[k]=base[k].clone() + (torch.tensor([0.2,0.]) if i>=8 else 0)
            fixed[k]=base[k].clone() + torch.tensor([0.2,0.1])
    torch.save({"model":base},champion)
    old_dir=out/"P0_CONTROL_SMOKE"
    old_dir.mkdir(parents=True)
    torch.save({"model":unfixed},old_dir/"checkpoint_0.pth")
    original={
        "model":{"ckpt":str(champion),"collab_mode":"legacy",
                 "gnn_scale":0.1,"gcn_layer":31,
                 "gnn_semantic_align":True,
                 "gnn_semantic_weight":0.0001,
                 "freeze_lora":False,
                 "l30_aux_weight":0.0},
        "run":{"seed":42,"max_train_steps":24,
               "skip_validation":True,"output_dir":str(old_dir)},
    }
    (run/"P0_CONTROL_SMOKE.yaml").write_text(
        yaml.safe_dump(original),encoding="utf-8")
    (run/"state.json").write_text(json.dumps({
        "champion":str(champion),"champion_sha256":sha(champion),
        "smoke_steps":24,"seed":42,"output_dir":str(out),
    }),encoding="utf-8")
    prepare(run)
    _,_,prior,_,newcfg,newdir=paths(run)
    updated=yaml.safe_load(newcfg.read_text())
    assert updated["model"]["fix_autocast_cache"] is True
    reverted=json.loads(json.dumps(updated))
    reverted["model"].pop("fix_autocast_cache")
    reverted["run"]["output_dir"]=str(old_dir)
    assert reverted==prior
    print("[P2A-CPU] configuration differs only in fix flag and output dir PASS",flush=True)

    newdir.mkdir()
    torch.save({"model":fixed},newdir/"checkpoint_0.pth")
    compare(run)
    report=json.loads((run/"P2A_AUTOCAST_CACHE_PROBE.json").read_text())
    assert report["original_first_eight_all_zero"]
    assert report["fixed_first_eight_all_updated"]
    assert report["layers"]["L00"]["old_norm"]==0
    assert report["layers"]["L30"]["old_norm"]>0
    print("[P2A-CPU] old 0–7 zero and fixed 0–7 changed PASS",flush=True)
    try:
        prepare(run)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Refuse overwrite of probe config/output")
    print("[P2A-CPU] refuses overwrite of an existing probe PASS",flush=True)

print("[P2A-CPU] ALL PASS",flush=True)
