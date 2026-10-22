#!/usr/bin/env python3
"""R5 paired training entry: hash MF/GNN/projector/LoRA at actual model creation."""
import argparse, hashlib, json, os, sys
from pathlib import Path

ROOT=Path("/home/bbx/user_lx/GraphLoRA_collab_v1")
EXP=ROOT/"experiments"/"collab_injection_v1"

def fingerprints(model):
    groups={"MF":[],"GNN":[],"Projector":[],"LoRA":[]}
    for name,p in model.named_parameters():
        if name.startswith("rec_encoder."): group="MF"
        elif name.startswith("gcn."): group="GNN"
        elif name.startswith("llama_proj."): group="Projector"
        elif ".lora_A." in name or ".lora_B." in name: group="LoRA"
        else: continue
        groups[group].append((name,p))
    result={}
    for group,params in groups.items():
        if not params: raise RuntimeError(f"No {group} parameters to fingerprint")
        h=hashlib.sha256()
        for name,p in sorted(params):
            t=p.detach().contiguous().cpu()
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(str(t.dtype).encode())
            h.update(t.numpy().tobytes())
        result[group]={"sha256":h.hexdigest(),
                       "trainable":sum(p.numel() for _,p in params if p.requires_grad),
                       "total":sum(p.numel() for _,p in params)}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cfg-path",required=True)
    ap.add_argument("--arm",choices=("control","cora"),required=True)
    ap.add_argument("--fingerprint",required=True)
    a=ap.parse_args()
    os.chdir(ROOT)
    sys.path[:0]=[str(ROOT),str(EXP)]
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    from minigpt4.models import minigpt4rec_v2 as graph
    runtime=None
    if a.arm=="cora":
        from r3_paper_adapters import install_runtime
        runtime=install_runtime("cora")
    original=graph.MiniGPT4Rec_v2.from_config
    def inspected(cls,cfg):
        model=original(cfg)
        has_adapter=getattr(model,"r3_adapter",None) is not None
        if has_adapter != (a.arm=="cora"):
            raise RuntimeError("Wrong CoRA adapter attachment for "+a.arm)
        data=fingerprints(model)
        dest=Path(a.fingerprint).resolve()
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("x") as f:
            json.dump({"arm":a.arm,"initial_shared":data},f,indent=2)
        print("[R5-FAIR-INITIAL] "+json.dumps(data,sort_keys=True),flush=True)
        return model
    graph.MiniGPT4Rec_v2.from_config=classmethod(inspected)
    import train_collm_mf_din as entry
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(Path(a.cfg_path).resolve())]
    try: entry.main()
    finally:
        if runtime is not None:
            print("[R3-COUNTERS] "+json.dumps(dict(runtime.counters)),flush=True)

if __name__=="__main__": main()
