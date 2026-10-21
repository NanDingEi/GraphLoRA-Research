#!/usr/bin/env python3
"""P1: CPU-only parameter-drift diagnostic for P0 S2 L30 auxiliary BCE.

Compare identical saved trainable parameters in S2, P0_CONTROL and
P0_L30_AUX002. Read exact checkpoint paths from the completed P0 manifest.
No model instance, CUDA, gradients, training, test set, or validation rerun.

Update vectors: dc = CONTROL - S2, da = AUX - S2, daux = AUX - CONTROL.
The parameter update direction cosine is NOT a gradient-conflict measure.
"""
import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

DEFAULT_RUN = (
    "/home/bbx/user_lx/GraphLoRA_collab_v1/logs/collab_v1/"
    "S2_L30_AUX_P0_20261021_165700"
)
ARM_WEIGHTS = {"P0_CONTROL": 0.0, "P0_L30_AUX002": 0.02}
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
PROJ_RE = re.compile(r"(?:^|\.)(q_proj|v_proj)(?:\.|$)")
LORA_RE = re.compile(r"(?:^|\.)(lora_A|lora_B)(?:\.|$)")


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        obj = torch.load(str(path), map_location="cpu", mmap=True,
                         weights_only=False)
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict) or not isinstance(obj.get("model"), dict):
        raise RuntimeError(f"Missing checkpoint model dictionary: {path}")
    for name, tensor in obj["model"].items():
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Non-tensor state value {name} in {path}")
    return obj


def tensor_category(key):
    if LORA_RE.search(key):
        return "llm_lora"
    if key.startswith("gcn.") or ".gcn." in key:
        return "gnn"
    if "llama_proj." in key:
        return "projector"
    if key.startswith("rec_encoder.") or ".rec_encoder." in key:
        return "small_recommender"
    return "other"


def tensor_groups(key):
    category = tensor_category(key)
    groups = [f"module/{category}"]
    if category == "llm_lora":
        lm = LAYER_RE.search(key)
        layer = f"L{int(lm.group(1)):02d}" if lm else "unassigned"
        proj = PROJ_RE.search(key)
        ab = LORA_RE.search(key)
        groups.extend([
            f"lora_layer/{layer}",
            f"lora_layer_proj/{layer}/{proj.group(1) if proj else 'other'}",
            f"lora_layer_ab/{layer}/{ab.group(1) if ab else 'other'}",
        ])
    return groups


def empty_accum():
    return {
        "n_tensors": 0, "n_elements": 0, "changed_control": 0,
        "changed_aux": 0, "changed_between": 0,
        "base_sq": 0.0, "control_update_sq": 0.0,
        "aux_update_sq": 0.0, "between_sq": 0.0,
        "update_dot": 0.0,
    }


def add_to_accum(accum, row):
    accum["n_tensors"] += 1
    accum["n_elements"] += row["n_elements"]
    accum["changed_control"] += int(row["control_update_sq"] > 0)
    accum["changed_aux"] += int(row["aux_update_sq"] > 0)
    accum["changed_between"] += int(row["between_sq"] > 0)
    for key in ("base_sq", "control_update_sq", "aux_update_sq",
                "between_sq", "update_dot"):
        accum[key] += row[key]


def finalize(accum):
    b = math.sqrt(accum["base_sq"])
    c = math.sqrt(accum["control_update_sq"])
    a = math.sqrt(accum["aux_update_sq"])
    gap = math.sqrt(accum["between_sq"])
    dot = accum["update_dot"]
    cos = dot / (a * c) if a > 0 and c > 0 else None
    if cos is not None:
        cos = max(-1.0, min(1.0, cos))
    return {
        "n_tensors": accum["n_tensors"],
        "n_elements": accum["n_elements"],
        "changed_control": accum["changed_control"],
        "changed_aux": accum["changed_aux"],
        "changed_between": accum["changed_between"],
        "S2_parameter_norm": b,
        "control_update_norm": c,
        "aux_update_norm": a,
        "aux_minus_control_norm": gap,
        "control_relative_drift": c / max(b, 1e-12),
        "aux_relative_drift": a / max(b, 1e-12),
        "aux_vs_control_relative_gap": gap / max(b, 1e-12),
        "aux_to_control_update_norm_ratio": a/c if c else None,
        "update_direction_cosine": cos,
    }


def tensor_update_summary(base, control, aux, chunk=1_000_000):
    """Streaming one-tensor updates, bounded transient memory.

    FP32 differences to match the actual saved weights; accumulate dot/sq
    in float64 so late layers cannot dominate through reduced accumulation
    precision. Shape and finiteness checked fail-closed.
    """
    if (base.shape != control.shape or base.shape != aux.shape or
            base.numel() == 0):
        raise ValueError("Invalid/mismatched comparison tensor")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    flat = [x.reshape(-1) for x in (base, control, aux)]
    row = {"n_elements": int(base.numel()), "base_sq": 0.0,
           "control_update_sq": 0.0, "aux_update_sq": 0.0,
           "between_sq": 0.0, "update_dot": 0.0}
    for start in range(0, base.numel(), chunk):
        stop = min(start + chunk, base.numel())
        b, c, a = [x[start:stop].to(torch.float64) for x in flat]
        if not all(bool(torch.isfinite(x).all()) for x in (b, c, a)):
            raise RuntimeError("Encountered NaN/Inf in saved parameter")
        dc = c - b
        da = a - b
        diff = a - c
        row["base_sq"] += float(torch.dot(b, b))
        row["control_update_sq"] += float(torch.dot(dc, dc))
        row["aux_update_sq"] += float(torch.dot(da, da))
        row["between_sq"] += float(torch.dot(diff, diff))
        row["update_dot"] += float(torch.dot(dc, da))
    return row


def analyze_states(base, control, aux, require_full_lora=True):
    """Analyze ONLY matched same-shape saved tensors, explicitly audit gaps."""
    states = (base, control, aux)
    key_sets = [set(x) for x in states]
    common = key_sets[0] & key_sets[1] & key_sets[2]
    gaps = {}
    for label, keys in zip(("S2", "control", "aux"), key_sets):
        gaps[label+"_only_or_not_in_all"] = sorted(keys - common)
    missing_lora = {
        label: sorted(k for k in keys-common if LORA_RE.search(k))
        for label, keys in zip(("S2", "control", "aux"), key_sets)
    }
    if require_full_lora and any(missing_lora.values()):
        raise RuntimeError("LoRA state-key mismatch; cannot infer layer drift: "
                           + repr({k:v[:10] for k,v in missing_lora.items()}))
    groups = defaultdict(empty_accum)
    tensors = []
    skipped_shapes = []
    for key in sorted(common):
        b, c, a = (state[key] for state in states)
        if b.shape != c.shape or b.shape != a.shape:
            skipped_shapes.append({
                "name": key, "shapes": [list(t.shape) for t in (b,c,a)]})
            continue
        if not b.is_floating_point():
            # e.g., int64 running counters must not be treated as weights.
            continue
        stats = tensor_update_summary(b, c, a)
        record = {"name":key, "category":tensor_category(key),
                  "shape":list(b.shape)}
        record.update(finalize({
            **empty_accum(),
            **stats,
            "n_tensors":1,
            "changed_control":int(stats["control_update_sq"] > 0),
            "changed_aux":int(stats["aux_update_sq"] > 0),
            "changed_between":int(stats["between_sq"] > 0),
        }))
        tensors.append(record)
        for group in tensor_groups(key):
            add_to_accum(groups[group], stats)
    if require_full_lora and any(LORA_RE.search(r["name"]) for r in skipped_shapes):
        raise RuntimeError("LoRA shape mismatch in saved checkpoints")
    if not groups.get("module/llm_lora") or (
        groups["module/llm_lora"]["n_tensors"] < 10
    ):
        raise RuntimeError("Too few matched LoRA tensors for credible drift analysis")
    aggregates = {g:finalize(row) for g,row in sorted(groups.items())}
    by_difference = sorted(tensors,
        key=lambda r:r["aux_minus_control_norm"],reverse=True)
    return {
        "matched_floating_tensors":len(tensors),
        "missing_state_keys":gaps,
        "shape_mismatch":skipped_shapes,
        "per_group":aggregates,
        "top_20_between":by_difference[:20],
        "per_tensor":tensors,
    }


def load_p0_manifest(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    path=run_dir/"state.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    state=json.loads(path.read_text(encoding="utf-8"))
    arm_info={}
    for name, expected in ARM_WEIGHTS.items():
        entry=state.get("arms",{}).get(name)
        if not entry or entry.get("status")!="complete":
            raise RuntimeError(f"{name} must have status=complete")
        if abs(float(entry.get("aux_weight",-1))-expected)>1e-12:
            raise RuntimeError(f"Unexpected aux weight for {name}")
        cfg=run_dir/f"{name}_FULL.yaml"
        if not cfg.is_file() or file_sha256(cfg)!=entry.get("config_sha"):
            raise RuntimeError(f"{name} config changed since training")
        model_path=Path(entry["best_checkpoint"]).resolve()
        if not model_path.is_file() or model_path.name!="checkpoint_best.pth":
            raise RuntimeError(f"{name} best checkpoint is unavailable")
        arm_info[name]={"checkpoint":model_path,"config_path":cfg,
                        "auc":float(entry["auc"]),
                        "uauc":float(entry["uauc"])}
    champion=Path(state["champion"]).expanduser().resolve()
    if not champion.is_file() or (
            file_sha256(champion)!=state["champion_sha256"]):
        raise RuntimeError("Original S2 checkpoint missing or altered")
    return run_dir,state,champion,arm_info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",default=DEFAULT_RUN)
    parser.add_argument("--output",default=None)
    parser.add_argument("--no-checkpoint-config-check",action="store_true",
                        help="For diagnostics only; usually keep guard enabled")
    args=parser.parse_args()
    run_dir,state,champion,arms=load_p0_manifest(args.run_dir)
    dest=(Path(args.output).expanduser().resolve() if args.output else
          run_dir/"P1_PARAMETER_DRIFT.json")
    if dest == champion or dest in (x["checkpoint"] for x in arms.values()):
        raise RuntimeError("Refusing to overwrite a checkpoint")
    b=load_checkpoint(champion)
    c=load_checkpoint(arms["P0_CONTROL"]["checkpoint"])
    a=load_checkpoint(arms["P0_L30_AUX002"]["checkpoint"])
    if not args.no_checkpoint_config_check:
        for label,obj,weight in (("P0_CONTROL",c,0.0),
                                 ("P0_L30_AUX002",a,0.02)):
            config=obj.get("config",{})
            model=config.get("model",{})
            run=config.get("run",{})
            if (abs(float(model.get("l30_aux_weight",-1))-weight)>1e-12
                or int(model.get("gcn_layer",-1))!=31
                or abs(float(model.get("gnn_scale",-1))-0.1)>1e-10
                or model.get("collab_mode")!="legacy"
                or Path(model.get("ckpt","")).resolve()!=champion
                or int(run.get("seed",-1))!=42):
                raise RuntimeError(f"{label} checkpoint config does not match P0")
    result=analyze_states(b["model"],c["model"],a["model"])
    report={
        "status":"complete","analysis":"P1_parameter_drift_no_training",
        "source_run":str(run_dir),
        "original_S2":str(champion),
        "arms":{
            name:{"checkpoint":str(info["checkpoint"]),
                  "auc":info["auc"],"uauc":info["uauc"]}
            for name,info in arms.items()},
        "metrics":{
            "aux_minus_control_auc":(
                arms["P0_L30_AUX002"]["auc"]-arms["P0_CONTROL"]["auc"]),
            "aux_minus_control_uauc":(
                arms["P0_L30_AUX002"]["uauc"]-arms["P0_CONTROL"]["uauc"]),
        },
        "diagnostic":result,
        "caveats":[
            "Parameter drift does not prove gradient conflict or causal mechanism",
            "Only matched, saved trainable tensors are comparable",
            "L30 auxiliary BCE affects upstream layers, not the frozen Yes head",
            "Both arms continued S2 for one epoch; not a fresh-retraining study",
            "A single seed and a single Valid assessment cannot imply significance",
        ],
    }
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),
                    encoding="utf-8")
    print(f"[P1-ANCHOR] S2 and both P0 arms verified, "
          f"matched_tensors={result['matched_floating_tensors']}",flush=True)
    for name in ("module/llm_lora","module/gnn","module/projector",
                 "module/small_recommender","module/other"):
        g=result["per_group"].get(name)
        if g is None:
            print(f"[P1-MODULE] {name} no common saved tensors",flush=True)
            continue
        print(f"[P1-MODULE] {name} control_drift={g['control_relative_drift']:.8g} "
              f"aux_drift={g['aux_relative_drift']:.8g} "
              f"gap={g['aux_vs_control_relative_gap']:.8g} "
              f"cos={g['update_direction_cosine']} "
              f"changed={g['changed_control']}/{g['changed_aux']}/"
              f"{g['changed_between']}/{g['n_tensors']}",flush=True)
    keys=sorted(k for k in result["per_group"] if k.startswith("lora_layer/"))
    for key in keys:
        g=result["per_group"][key]
        print(f"[P1-LAYER] {key:20s} "
              f"ctrl_norm={g['control_update_norm']:.7g} "
              f"aux_norm={g['aux_update_norm']:.7g} "
              f"gap={g['aux_minus_control_norm']:.7g} "
              f"cos={g['update_direction_cosine']}",flush=True)
    print(f"[P1-METRIC] aux_minus_control AUC="
          f"{report['metrics']['aux_minus_control_auc']:+.9f} UAUC="
          f"{report['metrics']['aux_minus_control_uauc']:+.9f}",flush=True)
    print(f"[P1-COMPLETE] CPU only; {dest}",flush=True)


if __name__=="__main__":
    main()
