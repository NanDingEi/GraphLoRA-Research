#!/usr/bin/env python3
"""Reproduce historical CoRA ensembles and select three improvements on VALID.

No model training. Two separate phases:
  select: read A/B/C VALID NPZ, select/lock hyperparameters using VALID only;
  test:   read frozen selection.json and A/B/C TEST NPZ, evaluate once.

The 3 checkpoints/splits must correspond to the exact same A/B/C members:
  A=RR1_L32_BASE (L32 Yes logit)
  B=RR5_USER_GATE_R31 (its trained gate-fused Yes logit)
  C=G3_FIXED_A025 (L32 + 0.25 * (L31 - L32))

No test-label-dependent tuning or weight selection exists in this program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 42
KEY_GROUPS = {
    "labels": ("labels", "label", "y", "targets"),
    "users": ("users", "user_ids", "UserID", "user_id"),
    "items": ("items", "item_ids", "TargetItemID", "item_id"),
    "layers": ("yes_logits", "layer_yes_logits", "layer_logits"),
    "fused": ("fused_logits", "fused_scores", "fused_yes_logits", "fused", "score", "scores"),
}


def auc(y, x):
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def uauc(y, users, x):
    result = []
    for u in np.unique(users):
        ix = users == u
        if np.unique(y[ix]).size == 2:
            result.append(auc(y[ix], x[ix]))
    if not result:
        raise ValueError("No user has both classes; UAUC undefined")
    return float(np.mean(result)), len(result)


def metrics(y, u, x):
    a = auc(y, x)
    b, n = uauc(y, u, x)
    return {"auc": a, "uauc": b, "uauc_users": n}



class ValidMetricEvaluator:
    """Cache all valid positive-negative comparisons for fast UAUC searches.

    Each user gets the same weight, not each pair. Ties count as 0.5.
    """
    def __init__(self, labels, users):
        self.y = labels
        self.u = users
        positives, negatives, weights = [], [], []
        eligible_users = 0
        for uid in np.unique(users):
            indices = np.flatnonzero(users == uid)
            p = indices[labels[indices] == 1]
            n = indices[labels[indices] == 0]
            if not len(p) or not len(n):
                continue
            eligible_users += 1
            positives.append(np.repeat(p, len(n)))
            negatives.append(np.tile(n, len(p)))
            weights.append(np.full(len(p)*len(n),1.0/(len(p)*len(n))))
        if not eligible_users:
            raise ValueError("No eligible UAUC users")
        self.pos=np.concatenate(positives)
        self.neg=np.concatenate(negatives)
        self.weights=np.concatenate(weights)/eligible_users
        self.users=eligible_users

    def metrics(self, prediction):
        diffs = prediction[self.pos]-prediction[self.neg]
        value = float(self.weights @ ((diffs > 0).astype(float) + 0.5*(diffs == 0)))
        return {"auc":auc(self.y,prediction),"uauc":value,"uauc_users":self.users}

def locate(npz, aliases):
    found = [s for s in aliases if s in npz.files]
    return found[0] if found else None


def choose_layer(scores, n, which):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape == (4, n):
        return scores[which].copy()
    if scores.shape == (n, 4):
        return scores[:, which].copy()
    raise ValueError(f"Expected layer matrix [4,N] or [N,4], got {scores.shape}")


@dataclass
class Member:
    path: str
    name: str
    labels: np.ndarray
    users: np.ndarray
    items: np.ndarray
    score: np.ndarray
    z31: np.ndarray | None
    z32: np.ndarray | None
    has_fused: bool
    keys: list[str]


def load_member(path, name, explicit_score_key=None):
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        candidate = list(p.glob("*.npz"))
        if len(candidate) != 1:
            raise ValueError(f"Directory {p} must contain exactly one NPZ; found {candidate}")
        p = candidate[0]
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        k = {x: locate(z, KEY_GROUPS[x]) for x in ("labels", "users", "items")}
        if any(v is None for v in k.values()):
            raise KeyError(f"Missing labels/users/items in {p}: keys={z.files}")
        y, u, i = [np.asarray(z[k[x]]).reshape(-1).astype(np.int64) for x in ("labels", "users", "items")]
        if not (len(y) == len(u) == len(i)):
            raise ValueError(f"Inconsistent lengths: {p}")
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"Nonbinary labels in {p}")
        n = len(y)
        layer_key = locate(z, KEY_GROUPS["layers"])
        z31 = z32 = None
        if layer_key:
            mat = np.asarray(z[layer_key])
            # Require the exact four late layers in the stored order L29/L30/L31/L32.
            z31, z32 = choose_layer(mat, n, 2), choose_layer(mat, n, 3)
        else:
            for suffix, ref in (("z31", "z31"), ("z32", "z32")):
                if suffix in z.files:
                    if ref == "z31":
                        z31 = np.asarray(z[suffix], dtype=np.float64).reshape(-1)
                    else:
                        z32 = np.asarray(z[suffix], dtype=np.float64).reshape(-1)
        fused_key = explicit_score_key or locate(z, KEY_GROUPS["fused"])
        if explicit_score_key and explicit_score_key not in z.files:
            raise KeyError(f"Explicit score key {explicit_score_key!r} missing from {p}")
        fused = np.asarray(z[fused_key], dtype=np.float64).reshape(-1) if fused_key else None
        if fused is not None and len(fused) != n:
            raise ValueError(f"Invalid fused shape {fused.shape} in {p}")
        if name == "A":
            if z32 is None:
                if explicit_score_key is None or fused is None:
                    raise ValueError("A requires L32 yes_logits, or an EXPLICIT --a-score-key")
                score = fused
            else:
                score = z32
        elif name == "B":
            if fused is None:
                raise ValueError("B requires saved RR5 fused logits (NOT plain z32)")
            score = fused
        elif name == "C":
            if z31 is None or z32 is None:
                if explicit_score_key is None or fused is None:
                    raise ValueError("C requires L31/L32, or an EXPLICIT --c-score-key")
                score = fused
            else:
                score = z32 + 0.25 * (z31 - z32)
                if fused is not None:
                    difference = float(np.max(np.abs(fused - score)))
                    if difference > 0.005:
                        raise ValueError(f"C fixed alpha=0.25 formula mismatch: maxerr={difference:.6g}")
        else:
            raise ValueError(name)
        if score.shape != (n,) or not np.isfinite(score).all():
            raise ValueError(f"Invalid {name} score shape or nonfinite values")
        return Member(str(p), name, y, u, i, score, z31, z32, fused is not None, list(z.files))


def check_align(members):
    a = members["A"]
    for name, m in members.items():
        if len(a.labels) != len(m.labels):
            raise ValueError(f"A/{name} sample count mismatch")
        for field in ("labels", "users", "items"):
            if not np.array_equal(getattr(a, field), getattr(m, field)):
                bad = np.flatnonzero(getattr(a, field) != getattr(m, field))
                raise ValueError(f"A/{name} {field} misalignment first_idx={int(bad[0])}; abort")
    unique = np.stack((a.users, a.items, a.labels), axis=1)
    digest = hashlib.sha256(np.ascontiguousarray(unique.astype('<i8')).tobytes()).hexdigest()
    return digest


def load_triplet(args, split):
    members = {name: load_member(getattr(args, f"{name.lower()}_{split}"), name,
                                 getattr(args, f"{name.lower()}_score_key")) for name in "ABC"}
    digest = check_align(members)
    return members, digest


def stats(x, first128=True):
    v = x[:min(128, len(x))] if first128 else x
    mean, std = float(np.mean(v)), float(np.std(v))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("Zero or nonfinite standard deviation during normalization")
    return [mean, std]


def normalize_members(members, mode, stats_by_member=None):
    scores, used = {}, {}
    for name, m in members.items():
        if mode == "test_first128" and stats_by_member is not None:
            raise ValueError("test_first128 does not accept saved validation stats")
        values = stats(m.score, first128=True) if stats_by_member is None else stats_by_member[name]
        scores[name] = (m.score - values[0]) / values[1]
        used[name] = values
    return scores, used


def _get_checkpoint_state(path):
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("RR5 checkpoint must contain a state dict")
    state = obj.get("model", obj)
    if not isinstance(state, dict):
        raise ValueError("RR5 checkpoint['model'] is not a state dict")
    return state, obj.get("epoch", None)


def audit_rr5_gate(member, checkpoint, tolerance=0.005):
    """Independently rebuild B using RR5's OWN trained MF table and gate."""
    import torch
    import torch.nn.functional as F
    if member.name != "B" or member.z31 is None or member.z32 is None:
        raise ValueError("RR5 audit needs B, z31/z32, and saved fused scores")
    state, epoch = _get_checkpoint_state(checkpoint)
    suffixes = ["0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias"]
    gate = {}
    for suffix in suffixes:
        matches = [(k, v) for k, v in state.items() if k.endswith("cora_residual_fusion.user_gate." + suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"RR5 gate tensor {suffix}: expected one, got {[x[0] for x in matches]}")
        gate[suffix] = matches[0][1].float().cpu()
    candidates = [(k,v) for k,v in state.items() if k.endswith("rec_encoder.user_embedding.weight")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one RR5 MF user table; got {[k for k,_ in candidates]}")
    table = candidates[0][1].float().cpu()
    reconstructions = []
    u = member.users
    for offset in (0, -1, 1):
        ids = u + offset
        if int(ids.min()) < 0 or int(ids.max()) >= table.shape[0]:
            continue
        with torch.no_grad():
            x = F.normalize(table[torch.from_numpy(ids).long()], p=2, dim=1, eps=1e-8)
            x = F.layer_norm(x, (x.shape[1],), gate["0.weight"], gate["0.bias"], eps=1e-5)
            x = F.gelu(F.linear(x, gate["1.weight"], gate["1.bias"]))
            strength = torch.sigmoid(F.linear(x, gate["3.weight"], gate["3.bias"]))[:, 0].numpy()
        fused = member.z32 + strength * (member.z31-member.z32)
        maxerr = float(np.max(np.abs(fused-member.score)))
        reconstruction_auc = auc(member.labels, fused)
        reconstructions.append((maxerr, offset, reconstruction_auc))
    if not reconstructions:
        raise RuntimeError("RR5 raw UserID does not index its MF table with offsets 0/-1/+1")
    besterr, offset, ra = min(reconstructions)
    if besterr > tolerance:
        raise RuntimeError(f"RR5 checkpoint gate/MF reconstruction fails: best max error={besterr:.6g}, tolerance={tolerance}, offsets={reconstructions}")
    print(f"[RR5 GATE AUDIT PASS] epoch={epoch} user_offset={offset} max_abs_error={besterr:.8g} reconstructed_auc={ra:.9f}",flush=True)
    return {"checkpoint":str(checkpoint),"checkpoint_epoch":epoch,"user_offset":offset,"max_abs_error":besterr,"reconstructed_auc":ra}


def base_ab(S, w):
    return w * S["A"] + (1.0 - w) * S["B"]


def blend(S, weights):
    return sum(S[k] * float(weights.get(k, 0.0)) for k in "ABC")


def user_disagreement_weights(S, users, q50, q90, lambda_max):
    difference = np.abs(S["A"]-S["B"])
    result = np.zeros(len(users), dtype=np.float64)
    for u in np.unique(users):
        ix = users == u
        d = float(difference[ix].mean())
        result[ix] = lambda_max * np.clip((d-q50)/max(q90-q50,1e-12),0.0,1.0)
    return result


def quantile_disagreement(S, users):
    d = np.abs(S["A"]-S["B"])
    val = np.array([d[users==u].mean() for u in np.unique(users)],dtype=np.float64)
    return [float(x) for x in np.quantile(val,[.5,.9])]


def candidate_metric(y,u,x):
    return metrics(y,u,x)


def safe_score(row, name, priorities=("auc","uauc")):
    return tuple(row[k] for k in priorities)


def select(args):
    members, digest = load_triplet(args,"valid")
    y,u = members["A"].labels, members["A"].users
    evaluator=ValidMetricEvaluator(y,u)
    S, means = normalize_members(members,"test_first128")
    audit = None
    if not args.skip_rr5_audit:
        if not args.rr5_checkpoint:
            raise RuntimeError("VALID selection requires --rr5-checkpoint for RR5 gate audit")
        audit = audit_rr5_gate(members["B"], args.rr5_checkpoint)
    reference = {m: evaluator.metrics(members[m].score) for m in "ABC"}
    print("[VALID MEMBERS]",json.dumps(reference),flush=True)
    fixed = {}
    fixed["E2_AB_HALF"]={"kind":"constant","weights":{"A":0.5,"B":0.5,"C":0.0}}
    fixed["E3_ABC_THIRD"]={"kind":"constant","weights":{"A":1/3,"B":1/3,"C":1/3}}
    grid=np.linspace(0,1,21)
    weight_metrics=[(evaluator.metrics(base_ab(S,float(w))),float(w)) for w in grid]
    ab_metric,w=max(weight_metrics,key=lambda m:(m[0]["auc"],-abs(m[1]-0.5)))
    fixed["E2_AB_AUC"]={"kind":"constant","weights":{"A":w,"B":1-w,"C":0}}
    if (args.expected_historical_w is not None and
            abs(w - args.expected_historical_w) > 1e-9 and
            not args.allow_different_valid_weight):
        raise RuntimeError(
            f"Historical E2_AB_AUC weight mismatch: selected w={w:.2f} "
            f"instead of expected {args.expected_historical_w:.2f}. "
            "Check that A=RR1 ep37, B=RR5 ep21 and VALID captures correspond "
            "to exactly those checkpoints. Use --allow-different-valid-weight "
            "ONLY when investigating a legitimately different protocol.")
    ab=base_ab(S,w)
    half=base_ab(S,.5)
    half_metric=evaluator.metrics(half)
    baseline_metric=evaluator.metrics(S["A"])
    # I1: 3-way constrained simplex, maximize VALID AUC while UAUC no worse than
    # both RR1 and AB_HALF. Infeasible => fall back to AB_HALF.
    floor=max(baseline_metric["uauc"],half_metric["uauc"])
    i1=[]
    for ai in range(21):
        for bi in range(21-ai):
            ci=20-ai-bi
            wt={"A":ai/20,"B":bi/20,"C":ci/20}
            met=evaluator.metrics(blend(S,wt))
            if met["uauc"]>=floor-1e-12:
                i1.append((met,wt))
    if i1:
        i1met, i1w=max(i1,key=lambda t:(t[0]["auc"],t[0]["uauc"],-t[1]["C"]))
    else:
        i1w={"A":.5,"B":.5,"C":0.0}; i1met=half_metric
    # I2: maximize VALID UAUC, permitting at most 5e-4 VALID AUC loss vs AB_AUC.
    ab_auc=ab_metric["auc"]
    i2=[]
    for lam in np.arange(0.0,.35001,.025):
        candidate=(1-lam)*ab+lam*S["C"]
        met=evaluator.metrics(candidate)
        if met["auc"]+1e-12>=ab_auc-0.0005:
            i2.append((met,float(round(lam,3))))
    if not i2:
        raise RuntimeError("I2 feasible set cannot be empty because lambda=0")
    i2met,i2lam=max(i2,key=lambda t:(t[0]["uauc"],t[0]["auc"],-t[1]))
    # I3: per-user label-free disagreement; select lambda on VALID only, preserving
    # an AUC tolerance vs AB_AUC. User quantiles frozen from VALID users.
    q50,q90=quantile_disagreement(S,u)
    i3=[]
    for lmax in np.arange(0.0,.4001,.05):
        uw=user_disagreement_weights(S,u,q50,q90,float(lmax))
        candidate=(1-uw)*ab+uw*S["C"]
        met=evaluator.metrics(candidate)
        if met["auc"]+1e-12>=ab_auc-0.0005:
            i3.append((met,float(round(lmax,3))))
    if not i3:
        raise RuntimeError("I3 feasible set empty (unexpected; lmax=0)")
    i3met,i3l=max(i3,key=lambda t:(t[0]["uauc"],t[0]["auc"],-t[1]))

    fixed["I1_SIMPLEX_AUC_UAUC_FLOOR"]={"kind":"constant","weights":i1w}
    fixed["I2_UAUC_PARETO_C"]={"kind":"shrink_c","ab_weight_a":w,"lambda_c":i2lam}
    fixed["I3_USER_DISAGREEMENT_C"]={"kind":"user_disagree_c","ab_weight_a":w,"lambda_max":i3l,"valid_user_disagreement_quantiles":{"q50":q50,"q90":q90}}
    valid_results={}
    for tag, spec in fixed.items():
        predicted=predict(S,u,spec)
        valid_results[tag]=evaluator.metrics(predicted)
    print("[VALID-ONLY LOCKED]")
    for tag,v in valid_results.items():
        print(f"{tag:30s} auc={v['auc']:.8f} uauc={v['uauc']:.8f} spec={fixed[tag]}",flush=True)
    report={"seed":SEED,"selection_split":"valid","valid_sha256_sample_alignment":digest,
            "source_npz":{n:m.path for n,m in members.items()},
            "valid_rr5_audit":audit,
            "norm_mode":args.norm,"valid_first128_stats":means,
            "reference_valid_members":reference,"valid_method_metrics":valid_results,
            "selection":{"valid_ab_auc_weight_a":w,"i1_uauc_floor":floor,"i2_auc_max_drop":0.0005,"i3_auc_max_drop":0.0005},
            "methods":fixed,
            "message":"All choices computed from VALID only. Never reselect using TEST. Separate TEST inputs must come from the same A/B/C checkpoints."}
    path=Path(args.selection).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not args.overwrite_selection:
        raise FileExistsError(f"Refuse overwriting existing locked selection: {path} (use new tag)")
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("[LOCKED VALID SELECTION]",path,flush=True)


def predict(S,u,method):
    kind=method["kind"]
    if kind=="constant":
        return blend(S,method["weights"])
    ab=base_ab(S,method["ab_weight_a"])
    if kind=="shrink_c":
        lam=method["lambda_c"]
        return (1-lam)*ab+lam*S["C"]
    if kind=="user_disagree_c":
        qs=method["valid_user_disagreement_quantiles"]
        lam=user_disagreement_weights(S,u,qs["q50"],qs["q90"],method["lambda_max"])
        return (1-lam)*ab+lam*S["C"]
    raise ValueError(kind)


def test(args):
    locked=Path(args.selection).resolve()
    if not locked.is_file():
        raise FileNotFoundError(f"First run --phase select on VALID: {locked}")
    spec=json.loads(locked.read_text(encoding="utf-8"))
    if spec.get("selection_split")!="valid" or spec.get("seed")!=SEED:
        raise RuntimeError("Invalid locked valid-selection file")
    members,digest=load_triplet(args,"test")
    valid_paths = set(str(Path(p).resolve()) for p in spec.get("source_npz",{}).values())
    test_paths = set(m.path for m in members.values())
    if valid_paths & test_paths:
        raise RuntimeError("VALID and TEST share an NPZ file; dataset leakage guard")
    if digest == spec.get("valid_sha256_sample_alignment"):
        raise RuntimeError("VALID and TEST sample identity digests are identical; abort")
    a=members["A"]
    tests={"A":0.739652,"B":0.739113,"C":0.7369}
    member_metrics={k:metrics(a.labels,a.users,m.score) for k,m in members.items()}
    for k,ref in tests.items():
        delta=abs(member_metrics[k]["auc"]-ref)
        if delta>args.reference_tol:
            raise RuntimeError(f"Member {k} checkpoint/split mismatch: AUC={member_metrics[k]['auc']:.8f}, expected={ref}, delta={delta:.8g}>tol={args.reference_tol}")
    audit=None
    if not args.skip_rr5_audit:
        if not args.rr5_checkpoint:
            raise RuntimeError("Set --rr5-checkpoint to audit gate/MF table, or explicitly --skip-rr5-audit (diagnostic only)")
        audit=audit_rr5_gate(members["B"],args.rr5_checkpoint)
        if abs(audit["reconstructed_auc"]-tests["B"])>0.0001:
            raise RuntimeError(f"RR5 reconstructed AUC does not match 0.739113 within 1e-4: {audit['reconstructed_auc']}")
    if spec["norm_mode"]=="valid_stats_128":
        S,used=normalize_members(members,"valid_stats_128",spec["valid_first128_stats"])
    else:
        S,used=normalize_members(members,"test_first128")
    rows=[]
    base=member_metrics["A"]
    for tag,method in spec["methods"].items():
        metric=metrics(a.labels,a.users,predict(S,a.users,method))
        row={"method":tag,"test_auc":metric["auc"],"test_uauc":metric["uauc"],
             "delta_auc_vs_RR1":metric["auc"]-base["auc"],
             "delta_uauc_vs_RR1":metric["uauc"]-base["uauc"],
             "uauc_users":metric["uauc_users"]}
        rows.append(row)
    print("[TEST, FROZEN VALID RULES, NO FITTING]")
    for r in rows:
        print(f"{r['method']:30s} auc={r['test_auc']:.8f} uauc={r['test_uauc']:.8f} delta_auc={r['delta_auc_vs_RR1']:+.8f} delta_uauc={r['delta_uauc_vs_RR1']:+.8f}")
    out=Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite existing TEST results: {out}")
    out.mkdir(parents=True)
    with (out/"test_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    full={"seed":SEED,"selection_file":str(locked),"selection_sha256":hashlib.sha256(locked.read_bytes()).hexdigest(),
          "test_sample_alignment_sha256":digest,"test_sources":{k:m.path for k,m in members.items()},
          "test_member_metrics":member_metrics,"RR5_audit":audit,"normalization_stats":used,
          "reference_tol":args.reference_tol,"results":rows,
          "notes":"Test labels are used ONLY for evaluating frozen configurations; no weight selection on test."}
    (out/"test_summary.json").write_text(json.dumps(full,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"[TEST WRITE DONE] {out}")


def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase",choices=["select","test"],required=True)
    for split in ("valid","test"):
        for name in "abc":
            p.add_argument(f"--{name}-{split}",dest=f"{name}_{split}",default=None)
    for name in "abc":
        p.add_argument(f"--{name}-score-key",dest=f"{name}_score_key",default=None)
    p.add_argument("--selection",default="out/CORA_ENSEMBLE_REPRO_20261113/selection.json")
    p.add_argument("--output-dir",default="out/CORA_ENSEMBLE_REPRO_20261113/TEST")
    p.add_argument("--norm",choices=["test_first128","valid_stats_128"],default="test_first128",
                   help="test_first128 reproduces historical transductive protocol; valid_stats_128 is stricter inductive normalization")
    p.add_argument("--rr5-checkpoint",default=None)
    p.add_argument("--expected-historical-w",type=float,default=0.40,
                   help="Historical AB AUC validation optimum; enforce exact match by default")
    p.add_argument("--allow-different-valid-weight",action="store_true",
                   help="VALID diagnostic only: allow a best weight different from historical 0.40")
    p.add_argument("--skip-rr5-audit",action="store_true")
    p.add_argument("--reference-tol",type=float,default=0.002)
    p.add_argument("--overwrite-selection",action="store_true",help="Overwrite VALID selection only; never test outputs")
    args=p.parse_args()
    required=[f"{x}_{args.phase}" for x in "abc"]
    if any(getattr(args,k) is None for k in required):
        p.error("Missing required NPZ arguments: "+", ".join("--"+k.replace('_','-') for k in required))
    if args.expected_historical_w is not None and not (0 <= args.expected_historical_w <= 1):
        p.error("--expected-historical-w must be in [0,1]")
    if args.reference_tol <= 0 or args.reference_tol > .01:
        p.error("--reference-tol must be in (0,.01]")
    return args


if __name__=="__main__":
    try:
        args=parse()
        if args.phase=="select":
            select(args)
        else:
            test(args)
    except Exception as e:
        print(f"[FAIL CLOSED] {type(e).__name__}: {e}",file=sys.stderr,flush=True)
        raise
