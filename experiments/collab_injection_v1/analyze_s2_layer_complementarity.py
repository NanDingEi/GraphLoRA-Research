#!/usr/bin/env python3
"""D1-D3 S2 late-layer prediction complementarity. CPU only, NO training.

Reads eval_s2_multilayer_vote.py's original Valid Small .npz and companion
.json. Measures correlations, *paired* AUC ranking rescues/damages, and
predeclared label-free disagreement-gated voting.

Important: rescue/damage and group outcomes use labels ONLY for post-hoc
diagnostics. They are NOT deployable gates or evidence of improvements until
a future held-out split corroborates. No model weights or Test data loaded.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HISTORICAL_AUC = 0.7296752596287361
HISTORICAL_UAUC = 0.6841670211619721


def to_python_float(x):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("Non-finite metric")
    return x


def array_zscore(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all() or x.size < 2:
        raise ValueError("Invalid scores for z calibration")
    std = float(x.std())
    if std < 1e-10:
        raise ValueError("Constant score column")
    return (x-x.mean())/std


def weighted_uauc(users, labels, score):
    """Match rec_base_task.uAUC_pandas: per-user AUC weighted by n_positive."""
    users, labels, score = map(np.asarray, (users, labels, score))
    auc_sum = 0.0
    positive_weight = 0
    eligible = 0
    for uid in np.unique(users):
        m = users == uid
        ys = labels[m]
        pos = int(np.count_nonzero(ys == 1))
        neg = int(np.count_nonzero(ys == 0))
        if not pos or not neg:
            continue
        auc_sum += pos * float(roc_auc_score(ys, score[m]))
        positive_weight += pos
        eligible += 1
    return (auc_sum/positive_weight if positive_weight else float("nan"),
            eligible, positive_weight)


def auc_metrics(user, labels, score):
    user = np.asarray(user)
    labels = np.asarray(labels)
    score = np.asarray(score)
    if not (user.shape == labels.shape == score.shape):
        raise ValueError("AUC arrays must have identical sample length")
    if np.unique(labels).tolist() != [0,1]:
        return {"n": int(len(labels)), "auc": None, "uauc": None,
                "eligible_users": 0}
    a = to_python_float(roc_auc_score(labels, score))
    ua, eligible, positive_weight = weighted_uauc(user, labels, score)
    return {"n": int(len(labels)), "auc": a,
            "uauc": to_python_float(ua) if eligible else None,
            "eligible_users": eligible,
            "uauc_positive_weight": positive_weight}


def pair_ledger(base_pos, base_neg, alt_pos, alt_neg, chunk=128):
    """Process labeled positive-negative pairs, bounded RAM for 5200 samples.

    Strict rescue: baseline negative margin, alternative positive margin.
    Strict damage: baseline positive margin, alternative negative margin.
    Ties get AUC half-credit; net_credit checks exact AUC difference,
    including ties and changes involving ties.
    """
    base_pos, base_neg, alt_pos, alt_neg = (
        np.asarray(t, dtype=np.float64) for t in
        (base_pos, base_neg, alt_pos, alt_neg)
    )
    assert base_pos.shape == alt_pos.shape and base_neg.shape == alt_neg.shape
    ledger = {"pairs": 0, "strict_rescue": 0, "strict_damage": 0,
              "base_correct": 0, "base_wrong": 0, "base_tie": 0,
              "alt_correct": 0, "alt_wrong": 0, "alt_tie": 0,
              "changed_tie_pairs": 0, "net_auc_credit": 0.0}
    for start in range(0, len(base_pos), chunk):
        a = base_pos[start:start+chunk, None] - base_neg[None, :]
        b = alt_pos[start:start+chunk, None] - alt_neg[None, :]
        n = int(a.size)
        ledger["pairs"] += n
        ac, aw, at = (a > 0), (a < 0), (a == 0)
        bc, bw, bt = (b > 0), (b < 0), (b == 0)
        ledger["strict_rescue"] += int(np.count_nonzero(aw & bc))
        ledger["strict_damage"] += int(np.count_nonzero(ac & bw))
        for key, mask in (
            ("base_correct", ac), ("base_wrong", aw), ("base_tie", at),
            ("alt_correct", bc), ("alt_wrong", bw), ("alt_tie", bt),
        ):
            ledger[key] += int(np.count_nonzero(mask))
        ledger["changed_tie_pairs"] += int(np.count_nonzero((at | bt) & (at != bt)))
        # sum of score differences in pairwise AUC credit, with ties at 1/2.
        ledger["net_auc_credit"] += float(
            np.count_nonzero(bc) - np.count_nonzero(ac) +
            0.5*(np.count_nonzero(bt)-np.count_nonzero(at)))
    if ledger["pairs"] != len(base_pos)*len(base_neg) or not ledger["pairs"]:
        raise RuntimeError("Positive-negative pair counting failed")
    n = ledger["pairs"]
    ledger["net_delta_auc"] = ledger["net_auc_credit"]/n
    ledger["strict_rescue_rate"] = ledger["strict_rescue"]/n
    ledger["strict_damage_rate"] = ledger["strict_damage"]/n
    ledger["rescue_over_baseline_strict_wrong"] = (
        ledger["strict_rescue"]/ledger["base_wrong"]
        if ledger["base_wrong"] else None)
    ledger["damage_over_baseline_strict_correct"] = (
        ledger["strict_damage"]/ledger["base_correct"]
        if ledger["base_correct"] else None)
    return ledger


def paired_complementarity(user, labels, baseline, alternate, chunk=128):
    """AUC-correct pair ledger, with separate pooled same-user pairs and UAUC.

    IMPORTANT pooled same-user pair frequencies are NOT the experiment's UAUC.
    UAUC weights per-user AUC by user positive interaction count, regardless
    of the number of negative interactions that user has.
    """
    user = np.asarray(user)
    labels = np.asarray(labels)
    baseline = np.asarray(baseline)
    alternate = np.asarray(alternate)
    p, n = labels == 1, labels == 0
    global_pairs = pair_ledger(
        baseline[p], baseline[n], alternate[p], alternate[n], chunk)
    b_auc = float(roc_auc_score(labels, baseline))
    a_auc = float(roc_auc_score(labels, alternate))
    if abs(global_pairs["net_delta_auc"]-(a_auc-b_auc)) > 1e-10:
        raise AssertionError("Global pair ledger != independent sklearn AUC delta")

    valid_users = 0
    positive_weights = 0
    weighted_base = 0.
    weighted_alt = 0.
    weighted_net = 0.
    weighted_rescue = 0.
    weighted_damage = 0.
    pooled = {"pairs": 0, "strict_rescue": 0, "strict_damage": 0,
              "base_wrong": 0, "base_correct": 0, "changed_tie_pairs": 0}
    for uid in np.unique(user):
        m = user == uid
        labels_u = labels[m]
        pu, nu = labels_u == 1, labels_u == 0
        npos = int(pu.sum())
        nneg = int(nu.sum())
        if not npos or not nneg:
            continue
        valid_users += 1
        b = baseline[m]
        a = alternate[m]
        r = pair_ledger(b[pu], b[nu], a[pu], a[nu], chunk)
        b_auc_u = float(roc_auc_score(labels_u, b))
        a_auc_u = float(roc_auc_score(labels_u, a))
        if abs(r["net_delta_auc"]-(a_auc_u-b_auc_u)) > 1e-10:
            raise AssertionError(f"User {uid}: pair ledger != AUC delta")
        positive_weights += npos
        weighted_base += npos*b_auc_u
        weighted_alt += npos*a_auc_u
        weighted_net += npos*r["net_delta_auc"]
        weighted_rescue += npos*r["strict_rescue_rate"]
        weighted_damage += npos*r["strict_damage_rate"]
        for k in pooled:
            pooled[k] += r[k]
    if not positive_weights:
        raise RuntimeError("No users with both labels")
    original_ua = weighted_base/positive_weights
    alt_ua = weighted_alt/positive_weights
    d_ua = (weighted_alt-weighted_base)/positive_weights
    if abs(weighted_net/positive_weights - d_ua) > 1e-10:
        raise AssertionError("User-pair credit != weighted UAUC delta")
    pooled["strict_rescue_fraction"] = pooled["strict_rescue"]/pooled["pairs"]
    pooled["strict_damage_fraction"] = pooled["strict_damage"]/pooled["pairs"]
    return {
        "global_auc_pairs": global_pairs,
        "within_user_pooled_pairs_not_uauc": pooled,
        "positive_weighted_uauc": {
            "eligible_users": valid_users,
            "positive_weight": positive_weights,
            "baseline_uauc": original_ua,
            "alternate_uauc": alt_ua,
            "delta_uauc": d_ua,
            "weighted_strict_rescue_fraction": weighted_rescue/positive_weights,
            "weighted_strict_damage_fraction": weighted_damage/positive_weights,
        },
    }


def diagnostic_groups(user, labels, raw_s2, raw_l30, discrepancy, groups=4):
    """Descriptive, equal-count sample quantiles of *unlabeled* disagreement.

    AUC in filtered subsets is NOT a decomposition of full-data AUC.
    We report it only to describe where predictions differ, not as a
    training/threshold selection signal.
    """
    order = np.argsort(discrepancy, kind="stable")
    blocks = np.array_split(order, groups)
    out = []
    for i, ids in enumerate(blocks):
        bm = auc_metrics(user[ids], labels[ids], raw_s2[ids])
        am = auc_metrics(user[ids], labels[ids], raw_l30[ids])
        out.append({
            "rank_quartile": i+1, "n_samples": int(len(ids)),
            "disagreement_min": float(discrepancy[ids].min()),
            "disagreement_max": float(discrepancy[ids].max()),
            "positive_rate": float(labels[ids].mean()),
            "native_S2": bm, "L30": am,
            "delta_auc": (am["auc"]-bm["auc"])
                if am["auc"] is not None and bm["auc"] is not None else None,
            "delta_uauc": (am["uauc"]-bm["uauc"])
                if am["uauc"] is not None and bm["uauc"] is not None else None,
        })
    return out


def fixed_diagnostic_gates(z_s2, z_l30):
    """Predeclared 10/25%-weight votes; thresholds use NO labels.

    Calibration and quantiles come from unlabelled Valid scores (transductive).
    This is exploratory only; evaluate any selected rule on a new held-out set.
    """
    discrepancy = np.abs(z_l30-z_s2)
    res = {
        "fixed_s2_90_l30_10": 0.90*z_s2+0.10*z_l30,
        "fixed_s2_75_l30_25": 0.75*z_s2+0.25*z_l30,
    }
    for q in (0.75,0.90):
        threshold = float(np.quantile(discrepancy, q))
        selected = discrepancy > threshold
        name = f"gate_disagreement_q{int(q*100)}_s2_75_l30_25"
        res[name] = z_s2 + np.where(
            selected, 0.25*(z_l30-z_s2), 0.)
    return res


def run_analysis(data, expected_n=5200, atol=2e-5):
    required = ("user", "item", "label", "s2_native",
                "layers", "layer_logits")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing NPZ keys: {missing}")
    user = np.asarray(data["user"]).reshape(-1)
    item = np.asarray(data["item"]).reshape(-1)
    labels = np.asarray(data["label"]).reshape(-1)
    native = np.asarray(data["s2_native"], dtype=np.float64).reshape(-1)
    layers = np.asarray(data["layers"]).astype(int).tolist()
    logits = np.asarray(data["layer_logits"], dtype=np.float64)
    n = len(user)
    if n != expected_n:
        raise ValueError(f"Expected {expected_n} samples, got {n}")
    if not (len(item) == len(labels) == len(native) == n):
        raise ValueError("NPZ sample alignment mismatch")
    if logits.shape != (n,len(layers)) or len(set(layers)) != len(layers):
        raise ValueError("NPZ layer index or logit matrix mismatch")
    if 30 not in layers or 28 not in layers or 31 not in layers:
        raise ValueError("Expected L28, L30, L31 in input NPZ")
    if np.unique(labels).tolist() != [0,1]:
        raise ValueError("Binary labels required")
    if not np.isfinite(native).all() or not np.isfinite(logits).all():
        raise ValueError("NPZ scores contain NaN/Inf")
    s2 = auc_metrics(user, labels, native)
    if abs(s2["auc"]-HISTORICAL_AUC) > atol:
        raise RuntimeError(
            f"S2 AUC parity failed: {s2['auc']} != {HISTORICAL_AUC}")
    if abs(s2["uauc"]-HISTORICAL_UAUC) > atol:
        raise RuntimeError(
            f"S2 UAUC parity failed: {s2['uauc']} != {HISTORICAL_UAUC}")
    scores = {"S2_native": native}
    scores.update({f"L{l}": logits[:,i] for i,l in enumerate(layers)})
    print(f"[D-ANCHOR] samples={n} S2 AUC={s2['auc']:.9f} "
          f"UAUC={s2['uauc']:.9f} PASS", flush=True)

    # D1: correlations, which do not imply complementary errors.
    d1 = {}
    for name in ("L28", "L30", "L31"):
        s = scores[name]
        pearson = float(np.corrcoef(s, native)[0,1])
        rho = float(spearmanr(s, native).statistic)
        if not np.isfinite(pearson) or not np.isfinite(rho):
            raise RuntimeError(f"Invalid correlation for {name}")
        d1[name] = {"pearson_vs_S2": pearson, "spearman_vs_S2": rho}
        print(f"[D1-CORR] {name:4s} Pearson={pearson:.8f} "
              f"Spearman={rho:.8f}", flush=True)
    d1["L28_vs_L30"] = {
        "pearson": float(np.corrcoef(scores["L28"],scores["L30"])[0,1]),
        "spearman": float(spearmanr(scores["L28"],scores["L30"]).statistic),
    }

    # D2: numerator/denominator of ranking rescue and damage are explicit.
    d2 = {}
    for name in ("L28", "L30", "L31"):
        d2[name] = paired_complementarity(user, labels, native, scores[name])
        global_row = d2[name]["global_auc_pairs"]
        ua_row = d2[name]["positive_weighted_uauc"]
        print(f"[D2-PAIRS] {name:4s} global_pairs={global_row['pairs']} "
              f"rescue={global_row['strict_rescue']} "
              f"damage={global_row['strict_damage']} "
              f"tie_affected={global_row['changed_tie_pairs']} "
              f"delta_AUC={global_row['net_delta_auc']:+.9f} "
              f"delta_UAUC={ua_row['delta_uauc']:+.9f}",flush=True)

    # D3: unlabeled z-score and discrepancy-gating comparisons.
    z_s2 = array_zscore(native)
    z_l30 = array_zscore(scores["L30"])
    discrepancy = np.abs(z_l30-z_s2)
    d3 = {
        "definition": "abs(z(L30)-z(S2_native)), unlabeled normalization",
        "sample_disagreement_quartiles": diagnostic_groups(
            user,labels,native,scores["L30"],discrepancy),
        "fixed_label_free_gates": {},
    }
    votes = fixed_diagnostic_gates(z_s2,z_l30)
    for name, vote in votes.items():
        m = auc_metrics(user, labels, vote)
        m["delta_auc_vs_S2"] = m["auc"]-s2["auc"]
        m["delta_uauc_vs_S2"] = m["uauc"]-s2["uauc"]
        d3["fixed_label_free_gates"][name] = m
        print(f"[D3-GATE] {name:38s} "
              f"AUC={m['auc']:.9f} UAUC={m['uauc']:.9f} "
              f"dAUC={m['delta_auc_vs_S2']:+.9f} "
              f"dUAUC={m['delta_uauc_vs_S2']:+.9f}", flush=True)
    for row in d3["sample_disagreement_quartiles"]:
        print(f"[D3-GROUP] Q{row['rank_quartile']} n={row['n_samples']} "
              f"gap=[{row['disagreement_min']:.4f},"
              f"{row['disagreement_max']:.4f}] "
              f"L30_vs_S2_AUC={row['delta_auc']} "
              f"L30_vs_S2_UAUC={row['delta_uauc']}", flush=True)

    return {
        "status": "complete", "split": "valid_small", "n_samples": n,
        "S2_anchor": s2, "layers": layers,
        "D1": d1, "D2": d2, "D3": d3,
        "interpretation_limits": [
            "True-label pair rescues are retrospective and NOT a deployable gate",
            "Users with a single class are excluded from positive-weighted UAUC",
            "Pooled same-user pair rates are NOT UAUC",
            "Subgroup AUC is not additive and subgroup label mixes may differ",
            "Voting rules are fixed without labels but unlabeled Valid "
            "quantiles/z-scores are transductive",
            "No candidate method is selected as a winner on this Valid set",
            "Further gate training should use separate training calibration "
            "and untouched held-out validation; no Test tuning",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="logs/collab_v1/S2_MULTILAYER_VOTE.npz")
    ap.add_argument("--source-report",
                    default="logs/collab_v1/S2_MULTILAYER_VOTE.json")
    ap.add_argument("--output",
                    default="logs/collab_v1/S2_LAYER_COMPLEMENTARITY.json")
    ap.add_argument("--expected-samples",type=int,default=5200)
    args = ap.parse_args()
    path = Path(args.input).expanduser().resolve()
    source = Path(args.source_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not path.is_file() or not source.is_file():
        raise FileNotFoundError(
            f"Need existing vote artifacts: {path}, {source}")
    prior = json.loads(source.read_text(encoding="utf-8"))
    if prior.get("status") != "complete" or prior.get("split") != "valid":
        raise RuntimeError("Refusing NPZ without complete Valid vote provenance")
    if Path(prior["prediction_archive"]).resolve() != path:
        raise RuntimeError("NPZ does not match source vote report")
    with np.load(path,allow_pickle=False) as data:
        report = run_analysis(data,args.expected_samples)
    # Extra provenance checks beyond NPZ AUC/UAUC parity.
    if prior.get("n_samples") != report["n_samples"]:
        raise RuntimeError("NPZ and source JSON sample counts differ")
    report["source_npz"] = str(path)
    report["source_json"] = str(source)
    report["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),
                      encoding="utf-8")
    print(f"[D-COMPLETE] CPU only; {output}",flush=True)


if __name__ == "__main__":
    main()
