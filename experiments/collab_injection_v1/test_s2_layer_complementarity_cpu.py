#!/usr/bin/env python3
"""CPU regression tests for D1-D3 sample alignment and exact pair accounting."""
import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_s2_layer_complementarity import (
    array_zscore, auc_metrics, diagnostic_groups, fixed_diagnostic_gates,
    pair_ledger, paired_complementarity, weighted_uauc,
)

# 2x2 synthetic positive-negative pairs; baseline wrong on one and
# alternate fixes that pair but damages another. Keep exact denominators.
bpos = np.array([0.3,0.8])
bneg = np.array([0.4,0.7])
apos = np.array([0.5,0.8])
aneg = np.array([0.4,0.9])
r = pair_ledger(bpos,bneg,apos,aneg,chunk=1)
assert r["pairs"] == 4
assert r["strict_rescue"] == 1
assert r["strict_damage"] == 1
assert r["net_delta_auc"] == 0
print("[D-CPU] strict rescue/damage counts and denominators PASS",flush=True)

# Ties must contribute 0.5 AUC credit and be distinct from strict wins.
r = pair_ledger(np.array([0.3]),np.array([0.3]),
                np.array([0.4]),np.array([0.3]))
assert r["pairs"] == 1 and r["base_tie"] == 1
assert r["net_auc_credit"] == 0.5 and r["net_delta_auc"] == 0.5
assert r["strict_rescue"] == r["strict_damage"] == 0
print("[D-CPU] AUC ties use half-credit, not false rescues PASS",flush=True)

# Same-user pooled pair rate differs from positive-weighted uAUC. Include
# single-class and single-sample users to verify exclusions.
user = np.array([1,1,1,1,2,2,2,3,3,4])
label = np.array([1,1,0,0,1,0,0,1,1,0])
s2 = np.array([0.2,0.7,0.3,0.8,0.9,0.4,0.6,0.5,0.8,0.1])
alt = np.array([0.4,0.9,0.2,0.6,0.7,0.9,0.1,0.6,0.7,0.4])
p = paired_complementarity(user,label,s2,alt,chunk=1)
a0 = roc_auc_score(label,s2)
a1 = roc_auc_score(label,alt)
ua0,u0,w0 = weighted_uauc(user,label,s2)
ua1,u1,w1 = weighted_uauc(user,label,alt)
assert w0 == w1 == 3 and u0 == u1 == 2
assert abs(p["global_auc_pairs"]["net_delta_auc"]-(a1-a0)) < 1e-12
assert abs(p["positive_weighted_uauc"]["delta_uauc"]-(ua1-ua0)) < 1e-12
assert p["within_user_pooled_pairs_not_uauc"]["pairs"] == 6
assert p["positive_weighted_uauc"]["positive_weight"] == 3
print("[D-CPU] sklearn global AUC and source-compatible weighted UAUC PASS",
      flush=True)

# Predeclared gates: no labels, deterministic, preserve baseline on low
# disagreement rows. Scale invariance from unlabeled z-score.
u = array_zscore(np.array([2.,1.,3.,8.,4.,5.,9.,7.]))
v = array_zscore(np.array([8.,7.,6.,5.,4.,3.,2.,1.]))
d = np.abs(v-u)
votes = fixed_diagnostic_gates(u,v)
assert len(votes) == 4
for k,score in votes.items():
    assert score.shape == u.shape and np.isfinite(score).all(),k
for q in (75,90):
    score = votes[f"gate_disagreement_q{q}_s2_75_l30_25"]
    assert np.allclose(score[d <= np.quantile(d,q/100.)],u[d <= np.quantile(d,q/100.)])
assert np.allclose(fixed_diagnostic_gates(u,v)["fixed_s2_75_l30_25"],
                   votes["fixed_s2_75_l30_25"])
print("[D-CPU] label-free fixed gate formulas and quantiles PASS",flush=True)

# D3 subgroup AUC must operate on its own samples, never silently use all
# 5200 labels. Repeated score values and ties should not crash.
d = np.arange(len(user),dtype=np.float64)
groups = diagnostic_groups(user,label,s2,alt,d,groups=2)
assert len(groups) == 2 and sum(g["n_samples"] for g in groups) == len(user)
assert groups[0]["rank_quartile"] == 1
print("[D-CPU] sorted disjoint disagreement groups PASS",flush=True)

print("[D-CPU] ALL PASS; no model, GPU or dataset required",flush=True)
