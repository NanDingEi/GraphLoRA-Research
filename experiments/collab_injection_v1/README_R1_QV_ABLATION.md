# R1: original S2 L31 Q/V GNN injection diagnostic (NO training)

Purpose: determine whether original GraphLoRA GNN injection helps the
final LLM Yes/No ranking, which L31 attention branch matters, and
whether half-strength V injection changes performance. All inference
uses original S2 champion model weights and native L31 Yes token.

Original checkpoint:
`/home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/L31_output/20261017143/checkpoint_best.pth`
Expected S2 Valid Small: AUC=0.7296752596287361, UAUC=0.6841670211619721, n=5200.

## Conditions, fixed before evaluation

| Name | Q beta | V beta | Interpretation |
|---|---:|---:|---|
| R1_A_S2_QV | 0.1 | 0.1 | Original S2 anchor |
| R1_B_OFF | 0.0 | 0.0 | Both L31 GNN additions disabled |
| R1_C_V_ONLY | 0.0 | 0.1 | Q disabled, original V |
| R1_D_Q_ONLY | 0.1 | 0.0 | V disabled, original Q |
| R1_E_V_HALF | 0.0 | 0.05 | Q disabled, half V |

This is a **post-hoc ablation on a trained checkpoint**, not a new
no-GNN training baseline. Only q_proj/v_proj MyLinear.gnn_scale changes;
all parameter tensors, GNN representation, prompt, data and evaluation
protocol remain unchanged. No LoRA/LM-head modification or trained gate.

## Commands (GPU0 foreground, valid only)

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
git merge --ff-only FETCH_HEAD
conda activate minigpt4
export CUDA_VISIBLE_DEVICES=0

python -m py_compile \
  experiments/collab_injection_v1/r1_eval_s2_qv_ablation.py \
  experiments/collab_injection_v1/r1_analyze_s2_qv_ablation.py \
  experiments/collab_injection_v1/test_r1_qv_ablation_cpu.py
python experiments/collab_injection_v1/test_r1_qv_ablation_cpu.py

# Only if e3 S2 control YAML does not already exist:
test -f tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml || \
  python experiments/collab_injection_v1/prepare_e3_configs.py

set -o pipefail
python -u experiments/collab_injection_v1/r1_eval_s2_qv_ablation.py \
  --config tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml \
  --result logs/collab_v1/S2_R1_QV_ABLATION.json \
  2>&1 | tee logs/collab_v1/S2_R1_QV_ABLATION.log

# CPU post-hoc diagnostics: train_ood2.pkl path derived automatically
# from the same config, with optional --train-pkl to override.
python -u experiments/collab_injection_v1/r1_analyze_s2_qv_ablation.py \
  --predictions logs/collab_v1/S2_R1_QV_ABLATION.npz \
  --manifest logs/collab_v1/S2_R1_QV_ABLATION.json \
  --out logs/collab_v1/S2_R1_QV_ANALYSIS.json \
  2>&1 | tee logs/collab_v1/S2_R1_QV_ANALYSIS.log
```

## Review

- [R1-RESULT] 5 conditions: native L31 Yes logit AUC/UAUC.
- [R1-COMPLETE]: predictions NPZ and checkpoint/config manifests.
- [R1-SCORE]: delta against S2, for all 5 conditions.
- [R1-PAIRS]: strict rescued/damaged positive-negative ranking pairs.
- [R1-POP]: separate observed/training-item-frequency descriptive groups.
- [R1-ANALYSIS-COMPLETE]: JSON of pair counts, UAUC weighting, and strata.

Strict rescue/damage counts refer to moving from S2 ON baseline to
each alternate condition (not the other way around); ties receive
half-credit in AUC delta. Positive-weighted UAUC is calculated
per-user, not from pooled same-user positive/negative pairs.

Item popularity counts EVERY training interaction row, without accessing
validation/test labels for popularity and without selecting threshold
using Valid performance. Unseen items are their own group. Frequency
rank bins are based on unique items occurring in train; ties may create
empty quantiles. AUC computed within a popularity subgroup is not
additively decomposable into whole-valid AUC.

## Interpretation / stopping

Anchoring AUC & UAUC must match historical original S2 within 2e-5.
Each condition must return 5200 predictions in exactly same user/item/
label order. ON vs OFF must actually differ or the probe stops.
No original S2 or P0 checkpoints are overwritten. If any condition fails
the program stops and does NOT write a 'complete' analysis archive.

R2 must not learn its gate using the same Valid labels that were used
to characterize R1 effects. Hold out user-disjoint data (or cross-fit)
before claiming a conditionally gated improvement.
