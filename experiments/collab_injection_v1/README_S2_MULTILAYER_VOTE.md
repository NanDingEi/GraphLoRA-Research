# S2 late-layer recommendation voting (Valid Small, no training)

New script: `experiments/collab_injection_v1/eval_s2_multilayer_vote.py`.

Run from the GraphLoRA_collab_v1 repository on GPU0:

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
# Save uncommitted work before checkout:
git checkout -B fix/collab-gated-optimizer FETCH_HEAD
export CUDA_VISIBLE_DEVICES=0
python -m py_compile experiments/collab_injection_v1/eval_s2_multilayer_vote.py
python experiments/collab_injection_v1/test_s2_multilayer_vote_cpu.py
python experiments/collab_injection_v1/prepare_e3_configs.py
set -o pipefail
python -u experiments/collab_injection_v1/eval_s2_multilayer_vote.py \
  --config tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml \
  --layers 24,26,28,30,31 \
  --result logs/collab_v1/S2_MULTILAYER_VOTE.json \
  2>&1 | tee logs/collab_v1/S2_MULTILAYER_VOTE.log
```

No training, no extra checkpoint, no Test predictions. The S2-L31 model remains frozen. Each validation batch receives exactly one unmodified full-model forward, and small forward hooks read decoder outputs at the first Yes prediction's actual final-prompt position (not the target-item token or appended answer).

For every chosen layer L the script computes `FP32 yes_logit = Yes_Row(lm_head)(final_RMSNorm(h_L))`. Intermediate layers use the pretrained shared final RMSNorm/LM head, with no extra trained classifier. This is a *logit-lens*, not an independently calibrated layer exit.

Reports: unmodified S2 native L31 AUC/UAUC (must match 0.7296752596); standalone AUC/UAUC per layer; equal soft voting; predeclared depth-linear weighted voting; S2-native anchored vote with 50% or 75% native weight. We z-score each logit stream without labels, as raw earlier-layer Yes logits may have very different scales. No validation labels are used to fit weights. This is an initial, unlabeled, distribution-calibrated transductive comparison; do not choose final weights by repeatedly tuning on Valid. Further claims need a distinct calibration set and untouched validation.

Output files:

- `logs/collab_v1/S2_MULTILAYER_VOTE.json`: complete metrics, score calibration means/std and S2 parity.
- `logs/collab_v1/S2_MULTILAYER_VOTE.npz`: users, items, labels, raw native S2 scores, all layer scores and vote scores.
- `logs/collab_v1/S2_MULTILAYER_VOTE.log`: full evaluation log; grep `VOTE-RESULT` and `VOTE-COMPLETE` for summary.

Failure guards: exact baseline AUC tolerance, full 5200 samples, valid-only evaluation, once-per-layer hook, finite scores and stable matched sample lengths.

To inspect additional layers in one run: `--layers 24,25,26,27,28,29,30,31`.

Do not draw conclusions until the script passes the native-S2 AUC parity gate. The tool is committed but not executed on the remote GPU.
## D1-D3: CPU-only cross-layer complementarity and selective voting

Requires the prior completed `S2_MULTILAYER_VOTE.npz` and matching `.json`.
No Vicuna loading, no GPU, no model updates, no Test evaluation.

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
conda activate minigpt4
python experiments/collab_injection_v1/test_s2_layer_complementarity_cpu.py
python -u experiments/collab_injection_v1/analyze_s2_layer_complementarity.py \
  --input logs/collab_v1/S2_MULTILAYER_VOTE.npz \
  --source-report logs/collab_v1/S2_MULTILAYER_VOTE.json \
  --output logs/collab_v1/S2_LAYER_COMPLEMENTARITY.json \
  2>&1 | tee logs/collab_v1/S2_LAYER_COMPLEMENTARITY.log
```

Outputs: `[D-ANCHOR]` for exact S2 AUC and weighted UAUC parity;
`[D1-CORR]` for L28/L30/L31 Pearson/Spearman correlations;
`[D2-PAIRS]` for global positive-negative pair ranking strict rescues, strict damages and ties, and user-weighted UAUC deltas;
`[D3-GATE]` for four predeclared *unlabeled* voting rules (S2 90/10 L30, 75/25 L30, high-disagreement Q75/Q90 only);
`[D3-GROUP]` for equal-count disagreement quantile groups.

Global pair corrections and damages are counted on labeled positive-negative pairs.
Ties get one-half AUC credit. User-AUC matches this repository's `uAUC_pandas`: positive-count-weighted per-user AUC;
raw pooled within-user pairs are NOT equivalent to UAUC.

True-label rescue/damage counts are retrospective diagnostics and do not constitute a deployable gating rule.
Unlabeled Valid normalization and quantiles make D3 explicitly transductive/exploratory.
Subgroup AUC cannot be added together to reconstruct full AUC.
Do not tune weights on this Valid set or Test; use a separate calibration partition for any later trained gate.
