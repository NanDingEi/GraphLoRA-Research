# Inference-only late-layer diagnostic: D4_TOPK_RES025 + C6_KD_BOTH

This is not a training experiment. It restores the existing best D4 trainable-only checkpoint and runs one pass over the Valid split. It does not run backward, optimizer steps, checkpoint saves, or Test evaluation.

## What it measures

- Decoder layers 29, 30, 31, and 32 (1-based).
- For each layer, capture the target-position hidden vector at the official answer position -t_posi, apply the shared final RMSNorm and LM Head, and read both Yes and No token logits. The script evaluates both the Yes-only score and the Yes-minus-No margin, then checks which one reproduces the official Layer-32 AUC.
- Layer 32 reconstruction is checked against the model's final forward-hook logit.
- Per-layer under both score conventions (Yes raw logit; Yes-minus-No margin): AUC, per-user UAUC, Brier score, accuracy, probability/logit distribution, saturation rate.
- Pairwise under both score conventions: Spearman correlation, prediction disagreement, error overlap/Jaccard.
- Complementarity: for each earlier layer, how many Layer-32 errors it corrects and how often it damages a correct Layer-32 prediction.
- Saves per-example user/item/label/logit/probability/prediction/error rows for audit.

UAUC is computed as the unweighted mean of per-user AUC over users having both positive and negative labels. The report includes the number of computed and excluded users. The script compares both Layer-32 score conventions with the official runner's aggregate AUC and selects the matching convention only when the absolute difference is at most 0.005. If neither matches, do not interpret intermediate-layer results until the official scoring expression is reconciled.

## Checkpoint and configuration

The script resolves D4_TOPK_RES025 from out/cora_all_valid_best.csv, requires a completed run and CKPT=FOUND, then loads the unique checkpoint_best.pth under that protocol. It uses the existing CORA_STRUCT10_20261101/D4_TOPK_RES025.yaml config and the same D4 neighbor-mix runtime setup. No weights are overwritten.

The checkpoint must be the trainable-only CoRA/MF checkpoint created by the project's compact atomic saver. Frozen Vicuna weights come from the configured local base model.

## Server command

Run inside the already working CoRA environment:

\`\`\`bash
cd /home/bbx/user_lx/CoRA_official

REV=584aa8b8642dec135bcbc59119dac26aa3d155cc
BASE=https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/$REV/experiments/no31_cora_original
curl -fL --retry 3 "$BASE/cora_late_layer_diag.py" -o cora_late_layer_diag.py

python -m py_compile cora_late_layer_diag.py

# Refresh the local all-experiment index; this does not train or evaluate.
python cora_all_summary.py --sort auc

# If the D4 config is absent only, create configs/manifests (no training):
test -f local_configs/CORA_STRUCT10_20261101/D4_TOPK_RES025.yaml || \
  python cora_struct10.py --plan

# One inference-only Valid pass, GPU0.
TAG="LATE_LAYER_DIAG_D4_C6_$(date +%Y%m%d_%H%M%S)"
mkdir -p "logs/$TAG"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u cora_late_layer_diag.py --run --tag "$TAG" \
  2>&1 | tee "logs/$TAG/run.log"
\`\`\`

## Output

- out/<TAG>/diagnostics.json
- out/<TAG>/per_layer_metrics_yes_raw_logit.csv
- out/<TAG>/per_layer_metrics_yes_minus_no_logit.csv
- out/<TAG>/pairwise_metrics_*.csv
- out/<TAG>/final_layer_rescue_*.csv
- out/<TAG>/sample_predictions.csv
- out/<TAG>/layer_logits_valid.npz
- logs/<TAG>/run.log

The script replaces the runner's train() entry point with a one-shot Valid evaluation, sets all model parameters to requires_grad=False after loading, and never calls the original training loop. It writes no model checkpoint.
