# CoRA Late-layer Voting / Mutual-consistency experiments

This suite implements the first of the two proposed directions. It uses D4_TOPK_RES025 as the fixed backbone and changes only how late-layer Yes-token predictions are trained/combined.

## Verified CoRA forward semantics

Official CoRA MiniGPT4Rec_vx.forward_v2 uses:
- answer target position -t_posi, where t_posi = padded_answer_length + 1;
- the positive answer token's raw logit;
- BCEWithLogits(yes_logit, label) as the main loss.

The runner hooks decoder layers 24, 28, 31, and 32 (1-based). It captures only the hidden vector at the exact same -t_posi location, then applies the same final LlamaRMSNorm and lm_head to obtain each layer's Yes logit. The final-layer reconstruction is checked against the official output (max absolute error <= 0.05). It does not enable output_hidden_states=True for all 32 layers and does not retain full sequence tensors.

## Four controlled trials

| ID | Training objective | Evaluation prediction |
|---|---|---|
| L1_LATE_SOFTVOTE | Replace final-layer main BCE with BCE of the mean of four late-layer Yes probabilities; retain C6 CTR/KD | Mean probability (soft vote) |
| L2_LATE_PAIRKL | Final-layer BCE + 0.02 × mean pairwise symmetric Bernoulli KL across four late layers; retain C6 CTR/KD | Final layer |
| L3_LATE_CONSENSUS | Same pairwise consistency objective as L2 | If at least 3/4 layers agree, use mean probability; otherwise use final layer; 2:2 ties fall back to final layer |
| L4_LATE_AUXBCE | Final-layer BCE + 0.10 × mean BCE of the four late layers; retain C6 CTR/KD | Mean probability (soft vote) |

Fixed controls:
- D4 inputs: four layer groups TopK40/20/10/5, z = 0.25 * MF + 0.75 * neighbor.
- C6 losses: CTR=0.005, KD=0.01, with agreement AND confidence (abs(p-0.5)>=0.20) filtering.
- CoRA rank16/alpha32; MF trainable; Vicuna frozen; original prompt.
- Seed42; CoRA LR=3e-4; MF LR=3e-5; batch2/accum4/400 microsteps; patience10.
- GPU0, serial, Valid only. Trainable-parameter-only atomic best checkpoint. No Test.

## Server commands

Run in the existing /home/bbx/user_lx/CoRA_official environment that successfully ran the previous CoRA rounds. This wrapper depends on the local CoRA source tree and the existing cora_joint8.py, cora_combo8.py, cora_rankgran7.py, and cora_struct10.py.

\`\`\`bash
cd /home/bbx/user_lx/CoRA_official

REV=661d71e2f95609a409776ff91bdbca49691bc103
BASE=https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/$REV/experiments/no31_cora_original

curl -fL --retry 3 "$BASE/cora_joint8.py" -o cora_joint8.py
curl -fL --retry 3 "$BASE/cora_combo8.py" -o cora_combo8.py
curl -fL --retry 3 "$BASE/cora_rankgran7.py" -o cora_rankgran7.py
curl -fL --retry 3 "$BASE/cora_struct10.py" -o cora_struct10.py
curl -fL --retry 3 "$BASE/cora_all_summary.py" -o cora_all_summary.py
curl -fL --retry 3 "$BASE/cora_latevote4.py" -o cora_latevote4.py

python -m py_compile cora_joint8.py cora_combo8.py cora_rankgran7.py \
  cora_struct10.py cora_all_summary.py cora_latevote4.py

# CPU-only numerical tests; no GPU training
python cora_latevote4.py --cpu-test

# Generate four isolated configs and run API/control preflight
python cora_latevote4.py --plan

# Serial GPU0 training, four trials, stop on first failure
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u cora_latevote4.py --run --patience 10 \
  2>&1 | tee -a logs/CORA_LATEVOTE4_20261102/launcher.log

# Refresh all-time Valid summary
python cora_all_summary.py --sort auc
tail -60 logs/CORA_ALL_BEST_SUMMARY.log
\`\`\`

To resume only unstarted trials, use e.g.:

\`\`\`bash
CUDA_VISIBLE_DEVICES=0 python -u cora_latevote4.py --run \
  --only L2_LATE_PAIRKL,L3_LATE_CONSENSUS,L4_LATE_AUXBCE --patience 10
\`\`\`

Existing trial logs are never overwritten. A failed trial stops the serial runner; inspect logs/CORA_LATEVOTE4_20261102/<ID>.log before continuing.

## Output

- Logs: logs/CORA_LATEVOTE4_20261102/<ID>.log
- Configs: local_configs/CORA_LATEVOTE4_20261102/<ID>.yaml
- Best checkpoints: output/CORA_LATEVOTE4_20261102/<ID>/.../checkpoint_best.pth
- Manifest: logs/CORA_LATEVOTE4_20261102/latevote4_manifest.json
- Global summary: logs/CORA_ALL_BEST_SUMMARY.log
- CSV/Markdown: out/cora_all_valid_best.csv and .md

This is code implementation, not a completed training result. The CPU test and plan must pass on the server before starting GPU training.
