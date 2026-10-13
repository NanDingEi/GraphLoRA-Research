#!/usr/bin/env bash
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
CFG=tran_configs/S2_candidate_history_A1.yaml
mkdir -p log
echo "A1 candidate-history fusion ON; evaluate with the same S2 data/protocol"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_collm_mf_din.py \
    --cfg-path "$CFG" \
    2>&1 | tee log/S2_CAND_A1_train.log
