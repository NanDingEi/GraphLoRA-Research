#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
for ATTEMPT in 1 2 3; do
  for i in $(seq 1 60); do
    FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
    [ "$FREE" -ge 18000 ] && break
    sleep 60
  done
  FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
  [ "$FREE" -lt 18000 ] && { echo "attempt $ATTEMPT: GPU busy, skip"; continue; }
  echo "=== attempt $ATTEMPT ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_collm_mf_din.py \
    --cfg-path tran_configs/collm_pretrain_mf_ood_b3_idlist_eval.yaml \
    --options run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_v5c_best run.batch_size_eval=1 \
    > eval_v5c_best.log 2>&1
  if grep -qE "rank_0 auc" eval_v5c_best.log; then
    grep -oE "rank_0 auc: 0\.[0-9]+" eval_v5c_best.log
    echo EVAL_DONE; exit 0
  fi
  echo "attempt $ATTEMPT failed (likely OOM), retrying after cooldown"; sleep 120
done
echo "ALL ATTEMPTS FAILED"; tail -3 eval_v5c_best.log | cut -c1-200
