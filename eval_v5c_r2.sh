#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
for i in $(seq 1 60); do
  FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
  [ "$FREE" -ge 16000 ] && break
  sleep 60
done
echo "=== eval v5c best r2 (batch1) ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_collm_mf_din.py \
  --cfg-path tran_configs/collm_pretrain_mf_ood_b3_idlist_eval.yaml \
  --options run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_v5c_best run.batch_size_eval=1 \
  > eval_v5c_best.log 2>&1
grep -oE "rank_0 auc: 0\.[0-9]+" eval_v5c_best.log || tail -3 eval_v5c_best.log | cut -c1-200
echo EVAL_DONE
