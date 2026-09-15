#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
for i in $(seq 1 60); do
  FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
  [ "$FREE" -ge 18000 ] && break
  sleep 60
done
FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
if [ "$FREE" -lt 18000 ]; then echo "GPU busy after 60min, abort"; exit 1; fi
echo "=== eval v5c best (test) ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_collm_mf_din.py \
  --cfg-path tran_configs/collm_pretrain_mf_ood_b3_idlist_eval.yaml \
  --options run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_v5c_best \
  > eval_v5c_best.log 2>&1
grep -oE "rank_0 auc: 0\.[0-9]+" eval_v5c_best.log
echo EVAL_DONE
