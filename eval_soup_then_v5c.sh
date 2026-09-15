#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
for i in $(seq 1 40); do
  FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
  [ "$FREE" -ge 18000 ] && break
  sleep 60
done
echo "=== eval soup_012 ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_collm_mf_din.py \
  --cfg-path tran_configs/collm_pretrain_mf_ood_b3_rank_eval.yaml \
  --options model.ckpt=/home/bbx/user_lx/GraphLoRA/output/graphlora_ml1m_b3_rank_lowlr/soup_012.pth \
            run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_soup_best \
  > eval_soup_best.log 2>&1
grep -oE "rank_0 auc: 0\.[0-9]+" eval_soup_best.log || tail -2 eval_soup_best.log | cut -c1-200
echo "=== launch v5c idlist batch1 ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY train_collm_mf_din.py \
  --cfg-path tran_configs/collm_pretrain_mf_ood_b3_idlist.yaml \
  --options run.batch_size_train=1 run.batch_size_eval=2 \
  > graphlora_b3_idlist_train.log 2>&1 &
echo "v5c PID: $!"
nohup bash gpu_monitor.sh b3_v5c_idlist >/dev/null 2>&1 &
