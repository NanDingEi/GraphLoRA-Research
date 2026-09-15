#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
for i in $(seq 1 90); do
  FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
  [ "$FREE" -ge 18000 ] && break
  sleep 60
done
echo "=== scratch_rel start $(date '+%H:%M') ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel.yaml \
    --options run.max_epoch=3 run.batch_size_train=1 run.batch_size_eval=2 \
              run.save_checkpoint_every_n_epochs=0 \
  > scratch_rel.log 2>&1
echo "--- scratch_rel AUCs ---"
grep -oE "rank_0 auc: 0\.[0-9]+" scratch_rel.log || tail -3 scratch_rel.log | cut -c1-180
D=$(ls -dt output/graphlora_ml1m_scratch_rel/*/ 2>/dev/null | head -1)
[ -n "$D" ] && find "$D" -name "*.pth" -delete && echo "checkpoints cleaned"
echo V11_DONE
