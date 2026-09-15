#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
for i in $(seq 1 120); do
  FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
  [ "$FREE" -ge 19500 ] && break
  sleep 60
done
echo "=== rel_ema retry start $(date '+%H:%M') ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_ema.yaml \
    --options run.max_epoch=8 run.batch_size_train=1 run.batch_size_eval=2 \
              run.save_checkpoint_every_n_epochs=0 \
  > rel_ema.log 2>&1
NAUC=$(grep -c "rank_0 auc" rel_ema.log)
echo "evals done: $NAUC/8"
grep -oE "rank_0 auc: 0\.[0-9]+" rel_ema.log
if [ "$NAUC" -ge 8 ]; then
  BEST=$(ls -t output/graphlora_ml1m_scratch_rel_ema/*/checkpoint_best.pth 2>/dev/null | head -1)
  for i in $(seq 1 90); do
    FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
    [ "$FREE" -ge 18000 ] && break; sleep 60
  done
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_ema.yaml \
      --options model.ckpt=$BEST run.evaluate=True run.batch_size_eval=1 \
                run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_rel_ema \
    > eval_rel_ema.log 2>&1
  echo "test:"; grep -oE "rank_0 auc: 0\.[0-9]+" eval_rel_ema.log
  find output/graphlora_ml1m_scratch_rel_ema output/eval_rel_ema -name "*.pth" -delete 2>/dev/null
fi
echo EMA_RETRY_DONE
