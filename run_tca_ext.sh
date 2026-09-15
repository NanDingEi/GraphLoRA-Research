#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
wait_gpu () {
  for i in $(seq 1 120); do
    FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
    [ "$FREE" -ge 19500 ] && return 0
    sleep 60
  done
  return 1
}
DONE=0
for ATTEMPT in 1 2; do
  wait_gpu || { echo "attempt $ATTEMPT: GPU busy"; continue; }
  echo "=== tca_full 16ep attempt $ATTEMPT start $(date '+%H:%M') ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_tca_full.yaml \
      --options run.max_epoch=16 run.batch_size_train=1 run.batch_size_eval=2 \
                run.save_checkpoint_every_n_epochs=0 \
    > tca_full_ext.log 2>&1
  NAUC=$(grep -c "rank_0 auc" tca_full_ext.log)
  echo "attempt $ATTEMPT: $NAUC/16 evals"
  [ "$NAUC" -ge 16 ] && { DONE=1; break; }
  sleep 120
done
[ "$DONE" != "1" ] && { echo "train failed"; exit 1; }
grep -oE "rank_0 auc: 0\.[0-9]+" tca_full_ext.log
BEST=$(ls -t output/graphlora_ml1m_scratch_tca_full/*/checkpoint_best.pth 2>/dev/null | head -1)
echo "best: $BEST"
wait_gpu || exit 0
echo "=== tca_full test eval ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_tca_full.yaml \
    --options model.ckpt=$BEST run.evaluate=True run.batch_size_eval=1 \
              run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_tca_full \
  > eval_tca_full.log 2>&1
echo "tca_full test:"; grep -oE "rank_0 auc: 0\.[0-9]+" eval_tca_full.log
find output -name "*.pth" -delete 2>/dev/null
echo TCA_EXT_DONE
