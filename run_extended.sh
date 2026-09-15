#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
wait_gpu () {
  for i in $(seq 1 120); do
    FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
    [ "$FREE" -ge 18000 ] && return 0
    sleep 60
  done
  return 1
}
for RUN in rel gate; do
  wait_gpu || { echo "run $RUN: GPU busy, skip"; continue; }
  echo "=== ext_$RUN train start $(date '+%H:%M') ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_$RUN.yaml \
      --options run.max_epoch=8 run.batch_size_train=1 run.batch_size_eval=2 \
                run.save_checkpoint_every_n_epochs=0 \
    > ext_$RUN.log 2>&1
  echo "--- ext_$RUN valid AUCs ---"
  grep -oE "rank_0 auc: 0\.[0-9]+" ext_$RUN.log
  BEST=$(ls -t output/graphlora_ml1m_scratch_$RUN/*/checkpoint_best.pth 2>/dev/null | head -1)
  if [ -n "$BEST" ]; then
    wait_gpu || { echo "eval $RUN: GPU busy, keep ckpt for later"; continue; }
    echo "--- ext_$RUN test eval ---"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_$RUN.yaml \
        --options model.ckpt=$BEST run.evaluate=True run.batch_size_eval=1 \
                  run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_ext_$RUN \
      > eval_ext_$RUN.log 2>&1
    grep -oE "rank_0 auc: 0\.[0-9]+" eval_ext_$RUN.log || tail -2 eval_ext_$RUN.log | cut -c1-150
    find output/graphlora_ml1m_scratch_$RUN -name "*.pth" -delete
    find output/eval_ext_$RUN -name "*.pth" -delete 2>/dev/null
    echo "checkpoints cleaned for $RUN"
  fi
done
echo EXT_DONE
