#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python

wait_gpu () {
  for i in $(seq 1 90); do
    FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
    [ "$FREE" -ge 18000 ] && return 0
    sleep 60
  done
  return 1
}

# ---- stage 0: smoke ----
SMOKE_OK=0
for ATTEMPT in 1 2; do
  wait_gpu || { echo "smoke: GPU busy"; break; }
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY smoke_test_schemes.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3_smoke.yaml \
    > smoke_test_schemes.log 2>&1
  if grep -q "ALL SMOKE TESTS PASSED" smoke_test_schemes.log; then
    grep -E "^\[[0-9]/6\]" smoke_test_schemes.log; SMOKE_OK=1; break
  fi
  echo "smoke attempt $ATTEMPT failed:"; tail -4 smoke_test_schemes.log | cut -c1-180; sleep 60
done
if [ "$SMOKE_OK" != "1" ]; then echo SMOKE_FAILED; exit 1; fi

# ---- stage 1..4: sequential scratch runs ----
for RUN in gate expert expert_ml idlist; do
  wait_gpu || { echo "run $RUN: GPU busy, skip"; continue; }
  echo "=== scratch_$RUN start $(date '+%H:%M') ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_$RUN.yaml \
      --options run.max_epoch=3 run.batch_size_train=1 run.batch_size_eval=2 \
                run.save_checkpoint_every_n_epochs=0 \
    > scratch_$RUN.log 2>&1
  echo "--- scratch_$RUN AUCs ---"
  grep -oE "rank_0 auc: 0\.[0-9]+" scratch_$RUN.log || tail -3 scratch_$RUN.log | cut -c1-180
  # 磁盘纪律：只留 log.txt，删 checkpoint
  D=$(ls -dt output/graphlora_ml1m_scratch_$RUN/*/ 2>/dev/null | head -1)
  [ -n "$D" ] && find "$D" -name "*.pth" -delete && echo "checkpoints cleaned for $RUN"
done
echo QUEUE_DONE
