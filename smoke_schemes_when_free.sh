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
  [ "$FREE" -lt 18000 ] && { echo "attempt $ATTEMPT: GPU busy"; continue; }
  echo "=== smoke attempt $ATTEMPT ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY smoke_test_schemes.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3_smoke.yaml \
    > smoke_test_schemes.log 2>&1
  if grep -q "ALL SMOKE TESTS PASSED" smoke_test_schemes.log; then
    grep -E "^\[[0-9]/6\]" smoke_test_schemes.log
    echo SMOKE_DONE; exit 0
  fi
  echo "attempt $ATTEMPT failed:"; tail -5 smoke_test_schemes.log | cut -c1-200; sleep 60
done
echo SMOKE_FAILED
