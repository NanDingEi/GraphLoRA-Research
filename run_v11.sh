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
# smoke 回归（重构涉及 b3 路径）
wait_gpu || { echo "smoke: GPU busy"; exit 1; }
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY smoke_test_schemes.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3_smoke.yaml \
  > smoke_test_schemes.log 2>&1
if ! grep -q "ALL SMOKE TESTS PASSED" smoke_test_schemes.log; then
  echo SMOKE_FAILED; tail -5 smoke_test_schemes.log | cut -c1-200; exit 1
fi
grep -E "^\[[0-9]/6\]" smoke_test_schemes.log
# v11 训练
wait_gpu || { echo "v11: GPU busy"; exit 1; }
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
