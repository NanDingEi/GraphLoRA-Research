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
wait_gpu || { echo "smoke: GPU busy"; exit 1; }
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY smoke_test_schemes.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3_smoke.yaml \
  > smoke_test_schemes.log 2>&1
if ! grep -q "ALL SMOKE TESTS PASSED" smoke_test_schemes.log; then
  echo SMOKE_FAILED; tail -5 smoke_test_schemes.log | cut -c1-200; exit 1
fi
grep -cE "^\[[0-9]/6\]" smoke_test_schemes.log | xargs echo "smoke checks passed:"
wait_gpu || exit 1
echo "=== v12 rel_pref train start $(date '+%H:%M') ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_pref.yaml \
    --options run.max_epoch=8 run.batch_size_train=1 run.batch_size_eval=2 \
              run.save_checkpoint_every_n_epochs=0 \
  > rel_pref.log 2>&1
echo "--- v12 valid AUCs ---"
grep -oE "rank_0 auc: 0\.[0-9]+" rel_pref.log || tail -3 rel_pref.log | cut -c1-180
BEST=$(ls -t output/graphlora_ml1m_scratch_rel_pref/*/checkpoint_best.pth 2>/dev/null | head -1)
echo "best: $BEST"
[ -z "$BEST" ] && { echo "no ckpt"; exit 1; }
# test eval beta=0
for B in 0.0 0.25 0.5; do
  wait_gpu || break
  echo "=== v12 test eval beta=$B ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_pref.yaml \
      --options model.ckpt=$BEST run.evaluate=True run.batch_size_eval=1 \
                model.eval_fuse_beta=$B \
                run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_rel_pref_b$B \
    > eval_rel_pref_b$B.log 2>&1
  echo "beta=$B:"; grep -oE "rank_0 auc: 0\.[0-9]+" eval_rel_pref_b$B.log || tail -2 eval_rel_pref_b$B.log | cut -c1-150
done
find output/graphlora_ml1m_scratch_rel_pref output/eval_rel_pref_b* -name "*.pth" -delete 2>/dev/null
echo V12_DONE
