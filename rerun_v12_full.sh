#!/bin/bash
# v12 完整重训（8 epoch），OOM 自动重试，完成后 test eval + 融合 eval
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
BEST=""
for ATTEMPT in 1 2 3; do
  wait_gpu || { echo "attempt $ATTEMPT: GPU busy"; continue; }
  echo "=== v12 full train attempt $ATTEMPT start $(date '+%H:%M') ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_pref.yaml \
      --options run.max_epoch=8 run.batch_size_train=1 run.batch_size_eval=2 \
                run.save_checkpoint_every_n_epochs=0 \
    > rel_pref.log 2>&1
  NAUC=$(grep -c "rank_0 auc" rel_pref.log)
  echo "attempt $ATTEMPT: $NAUC/8 evals completed"
  if [ "$NAUC" -ge 8 ]; then
    BEST=$(ls -t output/graphlora_ml1m_scratch_rel_pref/*/checkpoint_best.pth 2>/dev/null | head -1)
    break
  fi
  sleep 120
done
if [ -z "$BEST" ]; then
  BEST=$(ls -t output/graphlora_ml1m_scratch_rel_pref/*/checkpoint_best.pth 2>/dev/null | head -1)
fi
echo "best: $BEST"
[ -z "$BEST" ] && { echo "no ckpt"; exit 1; }
grep -oE "rank_0 auc: 0\.[0-9]+" rel_pref.log
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
echo V12_FULL_DONE
