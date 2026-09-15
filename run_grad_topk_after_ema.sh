#!/bin/bash
# 等 EMA 完全结束后接力 grad/topk 两个 run
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
while ! grep -q "EMA_RETRY_DONE" /home/bbx/.zcode/cli/exec/sess_f88f002e-2ef7-48de-be59-a499b6e484a3/call_684795d97364417ca98bc917-stdout.log 2>/dev/null; do
  sleep 120
done
wait_gpu () {
  for i in $(seq 1 120); do
    FREE=$((24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)))
    [ "$FREE" -ge 19500 ] && return 0
    sleep 60
  done
  return 1
}
for RUN in grad topk; do
  DONE=0
  for ATTEMPT in 1 2; do
    wait_gpu || { echo "run $RUN attempt $ATTEMPT: GPU busy"; continue; }
    echo "=== rel_$RUN attempt $ATTEMPT start $(date '+%H:%M') ==="
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_$RUN.yaml \
        --options run.max_epoch=8 run.batch_size_train=1 run.batch_size_eval=2 \
                  run.save_checkpoint_every_n_epochs=0 \
      > rel_$RUN.log 2>&1
    NAUC=$(grep -c "rank_0 auc" rel_$RUN.log)
    echo "attempt $ATTEMPT: $NAUC/8 evals"
    [ "$NAUC" -ge 8 ] && { DONE=1; break; }
    sleep 120
  done
  [ "$DONE" != "1" ] && { echo "run $RUN failed, skip"; continue; }
  echo "--- rel_$RUN valid AUCs ---"
  grep -oE "rank_0 auc: 0\.[0-9]+" rel_$RUN.log
  BEST=$(ls -t output/graphlora_ml1m_scratch_rel_$RUN/*/checkpoint_best.pth 2>/dev/null | head -1)
  [ -z "$BEST" ] && continue
  wait_gpu || continue
  echo "=== rel_$RUN test eval ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_$RUN.yaml \
      --options model.ckpt=$BEST run.evaluate=True run.batch_size_eval=1 \
                run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_rel_$RUN \
    > eval_rel_$RUN.log 2>&1
  echo "rel_$RUN test:"; grep -oE "rank_0 auc: 0\.[0-9]+" eval_rel_$RUN.log
  find output/graphlora_ml1m_scratch_rel_$RUN output/eval_rel_$RUN -name "*.pth" -delete 2>/dev/null
done
echo GRAD_TOPK_DONE
