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
run_one () {  # $1=cfg-suffix $2=epochs $3=logname
  DONE=0
  for ATTEMPT in 1 2 3; do
    wait_gpu || { echo "attempt $ATTEMPT: GPU busy"; continue; }
    echo "=== $3 attempt $ATTEMPT start $(date '+%H:%M') ==="
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_$1 \
        --options run.max_epoch=$2 run.batch_size_train=1 run.batch_size_eval=2 \
                  run.save_checkpoint_every_n_epochs=0 \
      > $3.log 2>&1
    NAUC=$(grep -c "rank_0 auc" $3.log)
    echo "attempt $ATTEMPT: $NAUC/$2 evals"
    [ "$NAUC" -ge "$2" ] && { DONE=1; break; }
    sleep 120
  done
  [ "$DONE" != "1" ] && { echo "$3 failed, skip"; return 1; }
  grep -oE "rank_0 auc: 0\.[0-9]+" $3.log
  BEST=$(ls -t output/graphlora_ml1m_scratch_rel_pref*/2*/checkpoint_best.pth output/graphlora_ml1m_scratch_rel_pref*/*/checkpoint_best.pth 2>/dev/null | head -1)
  echo "$3 best: $BEST"
  echo "$BEST" > /tmp/last_best_$3.txt
  return 0
}
# (a) 12-epoch
run_one scratch_rel_pref_12ep 12 rel_pref_12ep
BEST_A=$(cat /tmp/last_best_rel_pref_12ep.txt 2>/dev/null)
# (b) K=3
run_one scratch_rel_pref_k3 8 rel_pref_k3
BEST_B=$(cat /tmp/last_best_rel_pref_k3.txt 2>/dev/null)
# test eval 两个 best（β=0）
for PAIR in "12ep:$BEST_A" "k3:$BEST_B"; do
  TAG=${PAIR%%:*}; BEST=${PAIR#*:}
  [ -z "$BEST" ] && continue
  wait_gpu || continue
  echo "=== test eval $TAG ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel_$TAG.yaml \
      --options model.ckpt=$BEST run.evaluate=True run.batch_size_eval=1 \
                run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_$TAG \
    > eval_$TAG.log 2>&1
  echo "$TAG test:"; grep -oE "rank_0 auc: 0\.[0-9]+" eval_$TAG.log || tail -2 eval_$TAG.log | cut -c1-150
done
find output -name "*.pth" -delete 2>/dev/null
echo FINAL_PUSH_DONE
