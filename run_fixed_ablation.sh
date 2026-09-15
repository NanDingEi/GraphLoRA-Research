#!/bin/bash
# 修复后的真实消融：全部 8 epoch、lr 1e-4、IDList、batch 1
# (a) tca_full: REL + pref_tca 双向互优化（互学习真正进 backward）
# (b) rel_pref: v12 配置重跑（REL 进 backward 的版本）
# (c) plain: 纯 L_rec 对照（mutual off）
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
run_one () {  # $1=cfg $2=epochs $3=log $4=extra-options
  DONE=0
  for ATTEMPT in 1 2; do
    wait_gpu || { echo "attempt $ATTEMPT: GPU busy"; continue; }
    echo "=== $3 attempt $ATTEMPT start $(date '+%H:%M') ==="
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY train_collm_mf_din.py --cfg-path tran_configs/$1 \
        --options run.max_epoch=$2 run.batch_size_train=1 run.batch_size_eval=2 \
                  run.save_checkpoint_every_n_epochs=0 $4 \
      > $3.log 2>&1
    NAUC=$(grep -c "rank_0 auc" $3.log)
    echo "attempt $ATTEMPT: $NAUC/$2 evals"
    [ "$NAUC" -ge "$2" ] && { DONE=1; break; }
    sleep 120
  done
  [ "$DONE" != "1" ] && { echo "$3 failed, skip"; return 1; }
  grep -oE "rank_0 auc: 0\.[0-9]+" $3.log
  BEST=$(ls -t output/graphlora_ml1m_scratch_*/*/checkpoint_best.pth 2>/dev/null | head -1)
  echo "$3 best: $BEST"; echo "$BEST" > /tmp/fix_best_$3.txt
  wait_gpu || return 0
  echo "=== $3 test eval ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/$1 \
      --options model.ckpt=$BEST run.evaluate=True run.batch_size_eval=1 \
                run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_fix_$3 \
    > eval_fix_$3.log 2>&1
  echo "$3 test:"; grep -oE "rank_0 auc: 0\.[0-9]+" eval_fix_$3.log || tail -2 eval_fix_$3.log | cut -c1-150
  find output -name "*.pth" -delete 2>/dev/null
  return 0
}
run_one collm_pretrain_mf_ood_scratch_tca_full.yaml 8 fix_tca_full ""
run_one collm_pretrain_mf_ood_scratch_rel_pref.yaml 8 fix_rel_pref ""
run_one collm_pretrain_mf_ood_scratch_idlist.yaml 8 fix_plain "model.mutual_mode=off model.lambda_llm_to_gnn=0.0 model.lambda_gnn_to_llm=0.0"
echo FIXED_ABLATION_DONE
