#!/bin/bash
cd /home/bbx/user_lx/GraphLoRA
PY=~/anaconda3/envs/minigpt4/bin/python
for PAIR in "v4:output/graphlora_ml1m_b3_rank_lowlr/20260906110/checkpoint_best.pth" "ctrl:output/graphlora_ml1m_ctrl_lowlr/20260906114/checkpoint_best.pth"; do
  TAG=${PAIR%%:*}; CKPT=${PAIR#*:}
  echo "=== eval $TAG ($CKPT) ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_collm_mf_din.py     --cfg-path tran_configs/collm_pretrain_mf_ood_b3_rank_eval.yaml     --options model.ckpt=$CKPT run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_${TAG}_best     > eval_${TAG}_best.log 2>&1
  grep -oE "rank_0 auc: 0\.[0-9]+" eval_${TAG}_best.log | tail -3
done
echo ALL_EVALS_DONE
