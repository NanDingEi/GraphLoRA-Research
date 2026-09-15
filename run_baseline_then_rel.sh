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

# ---- 阶段 1：重训基线 GraphLoRA（重建热启动权重）----
wait_gpu || { echo "baseline: GPU busy"; exit 1; }
echo "=== baseline retrain start $(date '+%H:%M') ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood.yaml \
    --options run.max_epoch=12 run.batch_size_train=1 run.batch_size_eval=2 \
              run.save_checkpoint_every_n_epochs=0 \
              run.output_dir=/home/bbx/user_lx/GraphLoRA/output/graphlora_ml1m_retrain \
  > baseline_retrain.log 2>&1
echo "--- baseline valid AUCs ---"
grep -oE "rank_0 auc: 0\.[0-9]+" baseline_retrain.log
BASE_BEST=$(ls -t output/graphlora_ml1m_retrain/*/checkpoint_best.pth 2>/dev/null | head -1)
echo "baseline best: $BASE_BEST"
[ -z "$BASE_BEST" ] && { echo "no baseline ckpt, abort"; exit 1; }
find output/graphlora_ml1m_retrain -name "checkpoint_[0-9]*.pth" -delete

# ---- 阶段 2：热启动 REL（8 epoch 到收敛）+ test 评估 ----
wait_gpu || { echo "rel warm: GPU busy"; exit 1; }
echo "=== rel warm-start start $(date '+%H:%M') ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel.yaml \
    --options model.ckpt=$BASE_BEST run.init_lr=2e-5 run.min_lr=1e-5 \
              run.max_epoch=8 run.batch_size_train=1 run.batch_size_eval=2 \
              run.save_checkpoint_every_n_epochs=0 \
              run.output_dir=/home/bbx/user_lx/GraphLoRA/output/graphlora_rel_warm \
  > rel_warm.log 2>&1
echo "--- rel_warm valid AUCs ---"
grep -oE "rank_0 auc: 0\.[0-9]+" rel_warm.log
REL_BEST=$(ls -t output/graphlora_rel_warm/*/checkpoint_best.pth 2>/dev/null | head -1)
if [ -n "$REL_BEST" ]; then
  wait_gpu || { echo "eval: GPU busy"; exit 1; }
  echo "--- rel_warm test eval ---"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_rel.yaml \
      --options model.ckpt=$REL_BEST run.evaluate=True run.batch_size_eval=1 \
                run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_rel_warm \
    > eval_rel_warm.log 2>&1
  grep -oE "rank_0 auc: 0\.[0-9]+" eval_rel_warm.log || tail -2 eval_rel_warm.log | cut -c1-150
fi
# 磁盘纪律：全部权重即用即删
find output/graphlora_ml1m_retrain output/graphlora_rel_warm output/eval_rel_warm -name "*.pth" -delete 2>/dev/null
echo PIPELINE_DONE
