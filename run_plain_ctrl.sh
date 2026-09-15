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
DONE=0
for ATTEMPT in 1 2; do
  wait_gpu || { echo "attempt $ATTEMPT: GPU busy"; continue; }
  echo "=== fix_plain attempt $ATTEMPT start $(date '+%H:%M') ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_idlist.yaml \
      --options run.max_epoch=8 run.batch_size_train=1 run.batch_size_eval=2 \
                run.save_checkpoint_every_n_epochs=0 \
                model.lambda_llm_to_gnn=0.0 model.lambda_gnn_to_llm=0.0 \
    > fix_plain.log 2>&1
  NAUC=$(grep -c "rank_0 auc" fix_plain.log)
  echo "attempt $ATTEMPT: $NAUC/8 evals"
  [ "$NAUC" -ge 8 ] && { DONE=1; break; }
  sleep 120
done
[ "$DONE" != "1" ] && { echo "plain failed"; exit 1; }
grep -oE "rank_0 auc: 0\.[0-9]+" fix_plain.log
PLAIN_TEST=$(ls -t output/graphlora_ml1m_scratch_idlist/*/checkpoint_best.pth 2>/dev/null | head -1)
echo "plain best ckpt: $PLAIN_TEST"
wait_gpu || exit 0
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_scratch_idlist.yaml \
    --options model.ckpt=$PLAIN_TEST run.evaluate=True run.batch_size_eval=1 \
              run.output_dir=/home/bbx/user_lx/GraphLoRA/output/eval_fix_plain \
    > eval_fix_plain.log 2>&1
echo "plain test:"; grep -oE "rank_0 auc: 0\.[0-9]+" eval_fix_plain.log
find output -name "*.pth" -delete 2>/dev/null
echo PLAIN_DONE
