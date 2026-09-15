#!/bin/bash
# 等 GPU 空闲 -> smoke test -> 通过则启动 v3 训练 + GPU 监控
cd /home/bbx/user_lx/GraphLoRA
for i in $(seq 1 40); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$USED" -lt 2000 ] && break
  sleep 60
done
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -ge 2000 ]; then echo "GPU busy after 40min, abort"; exit 1; fi
echo "GPU free, running smoke test..."
CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/minigpt4/bin/python smoke_test_mutual.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3_rank.yaml > smoke_test_mutual_rank.log 2>&1
if ! grep -q "ALL SMOKE TESTS PASSED" smoke_test_mutual_rank.log; then
  echo "SMOKE TEST FAILED, not launching training"; exit 2
fi
echo "smoke test passed, launching v3 rank training"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ~/anaconda3/envs/minigpt4/bin/python train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3_rank.yaml > graphlora_b3_rank_train.log 2>&1 &
echo "train PID: $!"
nohup bash gpu_monitor.sh b3_v3_rank >/dev/null 2>&1 &
echo "monitor started"
