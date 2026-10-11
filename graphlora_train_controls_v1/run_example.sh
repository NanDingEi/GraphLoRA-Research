#!/usr/bin/env bash
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
unset GLORA_GRAD_PROBE
mkdir -p logs
python -u train_collm_mf_din.py \
  --cfg-path output/overnight_auc_sweep_3/configs/S2_sem_proj_0005.yaml \
  --options \
  run.init_lr=1e-4 run.min_lr=5e-5 run.warmup_lr=1e-5 \
  run.accum_grad_iters=3 \
  run.lr_scale_rec=0.5 run.lr_scale_gnn=0.25 \
  run.lr_scale_proj=0.5 run.lr_scale_lora=1.0 \
  model.gnn_semantic_weight=1e-4 model.proj_reg_weight=5e-4 \
  model.ckpt=null run.resume_ckpt_path=null \
  run.output_dir=/home/bbx/user_lx/GraphLoRA/output/S2_groups_acc3 \
  2>&1 | tee logs/S2_groups_acc3.log
