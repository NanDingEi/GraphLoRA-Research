# CoRA 单模型联合训练：RR1 L32 + RR5 User-Gated L31

分支：exp/cora-joint-l32-gate-20261113
脚本：experiments/no31_cora_original/cora_joint_l32_gate.py

## 设计
只加载一个 Vicuna-7B。单个 CoRA 模型同时计算 L32 和 L31 用户门控残差。不加载 RR5 checkpoint；无需双教师、离线预测、KD 蒸馏。
所有方案都从相同 RR1 epoch37 仅模型权重 checkpoint 续训，使用全新 AdamW（原 checkpoint 不含 optimizer）；固定 seed42。
D4_TOPK_RES025 + C6_KD_BOTH 保持不变（CTR=.005、原 MF KD=.01、分层 TopK40/20/10/5、25% MF residual）。

J0_RR1_CONTINUE: original L32 BCE + C6。
J1_DUAL_BCE: original L32 BCE + C6 + 0.25*fused BCE + RR5 gate prior。
J2_DUAL_USER_RANK: J1 + 0.02*same-user positive-negative ranking loss。

z_fused = z32 + sigmoid(g(u))*(z31-z32)。
注意原始 C6 loss 已包含 L32 BCE，不能重复累加。
三组使用相同 deterministic sampler：每四个 micro-batch 有一个同用户正负样本对。
GPU0，serial foreground，batch2/accum4/400 microsteps每轮；CoRA LR3e-5，MF LR3e-6。
每轮完整 VALID，记录 L32 和 fused AUC/UAUC，保存所有轮次 compact checkpoint；不运行 TEST。

## 更新与执行

在 /home/bbx/user_lx/CoRA_official 原目录：

    cd /home/bbx/user_lx/CoRA_official
    BRANCH=exp/cora-joint-l32-gate-20261113
    DIR=experiments/no31_cora_original
    git fetch git@github.com:NanDingEi/GraphLoRA-Research.git "refs/heads/$BRANCH"
    git show "FETCH_HEAD:$DIR/cora_joint_l32_gate.py" > "$DIR/cora_joint_l32_gate.py.tmp"
    mv -f "$DIR/cora_joint_l32_gate.py.tmp" "$DIR/cora_joint_l32_gate.py"
    python -m py_compile "$DIR/cora_joint_l32_gate.py"
    python "$DIR/cora_joint_l32_gate.py" --cpu-test
    python "$DIR/cora_joint_l32_gate.py" --plan

串行训练：

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_joint_l32_gate.py --run --only J0_RR1_CONTINUE,J1_DUAL_BCE,J2_DUAL_USER_RANK --epochs 6 --patience 6

自动输出：
- experiments/no31_cora_original/log/CORA_JOINT_L32_GATE_20261113/J*.log
- experiments/no31_cora_original/out/CORA_JOINT_L32_GATE_20261113/J*/valid_every_epoch.csv
- experiments/no31_cora_original/output/CORA_JOINT_L32_GATE_20261113/J*/<timestamp>/checkpoint_*.pth

原 checkpoint 不覆盖。任一组失败会阻止后续组启动。
只有 Valid 选定的 checkpoint 才应评估 Test；已有 Test 不用于调超参数。