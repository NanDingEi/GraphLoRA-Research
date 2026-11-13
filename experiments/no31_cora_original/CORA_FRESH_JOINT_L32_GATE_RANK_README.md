# CoRA 单模型联合训练（全新初始化，非续训）

本分支只新增 `cora_fresh_joint_l32_gate_rank.py` 训练入口。**不读取 RR1、RR5 checkpoint，不加载双教师，不使用此前测试集预测进行调参。**

## 初始条件

- Vicuna-7B：使用 CoRA 原配置指定的预训练基础模型（冻结基座），**不是随机初始化整个 7B**。
- MF：使用原始 MF 预训练文件初始化，训练中 MF 解冻。
- CoRA：fresh dynamic Q/K/V/O LoRA，rank=16 / alpha=32；J1/J2 门控从头初始化。
- D4：TopK=40/20/10/5，MF residual personal share=0.25；保留原模型内部混合输入。
- C6：CTR=.005、agreement+confidence MF KD=.01，与原训练相同。
- 3组独立子进程：J0、J1、J2；物理 GPU0，seed=42。
- CoRA 学习率=3e-4，MF 学习率倍率=.1；microbatch=2、accum=4、400 microsteps/epoch。
- 每轮记录 VALID AUC/UAUC 并保存 epoch checkpoint；仅根据 Valid 决定训练方案，不自动评估 Test。

## 实验定义

| ID | 训练总损失 | Valid 输出 |
|---|---|---|
| J0_FRESH_L32 | 原始 L32 BCE + C6 CTR/KD | L32 Yes logit |
| J1_FRESH_JOINT | J0 + .25 BCE(fused) + 1e-3(mean_gate-prior)^2 | fused |
| J2_FRESH_JOINT_RANK | J1 + .02 same-user pos-neg softplus ranking | fused |

`fused=z32+sigmoid(g(u))*(z31-z32)`，门控从新随机初始化（末层权重零初始化，bias=-1.5），不存在对旧 RR5 权重的加载。J0/J1/J2 使用完全一致的 TRAIN 混合采样：每 4 个 microbatch 中 1 个来自同用户的正负样本，其余随机。这样 J2 的收益不能简单归因于不同采样；但 J0 与**历史** RR1 的采样不同，报告时须分开说明。

## 同步到服务器现有目录（无需 clone、无需 Fork）

```bash
cd /home/bbx/user_lx/CoRA_official
set -e
BRANCH=exp/cora-fresh-joint-l32-gate-rank-20261113
DIR=experiments/no31_cora_original
git fetch git@github.com:NanDingEi/GraphLoRA-Research.git "refs/heads/$BRANCH"
git show "FETCH_HEAD:$DIR/cora_fresh_joint_l32_gate_rank.py" > "$DIR/cora_fresh_joint_l32_gate_rank.py.tmp"
mv -f "$DIR/cora_fresh_joint_l32_gate_rank.py.tmp" "$DIR/cora_fresh_joint_l32_gate_rank.py"
python -m py_compile "$DIR/cora_fresh_joint_l32_gate_rank.py"
python "$DIR/cora_fresh_joint_l32_gate_rank.py" --cpu-test
python "$DIR/cora_fresh_joint_l32_gate_rank.py" --plan
```

## 正式 GPU0 从头训练（三组串行前台）

```bash
cd /home/bbx/user_lx/CoRA_official
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_fresh_joint_l32_gate_rank.py \
  --run --only J0_FRESH_L32,J1_FRESH_JOINT,J2_FRESH_JOINT_RANK \
  --screen-epochs 0 --patience 10
```

`--screen-epochs 0` 表示不人为限制最大训练轮次，使用原 CoRA 1000 epoch scheduler，按 Valid AUC patience=10 早停；不等于只训练 0 轮。每轮在 `experiments/no31_cora_original/out/CORA_FRESH_JOINT_L32_GATE_RANK_20261113/<trial>/valid_every_epoch.csv` 保存 AUC、UAUC，并在 `experiments/no31_cora_original/output/CORA_FRESH_JOINT_L32_GATE_RANK_20261113/<trial>/...` 保存 epoch/best checkpoints。前台完整日志保存在 `experiments/no31_cora_original/log/CORA_FRESH_JOINT_L32_GATE_RANK_20261113/<trial>.log`。有既存日志或结果会拒绝覆盖，避免毁坏训练结果。

**服务器 GPU 真正训练尚未运行**；先做 CPU 检查和 `--plan`，之后执行 `--run`。如果运行时的本地 CoRA 修改与 GitHub 文件的 API 不同，立即检查首个错误日志，不要修改环境版本。

## 常见认知误区

- 这不是 RR1 epoch37 的续训，不是 RR5 epoch21 的复训，也不是双模型 ensemble/KD。
- 使用 Vicuna-7B 预训练底座与 MF 预训练初始化，符合从初始 CoRA 训练；“从头”不等于让 7B 在随机参数上预训练。
- 训练内 J1/J2 额外分支用 label BCE，原 C6 的 CTR/KD 均保留，不覆盖 L32 主损失。
- Test 历史 AUC 0.741707/UAUC 0.707344 仅作为最后基准，不能反复据此挑选 checkpoint。
