# CoRA RR1 Warm-start T0–T3：训练优化（2026-11-13）

**源代码：** `NanDingEi/GraphLoRA-Research` 实验分支 `exp/cora-train4-20261113`。
**服务器目录：** `/home/bbx/user_lx/CoRA_official`。

目标：以真实 RR1 Test AUC=0.739654、UAUC=0.706352 和预先复现的 RR1/RR5 等权集成 AUC=0.741707、UAUC=0.707231 为基准，研究单模型可否通过用户排序损失和集成教师蒸馏改善 Valid AUC/UAUC。**没有声称训练指标已提升**。历史 Test 仅作最终独立比较，实验中不用于调参。

## 已实现的四个实验

| ID | 起点 | 同用户 pair sampler | BCE+C6 | User-rank | Ensemble KD | MF anchor |
|---|---|---|---|---|---|---|
| T0_CONTINUE | RR1 best | 是 | 是 | 0 | 0 | 0 |
| T1_USER_RANK | RR1 best | 是 | 是 | 0.02 | 0 | 0 |
| T2_ENSEMBLE_KD | RR1 best | 是 | 是 | 0 | 0.05 | 0 |
| T3_JOINT | RR1 best | 是 | 是 | 0.02 | 0.05 | 5e-4 |

- **真实梯度训练**：每一个 T 都从 RR1 best 的 compact *model-only* checkpoint 重新加载训练参数，再新建 AdamW；不是恢复优化器状态。四组对照完全相同的用户均匀正负成对采样，每个 batch=2（1正1负，同UserID），accum=4，400 microsteps/epoch。原始 C6 CTR=0.005、C6 gated-KD=0.010 均保留。
- User-rank `softplus(-(s_pos-s_neg))`：真正是同用户两个当前 forward 分数共同参与梯度，不是历史 memory bank 或仅 batch 混合。
- Ensemble KD：分别从 RR1 和 RR5 checkpoint **离线生成 TRAIN logits**，归一化并等权融合（仅 Train 统计），Train 上拟合一维 Logistic 校准成教师 logit，T=2；梯度只进入 RR1 Student，teacher 永远不占训练显存。
- MF anchor：只对**当前 batch 访问的用户/物品 MF 行**施加对 RR1 checkpoint 初始化的平方偏移约束，系数 5e-4；**不是**全参数 LoRA anchor。
- D4 多粒度输入（TopK40/20/10/5，个人残差=0.25）、CoRA动态LoRA、C6 remain unchanged. 初始LR=3e-5，MF分组0.1x，seed42，单卡GPU0，前台串行，运行时 GPU0 free < 22000 MiB 则拒绝装载模型。
- 每轮完整 Valid；打印/保存 Valid AUC + UAUC (`valid_auc_uauc.csv`)；保存每轮 compact checkpoint (`checkpoint_0.pth`...以及 `checkpoint_best.pth`)；**不会运行 Test**。

## 安装同步（只更新新增文件）

```bash
cd /home/bbx/user_lx/CoRA_official
bash -e <<'SH'
BRANCH=exp/cora-train4-20261113
DIR=experiments/no31_cora_original
git fetch git@github.com:NanDingEi/GraphLoRA-Research.git "refs/heads/$BRANCH"
for f in cora_train4_20261113.py CORA_TRAIN4_README.md; do
  git show "FETCH_HEAD:$DIR/$f" > "$DIR/$f.tmp"
  mv -f "$DIR/$f.tmp" "$DIR/$f"
done
SH
```

## CPU 自测 + 预生成 T0–T3 配置

```bash
cd /home/bbx/user_lx/CoRA_official
PYTHON=python
SCRIPT=experiments/no31_cora_original/cora_train4_20261113.py
$PYTHON -m py_compile "$SCRIPT"
$PYTHON "$SCRIPT" --cpu-test
$PYTHON "$SCRIPT" --plan
```

### 第一步：训练 T0 和 T1（不需要教师预测）

```bash
cd /home/bbx/user_lx/CoRA_official
export CUDA_VISIBLE_DEVICES=0
python -u experiments/no31_cora_original/cora_train4_20261113.py \
  --run --only T0_CONTINUE,T1_USER_RANK
```

运行输出中须确认 `[WARM START]` RR1 checkpoint 加载；`[TRAIN4 PAIR LOADER]` 有同用户配对采样；`[TRAIN4 VALID]` 每轮输出 AUC+UAUC；日志 `experiments/no31_cora_original/logs/CORA_TRAIN4_RR1_WARMSTART_20261113/<trial>.log`，结果目录 `experiments/no31_cora_original/output/CORA_TRAIN4_RR1_WARMSTART_20261113/<trial>/<job_id>/`。

### 第二步：串行捕获 RR1/RR5 Train 教师（只训练集、纯推理）

```bash
cd /home/bbx/user_lx/CoRA_official
export CUDA_VISIBLE_DEVICES=0
python -u experiments/no31_cora_original/cora_train4_20261113.py --capture RR1
python -u experiments/no31_cora_original/cora_train4_20261113.py --capture RR5
python experiments/no31_cora_original/cora_train4_20261113.py --prepare-teacher
```

新 NPZ 将写到 `experiments/no31_cora_original/out/CORA_TRAIN_TEACHERS_20261113/`；训练时验证原始 `dataset.annotation` 的逐行 `(UserID,TargetItemID,label,index)` 完全一致（不是仅仅用 triple 字典去查找，重复 triples 不会污染行身份）。如果 checkpoint、样本数、顺序不匹配则立刻停止。不会读 Valid/Test 来拟合教师。

### 第三步：训练 T2 和 T3

```bash
cd /home/bbx/user_lx/CoRA_official
export CUDA_VISIBLE_DEVICES=0
python -u experiments/no31_cora_original/cora_train4_20261113.py \
  --run --only T2_ENSEMBLE_KD,T3_JOINT
```

训练日志和配置按上述路径分别隔离；有同名 log 则自动拒绝覆盖，以免覆盖实验记录。测试前根据每轮 Valid AUC、UAUC 选择 Pareto checkpoint，不允许根据已知 Test 结果调整超参数。

## 关键要求

- 使用服务器现有 `minigpt4` 环境，不升级 PyTorch/CUDA/transformers、不修改官方 CoRA 主体文件。
- 必需 RR1：`experiments/no31_cora_original/output/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE/20261103092/checkpoint_best.pth`。
- 必需 RR5：`experiments/no31_cora_original/output/CORA_RESIDUAL_ROUTING6_20261103/RR5_USER_GATE_R31/20261103153/checkpoint_best.pth`。
- 依赖已有已通过测试的 `cora_residual_routing6.py`、`cora_latevote4.py`、`cora_combo8.py`、`cora_joint8.py`、`cora_struct10.py`、`cora_rankgran7.py`。
- 训练数据来自原 `train_collm_mf_din.py` 对应 train split。此 runner 不使用 Test split，Valid 仅用于追踪 AUC/UAUC/选择模型。
- 若 GPU 显存低于22GiB，因原 R6 保护设置拒绝开始。要先腾出 GPU0 显存，不允许随意修改这层保护。
- 若需比较训练步数公平，确保 T0–T3 都完成相同 epoch/microstep；相比 RR1 原训练，这些都是 warm-start 额外续训实验。
- 当前仅进行了本地 CPU 结构自测；尚未在服务器执行真实 7B checkpoint 训练，首轮应先核对 `[WARM START]` 和 `[TRAIN4 VALID]` 是否出现。
