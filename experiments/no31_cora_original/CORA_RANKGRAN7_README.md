# CoRA/MF 第四轮：C8 补跑 + Rank 与多粒度协同信息（7 组）

## 实验目的与统一控制

本轮先在一个新的隔离协议目录中重新完整运行未完成的 C8_MAIN_HALF，再运行 6 组结构实验。旧的 C8 日志和 checkpoint 不会覆盖；新批次统一保存在 `CORA_RANKGRAN7_20261101`。

- C8：沿用上一轮 CTR=0.005、KD=0.01、LLM 主任务传回 MF 的梯度增益 0.5。
- R1–G4：统一采用当前 AUC 最优的 C6 损失组合：CTR=0.005、KD=0.01；KD 只在 LLM 预测与标签一致且 `abs(p-0.5)>=0.20` 时启用。
- 原始 CoRA Yes-logit BCE 保留；Vicuna 主干冻结，CoRA Q/K/V/O 动态参数生成器和 MF embedding 可训练。
- seed=42，CoRA LR=3e-4，MF LR=3e-5，batch_size=2，gradient accumulation=4，400 microbatches/epoch。
- Valid only，不运行 Test；Valid AUC 连续 10 轮不提升即早停。
- GPU0 前台串行；只保存可训练参数，不保存冻结 Vicuna、optimizer 或 GradScaler。
- 不改 Prompt，不修改官方 CoRA 源文件；实验机制通过独立 runner 的运行时 patch 实现。

## 七组实验

| 顺序 | ID | 核心设置 |
|---:|---|---|
| 1 | C8_MAIN_HALF | 补跑上轮未完成实验；原 C8 配置不变 |
| 2 | P1_RANK_46810 | 32 层按 8 层分组，Rank=[4,6,8,10] |
| 3 | P2_RANK_8121620 | 32 层按 8 层分组，Rank=[8,12,16,20] |
| 4 | G1_CLUSTER_K32 | 所有层均使用 K=32 聚类中心 |
| 5 | G2_COARSE_FINE | 0–7 层 K32；8–15 层 K128；16–23 层 TopK20；24–31 层原始 MF |
| 6 | G3_CLUSTER_RESIDUAL | K32 聚类中心与个体残差按 0、1/3、2/3、1 逐层融合 |
| 7 | G4_MULTI_TOPK | 0–7 层 TopK5；8–15 层 TopK10；16–23 层 TopK20；24–31 层 TopK40 |

## Rank 实现细节

CoRA 原始生成器输出一个共享的动态矩阵 `W`，其末维为 Rank；每个 Transformer 层的 Q/K/V/O PLora 投影层使用该矩阵。为了实现逐层不同 Rank，本轮将生成器输出末维改为该 profile 的最大 Rank，并让每层取前 `r_l` 维，同时将每层独立的 `lora_P` 投影输入维度改为 `r_l`。

Alpha 按 `alpha/r=2` 同步设置，因此 R1 的 Alpha=[8,12,16,20]，R2 的 Alpha=[16,24,32,40]。所有层仍使用同一个 CoRA query generator，避免额外增加 32 套生成器参数。每层的 P 投影仍独立训练并零初始化。

截取较短 Rank 前缀后会重新归一化为单位 Frobenius 范数，避免低秩层仅因切片而自动获得更小的动态权重幅值。Alpha/Rank 也保持为 2。\n\n这是一种嵌套前缀秩分配，不是每层独立生成不同的协同权重矩阵。这样能控制变量，只比较层间容量分配。

## 多粒度信息构建

- K-Means 使用预训练 MF 表示（仅由训练数据得到）在 CPU 上建立固定用户/物品聚类分配；K=32 和 K=128 分别建立。
- 聚类中心每个 forward 都由当前可训练 MF embedding 表计算，因此中心仍有梯度，不是冻结的初始中心。
- 邻居 ID 根据预训练 MF embedding 的余弦相似度预计算并固定；每个 forward 使用当前可训练 MF embedding 重新计算邻居相似度和加权聚合，因此邻居向量仍可接收梯度。
- G2 的 TopK20 与 G4 的 TopK5/10/20/40 均排除 padding ID=0 和节点自身。
- G2–G4 将 4 组表示按批次拼接，一次批量调用共享 CoRA generator，再按 8 层一组路由对应动态权重；G1 只调用一次 generator。

## 执行指令

在服务器的 `minigpt4` 环境中执行：

```bash
cd /home/bbx/user_lx/CoRA_official
conda activate minigpt4

BRANCH=exp/night26-mf-graph-lambda-adaptive-20261028
BASE=https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/$BRANCH/experiments/no31_cora_original

# 更新实验入口、C8 协议支持和全局汇总脚本
curl -fL --retry 3 "$BASE/cora_joint8.py" -o cora_joint8.py
curl -fL --retry 3 "$BASE/cora_combo8.py" -o cora_combo8.py
curl -fL --retry 3 "$BASE/cora_all_summary.py" -o cora_all_summary.py
curl -fL --retry 3 "$BASE/cora_rankgran7.py" -o cora_rankgran7.py

# 语法检查
python -m py_compile cora_joint8.py cora_combo8.py cora_all_summary.py cora_rankgran7.py

# CPU 单元测试：聚类、聚类中心梯度、邻居聚合梯度、Rank profile
python cora_rankgran7.py --cpu-test

# 只生成配置和 manifest，不启动训练
python cora_rankgran7.py --plan

# 七组实验：先 C8，再 P1、P2、G1、G2、G3、G4；GPU0 前台串行
CUDA_VISIBLE_DEVICES=0 python -u cora_rankgran7.py --run --patience 10

# 查看全局历史最优指标
python cora_all_summary.py --sort auc
```

如果 C8 已经在新协议目录完成，可跳过 C8：

```bash
CUDA_VISIBLE_DEVICES=0 python -u cora_rankgran7.py --run --skip-c8 --patience 10
```

也可以只运行尚未开始的组，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -u cora_rankgran7.py --run --only P1_RANK_46810,P2_RANK_8121620 --patience 10
```

不要覆盖同名 log。某一组失败时，runner 会停止后续组并刷新全局汇总；修复原因后应使用新的协议目录重跑，或明确归档失败日志后再重跑。

## 输出路径

- Configs：`local_configs/CORA_RANKGRAN7_20261101/`
- Logs：`logs/CORA_RANKGRAN7_20261101/`
- Manifest：`logs/CORA_RANKGRAN7_20261101/rankgran7_manifest.json`
- Checkpoints：`output/CORA_RANKGRAN7_20261101/<trial>/`
- Global best metrics：`logs/CORA_ALL_BEST_SUMMARY.log`
- CSV/Markdown：`out/cora_all_valid_best.csv`、`out/cora_all_valid_best.md`

所有七组的 Valid AUC/UAUC、epoch、状态和 checkpoint 状态都登记在 `cora_all_summary.py`。UAUC 记录与该实验最佳 Valid AUC 对应的 epoch，不是单独挑选 UAUC 最大 epoch。

## 解释限制

单次 seed=42 的结果用于筛选方向，不代表统计显著性。Rank 方案改变了各层 P 投影的参数量；G2–G4 改变了每层输入表示粒度。若有效果，下一步需要用独立随机种子复核，并进一步分析计算开销。聚类分配与邻居 ID 固定于预训练 MF 表示，训练中不会使用验证集或测试集交互来构造图或簇。

代码已提交 GitHub；CPU smoke test 和真实 GPU 训练尚需在用户服务器执行。