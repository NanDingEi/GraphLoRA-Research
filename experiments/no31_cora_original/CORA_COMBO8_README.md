# CoRA × MF：第三轮 8 组损失组合实验

## 前两轮已经观察到的 Valid 结果（用户日志）

| 实验 | 监督配置 | Valid AUC | UAUC（在 AUC 最佳 epoch） |
|---|---|---:|---:|
| R2_MF_CTR | CTR=0.05 | 0.725905 | 0.698475 |
| R3_MF_KD | KD=0.05 | 0.726337 | 0.695700 |
| R4_CTR_KD | CTR=0.05, KD=0.05 | 0.725750 | 0.689830 |
| N1_CTR_001 | CTR=0.01 | 0.726860 | 0.696300 |
| N3_KD_001 | KD=0.01 | **0.727718** | 0.696547 |
| N5_CTR5_KD1 | CTR=0.05, KD=0.01 | 0.717235 | 0.690201 |
| N6_CTR1_KD5 | CTR=0.01, KD=0.05 | 0.718409 | 0.694138 |

结论是**单独较弱辅助监督有效，但此前简单的较大权重混合未奏效**。本轮只测试新的组合机制；不声称这些新实验会提高指标。

## 本轮实验设计（8 组）

| ID | CTR | KD | KD 样本选择 | 其他 | 验证假设 |
|---|---:|---:|---|---|---|
| C1_WEAK_BOTH | 0.005 | 0.005 | 所有样本 | — | 双损失同时弱化是否好于此前直接叠加 |
| C2_CTR_DOM | 0.010 | 0.002 | 所有样本 | — | 保持 CTR 主导，KD 仅轻辅助 |
| C3_KD_DOM | 0.002 | 0.010 | 所有样本 | — | 在前轮最佳 KD=0.01 上只加极弱 CTR |
| C4_KD_AGREE | 0.005 | 0.010 | 仅 LLM 预测与真实标签一致 | — | 过滤误导性教师预测 |
| C5_KD_CONF | 0.005 | 0.010 | 仅 LLM 置信度 >= 0.7 或 <= 0.3 | — | 降低不确定教师的 KD 干扰 |
| C6_KD_BOTH | 0.005 | 0.010 | 预测正确 **且** 达到置信度要求 | — | 可靠性双重筛选 |
| C7_BIDIR | 0.005 | 0.010 | 所有样本 | MF→LLM 反向 KD=0.005，限定 MF 置信且预测正确 | 检验小模型能否反向帮助 LLM，而非只优化 MF |
| C8_MAIN_HALF | 0.005 | 0.010 | 所有样本 | LLM 主损失→MF 梯度乘 0.5 | 兼顾双模型优化与抑制主损失对 MF 的扰动 |

说明：所有实验都保留原始 CoRA 的 LLM Yes-logit BCE 主损失。

CTR = BCEWithLogits(MF 用户-物品点积, y)，只直接优化 MF。

KD = BCEWithLogits(MF 用户-物品点积, stopgrad(sigmoid(LLM Yes logit)))，仅把当前 LLM 偏好信息蒸馏到 MF。

C7 额外项 = 0.005 × BCEWithLogits(LLM Yes logit, stopgrad(sigmoid(MF 点积)))。该项只对 MF 预测与真实标签一致且 |MF 概率-0.5| >= 0.2 的样本启用，以减少错误协同监督。此项更新 LLM 的 CoRA 参数（LLM 主干仍冻结），并可通过 CoRA 条件路径回传 MF；MF teacher soft target 本身 stop-grad。

在 C4–C6 中按样本掩码将 KD 损失置零，再对**原始 batch 数量**求平均，避免选中样本少时损失被人为放大。batch_size=2，因此这种掩码在不同 batch 上会有较大波动；会记录 kd_active 利用率。

C8 在 CoRA 权重生成器输入实现 ui_detach + 0.5*(ui-ui_detach)，**前向数值不变**，只把 LLM 主任务传回 MF 的梯度减半（不是冻结 MF，也不是降低 CoRA 自身梯度）。

## 一致性与保存规范

- 原始 Vicuna-7B 冻结；CoRA Q/K/V/O 动态 LoRA 正常训练，MF 解冻，所有实验使用同一预训练 MF 初始化。
- **seed=42**；CoRA LR=3e-4，MF LR=3e-5；batch_size=2，accum=4，每 epoch=400 microbatches。
- Valid_small=5200；比较 **Valid AUC** 与其最佳 epoch 对应的 UAUC；不运行 Test。
- 10 个连续验证 epoch 的 AUC 无提升即早停；保留原本 1000-epoch cosine horizon。
- 所有模型从头独立开始、只用 GPU0、**前台串行**；单组失败立即停止后面的组。
- **只保存可训练的 CoRA/MF 参数**，不保存冻结 LLM、不保存 optimizer/scaler；临时文件完整校验后原子替换。
- 旧实验的 checkpoint 即使已经删除，本轮也不依赖它们；只依赖原始预训练 MF 文件和 Vicuna 权重。
- 新目录：logs/CORA_COMBO8_20261031、output/CORA_COMBO8_20261031、local_configs/CORA_COMBO8_20261031。

## 服务器命令

```bash
cd /home/bbx/user_lx/CoRA_official

# 先更新基础文件（含精简可训练参数 checkpoint 保存）
curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/2d30a01c50e6faf2be274b628d306d5b33844c9c/experiments/no31_cora_original/cora_joint8.py -o cora_joint8.py

# 第三轮代码
curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/d2ebbcb39bf14b2d4371a27f28c66d0ff8b05dc7/experiments/no31_cora_original/cora_combo8.py -o cora_combo8.py

# 语法检查和准备配置（不训练）
python -m py_compile cora_joint8.py cora_combo8.py
python cora_combo8.py

# GPU0 前台串行运行八组；禁止后台训练
CUDA_VISIBLE_DEVICES=0 python -u cora_combo8.py --run --patience 10

# 只查看当前轮结果，无训练
python cora_combo8.py --summary
```

若发生异常，先保留失败的日志，不要删除已经完成的实验结果。`--only C3_KD_DOM,C4_KD_AGREE` 可指定尚未运行的实验；当前脚本拒绝覆盖已存在的同名日志。

## 综合跨轮次汇总

更新 cora_all_summary.py 后，运行 `python cora_all_summary.py --sort auc` 查看 R0–R7、N1–N8、C1–C8 的同一张表，并生成 out/cora_all_valid_best.csv 和 .md。被删除 checkpoint 的旧结果仍保留日志指标，但 CKPT 会显示 MISSING，不能以为可直接 Test。

## 结果解释

最佳选择同时参考 AUC、UAUC、最佳 epoch 和有效训练状态。一次 seed=42 的小幅提升不能当作确定的泛化收益。尤其 C4–C7 的样本筛选可能使损失有效样本减少，要同时比较 kd_active/rev_active。优先与 N3_KD_001 (AUC=0.727718) 及 R2_MF_CTR (UAUC=0.698475) 作日志级对照。

代码已提交 GitHub，但未在用户 GPU0 上进行训练验证；所有 C1–C8 指标仍为未知。
