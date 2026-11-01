# CoRA/MF 第五轮：C6 梯度、残差融合、选择性注入、均匀秩对照

## 实验依据（用户 Valid 结果）

- C6_KD_BOTH：AUC 0.732154，UAUC@bestAUC 0.694846（当前最高 AUC）。
- C7_BIDIR：AUC 0.726525，UAUC 0.698707（当前最高 UAUC@bestAUC）。
- C8_MAIN_HALF：AUC 0.728791，UAUC 0.697618；其 KD 未筛选，不能直接归因于梯度缩放。
- G1_CLUSTER_K32：AUC 0.730836，UAUC 0.672262；聚类保留总体 AUC，用户排序可能变差。
- G4_MULTI_TOPK：AUC 0.727153，UAUC 0.691886。
- P1 rank=[4,6,8,10] AUC 0.719009；P2=[8,12,16,20] AUC 0.723650。需要均匀秩对照来区别预算与层次分配。

## 十组设计

| ID | 研究变量 | 具体实现 |
|---|---|---|
| D1_C6_GRAD_HALF | 仅 MF 主任务梯度增益 | C6 正确性+置信度 KD 过滤不变，LLM 主损失传回 MF 梯度乘 0.5；保留原始 rank16/alpha32、原始权重初始化，不额外重建 LoRA P |
| D2_K32_RES025 | 聚类+25%个体 | 所有层输入 z=0.25·原始MF +0.75·K32聚类中心 |
| D3_K32_RES050 | 聚类+50%个体 | 所有层 z=0.50·原始MF+0.50·K32聚类中心 |
| D4_TOPK_RES025 | 邻域+25%个体 | 四组层分别 TopK40/20/10/5，z=0.25·原始MF+0.75·邻域聚合 |
| D5_TOPK_RES050 | 邻域+50%个体 | 四组层分别 TopK40/20/10/5，z=0.50·原始MF+0.50·邻域聚合 |
| D6_LOW_K32 | 仅低层群体信息 | L0–15 使用 K32 聚类中心，L16–31 使用原始 MF |
| D7_HIGH_TOPK20 | 仅高层邻域信息 | L0–15 使用原始 MF，L16–31 使用 TopK20 邻居聚合 |
| D8_UNIFORM_R8 | 均匀秩预算对照 | 全 32 层、Q/K/V/O，rank=8、alpha=16 |
| D9_UNIFORM_R12 | 均匀秩预算对照 | 全 32 层、Q/K/V/O，rank=12、alpha=24 |
| D10_UNIFORM_R20 | 均匀秩预算对照 | 全 32 层、Q/K/V/O，rank=20、alpha=40 |

## 控制变量

- 十组均固定 seed=42、MF 从相同预训练权重初始化、Vicuna-7B 冻结、原始标题 prompt，不增加混合提示词。
- 全部使用 C6：原始 LLM Yes-logit BCE 主任务 + MF CTR=0.005 + KD=0.01，KD 仅选择 LLM 预测正确且 |sigmoid(Yes)-0.5|≥0.20 的样本。
- D1 是**唯一**将主损失 MF 梯度增益 gamma 改为 0.5 的实验，其他九组 gamma=1.0。
- D1–D7 使用固定 rank16、alpha32；D8–D10 仅改变所有层一致的 rank 与 alpha，固定 alpha/r=2。
- CoRA LR=3e-4、MF LR=3e-5、batch2/accum4、400 microbatches/epoch、10 个验证 epoch 的 patience。
- 不使用 Test 进行模型选择；记录最佳 Valid AUC 对应的 UAUC 及 epoch。
- 聚类和邻居 ID 只从训练集正样本和训练历史实体构建，验证集/测试集冷实体不参与邻居构建，缺失协同信号时退回原始 MF。
- 每次前台串行一组 GPU0；一组失败就停止，保留日志。
- checkpoint 严格只保存训练参数，原子写盘，不保存冻结 Vicuna、optimizer 或 scaler。
- 所有文件均在新协议 CORA_STRUCT10_20261101 下；旧实验权重和日志不覆盖。

## 服务器操作（已经验证前几轮能运行的 CoRA_official 根目录）

```bash
cd /home/bbx/user_lx/CoRA_official
# 保持原先成功运行 cora_rankgran7.py 的 Python/conda 环境

REV=1a44f8de2d4c27b0abcf1f0e1e8374262277a265
BASE=https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/$REV/experiments/no31_cora_original

# 保留本机已经存在的代码版本
for f in cora_joint8.py cora_combo8.py cora_rankgran7.py cora_all_summary.py; do
  if [ -f "$f" ] && [ ! -f "$f.before_struct10" ]; then cp "$f" "$f.before_struct10"; fi
done

curl -fL --retry 3 "$BASE/cora_joint8.py" -o cora_joint8.py
curl -fL --retry 3 "$BASE/cora_combo8.py" -o cora_combo8.py
curl -fL --retry 3 "$BASE/cora_rankgran7.py" -o cora_rankgran7.py
curl -fL --retry 3 "$BASE/cora_all_summary.py" -o cora_all_summary.py
curl -fL --retry 3 "$BASE/cora_struct10.py" -o cora_struct10.py

# 必须先通过不使用 GPU 的检查
python -m py_compile cora_joint8.py cora_combo8.py cora_rankgran7.py cora_all_summary.py cora_struct10.py
python cora_struct10.py --cpu-test
python cora_struct10.py --plan

# 运行全部十组；只用 GPU0，前台串行，自动维护全局汇总
CUDA_VISIBLE_DEVICES=0 python -u cora_struct10.py --run --patience 10 \
  2>&1 | tee -a logs/CORA_STRUCT10_20261101/launcher.log

# 结束后/中途查看全部历史最佳 Valid 排名
python cora_all_summary.py --sort auc
tail -50 logs/CORA_ALL_BEST_SUMMARY.log
```

如果只跑几组，或从已完成前几组继续：

```bash
CUDA_VISIBLE_DEVICES=0 python -u cora_struct10.py --run \
  --only D4_TOPK_RES025,D5_TOPK_RES050,D6_LOW_K32,D7_HIGH_TOPK20,D8_UNIFORM_R8,D9_UNIFORM_R12,D10_UNIFORM_R20 \
  --patience 10
```

已有同名 `*.log` 的实验不能覆盖，必须从 `--only` 指定尚未启动的实验，或者先明确归档失败日志。失败后检查 `logs/CORA_STRUCT10_20261101/<ID>.log`，不能直接把早期表现当成最终结果。

## 结果/汇总

- 原始日志：`logs/CORA_STRUCT10_20261101/<ID>.log`。
- 配置：`local_configs/CORA_STRUCT10_20261101/<ID>.yaml`。
- 精简 checkpoint：`output/CORA_STRUCT10_20261101/<ID>/.../checkpoint_best.pth`。
- 协议元数据：`logs/CORA_STRUCT10_20261101/struct10_manifest.json`。
- **历史最优汇总**：`logs/CORA_ALL_BEST_SUMMARY.log`，同时有 `out/cora_all_valid_best.csv`、`.md` 和 `.log`。
- 每组完成或出错后由实验总控自动刷新全局汇总；无结果的组显示 NOT_FOUND，不补造指标。

## 限制

这是实现提交和代码设计，不是已完成的训练。首次运行必须以服务器端 `--cpu-test`、`--plan` 和真正的前向/反向为准。单个 seed42 的细小提升不能认定为稳定显著提升。