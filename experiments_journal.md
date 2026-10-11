# B2/B3 双向互学习实验记录

> 目标：LLM 语义视图 ↔ GNN 协同视图双向互学习，最终以 LLM 推荐 AUC/uAUC 为准。
> 判定标准：valid AUC 是否保持/超过热启动点（基线 GraphLoRA best ≈ 0.6995）。
> 共享 GPU 注意：有其他项目在调试，训练前查 `logs_monitor/`，batch=2 时训练占 ~18.5GB。

## K/L/M 系列：层扫描、邻域精细、正交旋钮、上游对齐（2026-09-28～29，自主循环 + 用户指令）

**新实现**：`sem_layer`（语义提取层参数化，默认 8）、`sem_layers: "4,8,12"`（多层关系矩阵集成，单次 teacher 前向取多层隐状态各自算 cos 后平均）、`rec_lr_scale`（差分学习率）、lora_config 真正接线、cosine 语义模式。

### L 系列：语义提取层扫描（1-epoch 压缩调度）
层 2/4/6/8/10/16/24/30 → 0.6611 / 0.6482 / 0.6242 / **0.7076** / 0.6420 / 0.6486 / 0.6297 / 0.6557。
- **层 8 是针尖状尖峰**：与 L4 只差一行 `sem_layer: 4→8`（用户要求的单变量复制对照 L8ctrl）= **+5.9pt**；±2 层即 -6.5~8.5pt。
- **调度混淆发现（用户质疑触发）**：`max_epoch: 1` 会把 cosine 衰减总长同步压缩到 1 epoch（lr 提前到底），L8ref（层 8+1ep 调度）0.7076 vs I3（层 8+3ep 调度）0.7326 = **调度本身 -2.5pt**。教训：**1-epoch 探针必须用 `max_epoch: 6 + iters_per_epoch: 5649` 保持调度总长**，不能用 max_epoch: 1。
- L4 "3-epoch 重测"实为同配置复跑（驱动指向旧文件，0.6482 逐位复现）——已纠正并重跑，结论以 1-epoch 调度内相对排序 + 层 8 的 3-epoch 锚点为准。

### K/M 系列：正交旋钮 + 上游对齐（全部未超 I3/A）
| run | 配置 | 结果 |
|---|---|---|
| K1 | gs02 半epoch探针（6 评估点）| 峰值在 1.0ep 边界（0.7228，重洗数据 -1pt 噪声）|
| K2 | gs02 + 门关 | 0.7196 → 0.6444 → 0.6247（H1 的稳衰减不能平移到 gs02）|
| K3 | gs02 + cosine 语义模式 | 0.7178（低于 MSE 1.5pt，"MSE 过强"再否定）|
| K5 | **多层关系集成 4,8,12** | 0.6837 ✂（用弱层稀释层 8 有害——尖峰非噪声）|
| K6 | gnn_scale 0.25 | 0.6402 ✂ |
| K7 | **min_lr 5e-5→8e-5（上游对齐）** | 0.7319 → 0.7305 → **0.6913** — ep1 微升但 ep2 更差，**min_lr 假设否定：ep2 衰减是数据级过拟合，非 lr 轨迹产物** |
| M1/M2/M3 | TopK3/2 × w 2e-4/3e-4 | 0.6380 / 0.6640 / 0.6395 全✂ — **TopK 家族权重上限就是 1e-4，不随 K 上移** |
| I4 | gnn_scale 0.05 | 0.6379 ✂（曲线左端闭合）|

### 关键响应曲线汇总（全部针尖状）
- **gnn_scale**：0.05→0.638 / 0.1→0.729 / 0.15→0.718 / **0.2→0.733** / 0.25→0.640 / 0.3→0.664
- **语义提取层**（1ep 调度）：2→0.661 / 4→0.648 / 6→0.624 / **8→0.708** / 10→0.642 / 16→0.649 / 24→0.630 / 30→0.656
- **w_sem**（MSE）：5e-5→0.717 / **1e-4→0.729** / 2e-4(rank)→0.638 / 3e-4(rank)→0.664 / 5e-4→0.642
- **TopK**(rank)：2→0.726 / **3→0.728** / 5→0.648 / 10→0.641
- A_validate（第 5 次位级验证）：0.7291/0.7221/0.7047 逐位复现——代码无漂移。

### 当前 standings（42 个变体后）
- **test 最高：A（gs01）0.7378**；**valid 最高：I3（gs02）0.7326**（test 0.7368）
- 两者只差 gnn_scale；所有机制创新（rank/TopK/手术/集成/门控/两阶段/层移动）均未超越
- ep2 衰减定位为数据级过拟合（K7 否定 lr 假设）；峰值在 ep0 末尾（K1 探针）

### K7 test 补测（2026-09-29 深夜，用户指令：选型必须以 test 为准）
- K7 重训（逐位复现 0.7319/0.7305/0.6913）后 test 评估 best(ep0) = **0.7376**
- **test 终榜**：A **0.7378** ≈ K7(min_lr8e-5) **0.7376** ≈ I3 0.7368 —— 三者统计打平（差距 ≤0.1pt）
- valid 层面的排序（I3>K7>A）在 test 上不成立；valid 只能用于初筛，最终裁决必须 test
- 实操结论：三个配置可任选；若在意 ep1/ep2 的稳定性（部署时无 best-checkpoint 重载的场景）选 K7，否则 A/I3 均可

## I/J 系列：自主调参循环（2026-09-27～28，用户指令：自行设计方案持续尝试）

**新机制实现**（minigpt4rec_v2.py，全部配置门控默认关）：`sem_detach_mf`（语义分支用 detach 的 MF 嵌入跑第二次 GNN 前向，语义梯度只训 NGCF 结构）、`two_phase_gate`（rec→GNN 门只在 ep0 开）、`sem_loss_mode: cosine`（尺度无关对齐）、lora_config 真正接线（此前 r/alpha/dropout 改 yaml 无效）。

| run | 配置 | 结果 |
|---|---|---|
| **I3** | **gnn_scale 0.1→0.2** | **0.7326 → 0.7290 → 0.7193，test 0.7368 — valid 新冠军** |
| I1 | sem_detach_mf + w=5e-4 | 0.6992 击杀（detach 把同剂量从 0.6420 救回 +5.7pt，机制证实但不够）|
| I2 | sem_detach_mf + w=1e-4 | 0.6494 击杀 → **机制发现：语义损失的杠杆就是 MF 嵌入表**（切断后 GNN 退化近随机 ≈H2）|
| I6 | two_phase_gate | 0.7291 → 0.7246 → 0.7046（与 A 逐位起点、ep1 微升、ep2 归同——衰减根源不在 GNN 信号来源）|
| J1 | gnn_scale 0.3 | 0.6639 击杀（峰值在 0.2）|
| J2 | gnn_scale 0.15 | 0.7180 → 0.6520 → 0.6324（ep1 塌入坏盆地）|
| I5 | LoRA dropout 0.05→0.1 | 0.6125 击杀（拖慢 ep0 快速适配）|
| I7 | weight_decay 1e-3→3e-3 | 0.6403 击杀 |
| I8 | iters_per_epoch 误配（实际=A 复刻）| 0.7291 逐位一致，第 4 次确定性确认，终止 |

**gnn_scale 响应曲线**：0.05? / 0.15→0.7180(后塌) / **0.1→0.7291 / 0.2→0.7326(峰)** / 0.3→0.6639。K 系列（半 epoch 峰值探针、gs02+门关、cosine）进行中。

## 语义损失形态改造：Rank + TopK + 梯度手术（2026-09-25，用户指令：解决双损失梯度冲突与 MSE 过强）

**实现**（minigpt4rec_v2.py + base_task.py，全部配置门控、默认关闭 = 行为与 A 完全一致）：
- `sem_loss_mode: rank`：MSE → 置信度加权排序 hinge。三元组 (anchor i, j, k)：`L = w·relu(m − sign(d_L)·d_G)`，`d_L=S_L(i,j)−S_L(i,k)`，`w=min(|d_L|/τ,1)`，m=0.02，τ=0.05。LLM 只教"谁更近"，不教绝对 cosine。
- `sem_topk: K`：每个 anchor 只保留 LLM 语义空间 Top-K 邻居；rank 模式对齐 (邻居, 非邻居) 三元组；mse 模式只对齐 (anchor, TopK 邻居) 对。
- `grad_surgery: True`：PCGrad 式全局冲突投影。`dot(g_rec,g_sem)<0` 时 `g_sem' = g_sem − (g_sem·g_rec/||g_rec||²)g_rec`，组合 `g = g_rec + w·g_sem'`。实现要点：**先 backward loss_sem（其图只含 GNN 子图，teacher 已 detach，retain 代价极小），再 backward loss_rec**——反序会因 retain 整个 LLM 激活图在 batch 3 下 OOM（E4 首跑踩坑）。手术在 GradScaler 缩放空间进行（缩放因子在投影公式中抵消），`scaler.step` 统一 unscale。支持参数：`sem_rank_margin/sem_rank_tau/sem_loss_mode/sem_topk/grad_surgery`。

**结果**（全部 3 epoch 上限，ep0 < 0.70 击杀；对标 A：valid 0.7291@ep0 / test 0.7378）：
| run | 配置 | 轨迹 | 结论 |
|---|---|---|---|
| E1 | rank 全对，w=1e-4 | ep0 **0.6550** 击杀 | 排序信号太稀（teacher gap p50 仅 0.005），hinge 不自熄（恒 ~0.012）持续扰动，eval loss 1.04 |
| **E2** | **rank + TopK3，w=1e-4** | **0.7156 → 0.7281@ep1 → 0.7184**，**test 0.7232** | TopK 救活排序方案（+6pt vs E1）且健康（eval loss 0.625，唯一后程上升的 run），但 valid/test 均未超 A |
| E3 | rank + TopK3，w=1e-3 | ep0 **0.6320** 击杀 | 10 倍剂量毁灭性 |
| E4 | rank+TopK3+手术，w=1e-3 | ep0 **0.6458** 击杀 | 手术只剔冲突分量；10 倍剂量下不冲突的噪声部分仍扰动 |
| E5 | mse+手术（= A+手术），w=1e-4 | ep0 **0.6431** 击杀，eval loss 1.07 | **深挖见下** |

### E5 深挖：手术实现被证明数学惰性，暴露"刀锋边缘"训练 regime（重要）
E5 = A 配置 + 纯手术（MSE 模式），理论上应与 A 逐位一致，却从 iter ~50 分歧并塌缩（ep1 train loss → 0.0084，yes-logit 饱和过拟合）。系统性排查（probe 已删，方法记录在此）：
1. **梯度等价性**：同一 batch、同 RNG 状态下，手术 backward 与单遍合并 backward 的 148 个可训练参数梯度**逐参数一致**（60 个连续 batch 最大相对差 6.2e-7）。
2. **初始权重一致**：两配置构建的全部 148 个可训练参数逐位相同。
3. **确定性**：A 配置在当前代码上重跑 250 iter 与 A 原始 log 逐位一致；E5 自身两次运行也逐位一致。
4. **诊断数据**（surgery 日志）：`w·coef ≈ 1e-8`（投影修正量可忽略），`||g_sem||/||g_rec|| = 0.03–0.08`——**语义梯度比推荐梯度小一个数量级，冲突场景在本设置下根本不绑定**。用户设想的"冲突损害主任务"机制在 w=1e-4 剂量下不成立。
5. **根因**：md5 级差异来自 embedding-backward/sparse-mm 的 CUDA 原子操作浮点非结合性（两次 backward 的累加顺序 vs 单次）——bit 级噪声被 11297 步混沌放大。E5 相对 A 等价于"换了一个有效随机种子"。
6. **双峰稳定性**：本 regime 存在健康盆地（A、E2：eval loss ~0.63）与塌缩盆地（C/E1/E3/E4/E5：eval loss 1.0–1.4、yes-logit 饱和、train loss→0）。bit 级扰动足以决定落点 ⇒ **单 run 对比在此预算下不可靠**，A 的 0.7291 与 E2 的 0.7281 是同分布抽样。

### 结论与遗留
1. **A（纯净 RLM 原配）仍是冠军**：valid 0.7291 / test 0.7378。E2 是唯一健康变体（test 0.7232）。
2. 用户方向 1（rank）单独有害、配 TopK 后中性略负；方向 2（TopK）对 rank 是必要稳定器；方向 3（手术）实现正确但所防御的冲突在本剂量下不存在（||g_sem||≪||g_rec||）——若要提高语义剂量（≥1e-3）则手术是正确框架，但 10 倍剂量本身即塌缩，需先解决剂量-稳定性问题。
3. **后续必做：seed 方差测量**（A 配置 seed 43/44 各一次）。若 ep0 AUC 波动 >2pt，则历史所有单 run 结论需重新校准；方向 4（Semantic Residual Corrector，用户建议的第二阶段）在测量噪声基线之前不应开工。
4. 新配置文件：`collm_pretrain_mf_ood_rlm_E1_rank/E2_rank_tk3/E3_tk5_w3/E4_combo/E5_mse_surg.yaml` + `collm_pretrain_mf_ood_rlm_eval_e2.yaml`（test 评估模板）。

## F 系列：rank+TopK 参数量级扫描（2026-09-25 晚～26 早，用户指令：多试参数量级，E2 试 5e-4）

围绕 E2（rank+TopK3, w=1e-4）做邻域/剂量/损失形状扫描，8 个变体全部跑完（ep0<0.70 击杀）：

| run | TopK | w_sem | 其他 | ep0 | 结局 |
|---|---|---|---|---|---|
| F1 | 5 | 1e-4 | — | 0.6475 | 击杀（L_sem 不自熄 ~0.01-0.03）|
| **F2** | **2** | **1e-4** | — | **0.7174** → 0.7263@ep1 → 0.7131 | **完成**（唯一幸存，test 未评）|
| F3 | 5 | 3e-4 | — | 0.6167 | 击杀 |
| F7 | 3 | **5e-4**（用户指定）| — | 0.6471 | 击杀 |
| F6 | 10 | 1e-4 | — | 0.6413 | 击杀 |
| F8 | 5 | 5e-4 | — | 0.6320 | 击杀 |
| F4 | 3 | 1e-4 | margin 0.02→0.05 | 0.6295 | 击杀 |
| F5 | 3 | 1e-4 | tau 0.05→0.02 | 0.6495 | 击杀 |

### F 系列结论
1. **健康窗口极窄且完全由剂量决定**：只有 (TopK≤3, w=1e-4) 存活；邻域扩大（K≥5，三元组数平方增长）或剂量提高（3e-4/5e-4）一律塌进坏盆地（0.62-0.65）。用户指定的 5e-4 剂量（F7）即使配 TopK3 的自熄信号也不行。
2. **L_sem 自熄性是健康的最可靠先行指标**：健康 run 的 L_sem 从 ~1e-4 量级单调压到 1e-5 以下（F2: 0.000009-0.005）；塌缩 run 的 L_sem 恒定在 0.005-0.03 不熄灭。后续可把"L_sem 500 步不降"作为训练内早停信号，省掉等 ep0 eval 的 70 分钟。
3. F2 与 E2 轨迹几乎重合（0.7174/0.7263 vs 0.7156/0.7281，且都 ep1 达峰）——该剂量下 rank 信号贡献 ~1e-6/步，两个 run 本质上都是"A + 微小扰动"的抽样，再次印证单 run 方差问题。
4. **最终定论：rank+TopK 方向在此数据/预算上不产生超越 MSE 基线的增益，参数空间已扫完**。项目最佳仍是 A（valid 0.7291 / test 0.7378）。下一优先级不变：seed 方差测量 → 再决定是否做 Residual Corrector。

## H 系列：梯度门控隔离实验——语义损失独立有效性（2026-09-27，用户设计：关闭主任务损失回传 GNN）

**设计**：`train_gnn_with_rec_loss: False`（rec 梯度不回传 NGCF，仅语义损失训练 GNN），与 G1/构成 2×2 消融（rec→GNN 门 × 语义损失）。种子实验按用户指示取消。

| run | rec→GNN | 语义 | valid 轨迹 | test |
|---|---|---|---|---|
| A | ✓ | ✓ 1e-4 | 0.7291 → 0.7221 → 0.7047 | **0.7378** |
| G1 | ✓ | ✗ | 0.7282 → 0.7146 → 0.6393 | 0.7276 |
| H1 | ✗ | ✓ 1e-4 | 0.7286 → 0.7248 → 0.7117 | 0.7269 |
| H2 | ✗ | ✗（NGCF 永远随机初始化=纯随机投影对照）| 0.6612 | 击杀 |

### H 系列结论（本项目最干净的消融）
1. **语义损失独立有效实锤**：门关后语义损失是 NGCF 唯一训练信号，H1 − H2 = **+6.7pt**——语义对齐把随机初始化的 NGCF 训练到与全信号版本持平的水平（0.7286 vs 0.7291）。没有语义损失时 NGCF 就是随机投影，性能塌到 0.66。
2. **rec→GNN 回传在 test 上仍有独立贡献**：A − H1 = +1.1pt（valid 上两者持平）。完整 2×2 的 test 排布非常有规律：三个"缺一件"的配置全部落在 0.7269–0.7276，只有双组件齐全的 A 到达 0.7378——**两路梯度互补、缺一不可，齐全才有超加性收益**。
3. **论文级消融故事成型**：rec 损失负责 LLM/LoRA 与 MF 嵌入（经 llama_proj 路径），语义损失独立承担 GNN 结构学习（H1 证明其充分性），两者在注入路径上汇合时效果最佳（A）。
4. H1 的衰减也更平缓（ep2 0.7117 vs A 0.7047/G1 0.6393）——语义-only 训练的 GNN 过拟合更慢，值得后续在更长预算下复核。

**配置留存**：`collm_pretrain_mf_ood_rlm_H1_gate_sem/H2_gate_nosem.yaml` + `eval_h1.yaml`。此前用户观察"加语义损失 > 原基线、可达 0.73+"与本矩阵一致：原基线 0.7149 → +IDList ≈ 0.7276 → +语义 ≈ 0.7378。

## G 系列："原模型已最优"假设检验（2026-09-26，用户假设：ep0 达峰后衰减是因为原模型已最优，应降协同侧学习率或降 LLM→GNN 更新权重）

**实现**：差分学习率框架——`run.rec_lr_scale` 让 rec_encoder+gcn 参数进独立 AdamW 组，LR 调度器对每组乘 `lr_scale`（optims.py + runner_base.py，默认 1.0 = 行为不变，单测验证）。

| run | 配置 | 轨迹 | best |
|---|---|---|---|
| G1 | **语义全关**（align=False，缺失的关键消融）| 0.7282 → 0.7146 → 0.6393 | 0.7282@ep0，**test 0.7276** |
| G2 | rec_lr_scale=0.1（协同侧 1e-5）| 0.6055 | 击杀 |
| G3 | freeze_rec=True（MF 全冻）| 0.7161 → 0.6391 → 0.6129 | 0.7161@ep0（塌缩更快）|
| G4 | w_sem=5e-5（半剂量）| 0.7095 → 0.7168 → 0.6951 | 0.7168@ep1 |

### G 系列结论（对用户假设的回答）
1. **"降低协同侧更新"不成立**：降 LR 10 倍（G2，-12pt）或全冻（G3，-1.3pt 且塌缩更陡）都显著变差——MF 嵌入/GNN 在 ep0 需要**全速适配**，它们并非"已经不动最优"，而是"必须在第一个 epoch 内快速走完适配"。衰减是整个 regime 的性质（所有 13+ 个 run 都 ep0/ep1 达峰后下降），不是语义损失或协同侧更新的锅。
2. **w_sem 完整剂量曲线补齐**：0（0.7282/0.7276）→ 5e-5（0.7168）→ **1e-4（0.7291/0.7378，最优）** → 3e-4/5e-4（塌缩）。语义损失在安全剂量下 valid 中性（G1 vs A 差 0.0009 ≈ 噪声），**test 上 A 反而好 1pt**（0.7378 vs 0.7276）——语义损失以 1e-4 的剂量提供的泛化增益虽然小但方向为正，去掉它模型更简单但 test 变差。
3. **all-run 汇总**：13 个 G/E/F 变体 + A 本身，无任何扰动超过 A。A 的配方（w_sem=1e-4, batch 3, lr 1e-4, 全参数更新）就是当前数据/预算下的局部最优解。
4. 遗留：seed 方差测量（A 配置 seed 43/44）仍是判断"哪些差距是真"的唯一手段；Residual Corrector 等噪声基线明确后再议。

## 纯净 RLM 调试（2026-09-24 深夜～25 上午，用户指令：基于新版纯净 RLM 调试+调参）

**背景**：模型代码重写为纯净版（`gnn_semantic_align` 单向 LLM 语义→GNN 修正，`_compute_gnn_semantic_loss` = batch 内 item-item 关系矩阵 MSE 对齐，teacher=第 8 层 LLM 隐状态 stop-grad）。配置 `collm_pretrain_mf_ood_scratch_rlm.yaml`（IDList prompt，NGCF，gnn_scale 0.1，gcn_layer 31，lr 1e-4，**batch 3**，w_sem=1e-4，3 epoch）。停机规则：ep1 AUC 明显低于 0.72（操作化为 <0.70，因历史上所有健康 run 的 ep1 都在 0.70-0.71、峰值在 ep2）即杀。

| run | 配置 | valid 轨迹 | 结论 |
|---|---|---|---|
| **A_base** | 原配 w=1e-4，batch 3 | **0.7291**@ep0 → 0.7221 → 0.7047 | **ep0 即达峰且为全项目历史最高 valid**；**test 0.7378**（新纪录，超 gate_disagree 0.7263 达 +1.15pt） |
| C_sem5 | 仅 w→5e-4（journal REL 剂量），batch 3 | ep0 **0.6420**，eval loss 1.406 | 击杀。**剂量灾难**：与用户当日 20:42 调试 run 逐位复现（loss 至小数 6 位一致），确认该 run 即 5e-4 剂量 |
| B_v12 | A + 偏好蒸馏 0.5 + κ 头 0.1（v12 配方移植），batch 1 组(2行)，w_sem 回 1e-4 | ep0 **0.6769**，eval loss 0.764 | 击杀。候选组格式（每样本加均匀负例进 L_rec + KL 蒸馏）对纯净 RLM 是净伤害。pref 蒸馏/κ 移植代码保留（单元测试验证梯度正确），配置 `collm_pretrain_mf_ood_scratch_rlm_v12.yaml` |
| D2_lr5 | 仅 lr→5e-5（min 2e-5），batch 3 | ep0 **0.6284**，eval loss 1.140 | 击杀。低 lr 使 LoRA 适配不足（batch 3 每 epoch 仅 11297 步）；**本配置的强度恰恰来自 lr 1e-4 的快速适配** |

### 核心结论
1. **纯净 RLM 原配（w_sem=1e-4，batch 3，lr 1e-4）= 当前全项目最强**：valid 0.7291@ep0 / **test 0.7378**。best checkpoint：`output/graphlora_ml1m_semantic/20260924231/checkpoint_best.pth`（=ep0）。
2. **峰值在 ep0**（0.7291→0.7221→0.7047 单调降）——与历史 batch-1 时代"ep2 达峰"完全不同；batch 3 的大步长快速适配 + 预训练 MF 编码器在第一个 epoch 内即达最优平衡，之后过拟合。
3. **语义剂量极敏感且方向明确**：1e-4 恰好；5e-4 毁灭性（-8.7pt）。机理：大批量 → 每步 item 对数 3 倍 → 关系对齐梯度对预训练 MF 嵌入表的冲击过大（呼应 9-13 的梯度流发现：L_sem 主要训练 MF 嵌入表）。**在此 batch 曝露下不要提高 w_sem**。
4. **v12 配方（pref 蒸馏+κ 头）不可直接移植**：候选组格式改变了 L_rec 任务分布（均匀负例入损失），ep0 -5.2pt。
5. 训练确定性再次确认：A 与误覆盖 lr 的重复 run 轨迹逐位一致。**注意：`--options` 无法覆盖 run: 段配置**（键被合并到顶层），调 run 参数必须改 yaml（本次 D 第一次启动即踩此坑）。

### 遗留物
- 变体配置：`collm_pretrain_mf_ood_scratch_rlm_sem5.yaml`（5e-4，已证伪）、`..._v12.yaml`（pref+κ@1e-4，已证伪）、`..._lr5.yaml`（lr 5e-5，已证伪）、`collm_pretrain_mf_ood_rlm_eval_base.yaml`（A 的 test 评估配置）。
- 代码：minigpt4rec_v2.py 新增 `pref_distill`/`pref_distill_weight`/`pref_tau_l`/`pref_tau_g`/`pref_kappa_init`/`gnn_conf_loss_weight`（默认全关，不影响原行为；`compute_feedback_gcn` 的 token_gnn 视图条件改为 `gnn_semantic_align or pref_distill`）。
- 未试：mid-epoch checkpoint（ep0 内部峰值探测）、seed 稳定性、gnn_scale 扫描。

## 崩坏原因分析（b3_v2，2026-09-06）

| 证据 | 数据 | 结论 |
|---|---|---|
| train loss 持续降、valid AUC 崩 | loss 0.512→0.469→0.466→**0.401**；AUC 0.708→0.708→0.678→0.655 | 模型在**主动变形拟合训练图**，不是数值不稳定 |
| 基线对照 | 基线 valid AUC 在 0.66-0.72 震荡维持 ~19 epoch 才衰退 | 排除"热启动+lr 大"主因；是 mutual loss 加速了崩坏 |
| 崩坏起点 | KL 降到 ~0.45/0.73 以下（epoch 2 起）才开始掉，KL 全程单调 | **问题在低 KL 区间的持续碾压** |

**机制**：`KL(softmax)` 要求两分布的**绝对置信度**一致。排序一致后 KL 要继续降，唯一途径是把 LLM 中间层余弦几何的幅度掰成 NGCF 256 维空间的幅度（G→L 方向 λ=0.02 最强，主犯）。eval 用的是与训练不同的 valid/test 图（OOD 设计），变形的表示换图后泛化更差 → train 越拟合 valid 越崩。方案原意"只关心谁更近谁更远"没有被 KL 执行到位——它多传递了不该传递的**幅度**信息。

**修复（v3，mutual_rank_loss）**：KL → **带教师权重的 pairwise 排序 hinge**：
- 教师（detach）只提供每对候选的偏好**符号** + 果断度权重 `w=min(1,|Δ_teacher|/0.05)`（接近并列不施压）
- 学生的 hinge：`w·relu(rank_margin − sign·Δ_student)`，margin=0.02
- 两边排序一致后损失**精确归零、梯度归零**（自终止），无尺度传递，结构上消灭低 KL 碾压
- λ 重标定（hinge 量级 ~0.02 vs KL ~1）：λ_L→G=0.5，λ_G→L=2.0（保持 4:1 不对称）
- 另加 `mutual_warmup_iters` 旋钮（默认 0=关）
- v3 配置：`tran_configs/collm_pretrain_mf_ood_b3_rank.yaml`，output_dir `output/graphlora_ml1m_b3_rank`

### 方案级优化（2026-09-06 下午，用户指令：不调超参，试不同方案）
- **方案 A：checkpoint soup**（v4 ep0/1/2 权重平均）→ eval AUC 0.7147 ≈ ctrl best 0.7149，**无增益**，弃。output/graphlora_ml1m_b3_rank_lowlr/soup_012.pth
- **方案 B：ItemIDList prompt（v5c，进行中）**——历史物品以 ID token 进入 LLM 输入与 GNN 注入（CoLLM 原始设计）。收益点：①LLM 输入多了 ID 级历史信号；②mutual 候选集从 K=2（ctrl 已证无增益）变 K=11/样本；③batch=1 即可用（每样本自带候选）。配置 collm_pretrain_mf_ood_b3_idlist.yaml，prompts/collm_movie_idlist.txt
- **修复预存 bug**：`encode_recdata_v2` 的 node_ids 循环硬编码假设 user 段在 pos 0，`get_ids_order` 重排后错位吞 node（merged=2 vs nodes=1，冷用户必触发）→ 已改为按段边界判断；`recprompt_wrap_v2` 的 Warning-Fix 截断路径改为三者一致截断。修复对坏顺序 prompt 也验证通过（800 batch × 2 = 0 不一致）
- **发现**：train/valid/test 三个 graph npz 是同一份 train 图的复制（build_graph.py 设计如此），eval 用的就是全量 train 图——图结构无问题
- **发现**：存在全 0 历史的冷用户样本（nonpad=0），ItemIDList 展开为空

| 版本 | 配置差异 | 关键观察 | valid AUC | 结论 |
|------|----------|----------|-----------|------|
| b3_v1 | T=1.0, batch=2, λ=0.005/0.02, warm-start 基线 best | KL 仅 ~0.013 且两方向几乎相等：T=1 把 cos gap(~0.03-0.05) 抹平成近均匀分布，辅助信号无效（λ 加权后 ~5e-5） | 未评估即停 | 温度问题，弃 |
| b3_v2 | T=0.05, batch=2, λ=0.005/0.02, 同 ckpt | 诊断：LLM cos gap mean=0.045/p50=0.005，GNN mean=0.029/p50=0.021 → 取 T=0.05。KL 从 1.11/1.75 快速下降至 0.26/0.44 | ep0: **0.7071** → ep1: **0.7079**(best) → ep2: 0.6777 → ep3: 0.6555，连续两轮大跌，触发停机 | **过度对齐**：KL 被持续压低时 G→L 方向把 LLM 中间表示拉向协同结构，损害主任务。best=epoch1 已超基线 0.6995 |
| b3_v3_rank | T 同 v2（rank 模式不用 T），hinge 排序损失，λ=0.5/2.0，同 ckpt，lr/batch 同 v2 | epoch0 0.7107 > v2 同期；但 ep1 0.7076、ep2 0.6961 连续下降，触发停机。退化形态与 v2 重复（best 都在 ep0-1），只是幅度温和（ep2 -0.012 vs -0.030） | ep0: **0.7107**(best) → ep1: 0.7076 → ep2: 0.6961 | hinge 修复了 KL 的置信度碾压（降幅减半以上），但 lr=1e-4 对热启动太大仍是主因 |
| b3_v4_rank_lowlr | 同 v3 + lr 2e-5 | ep3 突降停机 | ep0: **0.7203**(best) → 0.7176 → 0.7126 → 0.6947 | 与对照几乎重合 |
| ctrl_lowlr | **无 mutual**，lr 2e-5 | **决定性对照**：与 v4 轨迹几乎重合 | ep0: 0.7197 → 0.7161 → 0.7124 | **0.72 来自低 lr 微调本身，mutual 无可测增益** |
| soup_012 | v4 ep0/1/2 权重平均 | 无增益 | 0.7147 | 弃 |
| b3_v5c_idlist | ItemIDList prompt + rank loss + lr 2e-5 + batch=1 | **唯一持续上升的 run**：epoch1 起不降反升，epoch2-3 稳定新高；ep4-5 缓降（比 ctrl 晚 2 epoch、幅度减半）。best ckpt = ep2/3 | ep0: 0.7184 → ep1: 0.7194 → ep2: **0.7227** → ep3: **0.7227** → ep4: 0.7202 → ep5: 0.7191 | **IDList+mutual 真实增益确立**；**test AUC = 0.7262**（output/eval_v5c_best/）|

### 最终 test AUC 对比（同一 eval 协议）
| 模型 | test AUC |
|---|---|
| ctrl best（无 mutual，低 lr 微调） | 0.7149 |
| v4 best（mutual rank，无 IDList） | 0.7164 |
| **v5c best（IDList + mutual rank）** | **0.7262** |

v5c 相对 ctrl **+1.1pt**、相对 v4 **+1.0pt**。方案级结论：把历史物品 ID token 引入 LLM 输入与 GNN 注入路径（CoLLM 原始设计）+ 修复后的 hinge 关系互学习（K=11 候选）= 真实有效的方案组合。

## B9/B10/B11 实现与从零对比（2026-09-06 晚，用户指令：尝试 B9-B12）

### 实现记录（minigpt4/models/minigpt4rec_v2.py）
- **B11 分歧门控**：`mutual_gate: threshold|disagreement|off`；GNN 置信度头 `s_G = κ·cos(z_u, z_target)`（κ 可学习，自有 BCE 校准，权重 0.1）；hinge 互学习损失重构为 per-user 向量，按样本门控 `m_LG = 1[q_L>τ_h ∧ q_G<τ_l]`（反向对称）；"该传才传"
- **B9 独立协同专家**：`injection_mode: expert`——P_G(256→8) + B_G(8→4096) 独立专家，不再与任务 LoRA 共享 rank-8 瓶颈；decoder layer forward-hook 在隐空间注入 `h' = h + α_l·ρ·γ·(dh/‖dh‖)·‖h‖`；ρ 为 per-token router（[h;z]→sigmoid）
- **B10 多层注入**：`injection_layers: [7,15,23,31]` 每层独立 adapter + layer router α（z_u→softmax）
- **调试中修掉的三个坑**：①zero-init B_G + fp16 隐状态（量级~100）→ add 被舍入为 0、梯度死锁 → 改相对范数注入；②relative-norm 下 B_G 零初始化残留 → dh≡0 → 改随机方向初始化；③smoke 断言用 loss——未训练模型 Yes-logit 饱和使 BCE 不敏感 → 改用 logits 差异（实测 L31 隐状态差异 6.06，注入正常）
- **B12 搁置**：bilevel 优化需二阶梯度穿过 7B 模型，24GB 共享卡不可行，留待有更大显存时做

### 从零训练对比（IDList prompt，lr 1e-4，batch 1，3 epoch，完全同预算）
| 方案 | ep0 | ep1 | ep2 | best | 备注 |
|---|---|---|---|---|---|
| **v6 门控（B11）** | **0.7040** | 0.7098 | **0.7244** | **0.7244** | 全程领先对照 |
| **v7 专家（B9）** | 0.6878 | 0.7156 | 0.7205 | 0.7205 | 单层 31 |
| v8 多层（B9+B10） | 0.6685 | 0.6729 | 0.7028 | 0.7028 | 从零训练下低层注入干扰早期表示 |
| 对照（v5c 复刻，无新方案） | 0.6088 | 0.6955 | 0.7100 | 0.7100 | ep0 显著落后 |

**关键发现**：对照在 ep0 只有 0.6088——从零训练早期模型乱猜时强行互学习有伤害；v6 的门控恰好规避（早期双方都不自信 → 门自动关闭 → 纯 L_rec），**直接验证了 B11 的设计目标**。方案排序：B11 > B9 > B10 > 对照。B10 的多层注入对从零训练有害、对热启动可能不同（待热启动验证）。

### 空间管理（用户指令：只保留纯粹代码）
2026-09-06 已清理 output/ 全部训练产出（~282GB→8KB），保留代码+配置+数据集(131MB)+MF 小权重(4.1MB)+journal。热启动 ckpt 已随之删除——从零对比因此统一从零训练。**此后约定**：每 run 结束立即删 checkpoint 只留 log 数字；定期 du 检查。

## 项目暂停总结（2026-09-06 深夜，用户指令：停一停，总结方案与结果）

### v10 归因实验的意外发现（中止于 ep1，0.6383→0.5635）
v10 = gate OFF + κ 头 + **全量互学习**（masks 全 1）——比对照还差得多。结合 v6（gate ON，互学习只对稀有分歧样本开 + κ 头）最优，归因链完整了：
- **全量互学习是伤害源**（对照 ep0 0.6088 / v10 更差）；**稀疏门控互学习是收益源**（v6）；κ 头本身在 v6 里无害
- 门控统计（mLG/mGL≈0）+ v6>v10+v6>ctrl 三条证据互洽：**互学习的价值在于"选对时机传"，而非"传得多"**
- 后续待做：完全解耦实验"gate ON + 无 κ 头"（目前 κ 与 gate 在代码里耦合创建）

### 全部方案与结果一览
**热启动时代**（基线 best 存在时，lr 2e-5，IDList 除非注明）：
| run | 方案 | best valid | test |
|---|---|---|---|
| v5c | IDList prompt + hinge 互学习 | **0.7227** | **0.7262**（全局最佳）|
| v4 | hinge 互学习（旧 prompt） | 0.7203 | 0.7164 |
| ctrl | 无互学习 | 0.7197 | 0.7149 |
| v3 | hinge（lr 1e-4） | 0.7107 | — |
| v2 | KL（lr 1e-4） | 0.7079 | —（崩坏：KL 低区间碾压）|
| soup | v4 权重平均 | 0.7147 | 无增益 |
| b5（另一会话）| B3-Pred | 0.7224 | — |

**从零时代**（3 epoch 同预算，lr 1e-4，batch 1）：
| run | 方案 | best valid |
|---|---|---|
| v6 | **B11 门控 + κ 头** | **0.7244** |
| v9 | B11 门控 + B9 专家 | 0.7229 |
| v7 | B9 单层专家 | 0.7205 |
| ctrl | 纯互学习（v5c 复刻） | 0.7100 |
| v8 | B9+B10 多层专家 | 0.7028 |
| v10 | 全量互学习 + κ 头 | 0.6383→0.5635（中止，崩坏中）|

### 核心结论
1. **最大的单项收益来自 ID token 进输入/注入路径**（v5c，test +1.1pt vs ctrl）——CoLLM 原始设计的回归
2. **互学习"传得多"有害、"选对时机传"有益**：全量互学习从零 ep0 0.6088/热启动缓降崩坏；门控版从零 +1.4pt
3. B9 专家注入从零 +1.0pt，与门控无叠加收益；B10 多层从零有害
4. 崩坏机理 = 过度对齐（v2 的 KL 置信度匹配 / v10 的全量互学习+κ 头）；解 = 门控 + 早停
5. 环境教训：共享 GPU（另一会话 15-18GB）需等待+重试脚本；磁盘纪律（checkpoint 即用即删）

## 方案 2/4 实验（2026-09-13，用户指定补齐未试方案）

### 方案 4：Disagreement 门控（已实现未跑 → 补跑）
- **配置**：scratch_gate_disagree.yaml（b3 hinge + `mutual_gate: "disagreement"`，|q_L−q_G|>δ=0.1 才互学）
- **轨迹**：0.6756→0.7067→0.7204→0.7159→0.7172→0.7145→0.7191→0.7172（ep2 后稳定 0.714-0.720，无崩坏）
- **test 0.7263** —— **新纪录**（微超 v12 的 0.7262，噪声边缘但至少无害）
- 注意：disagree 模式下 mLG/mGL=1.00 全开（q_L 与 q_G 高度相关但差值普遍 >0.1）——**δ=0.1 的带宽下 disagreement 门控实际上等价于全量互学习**，其增益来自 b3 hinge 配方本身

### 方案 2：Semantic Graph Refinement（新实现 + 实验）
- **实现**：`_refine_adjacency()` + compute_feedback_gcn 两遍 GNN——pass A（普通图）出 gate 上下文；门控 `γ_ij = σ(g(h_i+h_j, z_i+z_j))`（对称、h detach、bias=-2 边缘初始闭合）；边权 `w_ij = γ_ij·relu(cos(h_i,h_j))`（LLM 提供候选语义边）；加进子图邻接后 NGCF 内部度归一化让"图决定"边的实际参与度；eval 也跑 no-grad pass-1 保持 train/eval 图一致
- **结果**：valid best 0.7102@ep1，**test 0.7057**；轨迹波动大（ep2 深谷 0.6577、ep7 0.6473）
- **结论**：**证伪**——门控语义边改变了图结构但没带来提升，且训练稳定性反而下降（图结构随训练漂移更剧烈）

### 补齐后的完整终榜（test AUC）
（v13 轮补充：tca_full 16ep 延长 test = 0.7147——慢热曲线在 16ep 内未超越 rel_pref/gate_disagree 的 0.7263；双向形态的健康曲线已证实，但绝对指标未占优）
| 排名 | run | 方案 | test |
|---|---|---|---|
| 1 | **gate_disagree** | **B3 hinge + disagreement 门控** | **0.7263** |
| 2 | v12 | 从零 REL + pref 蒸馏 | 0.7262 |
| 3 | v5c（历史） | 热启动(强基线) + hinge | 0.7262 |
| 4 | ext_rel | 从零 REL | 0.7256 |
| 5 | rel_warm / rel_grad | 热启动 REL / 梯度投影 | 0.7230 / 0.7162 |
| 6 | rel_ema | EMA 教师 | 0.7154 |
| 7 | rel_refine | 语义图精化 | 0.7057 |
| 8 | ext_gate / ext_gate_ml | B11 门控(8ep) / 多层注入 | 0.7029 / — |

**用户全部方案的最终对账**：B9 ✅、B10 ✅、B11 ✅（threshold+disagreement 两模式）、B12 ⛔（显存不可行）、Relation-level ✅、TopK 邻域 ✅、Semantic Graph Refinement ✅、Cross-view 硬负例 ⛔（未实现，pref 用均匀负例）、Ranking-level G→L ✅（v12 pref）、EMA 教师 ✅、Gradient-aware ✅。**12/14 已试，2 个有明确不可行/未实现理由**。

## ReLLaX 式 CFLoRA 实验（2026-09-13，Graph-conditioned Fully-Interactive LoRA）
- **实现**：`injection_mode: "cflora"`——协同表示 h_G=[z_u; mean(z_hist); z_t]（768 维）→ projector（768→1024→64，末层零初始化）→ M_G ∈ R^(8×8)；LoRA 瓶颈变为 `r' = (I + M_G)·r`（残差形式，零初始化=启动时等价标准 LoRA）；替换原 8-d hidden 注入；梯度经 M_G 流回 GNN/MF（ReLLaX 式端到端）
- **V1（纯 CFLoRA，无互学习）**：valid best 0.7123@ep5，**低于 plain 0.7210 和 rel_pref 0.7232**——参数级融合单独使用无优势
- **V2（CFLoRA + TCA 双向互优化）**：valid best 0.7041@ep3，**test 0.6969**——**全项目最差**；CFLoRA 的 rank 交互调制与 TCA 式偏好蒸馏叠加后互相干扰
- **结论**：ReLLaX 的参数级融合在 GraphLoRA 上的两种组合（单独/叠加互学习）均未超过现有方案。**最稳定的配置仍是 v12（test 0.7262）和 gate_disagree（test 0.7263）**。可能原因：rank 交互矩阵的自由度（64 参数/样本）在从零训练的短预算内难以学到有意义的组合模式；且 M_G 的梯度路径与 LoRA B 的零初始化存在激活延迟耦合

### 实验对账更新
| 排名 | run | 方案 | test |
|---|---|---|---|
| 1 | gate_disagree | B3 hinge + disagreement 门控 | **0.7263** |
| 2 | v12 | REL + pref 蒸馏 | 0.7262 |
| 3 | plain 对照 | 纯 L_rec | 0.7237 |
| 4 | ext_rel | 纯 REL | 0.7256(valid)/0.7256 |
| 5 | V1 纯 CFLoRA | ReLLaX 参数融合 | 0.7123(valid) |
| 6 | V2 CFLoRA+TCA | 参数融合+双向互优化 | 0.6969 |
| run | 方案 | best valid | test |
|---|---|---|---|
### 稳定器实验（EMA / 梯度投影 / TopK，2026-09-12，全部无增益）
| rel_ema | EMA 动量教师（m=0.99，bank 语义记忆） | 0.7188@ep2 | 0.7154 |
| rel_grad | 梯度冲突投影（PCGrad 式） | 0.7202@ep2 | 0.7162 |
| rel_topk | TopK=5 局部邻域对齐 | 0.7188@ep2 | （未单独 eval，valid 已低于 v12）|
| **v12 对照** | **纯 REL + pref** | **0.7232** | **0.7262** |

三个稳定器全部低于 v12——针对"越训越差"的强化约束在 REL 上是负增益（REL 的缓降本就温和）。

### 调试中发现的梯度流现象（重要，留作后续改进）
- **现象**：L_S2G 对 NGCF 权重（W1/W2）的梯度 ≈ 3e-4 量级（近零），但对 MF 嵌入表的梯度 ~11（大）——有限差分证实 autograd 正确
- **机理**：token_gnn = `F.normalize(h)` 后再算 cos——梯度反传经过 normalize 的 1/‖h‖ 因子；同时 NGCF 的 `output = normalize(h)×0.1` 再注入，双重缩放
- **影响**：REL 的语义关系约束实际主要训练 **MF 嵌入表**（经 llama_proj 影响输入）而非 NGCF 权重——GNN 权重的学习几乎全部依赖 κ 头与注入回流
- **改进方向**：penult 去掉 normalize 或加可学习缩放；或者 REL 的视图直接用未归一化 h
- 调试过程还修掉一个回归：forward_v2 重构时 diag 的 detach loss 覆盖了带梯度的 ret["loss"]（导致 runner backward 报错），已加 guard

## REL 模式（CLLM4Rec 启发）实现与结果（2026-09-12）

### 方案（用户设计的 Semantic-Collaborative Mutual GraphLoRA 第一版）
- **不做 feature injection**（moving-target-chasing-moving-target 的教训），做 **relation-level mutual guidance**：
  - `mutual_mode: "rel"`：batch 内 item-item 关系矩阵对齐——`S_L = cos(h_i,h_j)`（LLM 第 8 层语义关系）教 `S_G = cos(z_i,z_j)`（GNN 协同关系），MSE + 教师 stop-grad
  - **置信度自适应加权**（CLLM4Rec λ_c 洞察）：`w_i = q_L(sample(i))`——LLM 对样本越有把握，语义信号对 GNN 越强
  - 第一版只开 S→G（λ_gnn_to_llm=0）；κ 头承担 L_GNN^rec（权重 0.1）
- 与已有 B3 的区别：B3 是 user→candidates 的**偏好关系**（user 中心），rel 是 item-item 的**语义结构关系**（结构迁移）
- Graph→LLM 方向仍由 GraphLoRA 注入承担（两个方向载体不对称：协同→语义用 LoRA adaptation，语义→协同用 relation constraint）

### 结果（从零，IDList prompt，lr 1e-4，batch 1）
| run | 轨迹 | best valid | test |
|---|---|---|---|
| **v11/ext_rel（8 epoch）** | 0.7113→0.7056→**0.7239**→0.7216→0.7158→0.7203→0.7078→0.6908 | **0.7239@ep2** | **0.7256** |
| v11 3ep | 0.7073→0.7117→0.7234 | 0.7234 | — |

**ext_rel test 0.7256 ≈ 历史最佳 v5c 热启动 0.7262，但无需热启动/基线预训练**——ep0 即 0.7073（从零史上最快起步，对照 0.6088）。ep2 达峰后缓降（与普遍模式一致，早停/门控仍是必要的）。ext_gate（8 epoch 对照）运行中。

### 8-epoch 延长训练终局对比（2026-09-12）
| run | valid 轨迹 | best valid | **test** |
|---|---|---|---|
| **ext_rel（v11, REL 模式）** | 0.7113→0.7056→**0.7239**→0.7216→0.7158→0.7203→0.7078→0.6908 | 0.7239@ep2 | **0.7256** |
| ext_gate（B11 门控） | 0.7007→0.7041→0.7111→0.7022→**0.7149**→0.6803→0.6859→0.6704 | 0.7149@ep4 | 0.7029 |

**终局结论：REL 模式是当前全项目最强方案**——test 0.7256 与热启动时代最佳 v5c（0.7262）持平，且从零训练即可达成（无需基线预训练/热启动）。8-epoch 长训下 REL 相对门控的优势拉大（0.7256 vs 0.7029）：门控几乎不激活（q_L/q_G 高度相关），长训中无持续正则收益；REL 的语义关系约束则持续提供温和的结构信息。两者 ep2-4 后均缓降——早停组件仍然必要。

### 热启动 REL 验证（2026-09-12，用户指令：预热或多训到收敛）
- 基线重训 12 epoch：best 0.7170@ep2（弱于被删的历史基线 ~0.7212@ep12），后期下滑至 0.676（典型基线过拟合）
- rel_warm（热启动 REL，lr 2e-5，8ep）：**ep0 即达峰 0.7213**（基线 0.7170 + 0.0043），随后单调缓降至 0.6979；**test 0.7230**
- **热启动没有超过从零**（0.7230 < 0.7256）：本轮基线偏弱 + REL 的收益在最初几步即兑现完毕。REL 方案的核心价值恰恰是**不依赖基线质量**——这简化了整个训练流程（跳过基线预训练阶段）

### 全项目 test AUC 终榜（2026-09-12 更新：v12 偏好级蒸馏）
| 排名 | run | 方案 | test |
|---|---|---|---|
| 1 | **v12** | **从零 REL + IDList + 偏好级蒸馏，8ep** | **0.7262** |
| 2 | v5c（历史） | 热启动(强基线) + IDList + hinge 互学习 | 0.7262 |
| 3 | ext_rel | 从零 REL + IDList，8ep | 0.7256 |
| 4 | rel_warm | 热启动(弱基线 0.7170) + REL | 0.7230 |
| 5 | v4 | 热启动 + hinge（旧 prompt） | 0.7164 |
| 6 | ctrl | 热启动 + 无互学习 | 0.7149 |
| 7 | ext_gate | 从零 B11 门控 8ep | 0.7029 |

### 最终冲刺：延长训练与候选数（2026-09-12 深夜，均无增益）
| run | 方案 | best valid@ep | 备注 |
|---|---|---|---|
| v12（8ep，K=2） | REL + pref 蒸馏 | **0.7232**@ep2 → test **0.7262** | **最优预算与配置** |
| rel_pref_12ep | 同 v12，拉长至 12 epoch | 0.7202@ep2，ep10-11 跌至 0.64-0.65 | 拉长 cosine 使 ep2 后劣化更快，**延长有害** |
| rel_pref_k3 | 候选 K=2→3 | 0.7217@ep2 | 无增益（均匀负例的信息量未增加）|

**最终结论**：v12 配置（从零 REL + pref 蒸馏 + IDList，lr 1e-4，batch 1，8 epoch，K=2）是本项目的可复现最优——test 0.7262。延长训练、增加候选、三个稳定器（EMA/梯度投影/TopK）、推理融合、门控、多层注入全部试过均无增益。峰值稳定出现在 ep2（patience 2-3 的早停即为正确部署方式）。

## 重大修复与真实消融（2026-09-13，改变结论的一轮）

### 发现 ret["loss"] 重绑定 bug 及其修正
- forward_v2 中 `ret = {"loss": loss}` 在辅助损失累加**之前**绑定，后续 `loss = loss + λ·L_aux` 只重绑局部变量——**返回的 loss 只含 L_rec，互学习损失从未进 backward**
- 修复：training 块末尾显式 `ret["loss"] = loss` 重绑定（grad_projection 的 trigger 覆盖保留在其后）
- 修正一个错误推断：修复后 rel_pref 重跑与修复前 v12 轨迹逐位相同 → 证明历史 run 的互学习损失**确实进了 backward**（diag 覆盖问题只存在于 TCA 接线轮的短暂窗口，影响的是 ema/grad/topk 三个 run 的首次尝试）
- 训练确定性证实：同配置重跑 AUC 逐位相同

### 修复后三方真实消融（8 epoch，lr 1e-4，IDList，同预算）
| run | valid 轨迹 | best valid | test |
|---|---|---|---|
| rel_pref（REL+pref KL） | 0.7016→0.7024→**0.7232**→0.7194→0.7198→0.7103→0.7193→0.7192 | 0.7232@ep2 | **0.7262** |
| tca_full（REL+TCA 双向） | 16ep：0.691→0.699→0.718→...→**0.7205**@ep13→0.706 | 0.7205@ep13 | 0.7147 |
| plain（纯 L_rec 对照） | 0.7017→0.7061→**0.7210**→0.7161→0.7162→0.7152→0.6947→0.6925 | 0.7210@ep2 | 0.7237 |

（plain 的 test 0.7237 已补测完成）

**关键修正后的结论**：
1. **互学习的真实增益 = rel_pref/plain = +0.25pt**（修复后才是真实的互学习对比）
2. **两种互学习形态的收敛曲线截然不同**：rel_pref"早熟"（ep2 峰值后震荡）；tca_full"慢热持续上升"（ep7 未饱和）——TCA 式软目标的梯度性质（依赖 LLM 当前预测）避免了早熟峰值
3. tca_full 16-epoch 延长跑进行中（验证上升曲线的上限）

### v12：偏好级蒸馏（GNN 偏好分布 → LLM Yes 概率，2026-09-12）
- **机制**：每样本扩展为候选组（真实 target + 均匀负例，`PrefCandidateDataset` 包装器）；GNN 教师分布 `P_G = softmax(κ·cos(z_u,z_c)/τ_G)`（stop-grad）蒸馏进 LLM 的 `P_L = softmax(yes_logit/τ_L)`，按 GNN 置信度 q_G 加权——**协同空间认为更匹配的物品，其 Yes 概率被训练直接抬高**（偏好级 G→L，与 REL 的结构级 S→G 正交互补）
- **结果**：valid best 0.7232@ep2（≈ ext_rel 0.7239），**test 0.7262**——追平历史最佳且从零可复现、无基线依赖；蒸馏项无害微正
- **推理级融合对照（同一 ckpt）**：β=0.25 → 0.7186，β=0.5 → 0.6878——**训练好的强模型上融合同样有害**（ep0 弱模型上 β 也单调恶化：0.6988→0.6702→0.6302）。两个层次的一致结论：协同信息必须通过训练进入 LLM 的打分函数，推理期集成（弱 GNN 评分器 + 强 LLM）必然稀释
- 实现位置：`pref_candidates.py`（包装器）、`_pref_distill_loss()`、`_row_gnn_scores()`、eval 的 `eval_fuse_beta` 开关；配置 `collm_pretrain_mf_ood_scratch_rel_pref.yaml`

### v5c 为什么有效（对比 ctrl 的归因逻辑）
- ctrl（无 mutual，旧 prompt，lr 2e-5）：0.7197 起单调下降
- v5c（IDList + mutual rank，lr 2e-5）：同起点水平（0.7184），ep1 起上升至 0.7227 平台，退化推迟 2 epoch
- 差异变量只有两个：①LLM 输入多了历史 ID token（信息增益）；②mutual 候选从 K=2 → K=11/样本（hinge 有真实信号可传）。两者都指向"ID token 进入输入/注入路径"这一方案变更

### 关键旁证（另一会话的 b5 run，2026-09-06）
- `output/graphlora_ml1m_b5_lowlr_debug/20260906095/`：B3-Pred + rel_temp 0.1 + λ=0.05/0.2 + **lr 2e-5** + 300 iters/epoch，5 epoch valid AUC **0.7215/0.7224/0.7204/0.7186/0.7197 全程稳定**（best ep1 0.7224）
- 结论：低 lr 下热启动训练不崩；支持 v4 的 lr 假设

### b3_v2 完整轨迹（output/graphlora_ml1m_b3/20260906013/）
- checkpoint_best.pth = epoch 1（valid AUC 0.7079, uAUC 见 log）；另有 checkpoint_{0..3}.pth 快照
- valid AUC: 0.7071 / 0.7079 / 0.6777 / 0.6555 (epoch 0/1/2/3)
- mutual KL: L→G 1.11→0.33→0.26, G→L 1.75→0.57→0.44（单调下降，无反弹）

### 跑全量数据前的建议（按优先级）
1. **限制 G→L 方向强度或延迟启用**：λ_G→L 降到 0.005，或给 mutual loss 加 warmup（前 1-2 epoch 为 0，再线性升到目标 λ）——epoch 0-1 证明温和对齐有效（+0.008），崩坏发生在 KL 深度压低之后
2. **lr 降到 5e-5**（biloop_ft 同款温和微调思路），热启动下 1e-4 偏大
3. **早停耐心调小**（patience 2）：本方法的有效窗口看起来就在前 1-2 epoch
4. 对照试验：λ_L→G 单独开（0.01）、λ_G→L=0，验证是 G→L 方向导致崩坏
5. 备选：跑 B2（tran_configs/collm_pretrain_mf_ood_b2.yaml，代码已就绪）做方案间对比

## 诊断数据（diag_cos_gap.log, 2026-09-06）
- LLM 视图 top1-top2 cos gap: mean=0.0449, p50=0.0053, p90=0.1228
- GNN 视图 top1-top2 cos gap: mean=0.0294, p50=0.0206, p90=0.0606

## 参考基线
- GraphLoRA 基线（output/graphlora_ml1m/20260903095/checkpoint_best.pth）: valid AUC ≈ 0.6995
- BiLoop ctrl: valid AUC ≈ 0.6848（未超过基线）

## 运行命令备忘
```
# 监控
nohup bash gpu_monitor.sh <tag> &
# 训练
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ~/anaconda3/envs/minigpt4/bin/python \
  train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3.yaml \
  > graphlora_b3_<tag>.log 2>&1 &
# 调参覆盖
--options lambda_llm_to_gnn=... rel_temp_llm=... batch_size_train=...
```

## 仓库清理（2026-10-01，用户指令：代码回溯到净化版，只保留最优结果）
- `minigpt4rec_v2.py` 替换为用户提供的净化版（保留 GraphLoRA 核心 + 语义对齐 MSE + pref_distill 机制 + 单层注入；删除 rank/TopK/手术/集成/门控/Q-V/sem_detach/两阶段/层参数化等全部实验期代码）。旧版备份：/tmp/minigpt4rec_v2_pre_cleanup_20260930.py
- base_task.py / runner_base.py / optims.py 回退到会话前状态（移除手术分支、dict 分支、rec_lr_scale、lr_scale）
- 实验期追加的全部变体配置已删除；保留：scratch_rlm.yaml（冠军 A 配置）、eval_base.yaml（A 的 test 复现配置）、上游两个原始配置
- output/ 只保留最优结果：graphlora_ml1m_semantic/20260924231/checkpoint_best.pth（A，valid 0.7291 / **test 0.7378**，全项目最高）
- 清理前共 45 个变体实验、6 次位级验证、完整响应曲线，全部结论记录于本文件上方

### Soup 实验（2026-10-02，计划一）：A+I3 权重平均 = 负结果
- A(gs01) 与 I3 重训(gs02，三轮逐位复现 0.7326/0.7290/0.7193) 的 best checkpoint 逐 tensor 平均后 test 评估：
  - gs=0.1 → **0.6675**；gs=0.15 → 0.6675（完全相同——soup 权重对 gnn_scale 不敏感）
- 结论：两个冠军只在 gnn_scale 上不同（0.1/0.2），权重空间中点落在针尖状响应的峰外。**跨 gnn_scale 的 soup 有害**，与全部刀锋边缘发现一致。soup 路线关闭。

## RNG 泄漏根因发现 + 锚定修正首测（2026-10-02～03，用户方案二实施）
### 根因链（"塌缩抽签"的机制定位）
1. 新模块创建消耗全局 CPU RNG → llama_proj 等后续初始化全变（init lottery）；
2. fork_rng 补丁的 torch.manual_seed 同时重置了 CUDA RNG，而 fork_rng(devices=[]) 只恢复 CPU → **所有训练期 dropout mask 改变**（draw lottery）；
3. 修复：fork 内用 torch.random.default_generator.manual_seed（CPU-only），CUDA 流完全不动。
### R0S-v3 = 方案二最小剂量首测（rho=0.05, w=0.04, adapter 零初始化）
- ep0 = **0.7295**（I3 0.7326 / A 0.7291 —— 持平，健康盆地）✓ 方案二最小剂量无害且在场
- 结构对照（rho=0 纯惰性）被本 run 取代——它同时回答了"最温和修正是否安全"
- 待 ep1/ep2：锚定修正是否消除 ep2 衰减（I3 ep2 0.7193）

### R0S-v3 终局（2026-10-03）
- 三轮：0.7295 / 0.7179 / 0.7011 —— 每轮略低于 I3（0.7326/0.7290/0.7193）0.3~1.8pt，**无塌缩、无衰减加剧**
- 结论：方案二（锚定目标 ρ=0.05 + 补偿权重 0.04 + Adapter 零初始化）在最小剂量下**无害但无增益**
- 三保险丝验证：α=0 路径恒等 ✓、ρ 缩放语义拉力 ✓、Adapter 零初始化起点=冠军 ✓（R0S-v3 即带电运行的最温和形式）
- RNG 泄漏修复后，"新模块 = 新抽签"的 init lottery 已消除（fork_rng + CPU-only 播种）——后续变体对比将纯净化

## A0/A1 接线 + 位级验证 + A0 诊断 run 启动（2026-10-04，文献报告路线落地）

### 背景与定级
文献报告的 A/B/C 三方案按 journal 既有证据重新定级：**B（PCGrad）正式关闭**（E5 已证明 ||g_sem||/||g_rec||=0.03-0.08、投影修正 ~1e-8，冲突机制在本剂量下不绑定；rel_grad 0.7162<v12 0.7262 佐证）；**C（CF soft supervision）基本关闭**（pref_distill 已做=test 0.7262 无害微正；late-fusion screening 已做=有害）。执行路线：**A0 注入诊断 → A1 条件式 warm-up → teacher 置信度诊断 → confidence-weighted MSE**；频谱分析并行。

### 代码接线（minigpt4/models/minigpt4rec_v2.py，全部配置门控、默认关闭）
- **A0 注入诊断**（`inj_diag: True` 开启）：在 MyLinear 注入点（LoRA-B 之前）记录每步 a=LoRA-A 输出、g=gnn_scale×warmup_factor×z_G，输出窗口汇总（每 100 步）：`R_med/R_p90`（=‖g‖/‖a‖）、`C_med/C_p10`（=cos(a,g)）、|a|、|g|，q_proj/v_proj 分开。纯 no-grad 统计，不碰训练图、不消耗 RNG。
- **A1 warm-up**（`gnn_injection_warmup_ratio: 0.1` + `gnn_injection_warmup_total_steps: 33891` 开启）：训练时有效 gnn_scale = β·min(1, t/(ratio·T))，t 取 fb["step"]（0-based 全局步）；**只在训练路径生效，eval 恒 factor=1.0**（评估协议与 I3 完全一致）；factor==1.0 时走原表达式逐位等价；无任何新参数/新 RNG 消耗。
- 修复口径 bug：diag 初版漏乘 β，导致 A1 下 R 轨迹会失真——已改为乘有效 scale。
- 新配置：`..._rlm_A0_diag.yaml`（I3+diag）、`..._rlm_A1_warmup.yaml`（I3+warmup10%+diag）。
- 新离线脚本（零训练成本）：`diag_teacher_confidence.py`（层 6/8/10 关系方差 vs |S8−S_G| 分桶，需 GPU 加载 7B，待 A0 结束后跑）、`diag_gnn_spectrum.py`（checkpoint→NGCF 全图 penult 谱，CPU/小显存可跑）。

### 位级验证（三个冒烟，全部通过）
| 冒烟 | 结果 |
|---|---|
| 新代码 + I3 原配置（全关） | 前 350 步窗口 loss 与 rlm_I3_retrain_soup.log **逐位一致**（1.0524/1.3734/1.4467/0.5738/1.1173/0.5585/0.9879）|
| A0 配置（diag 开） | loss 仍逐位一致 → 诊断无扰动 |
| A1 配置（warmup 开） | `[A1] warmup_steps=3389` 横幅出现；首 100 步 |g|=0.0003（斜坡生效）|

### A0 run（进行中）+ 频谱基线
- `a0_diag_run.log`：I3+被动日志，3 epoch。它的 valid 轨迹应逐位复现 I3（0.7326/0.7290/0.7193），并充当 A1 的 matched control。
- 早期读数（前 400 步）：R_med ≈ 0.03 稳定无 early spike；q 路径 C_med ∈ [-0.13, 0]（方向杂/略反），v 路径 C_med ≈ +0.33~+0.48（同向）——q/v 行为确有分化。
- **频谱基线（A 冠军 checkpoint，diag_gnn_spectrum.py）**：MF 表（GNN 前）有效秩 ≈45、σ1/Σσ=0.047，训练前后谱几乎不变；Z_G penult 有效秩 **18.7**、PR 4.73、σ1/Σσ=0.079、top-50 能量 0.587 —— 图传播显著低频化，冠军表示的频谱指纹留档。待有塌缩 run checkpoint 时对比"强对齐→谱塌缩"假说。

### 判决规则（A0 结束后）
- R(t) 全程平稳（图 B）→ warm-up 机制依据不存在，A1 关闭，转 teacher 置信度诊断；
- R early spike（图 A）→ 启动 A1 matched pair（A0 run 即 control，只差 β(t)）；
- R 递增（图 C）→ 问题不在早期，同样关闭。

### A0 终局（2026-10-04 下午，判决完成）
- **valid 轨迹逐位复现 I3：0.73262 / 0.72904 / 0.71931** —— diag 零扰动再次确认，本 run 同时是合格的 matched control。
- **判决：图 C（递增），A1 warm-up 正式关闭**。early/late R_med 比值：q 路径 0.76、v 路径 0.17——注入比随训练**上升**而非早期过高，warm-up 要防御的"early mismatch"在本设置中不存在。
- **机制发现（论文 analysis 素材）**：后期 R 上升的驱动是 **LoRA-A 输出萎缩**（v 路径 |a| 从 ~0.74 塌到 ~0.16-0.18，ep1-2），而 |g| 恒 0.02；v 路径 R_med 涨到 ~0.43、C_med≈-0.48（注入方向与 LoRA 方向反向）。ep2 数据级过拟合期，协同分量在 bottleneck 中的**相对**占比被动放大且方向渐反——与 G 系列"ep2 衰减是数据级过拟合"的结论互洽。
- q/v 分化全程存在：q 路径 R_med ~0.05、C_med 全程 -0.2~-0.3（方向近乎正交略反）；v 路径早期 C_med>0（+0.33）后期转负。
- A0 checkpoint 已按纪律删除（与 I3 逐位同权重，无保留价值）。
- 待办转移：teacher 置信度诊断（diag_confidence_A.log，对 A 冠军、valid 400 batch）运行中 → 决定 confidence-weighted MSE 是否立项。

## 置信度诊断结果 + CW1 立项（2026-10-04 下午，A0 判决后）

### teacher 置信度诊断（diag_teacher_confidence.py，A 冠军，valid 400 batch = 1,315,457 个 item pair）
- 层间方差 var_L = Var_l(S_l)，l∈{6,8,10}；对齐误差 err = |S8 − S_G|（Layer8 目标 vs GNN 当前关系）。
- **方向支持假说但效应弱**：err 按 var 三分位：low 0.0697 / mid 0.0615 / high 0.0761；极端十分位比 bottom10% 0.0727 vs top10% 0.0889 = **1.22×**。pearson(var, err)=0.12、spearman=0.056，且 mid 桶低于 low（非单调）。
- consensus 目标与 S8 的差异集中在高 var pair：spread（max−min 层间极差）low 0.013 / high 0.101 —— consensus 教师恰好在"当前监督最差"的 pair 上给出实质不同的目标。
- τ=1e-3 时权重温和（w p10=0.03 / p50=0.70 / p90=0.98，有效样本比 0.60）。
- 数据存档：diag_confidence_A.npz。

### 判决与立项
按报告十三的规则（high uncertainty ⇒ larger semantic error 则值得训练），条件弱成立 → **CW1 立项**：confidence-weighted consensus teacher，**一个 matched pair**（control = A0 run，已逐位验证）。

### CW1 终局（2026-10-04 傍晚）：塌缩，提前击杀
- **ep0 = 0.7145（Δ-1.81pt vs control 0.7326）、ep1 = 0.6563（Δ-7.3pt）**——ep1 明确塌入坏盆地（ep2 开局 train loss 0.0069，yes-logit 饱和），按判决表提前击杀，checkpoint 已删。
- **归因**：consensus 目标（层 6/8/10 均值）把监督从层 8 针尖拉走 = **K5 的机制以更隐蔽的形式复发**（多层集成稀释层 8）。诊断只证明了"高方差 pair 误差更大"，没证明"共识目标是更好的教师"——立项时对 K5 先验加权不足，教训记下。
- **副产品：频谱对比反转了直觉**（diag_gnn_spectrum.py，champion vs CW1 塌缩 checkpoint_1）：
  - A 冠军 Z_G：有效秩 **18.7**、PR 4.7、σ1/Σσ 0.079（强低通、集中）；
  - CW1 塌缩 Z_G：有效秩 **40.0**、PR 12.7、σ1/Σσ 0.051——**退火回 MF 表形状**（原始表 ~45）；
  - 解释：健康 NGCF 做强图平滑（低频集中）；错误语义目标破坏 NGCF 权重学习后，Z_G 失去图平滑、退化为"近 MF 投影"。即本设置下**塌缩的谱特征是"图平滑消失"而非"过度集中"**，与 UniSpecRec 式"过强对齐→谱塌缩"的预期方向相反（caveat：n=1 塌缩 run）。论文 analysis 素材。
- 代码后续：`sem_consensus_target: "mean"|"s8"` 键加入（默认 mean 保持 CW1 语义）。

### CW2 终局（2026-10-04 晚）：同样塌缩，CW 家族整体归档
- **ep0 = 0.6191（Δ-11.3pt）**，ep1 开局 train loss 0.0067（饱和塌缩）——比 CW1 更差，按纪律击杀，checkpoint 已删。
- **意外归因反转**：CW2（目标=层 8 不动，仅加权）比 CW1（均值目标+加权）更差 → **置信度加权本身即是主要伤害源**，不只是目标偏移。可能机制：低 var pair 多为"层间一致的平凡对"，其残差本就接近 0，加权后有效梯度集中在这些近零残差上，同时剥夺了高 var pair（GNN 真正需要移动的方向）的监督——但此解释未做专项验证。
- **draw-lottery caveat（诚实记录）**：本 regime 任何损失形态扰动的塌缩率 ~50-60%，CW1/CW2 两次连续塌缩在纯抽签假设下联合概率 ~0.25-0.36，不能完全排除"两次都是坏 draw"。但区分机制/抽签需要 2 个额外 draw（~7 GPU-h），而该家族的最好情形也仅是 TopK 级中性（历史 0.726-0.728）——**信息增益为负，不再投入**。CW 归档判词：置信度信号真实存在（1.22×），但在此剂量与训练形态下不可利用。
- **CW 家族终榜**：CW1（consensus target）ep0 0.7145 / ep1 0.6563 ✂；CW2（weight-only）ep0 0.6191 ✂。对照 A0=I3 0.7326/0.7290。

### 2026-10-04 全天总结：报告路线全部走完
| 路线 | 判决 | 证据 |
|---|---|---|
| A0 注入诊断 | ✅ 完成 | 逐位复现 I3，获得永久 matched control |
| A1 warm-up | ❌ 关闭（诊断判决） | 图 C 递增（q 0.76 / v 0.17），无 early mismatch；驱动是 LoRA-A 萎缩 |
| B PCGrad | ❌ 关闭（历史证据） | E5：g_sem/g_rec=0.03-0.08，修正量 1e-8；rel_grad 负 |
| C CF soft supervision | ❌ 关闭（历史证据） | pref_distill 已做=中性；fusion screening=有害 |
| 频谱分析 | ✅ 完成（analysis 素材） | 冠军=强低通（秩 18.7）；塌缩=退火回 MF 形状（秩 40，n=1） |
| Teacher 置信度 | ❌ 归档（CW1/CW2 塌缩） | 信号真实（1.22×）但不可利用 |
| **今日正收益** | matched-control 体系落地 + 3 个位级验证 + 冠军/塌缩频谱指纹 + 注入动力学数据 | 全部记录于本文件 |

**下一阶段待用户决策的方向**（均在报告"后备主线"或更远）：
1. C2F/AlphaLoRA 式容量分配（报告定为中高风险后备主线，违背"停止结构搜索"原则，需明确批准）；
2. champion 的 5-draw 方差测量（需换 seed 43/44，等待用户批准）；
3. 论文 analysis 章节整合（频谱指纹、注入动力学、梯度哑弹 E5、CW 负结果——均为零训练成本写作素材）。

## 2026-10-04 晚：用户批复新路线 —— seed 地基 + 容量重分配（RC 系列）启动

### 用户决策（第二轮报告批复）
- **批准换 seed**：champion 补 seed 43-46（42 复用 A0 逐位结果）；所有新方案主统计量改为 matched-seed Δ_s 均值。
- **CW 判决措辞修正**（采纳）：论文写 "relation confidence ≠ training utility"，不写"已证明置信度加权有害"（两个 treatment 非 repeated draw，p² 推断无效；历史塌缩率混杂 RNG 泄漏期，不可沿用）。
- **唯一批准的新训练结构**：预算中性 LoRA rank 重分配 [8,8,8,8]→[6,8,10,8]，r₃₁=8 永不动，α/r 恒定，matched-seed Δ≥+0.003 才扩展。
- 若该路线 matched seeds 下也不成立 → 停止结构搜索，转论文写作。

### RC 代码接线（minigpt4rec_v2.py）
- `lora_rank_groups: [[0,7,r],[8,15,r],[16,23,r],[24,31,r]]`（闭区间）；组 rank=8 的模块完全不触碰 → uniform 分组逐位等价；注入层 gcn_layer 强制不重排（其 bottleneck 即 GraphLoRA 协同投影通道）；scaling["default"]=2·r_new 保持 α/r=2 恒定（基线 alpha=16/r=8 硬编码比率）；重排模块 A kaiming / B 零初始化（dW=0 起步），init 包裹 fork_rng+CPU-only 播种（RNG 泄漏教训）。
- 配置：`..._RC0_uniform.yaml`（位级对照）、`..._RC1_rank68108.yaml`（treatment）、`..._champ_seed43/44/45/46.yaml`（仅 seed+output_dir 不同）。

### 位级/功能验证
| 冒烟 | 结果 |
|---|---|
| RC0 uniform [8,8,8,8] | re-ranked 0 modules；前 150 步 loss 与 I3 **逐位一致**（1.0524/1.3734/1.4467）|
| RC1 [6,8,10,8] | re-ranked 32 modules（L0-7、L16-23 × q/v）；total=256；训练正常，loss 轨迹如预期偏离 |
| （调试记）| 首版正则漏了 `self_attn.` 段致 0 匹配，已修 |

### Champion 多 seed 地基（2026-10-05，重大发现：champion 本身是双峰抽签）
- 五 seed valid AUC 轨迹（配置=I3 逐字节相同，仅 seed 不同，代码=RNG 修复后）：
  | seed | ep0 | ep1 | ep2 | 状态 |
  |---|---|---|---|---|
  | 42（=A0） | 0.7326 | 0.7290 | 0.7193 | 健康（test 0.7378）|
  | 43 | 0.6239 | 0.6245 | 0.6002 | **塌缩**（uauc 0.569-0.577）|
  | 44 | 0.6551 | 0.6136 | 0.5976 | **塌缩**（uauc 0.562-0.606）|
  | 45 | 0.7247 | 0.7161 | 0.7140 | 健康 |
  | 46 | 0.7243 | (运行中) | — | 目前健康 |
- **核心结论：RNG 修复没有消除塌缩抽签**。champion 配方本身 p(塌缩)≈2/5~3/5，与历史 50-60% 一致——塌缩是训练动力学内禀属性（数据顺序/dropout 抽签），不是结构扰动的产物。valid best 分布双峰：健康盆地 0.7247-0.7326，塌缩盆地 0.5976-0.6551，中间无人。
- **三层重估**：
  1. **champion 0.7326/0.7378 是幸运尾部**，不是配方的稳定属性；论文主结果必须报分布（健康盆地均值±SD + 塌缩概率），单 seed 数字不可再引用。
  2. **CW1/CW2 的塌缩失去机制归因**：base 配方本身以 ~0.5 概率塌缩，两个 treatment 各一次塌缩与抽签假设完全相容。CW 关闭维持（期望收益 TopK 级中性 + 两次 draw 都在塌缩盆地），但论文表述必须写"未能证明其有效"，不能写"证明其有害"。
  3. **RC1 单 matched pair 判决协议作废**：p(塌缩)≈0.5 下单对比较是掷硬币。改为：RC1 与 control 各跑 ≥3 个 seed 的 matched draws，**以"健康盆地条件下的 Δ"为主统计量**（双方各自条件于存活），或报 5-seed Δ_s 全分布。
- seed 46 完成后：出五 seed 正式统计表 + UAUC + best-valid→test 协议执行（塌缩 seed 的 test 是否评估待定）。

## 2026-10-05：seed 工作终止 + L0-7 零梯度根因破案 + FX 方案启动

### 用户指令
**终止全部 seed 相关实验**（含运行中的 seed46 与计划的 RC1 多 seed 协议）。此后只用 seed 42 位级复现做对照。

### 根因链：层 0-7 LoRA 梯度精确零 = autocast 缓存中毒（真 bug）
探针逐层排除（probe_lora_grad.py / _grad2.py + 三轮 hook 实验）：
1. 前向连通：扰动 B(L0) 改变 loss（Δ=3.3e-3）；grad(loss→L0 output)=1.26 非零；llama_proj 梯度非零。
2. 但 L0-7 的 A/B 权重 **UNUSED — NOT IN GRAPH**（autograd.grad allow_unused 直接判定）；dA=0 可由 B=0 解释，dB=0 无法用链式法则解释（dL/dy=0.126、z=|A·x|=1.45 均非零）。
3. **根因**：语义 teacher pass 在 `torch.no_grad()` + autocast 下先跑 **层 0-7**（exit_layer=8），把这些层的 LoRA 权重以无梯度方式 cast 成 fp16 并写入 **autocast 权重缓存**；主前向处于同一个外层 autocast 区域（maybe_autocast 默认 cache_enabled=True），直接复用缓存的无梯度副本 → 层 0-7 的 adapter 永久断开 autograd 图。**层 8-31 未被 teacher 访问，首次 cast 发生在主前向中，连接正常——精确的 L8 边界 = teacher 的 exit_layer**。
4. 影响面：所有 `gnn_semantic_align: True` 的 run（A/I3/全部冠军变体）浅层 1/4 LoRA 从未训练；修复 = teacher pass 改用 `autocast(cache_enabled=False)`（FX），flag 关闭时行为逐位不变。

### FX1 方案（运行中）+ 判决
- **FX1 = 冠军 + fix_autocast_cache: True**：恢复全部 32 层 adapter 训练（预算中性、不动任何已验证机制——修 bug 而非加机制）。
- 探针验证：flag on 后 L0-7 **dB 恢复非零**（1.1e-2~2.8e-2）；dA 第一步仍为 0（dA=Bᵀ·dy，B 从 0 起步）但首步优化即推动 B，第二步起 A/B 双双激活。
- flag-off 位级冒烟：150 步 loss 与 I3 逐位一致（1.0524/1.3734/1.4467）。
- 对照 = I3/A0（位级）。GPU 队列：FX1 → RC1 串行（约 7h）。
- **附带重估**：①"champion 只训了 3/4 容量"解释了容量诊断的浅层死区，RC1 的"L0-7 降 rank"确认为空操作、RC1 实际效应 = L16-23 容量+25%；②塌缩抽签（43/44）可能与"浅层 adapter 恒零 + 8 层起活跃"的非常规训练动力学有关——FX 修复后塌缩率是否变化本身就是信息；③若 FX 正向，此前所有负结果（多层注入/关系注入/CW）都值得在"全容量训练"下重测——它们当年全部运行在残缺容量上。

### FX1 终局（2026-10-05 上午）：冠军配方下塌缩，击杀
- **ep0 = 0.6424（Δ-9.0pt vs 0.7326）**，valid loss 0.897、ep1 开局 train loss 0.0665（饱和轨迹）→ 按纪律击杀，checkpoint 已删。
- **解读（重要，勿过度引申）**：修复恢复了全部 32 层 adapter 训练，但冠军配方（lr 1e-4、batch 3 等）是**在被 bug 截断的 3/4 容量上调好的**——新激活的浅层容量以同一剂量全速训练破坏了原有平衡。单次塌缩 ≠ "修复有害"：塌缩抽签本身 ~0.5 基率，且 0.6424 落在塌缩盆地。准确表述是"**全容量训练与冠军配方不兼容**，需要配方协同调整"。
- FX 后续候选（未立项，待用户定）：FX2 = 修复 + 新激活层差分学习率（浅层 adapter 低 lr 缓启动，保住已调平衡）。需要重建 param-group 基础设施（清理时已删 rec_lr_scale）。
### RC1 终局（2026-10-05 中午）：塌缩，profile 关闭
- **ep0 = 0.6446（Δ-8.8pt）**，ep1 开局 train loss 0.1213（饱和）→ 击杀，checkpoint 已删。[6,8,10,8] 预算中性重分配在 seed 42 判决为负，profile 关闭（按报告协议 Δ<0 即关）。

### FX2 立项与实现（2026-10-05 中午，FX 线延续）
- **机制假设**：FX1 的失败不是"修复有害"而是"配方失配"——新激活的浅层 adapter 以冠军剂量（1e-4）全速训练破坏了在 3/4 容量上调好的平衡。FX2 = 修复 + 浅层温和激活（lr×0.1），保住已调平衡的同时让新容量以慢速参与。
- **实现**：optims.py 三个 schedule 函数支持 per-group `lr_scale` 乘子（无该键逐位等价）；runner_base.py optimizer 构建按 `run.shallow_lora_lr_scale` + `run.shallow_lora_layers`（如 "0-7"）把浅层 LoRA 参数分入独立组（乘子组附加在基础组之后，lr 日志取 param_groups[0] 不变）。默认配置完全走原路径。
- **双冒烟**：默认路径 150 步逐位一致；FX2 路径 [FX] banner + [FX2] 分组 32 参数 + loss 偏离 I3（机制生效）。
### FX2 ep0（2026-10-05 下午）：健康盆地，跑满中
- **ep0 = 0.7244（Δ-0.82pt），valid loss 0.6646**——温和激活把 FX1 的塌缩（0.6424）拉回健康盆地，**8.2pt 恢复证实"配方失配"假设**：问题不在修复本身，在新增容量的优化剂量。
- 判决表：低 0.3-1pt → 必须跑满，不早停。看点 = ep1/ep2 是否出现慢容量后期收益（对照是 ep0 峰值后单调衰减）。

### FX3 立项（用户批复，LoRA+ 式 A/B 非对称 lr）
- **机制**：FX2 把 A/B 一起压到 0.1× 可能过度——B（低秩特征→输出映射）是产生 LoRA output 的主动项，A（特征子空间）应缓变。ICML'24 LoRA+（η_B>η_C）与 LoRA asymmetry（B 比 A 重要）支持拆分。
- **FX3 配置**：L0-7 浅层 **A 0.1× / B 0.3×**，其余 = FX2（fix_autocast_cache + 其余全 1×）。不用教科书 (0.1, 1.6)——必须保护 Layer31 GraphLoRA 平衡，0.3 是边界内的第一枪。
- **实现**：runner 分组重构为 per-branch（A/B 独立 param group + 独立 lr_scale），`shallow_lora_A_lr_scale`/`shallow_lora_B_lr_scale` 覆盖 `shallow_lora_lr_scale`；无键时路径不变。FX2 的 both-key 语义向后兼容（A=B=0.1）。
- **判决表（用户定，以超 I3 为目标）**：ep0 <0.720 杀；0.720-0.724 不如 FX2 杀；0.724-0.730 最多看 ep1；0.730-0.733 强信号跑满；>0.733 完整跑；>0.7356 值得围绕它继续优化。
- **FX2 三分支预案**（用户定）：FX2 上升（0.729+→0.732+）= 容量需要时间，FX3 优先级升；平行衰减 = 0.1× 只是延缓破坏，FX3 仍打但判决从严；FX2 后期塌回 0.64 = 放弃 FX3 改 delayed activation。
- **rank profile（RC 系列）冻结**：用户明确不再动 rank allocation——线索在 optimizer dynamics，不在函数空间。
### FX2 终局（2026-10-05 下午）：第三分支确认，FX3 接力链招回
- **0.7244 / 0.6430 / 0.6106**——ep1 塌入坏盆地（eval loss 1.03→1.21，标准饱和形态）。**0.1× 温和激活只把不稳定推迟了一个 epoch，没有消除它**。按用户预设三分支：放弃 FX3（配置保留，未跑），改 **FX4 = delayed activation**。
- FX 系列教训统一：新增浅层容量的破坏是**累积性**的——全速（FX1）ep0 即塌，0.1×（FX2）撑过 ep0 后塌。剂量只改变跨越阈值的时刻。

### FX4：delayed activation（实现 + 运行中）
- **设计**：浅层 L0-7 adapter 在 ep0 完全冻结（lr_scale=0），**ep1 起（global step 11297）以 0.1× 激活**。关键性质：B 恒 0 + lr 0 → 浅层参数全程不动 → **ep0 逐位等于 I3（0.7326 保底）**，ep1/ep2 的任何 ΔAUC 干净归因于容量激活本身（AdamW m/v 在 ep0 累积浅层梯度也属于机制的一部分）。
- **实现**：optims.py 增加 `_effective_scale`（支持 per-group `lr_scale_start_step`，全局步前乘子=0）；LinearWarmupCosine 的 warmup/cosine 路径都传 total_step；runner 分组读取 `run.shallow_lora_lr_start_step`。单元测试：step<10 时 0、step≥10 后 0.1×cosine ✓。
- **冒烟**：150 步 loss 与 I3 逐位一致（1.0524/1.3734/1.4467）✓。
- **判决规则**：ep0 应=0.7326（位级验证项）；**ep1 ≥0.7290（I3 ep1）且 ep2 ≥0.7193** → 浅层容量对晚期阶段有正贡献，FX 线复活（下一步剂量轴：激活时刻/剂量）；ep1 明显低于 0.7290 → 容量在已调平衡的晚期阶段仍是净干扰，**FX 线整体关闭**（bug 修复本身保留为 flag，默认关）。
- FX4 运行中（fx4_run.log），ep1 结果约 2.3h 后出。

### FX4 ep1 判决 + FX5 立项（2026-10-05 傍晚，用户指令：转向改进）
- **FX4 ep0 = 0.7326（位级复现 I3 ✓，延迟激活实现正确）；ep1 = 0.7282**（I3 ep1 = 0.7290，Δ-0.08pt，valid loss 0.629 健康）——**延迟到 ep1 的浅层容量在晚期阶段中性无害**（对比 FX2 同期 0.6430 塌缩）。ep0 快速适配期是容量干扰的窗口，晚期平衡已形成后容量可以共存。
- **0.1× 中性 → 改进靠剂量**。FX4 ep2 无增量信息（best 已锁 ep0 0.7326），提前切 FX5：**同一延迟点（step 11297）、剂量升到 A 0.1×/B 0.3×**（LoRA+ 式，复用 FX3 剂量）。FX4 checkpoint 已删。
- **FX 系列图景**：ep0 容量=干扰（FX2 0.7244 / FX1 塌缩），ep1 容量=中性（FX4 0.7282）→ 剂量轴在晚期展开：0.1（中性）→ 0.3（FX5，运行中）→ 视结果 0.5 或回撤。若 0.3 仍中性，改进方向转出 FX 线。
- FX5 运行中（fx5_run.log），ep1 判决点 vs 0.7290。

### FX5 终止 + FX6 立项（2026-10-05 晚，用户指令：最优值都在 ep0，直接测 ep0）
- 用户指出 FX5 的设计错误：延迟到 ep1 意味着 **ep0 = 位级复现（0.7326 已知）**，而全部最优值都在 ep0——改进必须发生在 ep0 窗口内。FX5 击杀（配置保留）。
- **FX6 = 容量从头激活 + 剂量减半（A/B 均 0.05×）**，ep0 直接判决。设计逻辑：ep0 剂量响应现有两点 0×→0.7326、0.1×→0.7244，0.05× 探中间——
  - **内点最优（>0.7326）**：容量在 ep0 有被干扰淹没的正贡献 → 细化剂量轴继续；
  - **单调（0.7244 < 结果 < 0.7326）**：容量对 ep0 单调有害，三点闭环，"浅层容量改进 ep0"方向关闭（FX 系列结论=修复真实但容量在该配方下无增益）。
- 教训记录：pkill 模式包含自身命令行里的路径字符串（rm 路径含 rlm_FX5）导致自杀，复合命令需用 `[F]X5` 式转义或分步执行。
- FX6 运行中（fx6_run.log），ep0 监控已挂。

### FX6 终局 + 剂量线关闭（2026-10-05 晚）
- **FX6 ep0 = 0.6385**（eval loss 1.32，塌缩盆地）→ 击杀。
- **关键方法论结论：ep0 剂量响应在"每剂量单 draw"下不可测**。FX 系列三个单点 = 一次健康（FX2 0.7244）两次塌缩（FX1 0.6424 / FX6 0.6385），与 ~50% 塌缩基率完全相容——抽签噪声（±9pt）淹没任何剂量效应。区分剂量需要多 draw（被禁），故"浅层容量改进 ep0"方向**因不可测而关闭**（非因负证据）。剂量响应三点（0/0.05/0.1）留档但只有 FX2 点位可信。
- **对抽签免疫的改进杠杆转向：ep0 内部峰值捕获（PK0）**。冠军峰值在 ep0，但从未验证是否在 ep0 **内部**——若真实峰值在 step<11297，直接白捡提升。

### PK0：mid-epoch 峰值捕获（实现 + 运行中）
- **实现**：`run.mid_epoch_eval_iters: "2000,4000,6000,8000,10000"` → runner 在训练内层循环挂钩子，于指定步触发 valid 评估；**评估前后快照/恢复全部 RNG 状态（python/CPU/CUDA）**，训练轨迹保持位级不变（与 champion 同 draw，ep0 边界必然=0.7326）。
- **冒烟**：step 100 触发一次评估后，150 步 loss 与 I3 逐位一致（1.3734/1.4467）✓ 钩子无扰动验证通过。
- PK0 = I3 配置 + 5 个内部评估点（+15 min 评估时间），运行中（pk0_run.log）。
- **判决**：内部点 >0.7326 → 新冠军 + "峰值在 ep0 内部"结论（继续加密采样）；全部 ≤0.7326 → 确认边界即峰值，评估粒度问题排除。

### PK0 终局（2026-10-05 晚）：ep0 内部无峰值，负结果干净
- **ep0 内部 AUC 曲线（RNG 恢复式中途评估，同 draw）**：step 2000→0.6274、4000→0.6950、6000→0.7025、8000→0.7249、10000→0.7264、边界 11297→0.7326——**单调上升至边界，无内部峰值**。评估粒度假设排除，0.7326 是 ep0 真实顶点。run 提前终止（ep1/ep2 已知位复现），checkpoint 已删。
- mid-epoch eval 钩子（RNG 快照恢复）作为通用基础设施留档：`run.mid_epoch_eval_iters`，零扰动已验证。

### 2026-10-05 收盘：改进杠杆全面盘点（供用户决策）
单 draw 纪律 + seed 禁令下，所有已试杠杆及其结论：
| 杠杆 | 结论 |
|---|---|
| 结构搜索（报告 A/B/C、CW、RC rank、多层注入等） | 全部关闭，无超 A 者 |
| 浅层容量（FX 系列：修复真实，剂量/时机不可测） | 因抽签噪声不可测而关闭 |
| 评估粒度（PK0 内部峰值） | 边界即峰值，关闭 |
| 权重/预测融合 | 历史已关闭（soup/fusion 均负）|
| **抽签本身** | 塌缩率 ~50% 是最大未解决问题：E[run] 被拉低 ~5pt，但 champion 数字来自健康 draw，降低塌缩率不提升 best-of-record |
**剩余可选项（需用户拍板）**：① 解禁 seed/多 draw，测"降低塌缩率"的干预（grad clip、lr 温和化——现有 config 支持未测）；② 换数据规模/配方 regime（超参重扫，多 GPU 时）；③ 转论文写作（素材充分：机制分析链完整）。

## 2026-10-05 深夜：LLaRA 基线适配（用户指令）

### 任务
LLaRA（github ljy0ustc/LLaRA，输入侧协同嵌入注入）改造到我们的 ml-1m OOD 数据集 + Yes/No 任务 + AUC/UAUC 指标。现有 GraphLoRA 代码不动，代码放独立目录 `/home/bbx/user_lx/LLaRA_baseline/`。

### LLaRA 原理（保留其核心）
LLaMA-7B + LoRA（r8/α32/d0.1，all-proj，bf16）+ **输入侧注入**：冻结的 SASRec 物品嵌入经 MlpProjector（256/64→4096 两层 MLP）映射到词嵌入空间，替换输入中 `[HistoryEmb]/[ItemEmb]` 特殊 token 位置；**curriculum prompting**：训练早期用 `[PH]` 占位（projector 冻结），后期换真实注入。

### 适配实现（LLaRA_baseline/ours/，核心模型零修改）
- `data_ours.py`：我们的 OOD pkl → LLaRA 格式（历史截断最近 10 个非空 title+id，target=item title），collater 保留 curriculum 逻辑（[PH]/真实注入按步数概率切换 + projector 冻结联动），目标 token 改 Yes/No（无前导空格——' Yes' 首 token 是空格片段，Yes/No 首段相同无法做 margin，与 GraphLoRA ans_type v2 对齐）。
- `model_ours.py`：LLaRA 原装 load_llm/projector/wrap_emb；协同侧换我们的**预训练 MF 表**（[3256,256] 冻结，与 GraphLoRA 热启动同源，保持可比）；score() = 末位 logit 的 yes-no margin。
- `run_ours.py`：纯 PyTorch 循环（共享环境未装 PL，避免污染），Adam + LLaRA 的 LinearWarmupCosine（lr 8e-4，warmup 5%），bs2×accum16（bs4 峰值 22.6GB 贴上限），每 epoch valid AUC/UAUC，best-valid 权重终测 test。
- UAUC 按用户配对协议（双方类别齐全的用户取 per-user AUC 均值）。
- 调试修掉：Yes/No 首 token 冲突、BatchEncoding 未上卡。
- 冒烟：loss 17→0.017（千步内收敛），19.9GB 稳定。

### 运行（llara_ours_run.log，约 2h）
3 epoch，~34min/epoch。判读基准：GraphLoRA champion valid 0.7326 / test 0.7378。

### LLaRA-ours v1 判决：valid AUC 0.5043（随机），train loss 0.015 —— 记忆读出
- 无标签泄漏（target 不在历史，0/33891）。机制推断：**MF 表就是在该 train 集上 BCE 训练的**，注入的嵌入里编码了训练对标签；lr 8e-4 的 all-proj LoRA 在注入开启后（curriculum ~5%）仅 9 个 update 就学会读出记忆分量 → train CE 0.015，valid 纯噪声。
- 诊断增强：run_ours 加 train-subset（5200 样本）AUC 对照——健康配置应 train-valid 同步；记忆读出则 train 高 valid 随机。
- pkill 自匹配连环坑记录：模式或复合命令其他部分含目标字符串即自杀，统一用 `[F]X` 转义 + 分步执行。

### 夜间扫参（sweep_night.sh，5 个单 epoch 探针，~4.5h）
| 探针 | 配置 | 目的 |
|---|---|---|
| v1b | lr 2e-4 | 降 lr 抑制记忆读出 |
| v1c | lr 1e-4 | 剂量再降 |
| v1d | lr 2e-4 + no_curriculum | 排除 PH 阶段干扰 |
| v2 | lr 2e-4 + mf_lr 1e-4 | 协同侧梯度回传（用户方向 1）|
| v3 | v2 + sem_align_weight 1e-4（层 8 关系 MSE，移植 I3 机制） | 语义相似度更新（用户方向 2）|
- v2/v3 代码就绪：MF 表 nn.Embedding 可训练开关（mf_lr>0）、L_sem（LLM 层 8 隐状态在注入 token 位置的关系矩阵 teacher，stop-grad，监督 MF 关系）、修复 loss 双计 bug。
- 判据：valid AUC 与 train-diag AUC 的关系（同步↑=泛化；train↑valid 平=记忆读出未除）；胜者跑 3 epoch 全程。

### **BUG 破案：scoring 位置错误（用户判定 v1 0.5043 必有 bug，正确）**
- **根因**：LLaMA pair 编码在 prompt 与答案之间插入分隔 BOS（实测 `...Answer:, [BOS]=1, 'Yes'`）。训练时答案 token 的预测条件是 **BOS 之后**；而 `score()` 在 prompt 末位直接取 logit——该位置的训练目标是 **BOS 本身**，Yes/No margin 是噪声 → valid 0.5043。**v1 的"记忆读出"推断作废，0.015 的 train loss 未必是坏事**（等 fixed 重跑判定）。
- **修复**：score() 末尾拼接 BOS 嵌入，读取其后的 logit 做 margin；evaluate 加 p(Yes) 分布诊断（检测退化输出）；best checkpoint 落盘（此前只存内存，v1 被杀即丢——教训）。
- 重跑扫参 sweep_night2：v1（原配方 8e-4，直接验证 bug 修复效果）→ v1b(2e-4) → v1c(1e-4) → v2(mf_lr) → v3(+sem)。v1 结果是分水岭：若 valid 大幅 >0.5，bug 确认且原配方可能直接可用。

### 方法探究路线（当前）
**RC1（运行中）→ 按结果决定**。FX2（差分 lr 温和激活）为候选，需用户批准方向。论文写作仍是兜底，不进当前计划。

### CW post-mortem（零成本，diag_confidence_A.npz）
- **推翻了我此前"权重集中在已对齐 pair"的猜测**：E[e|c] 沿 c 分箱近乎平坦（0.0068-0.0121），corr(c,e) pearson = **-0.055**——层间置信度对"哪些 pair 失配"几乎零信息。
- 梯度质量：top-50% 置信 pair 占 76.7% 质量，但其残差（0.0080）与 bottom-50%（0.0086）几乎相同；N_eff(τ=1e-3)=74.9% pairs。
- 结论表述：CW 加权 = 给语义梯度注入与目标难度结构**无关**的逐 pair 各向异性（c 值域 0-1）。"confidence ≠ utility" 现在有机制数据支撑，不只是措辞谨慎。数据在 cw_post_mortem 输出（本轮 shell）。

### 容量诊断（diag_lora_capacity.py，champion checkpoint）——**RC 路线最强先验确认**
- **层 0-7 的 LoRA ΔW ≡ 0（精确零）**：16 个模块（q/v×8 层）的 B 全零、|A|max=0.015624 恰等于 kaiming 上界 0.015625（L8 为 0.0429，明显离开初始化）→ 浅层 LoRA 参数在全部 33891 步中**梯度精确为零**，预算的 1/4（64 rank 单位）证明性闲置。
- 活跃分布：q 路径 thirds 0.245/1.534/1.751，argmax = **L31（|F|=5.52，注入层特殊活跃）**；v 路径 argmax = L8（1.42），中后段总体高于浅段。
- 含义：① RC1 里 L0-7 降 rank 在功能上是空操作（ΔW 恒 0 与 r 无关），RC1 的实际效应 = L16-23 容量 +25%，归因更干净；② **层 8 精确边界待查**——为何梯度恰从 L8 开始非零（与语义教师层 8 同号是巧合还是结构问题？）需 GPU 空闲后做单批梯度探针（L0 vs L8 的 lora_B grad norm）；若证明是优化器/图结构 bug，"修复后浅层 LoRA 激活"本身就是新实验。

### CW 实现（minigpt4/models/minigpt4rec_v2.py，配置门控默认关）
- `sem_consensus: True` + `sem_consensus_layers: "6,8,10"` + `sem_conf_tau: 1e-3`。
- teacher pass exit_layer 提到 10，抽取三层 ID-token 隐状态（stack，detach）；L_sem 变为加权 MSE：target = sg(mean_l S_l)，c_ij = exp(−Var_l S_l/τ)，L = Σc·(S_G−target)²/Σc。GNN 梯度路径不变（纯 teacher 侧替换）；R 系列 corr loss 不受影响（仍读 h_semantic 的 layer8 切片）。
- 配置：`..._rlm_CW1_confmse.yaml`（= I3 + 上述三键）。
- 冒烟：off 路径须逐位复现 I3 前 150 步；on 路径检查 [CW] 横幅、L_sem 量级、无 NaN。

## RC0/RA3/RA2：结构对照通过 + 锚定修正剂量实验（2026-10-03，用户方案二延续）
- **RC0（纯结构对照，ρ=0/w=0/α=0，分支在图中但零加权）ep0=0.7295/ep1=0.7179/ep2=0.7011 —— 与 R0S-v3（分支完全不在图中的另一种惰性实现）逐位一致**。两种不同的惰性实现 → 相同轨迹：惰性在图结构层面被最强验证（用户 R0-S 设计的对照目的达成）。
- **语义对齐的价值定量确认**：I3（旧 L_sem 开）vs RC0/R0S-v3（等价语义关）= +0.3/+1.1/+1.8pt（ep0/1/2），随深度增长——旧 L_sem 真实有益。
- 锚定修正 α>0 的 run：RA3（gs02+衰减）ep0=0.6416✂、RA2（gs01+恒定 w=0.1）ep0=0.6442✂、对照 R2b（gs01+α=0.05 无衰减）ep0=0.7289 健康——**base/draw 混杂，单 run 不可判读**。
- 方法论终稿：同配置重跑逐位确定；任何图结构扰动 = 新 draw（塌缩率 ~50-60%）；**结论需 2-3 draw/配置**。

## 2026-10-06/07：LLaRA 基线适配全记录（独立目录 /home/bbx/user_lx/LLaRA_baseline/）

### 适配实现
LLaRA（github ljy0ustc/LLaRA）核心零修改：LLaMA-7B + LoRA（r8/α32/d0.1，all-proj，bf16，grad ckpt）+ 输入侧协同嵌入注入（协同侧嵌入经 MlpProjector 256/64→4096 替换 [HistoryEmb]/[ItemEmb] 特殊 token 位置）+ curriculum prompting（早期 [PH] 占位）。改造仅四处：任务头（候选生成→Yes/No，CE on 答案 token）、数据接口（OOD pkl / 原版 df → LLaRA 格式，历史截断最近 10）、评估（末位 logit margin → AUC/UAUC/session-UAUC）、协同侧可插拔（0912 MF 表 / SASRec）。

### 排查修掉的 bug 链（按发现顺序）
1. scoring 读错位置：pair 编码的 BOS 分隔符使答案预测条件在 BOS 后，score() 读 prompt 末位 = 噪声（0.5043 事件）；
2. Embedding 下标 / wrap_emb 返回值不一致（v2/v3 改造回归）；
3. 旧版 transformers ckpt 参数不兼容 + **PH 阶段零梯度断图**（无注入点的批 inputs_embeds 无梯度 → 重入式 ckpt 断链，修复 = grad-enabled 守卫的 requires_grad 钩子）；
4. 注入原地赋值在 leaf 上崩溃 → out-of-place index_put；
5. bs4 长 prompt OOM → ckpt + OOM 跳批；
6. pkill 自匹配连环自杀（模式必须 [X] 转义）；
7. llara_ml 数据 df 的 tuple 解析、len_seq 字段缺失；
8. SASRec 训练数据污染：label=0 行的目标是采样负样本，必须 label==1 过滤（HR@10 0.0178→0.0236）；
9. 评估集对齐：valid_small_ood2（5200）= GraphLoRA 冠军评估集，全量 valid_ood2（10401）不可比。

### 关键诊断方法（可复用）
- 逻辑回归特征探针：注入特征 valid 0.6706 可迁移 → 排除特征/评估问题，锁定训练侧；
- MF u·i train AUC 0.9994：MF 表编码训练标签记忆 → 解释高 lr 崩溃通道；
- score-diag p(Yes) 分布：常数=无信号，全幅=真实判别。

### 最终结果矩阵（valid_small 对齐评估，我们数据，seed 1234，1-3 epoch）
| 协同侧 | LoRA/lr | valid AUC | test AUC |
|---|---|---|---|
| 下一项 SASRec（重训，HR@10 0.0236）| all/1e-4 | 0.6728 | 0.6941 |
| CTR-SASRec（任务对齐训练，AUC 0.6476）| all/1e-4 | 0.6771 | 0.7102 |
| **0912 MF 表（收敛 CTR 训练）**| all/1e-4 | **0.7023**（峰值 ep1）| **0.7110** |
| 0912 MF 表 | qv/2e-4 | 0.6953 | 0.7051 |
| GraphLoRA champion（MF+NGCF，Layer31 注入，不同架构）| — | 0.7326 | 0.7378 |
| llara_ml（原版 SASRec + 原版数据，上限参考）| — | 0.8588 | 0.8488 |

### 结论
1. 任务改写与管线验证无罪：llara_ml 0.8588 证明 LLaRA 核心 + 我们的 Yes/No 任务头/指标全部正常；
2. 我们数据上 LLaRA-ours 最优 = MF 表协同侧 0.7023/0.7110（lr 8e-4 会读出 MF 的训练标签记忆 → 0.50 崩溃，降 lr 解决）；
3. 协同侧替换（两种 SASRec 目标）均不敌 MF 表——OOD 数据的序列信号太弱（HR@10 0.0236 vs 原版 0.1064）；
4. 与 champion 差距（valid -3pt）归因：任务难度（真实负样本 vs llara_ml 随机负样本，天花板差异）+ 注入位置（输入层 vs Layer31 深层）；
5. 任务本质澄清：label = rating≥4（偏好预测），valid 负样本 = 真实看过但不喜欢的电影（困难负样本，非随机）。

### 小模型（SASRec）质量深挖（2026-10-06 深夜，用户指令：解决小模型质量问题）
- **过拟合测试通过**：185 样本 150 步 train AUC→1.0 —— CTR 训练代码正确，无标签/对齐 bug。
- **训练移动量测量**：嵌入表仅移动初始范数的 11.3%，s_fc 移动 155% —— BCE 配方（每会话 1 采样负例）把学习全部灌进输出层，嵌入表接近随机初始化。
- **嵌入质量矩阵**：所有 SASRec 变体（含 LLaRA 官方 movielens.pt）物品嵌入均为各向同性随机特征（两两 cos≈0.000、有效秩满秩）；MF 表有真实结构（cos 0.0385、有效秩 19.6/256）。**LLaRA 注入通道拿到的"协同信息"是随机向量，官方设计亦然**——llara_ml 0.8588 全靠标题文本 + 随机负样本的简单任务。
- **训练塌缩抽签**：全量 CTR 训练存在两种归宿——泛化解（valid 0.6476）与记忆解（train loss 0.19、valid 0.52 纯记忆）；同配置同种子重跑因 GPU 非确定性分岔（ep0 损失 2.97 vs 2.70）。尝试的稳定手段：in-batch 负例（0.53，流行度偏差，崩）、dropout 0.5（0.52）、hidden 128（0.54）、lr 5e-4（0.52）、MF-PCA 初始化（0.578，train/valid 同步但欠拟合）——均未稳定超越随机基线。
- **逐环探针矩阵（逻辑回归，valid_small）**：SASRec-LLaRA-form 原始特征 0.5369（序列结构不可迁移到偏好任务）→ +训练后 projector 0.5910（train 爆到 0.9232 = 标签解码器）；CTR-SASRec 原始 0.6511 → +projector 0.6140（受损）；MF 原始 0.6534。
- **最终裁定**：SASRec 的下一项序列结构与偏好判别任务错位，其嵌入质量（HR@10 0.2036）对任务 AUC 无贡献；**直接在任务上训练的 MF 表仍是最强协同侧**（独立打分 0.6775 + 注入后 0.7023/0.7110）。LLaRA-ours 的 0.69-0.70 平台 = 文本 + 弱注入增益，对协同嵌入质量不敏感（单 token 注入的信息瓶颈）。

### 协同侧最终对决（2026-10-07，valid_small 5200 对齐评估，全部 3 epoch 同配方）
| 协同侧 | 独立质量 | valid AUC | test AUC |
|---|---|---|---|
| text-only（零协同）| — | 0.6767 | 0.6929 |
| 下一项 SASRec（重训）| HR@10 0.0236 | 0.6728 | 0.6941 |
| **CTR-SASRec（任务对齐，最强 SASRec）**| AUC 0.6406 | 0.6748 | 0.6973 |
| LLaRA-form SASRec | HR@10 0.2036(train) | 0.6726 | 0.7039 |
| **0912 MF 表** | u·i AUC 0.6775 | **0.7023** | **0.7110** |
| GraphLoRA champion（MF+NGCF，Layer31）| — | 0.7326 | 0.7378 |
**最终结论**：① LLaRA 输入注入架构下，全部 SASRec 变体的注入在 valid 上不敌纯文本（0.672-0.675 vs 0.677），仅 test 微升；② 唯一有真实增益的协同侧是直接在偏好任务上训练的 MF 表（+2.6pt valid / +1.8pt test）；③ 协同嵌入质量（HR@10 跨 8.6 倍）与注入增益无关——序列结构信号与偏好判别任务错位是本质；④ 与 GraphLoRA champion 的 ~3pt 差距 = 深层注入（Layer31）+ NGCF 的架构增益，LLaRA 输入注入在此任务形态上不如深层注入。
