# B2/B3 双向互学习实验记录

> 目标：LLM 语义视图 ↔ GNN 协同视图双向互学习，最终以 LLM 推荐 AUC/uAUC 为准。
> 判定标准：valid AUC 是否保持/超过热启动点（基线 GraphLoRA best ≈ 0.6995）。
> 共享 GPU 注意：有其他项目在调试，训练前查 `logs_monitor/`，batch=2 时训练占 ~18.5GB。

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
