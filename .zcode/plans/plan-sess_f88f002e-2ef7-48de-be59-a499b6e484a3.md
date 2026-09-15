# B2/B3 双向互学习实现方案

## 目标
在现有 GraphLoRA（NGCF → 8-d 注入 LoRA 31 层 → Vicuna-7B → Yes/No BCE）之上，新增 LLM 语义视图 ↔ GNN 协同视图的**双向互学习**：
- **B3（关系分布 KL）**：LLM 第 l_s 层的 ID token 隐状态与 NGCF 倒数第二层 256 维嵌入，各自算 user→items 的余弦相似度分布，双向 KL（target 侧 stop-gradient），不对齐表示空间坐标
- **B2（对比式表示对齐）**：两个视图经 P_L(4096→256)/P_G(256→256) projector 归一化后，批内 in-batch InfoNCE 双向互学
- 总损失 `L = L_rec + λ_LG·L_L→G + λ_GL·L_G→L`，λ 默认 0.005/0.02；**不冻结任何一侧**（stop-grad 只在单个辅助 loss 内把 target 固定，LoRA 在 L_rec 和 L_G→L 中持续更新）
- 从基线最优 ckpt（output/graphlora_ml1m/20260903095/checkpoint_best.pth）热启动

## 代码改动（仅 1 个代码文件 + 配置 + smoke test）

### 1. `minigpt4/models/minigpt4rec_v2.py`
- **`NGCF.forward`/`LightGCN.forward`**：加 `return_penult=False` 参数，True 时额外返回倒数第二层 256 维归一化嵌入（fp32，不做 *0.1 缩放）；原 8-d 输出路径逐位不变
- **`__init__`**：新增配置 `mutual_mode("off"|"b2"|"b3"), mutual_layer=8, lambda_llm_to_gnn=0.005, lambda_gnn_to_llm=0.02, rel_temp_llm=1.0, rel_temp_gnn=1.0, b2_temp=0.1`；与 semantic_feedback 互斥（assert）；mutual_layer < gcn_layer（assert）；b2 模式创建 mutual_proj_L/mutual_proj_G（B3 零新参数，热启动无损）
- **`compute_feedback_gcn`**：新增 `elif mutual 激活且 training` 分支——复用 BiLoop 的 pass-1（exit_layer=mutual_layer, output_hidden_states=True）提取 `h_mutual = hs[l_s]` 于 ID token 位置（fp32），**不做** back_proj/gate/embedding 改写；GNN 调用带 return_penult；fb dict 增加 h_mutual/penult/token_node_ids。BiLoop 原分支逐字不动
- **新增 `compute_mutual_losses(fb)`**：
  - B3：按 sample 分组，`token_node_ids < num_users` 定位 user 行，其余为候选 items（交互+target，padding 已剔除）；cos → softmax(r/T) → `KL(sg(P_L)‖P_G)` 与 `KL(sg(P_G)‖P_L)`，fp32 关 autocast
  - B2：批内 ID token 实体 in-batch InfoNCE（同实体跨视图为正对），双向各 detach target
  - 每 100 iter 打印 loss 运行均值
- **`forward_v2`**：loss 加上 λ 加权辅助项，返回 dict 增加诊断键（train_step 只取 "loss"，无副作用）；两 λ 均为 0 时跳过 pass-1（保证与基线 RNG 逐位等价）
- **`from_config`**：透传新键

### 2. 配置 `tran_configs/`（仿 biloop 系列）
- `collm_pretrain_mf_ood_b3.yaml`：mutual_mode: b3、semantic_feedback: False、λ=0.005/0.02、热启动基线 ckpt、output_dir=graphlora_ml1m_b3
- `collm_pretrain_mf_ood_b2.yaml`：同上但 mutual_mode: b2
- 对应两份 `_eval.yaml`（evaluate: True，ckpt 指向新 run best 的占位注释）

### 3. `smoke_test_mutual.py`（仿 smoke_test_biloop.py，跳过大 ckpt）
- B3 梯度定向：仅 L_L→G 反传 → gcn/MF 表有梯度、LoRA/llama_proj 梯度为 None；仅 L_G→L → 反之
- λ=0 等价性：mutual 开/关同种子 loss 逐位一致
- KL 恒等（同分布=0）；B2 projector 双向各得其梯度；eval 路径 logits 不受 mutual 开关影响

## 验证与运行
```bash
~/anaconda3/envs/minigpt4/bin/python smoke_test_mutual.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3.yaml
~/anaconda3/envs/minigpt4/bin/python train_collm_mf_din.py --cfg-path tran_configs/collm_pretrain_mf_ood_b3.yaml
```

## 注意点
- 每步额外过前 8/32 层 LLM（约 +25% forward），可接受
- hs[l_s] 索引约定与 BiLoop 一致
- B2 projector 热启动时随机初始化（strict=False，日志确认 missing keys 仅这两个）