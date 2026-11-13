# CoRA Post-hoc Ensemble 复现 + 三组 Valid-only 改进

## 检查范围与可实现内容

- 已在 GitHub `NanDingEi/GraphLoRA-Research` 的 `exp/cora-rr1-test-20261110` 等 CoRA 分支找到：
  - `cora_residual_routing6.py`（RR1/RR5 训练和门控融合）
  - `cora_late_layer_diag.py`（final RMSNorm + LM head 逐层 Yes logits 捕获）
  - `cora_residual_ensemble.py` / `_rank.py`（Valid-only 残差融合分析）
- 未发现：`cora_rr1_test_logits.py`、`cora_rr5_test_logits.py`、`cora_g6_capture_logits.py`、`cora_ensemble_final4.py` 的对应 GitHub 文件。若这些脚本仍在服务器，**继续使用原始捕获脚本**，不要用未验证的通用 runner 替换，否则容易丢失 D4/C6 或 RR5 运行时补丁。
- 此目录提供的是 **NPZ 已捕获之后的完整 CPU 实验**，包括历史集成复现、严密审计、valid 选参和 test 冻结评估。没有连接你的服务器，未真实执行 test。

## A/B/C 必须对应的 checkpoints

```text
A: experiments/no31_cora_original/output/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE/20261103092/checkpoint_best.pth
B: experiments/no31_cora_original/output/CORA_RESIDUAL_ROUTING6_20261103/RR5_USER_GATE_R31/20261103153/checkpoint_best.pth
C: experiments/no31_cora_original/output/CORA_GATE_TUNE_20261112/G3_FIXED_A025/20261112014/checkpoint_best.pth
```

捕获时应当安装各自的 D4 多粒度协同输入、C6 CTR/KD、RR5 fusion 模块，检查 checkpoint 和最终分数：

- A 的 test AUC 参考值 `0.739652`，取 L32 raw Yes logit。
- B 的 test AUC 参考值 `0.739113`，取 RR5 的 `z32+sigmoid(g(u))*(z31-z32)`，**不能把 RR5 的 L32 当 B**；RR5 MF 用户表来自 RR5 checkpoint。
- C 的 test AUC 参考值约 `0.7369`，本脚本用 `(L31,L32)` 以固定 `0.25` 重建。
- 全部必须对齐 `(label, UserID, TargetItemID)`，test 与 valid 不能交叉。
- 源 NPZ 至少提供 `labels`, `users`, `items`, `yes_logits`(4×N)；B 有 `fused_logits`（或 `fused_scores`/`fused_yes_logits`/`fused`/`score`）时会用其审计；若只有四层 `yes_logits`，脚本会从 RR5 checkpoint 的 MF 用户表与 6 个 gate 张量自动重建 B，不再要求额外 GPU 推理。缺少 fused 时使用明确指定的 user-id offset（默认 0），绝不通过 Test AUC 猜索引。
- 已存储的 `out/LATE_LAYER_DIAG_D4_C6_FIX_20261102_165625/layer_logits_valid.npz` **不保证**是 RR1(ep37) 的 valid logits。务必用相同 RR1 checkpoint 验证，不可静默替代。

## 原始捕获命令（仅服务器已有脚本时执行）

```bash
cd /home/bbx/user_lx/CoRA_official
export CUDA_VISIBLE_DEVICES=0
# 串行；以下命令是你已确认的原服务器命令，需先检查脚本确实存在
python -u experiments/no31_cora_original/cora_rr1_test_logits.py --run
python -u experiments/no31_cora_original/cora_rr5_test_logits.py --run
python -u experiments/no31_cora_original/cora_g6_capture_logits.py --run --trial G3_FIXED_A025 --split test
# valid 同样应对 A/B/C 精确 checkpoint 捕获并做 AUC/运行时校验。
```

建议服务器先搜索现存的 NPZ：

```bash
find out -name '*.npz' \( -iname '*logit*' -o -iname '*score*' \) -print | sort
```

## 正式用法（CPU）

在服务器 `/home/bbx/user_lx/CoRA_official` 下，将 `cora_ensemble_repro_improve.py` 放在工作目录或指定相对路径。下列 `<...>` 需替换为真实 NPZ **文件**路径；如果一个目录内只有一份 NPZ，可以直接传目录。

### Step 1: Valid-only 选参并锁定配置

```bash
cd /home/bbx/user_lx/CoRA_official
mkdir -p log out/CORA_ENSEMBLE_REPRO_20261113
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u cora_ensemble_repro_improve.py \
  --phase select \
  --a-valid '<RR1_ep37_valid.npz>' \
  --b-valid '<RR5_ep21_valid.npz>' \
  --c-valid '<G3_valid.npz>' \
  --norm test_first128 \
  --rr5-checkpoint experiments/no31_cora_original/output/CORA_RESIDUAL_ROUTING6_20261103/RR5_USER_GATE_R31/20261103153/checkpoint_best.pth \
  --selection out/CORA_ENSEMBLE_REPRO_20261113/selection.json \
  2>&1 | tee log/CORA_ENSEMBLE_REPRO_20261113_valid.log
```

`test_first128` 是历史 transductive 协议：每个模型各自使用该应用 split 的**前 128 个无标签 logits** 计算均值和标准差。要评估严格 inductive 归一化，**另建独立实验 tag**，把 `--norm` 改成 `valid_stats_128`（test 采用 valid 前128矩），避免与历史数值混合汇报。

复现默认要求 valid 选出的 AB_AUC 权重精确等于历史 `0.40`，否则拒绝输出 selection.json；仅定位差异时可显式添加 `--allow-different-valid-weight`。Valid 阶段同样强制重建 RR5 gate 并校验 fused scores。

脚本会输出历史 `E2_AB_HALF`、`E2_AB_AUC`、`E3_ABC_THIRD` 的 valid 表现，并额外在 valid 上锁定以下三个方案：

| 试验 | 权重或公式 | 优化目标 | 训练开销 |
|---|---|---|---|
| I1_SIMPLEX_AUC_UAUC_FLOOR | `a*A + b*B + c*C`，三权重非负且和为1，步长0.05 | valid UAUC 不低于 RR1 和 AB_HALF 时最大化 AUC | 无 LLM 训练 |
| I2_UAUC_PARETO_C | `(1-λ)*AB_validAUC + λ*C`，`λ=0..0.35, step0.025` | valid AUC 损失不超过 0.0005，最大化 UAUC | 无 LLM 训练 |
| I3_USER_DISAGREEMENT_C | `(1-λ_u)*AB_validAUC + λ_u*C`，`λ_u=λmax·clip((d_u-q50)/(q90-q50),0,1)`，`d_u=mean_i |A_ui-B_ui|` | 无标签用户分歧高时提高 C 的参与度；AUC 保底 | 无 LLM 训练 |

I3 的 `q50/q90` 从 valid 用户统计中计算，`λmax=0,0.05,...,0.40` 在 valid 上选。test 只计算无标签用户差异，用锁定的阈值和权重映射，不读 test label 做任何选择。

### Step 2: 一次性 Test 评估（已冻结的 6 个预注册配置）

```bash
cd /home/bbx/user_lx/CoRA_official
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u cora_ensemble_repro_improve.py \
  --phase test \
  --a-test '<RR1_ep37_test.npz>' \
  --b-test '<RR5_ep21_test.npz>' \
  --c-test '<G3_test.npz>' \
  --rr5-checkpoint experiments/no31_cora_original/output/CORA_RESIDUAL_ROUTING6_20261103/RR5_USER_GATE_R31/20261103153/checkpoint_best.pth \
  --selection out/CORA_ENSEMBLE_REPRO_20261113/selection.json \
  --output-dir out/CORA_ENSEMBLE_REPRO_20261113/TEST \
  2>&1 | tee log/CORA_ENSEMBLE_REPRO_20261113_test.log
```

**硬校验：** 逐行 `(label,users,items)` 完全对齐；A/B/C 的 test 单模型 AUC 与历史值相差不超过默认 `0.002`；C 固定公式重建检查；RR5 从 checkpoint 中寻找 6 个 gate 张量和 `rec_encoder.user_embedding.weight`，若 NPZ 保存了 fused，则通过 0/-1/+1 索引候选的无标签重建误差审计（最大误差不超过 `0.005`）；若 NPZ 只有逐层 logits，则明确使用 `--rr5-user-offset`（默认 0）重建，之后强制核对重建后的 Test AUC，并要求重算的 test AUC 距 `0.739113` 不超过 `1e-4`。任何失败都拒绝输出完整 test 汇总。

如果 RR5 权重或 userID 对应关系有别于原程序，必须先在原始运行时确认，**不能使用** `--skip-rr5-audit` 得到的结果作为严格复现数据。测试成功后写入 `test_summary.csv`、`test_summary.json`。已有 TEST 目录会被拒绝覆盖，避免无意中反复优化 test。

### CPU 环境检测与自测

```bash
python -c 'import numpy, sklearn, torch; print("analysis deps OK")'
python test_ensemble_repro_improve.py
```

自测采用合成数据，只能验证公式、输入检查和 CPU gate 重建是否正确，不代表真实 RR1 / RR5 / G3 的指标。

## 下一阶段：把提升压回单模型，而不是仅依赖集成

**I4_ENSEMBLE_DISTILL_SINGLE（单模型知识蒸馏，待 I1–I3 结果再执行）**：离线捕获训练样本的 A/B/C logits，形成 valid 锁定权重的 ensemble teacher；以 RR1 checkpoint 为 student 起点，仅微调 RR1 的 LoRA/已有投影，不把三套 7B 模型同时装入显存。主任务保留 C6 CTR/KD，额外引入：

`L = L_rec + λ_KD * BCEWithLogits(student/T, sigmoid(teacher/T))*T² + λ_rank * L_pair_user`。

建议控制组 `λ_KD=0, λ_rank=0`、知识蒸馏组 `(0.05,0)`、联合排序组 `(0.05,0.02)`，固定 seed42 / 相同训练步数 / 所有轮次 VALID / 仅选定 checkpoint 后一次 TEST。训练数据的 teacher logits 必须严格对齐 train 样本，禁止先前仅有的 **test** logits 参与训练。若 24GB 显存紧张，teacher 分数完全预计算，GPU0 只加载学生模型。此 I4 是下一阶段实验设计，**本包没有实现/运行训练脚本**。

## 不要混淆的结论

- 历史表：RR1 test `(0.7397, 0.7064)`，E2_HALF `(0.7417,0.7068)`，E2_AB_AUC `(0.7415,0.7072)`，E3_ABC `(0.7410,0.7081)` —— 用户提供的**历史数据**。
- 本包新 I1–I3 的真实 valid/test 数字目前**尚未计算**，不能作为实验成果填表。
- E2_AB_AUC 必须在与 test 同 checkpoint 的 A/B valid 中重新得到约 `w=0.40`，否则应先查明捕获差异，而不是强设 0.40 伪造复现。
- 用同一 test 集评估多个预注册方案可以作为一次批量的最终比较，但不能观察 test 指标后继续改权重并再次把结果称为独立测试。

## 一键服务器入口（推荐）

把 `cora_ensemble_repro_improve.py` 和 `cora_ensemble_server.sh` 下载到服务器 `experiments/no31_cora_original/`，然后：

```bash
cd /home/bbx/user_lx/CoRA_official
# 第一次先用 find -name '*.npz' 查找真实的有效文件；不要把 D4 valid 误认成 RR1 ep37。
find experiments/no31_cora_original/out out -type f -name '*.npz' 2>/dev/null | sort

export A_VALID=/absolute/path/to/RR1_ep37_valid.npz
export B_VALID=/absolute/path/to/RR5_ep21_valid.npz
export C_VALID=/absolute/path/to/G3_valid.npz
export A_TEST=/absolute/path/to/RR1_ep37_test.npz
export B_TEST=/absolute/path/to/RR5_ep21_test.npz
export C_TEST=/absolute/path/to/G3_test.npz

# CPU, no GPU; separate, explicit stages.
bash experiments/no31_cora_original/cora_ensemble_server.sh select
bash experiments/no31_cora_original/cora_ensemble_server.sh test
cat out/CORA_ENSEMBLE_REPRO_20261113/TEST/test_summary.csv
```

六个真实 NPZ 路径必须从服务器实际文件得到；上述 `/absolute/path/to` 是占位符，不可直接运行。需要重新捕获时，使用服务器上已经验证能正确安装完整 D4+C6+RR5 运行时的原脚本；此仓库目前没有这些捕获脚本，不能假装仅凭这套 CPU 融合代码就能完成 LLM 推理。

## 2026-11-13 修复：RR5 Test NPZ 无 fused 字段

已收到服务器实测：RR1/RR5 的 `layer_logits_test.npz` 均为 `yes_logits` `(4,7331)`，RR5 没有 `fused_logits`。`cora_ensemble_repro_improve.py` 已更新：缺失 fused 时从 RR5 checkpoint 自动重建；默认 `--rr5-user-offset 0`，禁止通过 Test AUC 优选 offset。没有 valid 三成员或 C/G3 logits 时仍不能合法运行完整选权与三模型 Test，必须先确认/捕获。见服务器对应分支最新提交。
