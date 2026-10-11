# GraphLoRA AUC 优先：三组过夜实验

**无需重跑 GraphLoRA 基线**：使用已有「GraphLoRA + LLM→GNN 语义对齐」（历史最佳 AUC 0.7378）作为参考。仅对输入 Projector 施加 Cosine Similarity Preservation，三个实验只改变权重。

| 实验 | `gnn_semantic_align` | `gnn_semantic_weight` | `proj_reg_weight` | `proj_reg_items` |
|---|---|---:|---:|---:|
| S1_sem_proj_0001 | True | 0.0001 | 0.0001 | 64 |
| S2_sem_proj_0005 | True | 0.0001 | 0.0005 | 64 |
| S3_sem_proj_0010 | True | 0.0001 | 0.001 | 64 |

三组均保留 `train_gnn_with_rec_loss=True`、`gcn_layer=31`、`gnn_scale=0.1`、`lora_scale=1.0`。其余参数继承源 YAML（包括 batch size、epoch、学习率、seed、数据划分）。保留源 YAML 中已有的 `injection_mode=legacy`；不会增加本地版本未知的参数。

## 服务器运行

在 GraphLoRA 项目根目录解压后执行：

```bash
cd /home/bbx/user_lx/GraphLoRA
conda activate minigpt4

python graphlora_overnight_auc3/run_three_experiments.py \
  --base-config tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml \
  --dry-run

# 建议在 tmux 会话中运行
python -u graphlora_overnight_auc3/run_three_experiments.py \
  --base-config tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml \
  2>&1 | tee overnight_auc3_master.log
```

使用新的独立输出根目录 `output/overnight_auc_sweep_3/`。自动生成三个 YAML；依次在单卡上训练，单组失败时保留日志并继续，重新运行跳过已成功且配置 hash 没变化的实验。每组 GraphLoRA Runner 会在独立的 job-id 子目录内存储 `checkpoint_best.pth` 及 `valid_log.txt`。脚本生成 `valid_summary.csv` 并根据 **Valid AUC** 挑选各实验的最佳 epoch，不使用 Test 选模型。

## 注意事项

1. 请先确认投影器正则 v2 补丁确已安装（模型文件内包含 `projector_regularized_forward`）。**不要再次 apply**。
2. 此脚本使用先前 `scratch_rlm` YAML 作为母版；它会覆盖语义对齐、投影器正则两类参数，不会自动关闭其他自定义 N1/RLM 功能。请确认母版里没有其他未预期的辅助损失。
3. 先确认母版 `model.ckpt: null`、`run.evaluate: False`，以及你希望的验证文件 (`valid_ood2.pkl` 或 `valid_small_ood2.pkl`) 一致。**历史 0.7378 只有在相同数据划分/训练评估协议下才可直接比较**。
4. 三组均没有测试「纯正则」：原因是实验目标已改为在已知较优语义方案上进一步提升 AUC，而非进行完整因素消融。若三组均未超过旧最佳，则当前证据不支持继续投入更多正则权重搜索。
5. 默认不在训练中测试所有 Test epoch，以避免用测试集优化超参数。原 Runner 训练模式按最大 Valid AUC 存 best checkpoint；若要每轮 Test，可在之后单独加入调试模式。
