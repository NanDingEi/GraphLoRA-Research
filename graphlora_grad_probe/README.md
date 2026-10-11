# GraphLoRA: Projector 梯度冲突诊断（v1）

仅适用于已安装 `graphlora_projector_reg_v2` 的 `minigpt4/models/projector_structure_reg.py`。**只修改这个正则化模块，不修改 `minigpt4rec_v2.py` 或训练 YAML。** 自动备份原文件。不设置环境变量时诊断完全关闭。

## 1. 安装

在 GraphLoRA 项目根目录（例如 `/home/bbx/user_lx/GraphLoRA`）运行：

```bash
unzip -o graphlora_grad_probe_v1.zip
python graphlora_grad_probe/install_grad_probe.py --check
python graphlora_grad_probe/install_grad_probe.py --apply
python -m py_compile minigpt4/models/projector_structure_reg.py
```

只执行一次 `--apply`。如果脚本检测到重复安装，会拒绝改写。

## 2. 单组诊断：从 S2 最佳 checkpoint 开始

```bash
cd /home/bbx/user_lx/GraphLoRA
CKPT=$(find output/overnight_auc_sweep_3/runs/S2_sem_proj_0005 -name checkpoint_best.pth -print -quit)
test -n "$CKPT" || { echo "S2 checkpoint not found"; exit 1; }

GLORA_GRAD_PROBE=1 \
GLORA_GRAD_FIRST=5 \
GLORA_GRAD_EVERY=100 \
GLORA_GRAD_SCOPE=first \
python -u train_collm_mf_din.py \
  --cfg-path output/overnight_auc_sweep_3/configs/S2_sem_proj_0005.yaml \
  --options \
  model.ckpt="$(realpath "$CKPT")" \
  run.max_epoch=1 \
  run.output_dir="$(pwd)/output/probe_S2_nooverwrite" \
  2>&1 | tee probe_S2.log
```

在观察到 `step=1...5` 以及 `step=100` 等若干条 `[grad-probe]` 日志后，按 Ctrl+C 停止（**此命令只是诊断，勿当作正式训练**）。它使用临时 output_dir，不会覆盖之前的 checkpoint。也可以完整跑完一个 Epoch。

S1 同理，将 `S2_sem_proj_0005` 替换为 `S1_sem_proj_0001`、`probe_S2` 替换为 `probe_S1`。

诊断使用 `run.evaluate=False`，才能构建反向图；切勿将它设为 `True`。

## 3. 输出释义

```
[grad-probe] step=1 scope=first params=0.weight base_norm=0.018 reg_norm=0.15 weighted_reg_norm=0.000075 ratio=0.00417 cosine=-0.24 conflict=YES active_base=1/1 active_reg=1/1
```

* `base_norm`：原始 `forward_v2` 返回损失对选择的 Projector 参数的梯度范数，**包括推荐任务以及已开启的语义/其他辅助损失**。
* `reg_norm`：未加权 `L_proj` 梯度范数。
* `weighted_reg_norm`：`lambda * ||grad L_proj||`。
* `ratio`：`lambda * ||grad L_proj|| / ||grad L_base||`。例如 0.004 表示梯度幅值约为 0.4%，不是损失值的比值。
* `cosine`：两个梯度向量的余弦值。负数代表该批次在该参数子集上存在方向冲突，接近零代表近似正交，正数代表同向；**单一 batch 不能证明整体训练冲突**。
* `scope=first` 默认仅诊断 Projector 第一层权重（省显存）；`GLORA_GRAD_SCOPE=all` 会同时检查所有 Projector 参数但增加显存与时间开销；`last` 检查最后一层权重。
* `active_base`、`active_reg` 可检查是否对选定参数产生梯度。若 base 为 0/1，必须排查主任务是否使用/意外 detach 了 Projector。

因为当前 GraphLoRA 还启用了 `gnn_semantic_align=True`，本诊断测量的是 **原模型总损失（推荐+语义）与 Projector Reg 的关系**，无法仅凭该结果拆分 `L_rec` 与 `L_sem`。若需分别测，必须在本地定制的 `forward_v2` 中暴露两个标量损失。

## 4. 保持原行为 / 恢复

* 正常训练不设置 `GLORA_GRAD_PROBE`，新增加的函数立即返回，不调用额外 autograd。
* 手动恢复：将安装时生成的 `minigpt4/models/projector_structure_reg.py.before_grad_probe_YYYYMMDD_HHMMSS` 复制覆盖原模块。
* 诊断在正常 `.backward()` 前调用 `torch.autograd.grad(..., retain_graph=True)`；不会清空、累积或修改 `.grad`，但会额外进行两次有针对性的反向传播，可能增加运行时间、显存占用。默认用第一层以降低开销。
