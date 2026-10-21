# P1 — P0 LoRA parameter-drift diagnostic (CPU only)

P0 CONTROL (1 epoch): Valid AUC 0.7219307045640613, UAUC 0.6659307008748429.
P0 AUX002 (1 epoch): Valid AUC 0.7217675150885272, UAUC 0.6650199422913636.
Original S2 checkpoint: AUC 0.7296752596287361, UAUC 0.6841670211619721.
These single-seed results do not support multi-layer BCE expansion.

## Commands

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
git merge --ff-only FETCH_HEAD
conda activate minigpt4
python -m py_compile experiments/collab_injection_v1/analyze_p0_parameter_drift.py experiments/collab_injection_v1/test_p0_parameter_drift_cpu.py
python experiments/collab_injection_v1/test_p0_parameter_drift_cpu.py
RUN=logs/collab_v1/S2_L30_AUX_P0_20261021_165700
set -o pipefail
python -u experiments/collab_injection_v1/analyze_p0_parameter_drift.py \
  --run-dir "$RUN" \
  --output "$RUN/P1_PARAMETER_DRIFT.json" \
  2>&1 | tee "$RUN/P1_PARAMETER_DRIFT.log"
```

The diagnostic reads P0 state.json and validates completion, original S2 checkpoint SHA256, and saved config checksums.
The tool loads three trainable-parameter checkpoints on CPU, using mmap when available. It does not initialize Vicuna, train a model, or evaluate Test data.

## Measurements

- [P1-MODULE]: total LoRA/GNN/projector drift from S2, control-vs-AUX parameter distance, and direction cosine.
- [P1-LAYER]: each decoder layer's control/S2 update norm, auxiliary/S2 update norm, auxiliary-minus-control norm, update cosine.
- [P1-METRIC]: existing one-epoch Valid AUC/UAUC deltas between the two P0 arms.
- [P1-COMPLETE]: machine-readable JSON report path.

Define dc = theta_CONTROL - theta_S2, da = theta_AUX - theta_S2. The update cosine is cos(dc,da).
This is NOT a gradient conflict measurement. It only describes where learned parameters ended after training.
Group relative drift can be large if the source group is near zero; inspect absolute norms as well.

State tensors missing from any checkpoint are reported and are not counted as unchanged. Missing LoRA trainable keys or incompatible shapes fail closed.
The next step after P1, if warranted, is limited train-step gradient alignment diagnosis, not launching new full L28/L30 training.
