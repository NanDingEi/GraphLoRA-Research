#!/usr/bin/env bash
# Fair R5: one pretrained-MF GraphLoRA CONTROL and one CoRA, TWO epochs each.
# Smoke both first, compare identical initial MF/NGCF/Projector/LoRA SHA256,
# then run FULL_E2 serially on GPU0. NO TEST, NO S2.
set -Eeuo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[R5-FAIR-STOP] Activate conda minigpt4 first" >&2
  exit 2
fi

BASELINE="tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml"
RUNNER="experiments/collab_injection_v1/r5_fair_pair.py"
ENTRY="experiments/collab_injection_v1/r5_fair_entry.py"
if [[ ! -f "$BASELINE" ]]; then
  echo "[R5-FAIR-STOP] Historical baseline YAML missing: $BASELINE" >&2
  exit 2
fi

python -m py_compile "$RUNNER" "$ENTRY" \
  experiments/collab_injection_v1/r3_paper_adapters.py
python -u experiments/collab_injection_v1/test_r3_cpu.py

if pgrep -af 'python .*train_collm_mf_din.py|python .*r[345]_entry.py' | grep -v grep; then
  echo "[R5-FAIR-STOP] Another GraphLoRA training detected" >&2
  exit 3
fi

OUT="logs/collab_v1/R5_FAIR_MF_CORA_20261022"
if [[ -e "$OUT" || -e "output/R5_FAIR_MF_CORA_20261022" ]]; then
  echo "[R5-FAIR-STOP] Existing result directory; refusing overwrite" >&2
  exit 4
fi

echo "[R5-FAIR] Original pretrained MF, matched fresh NGCF/LoRA/projector, two methods, two epochs each."
python -u "$RUNNER" --baseline-yaml "$BASELINE"
cat "$OUT/fair_comparison.csv"
