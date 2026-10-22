#!/usr/bin/env bash
# One experiment only: fresh GraphLoRA + CoRA, TWO full epochs, GPU0 foreground.
set -Eeuo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[R5-FRESH-STOP] conda activate minigpt4 first" >&2
  exit 2
fi

RUNNER="experiments/collab_injection_v1/r5_cora_fresh2.py"
ENTRY="experiments/collab_injection_v1/r3_entry.py"
ADAPTER="experiments/collab_injection_v1/r3_paper_adapters.py"
TEST="experiments/collab_injection_v1/test_r3_cpu.py"
for file in "$RUNNER" "$ENTRY" "$ADAPTER" "$TEST"; do
  test -f "$file" || { echo "[R5-FRESH-STOP] Missing $file" >&2; exit 2; }
done
python -m py_compile "$RUNNER" "$ENTRY" "$ADAPTER"
python -u "$TEST"

SOURCE="logs/collab_v1/R4_ISOLATION_20261022/R4_02_CORA_ONLY/FULL_E1.yaml"
if [[ ! -f "$SOURCE" ]]; then
  echo "[R5-FRESH-STOP] Missing R4-02 config for dataset/path template: $SOURCE" >&2
  exit 2
fi

# Explicitly reject concurrently running GraphLoRA training processes.
if pgrep -af '^python .*train_collm_mf_din\.py|^python .*r[345]_entry\.py' | grep -v grep; then
  echo "[R5-FRESH-STOP] Another training process appears active" >&2
  exit 3
fi

mkdir -p logs/collab_v1/R5_CORA_FRESH2_20261022
echo "[R5-FRESH-START] one method=cora, 2 epochs, original Vicuna only, GPU0 $(date -Is)"
python -u "$RUNNER" 2>&1 | tee logs/collab_v1/R5_CORA_FRESH2_20261022/master.log

grep -q '\[R5-FRESH-COMPLETE\]' logs/collab_v1/R5_CORA_FRESH2_20261022/master.log || {
  echo "[R5-FRESH-STOP] Missing complete and validated two-epoch result" >&2
  exit 4
}
cat logs/collab_v1/R5_CORA_FRESH2_20261022/CORA_FROM_SCRATCH/manifest.json
