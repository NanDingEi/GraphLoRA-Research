#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG_DIR="tran_configs/collab_v1/e3"
LOG_DIR="logs/collab_v1"
mkdir -p "$LOG_DIR"

# Print AMP overflow status and exact parameter deltas at optimizer boundaries.
export GRAPHLoRA_DEBUG_OPTIMIZER=1

if [[ ! -f "$CONFIG_DIR/S2_L31_CONTROL.yaml" ||
      ! -f "$CONFIG_DIR/S2_GATED_L31.yaml" ||
      ! -f "$CONFIG_DIR/S2_CONDITIONAL_LORA.yaml" ]]; then
  echo "[ERROR] E3 configs not found. Run prepare_e3_configs.py first."
  exit 2
fi

if pgrep -af '[t]rain_collm_mf_din.py' >/dev/null; then
  echo "[ERROR] Another train_collm_mf_din.py process is running."
  pgrep -af '[t]rain_collm_mf_din.py' || true
  exit 3
fi

echo "[INFO] Repository: $ROOT"
echo "[INFO] Confirm GPU is available before continuing:"
nvidia-smi

# Preflight the gated module for 12 mini-batches before spending hours on
# full runs. This is only a wiring/learning check; its metrics are not reported.
echo "===== GATED PREFLIGHT START $(date) ====="
python -u train_collm_mf_din.py \
  --cfg-path "$CONFIG_DIR/S2_GATED_L31.yaml" \
  --options \
  run.max_epoch=1 \
  run.iters_per_epoch=12 \
  run.warmup_steps=0 \
  run.accum_grad_iters=3 \
  run.lr_scale_other=1.0 \
  run.output_dir="$ROOT/output/SMOKE_GATED_PREFLIGHT" \
  2>&1 | tee "$LOG_DIR/SMOKE_GATED_PREFLIGHT.log"

PREFLIGHT_CKPT="$(find "$ROOT/output/SMOKE_GATED_PREFLIGHT" -type f -name checkpoint_best.pth -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$PREFLIGHT_CKPT" ]]; then
  echo "[ERROR] Gated preflight did not produce checkpoint_best.pth."
  exit 4
fi

python - "$PREFLIGHT_CKPT" <<'PY'
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", {}))
keys = [k for k in state if "collab_gate" in k]
print("[PREFLIGHT CHECKPOINT]", path)
print("[GATE PARAMS]", len(keys))
for k in keys:
    v = state[k].float()
    print(k, "norm=", float(v.norm()), "nonzero=", int(torch.count_nonzero(v)))
if not keys:
    raise SystemExit("[FAIL] collab_gate parameters missing from checkpoint")
total = sum(float(state[k].float().norm()) for k in keys)
if total == 0.0:
    raise SystemExit(
        "[FAIL] collab_gate remained exactly zero after preflight; "
        "full experiments aborted to avoid wasting GPU time."
    )
print("[PASS] Gate parameters changed; full E3 experiments may proceed.")
PY

run_one() {
  local name="$1"
  local mode="$2"
  local config="$CONFIG_DIR/$name.yaml"
  local log="$LOG_DIR/${name}_E3.log"
  local out="$ROOT/output/${name}_E3"

  echo "===== START $name ($mode) $(date) ====="
  python -u train_collm_mf_din.py \
    --cfg-path "$config" \
    --options \
    run.max_epoch=3 \
    run.accum_grad_iters=3 \
    run.lr_scale_other=1.0 \
    "run.output_dir=$out" \
    2>&1 | tee "$log"

  local ckpt
  ckpt="$(find "$out" -type f -name checkpoint_best.pth -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$ckpt" ]]; then
    echo "[ERROR] $name did not produce checkpoint_best.pth."
    return 5
  fi
  echo "[BEST CHECKPOINT] $ckpt"

  # Validate custom module presence and parameter movement in the best checkpoint.
  python - "$ckpt" "$mode" <<'PY'
import sys
import torch

path, mode = sys.argv[1:3]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", {}))
token = {"gated_l31": "collab_gate", "conditional_lora": "collab_conditioner"}.get(mode)
if token:
    keys = [k for k in state if token in k]
    print(f"[MODULE CHECK] mode={mode} keys={len(keys)}")
    for k in keys:
        print(k, "norm=", float(state[k].float().norm()))
    if not keys:
        raise SystemExit(f"[FAIL] Missing {token} in best checkpoint")
    total = sum(float(state[k].float().norm()) for k in keys)
    if total == 0.0:
        raise SystemExit(f"[FAIL] {token} did not update; do not interpret this run as a valid method result")
print("[CHECKPOINT CHECK] PASS")
PY
  echo "===== DONE $name $(date) ====="
}

run_one S2_L31_CONTROL legacy
run_one S2_GATED_L31 gated_l31
run_one S2_CONDITIONAL_LORA conditional_lora

echo "===== ALL THREE E3 RUNS COMPLETED $(date) ====="
echo "Training is validation-only for model selection. Run test evaluation separately on the selected checkpoints."
