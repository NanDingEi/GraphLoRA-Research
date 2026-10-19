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

echo "[FIX-BRANCH] collab-gated-optimizer / optimizer-diagnostic-v2"
echo "[INFO] Repository: $ROOT"

# Fail closed if this checkout cannot emit the optimizer-step diagnostics.
if ! grep -q '\[OPT-UPDATE-DIAG\]' minigpt4/tasks/base_task.py; then
  echo "[ERROR] This checkout lacks optimizer-step diagnostics in minigpt4/tasks/base_task.py."
  echo "[ERROR] Fetch/check out fix/collab-gated-optimizer before training."
  exit 20
fi
if ! grep -q 'amp_init_scale' experiments/collab_injection_v1/prepare_e3_configs.py; then
  echo "[ERROR] Config generator lacks amp_init_scale support; refusing stale run."
  exit 21
fi

python - "$CONFIG_DIR/S2_GATED_L31.yaml" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
run = cfg.get("run", {})
print("[FIX-CONFIG] collab_mode=", cfg.get("model", {}).get("collab_mode"))
print("[FIX-CONFIG] amp=", run.get("amp"))
print("[FIX-CONFIG] amp_init_scale=", run.get("amp_init_scale"))
print("[FIX-CONFIG] accum_grad_iters=", run.get("accum_grad_iters"))
if float(run.get("amp_init_scale", 0)) != 1024.0:
    raise SystemExit("[ERROR] Expected amp_init_scale=1024. Regenerate configs before running.")
PY

echo "[INFO] Confirm GPU is available before continuing:"
nvidia-smi

# Preflight the gated module for 12 mini-batches before spending hours on
# full runs. This is only a wiring/learning check; its metrics are not reported.
# Remove stale best checkpoints so a failed fresh run cannot accidentally
# validate an earlier smoke-test artifact.
mkdir -p "$ROOT/output/SMOKE_GATED_PREFLIGHT"
find "$ROOT/output/SMOKE_GATED_PREFLIGHT" -type f -name 'checkpoint*.pth' -delete

echo "===== GATED PREFLIGHT START $(date) ====="
python -u train_collm_mf_din.py \
  --cfg-path "$CONFIG_DIR/S2_GATED_L31.yaml" \
  --options \
  run.max_epoch=1 \
  run.iters_per_epoch=12 \
  run.warmup_steps=0 \
  run.accum_grad_iters=3 \
  run.lr_scale_other=1.0 \
  run.skip_validation=True \
  run.output_dir="$ROOT/output/SMOKE_GATED_PREFLIGHT" \
  2>&1 | tee "$LOG_DIR/SMOKE_GATED_PREFLIGHT.log"

PREFLIGHT_CKPT="$(find "$ROOT/output/SMOKE_GATED_PREFLIGHT" -type f \( -name checkpoint_0.pth -o -name checkpoint_best.pth \\) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$PREFLIGHT_CKPT" ]]; then
  echo "[ERROR] Gated preflight did not produce checkpoint_0.pth/checkpoint_best.pth."
  exit 4
fi

python - "$PREFLIGHT_CKPT" <<'PY'
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("state_dict", {}))
gate_keys = [k for k in state if "collab_gate" in k]
lora_b_keys = [
    k for k in state
    if "layers.31.self_attn." in k and "lora_B" in k
    and (".q_proj." in k or ".v_proj." in k)
]
print("[PREFLIGHT CHECKPOINT]", path)
print("[GATE PARAMS]", len(gate_keys))
for k in gate_keys:
    v = state[k].float()
    print("[GATE]", k, "norm=", float(v.norm()), "nonzero=", int(torch.count_nonzero(v)))
print("[LAYER31 LORA_B PARAMS]", len(lora_b_keys))
for k in lora_b_keys:
    v = state[k].float()
    print("[LORA-B]", k, "norm=", float(v.norm()), "nonzero=", int(torch.count_nonzero(v)))
if not gate_keys:
    raise SystemExit("[FAIL] collab_gate parameters missing from checkpoint")
if len(gate_keys) != 2 or any(".v_proj." not in k for k in gate_keys):
    raise SystemExit(
        "[FAIL] Expected exactly v_proj.collab_gate.weight/bias; "
        "the Q gate should not exist because prefix-position Q injection is causally ineffective."
    )
if not lora_b_keys:
    raise SystemExit("[FAIL] layer-31 q/v LoRA-B parameters missing; cannot diagnose gate")
gate_total = sum(float(state[k].float().norm()) for k in gate_keys)
lora_b_total = sum(float(state[k].float().norm()) for k in lora_b_keys)
if lora_b_total == 0.0:
    raise SystemExit(
        "[FAIL-ROOT] Layer-31 LoRA-B is still exactly zero. "
        "The gate's zero gradient is likely downstream of LoRA-B; "
        "inspect [OPT-UPDATE-DIAG] for skipped optimizer steps / zero parameter deltas. "
        "Full experiments aborted."
    )
if gate_total == 0.0:
    raise SystemExit(
        "[FAIL-GATE] LoRA-B changed but collab_gate remained zero. "
        "Inspect gate gradients and optimizer-group membership. Full experiments aborted."
    )
print("[PASS] Both layer-31 LoRA-B and gate parameters changed.")
PY

if [[ "${RUN_FULL_E3_AFTER_PREFLIGHT:-0}" != "1" ]]; then
  echo "[STOP] Preflight passed. Full 3-epoch runs were NOT started."
  echo "[NEXT] Inspect $LOG_DIR/SMOKE_GATED_PREFLIGHT.log for [OPT-UPDATE-DIAG]."
  echo "[NEXT] If LoRA-B and gate deltas are nonzero and found_inf=0, rerun with:"
  echo "       RUN_FULL_E3_AFTER_PREFLIGHT=1 bash experiments/collab_injection_v1/run_e3.sh"
  exit 0
fi

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
