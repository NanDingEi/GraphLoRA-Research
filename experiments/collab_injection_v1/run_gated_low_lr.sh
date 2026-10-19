#!/usr/bin/env bash
# S2 gated L31: reduced LR, same full scheduler horizon, validation after 501/2001/5001
# mini-batches, and stop after 5001 batches. Gate already passed its 12-batch preflight.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="tran_configs/collab_v1/e3/S2_GATED_L31.yaml"
OUT="$ROOT/output/S2_GATED_L31_LOWLR_DIAG"
LOG="$ROOT/logs/collab_v1/S2_GATED_L31_LOWLR_DIAG.log"

if pgrep -af '[t]rain_collm_mf_din.py' >/dev/null; then
  echo "[ERROR] Another train_collm_mf_din.py process is running." >&2
  pgrep -af '[t]rain_collm_mf_din.py' >&2 || true
  exit 3
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing $CONFIG. Run prepare_e3_configs.py first." >&2
  exit 2
fi

if ! grep -q 'max_train_steps=self.config.run_cfg.get' minigpt4/runners/runner_base.py ||
   ! grep -q 'effective_iters = min(max_train_steps, iters_per_epoch)' minigpt4/tasks/base_task.py; then
  echo "[ERROR] Stale checkout would alter the LR horizon." >&2
  echo "[ERROR] Fetch fix/collab-gated-optimizer before continuing." >&2
  exit 4
fi

python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
model, run = config["model"], config["run"]
ckpt = model.get("ckpt")
if not ckpt or not Path(ckpt).is_file():
    raise SystemExit("[ERROR] No valid warm-start checkpoint at model.ckpt.")
if model.get("collab_mode") != "gated_l31" or float(model.get("gnn_scale", -1)) != 0.1:
    raise SystemExit("[ERROR] Expected gated_l31 with gnn_scale=0.1.")
if run.get("resume_ckpt_path"):
    raise SystemExit("[ERROR] Optimizer resume is enabled; not a fresh-optimizer diagnosis.")
if str(run.get("mid_epoch_eval_iters")) != "500,2000,5000":
    raise SystemExit("[ERROR] Regenerate config to set probes 500,2000,5000.")
if "iters_per_epoch" in run:
    raise SystemExit("[ERROR] Config overrides epoch steps; refusing altered LR horizon.")
if int(run.get("seed", -1)) != 42:
    raise SystemExit("[ERROR] Expected seed 42 for matched comparison.")
print("[GATED-LOWLR-DIAG] model weights:", ckpt, flush=True)
print("[GATED-LOWLR-DIAG] full LR horizon; 5001-batch cap; probes 500,2000,5000.", flush=True)
print("[GATED-LOWLR-DIAG] overrides: init_lr=1e-5 min_lr=5e-6 warmup_lr=1e-6.", flush=True)
PY

if [[ -e "$OUT" ]] && find "$OUT" -type f -name 'checkpoint*.pth' -print -quit | grep -q .; then
  echo "[ERROR] $OUT already contains checkpoints; refusing overwrite." >&2
  exit 5
fi

mkdir -p "$OUT" "$(dirname "$LOG")"
export GRAPHLoRA_DEBUG_OPTIMIZER=0
echo "[GATED-LOWLR-DIAG] output=$OUT log=$LOG"
python -u train_collm_mf_din.py \
  --cfg-path "$CONFIG" \
  --options \
  run.max_epoch=3 \
  run.max_train_steps=5001 \
  run.diagnostic_stop_after_epoch0=True \
  run.skip_validation=True \
  run.accum_grad_iters=3 \
  run.init_lr=0.00001 \
  run.min_lr=0.000005 \
  run.warmup_lr=0.000001 \
  "run.output_dir=$OUT" \
  2>&1 | tee "$LOG"

echo "[GATED-LOWLR-DIAG] Saved checkpoints:"
find "$OUT" -type f -name 'checkpoint_ep0_step*.pth' -printf '%p\n' | sort -V
echo "[GATED-LOWLR-DIAG] AUC summary:"
grep -E '\[mid-epoch-eval\] step .* agg=|\[CONTROL-DIAG\]|rank_0 auc:' "$LOG" || true

echo "[GATED-LOWLR-DIAG] Gate parameter check at 500-batch checkpoint:"
python - "$OUT" <<'PY'
import sys
from pathlib import Path
import torch
checkpoints = sorted(Path(sys.argv[1]).rglob("checkpoint_ep0_step500.pth"))
if len(checkpoints) != 1:
    raise SystemExit("[FAIL] Expected exactly one step-500 checkpoint.")
ckpt = torch.load(str(checkpoints[0]), map_location="cpu", weights_only=False)
state = ckpt["model"]
keys = [k for k in state if ".v_proj.collab_gate." in k]
print("[GATE-CHECK] keys:", keys)
if len(keys) != 2:
    raise SystemExit("[FAIL] Gate weight and bias missing from step-500 checkpoint.")
for k in keys:
    v = state[k].float()
    print(f"[GATE-CHECK] {k}: norm={float(v.norm()):.10g}")
if sum(float(state[k].float().norm()) for k in keys) == 0:
    raise SystemExit("[FAIL] Gated-L31 parameters did not move.")
print("[GATE-CHECK] PASS")
PY
