#!/usr/bin/env bash
# S2 legacy control: original LR horizon, validation after 501/2001/5001
# mini-batches, and stop after 5001 batches. No gate preflight.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml"
OUT="$ROOT/output/S2_L31_CONTROL_DIAG_V2"
LOG="$ROOT/logs/collab_v1/S2_L31_CONTROL_DIAG_V2.log"

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
if model.get("collab_mode") != "legacy" or float(model.get("gnn_scale", -1)) != 0.1:
    raise SystemExit("[ERROR] Expected legacy S2-L31 with gnn_scale=0.1.")
if run.get("resume_ckpt_path"):
    raise SystemExit("[ERROR] Optimizer resume is enabled; not a fresh-optimizer diagnosis.")
if str(run.get("mid_epoch_eval_iters")) != "500,2000,5000":
    raise SystemExit("[ERROR] Regenerate config to set probes 500,2000,5000.")
if "iters_per_epoch" in run:
    raise SystemExit("[ERROR] Config overrides epoch steps; refusing altered LR horizon.")
print("[CONTROL-DIAG-V2] model weights:", ckpt, flush=True)
print("[CONTROL-DIAG-V2] full LR horizon; 5001-batch cap, checkpoints at indices 500,2000,5000.", flush=True)
PY

if [[ -e "$OUT" ]] && find "$OUT" -type f -name 'checkpoint*.pth' -print -quit | grep -q .; then
  echo "[ERROR] $OUT already contains checkpoints; refusing overwrite." >&2
  exit 5
fi

mkdir -p "$OUT" "$(dirname "$LOG")"
export GRAPHLoRA_DEBUG_OPTIMIZER=0
echo "[CONTROL-DIAG-V2] output=$OUT log=$LOG"
python -u train_collm_mf_din.py \
  --cfg-path "$CONFIG" \
  --options \
  run.max_epoch=3 \
  run.max_train_steps=5001 \
  run.diagnostic_stop_after_epoch0=True \
  run.skip_validation=True \
  run.accum_grad_iters=3 \
  "run.output_dir=$OUT" \
  2>&1 | tee "$LOG"

echo "[CONTROL-DIAG-V2] Saved checkpoints:"
find "$OUT" -type f -name 'checkpoint_ep0_step*.pth' -printf '%p\n' | sort -V
echo "[CONTROL-DIAG-V2] AUC summary:"
grep -E '\[mid-epoch-eval\] step .* agg=|\[CONTROL-DIAG\]|rank_0 auc:' "$LOG" || true
