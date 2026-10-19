#!/usr/bin/env bash
# Gated-L31 warm-start from S2 champion, low-LR, ONE COMPLETE EPOCH.
# Scheduler preserves its original three-epoch horizon for fair comparison;
# diagnostic_stop_after_epoch0=True stops after exactly one actual epoch.
# One full valid evaluation at the end; no mid-epoch probes/preflight.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOURCE_CONFIG="tran_configs/collab_v1/e3/S2_GATED_L31.yaml"
RUN_CONFIG="tran_configs/collab_v1/e3/S2_GATED_L31_LOWLR_E1.yaml"
OUT="$ROOT/output/S2_GATED_L31_LOWLR_E1"
LOG="$ROOT/logs/collab_v1/S2_GATED_L31_LOWLR_E1.log"

if pgrep -af '[t]rain_collm_mf_din.py' >/dev/null; then
  echo "[ERROR] Another GraphLoRA training process is running." >&2
  pgrep -af '[t]rain_collm_mf_din.py' >&2 || true
  exit 3
fi
if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "[ERROR] Missing $SOURCE_CONFIG; run prepare_e3_configs.py first." >&2
  exit 2
fi
if ! grep -q 'diagnostic_stop_after_epoch0' minigpt4/runners/runner_base.py; then
  echo "[ERROR] Stale code checkout; update fix/collab-gated-optimizer." >&2
  exit 4
fi
if [[ -d "$OUT" ]] && find "$OUT" -type f -name 'checkpoint*.pth' -print -quit | grep -q .; then
  echo "[ERROR] Output already contains checkpoints: $OUT; refusing overwrite." >&2
  exit 5
fi

python - "$SOURCE_CONFIG" "$RUN_CONFIG" "$OUT" <<'PY'
import sys
from pathlib import Path
import yaml

source, target, output = map(Path, sys.argv[1:4])
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
model, run = cfg["model"], cfg["run"]
checkpoint = model.get("ckpt")
if not checkpoint or not Path(checkpoint).is_file():
    raise SystemExit("[ERROR] Missing historical S2 checkpoint at model.ckpt.")
if model.get("collab_mode") != "gated_l31" or float(model.get("gnn_scale", -1)) != 0.1:
    raise SystemExit("[ERROR] Expected gated_l31 at GNN scale 0.1.")
if run.get("resume_ckpt_path"):
    raise SystemExit("[ERROR] Optimizer resume is not allowed for this comparison.")
if int(run.get("seed", -1)) != 42:
    raise SystemExit("[ERROR] Expected seed 42.")

# Preserve scheduler 3-epoch horizon of matched control, but run ONLY epoch 0.
run["max_epoch"] = 3
run["diagnostic_stop_after_epoch0"] = True
run["skip_validation"] = False
run["evaluate"] = False
run["resume_ckpt_path"] = None
run.pop("max_train_steps", None)
run.pop("iters_per_epoch", None)
run.pop("max_iters", None)
run.pop("mid_epoch_eval_iters", None)
run["init_lr"] = 1e-5
run["min_lr"] = 5e-6
run["warmup_lr"] = 1e-6
run["accum_grad_iters"] = 3
run["save_checkpoint_every_n_epochs"] = 1
run["output_dir"] = str(output)

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("[GATED-E1] checkpoint:", checkpoint, flush=True)
print("[GATED-E1] 1 complete epoch (11,297 batches), full validation only at end.", flush=True)
print("[GATED-E1] no max_train_steps and no mid-epoch evaluation.", flush=True)
print("[GATED-E1] LR: init=1e-5 min=5e-6 warmup=1e-6; scheduler 3-epoch horizon.", flush=True)
print("[GATED-E1] config:", target, flush=True)
PY

mkdir -p "$OUT" "$(dirname "$LOG")"
export GRAPHLoRA_DEBUG_OPTIMIZER=0

python -u train_collm_mf_din.py --cfg-path "$RUN_CONFIG" 2>&1 | tee "$LOG"

CKPT="$(find "$OUT" -type f -name checkpoint_best.pth -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$CKPT" ]]; then
  echo "[ERROR] One-epoch run did not produce checkpoint_best.pth." >&2
  exit 6
fi
echo "[GATED-E1] best checkpoint: $CKPT"
python - "$CKPT" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = checkpoint["model"]
keys = sorted(k for k in state if ".v_proj.collab_gate." in k)
print("[GATE-CHECK] keys:", keys)
if len(keys) != 2:
    raise SystemExit("[FAIL] V-only gate weight/bias missing.")
for key in keys:
    print("[GATE-CHECK]", key, "norm=", float(state[key].float().norm()))
if sum(float(state[key].float().norm()) for key in keys) == 0:
    raise SystemExit("[FAIL] Gate remains untrained.")
print("[GATE-CHECK] PASS")
PY
echo "[GATED-E1] validation metrics:"
grep -E 'Averaged stats:.*\*\*\*auc:|rank_0 auc:|Training time|\[GATE-CHECK\]' "$LOG" | cut -c1-240 || true
