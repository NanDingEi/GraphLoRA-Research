#!/usr/bin/env bash
# Target-aware candidate/user/history GNN fusion at the final Yes/No decision.
# Warm start S2-L31, KEEP historical injection, zero-init new residual.
# Preflight (24 batches) checks real gradient updates; then one complete epoch.
# Eval once on valid_small at the end; NO test set, NO mid-epoch evaluations.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if pgrep -af '[t]rain_collm_mf_din.py' >/dev/null; then
  echo "[ERROR] Training process is already running; only gpu0, serial experiments." >&2
  pgrep -af '[t]rain_collm_mf_din.py' >&2 || true
  exit 2
fi
if ! grep -q 'class TargetAwareDecisionFusion' minigpt4/models/minigpt4rec_v2.py; then
  echo "[ERROR] Old checkout; fetch the latest fix/collab-gated-optimizer branch." >&2
  exit 3
fi

python -m py_compile minigpt4/models/minigpt4rec_v2.py
python experiments/collab_injection_v1/test_target_fusion_cpu.py
python experiments/collab_injection_v1/prepare_e3_configs.py

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="$ROOT/output/S2_TARGET_FUSION_${STAMP}"
ROOT_LOG="$ROOT/logs/collab_v1/S2_TARGET_FUSION_${STAMP}"
mkdir -p "$ROOT_OUT" "$ROOT_LOG"

python - "$ROOT_OUT" "$ROOT_LOG" <<'PY'
import sys
from pathlib import Path
import yaml

out, logs = map(Path, sys.argv[1:3])
src = Path("tran_configs/collab_v1/e3/S2_TARGET_FUSION.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
m, r = cfg["model"], cfg["run"]
if (m.get("collab_mode") != "target_fusion"
    or int(m.get("gcn_layer", -1)) != 31
    or float(m.get("gnn_scale", -1)) != 0.1):
    raise SystemExit("[FAIL] Wrong target fusion model configuration")
if not Path(m.get("ckpt", "")).is_file():
    raise SystemExit("[FAIL] Missing historical S2 champion checkpoint")
if m.get("rel_inject") or m.get("rel_rank_mod"):
    raise SystemExit("[FAIL] Relation modules must remain disabled")
if r.get("resume_ckpt_path"):
    raise SystemExit("[FAIL] Do not resume optimizer on warm-start experiment")
r["seed"] = 42
r["max_epoch"] = 3  # 3-epoch scheduler horizon preserved
r["diagnostic_stop_after_epoch0"] = True  # one real epoch only
r["skip_validation"] = False
r["evaluate"] = False
r["save_checkpoint_every_n_epochs"] = 1
r["accum_grad_iters"] = 3
r["init_lr"] = 1e-5
r["min_lr"] = 5e-6
r["warmup_lr"] = 1e-6
r["output_dir"] = str(out / "FULL_E1")
for key in ("max_train_steps", "iters_per_epoch", "max_iters", "mid_epoch_eval_iters"):
    r.pop(key, None)
path = logs / "FULL_E1.yaml"
path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8")

r["max_train_steps"] = 24
r["skip_validation"] = True
r["output_dir"] = str(out / "PREFLIGHT")
smoke = logs / "PREFLIGHT.yaml"
smoke.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")
print("[TARGET-RUN] champion:", m["ckpt"], flush=True)
print("[TARGET-RUN] preflight: 24 minibatches; full: 1 epoch (11297 minibatches)", flush=True)
print("[TARGET-RUN] init_lr=1e-5, min_lr=5e-6, warmup_lr=1e-6, seed=42", flush=True)
print("[TARGET-RUN] new fusion starts as exact zero residual; historical L31 injection stays", flush=True)
print("[TARGET-RUN] output:", out, flush=True)
print("[TARGET-RUN] logs:", logs, flush=True)
PY

echo "[TARGET-RUN] Smoke test starts"
python -u train_collm_mf_din.py --cfg-path "$ROOT_LOG/PREFLIGHT.yaml" \
  2>&1 | tee "$ROOT_LOG/PREFLIGHT.log"

python - "$ROOT_OUT/PREFLIGHT" <<'PY'
import sys
from pathlib import Path
import torch

files = list(Path(sys.argv[1]).rglob("checkpoint_0.pth"))
if len(files) != 1:
    raise SystemExit("[FAIL] Preflight checkpoint_0.pth not uniquely found")
checkpoint = torch.load(str(files[0]), map_location="cpu", weights_only=False)
state = checkpoint["model"]
suffix = "target_fusion.to_hidden.weight"
keys = [k for k in state if k.endswith(suffix)]
if len(keys) != 1:
    raise SystemExit(f"[FAIL] Missing target fusion output projector: {keys}")
w = state[keys[0]].float()
norm = float(w.norm())
print(f"[TARGET-PREFLIGHT] fusion output projector norm after 24 batches: {norm:.9g}", flush=True)
if not torch.isfinite(w).all() or norm <= 0:
    raise SystemExit("[FAIL] Target fusion output projector never updated")
print("[TARGET-PREFLIGHT] PASS, gradient reaches new decision fusion", flush=True)
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[TARGET-RUN] preflight-only mode complete"
  exit 0
fi

echo "[TARGET-RUN] Full one-epoch train starts"
python -u train_collm_mf_din.py --cfg-path "$ROOT_LOG/FULL_E1.yaml" \
  2>&1 | tee "$ROOT_LOG/FULL_E1.log"

python - "$ROOT_OUT/FULL_E1" "$ROOT_LOG/FULL_E1.log" <<'PY'
import re
import sys
from pathlib import Path

out, log = map(Path, sys.argv[1:3])
checkpoints = list(out.rglob("checkpoint_best.pth"))
if len(checkpoints) != 1:
    raise SystemExit("[FAIL] Missing best checkpoint after full-epoch validation")
auc = []
pattern = re.compile(r"\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\(\s*([0-9.]+)")
for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
    if "Averaged stats:" in line:
        match = pattern.search(line)
        if match:
            auc.append((float(match.group(1)), float(match.group(2))))
if len(auc) != 1:
    raise SystemExit(f"[FAIL] Expected exactly 1 full valid AUC/UAUC, got {len(auc)}")
print("[TARGET-RESULT] Valid AUC={:.9f} UAUC={:.9f}".format(*auc[0]), flush=True)
print("[TARGET-RESULT] best checkpoint:", checkpoints[0], flush=True)
PY

echo "[TARGET-RUN] Complete; no test-set evaluation was run."
