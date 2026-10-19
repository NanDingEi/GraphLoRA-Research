#!/usr/bin/env bash
# S2 Gate vs V-only, 8 epoch total: (2 modes x 2 seeds x 2 full epochs).
# From the SAME S2-L31 checkpoint; low learning rate and 3-epoch LR horizon.
# Four arms run SERIAL, with valid AUC/UAUC and trainable-only checkpoints
# after EVERY epoch. No mid-epoch eval and NO test-set evaluation.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

if pgrep -af '[t]rain_collm_mf_din.py' >/dev/null; then
  echo "[ERROR] Existing training process detected; do not overlap jobs." >&2
  pgrep -af '[t]rain_collm_mf_din.py' >&2 || true
  exit 2
fi
if ! grep -q 'stop_after_n_epochs' minigpt4/runners/runner_base.py ||
   ! grep -q 'with torch.random.fork_rng(devices=\[\]):' minigpt4/models/minigpt4rec_v2.py ||
   ! grep -q '"v_only"' minigpt4/models/minigpt4rec_v2.py; then
  echo "[ERROR] Stale checkout; fetch latest fix/collab-gated-optimizer." >&2
  exit 3
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
JOB="S2_GATE_VONLY_8EP_${STAMP}"
LOG_DIR="$ROOT/logs/collab_v1/$JOB"
OUT_DIR="$ROOT/output/$JOB"
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "[8EP] job=$JOB"
echo "[8EP] logs=$LOG_DIR"
echo "[8EP] outputs=$OUT_DIR"
python experiments/collab_injection_v1/prepare_e3_configs.py \
  > "$LOG_DIR/prepare.log" 2>&1 || { cat "$LOG_DIR/prepare.log"; exit 4; }

python - "$ROOT" "$LOG_DIR" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

root, log_dir, out_dir = map(Path, sys.argv[1:4])
src_dir = root / "tran_configs/collab_v1/e3"
arms = [
    ("GATED_S42", "S2_GATED_L31.yaml", "gated_l31", 42),
    ("VONLY_S42", "S2_V_ONLY_L31.yaml", "v_only", 42),
    ("GATED_S43", "S2_GATED_L31.yaml", "gated_l31", 43),
    ("VONLY_S43", "S2_V_ONLY_L31.yaml", "v_only", 43),
]
champions = set()
for arm, file, mode, seed in arms:
    cfg = yaml.safe_load((src_dir / file).read_text(encoding="utf-8"))
    model, run = cfg["model"], cfg["run"]
    if model.get("collab_mode") != mode:
        raise SystemExit(f"[ERROR] Wrong mode for {arm}: {model.get('collab_mode')}")
    if int(model.get("gcn_layer", -1)) != 31 or float(model.get("gnn_scale", -1)) != 0.1:
        raise SystemExit(f"[ERROR] Invalid L31 injection for {arm}")
    if model.get("rel_inject") or model.get("rel_rank_mod"):
        raise SystemExit(f"[ERROR] Relation modules unexpectedly enabled for {arm}")
    ckpt = Path(model.get("ckpt", ""))
    if not ckpt.is_file():
        raise SystemExit(f"[ERROR] Missing champion checkpoint: {ckpt}")
    champions.add(str(ckpt.resolve()))
    run["seed"] = seed
    run["max_epoch"] = 3  # LR horizon; actual training stops after two epochs
    run["stop_after_n_epochs"] = 2
    run["diagnostic_stop_after_epoch0"] = False
    run["evaluate"] = False
    run["skip_validation"] = False
    run["resume_ckpt_path"] = None
    run["accum_grad_iters"] = 3
    run["init_lr"] = 1e-5
    run["min_lr"] = 5e-6
    run["warmup_lr"] = 1e-6
    run["save_checkpoint_every_n_epochs"] = 1
    run["output_dir"] = str(out_dir / arm)
    for key in ("max_train_steps", "iters_per_epoch", "max_iters", "mid_epoch_eval_iters"):
        run.pop(key, None)
    (log_dir / f"{arm}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[8EP-CONFIG] {arm} mode={mode} seed={seed} "
          f"real_epochs=2 full_steps_per_epoch=11297 "
          f"LR=1e-5/5e-6/1e-6", flush=True)
if len(champions) != 1:
    raise SystemExit(f"[ERROR] Arms have different initial checkpoints: {champions}")
print("[8EP] SAME initial checkpoint:", next(iter(champions)), flush=True)
PY

SUMMARY="$LOG_DIR/summary.tsv"
printf 'arm\tepoch\tauc\tuauc\tbest_checkpoint\n' > "$SUMMARY"

for ARM in GATED_S42 VONLY_S42 GATED_S43 VONLY_S43; do
  echo "[8EP-START] $ARM $(date -Is)"
  echo "[8EP-START] $ARM $(date -Is)" >> "$LOG_DIR/status.txt"
  python -u train_collm_mf_din.py \
    --cfg-path "$LOG_DIR/$ARM.yaml" \
    2>&1 | tee "$LOG_DIR/$ARM.log"

  python - "$LOG_DIR/$ARM.log" "$ARM" "$SUMMARY" "$OUT_DIR/$ARM" <<'PY'
import re
import sys
from pathlib import Path

log_path, arm, summary_path, out_dir = sys.argv[1:5]
num = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
pattern = re.compile(r"\*\*\*auc:\s*" + num + r".*?\*\*\*uauc:\(\s*" + num)
scores = []
with open(log_path, encoding="utf-8", errors="replace") as stream:
    for line in stream:
        if "Averaged stats:" in line:
            match = pattern.search(line)
            if match:
                scores.append((float(match.group(1)), float(match.group(2))))
if len(scores) != 2:
    raise SystemExit(f"[FAIL] {arm}: expected 2 end-of-epoch AUC/UAUC records; got {len(scores)}")
out = Path(out_dir)
best = sorted(out.rglob("checkpoint_best.pth"))
for epoch in (0, 1):
    if not list(out.rglob(f"checkpoint_{epoch}.pth")):
        raise SystemExit(f"[FAIL] {arm}: checkpoint_{epoch}.pth missing")
if len(best) != 1:
    raise SystemExit(f"[FAIL] {arm}: expected 1 checkpoint_best.pth, got {len(best)}")
with open(summary_path, "a", encoding="utf-8") as f:
    for epoch, (auc, uauc) in enumerate(scores):
        f.write(f"{arm}\t{epoch}\t{auc:.9f}\t{uauc:.9f}\t{best[0]}\n")
        print(f"[8EP-VALID] {arm} epoch={epoch} AUC={auc:.9f} UAUC={uauc:.9f}", flush=True)
print("[8EP-SAVED] best", best[0], flush=True)
PY

  echo "[8EP-DONE] $ARM $(date -Is)" | tee -a "$LOG_DIR/status.txt"
done

echo "[8EP-COMPLETE] all four arms / eight epochs finished."
cat "$SUMMARY"
echo "[8EP] summary=$SUMMARY"
