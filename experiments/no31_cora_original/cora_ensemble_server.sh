#!/usr/bin/env bash
# CPU-only entrypoint. The three existing model checkpoints are not modified.
set -euo pipefail

ROOT="${CORA_ROOT:-/home/bbx/user_lx/CoRA_official}"
cd "$ROOT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
TAG="${ENSEMBLE_TAG:-CORA_ENSEMBLE_REPRO_20261113}"
NORM="${ENSEMBLE_NORM:-test_first128}"
SEL="out/$TAG/selection.json"
RESULT="out/$TAG/TEST"
RR5_CKPT="${RR5_CKPT:-experiments/no31_cora_original/output/CORA_RESIDUAL_ROUTING6_20261103/RR5_USER_GATE_R31/20261103153/checkpoint_best.pth}"

need_path() {
  local key="$1"
  local value="${!key:-}"
  if [[ -z "$value" ]]; then
    echo "[INPUT MISSING] export ${key}=/absolute/path/to/${key,,}.npz" >&2
    exit 2
  fi
  if [[ ! -f "$value" ]]; then
    echo "[FILE MISSING] ${key}=${value}" >&2
    exit 2
  fi
}
if [[ $# -ne 1 || ( "$1" != "select" && "$1" != "test" ) ]]; then
  echo "Usage: $0 select|test" >&2
  echo "Export A_VALID,B_VALID,C_VALID for select; A_TEST,B_TEST,C_TEST for test." >&2
  exit 2
fi

if [[ ! -f "$HERE/cora_ensemble_repro_improve.py" ]]; then
  echo "[FILE MISSING] $HERE/cora_ensemble_repro_improve.py" >&2
  exit 2
fi
if [[ ! -f "$RR5_CKPT" ]]; then
  echo "[CHECKPOINT MISSING] $RR5_CKPT" >&2
  exit 2
fi

# CPU only, no LLM, no training, no extra GPU allocation.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ "$1" == "select" ]]; then
  need_path A_VALID
  need_path B_VALID
  need_path C_VALID
  echo "[SELECT] seed=42; norm=$NORM; locked output=$SEL"
  exec "$PYTHON" -u "$HERE/cora_ensemble_repro_improve.py" \
    --phase select --a-valid "$A_VALID" --b-valid "$B_VALID" --c-valid "$C_VALID" \
    --rr5-checkpoint "$RR5_CKPT" --norm "$NORM" --selection "$SEL"
fi

need_path A_TEST
need_path B_TEST
need_path C_TEST
if [[ ! -f "$SEL" ]]; then
  echo "[LOCKED VALID SELECTION MISSING] $SEL; run select first" >&2
  exit 2
fi
echo "[TEST ONCE] seed=42; frozen selection=$SEL; output=$RESULT"
exec "$PYTHON" -u "$HERE/cora_ensemble_repro_improve.py" \
  --phase test --a-test "$A_TEST" --b-test "$B_TEST" --c-test "$C_TEST" \
  --rr5-checkpoint "$RR5_CKPT" --selection "$SEL" --output-dir "$RESULT"
