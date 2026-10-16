#!/usr/bin/env bash
# Layer-P2 repaired pilot:
#   0) verify historical S2 Valid AUC checkpoint under legacy loading
#   1) train S2-recipe L31 for ONE epoch (scheduler still knows original 3-epoch horizon)
#   2) ONLY IF L31 AUC >=0.70, train matched L24 for one epoch
# Runs Valid ONLY. No Test. Outputs/logs are isolated.
# Usage: nohup env GPU=0 bash run_layer_P2_verified.sh > logs/LAYER_P2_launcher.log 2>&1 < /dev/null &
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
S2="/home/bbx/user_lx/GraphLoRA/output/S2_groups_acc3/20261010223/checkpoint_best.pth"
S2_CFG="tran_configs/S2_groups_acc3_recovered.yaml"
test -f "$S2" || { echo "[P2-ERROR] missing historical S2 checkpoint: $S2"; exit 2; }
test -f "$S2_CFG"
python -m py_compile minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py
python - <<'PY'
import copy,yaml
names=["tran_configs/LAYER_P2_L31.yaml", "tran_configs/LAYER_P2_L24.yaml"]
a,b=[yaml.safe_load(open(x)) for x in names]
for x in [a,b]:
    assert x["model"]["gnn_semantic_align"] is True, "S2 semantic alignment must be enabled"
    assert x["model"]["train_gnn_with_rec_loss"] is True
    assert x["model"]["preserve_source_qv_bias"] is True
    assert x["model"]["ckpt"] is None
    assert x["model"]["lora_gnn"] == "ngcf"
    assert x["run"]["batch_size_train"] == 1
    assert x["run"]["batch_size_eval"] == 1
    assert x["run"]["accum_grad_iters"] == 9
    assert x["run"]["max_epoch"] == 3
    assert x["run"]["stop_after_epoch"] == 1
    assert abs(x["run"]["init_lr"] - 1e-4) < 1e-10
a2,b2=copy.deepcopy(a),copy.deepcopy(b)
la,lb=a2["model"].pop("gcn_layer"),b2["model"].pop("gcn_layer")
a2["run"].pop("output_dir");b2["run"].pop("output_dir")
assert (la,lb)==(31,24), (la,lb)
assert a2==b2, "P2 layer arms differ beyond gcn_layer/output_dir"
print("[P2-STATIC] both arms identical except layer; S2 semantic recipe preserved",flush=True)
PY

extract_and_check () {
  local LOG="$1" LABEL="$2" MIN="$3" REF="${4:-}"
  python - "$LOG" "$LABEL" "$MIN" "$REF" <<'PY'
import sys,re
from pathlib import Path
name,tag,lower,reference=sys.argv[1:]
body=Path(name).read_text(errors="replace")
auc=[float(x) for x in re.findall(r'\*\*\*auc:\s*([0-9.]+)',body)]
uauc=[float(x) for x in re.findall(r'\*\*\*uauc:\s*\(?([0-9.]+)',body)]
if not auc or not uauc:
    raise SystemExit("[P2-ERROR] "+tag+" Valid AUC/UAUC missing in "+name)
a,u=auc[-1],uauc[-1]
print("[P2-METRIC]",tag,"Valid AUC=%.9f"%a,"Valid UAUC=%.9f"%u,flush=True)
if a < float(lower):
    raise SystemExit("[P2-GATE] "+tag+" too low (AUC %.6f < %.3f); STOP without running next experiment"%(a,float(lower)))
if reference:
    expected=float(reference)
    if abs(a-expected)>0.015:
        raise SystemExit("[P2-GATE] historic S2 baseline changed by >0.015; stop to debug loading before retraining")
PY
}

S2_LOG="logs/LAYER_P2_${STAMP}_S2_ckpt_valid.log"
echo "[P2] 0/2: verifying historic S2 on Valid only, expect AUC around 0.73288"
python -u train_collm_mf_din.py --cfg-path "$S2_CFG" --options \
  "model.ckpt=${S2}" "run.evaluate=true" "run.test_splits=[valid]" \
  "run.batch_size_eval=1" "run.output_dir=output/LAYER_P2_${STAMP}_S2_verify" \
  2>&1 | tee "$S2_LOG"
extract_and_check "$S2_LOG" "historical-S2" 0.70 0.7328835632

for L in 31 24; do
  CFG="tran_configs/LAYER_P2_L${L}.yaml"
  LOG="logs/LAYER_P2_${STAMP}_L${L}.log"
  echo "[P2] START L${L} | fresh S2 recipe, bs1 accum9, 1 epoch pilot (3-epoch LR schedule)"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.output_dir=output/LAYER_P2_${STAMP}_L${L}" \
    2>&1 | tee "$LOG"
  grep -Fq "[LAYER-EXACT] decoder layer ${L}: q_proj and v_proj" "$LOG" || {
    echo "[P2-ERROR] exact layer-QV hook assertion missing"; exit 1;
  }
  grep -Fq "[LAYER-BIAS] layer=${L} tag=q source_bias=none" "$LOG" || {
    echo "[P2-ERROR] q_proj source bias was not correctly preserved"; exit 1;
  }
  grep -Fq "[LAYER-BIAS] layer=${L} tag=v source_bias=none" "$LOG" || {
    echo "[P2-ERROR] v_proj source bias was not correctly preserved"; exit 1;
  }
  grep -Fq "[LAYER-PILOT] stopping after 1 completed epoch(s)" "$LOG" || {
    echo "[P2-ERROR] pilot epoch-stop missing"; exit 1;
  }
  if [[ "$L" == 31 ]]; then
    extract_and_check "$LOG" "L31-fresh" 0.70
  else
    extract_and_check "$LOG" "L24-fresh" 0.0
  fi
  echo "[P2] FINISHED L${L} | log: $LOG"
done

echo "[P2] P2 L31 and L24 Valid complete. Compare AUC/UAUC; no Test run."
