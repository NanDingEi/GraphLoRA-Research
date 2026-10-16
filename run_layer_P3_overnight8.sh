#!/usr/bin/env bash
# Overnight 8 new single-layer GraphLoRA ID-Q/V experiments; STRICTLY SERIAL.
# Baseline anchors: P2 L31 AUC 0.714338 / UAUC 0.677975; L24 0.623100 / 0.582282.
# nohup env GPU=0 bash run_layer_P3_overnight8.sh > logs/LAYER_P3_launcher.log 2>&1 < /dev/null &
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
mkdir -p logs
exec 9> logs/LAYER_P3_serial.lock
flock -n 9 || { echo "[P3-FATAL] another P3 launcher is active"; exit 3; }
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="logs/LAYER_P3_${STAMP}"
mkdir -p "$RUN_DIR"
SUMMARY="${RUN_DIR}/summary.tsv"
LAYERS=(30 29 28 27 26 20 16 8)
printf 'layer\tvalid_auc\tvalid_uauc\ttrain_loss\tvalid_loss\tq_R_med\tv_R_med\tq_C_med\tv_C_med\tq_loraA_norm\tv_loraA_norm\tlog\n' > "$SUMMARY"
echo "[P3] GPU=${CUDA_VISIBLE_DEVICES}  stamp=$STAMP  sequence=${LAYERS[*]}  commit=$(git rev-parse --short HEAD)"
echo "[P3] Strictly serial: one FULL epoch per layer; Valid only, no Test."
echo "[P3] Result file: $SUMMARY"
python -m py_compile minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py

# Preflight asserts only location/output-dir differ from already checked P2-L31.
python - <<'PY'
import copy,yaml
from pathlib import Path
layers=[30,29,28,27,26,20,16,8]
r=yaml.safe_load(Path("tran_configs/LAYER_P2_L31.yaml").read_text())
def normalized(x):
    x=copy.deepcopy(x)
    x["model"]["gcn_layer"]=-1
    x["run"]["output_dir"]="IGNORED"
    return x
assert r["model"]["gcn_layer"]==31
for layer in layers:
    p=Path(f"tran_configs/LAYER_P3_L{layer}.yaml")
    x=yaml.safe_load(p.read_text())
    assert x["model"]["gcn_layer"]==layer,p
    assert normalized(x)==normalized(r),f"Unexpected differences from P2-L31 in {p}"
    assert x["model"].get("d1_mode","baseline")=="baseline"
    assert x["model"]["preserve_source_qv_bias"] is True
    assert x["model"]["gnn_semantic_align"] is True
    assert x["model"]["train_gnn_with_rec_loss"] is True
    assert x["run"]["batch_size_train"]==1
    assert x["run"]["batch_size_eval"]==4
    assert x["run"]["accum_grad_iters"]==9
    assert x["run"]["max_epoch"]==3
    assert x["run"]["stop_after_epoch"]==1
    assert x["run"]["seed"]==42
    assert abs(x["run"]["init_lr"]-1e-4)<1e-12
    assert abs(x["model"]["gnn_scale"]-0.1)<1e-12
print("[P3-STATIC] PASS: 8 matched P2 configs, location/output-dir only",flush=True)
PY

for i in "${!LAYERS[@]}"; do
  L="${LAYERS[$i]}"
  IDX="$((i+1))"
  CFG="tran_configs/LAYER_P3_L${L}.yaml"
  LOG="${RUN_DIR}/L${L}.log"
  OUT="output/LAYER_P3_${STAMP}_L${L}"
  echo "[P3-START] ${IDX}/8 L${L} $(date -Is)"
  # pipefail ensures crashes are not swallowed by tee; stop on fatal errors.
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.output_dir=${OUT}" 2>&1 | tee "$LOG"
  grep -Fq "[LAYER-EXACT] decoder layer ${L}: q_proj and v_proj" "$LOG" || {
    echo "[P3-FATAL] expected layer hook missing, L${L}"; exit 10;
  }
  grep -Fq "[LAYER-BIAS] layer=${L} tag=q source_bias=none" "$LOG" || {
    echo "[P3-FATAL] original q bias not preserved, L${L}"; exit 11;
  }
  grep -Fq "[LAYER-BIAS] layer=${L} tag=v source_bias=none" "$LOG" || {
    echo "[P3-FATAL] original v bias not preserved, L${L}"; exit 12;
  }
  grep -Fq "[LAYER-PILOT] stopping after 1 completed epoch(s)" "$LOG" || {
    echo "[P3-FATAL] completed pilot marker missing, L${L}"; exit 13;
  }
  python - "$LOG" "$L" "$SUMMARY" <<'PY'
import re,sys
from pathlib import Path
log,layer,summary=sys.argv[1:]
lines=Path(log).read_text(errors="replace").splitlines()
valid=[x for x in lines if "Averaged stats:" in x and "***auc:" in x and "***uauc:" in x]
train=[x for x in lines if "Averaged stats:" in x and "***auc:" not in x and "loss:" in x]
if len(valid)!=1 or not train:
    raise SystemExit(f"[P3-FATAL] L{layer}: need exactly one full Valid AUC and one train loss; got {len(valid)}/{len(train)}")
def val(pattern,s,label):
    m=re.search(pattern,s)
    if not m: raise SystemExit(f"[P3-FATAL] L{layer}: missing {label}")
    return float(m.group(1))
v=valid[0]
auc=val(r"\*\*\*auc:\s*([0-9.]+)",v,"AUC")
uauc=val(r"\*\*\*uauc:\s*\(?([0-9.]+)",v,"UAUC")
vl=val(r"\bloss:\s*([0-9.]+)",v,"Valid loss")
tl=val(r"\bloss:\s*([0-9.]+)",train[-1],"Train loss")
def diag(tag,k):
    matching=[x for x in lines if f"[inj-diag][{tag}]" in x]
    if not matching: raise SystemExit(f"[P3-FATAL] L{layer}: missing diagnostic {tag}")
    pats={"R":r"\bR_med=([0-9.]+)","C":r"\bC_med=(-?[0-9.]+)","A":r"\|a\|=([0-9.]+)"}
    return val(pats[k],matching[-1],f"{tag}_{k}")
row=[layer,auc,uauc,tl,vl,diag("q","R"),diag("v","R"),
     diag("q","C"),diag("v","C"),diag("q","A"),diag("v","A"),log]
with open(summary,"a") as f: f.write("\t".join(str(z) for z in row)+"\n")
print(f"[P3-METRIC] L{layer}: Valid AUC={auc:.9f} UAUC={uauc:.9f} TrainLoss={tl:.6f} ValidLoss={vl:.6f} Q-R={row[5]:.5f} V-R={row[6]:.5f}",flush=True)
PY
  echo "[P3-DONE] ${IDX}/8 L${L} $(date -Is) summary=$SUMMARY"
done

echo "[P3-COMPLETE] ALL 8 positions finished. Valid only, no Test."
echo "[P3-COMPLETE] $SUMMARY"
python - "$SUMMARY" <<'PY'
import csv,sys
with open(sys.argv[1]) as f: rows=list(csv.DictReader(f,delimiter="\t"))
assert len(rows)==8,len(rows)
print("[P3-RANKED] By Valid AUC:")
for r in sorted(rows,key=lambda x:float(x["valid_auc"]),reverse=True):
    print("L{:>2} AUC={:.6f} UAUC={:.6f} ValidLoss={:.6f}".format(
        r["layer"],float(r["valid_auc"]),float(r["valid_uauc"]),float(r["valid_loss"])))
print("[P3-ANCHORS] P2 L31 AUC 0.714338 UAUC 0.677975; P2 L24 AUC 0.623100 UAUC 0.582282")
PY
