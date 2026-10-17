#!/usr/bin/env bash
# Historical S2 AMP optimizer-update audit: same first N train microbatches,
# L31 with rec_loss_updates_gnn ON vs DETACH. No full epochs/Valid/Test/CKPT.
# Monitor actual AdamW state["step"] advance across GradScaler.step/update.
# Isolated original source worktree; only logging and early exit patched.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
ID="${S2_RUN_ID:-20261017_143611}"
SOURCE="$MAIN/logs/S2_STRICT_${ID}/L31.yaml"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
export S2_AMP_PREFIX_MICRO="${S2_AMP_PREFIX_MICRO:-600}"
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_AMP_PREFIX.lock
flock -n 9 || { echo "[S2-AMP-ERROR] prefix audit already running"; exit 2; }
test -f "$SOURCE"
[[ $(( S2_AMP_PREFIX_MICRO % 3 )) -eq 0 ]] || {
  echo "[S2-AMP-ERROR] use a multiple of 3 microbatches"; exit 2;
}
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_AMP_PREFIX_${STAMP}"
WT="$MAIN/../GraphLoRA_S2_AMP_PREFIX_${STAMP}"
mkdir -p "$OUT"
git worktree add --detach "$WT" "$SHA"
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
if [[ ! -e "$WT/dataset" ]]; then
  ln -s "$MAIN/dataset" "$WT/dataset"
else
  mkdir -p "$WT/dataset/ml-1m"
  for NAME in movie_graph_train.npz movie_graph_valid.npz movie_graph_test.npz; do
    if [[ ! -e "$WT/dataset/ml-1m/$NAME" ]]; then
      ln -s "$MAIN/dataset/ml-1m/$NAME" "$WT/dataset/ml-1m/$NAME"
    fi
  done
fi
test -f "$WT/dataset/ml-1m/movie_graph_train.npz"
python - "$WT" "$SOURCE" "$OUT" <<'PY'
import sys,copy,yaml
from pathlib import Path
wt=Path(sys.argv[1]);src=Path(sys.argv[2]);out=Path(sys.argv[3])
cfg=yaml.safe_load(src.read_text())
assert cfg["model"]["gcn_layer"]==31
assert cfg["model"]["gnn_scale"]==0.1
assert cfg["model"]["train_gnn_with_rec_loss"] is True
assert cfg["run"]["amp"] is True
assert cfg["run"]["accum_grad_iters"]==3
assert cfg["run"]["max_epoch"]==3
assert cfg["run"]["seed"]==42
assert not cfg["model"].get("ckpt")
for tag,detach in (("ON",False),("DETACH",True)):
    c=copy.deepcopy(cfg)
    c["model"]["train_gnn_with_rec_loss"]=not detach
    c["run"]["output_dir"]=str(out/tag)
    assert {k for k in c["model"] if c["model"][k]!=cfg["model"][k]} == ({"train_gnn_with_rec_loss"} if detach else set())
    assert {k for k in c["run"] if c["run"][k]!=cfg["run"][k]}=={"output_dir"}
    (out/f"{tag}.yaml").write_text(yaml.safe_dump(c,sort_keys=False))

t=wt/"minigpt4/tasks/base_task.py"
s=t.read_text()
anchor="""            if i >= iters_per_epoch:
                break

            samples = next(data_loader)"""
patch="""            if i >= iters_per_epoch:
                break
            # S2-AMP-PREFIX: stop BEFORE consuming microbatch N; same prefix in both arms.
            if i >= int(os.environ.get("S2_AMP_PREFIX_MICRO","600")):
                break

            samples = next(data_loader)"""
assert s.count(anchor)==1
s=s.replace(anchor,patch)
old="""                    scaler.step(optimizer)
                    scaler.update()
                else:"""
new="""                    # S2-AMP-PREFIX: observe successful optimizer update by
                    # AdamW's ACTUAL per-parameter step counter, not scaler scale.
                    # The selected parameter has a gradient before GradScaler.step.
                    # This is read-only and consumes no random values.
                    _active = next((p for group in optimizer.param_groups
                                    for p in group["params"] if p.grad is not None),None)
                    assert _active is not None, "No active optimizer parameter"
                    _state = optimizer.state.get(_active,{})
                    _before = int(_state.get("step",0))
                    _scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    _after = int(optimizer.state.get(_active,{}).get("step",0))
                    _scale_after = float(scaler.get_scale())
                    assert _after in (_before,_before+1),("Unexpected AdamW step",_before,_after)
                    if not hasattr(self,"_s2_amp_audit"):
                        self._s2_amp_audit={"attempts":0,"updates":0,"skips":0}
                    _audit=self._s2_amp_audit
                    _audit["attempts"]+=1
                    _updated=_after>_before
                    _audit["updates" if _updated else "skips"]+=1
                    if _audit["attempts"]<=20 or _audit["attempts"]%25==0:
                        print(f'[S2-AMP-STEP] epoch={inner_epoch} micro={i+1} '
                              f'attempt={_audit["attempts"]} updated={int(_updated)} '
                              f'adam_before={_before} adam_after={_after} '
                              f'grad_scale_before={_scale_before} '
                              f'grad_scale_after={_scale_after}',flush=True)
                else:"""
assert s.count(old)==1
s=s.replace(old,new)
old_end="""        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {"""
new_end="""        metric_logger.synchronize_between_processes()
        _audit=getattr(self,"_s2_amp_audit",None)
        if _audit:
            print(f'[S2-AMP-SUMMARY] attempts={_audit["attempts"]} '
                  f'updates={_audit["updates"]} skips={_audit["skips"]} '
                  f'final_scale={float(scaler.get_scale())}',flush=True)
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {"""
assert s.count(old_end)==1
s=s.replace(old_end,new_end)
t.write_text(s)

r=wt/"minigpt4/runners/runner_base.py";a=r.read_text()
old="""                    self.log_stats(split_name="train", stats=train_stats)
                    # torch.cuda.empty_cache()"""
new="""                    self.log_stats(split_name="train", stats=train_stats)
                    logging.info("[S2-AMP-PREFIX-STOP] Early exit after exactly %s train microbatches. "
                                 "NO validation, test or checkpoint.",
                                 __import__("os").environ.get("S2_AMP_PREFIX_MICRO"))
                    return
                    # torch.cuda.empty_cache()"""
assert a.count(old)==1
r.write_text(a.replace(old,new))
print("[S2-AMP-PREFIX-PATCH] read-only AdamW step logging and early stop; "
      "optimizer/scheduler/data/forward/backward unchanged",flush=True)
PY
git -C "$WT" diff -- minigpt4/tasks/base_task.py minigpt4/runners/runner_base.py > "$OUT/only_prefix_and_monitor.diff"
python -m py_compile "$WT/minigpt4/tasks/base_task.py" "$WT/minigpt4/runners/runner_base.py"
for ARM in ON DETACH; do
    echo "[S2-AMP-START] arm=$ARM micro=$S2_AMP_PREFIX_MICRO $(date -Is)"
    (
      cd "$WT"
      python -u train_collm_mf_din.py --cfg-path "$OUT/$ARM.yaml" 2>&1 | tee "$OUT/$ARM.log"
    )
    grep -Fq "[S2-AMP-SUMMARY]" "$OUT/$ARM.log" || {
      echo "[S2-AMP-ERROR] arm=$ARM missing update summary"; exit 3;
    }
    grep -Fq "[S2-AMP-PREFIX-STOP]" "$OUT/$ARM.log" || {
      echo "[S2-AMP-ERROR] arm=$ARM missing early stop"; exit 3;
    }
done
python - "$OUT" <<'PY'
import re,sys
from pathlib import Path
out=Path(sys.argv[1]);stats={}
for name in ("ON","DETACH"):
    rows=re.findall(r"\[S2-AMP-SUMMARY\] attempts=(\d+) updates=(\d+) skips=(\d+) final_scale=([0-9.eE+-]+)",
                    (out/f"{name}.log").read_text(errors="replace"))
    assert len(rows)==1,(name,rows)
    a,u,s,scale=rows[0];a,u,s=int(a),int(u),int(s)
    assert a==u+s
    stats[name]=(a,u,s,scale)
    print(f"[S2-AMP-RESULT] {name} attempts={a} updates={u} skipped={s} "
          f"skip_rate={s/a:.5f} final_scale={scale}")
print(f"[S2-AMP-DELTA] DETACH minus ON successful updates="
      f"{stats['DETACH'][1]-stats['ON'][1]:+d}; "
      f"skipped updates={stats['DETACH'][2]-stats['ON'][2]:+d}")
print("[S2-AMP-COMPLETE] prefix training only, no Valid/Test/checkpoints")
PY
