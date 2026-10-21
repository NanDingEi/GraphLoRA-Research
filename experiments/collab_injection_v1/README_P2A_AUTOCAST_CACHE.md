# P2-A: early LoRA updates versus autocast-cache teacher pass

The completed P0 control+aux drift report showed L0-L7 LoRA update norms exactly zero.
The P0 control config explicitly enables gnn_semantic_align=True and does not set fix_autocast_cache; the model defaults it to False.
compute_feedback_gcn runs a no_grad early LLM semantic teacher forward; the existing opt-in branch uses torch.cuda.amp.autocast(cache_enabled=False) for that teacher only.

## Purpose and protocol

Run exactly ONE additional 24-step CONTROL smoke with the fix enabled. Reuse the existing P0_CONTROL_SMOKE checkpoint as the matched unfixed arm.
Do not use L30 aux supervision, change LR/seed/batch/steps, or run Valid/Test.
The configuration generation refuses to overwrite any existing fixed-probe outputs and checks that exactly two fields changed: model.fix_autocast_cache, run.output_dir.

## On the server

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
git merge --ff-only FETCH_HEAD
conda activate minigpt4
export CUDA_VISIBLE_DEVICES=0

python -m py_compile experiments/collab_injection_v1/run_p2a_autocast_cache_probe.py
python experiments/collab_injection_v1/test_p2a_autocast_cache_cpu.py

RUN=logs/collab_v1/S2_L30_AUX_P0_20261021_165700
python experiments/collab_injection_v1/run_p2a_autocast_cache_probe.py \
  --stage prepare --run-dir "$RUN"

set -o pipefail
python -u train_collm_mf_din.py \
  --cfg-path "$RUN/P2A_FIX_CACHE_SMOKE.yaml" \
  2>&1 | tee "$RUN/P2A_FIX_CACHE_SMOKE.log"

python -u experiments/collab_injection_v1/run_p2a_autocast_cache_probe.py \
  --stage compare --run-dir "$RUN" \
  2>&1 | tee "$RUN/P2A_AUTOCAST_CACHE_PROBE.log"
```

Expected output markers: [P2A-READY], [P2A-CHECK], 32 [P2A-LAYER] lines, [P2A-EARLY], [P2A-COMPLETE].
Key verdict: original_all_zero=True and fixed_every_layer_updated=True would demonstrate that this specific cache fix restores first-eight LoRA parameter updates under otherwise matched smoke conditions.

Important: this is a parameter-update recovery check, NOT proof of a validation AUC improvement or a direct autograd experiment.
If L0-L7 still do not update, check individual gradients/optimizer membership before claiming the cache explanation.
Do not overwrite the existing P0 checkpoints or train another full epoch until results are inspected.
