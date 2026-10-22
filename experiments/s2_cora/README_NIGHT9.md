# Tonight: 9 controlled S2 / CoRA runs

Historical S2 parent: d818fa2943ede4e9a810b032a3422e94d0c77ed0.
Historical success: Valid epoch0 AUC 0.729675, epoch1 AUC 0.732884.
Each of the 9 runs trains exactly ONE complete epoch (11,297 microbatches),
evaluates Valid (5200 rows), and saves checkpoint_0 and checkpoint_best.
run.max_epoch stays 3, and the added optional stop_after_n_epochs=1 runs
AFTER checkpointing, so the first-epoch learning rate curve remains identical
to the historical 3-epoch S2 curve. No test evaluations.

| # | Experiment | Seed | Modification compared with historical S2 |
|---|---|---|---|
| 01 | S2 control | 42 | None; mandatory historical AUC>=0.72 gate |
| 02 | CoRA full Q+V | 42 | Zero-initialized CoRA delta |
| 03 | CoRA Q-only | 42 | Only L31 Q CoRA delta active |
| 04 | CoRA V-only | 42 | Only L31 V CoRA delta active |
| 05 | CoRA gain 0.5 | 42 | Half of original CoRA delta (same init) |
| 06 | S2 control | 43 | Fresh seed-matched control |
| 07 | CoRA full Q+V | 43 | Paired CoRA |
| 08 | S2 control | 44 | Fresh seed-matched control |
| 09 | CoRA full Q+V | 44 | Paired CoRA |

No S2 trained checkpoints and no R4/R5 weights. The original pretrained MF
checkpoint IS reused. NGCF, input projector and LLM LoRA start fresh under
the matched seed. Each CoRA run checks exact SHA256 equality of its initial
MF/NGCF/Projector/LoRA parameters against the corresponding S2 seed control.

## Launch on the server (foreground, GPU0, sequential)

Run from the ORIGINAL existing project worktree (do not overwrite it):

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/s2-rollback-cora-minimal-20261022
if [ -d /home/bbx/user_lx/GraphLoRA_S2_CORA/.git ] || [ -f /home/bbx/user_lx/GraphLoRA_S2_CORA/.git ]; then
  git -C /home/bbx/user_lx/GraphLoRA_S2_CORA merge --ff-only FETCH_HEAD
else
  git worktree add -b s2-cora-local-20261022 /home/bbx/user_lx/GraphLoRA_S2_CORA FETCH_HEAD
fi
cd /home/bbx/user_lx/GraphLoRA_S2_CORA
conda activate minigpt4
bash experiments/s2_cora/run_night9.sh prepare
CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run_night9.sh run
```

Run all-in-one when no prior preparation exists:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run_night9.sh all
```

The first run checks S2 control AUC>=0.72; if not, the remaining eight
experiments are NOT launched. After every completed run a JSON result,
checkpoint, and summary.csv are saved, and the console prints the AUC/UAUC.

Summary: logs/S2_CORA_NINE_20261022/summary.csv
Logs: logs/S2_CORA_NINE_20261022/<experiment>.log
Weights: output/S2_CORA_NINE_20261022/<experiment>/

Nine full epochs plus validation may exceed the night; the runner does
NOT submit tasks to the background. If interrupted, completed runs are
skipped on the next invocation, and a partially failed run causes a
fail-closed stop rather than a silent overwrite.

## Notes

The archived S2 YAML was reconstructed from effective training logs.
The first historical S2 control is an empirical reproduction test, NOT a
guaranteed match. Do not interpret CoRA gains unless the S2 control
passes the gate, shared weight fingerprints match and both methods use
the identical Valid split.
