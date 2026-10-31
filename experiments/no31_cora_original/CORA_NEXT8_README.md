# CoRA + trainable MF: second round of 8 controlled experiments

Date: 2026-10-30. Use with the tested cora_joint8.py at commit 2117229582b12e306fbe44c70eeae16f87c3ef25. Existing CoRA files, checkpoints and experimental logs are not overwritten.

## First-round Valid observations (user's results)

| First round | Valid AUC | UAUC (at best AUC) |
| --- | ---: | ---: |
| R1: trainable MF, main loss only | 0.716869 | 0.695890 |
| R2: CTR .05 | 0.725905 | 0.698475 |
| R3: KD .05 | 0.726337 | 0.695700 |
| R4: CTR .05 plus KD .05 | 0.725750 | 0.689830 |
| R5: plus title-text semantic proxy | 0.716246 | 0.689536 |
| R6: plus MF anchor | 0.723497 | 0.691623 |
| R7: all losses | 0.715917 | 0.690015 |

These are one-seed observations, not causal or statistical proof. Since R2/R3 performed better than R1 but R4 did not improve on them, the next round isolates coefficient scale, relative strength, timing, and MF/main-loss gradient coupling. Do not reuse R5/R7 title-semantic proxy, which is NOT the prior GraphLoRA hidden-state semantic loss.

## Eight new experiments

| ID | CTR weight | KD weight | Main change | Comparison |
| --- | ---: | ---: | --- | --- |
| N1_CTR_001 | .01 | 0 | lower CTR strength | R2 CTR .05 |
| N2_CTR_010 | .10 | 0 | higher CTR strength | R2 CTR .05 |
| N3_KD_001 | 0 | .01 | lower KD strength | R3 KD .05 |
| N4_KD_010 | 0 | .10 | higher KD strength | R3 KD .05 |
| N5_CTR5_KD1 | .05 | .01 | CTR-dominant combination | R4 equal .05/.05 |
| N6_CTR1_KD5 | .01 | .05 | KD-dominant combination | R4 equal .05/.05 |
| N7_KD_RAMP10 | .05 | 0→.05 | KD ramps linearly over epoch 0–10; CTR stays .05 | R4 |
| N8_MF_DECOUPLE | .05 | .05 | detach MF features at CoRA generator input so main BCE no longer updates MF | R4 |

## Objective and exact gradient paths

LLM main loss: BCEWithLogits(LLM Yes logit, true label).
CTR: BCEWithLogits(MF user-item dot score, true label).
KD: BCEWithLogits(MF user-item dot score, stopgrad(sigmoid(LLM Yes logit))).

In N1–N7 the main loss updates both CoRA's dynamic generator and MF through its conditioning pathway; auxiliary losses update MF. In N8 the exact same MF feature VALUES enter the generator but through detach(), so only CTR and KD update MF, while the LLM main loss continues to train CoRA. Frozen LLM trunk remains frozen.

N7 uses lambda_KD(epoch) = .05 * min(epoch/10,1); epoch zero has zero KD weight, epoch ten and later have full .05. The schedule counts exactly 400 training microsteps per epoch. No optimizer scheduler change.

## Immutable controls

- Official source tree under /home/bbx/user_lx/CoRA_official and original input YAML unchanged.
- CoRA LR 3e-4, MF LR 3e-5 (ratio 0.1), seed 42, pretrained MF reused fresh for each experiment; CoRA adapters freshly initialized.
- Batch size 2, gradient accumulation 4, 400 microsteps per epoch, AMP initial scale 256 and supervised overflow handling from tested cora_joint8.py.
- Valid AUC as checkpoint selection metric, UAUC corresponding to that best AUC. Valid_small contains 5200 rows. No Test evaluation in this suite.
- 10 consecutive epochs without improved Valid AUC triggers early stop, original 1000-epoch LR schedule preserved.
- GPU0 only, serial foreground subprocesses; failure stops sequence, never overwrite existing nonempty logs.

## Server commands

cd /home/bbx/user_lx/CoRA_official

curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/2117229582b12e306fbe44c70eeae16f87c3ef25/experiments/no31_cora_original/cora_joint8.py -o cora_joint8.py

curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/70db0922d28562ef5c9194768554e2beb3ce7a34/experiments/no31_cora_original/cora_next8.py -o cora_next8.py

python -m py_compile cora_joint8.py cora_next8.py

python cora_next8.py

CUDA_VISIBLE_DEVICES=0 python -u cora_next8.py --run --patience 10

python cora_next8.py --summary

Files are under local_configs/CORA_NEXT8_20261030, logs/CORA_NEXT8_20261030, and output/CORA_NEXT8_20261030. Each trial has its own log and checkpoint directory. If a run crashes, preserve its failed log before trying the same ID again; the runner intentionally refuses to overwrite it.

## Interpretation

Compare N1/N2 to prior R2, N3/N4 to prior R3, N5–N8 to prior R4. Inspect Valid AUC, matched UAUC, best epoch, AMP skips, and whether each loss term is nonzero. Relative gain at seed 42 is not yet evidence of statistical significance. Recheck top variants at more seeds after these eight, and only evaluate Test after selecting by Valid.

The script was committed but has not yet been executed on the user's server; do not claim new AUC values before real runs.

## N8 checkpoint-write failure and compact-save fix (2026-10-31)

Observed at N8: torch.save failed at about 12.98 GB with an 'unexpected pos' message while saving checkpoint_best.pth. This is a serialization/storage failure, not proof of bad gradients or GPU OOM. Check disk free space, quota, and the saved model state size on the server.

Updated cora_next8.py commit ebd4594af0cc0b0b2c510c84e5936e4c7ae12d9c saves only actually trainable parameter keys (CoRA and MF), excluding frozen Vicuna and optimizer state, after measuring the expected size and available disk. It writes to checkpoint_best.pth.writing, checks ZIP archive CRC, and uses atomic os.replace only after successful completion. Failed writes preserve existing best checkpoint and leave the diagnostic temporary file.

These **model-only** checkpoints can be evaluated with model.load_state_dict(ckpt['model'], strict=False) after reconstructing the same Vicuna/CoRA/MF architecture. They CANNOT be used for full optimizer-state resume. They do not eliminate a real disk-full condition.

Inspect disk and N8 partial files before retrying:

```bash
cd /home/bbx/user_lx/CoRA_official
df -h .
df -i .
du -sh output/CORA_NEXT8_20261030/* 2>/dev/null
find output/CORA_NEXT8_20261030/N8_MF_DECOUPLE -type f -name 'checkpoint_*.pth' -printf '%s %p\n' 2>/dev/null | sort -nr
```

Preserve N1–N7 results. Inspect any failed/partial N8 checkpoint before deleting it. Rename only failed N8 log to N8_MF_DECOUPLE.failed.log before rerunning N8 with the updated code. A new timestamped job ID creates a new checkpoint subfolder.

## Single cross-round best-VALID summary (read-only, no GPU)

The old next8 `--summary` reported N1–N8 alone. The updated scripts read all known
R0–R7 and N1–N8 trial logs from `logs/CORA_*` and print: best Valid AUC,
corresponding Valid UAUC, best epoch, finished/failed status, and checkpoint existence.
Reports go to `out/cora_all_valid_best.csv` and `out/cora_all_valid_best.md`.
Failed N8's logged metrics are shown as failed, not as a successful checkpoint.
Structural ZIP checks cannot guarantee a checkpoint loads or matches the best metric.

Fetch and run from the CoRA server (these do not retrain):

```bash
cd /home/bbx/user_lx/CoRA_official
curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/2d30a01c50e6faf2be274b628d306d5b33844c9c/experiments/no31_cora_original/cora_joint8.py -o cora_joint8.py
curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/8bb2e87b16f7e4881f9d6a82d03e797176fa8991/experiments/no31_cora_original/cora_next8.py -o cora_next8.py
curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/08eb4c8a9bf5c3ec047baa90cd46e767b31b5eee/experiments/no31_cora_original/cora_all_summary.py -o cora_all_summary.py
python -m py_compile cora_joint8.py cora_next8.py cora_all_summary.py
python cora_next8.py --summary
# Or: python cora_all_summary.py --sort uauc
```

## Project checkpoint rule — all future experiments

Always save **trainable model parameters only** for best-model checkpoints.
Do not serialize frozen Vicuna-7B parameters, full optimizer/AdamW state,
or the AMP scaler in the best-model checkpoint. Both updated experiment
runners apply atomic trainable-only saving. Preserve existing older checkpoints.
Model-only checkpoints are suitable for loading after initializing the
same base model/MF setup, but are **not** a full training-resume snapshot.
If exact optimizer-state resumption is ever required, use a separately named,
explicitly requested recovery artifact; never silently expand best checkpoints.
