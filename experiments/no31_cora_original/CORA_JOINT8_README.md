# CoRA-MF: 8 independent joint-optimization screens (2026-10-29)

This suite applies the *official* CoRA-MF title-only prompt / full QKVO dynamic PLoRA to eight independent trainings, with trainable MF as the main change. It is **not** the earlier GraphLoRA (GNN/NO31) code, and no earlier GNN Layer31 injection is used.

## Fixed controls

- Source tree: `/home/bbx/user_lx/CoRA_official`, based on official `VanillaCreamer/CoRA`.
- Original baseline training configuration: `local_configs/cora_mf_ood2_official.yaml`. This file remains untouched; the eight new LR3e-4 trial configs are written under a distinct protocol directory so no older 1e-3 trial/checkpoint is overwritten.
- Each run independently loads the **same pretrained MF**, Vicuna-7B, fresh CoRA adapter and seed **42**. Never continue from a previous run's best checkpoint.
- BS=2, grad accumulation=4, original effective batch=8; 400 micro-steps = 100 optimizer updates per epoch.
- Original cosine scheduler horizon remains **1000**. By default, all eight runs train until **10 consecutive completed Valid evaluations have not improved the best Valid AUC** (patience=10), or the original 1000-epoch maximum is reached. The script tracks no-improve epochs and gracefully exits only after the original runner has saved any new best checkpoint. A finite `--screen-epochs N` flag adds an optional epoch cap for quick screening, independent of patience. The original upstream early-stop condition (>20) is still present in upstream code but the added runtime condition stops earlier, without altering original source files.
- CoRA remains rank=16, alpha=32, Q/K/V/O all layers, dropout 0.05.
- AdamW, updated CoRA / dynamic PLoRA initial LR **3e-4**, and MF LR multiplier **0.1**, i.e. MF peak LR **3e-5**. The original source YAML is not modified; this suite generates *new* experiment YAMLs with `run.init_lr: 0.0003`. `warmup_lr: 1e-5` and `min_lr: 8e-5` stay as in the earlier successful single-variable LR=3e-4 comparison. Each MF group has its scheduler-updated LR multiplied by 0.1 after each micro-step.
- Valid AUC and UAUC are the selection metrics. The training objective continues using the official **Yes-logit BCE**. Do not use Yes-minus-No, which already underperformed on Valid.
- GPU0 only, sequential, foreground, log per run, no TEST evaluations and no background training.
- **AMP stability fix (2026-10-30):** the previous R1 first-backward failure was caused by a diagnostic hook raising on the *loss-scaled* MF gradient before PyTorch GradScaler could perform overflow recovery. The updated suite uses an initial AMP loss scale of **256** (instead of default 65536), logs a handful of scaled gradient checks without aborting at the hook, and reports GradScaler overflow/step skips. **12 consecutive skipped optimizer updates** trigger an explicit failure rather than silently training without updates. This does not guarantee the backward graph is numerically stable; continued failures require deeper MF→QFormer→dynamic LoRA diagnosis. The updated suite writes to a new isolated protocol directory to preserve old R1 failure logs.
- The original two small source fixes must already exist: gradient accumulation `loss / accum_grad_iters`, and dictionary `keys()` in evaluation. No new CoRA source code is modified by this suite.

## Eight variants

| Run | MF | CTR (BCE w/ label) | KD (LLM Yes teacher, detached) | Text semantic | MF anchor |
|---|---|---:|---:|---:|---:|
| R0_FROZEN | frozen | — | — | — | — |
| R1_MF_MAIN | trainable | — | — | — | — |
| R2_MF_CTR | trainable | 0.05 | — | — | — |
| R3_MF_KD | trainable | — | 0.05 | — | — |
| R4_CTR_KD | trainable | 0.05 | 0.05 | — | — |
| R5_CTR_KD_SEM | trainable | 0.05 | 0.05 | 1e-4 | — |
| R6_CTR_KD_ANC | trainable | 0.05 | 0.05 | — | 5e-4 |
| R7_ALL | trainable | 0.05 | 0.05 | 1e-4 | 5e-4 |

**Loss:** `L = BCEWithLogits(z_Yes, y) + lambda_ctr BCEWithLogits(s_MF,y) + lambda_kd BCEWithLogits(s_MF,stopgrad(sigmoid(z_Yes))) + lambda_sem L_sem_text + lambda_anchor L_anchor`; `s_MF = dot(e_u^MF,e_i^MF)`.

**Gradient flow:** Main loss flows through MF -> CoRA conditioning -> dynamic QKVO adapters -> LLM Yes-logit. CTR/KD/sem/anchor update MF only; KD's LLM teacher logits detached. Frozen Vicuna trunk remains frozen; CoRA remains trainable. MF gradient hook prints its first norm and aborts if zero/nonfinite.

**Important distinction:** original GraphLoRA semantic alignment used hidden states from a specified LLM layer; here a lightweight teacher uses the detached *mean input-token embeddings of target movie titles*. This is a *semantic proxy*, **not** the prior hidden-state loss. Original Projector regularization cannot be reused because the title-only CoRA path bypasses input projector; instead `L_anchor` penalizes MF embedding row drift from its pretrained weights. Do not call it Projector loss.

## Run on server (no manual download)

```bash
cd /home/bbx/user_lx/CoRA_official
curl -fL --retry 3 https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/132a7cbe1ca227e47af5bb661da04d1630308cb7/experiments/no31_cora_original/cora_joint8.py -o cora_joint8.py
python -m py_compile cora_joint8.py
python cora_joint8.py                  # preflight and generate all 8 configs, NO training
CUDA_VISIBLE_DEVICES=0 python -u cora_joint8.py --run --patience 10 2>&1 | tee logs/cora_joint8_console.log
python cora_joint8.py --summary
```

Logs: `logs/CORA_JOINT8_LR3E4_AMPFIX_20261030/R*.log`. Best checkpoints: `output/CORA_JOINT8_LR3E4_AMPFIX_20261030/<run>/<job_id>/checkpoint_best.pth`. Configs: `local_configs/CORA_JOINT8_LR3E4_AMPFIX_20261030/R*.yaml`.

The program **stops on first failure** instead of silently skipping a broken experiment. It refuses to overwrite a nonempty existing trial log; on restart choose an explicit `--only` subset of not-yet-run names. For a first-stage test, run `--only R1_MF_MAIN --patience 10` (its log will be protected on subsequent runs). The default is **no 8-epoch cap**; for a deliberate short smoke screen specify `--screen-epochs 8` explicitly. Patience=10 cannot trigger if an 8-epoch cap ends training first.

## After screening

Compare all eight best **Valid AUC** values (UAUC secondary), against R0 frozen and R1 MF-only, and inspect corresponding best checkpoints. With the default 10-epoch no-improvement early stop, the eight runs have already been trained to that criterion. Only after choosing a final checkpoint on Valid, evaluate Test once. No numerical outcomes are implied by creation of these scripts.

## Known constraints

This is the *first runnable, minimal-transfer* design, not claimed to reproduce a previously reported GraphLoRA loss numerically. The original authors' stage-1 projection checkpoint was unavailable and the base config sets `model.ckpt=null`. There has been no execution on the user's server and no new AUC/UAUC measurements.
