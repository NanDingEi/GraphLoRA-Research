# R2: Eight sequential collaborative attention injection experiments

**IMPLEMENTED:** See [README_R2_RUN.md](README_R2_RUN.md) for actual programs, CPU guards, GPU0 commands and checkpoint evaluation rules. The remainder below is the original design and is retained as a protocol reference.

**Design only; no executable eight-run launcher or GPU run is implied.**
GPU0-only, one experiment at a time, foreground; original S2 checkpoint
remains unchanged. No label-driven gates or direct logit manipulation.

## Anchors
- S2 original ML-1M OOD2 Valid Small (5200): AUC 0.7296752596287361, UAUC 0.6841670211619721.
- R1: switching L31 Q/V ON->OFF changed only 243 strict positive-negative pair orders among 6,731,439 pairs; original Q+V matched V-only AUC/UAUC.
- P2B warm start 1 epoch, fix_autocast_cache true: AUC 0.7214924624586214, UAUC 0.6647772423708141; use only after checking configs and same source checkpoint.
- The final Prompt token (sample_embeds.shape[1]-1) predicts Yes/No, not the candidate-item ID token. Q at this decision position is not historical-position Q.
- Already tested/failed: historical/learned-attention gates, Target Fusion, candidate V gate, Target-Key variants. New proposal needs genuine distinction.

## Mechanism
For a valid historical item j and candidate i, derive the purely graph-side
relation from the original GNN 256-d penultimate embeddings:
  w[j|i] = 1 + 0.5 * tanh(cos(penult_j, penult_i)).
Normalize w among valid history-token positions per user so the mean is 1,
and bound to [0.5, 1.5]. Never reweight user/target ID positions.
Modified V low-rank path:
  low_rank_V[j] = alpha * A_V(x_j) + beta * w[j|i] * z_j.
GNN z_j is the existing 8-d injection vector and beta remains 0.1.

Probe decision-Q via the existing L31 Q LoRA-B:
  delta_Q[decision] = beta_q * B_Q(c_ui);
  c_ui = 0.5 * (z_user + z_target), beta_q=0.1.
Probe history K via fixed B_Q projection into pre-RoPE K only
if Q/K output shapes match; this is merely a pathway test, not a trained
method or claimed recommendation gain.

## Exactly eight experiment slots

| Slot | Experiment | Duration/budget | Key readout |
| --- | --- | --- | --- |
| R2-01 | CPU reanalyze R1 native Yes ON/OFF score shift, fp16 ties, margins | full R1 NPZ, no GPU | absolute/signed p50/p90/p99 score delta; near-margin fraction |
| R2-02 | L31 V injection amplitudes and reachability audit | fixed 512 Valid examples, no gradients | beta*z, B_V(beta*z), original V output norms, firing counts, valid masks |
| R2-03 | Inject graph context at last-Prompt decision Q ONLY, historical Q/V retained | same 512, no training | nonzero output response beyond repeat numerical jitter |
| R2-04 | Inject graph context to historical K ONLY (fixed steering projection, no trained K adapter) | same 512, no training | K path reachability and actual decision-token response |
| R2-05 | Candidate-aware history V graph-weighting, correct vs mismatched candidate features | same 512, no training | relation weights change & forward response, masking correctness |
| R2-06 | Train candidate-aware history V injection, no new output head | S2 original warm-start, full 1 epoch, full Valid | AUC/UAUC vs matched P2B continuation |
| R2-07 | Train candidate-aware history V + decision-Q (if R2-03 active); if inactive, predetermined fallback is V history relation WITHOUT per-history mean normalization | S2 original warm-start, full 1 epoch | incremental interaction effect or normalized-vs-unnormalized effect |
| R2-08 | Negative control: R2-06 with deterministic WRONG candidate GNN penultimate vector only in relational weighting; candidate text, ID, other GNN paths unchanged | S2 original warm-start, full 1 epoch | whether real user-item relationship matters |

### Scope and controls
- No experiments train logits/head directly, use original Yes/No BCE.
- R2-01 through R2-05 do not train. Their AUC on 512 examples must NOT
  be used to pick thresholds or claim improved performance.
- For R2-03/04/05 first verify a ZERO perturbation exactly preserves
  baseline S2 predictions on identical inputs; hooks activate only during
  real full-model forward, not the semantic teacher pass.
- Verify decision location equals inputs_embeds.shape[1] - t_posi
  in both train and eval, accounting for appended target answer labels.
- R2-05 relation computed solely from train graph-derived embeddings;
  mismatch alters only the target graph feature used in w[j|i].
- R2-06/07/08 require exact same original S2 checkpoint, seed 42, 1 epoch,
  batch 3, LR 2e-5, optimizer/scheduler horizon, semantic teacher options,
  train_gnn_with_rec_loss flag, LoRA/projector policies, fixed prompts,
  dataset, GPU0, and fix_autocast_cache=true. Do NOT continue from one
  new experimental checkpoint to the next.
- P2B as reference only after YAML/checkpoint/optimizer parity audit.
  If mismatch, spend the R2-07 training slot on a matched no-method
  baseline instead of comparing incomparable numbers.
- No Test while selecting methods; Test only after valid-based selection
  and independent replication.
- Prevent overwrites: per-run unique configs, output, logs, JSON and NPZ.

### Guards and stop conditions
1. Unit-test graph IDs, token role mapping, padding and batch B=1/3,
   tensor shapes/devices/dtypes and injection position.
2. 32-example CUDA smoke before each NEW path. Failure means repair,
   not a completed round.
3. If R2-02 finds no valid history injection or all zeros, stop and fix
   the information path before other GPU tests.
4. If R2-03/04 perturbation is indistinguishable from same-input
   no-perturbation numerical drift, mark pathway inactive and do not
   propose training it on an unverified mechanism.
5. If R2-05 correct/mismatched candidates produce identical weights,
   stop before R2-06/07/08.
6. If training becomes NaN, produces wrong source checkpoint or ignores
   optimizer parameter registration, stop rather than launch other runs.
7. All shifts/rescues must state direction, precision, repeated-baseline
   jitter and explicit sample count. Report not only AUC but UAUC.
8. One seed and one Valid do not establish significance. Next-day
   replication on user-disjoint or extra seeds before Test.

### Output per round
Unique JSON with source commit SHA, model checkpoint SHA, seed, validation
sample IDs, step count, CUDA peak, injected-token counts, tensor norms,
native Yes logits or its summary, AUC & UAUC if relevant, trained params,
checks and pass/inactive/fail reason.

Run order: R2-01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08.
**No background tasks, no automatic 8-run training; run only after
models, loaders, and tests are implemented.**

### Grounding
- GraphLoRA baseline: https://github.com/wgj15965/GraphLoRA
- CoRA (collaborative low-rank weights): https://ojs.aaai.org/index.php/AAAI/article/view/33334
- ReLLaX (CFLoRA + soft prompts): https://arxiv.org/abs/2501.13344
- iLoRA (instance-wise low-rank experts): https://proceedings.neurips.cc/paper_files/paper/2024/hash/cd476d01692c508ddf1cb43c6279a704-Abstract-Conference.html
