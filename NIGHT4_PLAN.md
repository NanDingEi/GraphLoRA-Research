# NIGHT4: Two-epoch LLM-only injection experiments

Goal: after D1-A Ep0 valid AUC=0.709148, test AUC=0.707153, separate weak injection effects from **continued fine-tuning of an S2 best checkpoint** (S2 Valid AUC 0.732884).

## Four strictly controlled runs

| Condition | config | Layer31 collaboration |
|---|---|---|
| N0 baseline continuation | NIGHT4_N0_S2_CONTINUE.yaml | original GraphLoRA ID-Q + ID-V |
| N1 evidence-only | NIGHT4_N1_ID_V_ONLY.yaml | ID-V only |
| N2 decision query | NIGHT4_N2_DECISION_Q_ID_V.yaml | decision-Q user/candidate pair + ID-V |
| N3 gated decision query | NIGHT4_N3_GATED_DECISION_Q_ID_V.yaml | LLM-query-conditioned rank-wise decision Q + ID-V |

All use trained S2 checkpoint `output/S2_groups_acc3/20261010223/checkpoint_best.pth` as the **identical starting point** and freeze MF and NGCF, including dropout (frozen NGCF always in eval mode). No GNN semantic auxiliary gradient: `gnn_semantic_align=false` for all four, avoiding the earlier no-grad teacher/autocast interaction and pointless auxiliary computation with a frozen target. Existing Projector cosine loss remains on.

Shared optimizer: init_lr=2e-5, min_lr=5e-6, warmup_lr=2e-6, 200 warmup steps, accumulation=3, batch_train=3, seed=42, max_epoch=2, validation=5200, Test=7331. Two **full** train epochs per condition, Valid after each; select best Valid checkpoint, Test that once. New N2/N3 pair projector is zero-initialized, so both start with N1's predictions; N0/N1 should produce the same valid predictions as S2 at initial checkpoint because ID-Q injected only into layer31 has no direct causal path to decision logit.

Per-run independent output/log filenames, serial execution on GPU 0. Every run has the same S2-Valid preflight (within ±0.003 or suite stops before training). The TEST checkpoint is completed using original frozen S2 MF/NGCF weights, as the runner saves only trainable weights.

Run:

```bash
cd /home/bbx/user_lx/GraphLoRA
git pull --ff-only origin exp/D1-decision-query-evidence-value-20261015
mkdir -p logs
nohup env GPU=0 bash run_D1_night4.sh > logs/night4_launcher.log 2>&1 < /dev/null &
echo "Night4 PID: $!"
```

This launches a separate child-shell background process; it does **not** alter the calling shell's `set -e` or call `exit`. Do not `source` the script. If errors occur it stops the suite rather than running invalid comparisons.

Monitor:
```bash
tail -f logs/night4_launcher.log
find logs -maxdepth 1 -name 'NIGHT4_*_train.log' -print
```

Results: `out/NIGHT4_<timestamp>_summary.tsv`, `logs/NIGHT4_<timestamp>_<group>_{preflight,train,test}.log`, and `minigpt4/output/NIGHT4_<timestamp>_<group>_train/<jobid>/checkpoint_best_complete.pth`.

Interpretation: compare N0 vs original S2 for fine-tuning effects, N1 vs N0 for ID-query reachability, N2 vs N1 for decision-Q usefulness, N3 vs N2 for learned semantic conditioning. If all four deteriorate, do NOT conclude D1's injection mechanism is inferior; first examine continued fine-tuning and the training/evaluation GNN projection mismatch. Only interpret Test after Valid-selected checkpoint.

Neither the GPU jobs nor unit tests can be run through this repository-only connector. Run the commands on the user's GPU host to obtain actual results.
