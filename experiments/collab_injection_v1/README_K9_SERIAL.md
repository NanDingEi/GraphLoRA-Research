# Nine independent serial Key experiments

Foreground GPU0 single-process scheduler:

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
python experiments/collab_injection_v1/test_target_k_cpu.py
set -o pipefail
python -u experiments/collab_injection_v1/run_k9_serial.py 2>&1 | tee logs/collab_v1/K9_console.log
```

Order: H0 H1 HS / T0 T1 TS / HT0 HT1 HTS.
Scopes: history / target / history_target. Conditions: no candidate / true candidate / mismatched candidate.
Original S2 L31 Q/V injection is preserved, and only the new zero-initialized L31 K adapter trains.
Each arm has its own S2 warm start (not the preceding arm), an independent 24-batch smoke run, a full 1-epoch training run, Valid Small AUC/UAUC, and parameter drift check.

To stop after a stage, use --stage history, --stage target or --stage history_target.
To resume a failed run, use the exact [K9-RESUME-CMD] printed by the launcher.
The script will skip successfully completed arms after checking saved checkpoints, config hash, Git commit and S2 checkpoint hash; it redoes a failed arm from S2, never from a partial checkpoint.

KS caveat: a negative candidate is chosen deterministically from a different candidate in the current batch, falling back to a different pre-existing GNN subgraph item for a single-ID batch. Neither text/labels nor S2 Q/V are altered. This is a batch-dependent feature-shuffle negative control, not a globally fixed item permutation; interpret results accordingly.

On the server, first make sure git status --short shows no uncommitted changes, then fetch and checkout the latest fix/collab-gated-optimizer branch. The runner performs its own Python syntax and CPU Key checks.

All runs are in the foreground, serial on GPU0, no Test evaluation and no background training.
Outputs: logs/collab_v1/S2_K9_SERIAL_TIMESTAMP/summary.tsv and state.json, plus per-arm logs; merged checkpoints in output/S2_K9_SERIAL_TIMESTAMP/.

Any failure stops the sequence immediately with an error and leaves prior completed arms intact.