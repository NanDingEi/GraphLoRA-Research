# A2 / A3 overnight serial experiment (2026-10-13)

Both experiments reproduce the S2 training recipe (NGCF layer 31, gnn_scale=0.1, MF, semantic alignment, projector regularization, seed 42, max_epoch 3, AUC/UAUC validation) and train **from scratch**. No S2/A1 checkpoint is reused for training.

| Version | Candidate-history aggregator | Injection | Configuration |
|---|---|---|---|
| A2 | fixed cosine attention using normalize(user+candidate) vs history | zero-initialized 8-rank gate on target GNN injection | \`tran_configs/S2_candidate_history_A2.yaml\` |
| A3 | learned identity-initialized query Wq(candidate) and key Wk(history) | same target GNN gate as A2 | \`tran_configs/S2_candidate_history_A3.yaml\` |

Both preserve GraphLoRA injection on all non-target ID tokens. The gate is 1 + 0.5*tanh(MLP([u, candidate, history evidence, candidate*evidence])), with zero-init output weights/bias, hence identity initially. Empty or missing history also produces identity; PAD and target item duplicates are masked. The original A1 additive residual is disabled for both A2 and A3 by \`candidate_history_mode: gate\`.

## Run

\`\`\`bash
cd /home/bbx/user_lx/GraphLoRA
git fetch origin
git switch exp/A3-learned-candidate-history-attention-20261013
git pull --ff-only
mkdir -p log out
nohup python -u run_A2_A3_serial.py --gpu 0 > log/A2_A3_launcher.log 2>&1 < /dev/null &
echo "PID=$!"
tail -f log/A2_A3_launcher.log
\`\`\`

The launcher first runs CPU preflight tests, then **A2 Train -> A2 Best Test -> A3 Train -> A3 Best Test**. The entire process uses one GPU and executes **serially**; no two model processes run concurrently. If A2 fails, its error is recorded and A3 is still attempted. Every stage produces independent experiment output directories. No full test terminal output is logged; evaluation outputs are summarized.

Each train run writes full logs to \`log/<experiment>_train_<stamp>.log\`, and checkpoints/artifacts to \`output/<experiment>_<stamp>/<jobid>/\`.

The final comparison is printed and saved to \`out/A2_A3_serial_<stamp>.txt\` and \`.csv\`. It includes Valid AUC/UAUC (best Valid checkpoint) and Test AUC/UAUC using that checkpoint. Partial results are saved after each experiment.

## Notes

- The current S2/A1 code has different GNN postprocessing in training and evaluation pathways. These runs intentionally preserve the preexisting S2/A1 behavior instead of changing an unrelated baseline assumption. Any future normalization fix must also be applied to a rerun S2 control.
- The A2 branch \`exp/A2-history-conditioned-gate-20261013\` stores the A2 implementation. The A3 branch extends A2 with its learned attention configuration and adds the serial runner, so both runs can use one working tree and shared implementation without branch switching during a GPU job.
- No remote GPU job is launched by the GitHub commit. A user must execute the shell command on the training server.
