# G2 — Task-priority GNN gradient coordination

**Task**: preserve S2 Yes-logit binary CTR with BCE and AUC/UAUC. No graph augmentation or new classifier.

Inside the existing S2 training step, calculate the weighted semantic loss gradient on NGCF trainable parameters using one extra autograd.grad and retain_graph. Normal backprop of the combined original loss remains unchanged for all other parameters. At the end of each accumulation window, recover the recommendation gradient as total_GNN_grad minus weighted_semantic_GNN_grad, and project the auxiliary part only if the dot product is negative. Write corrected GNN gradients at the same AMP scaling; original GradScaler and optimizer step remain in force. Negative-conflict frequency prints every 500 updates and at epoch end.

This implements a task-priority conflict projection in GNN only; it does **not** invent new supervisory labels and does not turn CTR into a ranking task. Past experiments suggested semantic gradient norms were much smaller than recommendation gradients, so the projection may rarely activate. We will log the actual conflict rate rather than assume improvement.

Run:
```bash
cd /home/bbx/user_lx/GraphLoRA
git fetch origin
git switch exp/G2-task-priority-grad-20261015
git pull --ff-only origin exp/G2-task-priority-grad-20261015
python -u run_G2_task_priority.py --gpu 1
```
The runner first runs CPU gradient projection tests and then trains all 3 epochs, selecting best Valid AUC checkpoint and evaluating Test exactly once. Outputs: log/G2_TASK_PRIORITY_train_<timestamp>.log, out/G2_task_priority_<timestamp>.txt/.csv and output/G2_TASK_PRIORITY_<timestamp>/. No early stops or TrainOff runs.
