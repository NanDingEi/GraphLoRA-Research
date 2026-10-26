# NIGHT18 — Explicit Collaborative Granularity (CoRA)

This experiment changes **only the frozen-MF conditioning route** of the established NIGHT17 E0 CoRA recipe. It does not alter original NO31 CTR/KD, training data, Q/K/V/O targets, Transformer layer schedules, layer-group projectors, LoRA-P, rank (16), alpha (48), seed (42), or optimizer defaults.

## Signal definitions
- **Coarse (entity-level)**: exact CoRA MF concatenation `[u;i]` from the frozen MF tables, shape `[B,512]`.
- **Fine (target-conditioned)**: history items weighted by candidate-item cosine similarity (excluding `padding_index`), followed by normalized user-target and target-history Hadamard interactions, shape `[B,512]`. Its norm matches the coarse vector. It is **not** equivalent to a GNN high-/low-hop decomposition.
- Fine-only replaces original CoRA generator conditioning.
- Coarse-gate/dual keep original CoRA generator inputs and multiply **only rank channels 8..15** by `1 + 0.1*tanh(MLP(signal))`, where signal=coarse or fine.
- MLP last layer zero-init; the original CoRA weights are preserved at initialization. The gate lives under `cora_weight_generator` and uses its current optimizer settings.

## Fixed baseline (no retraining)
- NIGHT17 E0_UNIFORM_4LAYER_CONTROL: historical Valid AUC=0.7299284744316927, UAUC=0.6866122315446266.
- NIGHT18 does NOT rerun the baseline. It trains only G1/G2/G3, each as an independent one-epoch experiment.
- Historical E0 comparison is indicative because runs occur in different sessions; keep their data, seed, and evaluation protocol aligned.

## Exact configurations
- G1_FINE_ONLY: original generator driven by target-conditioned interaction features.
- G2_COARSE_GATE: original MF generator + coarse-conditioned 8-rank gate (matched gate architecture).
- G3_DUAL_GATE: original MF generator + fine-conditioned 8-rank gate.

## Run on the original GPU0 server (foreground only)
```bash
cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
git status --short
# Stop if local modifications are present; do NOT stash/reset your experiments.
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/coarse-fine-cf-signals-20261026
git switch -c coarse-fine-n18-local FETCH_HEAD
conda activate minigpt4

python experiments/no31_cora_original/test_night18_cpu.py
python experiments/no31_cora_original/night18.py --plan
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night18.py --run
```
The run is serial and uses the same Valid-only protocol as NIGHT17.
It refuses to run if another compute process is using GPU0.
Output: `logs/NO31_CORA_NIGHT18_<timestamp>/summary.csv` and `output/NO31_CORA_NIGHT18_<timestamp>/`.

To run one arm: `CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night18.py --only G3_DUAL_GATE`.

## Important checks / limitations
- CPU smoke checks feature scale, padding safety, target-history sensitivity, rank-gate identity at initialization and final-layer gate gradients; they do not establish actual GPU predictive metrics.
- Historical NIGHT17 E0 is the fixed comparison. No E0 rerun and no anchor-drift gate.
- G3's gate begins as an exact identity modification to the original CoRA dynamic basis (assuming matched frozen embeddings); G1 is **not** initially identity because it uses different input features.
- Finite differences in predictions after training still do not establish clean semantic disentanglement. Follow up with history permutation, per-channel ablations, and controlled prediction probes.
- The dynamic condition is based on a frozen snapshot of MF embeddings, just as original CoRA was. NGCF supervision remains intact but is not explicitly split into graph-hop granularity here.
- If the current server checkout lacks the NIGHT17 prepared config at `logs/NO31_CORA_ORIGINAL_READY/train.yaml`, recreate/locate it according to prior archived CoRA setup before launching.
- GitHub source changes do not train a server GPU remotely. No fresh results are claimed.
