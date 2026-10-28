# NIGHT25 — three strictly serial LightGCN experiments

## Experiment definition

| Arm | Raw LightGCN initialization | Propagation | Graph BPR pretrain | LLM training |
| --- | --- | --- | --- | --- |
| M1_MF_INIT_L2 | Exact pretrained MF user/item table | 2 layers | 0 epochs | 1 epoch, Valid |
| M2_MF_INIT_L3 | Same exact pretrained MF table | 3 layers | 0 epochs | 1 epoch, Valid |
| M3_RANDOM_LONG_L2 | Random normal, no MF init | 2 layers | 500 epochs by default (500–1000 adjustable) | 1 epoch, Valid |

All arms keep the same frozen MF condition anchor and the same graph correction coefficient: MF_pair + 0.1 * (LightGCN_pair - MF_pair). This is a deliberate MF-anchored condition, not a complete replacement. In M3 the graph embedding is random-initialized, but original MF remains as the anchor and as the input prompt feature; this isolates the graph encoder initialization.

The graph ID table stays trainable through the LLM recommendation loss. Vicuna+CoRA trains normally; no LLM freezing. Original NGCF auxiliary CTR=0.05, KD=0.05 and semantic loss, MF mixed prompt, seed42, group8 CoRA full QKVO r16 alpha48, GPU0 are unchanged. N25 graph BCE auxiliary weight is zero in all arms. FP32 LightGCN, QFormer and CoRA dynamic branch; Vicuna retains its AMP policy (GradScaler starts at 256).

The two MF arms intentionally do not run separate graph BPR training: they directly compare MF initialization followed by 2 vs 3 graph propagation steps. The M3 graph gets 500-1000 full epochs of random-initialized BPR training using TRAIN positives only. This comparison changes both initialization and graph pretraining procedure; distinguish them in analysis.

Current TRAIN graph includes positive target edges during graph propagation. The known transductive target-edge caveat is unchanged across arms and needs a separate controlled leakage test.

## Install in a separate worktree

    cd /home/bbx/user_lx/GraphLoRA_NIGHT24
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night25-mf-init-lightgcn-depth-20261028
    git worktree add --detach /home/bbx/user_lx/GraphLoRA_NIGHT25 FETCH_HEAD
    cd /home/bbx/user_lx/GraphLoRA_NIGHT25
    conda activate minigpt4
    mkdir -p logs/NO31_CORA_ORIGINAL_READY
    cp /home/bbx/user_lx/GraphLoRA_NIGHT24/logs/NO31_CORA_ORIGINAL_READY/train.yaml logs/NO31_CORA_ORIGINAL_READY/train.yaml

If missing, source the prepared YAML from /home/bbx/user_lx/GraphLoRA_NO31_CORA/logs/NO31_CORA_ORIGINAL_READY/train.yaml.

## Preflight and run

**Important:** previous worktrees may remain in `PYTHONPATH` from
an earlier terminal. NIGHT25 forces its own root to the front of
`sys.path` and pins the path for every subprocess. Before launching
a long training run, verify the module path:

    cd /home/bbx/user_lx/GraphLoRA_NIGHT25
    PYTHONPATH="$PWD" python -c 'import experiments.no31_cora_original.graph_cf_pretrain as g; print(g.__file__); assert hasattr(g, "load_mf_init_table")'

The printed path **must** contain `GraphLoRA_NIGHT25` and never
`GraphLoRA_NIGHT24`. The preflight and plan now also print
`[N25-IMPORT-PATH-OK]`.

    python experiments/no31_cora_original/test_night25_cpu.py
    python experiments/no31_cora_original/night25.py --plan
    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night25.py --run

1000-epoch random graph pretrain instead of 500:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night25.py --run --random-epochs 1000

If an arm fails, later arms are not launched. For selective restart in a NEW result directory:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night25.py --run --start M2_MF_INIT_L3

## Outputs and checks

One new logs/NO31_CORA_NIGHT25_<timestamp> directory contains plan.json, each graph-only pretraining log, each train.log and checkpoint location, artifacts/*.npz with matching .json manifest, summary.csv and results.jsonl. Pretraining stage prints progress every 25 epochs, while writing complete per-epoch losses to the pretrain log. The final LLM metrics are Valid AUC and UAUC only, not Test.

Required markers: [N25-MF-WEIGHT-PARITY-PASS] verifies exact checkpoint embedding copy for M1 and M2; [N25-GRAPH-INIT] distinguishes MF and random graph initial states; [N25-GRAPH-RECEIVES-LLM-GRAD] confirms the LLM main-loss gradient reaches the graph even when graph auxiliary loss is zero; [N24-AMP-SUMMARY] shows successful/skipped updates; [N25-ARM-DONE] and [N25-COMPLETE] confirm all required serial stages.

Historical NIGHT17 E0 valid reference AUC 0.729928474, UAUC 0.686612232. This is not an exact same-precision paired control; a zero-lambda MF-only control would be required to isolate numerical precision effects.

No experimental metrics are asserted until runs finish.
