#!/usr/bin/env python3
"""CoRA/MF fifth-round controlled follow-ups: ten VALID-only independent runs.

D1: C6 loss + MF main-gradient x0.5 (no changed architecture).
D2/D3: all-layer K32 centroid + original MF residual (personal share 0.25/0.50).
D4/D5: four layer-group TopK neighbor aggregation + original MF residual
       (personal share 0.25/0.50).
D6/D7: selective bottom K32 or top TopK20 groups, others original MF.
D8/D9/D10: uniform Rank 8/12/20 with alpha/r=2.

Each experiment: same seed=42, original MF initialization, Vicuna-7B frozen,
Q/K/V/O CoRA generator, C6 loss (CTR=.005, gated KD=.01), except D1 changes
only MF main-gradient gain=0.5. No TEST. No overwriting previous rounds.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

import cora_rankgran7 as base

ROOT = Path(__file__).resolve().parent
PROTOCOL = "CORA_STRUCT10_20261101"

TRIALS = [
    dict(name="D1_C6_GRAD_HALF", mode="baseline", gamma=0.5),
    dict(name="D2_K32_RES025", mode="cluster_mix", personal_share=0.25),
    dict(name="D3_K32_RES050", mode="cluster_mix", personal_share=0.50),
    dict(name="D4_TOPK_RES025", mode="neighbor_mix", personal_share=0.25),
    dict(name="D5_TOPK_RES050", mode="neighbor_mix", personal_share=0.50),
    dict(name="D6_LOW_K32", mode="select_cluster"),
    dict(name="D7_HIGH_TOPK20", mode="select_neighbor"),
    dict(name="D8_UNIFORM_R8", mode="rank", ranks=[8, 8, 8, 8]),
    dict(name="D9_UNIFORM_R12", mode="rank", ranks=[12, 12, 12, 12]),
    dict(name="D10_UNIFORM_R20", mode="rank", ranks=[20, 20, 20, 20]),
]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
ORDER = [t["name"] for t in TRIALS]
PREV_MODEL_CONFIGURE = base._configure_initialized_model
PREV_GROUP_INPUTS = base._make_group_inputs


def configure():
    """Override the previous suite's globals in this process only."""
    base.PROTOCOL = PROTOCOL
    base.TRIALS = TRIALS
    base.TRIAL_BY_NAME = TRIAL_BY_NAME
    base.ALL_ORDER = ORDER
    base.configure_core()


def _interpolate(original, collaborative, personal_share):
    if original.shape != collaborative.shape:
        raise ValueError("Residual features must have equal shapes")
    if not 0 <= personal_share <= 1:
        raise ValueError(f"personal_share outside [0,1]: {personal_share}")
    return personal_share * original + (1 - personal_share) * collaborative


def group_inputs(generator, ui_embedding, ctx):
    mode = generator._rankgran_mode
    if mode not in ("cluster_mix", "neighbor_mix",
                    "select_cluster", "select_neighbor"):
        return PREV_GROUP_INPUTS(generator, ui_embedding, ctx)

    rec = ctx["rec_encoder"]
    user_ids = ctx["user_ids"].long().view(-1)
    item_ids = ctx["item_ids"].long().view(-1)
    user_table, item_table = rec.user_embedding.weight, rec.item_embedding.weight
    meta = generator._rankgran_meta
    trial = generator._struct10_trial
    if ui_embedding.shape[-1] != user_table.shape[-1] + item_table.shape[-1]:
        raise RuntimeError("Expected concatenated user/item MF vectors")
    if ui_embedding.shape[0] != user_ids.numel() or ui_embedding.shape[0] != item_ids.numel():
        raise RuntimeError("MF batch user/item IDs differ from generator batch")

    if mode in ("cluster_mix", "select_cluster"):
        u = base._cluster_repr(user_table, user_ids, meta["user_k32"])
        i = base._cluster_repr(item_table, item_ids, meta["item_k32"])
        group_signal = base._cat_ui(u, i)
        if mode == "cluster_mix":
            # Same feature for every Transformer layer; explicitly compare
            # cluster-only G1 and these two progressively personalized variants.
            return [_interpolate(ui_embedding, group_signal, trial["personal_share"])]
        # Group 0/1 = K32 centroid; groups 2/3 = original MF, no gate.
        return [group_signal, group_signal, ui_embedding, ui_embedding]

    if mode == "neighbor_mix":
        result = []
        # Same TopK profile as G4; only the residual coefficient changes.
        for k in (40, 20, 10, 5):
            u = base._neighbor_repr(user_table, user_ids, meta["user_neighbors"], k)
            i = base._neighbor_repr(item_table, item_ids, meta["item_neighbors"], k)
            result.append(_interpolate(ui_embedding, base._cat_ui(u, i),
                                       trial["personal_share"]))
        return result

    if mode == "select_neighbor":
        u = base._neighbor_repr(user_table, user_ids, meta["user_neighbors"], 20)
        i = base._neighbor_repr(item_table, item_ids, meta["item_neighbors"], 20)
        neighborhood = base._cat_ui(u, i)
        # Low layers keep raw MF; only L16..31 use neighborhood collaboration.
        return [ui_embedding, ui_embedding, neighborhood, neighborhood]
    raise RuntimeError(f"Unhandled struct10 mode: {mode}")


def configure_initialized_model(model, trial, PLoraLinear, LlamaDecoderLayer):
    mode = trial["mode"]
    if mode == "baseline":
        # D1 is strictly loss-path-only; never recreate P matrices,
        # reinitialize trainable CoRA weights or change rank/output capacity.
        layers = model.llama_model_lora.base_model.model.model.layers
        if len(layers) != 32:
            raise RuntimeError(f"Expected 32 Vicuna decoder layers, got {len(layers)}")
        modules = []
        for name, module in model.llama_model_lora.named_modules():
            match = base.RANK_RE.search(name)
            if match and isinstance(module, PLoraLinear):
                modules.append((name, module))
        if len(modules) != 128:
            raise RuntimeError(f"Expected 128 original CoRA QKVO modules, got {len(modules)}")
        for name, module in modules:
            active = module.active_adapter
            if int(module.r[active]) != 16 or abs(float(module.scaling[active]) - 2) > 1e-8:
                raise RuntimeError(f"D1 must retain original rank16 alpha32: {name}")
        generator = model.llama_model_lora.base_model.weight_generator
        if int(generator.r) != 16:
            raise RuntimeError("D1 must retain original dynamic generator Rank16")
        generator._rankgran_mode = "rank"  # original forward, no granularity changes
        print("[STRUCT10 BASELINE] original r16/alpha32, original P/generator "
              "initialized as in C6; only MF gradient gain changes", flush=True)
        return

    # Reuse verified previous-round initialization and fixed train-only
    # cluster/neighbor pools. Never change model source or prompt.
    supported = {
        "cluster_mix": "cluster_residual",  # builds train K32
        "neighbor_mix": "multi_topk",       # builds train top40
        "select_cluster": "cluster_k32",    # builds train K32
        "select_neighbor": "multi_topk",    # builds train top40
    }
    original_mode = supported.get(mode, mode)
    compatible_trial = dict(trial, mode=original_mode)
    PREV_MODEL_CONFIGURE(model, compatible_trial, PLoraLinear, LlamaDecoderLayer)
    gen = model.llama_model_lora.base_model.weight_generator
    gen._rankgran_mode = mode
    gen._struct10_trial = trial
    if mode == "rank":
        print(f"[STRUCT10 RANK] uniform rank={trial['ranks'][0]} alpha/r=2", flush=True)
    else:
        print(f"[STRUCT10 GRANULARITY] mode={mode} "
              f"personal_share={trial.get('personal_share','N/A')}", flush=True)


def _cpu_test():
    # Independent smoke checks using small train-only MF vectors:
    # deterministic cluster membership, gradient flow, no validation leakage.
    base._cpu_test()
    torch.manual_seed(42)
    personal = torch.randn(3, 512, requires_grad=True)
    group = torch.randn(3, 512, requires_grad=True)
    for alpha in (0.25, 0.50):
        mixed = _interpolate(personal, group, alpha)
        torch.testing.assert_close(
            mixed, personal * alpha + group * (1.0-alpha))
        gp, gg = torch.autograd.grad(mixed.sum(), (personal, group),
                                      retain_graph=True)
        torch.testing.assert_close(gp, torch.full_like(gp, alpha))
        torch.testing.assert_close(gg, torch.full_like(gg, 1-alpha))
    assert len(TRIALS) == 10
    assert len(set(ORDER)) == len(ORDER)
    assert [t["ranks"][0] for t in TRIALS[-3:]] == [8, 12, 20]
    assert all(t.get("mode") in (
        "baseline", "cluster_mix", "neighbor_mix",
        "select_cluster", "select_neighbor", "rank") for t in TRIALS)
    print("[STRUCT10 CPU PASS] residual mixing and gradients, "
          "experiment registry, uniform rank controls", flush=True)


def _preflight():
    configure()
    if len(TRIALS) != 10:
        raise RuntimeError("Ten independent experiments required")
    if base.combo.SEED != 42 or base.combo.BASE_LR != 3e-4:
        raise RuntimeError("C6 optimizer/seed control mismatch")
    c6 = base.combo.NAMES["C6_KD_BOTH"]
    assert c6["ctr"] == .005 and c6["kd"] == .010
    assert c6["kd_gate"] == "agree_conf" and c6["gamma"] == 1.0
    assert base.core.BASE.is_file()
    return c6


def prepare():
    c6 = _preflight()
    base.prepare()
    # base.prepare() creates current-suite configs only, all still with C6 loss.
    conf = base.core.CONFIG_DIR
    for trial in TRIALS:
        import yaml
        cfg = yaml.safe_load((conf / (trial["name"] + ".yaml")).read_text())
        if (cfg["run"]["seed"] != 42 or cfg["model"]["freeze_rec"] is not False or
            cfg["run"].get("test_splits") or cfg["run"].get("resume_ckpt_path") or
            cfg["run"]["iters_per_epoch"] != 400 or
            abs(float(cfg["run"]["init_lr"]) - 3e-4) > 1e-12):
            raise RuntimeError(f"Control drift in {trial['name']}")
    mf_path = base.core.BASE
    manifest = {
        "protocol": PROTOCOL,
        "experiment_type": "independent follow-ups after C6 AUC and G1/G4 UAUC divergence",
        "trials": TRIALS,
        "source": "cora_rankgran7.py extended without touching earlier logs/weights",
        "official_base_config": str(mf_path),
        "seed": 42,
        "gpu": "0",
        "fixed_optimization": {
            "main_loss": "original CoRA Yes-logit BCE",
            "ctr": c6["ctr"], "kd": c6["kd"],
            "gate": "LLM p aligns with label AND abs(p-.5)>=.2",
            "only_D1_gamma": 0.5, "others_gamma": 1.0,
            "cora_lr": 3e-4, "mf_lr": 3e-5,
            "patience": 10, "valid_only": True,
            "batch": 2, "accumulation": 4, "microsteps_per_epoch": 400,
        },
        "granularity": {
            "train_only_clusters": "train_ood2 positive and user-history entities only",
            "K32": "fixed train-seen cluster assignments, live differentiable centroids",
            "TopK": "fixed pretrained cosine train-seen IDs, live differentiable attention with T=.2",
            "cold_entities": "original individual MF embeddings (not clustered)",
            "personal_share": "alpha in z = alpha*MF + (1-alpha)*collaborative",
            "layer_groups": "L0-7, L8-15, L16-23, L24-31",
            "G4_topk_schedule": [40, 20, 10, 5],
        },
        "rank": {
            "uniform_profiles": [8, 12, 20],
            "alpha_over_rank": 2,
            "generator_max_rank": "uniform rank for each experiment",
            "baseline": "C6 uses r16 alpha32",
        },
        "checkpoints": "trainable-only atomic, no optimizer/scaler or frozen Vicuna",
        "no_mixed_prompt": True,
    }
    folder = base.core.LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "struct10_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[STRUCT10 PLAN] new isolated protocol:", PROTOCOL, flush=True)
    print("[STRUCT10 OUTPUT] configs:", base.core.CONFIG_DIR,
          "logs:", folder, "checkpoints:", base.core.OUT_DIR, flush=True)
    for trial in TRIALS:
        print("[STRUCT10 TRIAL]", trial["name"], trial, flush=True)


def worker(name, patience, screen_epochs):
    trial = TRIAL_BY_NAME[name]
    _preflight()
    if trial["name"] == "D1_C6_GRAD_HALF":
        altered = dict(base.combo.NAMES["C6_KD_BOTH"], gamma=0.5)
        base.combo.NAMES["C6_KD_BOTH"] = altered
        print("[STRUCT10 GRAD] C6 double-gated KD unchanged; only gamma=.5", flush=True)
    # Parent runner patches original CoRA classes and the compact checkpoint
    # saving path. Runtime customize model init and generator group signals.
    base._make_group_inputs = group_inputs
    base._configure_initialized_model = configure_initialized_model
    base.worker(name, patience, screen_epochs)


def _global_summary():
    script = ROOT / "cora_all_summary.py"
    if script.is_file():
        subprocess.run([sys.executable, str(script), "--sort", "auc"],
                       cwd=ROOT, check=False)


def run_selected(names, patience, screen_epochs):
    configure()
    if not base.core.CONFIG_DIR.is_dir():
        raise RuntimeError("No prepared configs: run python cora_struct10.py --plan first")
    for name in names:
        log = base.core.LOG_DIR / (name + ".log")
        if log.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing log {log}. "
                "Select only trials that have not started. Archive a failed "
                "trial log before rerunning it.")
    for name in names:
        log = base.core.LOG_DIR / (name + ".log")
        command = [sys.executable, "-u", str(ROOT / "cora_struct10.py"),
                   "--_worker", name, "--patience", str(patience),
                   "--screen-epochs", str(screen_epochs)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print("[STRUCT10 START]", name, "GPU0 foreground serial", flush=True)
        with log.open("x", encoding="utf-8") as out:
            proc = subprocess.Popen(command, cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                out.write(line)
            code = proc.wait()
        _global_summary()
        if code != 0:
            raise SystemExit(f"[STRUCT10 STOP ON FAILURE] {name} exit={code} log={log}")
    print("[STRUCT10 DONE] All requested runs terminated; check Valid metrics.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Validate and create configs only")
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--only", default="",
                        help="Comma-separated subset of D1..D10 IDs")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--screen-epochs", type=int, default=0,
                        help="Optional full epochs cap; default 0=patience based")
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.cpu_test:
        _cpu_test()
        return
    if args.summary:
        _global_summary()
        return
    if args._worker:
        if args._worker not in TRIAL_BY_NAME:
            parser.error(f"Unknown worker trial {args._worker}")
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience must be >=1; screen-epochs must be >=0")
    if not args.plan and not args.run:
        parser.error("Specify --plan or --run")
    if args.only:
        names = [x.strip() for x in args.only.split(",") if x.strip()]
    else:
        names = ORDER[:]
    if len(names) != len(set(names)) or any(x not in TRIAL_BY_NAME for x in names):
        parser.error("Invalid or duplicate IDs in --only")
    if args.plan:
        prepare()
        print("[STRUCT10 PLAN ONLY] no training started", flush=True)
        return
    if not base.core.CONFIG_DIR.is_dir():
        prepare()
    run_selected(names, args.patience, args.screen_epochs)


if __name__ == "__main__":
    main()
