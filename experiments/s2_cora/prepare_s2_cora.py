#!/usr/bin/env python3
"""Prepare two matching training YAMLs from the recorded S2_groups_acc3 recipe.

No S2 checkpoint warm-start, no MF reset, no R4 freeze. This only creates
configuration files; it does not execute training.
"""
import copy
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tran_configs/S2_groups_acc3_recovered.yaml"
ROOT_OUT = ROOT / "output/S2_ROLLBACK_CORA_20261022"
ROOT_LOG = ROOT / "logs/S2_ROLLBACK_CORA_20261022"
REQUIRED = {
    "model.gnn_semantic_align": True,
    "model.gnn_semantic_weight": 1e-4,
    "model.proj_reg_weight": 5e-4,
    "model.proj_reg_items": 64,
    "model.train_gnn_with_rec_loss": True,
    "model.gcn_layer": 31,
    "model.gnn_scale": 0.1,
    "run.accum_grad_iters": 3,
    "run.batch_size_train": 3,
    "run.init_lr": 1e-4,
    "run.min_lr": 5e-5,
    "run.lr_scale_rec": 0.5,
    "run.lr_scale_gnn": 0.25,
    "run.lr_scale_proj": 0.5,
    "run.lr_scale_lora": 1.0,
    "run.seed": 42,
    "run.max_epoch": 3,
}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    m, r = data["model"], data["run"]
    for nested, expected in REQUIRED.items():
        scope, name = nested.split(".")
        got = data[scope].get(name)
        if isinstance(expected, float):
            valid = got is not None and abs(float(got) - expected) <= 1e-12
        else:
            valid = got == expected
        if not valid:
            raise RuntimeError(f"S2 recipe mismatch: {nested} = {got!r}; expected {expected!r}")
    rec_path = Path(m["rec_config"]["pretrained_path"])
    if not rec_path.is_file():
        raise FileNotFoundError(f"Original pretrained MF weights missing: {rec_path}")
    if m.get("ckpt") not in ("", None) or r.get("resume_ckpt_path") not in ("", None):
        raise RuntimeError("Not fresh GraphLoRA training: pretrained S2 ckpt or resume specified")
    if m.get("rec_model") != "MF" or m.get("lora_gnn") != "ngcf":
        raise RuntimeError("Wrong S2 MF/NGCF model")
    for k in ("freeze_rec", "freeze_lora", "freeze_proj"):
        if m.get(k) is not False:
            raise RuntimeError(f"Historical S2 trains all modules: {k} must be false")
    if m.get("fix_autocast_cache", False) is not False:
        raise RuntimeError("S2 historical FX flag changed: refusing a different baseline")
    # Graph npz files are read RELATIVE to the S2 repository root even though
    # the train/valid/test pickle paths are absolute. Reuse the archived S2
    # graph files from the original GraphLoRA, never regenerate them.
    graph_root = ROOT / "dataset/ml-1m"
    reference_graph_root = Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m")
    for split in ("train", "valid", "test"):
        fname = f"movie_graph_{split}.npz"
        expected = reference_graph_root / fname
        actual = graph_root / fname
        if not expected.is_file():
            raise FileNotFoundError(
                f"Original S2 adjacency graph missing: {expected}")
        if not actual.exists():
            actual.parent.mkdir(parents=True, exist_ok=True)
            actual.symlink_to(expected)
            print(f"[S2-GRAPH-LINK] {actual} -> {expected}", flush=True)
        if not actual.is_file():
            raise FileNotFoundError(f"S2 graph is not a readable file: {actual}")
        def digest(path):
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        if digest(actual) != digest(expected):
            raise RuntimeError(f"S2 graph differs from original: {actual}")
    print("[S2-GRAPH-PARITY-OK] train/valid/test exact original graph files", flush=True)
    # Never write anything into the original S2_groups_acc3 output directory.
    if ROOT_LOG.exists() or ROOT_OUT.exists():
        raise FileExistsError(f"Existing experiment files: {ROOT_LOG} / {ROOT_OUT}")
    ROOT_LOG.mkdir(parents=True, exist_ok=False)
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for arm in ("BASELINE", "CORA"):
        cfg = copy.deepcopy(data)
        cfg["model"]["ckpt"] = None
        cfg["run"]["resume_ckpt_path"] = None
        cfg["run"]["evaluate"] = False
        cfg["run"]["test_splits"] = []       # Validation only; no test leakage
        cfg["run"]["output_dir"] = str(ROOT_OUT / arm)
        cfg["run"]["max_epoch"] = 3          # preserve historical cosine schedule
        cfg["run"]["save_checkpoint_every_n_epochs"] = 1
        cfg["run"].pop("diagnostic_stop_after_epoch0", None)
        cfg["run"].pop("stop_after_n_epochs", None)
        cfg["run"].pop("max_train_steps", None)
        # Do not change the model config between arms. The CoRA entrypoint,
        # not an altered S2 config, installs the additional module.
        (ROOT_LOG / (arm + ".yaml")).write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    base_cfg = yaml.safe_load((ROOT_LOG / "BASELINE.yaml").read_text())
    cora_cfg = yaml.safe_load((ROOT_LOG / "CORA.yaml").read_text())
    base_cfg["run"].pop("output_dir")
    cora_cfg["run"].pop("output_dir")
    if base_cfg != cora_cfg:
        raise RuntimeError("Configurations differ beyond output_dir!")
    (ROOT_LOG / "manifest.json").write_text(json.dumps({
        "status": "prepared", "historical_s2_commit": "d818fa2943ede4e9a810b032a3422e94d0c77ed0",
        "historical_s2_val_auc": 0.7328835632321707,
        "historical_s2_val_uauc": 0.6879526108156818,
        "historical_config": str(SOURCE), "historical_config_sha256": source_sha,
        "mf_pretrained_path": str(rec_path), "train_from_graphlora_ckpt": False,
        "training_policy": "MF pretrained; NGCF/Projector/LLM LoRA initialized fresh",
        "same_config_except_output_dir": True, "epochs": 3, "seed": 42,
        "changes_to_s2_model": "Expose penultimate NGCF only under CoRA runtime flag"
    }, indent=2), encoding="utf-8")
    print("[S2-RECIPE-PREPARED] historical_AUC=0.732883563 "
          "MF=pretrained CoRA=optional LoRA/GNN=training accum=3 "
          "lr_rec=.5 lr_gnn=.25 lr_proj=.5 reg=5e-4", flush=True)
    print("[S2-NO-TRAIN] YAMLs created; next run --probe-only to compare initial hashes",
          flush=True)


if __name__ == "__main__":
    main()
