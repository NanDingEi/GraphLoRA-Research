#!/usr/bin/env python3
"""Regression check for probe resumability after the historical counter bug.

Runs on CPU without loading Vicuna, reading any datasets, or launching training.
Only tests whether existing init-probe fingerprints can be safely reused.
"""
from pathlib import Path
import importlib.util
import json
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("no31_pair_for_resume_test", HERE / "run_pair.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)

groups = {g: {"sha256": g + "-mock-hash", "trainable": 1}
          for g in ("MF", "GNN", "Projector", "LoRA")}

with tempfile.TemporaryDirectory(prefix="no31-probe-resume-") as td:
    previous_log = pair.LOG
    try:
        pair.LOG = Path(td)
        for arm in ("no31", "cora"):
            (pair.LOG / (arm + "_probe_init.json")).write_text(json.dumps({
                "mode": arm, "groups": groups
            }), encoding="utf-8")
        (pair.LOG / "no31_probe.log").write_text(
            "[NO31-PROBE-OK] mode=no31\n", encoding="utf-8"
        )
        (pair.LOG / "cora_probe.log").write_text(
            "[NO31-PROBE-OK] mode=cora\n"
            '[NO31-CORA-COUNTERS] {"initialized": 1}\n'
            "Traceback (most recent call last):\n"
            "RuntimeError: CoRA branch q never executed\n",
            encoding="utf-8"
        )
        pair.launch("no31", 1, probe=True)
        pair.launch("cora", 1, probe=True)
        pair.check_hashes("probe")
        # Reject all unknown post-probe errors rather than incorrectly
        # interpreting a bad probe as healthy.
        (pair.LOG / "cora_probe.log").write_text(
            "[NO31-PROBE-OK] mode=cora\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: unknown unrelated failure\n",
            encoding="utf-8"
        )
        try:
            pair.launch("cora", 1, probe=True)
        except RuntimeError as exc:
            assert "Unsafe existing probe" in str(exc)
        else:
            raise AssertionError("Unexpectedly reused a failed non-counter probe")
    finally:
        pair.LOG = previous_log

entry = (HERE / "entry.py").read_text(encoding="utf-8")
assert "train.main()\n    # The @record training wrapper" in entry
assert "if args.probe_only:\n        return\n    if runtime is not None:" in entry
print("[NO31-PROBE-RECOVERY-CPU-PASS] both completed probes reused; unsafe errors rejected")
