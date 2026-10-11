import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiments", ROOT / "run_three_experiments.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class PlanTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "model": {"gnn_semantic_align": False, "gnn_semantic_weight": 0, "gnn_scale": 0.1,
                      "train_gnn_with_rec_loss": False, "rec_config": {"embedding_size": 256},
                      "injection_mode": "legacy", "proj_reg_weight": 0},
            "run": {"max_epoch": 3, "init_lr": 1e-4, "evaluate": True, "batch_size_train": 3,
                    "output_dir": "/old", "resume_ckpt_path": "/old/checkpoint.pth"},
            "datasets": {"amazon_ood": {"val_file": "valid_ood2.pkl"}},
        }

    def test_only_3_nonzero_semantic_configs(self):
        self.assertEqual(len(mod.RUNS), 3)
        self.assertEqual([a[2] for a in mod.RUNS], [0.0001, 0.0005, 0.001])
        self.assertTrue(all(a[1] for a in mod.RUNS))

    def test_unchanged_training_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, semantic, weight, _ in mod.RUNS:
                c = mod.prepare_config(self.base, name, semantic, weight, Path(tmp))
                self.assertTrue(c['model']['gnn_semantic_align'])
                self.assertEqual(c['model']['gnn_semantic_weight'], 1e-4)
                self.assertEqual(c['model']['proj_reg_weight'], weight)
                self.assertEqual(c['model']['injection_mode'], 'legacy')
                self.assertTrue(c['model']['train_gnn_with_rec_loss'])
                self.assertEqual(c['run']['max_epoch'], 3)
                self.assertEqual(c['run']['batch_size_train'], 3)
                self.assertEqual(c['datasets'], self.base['datasets'])
                self.assertEqual(c['run']['init_lr'], 1e-4)
                self.assertFalse(c['run']['evaluate'])
                self.assertIsNone(c['run']['resume_ckpt_path'])
                self.assertNotEqual(c['run']['output_dir'], self.base['run']['output_dir'])
        self.assertEqual(self.base['model']['gnn_semantic_align'], False)

    def test_metric_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'job_A'; p.mkdir()
            (p / 'valid_log.txt').write_text(
                '\n'.join(json.dumps(d) for d in [
                    {'valid_agg_metrics': .732, 'valid_uauc': [.71, ['u1']], 'valid_epoch': 0},
                    {'valid_agg_metrics': .740, 'valid_uauc': [.722, ['u1']], 'valid_epoch': 1},
                ]), encoding='utf-8')
            rows = mod.read_valid_metrics(Path(tmp))
            self.assertEqual(len(rows), 2)
            self.assertEqual(max(rows, key=lambda r: r['valid_auc'])['epoch'], 1)
            self.assertEqual(rows[1]['valid_uauc'], .722)

if __name__ == '__main__':
    unittest.main()
