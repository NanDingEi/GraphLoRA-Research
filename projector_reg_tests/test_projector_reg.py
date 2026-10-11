import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import ast
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = import_file('pr', HERE / 'minigpt4/models/projector_structure_reg.py')
ap = import_file('ap', HERE / 'apply_projector_reg.py')

FAKE = '''import torch
import torch.nn as nn
class MiniGPT4Rec_v2(nn.Module):
    def __init__(
            self,
            rec_model="MF",
            proj_token_num=1,
            gcn_layer =0
    ):
        super().__init__()
        self.proj_token_num = proj_token_num
        self.a = 1

    def forward_v2(self, samples):
        loss = torch.tensor(1.)
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        rec_model = cfg.get('rec_model', 'MF')
        proj_mid = cfg.get('proj_mid_times')
        model = cls(
            rec_model=rec_model,
            proj_mid=proj_mid,
            gcn_layer=cfg.get('gcn_layer', 0),
        )
        return model
'''
CFG = 'model:\n  arch: mini_gpt4rec_v2\n  proj_mid_times: 10\n  gcn_layer: 31\nrun:\n  seed: 42\n'


class ProjectorRegTests(unittest.TestCase):
    def test_identity(self):
        torch.manual_seed(8)
        proj = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            proj.weight.copy_(torch.eye(8))
        table = torch.randn(100, 8, requires_grad=True)
        loss = pr.projector_cosine_reg_loss(proj, table, num_items=40, padding_index=0)
        self.assertLess(loss.item(), 1e-12)

    def test_grad_projector_only(self):
        torch.manual_seed(17)
        proj = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 12))
        table = torch.randn(90, 8, requires_grad=True)
        loss = pr.projector_cosine_reg_loss(proj, table, num_items=32)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertIsNone(table.grad)
        self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in proj.parameters()))

    def test_patch_stock_source(self):
        src = ap.patch_source(FAKE)
        self.assertIn('proj_reg_weight=cfg.get', src)
        self.assertIn('@projector_regularized_forward', src)
        self.assertIn('proj_reg_weight=0.0', src)
        self.assertIn('gcn_layer =0,', src)
        self.assertEqual(src.count('@projector_regularized_forward'), 1)
        self.assertIn('return {"loss": loss}', src)
        ast.parse(src)
        with self.assertRaisesRegex(RuntimeError, 'already installed'):
            ap.patch_source(src)

    def test_patch_multiple_custom_return_branches_untouched(self):
        original = FAKE.replace('''        loss = torch.tensor(1.)
        return {"loss": loss}''', '''        if samples.get("early"):
            return {"loss": torch.tensor(0.5), "N1": 3}
        if samples.get("rlm"):
            return {"loss": total_loss, "semantic": semantics}
        outputs = {"loss": loss, "foo": 42}
        return outputs''')
        patched = ap.patch_source(original)
        self.assertIn('return {"loss": torch.tensor(0.5), "N1": 3}', patched)
        self.assertIn('return {"loss": total_loss, "semantic": semantics}', patched)
        self.assertIn('return outputs', patched)
        self.assertEqual(patched.count('@projector_regularized_forward'), 1)
        ast.parse(patched)

    def test_patch_custom_decorator(self):
        original = FAKE.replace('    def forward_v2(self, samples):', '    @some_other_decorator\n    def forward_v2(self, samples):')
        patched = ap.patch_source(original)
        self.assertIn('    @projector_regularized_forward\n    @some_other_decorator\n    def forward_v2', patched)

    def test_patch_config(self):
        result = ap.patch_config(CFG, 0.001, 64)
        self.assertIn('  proj_reg_weight: 0.001', result)
        self.assertIn('  gcn_layer: 31', result)
        with self.assertRaisesRegex(RuntimeError, 'already appears'):
            ap.patch_config(result, 0.001, 64)

    def test_decorator_preserves_other_losses_and_branch(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_reg_weight = 0.01
                self.proj_reg_items = 16
                self.llama_proj = nn.Sequential(nn.Linear(8, 10), nn.Tanh(), nn.Linear(10, 12))
                self.rec_encoder = SimpleNamespace(item_embedding=nn.Embedding(80, 8), padding_index=0)
                self._proj_reg_step = 0
            @pr.projector_regularized_forward
            def forward_v2(self, samples):
                if samples.get('rlm'):
                    return {'loss': (self.llama_proj.weight if False else torch.tensor(0.8, requires_grad=True)), 'RLM': 7}
                return {'loss': torch.tensor(0.9, requires_grad=True), 'other': torch.tensor(0.4)}
        m = Model().train()
        output = m.forward_v2({'rlm': True})
        self.assertEqual(output['RLM'], 7)
        self.assertGreater(output['loss'].item(), 0.8)
        output['loss'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.llama_proj.parameters()))
        self.assertIsNone(m.rec_encoder.item_embedding.weight.grad)
        self.assertEqual(m._proj_reg_step, 1)
        m.proj_reg_weight = 0
        untouched = m.forward_v2({'rlm': False})
        self.assertIn('other', untouched)
        self.assertEqual(m._proj_reg_step, 1)
        m.proj_reg_weight = 0.01
        m.eval()
        m.forward_v2({})
        self.assertEqual(m._proj_reg_step, 1)

    def test_missing_loss_in_training_rejects(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_reg_weight = 0.01
            @pr.projector_regularized_forward
            def forward_v2(self, samples):
                return {'logits': torch.ones(2)}
        with self.assertRaisesRegex(RuntimeError, 'without "loss"'):
            M().train().forward_v2({})


if __name__ == '__main__':
    unittest.main()
