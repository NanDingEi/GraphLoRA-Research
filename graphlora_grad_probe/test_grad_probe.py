import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import torch
from torch import nn

HERE=Path(__file__).parent

class TestGradientProbe(unittest.TestCase):
    def setUp(self):
        self.fixture_path = HERE / 'test_fixtures/projector_structure_reg_v2.py'

    def _module(self):
        src=self.fixture_path.read_text(encoding='utf-8')
        spec=importlib.util.spec_from_file_location('gp_installer',HERE/'install_grad_probe.py')
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        new=mod.patch(src)
        self.assertIn(mod.MARKER,new)
        ast.parse(new)
        ns={'__name__':'probe_test'}
        exec(compile(new,'inmemory','exec'),ns)
        return new,ns

    def test_double_patch_rejected(self):
        new,ns=self._module()
        spec=importlib.util.spec_from_file_location('gp_installer',HERE/'install_grad_probe.py')
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with self.assertRaisesRegex(RuntimeError,'already installed'):
            mod.patch(new)

    def test_grad_unmodified_and_similarity(self):
        new,ns=self._module()
        proj=nn.Sequential(nn.Linear(4,6),nn.ReLU(),nn.Linear(6,8))
        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.llama_proj=proj
                self.rec_encoder=type('R',(),{})()
                self.rec_encoder.item_embedding=nn.Embedding(40,4)
                self.proj_reg_weight=0.0005
                self.proj_reg_items=8
            @ns['projector_regularized_forward']
            def forward_v2(self, x):
                return {'loss':self.llama_proj(x).square().mean()}
        model=Stub()
        model.train()
        x=torch.randn(5,4)
        old={k:os.environ.get(k) for k in ['GLORA_GRAD_PROBE','GLORA_GRAD_FIRST','GLORA_GRAD_SCOPE']}
        os.environ['GLORA_GRAD_PROBE']='1'
        os.environ['GLORA_GRAD_FIRST']='5'
        os.environ['GLORA_GRAD_SCOPE']='first'
        try:
            result=model.forward_v2(x)
            self.assertEqual(len(result),1)
            self.assertIsNone(proj[0].weight.grad)
            result['loss'].backward()
            self.assertIsNotNone(proj[0].weight.grad)
            self.assertTrue(torch.isfinite(proj[0].weight.grad).all())
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v

    def test_probe_does_not_change_parameter_gradient(self):
        new, ns = self._module()
        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.llama_proj = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 8))
                self.rec_encoder = type('R', (), {})()
                self.rec_encoder.item_embedding = nn.Embedding(40, 4)
                self.proj_reg_weight = 0.0005
                self.proj_reg_items = 8
            @ns['projector_regularized_forward']
            def forward_v2(self, x):
                return {'loss': self.llama_proj(x).square().mean()}
        model = Stub()
        model.train()
        x = torch.randn(5, 4)
        keys = ['GLORA_GRAD_PROBE', 'GLORA_GRAD_FIRST', 'GLORA_GRAD_SCOPE']
        old = {k: os.environ.get(k) for k in keys}
        try:
            grads = []
            for probe in ('0','1'):
                os.environ['GLORA_GRAD_PROBE'] = probe
                os.environ['GLORA_GRAD_FIRST'] = '5'
                os.environ['GLORA_GRAD_SCOPE'] = 'all'
                torch.manual_seed(777)
                model.zero_grad(set_to_none=True)
                loss = model.forward_v2(x)['loss']
                loss.backward()
                grads.append([p.grad.clone() for p in model.llama_proj.parameters()])
            for off, on in zip(*grads):
                self.assertTrue(torch.equal(off, on))
        finally:
            for k,v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

    def test_cli_check_does_not_modify(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'minigpt4/models/projector_structure_reg.py'
            target.parent.mkdir(parents=True)
            src=self.fixture_path.read_bytes()
            target.write_bytes(src)
            out=subprocess.run(['python',str(HERE/'install_grad_probe.py'),'--project',d,'--check'],capture_output=True,text=True)
            self.assertEqual(out.returncode,0,out.stderr)
            self.assertEqual(target.read_bytes(),src)
            out=subprocess.run(['python',str(HERE/'install_grad_probe.py'),'--project',d,'--apply'],capture_output=True,text=True)
            self.assertEqual(out.returncode,0,out.stderr)
            self.assertIn('GLORA_GRAD_PROBE_V1',target.read_text())
            self.assertEqual(len(list(target.parent.glob('*.before_grad_probe_*'))),1)

if __name__=='__main__':unittest.main()
