import pathlib
import tempfile
import unittest
import importlib.util
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('train_patch', ROOT / 'install_training_controls.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

RUNNER = '''import logging
import torch
class Runner:
    @property
    def optimizer(self):
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                print(n)
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %d" % num_parameters)
            self._num_trainable_para = num_parameters > 0
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        return None
'''
SCHED = '''def cosine(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def warmup(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def step(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
'''
TASK = '''class BaseTask:
    def _train_inner_loop(self, model, scaler, optimizer, iters_per_epoch, accum_grad_iters=1):
        use_amp = scaler is not None
        for i in range(iters_per_epoch):
            loss = model(i)
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
'''

class AttrDict(dict):
    __getattr__ = dict.get

class PatchTests(unittest.TestCase):
    def test_patches_parse_and_marker(self):
        for fn, src in [(mod.patch_runner, RUNNER),(mod.patch_schedule,SCHED),(mod.patch_task,TASK)]:
            patched=fn(src)
            self.assertIn(mod.MARKER,patched)
            compile(patched, '<patched>', 'exec')
            with self.assertRaises(RuntimeError):
                fn(patched)

    def test_scheduler_per_group(self):
        s=mod.patch_schedule(SCHED)
        ns={}
        exec(s,ns)
        class Dummy: pass
        x=Dummy(); x.param_groups=[{'lr_scale':0.3},{'lr_scale':1.0},{'weight_decay':0}]
        for func in ('cosine','warmup','step'):
            ns[func](x,0.0001)
            self.assertAlmostEqual(x.param_groups[0]['lr'],0.00003)
            self.assertAlmostEqual(x.param_groups[1]['lr'],0.0001)
            self.assertAlmostEqual(x.param_groups[2]['lr'],0.0001)

    def test_optimizer_scale_groups(self):
        import torch.nn as nn
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.rec_encoder=nn.Linear(2,2)
                self.llama_proj=nn.Linear(2,2)
                self.gcn=nn.Linear(2,2)
                self.lora_A=nn.Linear(2,2)
        ns={}
        exec(mod.patch_runner(RUNNER),ns)
        r=ns['Runner']()
        r._optimizer=None
        r.model=Model()
        cfg=type('Cfg',(),{})()
        cfg.model_cfg=AttrDict(gcn_layer=31)
        cfg.run_cfg=AttrDict(init_lr=1e-4,weight_decay=0.001,lr_scale_gnn=0.25,lr_scale_proj=0.5)
        r.config=cfg
        gs={g['group_name']:g for g in r.optimizer.param_groups}
        self.assertAlmostEqual(gs['gnn_wd']['lr_scale'],0.25)
        self.assertAlmostEqual(gs['proj_no_wd']['lr_scale'],0.5)
        self.assertAlmostEqual(gs['lora_wd']['lr_scale'],1.0)
        self.assertEqual(gs['rec_no_wd']['weight_decay'],0)
        self.assertAlmostEqual(gs['gnn_wd']['weight_decay'],0.001)

    def test_accumulation_correct_divisible_and_tail(self):
        ns={}
        exec(mod.patch_task(TASK),ns)
        import torch.nn as nn
        torch.manual_seed(5)
        for steps in (1, 2, 3, 4, 5, 6, 7):
            for factor in (1, 2, 3, 4):
                model1=nn.Linear(1,1,bias=False)
                model2=nn.Linear(1,1,bias=False)
                model2.load_state_dict(model1.state_dict())
                opt1=torch.optim.SGD(model1.parameters(),lr=0.1)
                opt2=torch.optim.SGD(model2.parameters(),lr=0.1)
                xs=torch.arange(1,steps+1,dtype=torch.float32).view(-1,1)
                ys=xs*0.7+0.2
                def m1(i): return ((model1(xs[i:i+1])-ys[i:i+1])**2).mean()
                task=ns['BaseTask']()
                task._train_inner_loop(m1,None,opt1,steps,factor)
                for a in range(0,steps,factor):
                    b=min(steps,a+factor)
                    ((model2(xs[a:b])-ys[a:b])**2).mean().backward()
                    opt2.step();opt2.zero_grad()
                self.assertTrue(torch.allclose(model1.weight,model2.weight,atol=1e-5), (steps, factor, model1.weight,model2.weight))

    def test_no_code_mutation_check(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            for path,text in zip(mod.FILES,(RUNNER,SCHED,TASK)):
                q=root/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text)
            with patch('sys.argv',['install_training_controls.py','--root',td,'--check']):
                self.assertEqual(mod.main(),0)
            for path,text in zip(mod.FILES,(RUNNER,SCHED,TASK)):
                self.assertEqual((root/path).read_text(),text)
            with patch('sys.argv',['install_training_controls.py','--root',td,'--apply']):
                self.assertEqual(mod.main(),0)
            self.assertEqual(len(list(root.rglob('*.before_train_controls_*'))),3)
            for path in mod.FILES:
                self.assertIn(mod.MARKER,(root/path).read_text())

if __name__=='__main__':unittest.main()
