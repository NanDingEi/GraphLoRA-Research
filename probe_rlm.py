import warnings; warnings.filterwarnings('ignore')
import sys, os, random, torch
import pandas as pd
sys.path.insert(0, os.getcwd())
from torch.utils.data import DataLoader
import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.data_utils import prepare_sample

class A: pass
args = A(); args.cfg_path = 'tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml'; args.options = None
cfg = Config(args); cfg.model_cfg.ckpt = None
data_dir = cfg.datasets_cfg.amazon_ood.path
tr = pd.read_pickle(data_dir+'train_ood2.pkl'); va = pd.read_pickle(data_dir+'valid_ood2.pkl'); te = pd.read_pickle(data_dir+'test_ood2.pkl')
cfg.model_cfg.rec_config.user_num = int(max(tr.uid.max(), va.uid.max(), te.uid.max())+1)
cfg.model_cfg.rec_config.item_num = int(max(tr.iid.max(), va.iid.max(), te.iid.max())+1)
task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)
model = task.build_model(cfg); model.set_mode('v2'); model.to('cuda')
print(f"mutual_mode={model.mutual_mode}, bridge_dim={model.rlm_bridge_dim}, tau={model.rlm_tau}")

inner = list(datasets.values())[0]
loader = DataLoader(inner['train'], batch_size=1, shuffle=False, collate_fn=getattr(inner['train'],'collater',None))
it = iter(loader)
while True:
    batch = next(it)
    if int(batch['InteractedNum'][0]) > 3: break
samples = prepare_sample(batch, cuda_enabled=True)
model.train()
model.zero_grad(set_to_none=True)

fb_stash = {}
orig = model.compute_feedback_gcn
def spy(*a, **kw):
    out = orig(*a, **kw)
    fb_stash.update(out)
    return out
model.compute_feedback_gcn = spy
out = model(samples)
model.compute_feedback_gcn = orig
print(f"loss={out['loss'].item():.4f}")
for k in ("loss_rec","loss_l2g","loss_g2l"):
    if k in out: print(f"  {k}={out[k].item():.6f}")

# 梯度方向验证
tg = fb_stash["token_gnn"]
hm = fb_stash["h_mutual"]
g1 = torch.autograd.grad(out["loss_l2g"], [model.rlm_head_G.weight, model.rlm_head_L.weight,
                                            model.gcn.layers[0].W1.weight,
                                            model.llama_model_lora.base_model.model.model.layers[3].self_attn.q_proj.lora_A['default'].weight],
                         allow_unused=True)
print(f"loss_l2g grads: P_G={'None' if g1[0] is None else f'{g1[0].norm().item():.3e}'} "
      f"P_L={'None' if g1[1] is None else f'{g1[1].norm().item():.3e}'} "
      f"gcn.W1={'None' if g1[2] is None else f'{g1[2].norm().item():.3e}'} "
      f"lora={'None' if g1[3] is None else 'present'}")
g2 = torch.autograd.grad(out["loss_g2l"], [model.rlm_head_L.weight, model.rlm_head_G.weight], allow_unused=True)
print(f"loss_g2l grads: P_L={'None' if g2[0] is None else f'{g2[0].norm().item():.3e}'} "
      f"P_G={'None' if g2[1] is None else f'{g2[1].norm().item():.3e}'}")
print("PROBE_RLM DONE")
