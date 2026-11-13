#!/usr/bin/env python3
"""CPU-only synthetic contract tests; these do NOT reproduce real CoRA scores."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("ensemble",HERE/"cora_ensemble_repro_improve.py")
m=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def run():
    rng=np.random.default_rng(42)
    n=160
    users=np.repeat(np.arange(20),8)
    items=np.tile(np.arange(8),20)
    labels=np.tile([0,1,0,1,0,1,0,1],20).astype(np.int64)
    scores=rng.normal(size=n)+0.75*labels
    A=scores.copy()
    C=scores + rng.normal(0,.13,n)
    B=scores + rng.normal(0,.14,n)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        for key, score in (("A",A),("B",B),("C",C)):
            # C determined by layers to validate recomputation.
            z32=score.copy()
            z31=z32.copy()
            if key=="C":
                z32=scores.copy(); z31=scores+4*(C-scores)
            mat=np.stack((z32,z32,z31,z32))
            fields=dict(labels=labels,users=users,items=items,yes_logits=mat)
            if key=="B": fields["fused_logits"]=score
            np.savez(root/(key+"_valid.npz"),**fields)
        class Args: pass
        args=Args()
        for key in "abc": setattr(args,key+"_valid",str(root/(key.upper()+"_valid.npz")))
        for key in "abc": setattr(args,key+"_score_key",None)
        args.norm="test_first128"
        args.selection=str(root/"locked.json")
        args.overwrite_selection=False
        args.rr5_checkpoint=None
        args.skip_rr5_audit=True
        args.expected_historical_w=0.40
        args.allow_different_valid_weight=True
        mm,digest=m.load_triplet(args,"valid")
        assert len(digest)==64
        assert np.allclose(mm["C"].score,C)
        assert np.allclose(mm["A"].score,A)
        assert np.allclose(mm["B"].score,B)
        fast=m.ValidMetricEvaluator(labels,users)
        for pred in (A,B,C):
            got=fast.metrics(pred)
            expected=m.metrics(labels,users,pred)
            assert abs(got["uauc"]-expected["uauc"])<1e-12,(got,expected)
        m.select(args)
        locked=__import__('json').loads(Path(args.selection).read_text())
        assert len(locked["methods"])==6
        for a in ("E2_AB_HALF","E2_AB_AUC","E3_ABC_THIRD","I1_SIMPLEX_AUC_UAUC_FLOOR","I2_UAUC_PARETO_C","I3_USER_DISAGREEMENT_C"):
            assert a in locked["methods"]
        # Hard misalignment fail.
        swapped=dict(labels=labels.copy(),users=users.copy(),items=items.copy(),yes_logits=np.stack((A,A,A,A)))
        swapped['items'][0]+=123456
        np.savez(root/"A_wrong.npz",**swapped)
        try:
            m.check_align({"A":mm["A"],"C":m.load_member(str(root/"A_wrong.npz"),"A")})
        except ValueError as e:
            assert "misalignment" in str(e)
        else:
            raise AssertionError("Misaligned NPZ silently accepted")
        import torch
        import torch.nn.functional as F
        torch.manual_seed(42)
        tab = torch.randn(20,16)
        layer_norm_weight = torch.ones(16)
        layer_norm_bias = torch.zeros(16)
        fc1_w,fc1_b=torch.randn(6,16)*0.05, torch.randn(6)*0.05
        fc3_w,fc3_b=torch.randn(1,6)*0.05,torch.tensor([-1.5])
        state={
          'cora_residual_fusion.user_gate.0.weight':layer_norm_weight,
          'cora_residual_fusion.user_gate.0.bias':layer_norm_bias,
          'cora_residual_fusion.user_gate.1.weight':fc1_w,
          'cora_residual_fusion.user_gate.1.bias':fc1_b,
          'cora_residual_fusion.user_gate.3.weight':fc3_w,
          'cora_residual_fusion.user_gate.3.bias':fc3_b,
          'rec_encoder.user_embedding.weight':tab,
        }
        ckpt=root/'fake_RR5.pth'
        torch.save({'model':state,'epoch':21},ckpt)
        z32=B.copy()
        z31=z32+rng.normal(0,1,n)
        x=F.normalize(tab[torch.from_numpy(users).long()],p=2,dim=1,eps=1e-8)
        x=F.layer_norm(x,(16,),layer_norm_weight,layer_norm_bias,eps=1e-5)
        x=F.gelu(F.linear(x,fc1_w,fc1_b))
        gate=torch.sigmoid(F.linear(x,fc3_w,fc3_b))[:,0].numpy()
        recon=z32+gate*(z31-z32)
        member=m.Member('fake','B',labels,users,items,recon,z31,z32,True,['fused_logits'])
        result=m.audit_rr5_gate(member,str(ckpt))
        assert result['max_abs_error']<1e-6
        print("[CPU TEST PASS] A/B/C fusion, grouped UAUC, VALID selection, alignment fail-closed, independent RR5 gate+MF reconstruction",flush=True)
if __name__=="__main__": run()
