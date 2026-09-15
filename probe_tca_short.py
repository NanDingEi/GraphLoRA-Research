import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
r = subprocess.run([os.path.expanduser("~/anaconda3/envs/minigpt4/bin/python"),
                    "train_collm_mf_din.py",
                    "--cfg-path", "tran_configs/collm_pretrain_mf_ood_scratch_rel_pref.yaml",
                    "--options", "run.max_epoch=1 run.iters_per_epoch=3 run.batch_size_train=1",
                    "run.batch_size_eval=2 model.pref_tca=True",
                    "run.output_dir=/home/bbx/user_lx/GraphLoRA/output/probe_tca"],
                   capture_output=True, text=True)
for line in (r.stdout + r.stderr).splitlines():
    if "tca-dbg" in line or "mutual:rel" in line or "Averaged" in line or "Error" in line:
        print(line[:160])
