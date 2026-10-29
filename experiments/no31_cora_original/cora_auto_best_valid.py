#!/usr/bin/env python3
"""Evaluate the official CoRA best checkpoint on VALID (Yes logit vs Yes-No).

Run from CoRA_official, using --run. No upstream source edits, no training,
no TEST inference. The second score is an optional research ablation.
"""
import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path.cwd()
BASE_CONFIG = Path('local_configs/cora_mf_ood2_official.yaml')
EVAL_CONFIG = Path('local_configs/cora_eval_best_valid_auto.yaml')


def find_best(cfg):
    out = cfg['run'].get('output_dir', '')
    if not out:
        raise RuntimeError('run.output_dir is missing from original train config')
    folder = Path(out).expanduser()
    if not folder.is_absolute():
        folder = ROOT / folder
    if not folder.is_dir():
        raise RuntimeError(f'Train output directory missing: {folder}')
    checkpoints = [p for name in ('checkpoint_best.pth', 'best_checkpoint.pth')
                   for p in folder.rglob(name) if p.is_file()]
    if not checkpoints:
        raise RuntimeError(f'No checkpoint_best.pth found under {folder}')
    checkpoints.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    if len(checkpoints) > 1:
        print(f'[WARNING] Found {len(checkpoints)} best checkpoints; latest file chosen, '
              'not highest validation AUC across runs.', flush=True)
    print(f'[AUTO BEST] {checkpoints[0]}', flush=True)
    return checkpoints[0]


def prepare():
    if not (ROOT / 'train_collm_mf_din.py').is_file():
        raise RuntimeError('Run inside the CoRA_official repository directory')
    if not BASE_CONFIG.is_file():
        raise RuntimeError(f'Missing original config: {BASE_CONFIG}')
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8'))
    best = find_best(cfg)
    cfg['model']['ckpt'] = str(best)
    cfg['run']['evaluate'] = True
    cfg['run']['valid_splits'] = ['valid']
    cfg['run']['test_splits'] = ['valid']  # evaluate VALID, never TEST
    cfg['run']['batch_size_eval'] = 2
    cfg['run']['output_dir'] = str(ROOT / 'output' / 'CORA_EVAL_BEST_VALID_AUTO')
    EVAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    EVAL_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'[EVAL CONFIG] {EVAL_CONFIG}; TEST not accessed', flush=True)
    return EVAL_CONFIG


def worker(mode):
    if mode == 'margin':
        import torch.nn.functional as F
        from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
        upstream = MiniGPT4Rec_vx.generate_for_samples_v2

        def margin_valid(self, samples, return_all=False):
            outputs, yes = upstream(self, samples, return_all=True)
            answers = [self.pos_ans[0] if int(t) else self.neg_ans[0]
                       for t in samples['label']]
            answer_tokens = self.llama_tokenizer(
                answers, return_tensors='pt', padding='longest', truncation=True,
                max_length=self.max_txt_len, add_special_tokens=False)
            offset = answer_tokens.input_ids.shape[-1] + 1
            no_id = self.llama_tokenizer(
                self.neg_ans[0], add_special_tokens=False).input_ids[0]
            no = outputs.logits[:, -offset, no_id]
            no = no.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
            margin = yes - no
            loss = F.binary_cross_entropy_with_logits(yes, samples['label'].float())
            return {'loss': loss.nan_to_num(nan=0.0), 'logits': margin}

        MiniGPT4Rec_vx.generate_for_samples_v2 = margin_valid
        print('[SCORE] Yes logit minus No logit; inference-only monkeypatch', flush=True)
    else:
        print('[SCORE] Unmodified official Yes logit', flush=True)

    sys.argv = ['train_collm_mf_din.py', '--cfg-path', str(EVAL_CONFIG)]
    runpy.run_path(str(ROOT / 'train_collm_mf_din.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Evaluate both scores, serially on GPU0')
    parser.add_argument('--_worker', choices=['official', 'margin'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker)
        return
    prepare()
    if not args.run:
        print('Use --run for sequential VALID evaluations (GPU0 only).')
        return
    for mode in ('official', 'margin'):
        logfile = ROOT / 'logs' / f'cora_best_valid_{mode}.log'
        print(f'\n[START VALID] {mode}; log={logfile}', flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        with logfile.open('w', encoding='utf-8') as log:
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--_worker', mode]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = proc.wait()
        if code:
            raise SystemExit(f'[FAILED] {mode} validation, exit={code}; see {logfile}')
        print(f'[DONE VALID] {mode}', flush=True)
    print('\n[DONE] Both results are VALID-only. Margin is NOT original CoRA.')


if __name__ == '__main__':
    main()
