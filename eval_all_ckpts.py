"""Evaluate every saved checkpoint in an AVR logdir, dump metrics-vs-iter curve.

Usage:
    python eval_all_ckpts.py --logdir logs/MeshRIR/Meshrir_sliced_0.1s
    python eval_all_ckpts.py --logdir logs/TAU-SRIR/TAU_SRIR
"""
import os
import sys
import argparse
import json
import re
from pathlib import Path
import yaml
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/jooeun/AVR')
from infer import InferRunner


DATASET_DIR = {
    'MeshRIR': os.path.expanduser('~/data/dataset/MeshRIR/S1-M3969'),
    'TAU-SRIR': os.path.expanduser('~/data/dataset/TAU-SRIR/TAU-SRIR_DB/TAU-SRIR_DB'),
}


def eval_ckpt(runner, ckpt_path):
    runner.load_specific_checkpoint(str(ckpt_path))
    runner.renderer.eval()
    acc = {'Angle': 0, 'Amplitude': 0, 'Envelope': 0,
           'T60': 0, 'C50': 0, 'EDT': 0, 'multi_stft': 0}
    n = 0
    for batch in runner.test_iter:
        with torch.no_grad():
            ori_sig, position_rx, position_tx = batch
            if position_tx.dim() == 3 and position_tx.shape[1] == 1:
                position_tx = position_tx.squeeze(1)
            pred_sig = runner.renderer(position_rx.cuda(), position_tx.cuda())
            pred_sig = pred_sig[..., 0] + 1j * pred_sig[..., 1]
            ori_sig = ori_sig.cuda().to(pred_sig.dtype)
            _, m, _, _ = runner.calculate_metrics(pred_sig, ori_sig, runner.fs)
        for k in acc:
            acc[k] += float(m[k])
        n += 1
    return {k: v / n for k, v in acc.items()}, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', required=True)
    p.add_argument('--out', default=None,
                   help='Output JSON path (default: <logdir>/convergence.json)')
    args = p.parse_args()

    logdir = Path(args.logdir)
    cfg_path = logdir / 'avr_conf.yml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dataset_type = cfg['path']['dataset_type']
    dataset_dir = DATASET_DIR[dataset_type]

    out_path = args.out or str(logdir / 'convergence.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    runner = InferRunner(mode='train', dataset_dir=dataset_dir, batchsize=1, **cfg)

    ckpts_dir = logdir / 'ckpts'
    ckpts = sorted(
        [p for p in ckpts_dir.glob('*.tar')],
        key=lambda x: int(re.search(r'(\d+)', x.stem).group(1)))

    curve = list(existing.get('curve', []))
    seen = {entry['iter'] for entry in curve}

    for ckpt in tqdm(ckpts, desc='ckpts'):
        it = int(re.search(r'(\d+)', ckpt.stem).group(1))
        if it in seen:
            continue
        metrics, n = eval_ckpt(runner, ckpt)
        entry = {'iter': it, 'n_test': n, **metrics}
        curve.append(entry)
        curve.sort(key=lambda e: e['iter'])
        with open(out_path, 'w') as f:
            json.dump({'dataset_type': dataset_type, 'curve': curve}, f, indent=2)
        print(f'  iter={it:>6d}  Amp={metrics["Amplitude"]:.4f}  '
              f'T60={metrics["T60"]:.4f}  C50={metrics["C50"]:.4f}  '
              f'EDT={metrics["EDT"]:.4f}')
    print(f'\nSaved convergence curve ({len(curve)} ckpts) to {out_path}')


if __name__ == '__main__':
    main()
