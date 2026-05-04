"""Run AVR inference and render visualizations in the acousticGS repo's style.
Produces the same 6-panel signal_iter_<N>.png that this codebase uses.
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

AVR_REPO = '/home/jooeun/AVR'
sys.path.insert(0, AVR_REPO)
from infer import InferRunner


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def visualize_signal(mode_set, pred_freq, gt_freq, pred_time, gt_time, save_path):
    """6-panel signal visualization. Vendored from acousticGS/utils/visualize.py."""
    pred_freq = _to_numpy(pred_freq)
    gt_freq = _to_numpy(gt_freq)
    pred_time = _to_numpy(pred_time)
    gt_time = _to_numpy(gt_time)

    fig = plt.figure(figsize=(16, 12))
    plt.suptitle(f"Signal Visualization ({mode_set} set)", fontsize=16)

    plt.subplot(231)
    plt.title("Time Domain")
    plt.plot(pred_time, label='Predicted', linewidth=0.5)
    plt.plot(gt_time, alpha=0.7, label='Ground Truth', linewidth=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()

    plt.subplot(232)
    plt.title("Phase")
    plt.plot(np.angle(pred_freq), linewidth=0.5)
    plt.plot(np.angle(gt_freq), alpha=0.7, linewidth=0.5)

    plt.subplot(233)
    plt.axis('off')
    plt.legend(handles, labels, loc='lower left', fontsize=15)

    plt.subplot(234)
    plt.title("Magnitude")
    plt.plot(np.abs(pred_freq), linewidth=0.5)
    plt.plot(np.abs(gt_freq), alpha=0.7, linewidth=0.5)
    plt.ylim(0)

    plt.subplot(235)
    plt.title("Real")
    plt.plot(np.real(pred_freq), linewidth=0.5)
    plt.plot(np.real(gt_freq), alpha=0.7, linewidth=0.5)

    plt.subplot(236)
    plt.title("Imaginary")
    plt.plot(np.imag(pred_freq), linewidth=0.5)
    plt.plot(np.imag(gt_freq), alpha=0.7, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def render(ckpt_path, n_samples=5, out_subdir='inference_acgs_style'):
    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.parent.parent / 'avr_conf.yml'
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    expname = cfg['path']['expname']
    dataset_type = cfg['path']['dataset_type']
    log_dir = ckpt_path.parent.parent
    out_dir = log_dir / out_subdir
    out_dir.mkdir(exist_ok=True)

    # Map AVR's dataset_type back to a base_folder. For MeshRIR the trained
    # checkpoint was on `Meshrir_sliced_0.1s` -> `~/data/dataset/MeshRIR/S1-M3969_npy`.
    dataset_dir = {
        'MeshRIR': os.path.expanduser('~/data/dataset/MeshRIR/S1-M3969'),
    }.get(dataset_type)
    if dataset_dir is None:
        raise ValueError(f"Unsupported dataset_type {dataset_type!r}; add a path mapping")

    print(f"Loading AVR ckpt {ckpt_path}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Output dir: {out_dir}")

    runner = InferRunner(mode='train', dataset_dir=dataset_dir, batchsize=1, **cfg)
    runner.load_specific_checkpoint(str(ckpt_path))
    runner.renderer.eval()

    iter_label = ckpt_path.stem  # e.g. "500000"

    saved = 0
    metrics_accum = {'Angle': 0, 'Amplitude': 0, 'Envelope': 0,
                     'T60': 0, 'C50': 0, 'EDT': 0, 'multi_stft': 0}
    n_total = 0

    for batch in tqdm(runner.test_iter, desc='AVR infer (acGS style)'):
        with torch.no_grad():
            ori_sig, position_rx, position_tx = batch
            # MeshRIR's pos_src.npy is shape (1,3) so DataLoader yields (B,1,3);
            # the renderer expects (B,3). Squeeze any singleton mid-dim.
            if position_tx.dim() == 3 and position_tx.shape[1] == 1:
                position_tx = position_tx.squeeze(1)
            pred_sig = runner.renderer(position_rx.cuda(), position_tx.cuda())
            pred_sig = pred_sig[..., 0] + 1j * pred_sig[..., 1]
            ori_sig = ori_sig.cuda().to(pred_sig.dtype)

            _, metrics, ori_time, pred_time = runner.calculate_metrics(
                pred_sig, ori_sig, runner.fs)

        for k in metrics_accum:
            metrics_accum[k] += float(metrics[k])
        n_total += 1

        if saved < n_samples:
            save_path = out_dir / f'signal_iter_{iter_label}_{saved:03d}.png'
            visualize_signal(
                mode_set='test',
                pred_freq=pred_sig[0].detach().cpu(),
                gt_freq=ori_sig[0].detach().cpu(),
                pred_time=pred_time[0].detach().cpu(),
                gt_time=ori_time[0].detach().cpu(),
                save_path=str(save_path),
            )
            # Cropped early-window time signal (mirrors signal_gauss left panel).
            crop = min(1200, pred_time.shape[-1])
            ms = np.arange(crop) / runner.fs * 1000.0
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.set_title(f'Time Domain (first {crop} samples = {crop / runner.fs * 1000:.1f} ms)')
            ax.plot(ms, pred_time[0, :crop].detach().cpu(),
                    label='Predicted', linewidth=0.5)
            ax.plot(ms, ori_time[0, :crop].detach().cpu(), alpha=0.7,
                    label='Ground Truth', linewidth=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Pressure')
            ax.legend(loc='upper right', fontsize=9)
            fig.tight_layout()
            crop_path = out_dir / f'signal_crop_iter_{iter_label}_{saved:03d}.png'
            fig.savefig(crop_path, dpi=200)
            plt.close(fig)
            saved += 1

    avg = {k: v / n_total for k, v in metrics_accum.items()}
    print(f'\nAVR re-eval over {n_total} test samples:')
    for k, v in avg.items():
        print(f'  {k}: {v:.6f}')
    with open(out_dir / 'metrics.txt', 'w') as f:
        for k, v in avg.items():
            f.write(f'{k}: {v}\n')
    print(f'\nSaved {saved} signal_iter + {saved} signal_crop PNGs in {out_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--n_samples', type=int, default=5)
    args = p.parse_args()
    render(args.ckpt, n_samples=args.n_samples)
