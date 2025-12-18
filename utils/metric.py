import numpy as np
from scipy import stats
from scipy.signal import hilbert
import scipy
import auraloss
import torch

def metric_cal(ori_ir, pred_ir, fs=48000, window=32):
    """calculate the evaluation metric

    Parameters
    ----------
    ori_ir : np.array
        ground truth impulse response
    pred_ir : np.array
        predicted impulse response
    fs : int
        sampling rate, by default 48000

    Returns
    -------
    evaluation metrics
    """
    
    if ori_ir.ndim == 1:
        ori_ir = ori_ir[np.newaxis, :]
    if pred_ir.ndim == 1:
        pred_ir = pred_ir[np.newaxis, :]

    # prevent numerical issue for log calculation
    multi_stft = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[512, 256, 128], win_lengths=[300, 150, 75], hop_sizes=[60, 30, 8])
    multi_stft_loss = multi_stft(torch.tensor(ori_ir).unsqueeze(1), torch.tensor(pred_ir).unsqueeze(1))

    fft_ori = np.fft.rfft(ori_ir, axis=-1)
    fft_predict = np.fft.rfft(pred_ir, axis=-1)

    angle_error = np.mean(np.abs(np.cos(np.angle(fft_ori)) - np.cos(np.angle(fft_predict)))) + np.mean(np.abs(np.sin(np.angle(fft_ori)) - np.sin(np.angle(fft_predict))))
    amp_ori = scipy.ndimage.convolve1d(np.abs(fft_ori), np.ones(window))
    amp_predict = scipy.ndimage.convolve1d(np.abs(fft_predict), np.ones(window))
    amp_error = np.mean(np.abs(amp_ori - amp_predict) / amp_ori)

    # calculate the envelop error
    ori_env = np.abs(hilbert(ori_ir))
    pred_env = np.abs(hilbert(pred_ir))
    env_error = np.mean(np.abs(ori_env - pred_env) / np.max(ori_env, axis=1, keepdims=True))

    # derevie the energy trend
    ori_energy = 10.0 * np.log10(np.cumsum(ori_ir[:,::-1]**2 + 1e-9, axis=-1)[:,::-1])
    pred_energy = 10.0 * np.log10(np.cumsum(pred_ir[:,::-1]**2 + 1e-9, axis=-1)[:,::-1])

    ori_energy -= ori_energy[:, 0].reshape(-1, 1)
    pred_energy -= pred_energy[:, 0].reshape(-1, 1)
    
    # calculate the t60 percentage error and EDT time error
    ori_t60, ori_edt = t60_EDT_cal(ori_energy, fs=fs)
    pred_t60, pred_edt = t60_EDT_cal(pred_energy, fs=fs)
    t60_error = np.mean(np.abs(ori_t60 - pred_t60) / ori_t60)
    edt_error = np.mean(np.abs(ori_edt - pred_edt))

    # calculate the C50 error
    base_sample = 0
    samples_50ms = int(0.05 * fs) + base_sample  # Number of samples in 50 ms
    # Compute the energy in the first 50ms and from 50ms to the end
    energy_ori_early = np.sum(ori_ir[:,base_sample:samples_50ms]**2, axis=-1)
    energy_ori_late = np.sum(ori_ir[:,samples_50ms:]**2, axis=-1)
    energy_pred_early = np.sum(pred_ir[:,base_sample:samples_50ms]**2, axis=-1)
    energy_pred_late = np.sum(pred_ir[:,samples_50ms:]**2, axis=-1)

    # Calculate C50 for the original and predicted impulse response
    C50_ori = 10.0 * np.log10(energy_ori_early / energy_ori_late)
    C50_pred = 10.0 * np.log10(energy_pred_early / energy_pred_late)
    C50_error = np.mean(np.abs(C50_ori - C50_pred))

    return angle_error, amp_error, env_error, t60_error, edt_error, C50_error, multi_stft_loss, ori_energy, pred_energy


def t60_EDT_cal(energys, init_db=-5, end_db=-25, factor=3.0, fs=48000):
    """calculate the T60 and EDT metric of the given impulse response normalized energy trend
    t60: find the time it takes to decay from -5db to -65db.
        A usual way to do this is to calculate the time it takes from -5 to -25db, and multiply by 3.0
    
    EDT: Early decay time, time it takes to decay from 0db to -10db, and multiply the number by 6

    Parameters
    ----------
    energys : np.array
        normalized energy
    init_db : int, optional
        t60 start db, by default -5
    end_db : int, optional
        t60 end db, by default -25
    factor : float, optional
        t60 multiply factor, by default 3.0
    fs : int, optional
        sampling rate, by default 48000

    Returns
    -------
    t60 : float
    edt : float, seconds
    """

    t60_all = []
    edt_all = []

    for energy in energys:
        # find the -10db point
        edt_factor = 6.0
        energy_n10db = energy[np.abs(energy - (-10)).argmin()]

        n10db_sample = np.where(energy == energy_n10db)[0][0]
        edt = n10db_sample / fs * edt_factor

        # find the intersection of -5db and -25db position
        energy_init = energy[np.abs(energy - init_db).argmin()]
        energy_end = energy[np.abs(energy - end_db).argmin()]
        init_sample = np.where(energy == energy_init)[0][0]
        end_sample = np.where(energy == energy_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = energy[init_sample:end_sample + 1]
        
        # regress to find the db decay trend
        slope, intercept = stats.linregress(x, y)[0:2]
        db_regress_init = (init_db - intercept) / slope
        db_regress_end = (end_db - intercept) / slope

        # get t60 value
        t60 = factor * (db_regress_end - db_regress_init)
        
        t60_all.append(t60)
        edt_all.append(edt)

    t60_all = np.array(t60_all)
    edt_all = np.array(edt_all)

    return t60_all, edt_all



import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import hilbert
from auraloss.freq import MultiResolutionSTFTLoss

def metric_cal_torch(ori_ir_np, pred_ir_np, fs=44100, window=32, device="cuda"):

    if ori_ir_np.ndim == 1:
        ori_ir_np = ori_ir_np[np.newaxis, :]
    if pred_ir_np.ndim == 1:
        pred_ir_np = pred_ir_np[np.newaxis, :]
    
    # Convert to torch tensors on GPU (no gradients)
    ori_ir = torch.from_numpy(ori_ir_np).float().to(device)
    pred_ir = torch.from_numpy(pred_ir_np).float().to(device)

    with torch.no_grad():
        # ---- Multi-Resolution STFT Loss ----
        multi_stft = MultiResolutionSTFTLoss(w_lin_mag=1,
                                             fft_sizes=[512, 256, 128],
                                             win_lengths=[300, 150, 75],
                                             hop_sizes=[60, 30, 8]).to(device)
        multi_stft_loss = multi_stft(ori_ir.unsqueeze(1), pred_ir.unsqueeze(1))

        # ---- FFT-based metrics ----
        fft_ori = torch.fft.rfft(ori_ir, dim=-1)
        fft_pred = torch.fft.rfft(pred_ir, dim=-1)

        angle_ori = torch.angle(fft_ori)
        angle_pred = torch.angle(fft_pred)
        angle_error = torch.mean(torch.abs(torch.cos(angle_ori) - torch.cos(angle_pred)) +
                                 torch.abs(torch.sin(angle_ori) - torch.sin(angle_pred))).item()

        # Smoothed amplitude error
        amp_ori = torch.abs(fft_ori)
        amp_pred = torch.abs(fft_pred)
        kernel = torch.ones(1, 1, window, device=device) / window
        amp_ori_smoothed = F.conv1d(amp_ori.unsqueeze(1), kernel, padding=window // 2).squeeze(1)
        amp_pred_smoothed = F.conv1d(amp_pred.unsqueeze(1), kernel, padding=window // 2).squeeze(1)
        amp_error = torch.mean(torch.abs(amp_ori_smoothed - amp_pred_smoothed) / (amp_ori_smoothed + 1e-6)).item()

        # ---- Envelope error (Hilbert still uses CPU) ----
        ori_env = np.abs(hilbert(ori_ir_np, axis=-1))
        pred_env = np.abs(hilbert(pred_ir_np, axis=-1))
        env_error = np.mean(np.abs(ori_env - pred_env) / (np.max(ori_env, axis=1, keepdims=True) + 1e-6))

        # ---- Energy decay (reverse cumulative sum) ----
        def compute_energy_trend(x):
            x_rev = torch.flip(x ** 2, dims=[-1])
            cumsum = torch.cumsum(x_rev + 1e-13, dim=-1)
            log_energy = 10.0 * torch.log10(torch.flip(cumsum, dims=[-1]))
            log_energy = log_energy - log_energy[:, :1].clone()
            return log_energy
        
        ori_energy = compute_energy_trend(ori_ir)
        pred_energy = compute_energy_trend(pred_ir)

        # ---- T60 & EDT (compute on CPU for now) ----
        # ori_energy_np = ori_energy.cpu().numpy()
        # pred_energy_np = pred_energy.cpu().numpy()
        ori_t60, ori_edt = t60_EDT_cal_torch(ori_energy, fs)
        pred_t60, pred_edt = t60_EDT_cal_torch(pred_energy, fs)
        t60_error = torch.mean(torch.abs(ori_t60 - pred_t60) / (ori_t60 + 1e-6))
        edt_error = torch.mean(torch.abs(ori_edt - pred_edt))

        # ---- C50 error ----
        base_sample = 0
        samples_50ms = int(0.05 * fs) + base_sample
        early_ori = torch.sum(ori_ir[:, base_sample:samples_50ms] ** 2, dim=-1)
        late_ori = torch.sum(ori_ir[:, samples_50ms:] ** 2, dim=-1)
        early_pred = torch.sum(pred_ir[:, base_sample:samples_50ms] ** 2, dim=-1)
        late_pred = torch.sum(pred_ir[:, samples_50ms:] ** 2, dim=-1)
        C50_ori = 10.0 * torch.log10(early_ori / (late_ori + 1e-13))
        C50_pred = 10.0 * torch.log10(early_pred / (late_pred + 1e-13))
        C50_error = torch.mean(torch.abs(C50_ori - C50_pred)).item()

    return (angle_error, amp_error, env_error, t60_error, edt_error,
            C50_error, multi_stft_loss.cpu(), ori_energy.cpu().numpy(), pred_energy.cpu().numpy())


def t60_EDT_cal_torch(energy_db: torch.Tensor, fs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-friendly version of T60 and EDT calculation in PyTorch.
    
    Args:
        energy_db (torch.Tensor): [B, T] log-energy decay curve (in dB), already normalized.
        fs (int): sampling rate.
        
    Returns:
        t60 (torch.Tensor): [B] RT60 per sample (in seconds).
        edt (torch.Tensor): [B] EDT per sample (in seconds).
    """
    B, T = energy_db.shape
    device = energy_db.device
    time = torch.arange(T, device=device).float() / fs  # [T]
    time = time.expand(B, -1)  # [B, T]

    def fit_decay(decay_curve, target_db_start, target_db_end):
        """
        Fit decay between target_db_start and target_db_end, return slope.
        """
        # Create a mask for each sample where the dB is within the range
        mask = (decay_curve <= target_db_start) & (decay_curve >= target_db_end)  # [B, T]
        eps = 1e-6

        # Count valid points per sample
        valid_counts = mask.sum(dim=1).clamp(min=1)  # [B]
        mask_f = mask.float()

        # Linear regression: fit line to decay vs. time using least squares
        x = time * mask_f  # [B, T]
        y = decay_curve * mask_f  # [B, T]

        mean_x = x.sum(dim=1) / valid_counts  # [B]
        mean_y = y.sum(dim=1) / valid_counts  # [B]

        x_centered = x - mean_x.unsqueeze(1)
        y_centered = y - mean_y.unsqueeze(1)

        slope = (x_centered * y_centered * mask_f).sum(dim=1) / (
            (x_centered**2 * mask_f).sum(dim=1) + eps
        )  # [B]

        return slope

    with torch.no_grad():
        # T60: fit between 0 and -60 dB
        slope_t60 = fit_decay(energy_db, target_db_start=-5, target_db_end=-25.0)
        t60 = -60.0 / (slope_t60 + 1e-6)

        # EDT: fit between 0 and -10 dB
        slope_edt = fit_decay(energy_db, target_db_start=0.0, target_db_end=-10.0)
        edt = -60.0 / (slope_edt + 1e-6)

    return t60, edt