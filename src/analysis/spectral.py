"""频谱与功率谱密度 (PSD) 分析

对音高序列进行 PSD 分析，拟合 1/f^β 斜率，
验证生成旋律是否保留了原始噪声的统计特征。
"""

import numpy as np
from scipy import signal


def compute_psd(pitch_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算音高序列的功率谱密度。

    Returns:
        (frequencies, psd) 数组对
    """
    freqs, psd = signal.welch(
        pitch_sequence.astype(float),
        fs=1.0,          # 采样频率 = 1（每音符）
        nperseg=min(64, len(pitch_sequence) // 2),
        noverlap=None,
    )
    # 排除 DC 分量
    mask = freqs > 0
    return freqs[mask], psd[mask]


def fit_spectral_exponent(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    """在 log-log 空间拟合 PSD 斜率，估计频谱指数 β。

    PSD ∝ 1/f^β → log(PSD) = -β·log(f) + const

    Returns:
        (beta, r_squared): 频谱指数和拟合优度
    """
    log_f = np.log10(freqs)
    log_psd = np.log10(psd)

    # 线性拟合
    coeffs = np.polyfit(log_f, log_psd, 1)
    beta = -coeffs[0]  # 斜率的负数

    # 计算 R²
    predicted = np.polyval(coeffs, log_f)
    ss_res = np.sum((log_psd - predicted) ** 2)
    ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta, r_squared


def compute_audio_spectrogram(
    audio: np.ndarray, sr: int = 44100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算音频的 STFT 频谱图。

    Returns:
        (times, frequencies, spectrogram_db) 用于可视化
    """
    import librosa
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    return times, freqs, S_db
