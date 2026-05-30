"""可视化模块

生成项目所需的全部分析图表。
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_psd_comparison(
    results: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
):
    """绘制多种噪声旋律的 PSD 对比图。

    Args:
        results: {name: (freqs, psd, beta)} 字典
        output_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"pink": "#E91E63", "brown": "#795548", "white": "#9E9E9E"}

    for name, (freqs, psd, beta) in results.items():
        color = colors.get(name, None)
        ax.loglog(freqs, psd, label=f"{name} (β={beta:.2f})", color=color, lw=2)

    ax.set_xlabel("Frequency (cycles per note)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("PSD of Pitch Sequences (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_autocorrelation_comparison(
    results: dict[str, np.ndarray],
    output_path: Path,
):
    """绘制自相关函数对比图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"pink": "#E91E63", "brown": "#795548", "white": "#9E9E9E"}

    for name, acf in results.items():
        color = colors.get(name, None)
        ax.plot(acf, label=name, color=color, lw=2)

    ax.set_xlabel("Lag (notes)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation Function of Pitch Sequences")
    ax.axhline(y=0, color="k", ls="--", lw=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pitch_distribution(
    distributions: dict[str, dict[int, int]],
    output_path: Path,
):
    """绘制音高分布直方图。"""
    fig, axes = plt.subplots(1, len(distributions), figsize=(6 * len(distributions), 5))
    if len(distributions) == 1:
        axes = [axes]

    colors = {"pink": "#E91E63", "brown": "#795548", "white": "#9E9E9E"}

    for ax, (name, dist) in zip(axes, distributions.items()):
        pitches = sorted(dist.keys())
        counts = [dist[p] for p in pitches]
        color = colors.get(name, "steelblue")
        ax.bar(pitches, counts, color=color, alpha=0.7)
        ax.set_xlabel("MIDI Pitch")
        ax.set_ylabel("Count")
        ax.set_title(f"Pitch Distribution - {name}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_interval_distribution(
    distributions: dict[str, dict[int, int]],
    output_path: Path,
):
    """绘制音程分布直方图。"""
    fig, axes = plt.subplots(1, len(distributions), figsize=(6 * len(distributions), 5))
    if len(distributions) == 1:
        axes = [axes]

    colors = {"pink": "#E91E63", "brown": "#795548", "white": "#9E9E9E"}

    for ax, (name, dist) in zip(axes, distributions.items()):
        intervals = sorted(dist.keys())
        counts = [dist[i] for i in intervals]
        color = colors.get(name, "steelblue")
        ax.bar(intervals, counts, color=color, alpha=0.7)
        ax.set_xlabel("Interval (semitones)")
        ax.set_ylabel("Count")
        ax.set_title(f"Interval Distribution - {name}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram_comparison(
    spectrograms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sr: int,
    output_path: Path,
):
    """绘制音频频谱图对比。

    Args:
        spectrograms: {name: (times, freqs, S_db)}
        sr: 采样率
        output_path: 输出路径
    """
    import librosa.display

    fig, axes = plt.subplots(1, len(spectrograms), figsize=(7 * len(spectrograms), 5))
    if len(spectrograms) == 1:
        axes = [axes]

    for ax, (name, (times, freqs, S_db)) in zip(axes, spectrograms.items()):
        img = ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Spectrogram - {name}")
        ax.set_ylim(0, 4000)
        fig.colorbar(img, ax=ax, label="dB")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zipf(
    results: dict[str, dict],
    output_path: Path,
):
    """绘制 Zipf 定律 log-log 图（rank vs frequency）。

    Args:
        results: {name: {"slope": float, "r_squared": float, "ranks": list, "frequencies": list}}
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    colors = {"pink": "#E91E63", "brown": "#795548", "white": "#9E9E9E"}

    for ax, (name, data) in zip(axes, results.items()):
        ranks = np.array(data["ranks"])
        freqs = np.array(data["frequencies"])
        slope = data["slope"]
        r2 = data["r_squared"]
        color = colors.get(name, "steelblue")

        ax.scatter(ranks, freqs, color=color, alpha=0.7, s=30)

        # 拟合线
        if len(ranks) >= 3:
            log_r = np.log10(ranks)
            fit_freqs = 10 ** (slope * log_r + np.log10(freqs[0]))
            ax.plot(ranks, fit_freqs, "k--", lw=1.5,
                    label=f"slope={slope:.2f}, R²={r2:.2f}")

        # 参考 Zipf 斜率 -1
        ref_freqs = freqs[0] / ranks
        ax.plot(ranks, ref_freqs, color="gray", ls=":", lw=1, alpha=0.5,
                label="Zipf (slope=-1)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Zipf Analysis — {name}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dfa_scaling(
    results: dict[str, tuple[list, list, float, float]],
    output_path: Path,
):
    """绘制 DFA 标度关系图: log(n) vs log(F(n))。

    Args:
        results: {name: (scales, fluctuations, alpha, r2)}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"pink": "#E91E63", "brown": "#795548", "white": "#9E9E9E"}

    for name, (scales, fluct, alpha, r2) in results.items():
        color = colors.get(name, "steelblue")
        log_s = np.log(scales)
        log_f = np.log(fluct)

        ax.scatter(log_s, log_f, color=color, alpha=0.7, s=40, zorder=3)

        # 拟合线
        coeffs = np.polyfit(log_s, log_f, 1)
        fit_line = np.polyval(coeffs, log_s)
        ax.plot(log_s, fit_line, color=color, lw=2,
                label=f"{name}: α={alpha:.2f} (R²={r2:.2f})")

    ax.set_xlabel("ln(window size n)")
    ax.set_ylabel("ln(F(n))")
    ax.set_title("DFA Scaling: Detrended Fluctuation Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_interval_direction(
    results: dict[str, dict[str, float]],
    output_path: Path,
):
    """绘制音程方向堆叠条形图。

    Args:
        results: {name: {"ascending": float, "descending": float, "repeated": float}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_dir = {"ascending": "#4CAF50", "descending": "#F44336", "repeated": "#FFC107"}
    labels_cn = {"ascending": "上行", "descending": "下行", "repeated": "同音重复"}

    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.5

    bottom = np.zeros(len(names))
    for direction in ["ascending", "descending", "repeated"]:
        vals = [results[n][direction] for n in names]
        ax.bar(x, vals, width, bottom=bottom, color=colors_dir[direction],
               label=labels_cn[direction], alpha=0.8)
        # 百分比标注
        for i, v in enumerate(vals):
            if v > 0.05:
                ax.text(x[i], bottom[i] + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=10, fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set_ylabel("Proportion")
    ax.set_title("Interval Direction Distribution")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_sequence(
    sequences: dict[str, np.ndarray],
    output_path: Path,
):
    """绘制原始噪声序列波形。"""
    fig, axes = plt.subplots(len(sequences), 1, figsize=(12, 3 * len(sequences)))
    if len(sequences) == 1:
        axes = [axes]

    colors = {"pink": "#E91E63", "brown": "#795548", "white": "#9E9E9E"}

    for ax, (name, seq) in zip(axes, sequences.items()):
        color = colors.get(name, "steelblue")
        ax.plot(seq, color=color, lw=0.8)
        ax.set_title(f"Noise Sequence - {name}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Normalized value")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
