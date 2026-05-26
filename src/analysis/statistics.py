"""音高与音程统计分析

包含 Zipf 定律分析，用于评估旋律的"音乐性"。

参考：
- Manaris et al. (2005), "Zipf's Law, Music Classification, and Aesthetics"
- Levitin et al. (2012), PNAS
"""

import numpy as np
from collections import Counter


def pitch_distribution(pitches: np.ndarray) -> dict[int, int]:
    """统计各 MIDI 音高出现频次。"""
    return dict(Counter(pitches.tolist()))


def interval_distribution(pitches: np.ndarray) -> dict[int, int]:
    """统计相邻音符之间的音程 (半音数) 分布。"""
    intervals = np.diff(pitches)
    return dict(Counter(intervals.tolist()))


def interval_direction_stats(pitches: np.ndarray) -> dict[str, float]:
    """统计音程方向：上行、下行、同音重复的比例。"""
    intervals = np.diff(pitches)
    n = len(intervals)
    if n == 0:
        return {"ascending": 0, "descending": 0, "repeated": 0}

    ascending = np.sum(intervals > 0) / n
    descending = np.sum(intervals < 0) / n
    repeated = np.sum(intervals == 0) / n

    return {
        "ascending": float(ascending),
        "descending": float(descending),
        "repeated": float(repeated),
    }


def duration_distribution(durations: list[float]) -> dict[float, int]:
    """统计各时值出现频次。"""
    return dict(Counter(durations))


def shannon_entropy(sequence: np.ndarray) -> float:
    """计算序列的 Shannon 信息熵。

    熵越高 → 旋律越不可预测/复杂
    """
    _, counts = np.unique(sequence, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def melody_contour(pitches: np.ndarray) -> np.ndarray:
    """将音高序列转为轮廓序列。

    上行 → +1, 下行 → -1, 保持 → 0
    """
    intervals = np.diff(pitches)
    return np.sign(intervals).astype(int)


def zipf_analysis(pitches: np.ndarray) -> dict:
    """Zipf 定律分析：评估音高使用频率是否服从幂律分布。

    对音高频率排序后在 log-log 坐标下拟合直线。
    悦耳音乐的 Zipf 斜率接近 -1。

    Returns:
        {
            "slope": Zipf 斜率（接近 -1 表示 Zipf 分布）,
            "r_squared": 拟合优度,
            "ranks": 排名数组,
            "frequencies": 对应频率数组,
        }

    参考：Manaris et al. (2005), Computer Music Journal, 29(1), 55-69
    """
    _, counts = np.unique(pitches, return_counts=True)
    # 按频次降序排列
    sorted_counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)

    if len(ranks) < 3:
        return {"slope": 0.0, "r_squared": 0.0, "ranks": ranks, "frequencies": sorted_counts}

    log_ranks = np.log10(ranks)
    log_freqs = np.log10(sorted_counts)

    coeffs = np.polyfit(log_ranks, log_freqs, 1)
    slope = coeffs[0]

    predicted = np.polyval(coeffs, log_ranks)
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "slope": float(slope),
        "r_squared": float(r_squared),
        "ranks": ranks.tolist(),
        "frequencies": sorted_counts.tolist(),
    }


def summary_statistics(pitches: np.ndarray) -> dict:
    """计算音高序列的摘要统计。"""
    intervals = np.diff(pitches)
    zipf = zipf_analysis(pitches)
    return {
        "num_notes": len(pitches),
        "pitch_range": int(pitches.max() - pitches.min()),
        "pitch_mean": float(pitches.mean()),
        "pitch_std": float(pitches.std()),
        "interval_mean": float(np.mean(np.abs(intervals))),
        "interval_std": float(np.std(intervals)),
        "unique_pitches": int(len(np.unique(pitches))),
        "entropy": shannon_entropy(pitches),
        "direction": interval_direction_stats(pitches),
        "zipf_slope": zipf["slope"],
        "zipf_r2": zipf["r_squared"],
    }
