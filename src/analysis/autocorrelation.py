"""自相关分析与长程相关性

计算音高序列的自相关函数 (ACF)，估计 Hurst 指数和分形维度。

提供两种 Hurst 指数估计方法：
- R/S 分析法：经典方法，但对非平稳数据会系统性高估 H
- DFA（去趋势波动分析）：推荐方法，对非平稳数据更稳健

数学关系（文献依据）：
- DFA 标度指数 α = (β + 1) / 2
- fGn（α < 1）: α = H
- fBm（α > 1）: α = H + 1
- 分形维度 D = 2 - H

参考：
- Peng et al. (1994), Phys. Rev. E
- Jennings et al. (2004), Physica A
- Su & Wu (2007), Physica A
"""

import numpy as np


def compute_autocorrelation(
    sequence: np.ndarray, max_lag: int | None = None
) -> np.ndarray:
    """计算序列的归一化自相关函数。

    Args:
        sequence: 输入序列
        max_lag: 最大滞后量，默认为序列长度的一半

    Returns:
        自相关系数数组 (lag 0 到 max_lag)
    """
    seq = sequence.astype(float)
    seq = seq - np.mean(seq)
    n = len(seq)

    if max_lag is None:
        max_lag = n // 2

    var = np.var(seq)
    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.correlate(seq, seq, mode="full")
    acf = acf[n - 1:]  # 只取正滞后
    acf = acf[:max_lag + 1] / (var * n)

    return acf


def estimate_dfa(sequence: np.ndarray, order: int = 1) -> tuple[float, float]:
    """DFA（去趋势波动分析）估计标度指数 α。

    推荐方法：对非平稳数据比 R/S 更稳健，对短序列精度更高。

    算法步骤 (Peng et al., 1994):
    1. 构建积分序列（累积和）
    2. 分段，每段内拟合多项式趋势并去除
    3. 计算去趋势后的均方根波动 F(n)
    4. log(F(n)) vs log(n) 的斜率即为 α

    Args:
        sequence: 输入序列
        order: 去趋势多项式阶数（1=线性DFA-1，2=二次DFA-2）

    Returns:
        (alpha, r_squared): DFA 标度指数和拟合优度

    参考：
    - α ≈ 0.5: 无相关（白噪声）
    - α ≈ 1.0: 1/f 噪声
    - α ≈ 1.5: 棕色噪声/随机游走
    """
    seq = sequence.astype(float)
    n = len(seq)

    # 步骤 1: 构建积分序列
    profile = np.cumsum(seq - np.mean(seq))

    # 步骤 2-5: 多尺度波动分析
    # 窗口大小从 4 到 N/4
    min_window = max(order + 2, 4)
    max_window = n // 4

    scales = []
    fluctuations = []

    window = min_window
    while window <= max_window:
        n_segments = n // window
        if n_segments < 1:
            break

        local_fluctuations = []

        # 从两端分段（充分利用数据）
        for direction in [0, 1]:
            for seg in range(n_segments):
                if direction == 0:
                    start = seg * window
                else:
                    start = n - (seg + 1) * window
                end = start + window

                segment = profile[start:end]
                # 局部多项式去趋势
                x = np.arange(window)
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)
                # 计算去趋势后的方差
                local_fluctuations.append(np.mean((segment - trend) ** 2))

        if local_fluctuations:
            # F(n) = 均方根波动
            f_n = np.sqrt(np.mean(local_fluctuations))
            scales.append(window)
            fluctuations.append(f_n)

        # 对数等间距增加窗口大小
        window = max(window + 1, int(window * 1.25))

    if len(scales) < 3:
        return 0.5, 0.0  # 数据不足

    # 步骤 6: log-log 线性拟合
    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)

    coeffs = np.polyfit(log_scales, log_fluct, 1)
    alpha = coeffs[0]

    # R²
    predicted = np.polyval(coeffs, log_scales)
    ss_res = np.sum((log_fluct - predicted) ** 2)
    ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(alpha), float(r_squared)


def estimate_hurst_rs(sequence: np.ndarray) -> float:
    """使用 R/S 分析法估计 Hurst 指数。

    注意：此方法对非平稳数据系统性高估 H。
    推荐使用 estimate_dfa() 作为主方法。

    参考：Hurst (1951), Mandelbrot & Wallis (1969)
    """
    seq = sequence.astype(float)
    n = len(seq)

    min_window = 8
    scales = []
    rs_values = []

    window = min_window
    while window <= n // 2:
        rs_list = []
        for start in range(0, n - window + 1, window):
            segment = seq[start:start + window]
            mean = np.mean(segment)
            deviations = segment - mean
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(segment, ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            scales.append(window)
            rs_values.append(np.mean(rs_list))

        window *= 2

    if len(scales) < 2:
        return 0.5

    log_scales = np.log(scales)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_scales, log_rs, 1)

    return float(coeffs[0])


def compute_fractal_dimension(hurst_exponent: float) -> float:
    """从 Hurst 指数计算分形维度。

    D = 2 - H

    D ≈ 1.0: 平滑旋律（棕色噪声）
    D ≈ 1.2: "悦耳"区间（粉色噪声）
    D ≈ 1.5: 完全随机（白噪声）

    参考：Niklasson & Niklasson (2020), arXiv:2004.02612
    """
    return 2.0 - hurst_exponent


def compute_dfa_raw(sequence: np.ndarray, order: int = 1) -> dict:
    """返回 DFA 分析的完整数据（用于可视化）。"""
    seq = sequence.astype(float)
    n = len(seq)
    profile = np.cumsum(seq - np.mean(seq))

    min_window = max(order + 2, 4)
    max_window = n // 4

    scales = []
    fluctuations = []

    window = min_window
    while window <= max_window:
        n_segments = n // window
        if n_segments < 1:
            break

        local_fluctuations = []
        for direction in [0, 1]:
            for seg in range(n_segments):
                if direction == 0:
                    start = seg * window
                else:
                    start = n - (seg + 1) * window
                end = start + window
                segment = profile[start:end]
                x = np.arange(window)
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)
                local_fluctuations.append(np.mean((segment - trend) ** 2))

        if local_fluctuations:
            f_n = np.sqrt(np.mean(local_fluctuations))
            scales.append(window)
            fluctuations.append(f_n)

        window = max(window + 1, int(window * 1.25))

    if len(scales) < 3:
        return {"scales": [], "fluctuations": [], "alpha": 0.5, "r_squared": 0.0}

    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)
    coeffs = np.polyfit(log_scales, log_fluct, 1)
    alpha = coeffs[0]

    predicted = np.polyval(coeffs, log_scales)
    ss_res = np.sum((log_fluct - predicted) ** 2)
    ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "scales": scales,
        "fluctuations": fluctuations,
        "alpha": float(alpha),
        "r_squared": float(r_squared),
    }


def dfa_to_beta(alpha: float) -> float:
    """从 DFA 标度指数转换为频谱指数 β。

    β = 2α - 1

    参考：Peng et al. (1995)
    """
    return 2.0 * alpha - 1.0


def dfa_to_hurst(alpha: float) -> tuple[float, str]:
    """从 DFA 标度指数推导 Hurst 指数。

    Returns:
        (H, signal_type):
        - α < 1 → fGn 类型, H = α
        - α > 1 → fBm 类型, H = α - 1
    """
    if alpha < 1.0:
        return alpha, "fGn"
    else:
        return alpha - 1.0, "fBm"
