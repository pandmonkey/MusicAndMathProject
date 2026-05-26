"""噪声→音高映射

将 [0,1] 范围的噪声值量化到指定音阶的离散 MIDI 音高上。

文献依据：
- 线性映射存在"中音区聚集"问题（噪声近似高斯分布）
- 分位数映射可确保各音高使用频率均匀，推荐作为默认方法
- Voss & Clarke 原始方法为位累加法，不依赖显式映射
参考：McDonough & Herczynski (2023), Manaris et al. (2005)
"""

import numpy as np
from scipy.stats import rankdata

from src.config import MelodyConfig, SCALES


def build_pitch_pool(config: MelodyConfig) -> np.ndarray:
    """构建目标音阶的可用 MIDI 音高池。

    例如 C_major + octave 4-5 → [60,62,64,65,67,69,71, 72,74,76,77,79,81,83]
    """
    scale_pcs = SCALES[config.scale]
    pitches = []
    for octave in range(config.octave_low, config.octave_high + 1):
        base = (octave + 1) * 12  # MIDI: C4 = 60
        for pc in scale_pcs:
            p = base + pc
            if 0 <= p <= 127:
                pitches.append(p)
    return np.array(sorted(pitches))


def map_to_pitches(
    noise: np.ndarray,
    config: MelodyConfig,
    method: str = "quantile",
) -> np.ndarray:
    """将归一化噪声序列映射到 MIDI 音高。

    Args:
        noise: [0,1] 范围的噪声序列
        config: 旋律配置
        method: 映射方法
            - "quantile": 分位数映射（推荐，音高分布均匀）
            - "linear": 线性映射（简单但中音区聚集）

    Returns:
        MIDI 音高序列 (int array)
    """
    pool = build_pitch_pool(config)

    if method == "quantile":
        return _quantile_mapping(noise, pool)
    elif method == "linear":
        return _linear_mapping(noise, pool)
    else:
        raise ValueError(f"未知映射方法: {method}")


def _quantile_mapping(noise: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """分位数映射：利用经验 CDF 将噪声均匀映射到音高池。

    原理：对噪声值排名后均匀分配到音高池各区间，
    确保无论原始噪声分布如何（高斯、偏态），
    每个音高的使用频率大致相等。

    这避免了线性映射中由高斯分布导致的"中音区过度集中"问题。

    参考：McDonough & Herczynski (2023), Chaos, Solitons & Fractals
    """
    n = len(noise)
    k = len(pool)

    # 计算经验 CDF（通过排名）
    ranks = rankdata(noise, method="ordinal")  # 1-based ranks
    # 将 rank 映射到 [0, 1) 范围的分位数
    quantiles = (ranks - 0.5) / n
    # 将分位数均匀映射到音高池索引
    indices = np.clip((quantiles * k).astype(int), 0, k - 1)

    return pool[indices]


def _linear_mapping(noise: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """线性映射：noise 值线性缩放到音高池索引。

    注意：此方法因噪声的高斯分布特性导致中间音高过度使用。
    仅作为对比基线保留。
    """
    indices = np.clip(
        (noise * len(pool)).astype(int), 0, len(pool) - 1
    )
    return pool[indices]
