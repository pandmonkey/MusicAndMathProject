"""噪声序列生成

使用 colorednoise 库生成 1/f^β 噪声序列。
- β=0: 白噪声
- β=1: 粉色噪声 (1/f)
- β=2: 棕色噪声 (1/f²)
"""

import numpy as np
import colorednoise as cn

from src.config import NoiseConfig


def generate_noise(config: NoiseConfig) -> np.ndarray:
    """生成噪声序列。

    Args:
        config: 噪声配置参数

    Returns:
        归一化到 [0, 1] 范围的噪声序列
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    # 生成 1/f^β 噪声
    raw = cn.powerlaw_psd_gaussian(config.beta, config.num_samples)

    # 归一化到 [0, 1]
    normalized = (raw - raw.min()) / (raw.max() - raw.min())
    return normalized


def generate_noise_pair(
    num_samples: int = 256, seed: int | None = 42
) -> dict[str, np.ndarray]:
    """同时生成粉色噪声和棕色噪声序列（使用相同随机种子基准）。

    Returns:
        {"pink": array, "brown": array} 归一化噪声序列
    """
    result = {}
    for name, beta in [("pink", 1.0), ("brown", 2.0)]:
        cfg = NoiseConfig(
            beta=beta,
            num_samples=num_samples,
            seed=seed + hash(name) % 1000 if seed is not None else None,
        )
        result[name] = generate_noise(cfg)
    return result
