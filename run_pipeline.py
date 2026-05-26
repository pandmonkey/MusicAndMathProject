#!/usr/bin/env python3
"""
噪声音乐·随机旋律 Pipeline 入口

用法:
    # 激活虚拟环境
    source venv/bin/activate

    # 运行默认配置（粉色噪声 + 棕色噪声对比）
    python run_pipeline.py

    # 指定配置文件
    python run_pipeline.py --config configs/experiment_01.json

    # 只生成特定噪声类型
    python run_pipeline.py --noise-type pink

    # 批量生成多组
    python run_pipeline.py --batch 10 --seed 42
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig, NoiseConfig, MelodyConfig
from src.pipeline import run_comparison, run_single


def main():
    parser = argparse.ArgumentParser(description="噪声音乐旋律生成 Pipeline")
    parser.add_argument("--config", type=str, help="配置文件路径 (.json)")
    parser.add_argument("--noise-type", type=str, choices=["pink", "brown", "white"],
                        help="只生成指定噪声类型")
    parser.add_argument("--batch", type=int, default=1, help="批量生成组数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子基准")
    parser.add_argument("--bpm", type=int, default=None, help="覆盖 BPM")
    parser.add_argument("--scale", type=str, default=None, help="覆盖音阶")
    parser.add_argument("--measures", type=int, default=None, help="覆盖最少小节数")
    args = parser.parse_args()

    # 加载或创建配置
    if args.config:
        config = PipelineConfig.load(Path(args.config))
    else:
        config = PipelineConfig()

    config.noise.seed = args.seed
    if args.bpm:
        config.melody.bpm = args.bpm
    if args.scale:
        config.melody.scale = args.scale
    if args.measures:
        config.melody.min_measures = args.measures

    print("=" * 60)
    print("噪声音乐·随机旋律 Pipeline")
    print("=" * 60)

    noise_types = [args.noise_type] if args.noise_type else ["pink", "brown"]

    for i in range(args.batch):
        config.noise.seed = args.seed + i
        config.name = f"run_{i:03d}_seed{config.noise.seed}"

        print(f"\n--- Batch {i + 1}/{args.batch}: {config.name} ---")
        print(f"  噪声类型: {noise_types}")
        print(f"  BPM: {config.melody.bpm}, 音阶: {config.melody.scale}")
        print(f"  音域: octave {config.melody.octave_low}-{config.melody.octave_high}")
        print(f"  最少小节: {config.melody.min_measures}")

        if len(noise_types) > 1:
            run_comparison(config, noise_types)
        else:
            run_single(config, noise_types[0])

    print("\n" + "=" * 60)
    print("Pipeline 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
