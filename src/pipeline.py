"""Pipeline 主流程编排

将噪声生成→音高映射→MIDI输出→音频渲染→统计分析串联运行。
"""

import json
from pathlib import Path
import numpy as np

from src.config import PipelineConfig, NoiseConfig, PROJECT_ROOT
from src.noise_generation.generator import generate_noise
from src.noise_generation.pitch_mapping import map_to_pitches
from src.noise_generation.rhythm import build_note_sequence
from src.midi_generation.midi_writer import write_midi
from src.audio_rendering.renderer import render_with_fluidsynth, render_with_pretty_midi
from src.analysis.spectral import compute_psd, fit_spectral_exponent
from src.analysis.autocorrelation import (
    compute_autocorrelation,
    estimate_dfa,
    estimate_hurst_rs,
    compute_fractal_dimension,
    compute_dfa_raw,
    dfa_to_hurst,
)
from src.analysis.statistics import (
    pitch_distribution,
    interval_distribution,
    interval_direction_stats,
    zipf_analysis,
    summary_statistics,
)
from src.analysis.visualization import (
    plot_psd_comparison,
    plot_autocorrelation_comparison,
    plot_pitch_distribution,
    plot_interval_distribution,
    plot_noise_sequence,
    plot_zipf,
    plot_dfa_scaling,
    plot_interval_direction,
)

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = DATA_DIR / "analysis" / "figures"


def run_single(config: PipelineConfig, noise_type: str) -> dict:
    """运行单次 pipeline：生成一条噪声旋律。

    Args:
        config: pipeline 配置
        noise_type: "pink" / "brown" / "white"

    Returns:
        包含所有中间结果的字典
    """
    beta_map = {"pink": 1.0, "brown": 2.0, "white": 0.0}
    noise_cfg = NoiseConfig(
        beta=beta_map.get(noise_type, config.noise.beta),
        num_samples=config.noise.num_samples,
        seed=config.noise.seed,
    )

    # 1. 生成噪声
    noise = generate_noise(noise_cfg)

    # 2. 音高映射
    pitches = map_to_pitches(noise, config.melody)

    # 3. 构建音符序列（使用噪声序列的后半部分驱动节奏）
    mid = len(noise) // 2
    duration_noise = noise[mid:] if len(noise) > mid else None
    notes = build_note_sequence(pitches, config.melody, duration_noise)

    # 4. 写入 MIDI
    midi_dir = DATA_DIR / "midi" / noise_type
    midi_path = midi_dir / f"{config.name}.mid"
    write_midi(notes, config.melody, midi_path)

    # 5. 渲染音频
    audio_dir = DATA_DIR / "audio" / noise_type
    audio_path = audio_dir / f"{config.name}.wav"
    try:
        render_with_fluidsynth(midi_path, audio_path, config.render)
    except Exception as e:
        print(f"  FluidSynth 渲染失败 ({e}), 使用 pretty_midi 备选方案")
        render_with_pretty_midi(midi_path, audio_path, config.render)

    # 6. 统计分析
    freqs, psd = compute_psd(pitches)
    beta_psd, r2_psd = fit_spectral_exponent(freqs, psd)
    acf = compute_autocorrelation(pitches)

    # DFA（主方法）+ R/S（参考对比）
    dfa_alpha, dfa_r2 = estimate_dfa(pitches)
    dfa_raw = compute_dfa_raw(pitches)
    hurst_rs = estimate_hurst_rs(pitches)
    hurst_dfa, signal_type = dfa_to_hurst(dfa_alpha)
    fractal_dim = compute_fractal_dimension(hurst_dfa)

    zipf = zipf_analysis(pitches)
    direction = interval_direction_stats(pitches)

    stats = summary_statistics(pitches)
    stats["spectral_exponent_beta"] = beta_psd
    stats["spectral_fit_r2"] = r2_psd
    stats["dfa_alpha"] = dfa_alpha
    stats["dfa_r2"] = dfa_r2
    stats["dfa_beta"] = 2 * dfa_alpha - 1  # β = 2α - 1
    stats["hurst_dfa"] = hurst_dfa
    stats["hurst_rs"] = hurst_rs
    stats["signal_type"] = signal_type
    stats["fractal_dimension"] = fractal_dim

    return {
        "noise_type": noise_type,
        "config_name": config.name,
        "noise": noise,
        "pitches": pitches,
        "notes": notes,
        "midi_path": str(midi_path),
        "audio_path": str(audio_path),
        "psd": (freqs, psd, beta_psd),
        "acf": acf,
        "dfa_raw": dfa_raw,
        "zipf": zipf,
        "direction": direction,
        "stats": stats,
    }


def run_comparison(
    config: PipelineConfig,
    noise_types: list[str] = ("pink", "brown"),
) -> dict[str, dict]:
    """运行多种噪声类型的对比 pipeline。"""
    results = {}
    for noise_type in noise_types:
        print(f"[{noise_type}] 生成中...")
        results[noise_type] = run_single(config, noise_type)
        print(f"  MIDI: {results[noise_type]['midi_path']}")
        print(f"  Audio: {results[noise_type]['audio_path']}")
        stats = results[noise_type]["stats"]
        print(f"  PSD β = {stats['spectral_exponent_beta']:.2f}, "
              f"DFA α = {stats['dfa_alpha']:.2f} ({stats['signal_type']}), "
              f"H(DFA) = {stats['hurst_dfa']:.2f}, H(R/S) = {stats['hurst_rs']:.2f}")
        print(f"  Fractal D = {stats['fractal_dimension']:.2f}, "
              f"Entropy = {stats['entropy']:.2f}, "
              f"Zipf slope = {stats['zipf_slope']:.2f}")
        d = stats["direction"]
        print(f"  Direction: ↑{d['ascending']:.0%} ↓{d['descending']:.0%} "
              f"={d['repeated']:.0%}")

    # 生成对比图表
    print("\n生成分析图表...")
    _generate_comparison_plots(results)

    # 保存统计摘要
    summary_path = DATA_DIR / "analysis" / f"{config.name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v["stats"] for k, v in results.items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"统计摘要: {summary_path}")

    return results


def _generate_comparison_plots(results: dict[str, dict]):
    """生成所有对比分析图表。"""
    # 噪声序列波形
    plot_noise_sequence(
        {k: v["noise"] for k, v in results.items()},
        FIGURES_DIR / "noise_sequences.png",
    )

    # PSD 对比
    plot_psd_comparison(
        {k: v["psd"] for k, v in results.items()},
        FIGURES_DIR / "psd_comparison.png",
    )

    # 自相关对比
    plot_autocorrelation_comparison(
        {k: v["acf"] for k, v in results.items()},
        FIGURES_DIR / "acf_comparison.png",
    )

    # 音高分布
    plot_pitch_distribution(
        {k: pitch_distribution(v["pitches"]) for k, v in results.items()},
        FIGURES_DIR / "pitch_distribution.png",
    )

    # 音程分布
    plot_interval_distribution(
        {k: interval_distribution(v["pitches"]) for k, v in results.items()},
        FIGURES_DIR / "interval_distribution.png",
    )

    # Zipf 定律 log-log 图
    plot_zipf(
        {k: v["zipf"] for k, v in results.items()},
        FIGURES_DIR / "zipf_analysis.png",
    )

    # DFA 标度关系图
    plot_dfa_scaling(
        {k: (v["dfa_raw"]["scales"], v["dfa_raw"]["fluctuations"],
             v["dfa_raw"]["alpha"], v["dfa_raw"]["r_squared"])
         for k, v in results.items()},
        FIGURES_DIR / "dfa_scaling.png",
    )

    # 音程方向对比
    plot_interval_direction(
        {k: v["direction"] for k, v in results.items()},
        FIGURES_DIR / "interval_direction.png",
    )

    print(f"  图表已保存到 {FIGURES_DIR}/")
