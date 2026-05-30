"""跨批次汇总分析

汇总所有批次结果，生成对比表格和统计图表。
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = DATA_DIR / "analysis" / "figures"


def collect_summaries() -> dict[str, list[dict]]:
    """收集所有 batch 的统计摘要。"""
    analysis_dir = DATA_DIR / "analysis"
    pink_stats = []
    brown_stats = []

    for f in sorted(analysis_dir.glob("*_summary.json")):
        with open(f) as fh:
            data = json.load(fh)
        if "pink" in data:
            pink_stats.append({"run": f.stem, **data["pink"]})
        if "brown" in data:
            brown_stats.append({"run": f.stem, **data["brown"]})

    return {"pink": pink_stats, "brown": brown_stats}


def print_summary_table(all_stats: dict[str, list[dict]]):
    """打印跨批次汇总表。"""
    for noise_type, stats_list in all_stats.items():
        if not stats_list:
            continue

        betas = [s["spectral_exponent_beta"] for s in stats_list]
        dfas = [s["dfa_alpha"] for s in stats_list]
        h_dfas = [s["hurst_dfa"] for s in stats_list]
        h_rs = [s["hurst_rs"] for s in stats_list]
        frac_ds = [s["fractal_dimension"] for s in stats_list]
        entropies = [s["entropy"] for s in stats_list]
        intervals = [s["interval_mean"] for s in stats_list]
        zipf_slopes = [s.get("zipf_slope", 0.0) for s in stats_list]
        asc = [s.get("direction", {}).get("ascending", 0.0) for s in stats_list]
        desc = [s.get("direction", {}).get("descending", 0.0) for s in stats_list]

        print(f"\n{'='*60}")
        print(f"  {noise_type.upper()} NOISE — {len(stats_list)} runs 汇总")
        print(f"{'='*60}")
        print(f"  {'指标':<25} {'均值':>8} {'标准差':>8} {'最小':>8} {'最大':>8}")
        print(f"  {'-'*55}")

        for name, vals in [
            ("PSD β", betas),
            ("DFA α", dfas),
            ("Hurst (DFA)", h_dfas),
            ("Hurst (R/S)", h_rs),
            ("分形维度 D", frac_ds),
            ("Shannon 熵", entropies),
            ("平均音程 (半音)", intervals),
            ("Zipf 斜率", zipf_slopes),
            ("上行比例", asc),
            ("下行比例", desc),
        ]:
            arr = np.array(vals)
            print(f"  {name:<25} {arr.mean():>8.3f} {arr.std():>8.3f} "
                  f"{arr.min():>8.3f} {arr.max():>8.3f}")

        # 理论值对比
        theory_beta = 1.0 if noise_type == "pink" else 2.0
        theory_alpha = (theory_beta + 1) / 2
        print(f"\n  理论参考: β = {theory_beta:.1f}, α = {theory_alpha:.1f}")
        print(f"  实测均值: β = {np.mean(betas):.2f}, α = {np.mean(dfas):.2f}")
        print(f"  偏差: Δβ = {np.mean(betas) - theory_beta:+.2f}, "
              f"Δα = {np.mean(dfas) - theory_alpha:+.2f}")


def plot_cross_batch_comparison(all_stats: dict[str, list[dict]]):
    """生成跨批次对比图。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {"pink": "#E91E63", "brown": "#795548"}

    metrics = [
        ("spectral_exponent_beta", "PSD β", [1.0, 2.0]),
        ("dfa_alpha", "DFA α", [1.0, 1.5]),
        ("hurst_dfa", "Hurst (DFA)", None),
        ("hurst_rs", "Hurst (R/S)", None),
        ("fractal_dimension", "Fractal Dimension D", None),
        ("interval_mean", "Mean Interval (semitones)", None),
    ]

    for ax, (key, label, theory_vals) in zip(axes.flat, metrics):
        for i, (noise_type, stats_list) in enumerate(all_stats.items()):
            vals = [s[key] for s in stats_list]
            x = np.arange(len(vals)) + i * 0.3
            color = colors.get(noise_type, "gray")
            ax.bar(x, vals, width=0.28, color=color, alpha=0.7, label=noise_type)

            if theory_vals:
                theory = theory_vals[i]
                ax.axhline(y=theory, color=color, ls="--", lw=1, alpha=0.5)

        ax.set_xlabel("Run")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.2)

    fig.suptitle("Cross-Batch Comparison: Pink vs Brown Noise", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output = FIGURES_DIR / "cross_batch_comparison.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n跨批次对比图: {output}")


if __name__ == "__main__":
    all_stats = collect_summaries()
    print_summary_table(all_stats)
    plot_cross_batch_comparison(all_stats)

    # 保存汇总 JSON
    output_json = DATA_DIR / "analysis" / "cross_batch_summary.json"
    summary = {}
    for noise_type, stats_list in all_stats.items():
        vals = {
            "n_runs": len(stats_list),
            "beta_mean": float(np.mean([s["spectral_exponent_beta"] for s in stats_list])),
            "beta_std": float(np.std([s["spectral_exponent_beta"] for s in stats_list])),
            "dfa_alpha_mean": float(np.mean([s["dfa_alpha"] for s in stats_list])),
            "hurst_dfa_mean": float(np.mean([s["hurst_dfa"] for s in stats_list])),
            "hurst_rs_mean": float(np.mean([s["hurst_rs"] for s in stats_list])),
            "fractal_dim_mean": float(np.mean([s["fractal_dimension"] for s in stats_list])),
            "interval_mean": float(np.mean([s["interval_mean"] for s in stats_list])),
            "zipf_slope_mean": float(np.mean([s.get("zipf_slope", 0.0) for s in stats_list])),
            "ascending_mean": float(np.mean([s.get("direction", {}).get("ascending", 0.0) for s in stats_list])),
            "descending_mean": float(np.mean([s.get("direction", {}).get("descending", 0.0) for s in stats_list])),
        }
        summary[noise_type] = vals
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"汇总数据: {output_json}")
