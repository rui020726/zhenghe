"""
从已有数据文件读取覆盖率曲线并在同一张图上绘制多种通信条件的对比图。

支持两种输入数据源：
1) 每个条件有一个平均与标准差 CSV（包含 step, mean_coverage, std_coverage 列）
   -> 调用 load_mean_std_csv(filepath)
2) 每个条件有多条回合的覆盖率曲线（每条曲线为 list/1D-array），传入字典 condition -> list[curve]
   -> 调用 plot_conditions_from_episode_curves(...)

示例用法：
    # 方法A：从多个 average_coverage_data.csv 文件读取并绘图
    file_map = {
        "perfect": "metrics_out/perfect/average_coverage_data.csv",
        "poor": "metrics_out/poor/average_coverage_data.csv"
    }
    plot_conditions_mean_std_from_files(file_map, save_path="out/coverage_compare.png")

    # 方法B：如果你已经有每个回合的曲线数据（list of 1D arrays）
    data = {
        "perfect": [curve_ep1, curve_ep2, ...],
        "poor": [curve_ep1, curve_ep2, ...]
    }
    fig, ax = plot_conditions_from_episode_curves(data, max_steps=250, save_path="out/compare.png")

输出：返回 matplotlib.figure.Figure, matplotlib.axes.Axes
"""

import os
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pad_or_truncate_curve(curve: Sequence[float], max_steps: int) -> np.ndarray:
    """将单条曲线 pad 或截断到 max_steps 长度（pad 使用最后一个值填充）"""
    arr = np.asarray(curve, dtype=float)
    if arr.size == 0:
        return np.zeros(max_steps, dtype=float)
    if arr.size >= max_steps:
        return arr[:max_steps].copy()
    pad_len = max_steps - arr.size
    pad_vals = np.full(pad_len, arr[-1], dtype=float)
    return np.concatenate([arr, pad_vals], axis=0)


def _prepare_matrix_from_curves(curves: List[Sequence[float]], max_steps: int) -> np.ndarray:
    """把 list of curves 转换为 (num_episodes, max_steps) 的 numpy 矩阵"""
    if len(curves) == 0:
        return np.zeros((0, max_steps), dtype=float)
    mat = np.vstack([_pad_or_truncate_curve(c, max_steps) for c in curves])
    return mat


def load_mean_std_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取一个包含 step, mean_coverage, std_coverage 的 CSV 文件。
    支持列名的小变体（'step','mean','mean_coverage','std','std_coverage' 等）。

    返回:
        steps (1..N), mean_curve (length N), std_curve (length N)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    # 容错：允许多种列名
    col_map = {c.lower(): c for c in df.columns}
    # step 列
    if 'step' in col_map:
        steps = df[col_map['step']].to_numpy(dtype=int)
    else:
        # 如果没有 step 列就生成默认序列
        steps = np.arange(1, len(df) + 1)

    # mean 列
    mean_col_candidates = ['mean_coverage', 'mean', 'coverage', 'meancoverage']
    mean_col = None
    for cand in mean_col_candidates:
        if cand in col_map:
            mean_col = col_map[cand]
            break
    if mean_col is None:
        # 尝试匹配包含 mean 的列
        for k in col_map:
            if 'mean' in k:
                mean_col = col_map[k]
                break
    if mean_col is None:
        raise ValueError(f"No mean column found in CSV: {path}")

    mean_curve = df[mean_col].to_numpy(dtype=float)

    # std 列
    std_col_candidates = ['std_coverage', 'std', 'stdcoverage', 'std_dev']
    std_col = None
    for cand in std_col_candidates:
        if cand in col_map:
            std_col = col_map[cand]
            break
    if std_col is None:
        for k in col_map:
            if 'std' in k or 'sigma' in k or 'variance' in k:
                std_col = col_map[k]
                break
    if std_col is None:
        # 如果没有 std 列，可以使用 zeros
        std_curve = np.zeros_like(mean_curve)
    else:
        std_curve = df[std_col].to_numpy(dtype=float)

    return steps, mean_curve, std_curve


def plot_conditions_mean_std_from_files(file_map: Dict[str, str],
                                        save_path: Optional[str] = None,
                                        max_steps: Optional[int] = None,
                                        figsize: Tuple[int, int] = (10, 6),
                                        title: str = "Coverage Comparison (mean ± std)",
                                        xlabel: str = "Time Steps",
                                        ylabel: str = "Coverage Rate",
                                        colors: Optional[Dict[str, str]] = None,
                                        show_legend: bool = True,
                                        ylim: Tuple[float, float] = (0.0, 1.0),
                                        dpi: int = 200) -> Tuple[plt.Figure, plt.Axes]:
    """
    从多个平均+标准差 CSV 文件读取并在一张图上绘制（每个条件一条 mean 曲线及其 ±1σ 阴影带）。

    参数:
        file_map: dict { condition_name: path_to_csv }
        save_path: 若提供则保存图像
        max_steps: 若提供则对齐到 max_steps（会 pad/truncate）
    返回 fig, ax
    """
    if not file_map:
        raise ValueError("file_map 不能为空")

    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10.colors
    cond_names = list(file_map.keys())
    if colors is None:
        colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(cond_names)}
    else:
        # 补充缺失的颜色
        for i, name in enumerate(cond_names):
            if name not in colors:
                colors[name] = default_colors[i % len(default_colors)]

    # 读取每个文件并绘图
    for name, path in file_map.items():
        try:
            steps, mean_curve, std_curve = load_mean_std_csv(path)
        except Exception as e:
            print(f"跳过 {name}: 读取文件失败 ({path}) -> {e}")
            continue

        # 决定 max_steps
        if max_steps is None:
            target_steps = int(steps[-1])
        else:
            target_steps = int(max_steps)

        # pad 或截断
        mean_padded = _pad_or_truncate_curve(mean_curve, target_steps)
        std_padded = _pad_or_truncate_curve(std_curve, target_steps)
        steps_plot = np.arange(1, target_steps + 1)

        ax.plot(steps_plot, mean_padded, label=name, color=colors[name], linewidth=2.0)
        lower = np.clip(mean_padded - std_padded, 0.0, 1.0)
        upper = np.clip(mean_padded + std_padded, 0.0, 1.0)
        ax.fill_between(steps_plot, lower, upper, color=colors[name], alpha=0.2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(1, max_steps if max_steps is not None else None)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.3)
    if show_legend:
        ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig, ax


def plot_conditions_from_episode_curves(all_episode_data: Dict[str, List[Sequence[float]]],
                                        save_path: Optional[str] = None,
                                        max_steps: int = 250,
                                        figsize: Tuple[int, int] = (10, 6),
                                        title: str = "Coverage Comparison (mean ± std from episodes)",
                                        xlabel: str = "Time Steps",
                                        ylabel: str = "Coverage Rate",
                                        colors: Optional[Dict[str, str]] = None,
                                        show_individual: bool = False,
                                        individual_alpha: float = 0.12,
                                        linewidth: float = 2.0,
                                        show_legend: bool = True,
                                        ylim: Tuple[float, float] = (0.0, 1.0),
                                        dpi: int = 200) -> Tuple[plt.Figure, plt.Axes]:
    """
    从每回合曲线数据绘制对比图（先计算 mean/std，再绘图）。
    all_episode_data: { condition_name: [curve_ep1, curve_ep2, ...] }
    """
    if not all_episode_data:
        raise ValueError("all_episode_data 不能为空")

    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10.colors
    cond_names = list(all_episode_data.keys())
    if colors is None:
        colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(cond_names)}
    else:
        for i, name in enumerate(cond_names):
            if name not in colors:
                colors[name] = default_colors[i % len(default_colors)]

    steps = np.arange(1, max_steps + 1)

    for name in cond_names:
        curves = all_episode_data[name]
        mat = _prepare_matrix_from_curves(curves, max_steps)  # shape (num_episodes, max_steps)
        if mat.size == 0:
            print(f"注意: 条件 {name} 没有曲线数据，跳过")
            continue

        mean_curve = np.nanmean(mat, axis=0)
        std_curve = np.nanstd(mat, axis=0)

        if show_individual:
            for ep in range(mat.shape[0]):
                ax.plot(steps, mat[ep], color=colors[name], alpha=individual_alpha, linewidth=0.8)

        ax.plot(steps, mean_curve, color=colors[name], linewidth=linewidth, label=name)
        ax.fill_between(steps, np.clip(mean_curve - std_curve, 0.0, 1.0),
                        np.clip(mean_curve + std_curve, 0.0, 1.0),
                        color=colors[name], alpha=0.2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(1, max_steps)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.3)
    if show_legend:
        ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig, ax


# --------------------------
# 示例：如何基于你给出的 CSV 文件生成图像
# --------------------------
if __name__ == "__main__":
    # 假设下面这些文件分别是不同通信条件下计算得到的平均覆盖率 CSV（结构如你给出的 average_coverage_data.csv）
    file_map_example = {
        "perfect": "metrics_out/perfect/average_coverage_data.csv",
        "good": "metrics_out/good/average_coverage_data.csv",
        "medium": "metrics_out/medium/average_coverage_data.csv",
        "poor": "metrics_out/poor/average_coverage_data.csv",
        "very_poor": "metrics_out/very_poor/average_coverage_data.csv"
    }

    # 如果你只有一个文件（你提供的 average_coverage_data.csv），也可以这样绘制单条曲线：
    single_file = "average_coverage_data.csv"  # 这是你在对话中贴出的文件
    if os.path.exists(single_file):
        fig, ax = plot_conditions_mean_std_from_files({"example": single_file},
                                                      save_path="out/coverage_single_example.png",
                                                      max_steps=None)
        print("绘制完成: out/coverage_single_example.png")
    else:
        # 否则尝试绘制多个条件（如果这些文件实际存在）
        existing_files = {k: v for k, v in file_map_example.items() if os.path.exists(v)}
        if existing_files:
            fig, ax = plot_conditions_mean_std_from_files(existing_files,
                                                          save_path="out/coverage_multi_conditions.png",
                                                          max_steps=250)
            print("绘制完成: out/coverage_multi_conditions.png")
        else:
            print("未找到示例 CSV 文件，请将 average_coverage_data.csv 放在当前目录或指定 metrics_out/* 下的文件。")