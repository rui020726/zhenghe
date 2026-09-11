# -*- coding: utf-8 -*-
"""
优先级证据生成与可视化（适配新版 test.py / driver.py）

改动点：
- 使用 ModelTester（新版）进行推理，并沿用与 driver/test 一致的 padding、掩码与采样逻辑（通过 tester.test_single_env 触发 env 状态更新）
- 不直接调用旧版 tester._test_with_profiling（若你保留该方法也可用）；这里采用更稳的：逐集执行 test_single_env 更新 env.task_dic 的 finished/time_finish
- 完成后，按任务层面收集 time_finish、priority，生成 ECDF/散点/箱线图，并计算 front-load 与优先级一致性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env.task_env import TaskEnv
from test import ModelTester


# 分层规则
def priority_bins(p):
    if p >= 0.8:
        return 'ultra (≥0.8)'
    elif p >= 0.6:
        return 'high (0.6–0.8)'
    elif p >= 0.4:
        return 'medium (0.4–0.6)'
    else:
        return 'low (<0.4)'


def build_config(scale):
    if scale == 'small':
        return {'tasks': (10, 15), 'agents_per_species': (3, 5), 'species': (3, 4)}
    if scale == 'medium':
        return {'tasks': (20, 30), 'agents_per_species': (4, 6), 'species': (4, 6)}
    if scale == 'large':
        return {'tasks': (40, 50), 'agents_per_species': (6, 8), 'species': (5, 7)}
    if scale == 'xlarge':
        return {'tasks': (70, 90), 'agents_per_species': (8, 10), 'species': (6, 8)}
    raise ValueError(f'Unknown scale: {scale}')


def run_episode_with_tester(tester: ModelTester, env: TaskEnv, max_steps=200):
    """
    使用新版 ModelTester 执行一次完整 episode（仅推理），以便更新 env.task_dic 的 finished/time_finish。
    返回 makespan（env.current_time）和 success_rate。
    """
    metrics = tester.test_single_env(env, max_steps=max_steps)
    makespan = float(env.current_time) if hasattr(env, 'current_time') else np.nan
    success_rate = metrics.get('success_rate', 0.0) if isinstance(metrics, dict) else 0.0
    return makespan, success_rate


def collect_task_times(model_path='./model/save_12/checkpoint.pth', save_dir='./scalability_results'):
    """
    跑四档规模（每档 trial=0..9），收集每个任务的完成时间与优先级。
    使用 tester.test_single_env 执行episode来更新任务完成状态与 time_finish。
    返回：DataFrame[scale,trial,task_id,priority,band,finished,time_finish,makespan,norm_time,success_rate]
    """
    os.makedirs(save_dir, exist_ok=True)
    tester = ModelTester(model_path, device='cuda')

    scales = ['small', 'medium', 'large', 'xlarge']
    trials = list(range(10))

    rows = []
    for scale in scales:
        cfg = build_config(scale)
        for trial in trials:
            # 构建环境
            env = TaskEnv(
                prio_range=(0.3, 1.0),
                per_species_range=cfg['agents_per_species'],
                species_range=cfg['species'],
                tasks_range=cfg['tasks'],
                traits_dim=5,
                max_task_size=2,
                duration_scale=5,
                seed=trial
            )
            env.init_state()

            # 执行一集以填充任务完成时间
            try:
                makespan, sr = run_episode_with_tester(tester, env, max_steps=200)
            except Exception as e:
                print(f'⚠️ Run failed for {scale} trial={trial}: {e}')
                continue

            # 提取任务层数据
            for tid, task in env.task_dic.items():
                p = float(task.get('search_priority', 0.5))
                finished = bool(task.get('finished', False))
                tfin = float(task.get('time_finish', np.nan)) if finished else np.nan
                norm_time = (tfin / makespan) if (finished and makespan and makespan > 0) else np.nan
                band = priority_bins(p)
                rows.append({
                    'scale': scale,
                    'trial': trial,
                    'task_id': tid,
                    'priority': p,
                    'band': band,
                    'finished': finished,
                    'time_finish': tfin,
                    'makespan': makespan,
                    'norm_time': norm_time,
                    'success_rate': sr,
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(save_dir, 'task_finish_times.csv')
    df.to_csv(out_csv, index=False)
    print(f'✅ Saved task-level data to {out_csv}')
    return df


def plot_ecdf_by_band(df, save_dir='./scalability_results'):
    """
    ECDF（累计完成曲线）：横轴归一化时间（time_finish/makespan），纵轴为分层内完成比例。
    高优先级曲线应更靠左更靠上，表示更早完成。
    """
    os.makedirs(save_dir, exist_ok=True)
    bands_order = ['ultra (≥0.8)', 'high (0.6–0.8)', 'medium (0.4–0.6)', 'low (<0.4)']
    colors = {
        'ultra (≥0.8)': '#e74c3c',
        'high (0.6–0.8)': '#f39c12',
        'medium (0.4–0.6)': '#3498db',
        'low (<0.4)': '#2ecc71'
    }

    for scale in df['scale'].unique():
        sdf = df[(df['scale'] == scale) & (df['finished'] == True) & (~df['norm_time'].isna())]
        if sdf.empty:
            print(f'⚠️ ECDF skipped for {scale}: no finished tasks with valid times')
            continue
        plt.figure(figsize=(7, 5))
        for band in bands_order:
            bdf = sdf[sdf['band'] == band]
            if bdf.empty:
                continue
            x = np.sort(bdf['norm_time'].values.clip(0, 1))
            y = np.arange(1, len(x) + 1) / len(x)
            plt.step(x, y, where='post', label=band, color=colors.get(band, None), linewidth=2, alpha=0.9)
        plt.xlabel('Normalized completion time (t_finish / makespan)')
        plt.ylabel('Cumulative completion ratio')
        plt.title(f'Cumulative Completion by Priority Band ({scale})')
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend()
        out = os.path.join(save_dir, f'curves_completion_{scale}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'✅ Saved {out}')


def plot_scatter_priority(df, save_dir='./scalability_results'):
    """
    散点图：priority vs 归一化完成时间，附Spearman/Pearson相关系数。
    预期负相关（优先级越高，完成越早）。
    """
    os.makedirs(save_dir, exist_ok=True)
    for scale in df['scale'].unique():
        sdf = df[(df['scale'] == scale) & (df['finished'] == True) & (~df['norm_time'].isna())]
        if sdf.empty:
            print(f'⚠️ Scatter skipped for {scale}: no finished tasks with valid times')
            continue
        x = sdf['priority'].values
        y = sdf['norm_time'].values
        # 相关系数
        if len(x) >= 3:
            pearson = np.corrcoef(x, y)[0, 1]
            # Spearman
            rx = pd.Series(x).rank(method='average').values
            ry = pd.Series(y).rank(method='average').values
            spearman = np.corrcoef(rx, ry)[0, 1]
        else:
            pearson = np.nan
            spearman = np.nan

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, s=18, alpha=0.6, c=y, cmap='coolwarm')
        # 简单线性回归趋势线
        if len(x) >= 2 and np.ptp(x) > 1e-8:
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 100)
            plt.plot(xx, m * xx + c, color='black', linewidth=2, alpha=0.8, label='trend')
        plt.xlabel('Priority')
        plt.ylabel('Normalized completion time')
        plt.title(f'Priority vs Normalized completion time ({scale})\nPearson={pearson:.2f}, Spearman={spearman:.2f}')
        plt.grid(alpha=0.3)
        if len(x) >= 2:
            plt.legend()
        out = os.path.join(save_dir, f'scatter_priority_vs_time_{scale}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'✅ Saved {out}')


def plot_box_by_band(df, save_dir='./scalability_results'):
    """
    箱线图：按分层展示归一化完成时间的分布。
    """
    os.makedirs(save_dir, exist_ok=True)
    order = ['ultra (≥0.8)', 'high (0.6–0.8)', 'medium (0.4–0.6)', 'low (<0.4)']
    for scale in df['scale'].unique():
        sdf = df[(df['scale'] == scale) & (df['finished'] == True) & (~df['norm_time'].isna())]
        if sdf.empty:
            print(f'⚠️ Boxplot skipped for {scale}: no finished tasks with valid times')
            continue
        sdf = sdf.copy()
        sdf['band'] = pd.Categorical(sdf['band'], categories=order, ordered=True)
        sdf = sdf.sort_values('band')
        groups = [sdf[sdf['band'] == b]['norm_time'].dropna().values for b in order]
        labels = order
        plt.figure(figsize=(7, 5))
        plt.boxplot(groups, labels=labels, showmeans=True,
                    meanline=True, medianprops=dict(color='black'),
                    meanprops=dict(color='#e67e22'))
        plt.ylabel('Normalized completion time')
        plt.title(f'Completion time distribution by priority band ({scale})')
        plt.grid(axis='y', alpha=0.3)
        out = os.path.join(save_dir, f'box_time_by_band_{scale}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'✅ Saved {out}')


def compute_frontload_and_concordance(df, save_dir='./scalability_results'):
    """
    - Front-load 指数（越大越好）：FL = 1 - mean(norm_time)，按分层与规模
    - 优先级一致性（Concordance）：随机抽取任务对 (i,j)，统计 p_i>p_j 时是否有 t_i<t_j
    输出保存为 priority_evidence_summary.txt
    """
    os.makedirs(save_dir, exist_ok=True)
    report_lines = []

    # Front-load
    fl_rows = []
    for scale in df['scale'].unique():
        sdf = df[(df['scale'] == scale) & (df['finished'] == True) & (~df['norm_time'].isna())]
        if sdf.empty:
            continue
        for band, g in sdf.groupby('band', observed=False):
            if len(g) == 0:
                continue
            fl = 1.0 - float(g['norm_time'].mean())
            fl_rows.append({'scale': scale, 'band': band, 'frontload': fl})
    fl_df = pd.DataFrame(fl_rows)
    if not fl_df.empty:
        pivot = fl_df.pivot(index='band', columns='scale', values='frontload')
        report_lines.append('Front-load index (higher is better):\n' + pivot.round(3).to_string())
        report_lines.append('')

    # Concordance（抽样提升效率）
    def concordance_for_subset(sdf, sample_pairs=20000, rng=np.random.default_rng(0)):
        sdf = sdf[['priority', 'norm_time']].dropna()
        n = len(sdf)
        if n < 2:
            return np.nan
        idx = rng.integers(0, n, size=(sample_pairs, 2))
        i = idx[:, 0]; j = idx[:, 1]
        # 过滤 p_i != p_j
        mask = (sdf['priority'].values[i] != sdf['priority'].values[j])
        i = i[mask]; j = j[mask]
        if len(i) == 0:
            return np.nan
        p_i = sdf['priority'].values[i]; p_j = sdf['priority'].values[j]
        t_i = sdf['norm_time'].values[i]; t_j = sdf['norm_time'].values[j]
        wins = ((p_i > p_j) & (t_i < t_j)).sum() + ((p_j > p_i) & (t_j < t_i)).sum()
        total = len(i)
        return wins / total if total > 0 else np.nan

    overall = concordance_for_subset(df[(df['finished'] == True) & (~df['norm_time'].isna())])
    report_lines.append(f'Overall priority concordance: {overall:.3f} (probability higher-priority finishes earlier)')
    for scale in df['scale'].unique():
        sdf = df[(df['scale'] == scale) & (df['finished'] == True) & (~df['norm_time'].isna())]
        c = concordance_for_subset(sdf)
        report_lines.append(f'  {scale}: {c:.3f}')
    report = '\n'.join(report_lines)

    out_txt = os.path.join(save_dir, 'priority_evidence_summary.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'✅ Saved summary to {out_txt}')
    print('\n' + report)


def main():
    save_dir = './scalability_results'
    model_path = './model/save_12/checkpoint.pth'

    # 任务级数据
    df = collect_task_times(model_path=model_path, save_dir=save_dir)

    # 可视化（更直观）
    plot_ecdf_by_band(df, save_dir=save_dir)                 # 累计完成曲线（核心证据）
    plot_scatter_priority(df, save_dir=save_dir)             # 散点+相关
    plot_box_by_band(df, save_dir=save_dir)                  # 分层箱线图

    # 量化指标
    compute_frontload_and_concordance(df, save_dir=save_dir)


if __name__ == '__main__':
    main()