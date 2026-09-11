import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_priority_bins_stats_from_env(env):
    """
    从一个已运行完成的 env 中统计优先级分层完成率与加权完成时间。
    返回 dict:
      {
        'bin_counts': {'ultra_high':N, 'high':N, 'medium':N, 'low':N},
        'bin_finished': {'ultra_high':n, ...},
        'bin_rate': {'ultra_high': p, ...},  # 完成率
        'weighted_time': float,              # 优先级加权完成时间的均值
        'per_task_weighted_times': list      # 每个已完成任务的 p * time_finish
      }
    """
    priorities = np.array([t.get('search_priority', 0.5) for t in env.task_dic.values()])
    finished = np.array([t.get('finished', False) for t in env.task_dic.values()])
    time_finish = np.array([t.get('time_finish', 0.0) for t in env.task_dic.values()])

    # 分层区间（可调整）
    ultra_high_mask = priorities >= 0.8
    high_mask = (priorities >= 0.6) & (priorities < 0.8)
    medium_mask = (priorities >= 0.4) & (priorities < 0.6)
    low_mask = priorities < 0.4

    bin_counts = {
        'ultra_high': int(ultra_high_mask.sum()),
        'high': int(high_mask.sum()),
        'medium': int(medium_mask.sum()),
        'low': int(low_mask.sum()),
    }
    bin_finished = {
        'ultra_high': int((ultra_high_mask & finished).sum()),
        'high': int((high_mask & finished).sum()),
        'medium': int((medium_mask & finished).sum()),
        'low': int((low_mask & finished).sum()),
    }
    bin_rate = {k: (bin_finished[k] / bin_counts[k] if bin_counts[k] > 0 else 0.0) for k in bin_counts.keys()}

    # 优先级加权完成时间（只统计已完成任务）
    completed_priorities = priorities[finished]
    completed_times = time_finish[finished]
    per_task_weighted_times = (completed_priorities * completed_times).tolist()
    weighted_time = float(np.mean(completed_priorities * completed_times)) if len(completed_priorities) > 0 else 0.0

    return {
        'bin_counts': bin_counts,
        'bin_finished': bin_finished,
        'bin_rate': bin_rate,
        'weighted_time': weighted_time,
        'per_task_weighted_times': per_task_weighted_times,
    }


def attach_priority_stats_to_df(df, envs_stats_by_trial_scale):
    """
    将每个环境的优先级统计结果合并到已存在的 results_df 上。
    参数:
      - df: 由 test_scalability 返回的 DataFrame（包含 scale, trial 等字段）
      - envs_stats_by_trial_scale: dict[(scale, trial)] = stats_dict
    返回:
      - df_new: 在 df 基础上新增 prio_ultra_rate / prio_high_rate / prio_medium_rate / prio_low_rate / prio_weighted_time 字段
    """
    prio_ultra = []
    prio_high = []
    prio_medium = []
    prio_low = []
    prio_wtime = []

    for _, row in df.iterrows():
        key = (row['scale'], row['trial'])
        stats = envs_stats_by_trial_scale.get(key, None)
        if stats is None:
            prio_ultra.append(np.nan)
            prio_high.append(np.nan)
            prio_medium.append(np.nan)
            prio_low.append(np.nan)
            prio_wtime.append(np.nan)
        else:
            bin_rate = stats['bin_rate']
            prio_ultra.append(bin_rate.get('ultra_high', 0.0))
            prio_high.append(bin_rate.get('high', 0.0))
            prio_medium.append(bin_rate.get('medium', 0.0))
            prio_low.append(bin_rate.get('low', 0.0))
            prio_wtime.append(stats['weighted_time'])

    df_new = df.copy()
    df_new['prio_ultra_rate'] = prio_ultra
    df_new['prio_high_rate'] = prio_high
    df_new['prio_medium_rate'] = prio_medium
    df_new['prio_low_rate'] = prio_low
    df_new['prio_weighted_time'] = prio_wtime
    return df_new


def plot_priority_comparison(df, save_dir='./scalability_results'):
    """
    输入 df（含 prio_* 字段），按 scale 生成：
    - 高优先级完成率对比图（Ultra ≥0.8 / High 0.6–0.8）
    - 优先级加权完成时间对比图
    """
    os.makedirs(save_dir, exist_ok=True)
    if not {'prio_ultra_rate', 'prio_high_rate', 'prio_weighted_time'}.issubset(df.columns):
        print("⚠️ Priority stats not found in results. Skipping priority comparison plots.")
        return

    for scale in df['scale'].unique():
        sdf = df[df['scale'] == scale]

        # 1) 高优先级完成率（Ultra / High）
        plt.figure(figsize=(8, 5))
        ultra_mean = sdf['prio_ultra_rate'].mean()
        high_mean = sdf['prio_high_rate'].mean()
        labels = ['Ultra (≥0.8)', 'High (0.6–0.8)']
        values = [ultra_mean, high_mean]
        plt.bar(labels, values, color=['#e74c3c', '#f39c12'], alpha=0.85)
        plt.ylim(0, 1.05)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
        plt.ylabel('Completion Rate')
        plt.title(f'High-priority Completion Rates ({scale})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'priority_completion_rates_{scale}.png'), dpi=150)
        plt.close()

        # 2) 优先级加权完成时间
        plt.figure(figsize=(8, 5))
        w_mean = sdf['prio_weighted_time'].mean()
        plt.bar([f'{scale}'], [w_mean], color='#3498db', alpha=0.85)
        plt.ylabel('Mean Priority-weighted Completion Time')
        plt.title(f'Priority-weighted Completion Time ({scale})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'priority_weighted_time_{scale}.png'), dpi=150)
        plt.close()

    print(f"✅ Priority comparison plots saved under {save_dir}/")