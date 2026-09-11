import os
import pandas as pd
from env.task_env import TaskEnv
from test import ModelTester
from priority_stats import compute_priority_bins_stats_from_env, attach_priority_stats_to_df, plot_priority_comparison

def run_scalability_and_priority_plots(model_path='./model/save_8/checkpoint.pth', save_dir='./scalability_results'):
    """
    不改动 test.py：
    1) 使用 ModelTester.test_scalability 生成原始结果（CSV + 图）
    2) 在相同规模&trial的环境上再次运行（使用 tester._test_with_profiling 完成一集），统计优先级分层与加权完成时间
    3) 合并统计到结果DF，生成优先级分层完成率 + 加权时间图
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) 运行你原有的可扩展性实验
    tester = ModelTester(model_path, device='cuda')
    results_df = tester.test_scalability(save_dir=save_dir)
    if results_df is None or results_df.empty:
        print("❌ No scalability results to analyze.")
        return

    # 2) 针对每个 (scale, trial)，重新生成相同配置的环境，用 tester._test_with_profiling 跑到结束以获得 time_finish
    envs_stats = {}  # key=(scale, trial) -> stats_dict
    for _, row in results_df.iterrows():
        scale = row['scale']
        trial = int(row['trial'])

        # 对齐 test_scalability 中的规模配置
        if scale == 'small':
            cfg = {'tasks': (10, 15), 'agents_per_species': (3, 5), 'species': (3, 4)}
        elif scale == 'medium':
            cfg = {'tasks': (20, 30), 'agents_per_species': (4, 6), 'species': (4, 6)}
        elif scale == 'large':
            cfg = {'tasks': (40, 50), 'agents_per_species': (6, 8), 'species': (5, 7)}
        elif scale == 'xlarge':
            cfg = {'tasks': (70, 90), 'agents_per_species': (8, 10), 'species': (6, 8)}
        else:
            continue

        # 生成与 trial 相同 seed 的环境
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

        # 使用 tester._test_with_profiling 跑一次（与你的RL推理流程一致），以便填充 time_finish/finished 等字段
        try:
            _ = tester._test_with_profiling(env, max_steps=200)
        except Exception as e:
            print(f"⚠️ Profiling run failed for ({scale}, trial={trial}): {e}")
            continue

        # 统计优先级分层 & 加权时间
        stats = compute_priority_bins_stats_from_env(env)
        envs_stats[(scale, trial)] = stats

    # 3) 合并统计到 DF，保存并绘图
    df_with_prio = attach_priority_stats_to_df(results_df, envs_stats)
    out_csv = os.path.join(save_dir, 'scalability_results_with_priority.csv')
    df_with_prio.to_csv(out_csv, index=False)
    print(f"✅ Saved priority-augmented results to {out_csv}")

    plot_priority_comparison(df_with_prio, save_dir=save_dir)


if __name__ == '__main__':
    run_scalability_and_priority_plots()