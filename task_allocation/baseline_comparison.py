import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env.task_env import TaskEnv
from baselines import (
    HungarianBaseline,
    GreedyNearestBaseline,
    GreedyPriorityBaseline,
    GreedyCoalitionBaseline,
    CBBABaseline,
)

# 你的 RL 方法封装：调用现有 ModelTester.test_single_env
from test import ModelTester


def run_baseline_on_env(env, baseline_cls, max_steps=200):
    """在一个环境上运行某个基线方法，返回指标"""
    env_copy = copy.deepcopy(env)
    env_copy.init_state()
    baseline = baseline_cls(env_copy)

    start = time.time()
    steps = 0
    try:
        while not env_copy.finished and env_copy.current_time < 200 and steps < max_steps:
            release_agents, next_decision_time = env_copy.next_decision()
            env_copy.current_time = next_decision_time
            # 当前需要决策的智能体（释放 + 阻塞）
            to_decide = release_agents[0] + release_agents[1]
            if len(to_decide) == 0:
                break

            # 基线生成分配
            assignment = baseline.assign_tasks(to_decide)

            # 执行动作（传入 agent_step 的 task_id：0仓库，1..N任务）
            for aid in to_decide:
                task_id = assignment.get(aid, 0)
                try:
                    env_copy.agent_step(aid, task_id, 0)
                except Exception:
                    pass

            steps += 1

    except Exception as e:
        print(f"[Baseline {baseline.name}] Error: {e}")

    inference_time = time.time() - start
    reward, finished_tasks = env_copy.get_episode_reward(200)
    success_rate = float(np.sum(finished_tasks)) / float(len(finished_tasks)) if finished_tasks else 0.0

    return {
        'method': baseline.name,
        'success_rate': success_rate,
        'makespan': float(env_copy.current_time),
        'reward': float(reward),
        'inference_time': inference_time,
        'finished_tasks': int(np.sum(finished_tasks)) if finished_tasks else 0,
        'total_tasks': len(env_copy.task_dic),
    }


def run_your_method_on_env(env, tester: ModelTester):
    """在一个环境上运行你的RL方法（集中式）"""
    env_copy = copy.deepcopy(env)
    env_copy.init_state()

    start = time.time()
    metrics = tester.test_single_env(env_copy, max_steps=200, analyze_failure=False)
    metrics['method'] = 'Your Method (RL)'
    metrics['inference_time'] = time.time() - start
    return metrics


def compare_methods_same_scales(num_envs_per_scale=10, device='cuda', save_dir='./comparison_results'):
    """
    与 test.py 中可扩展性场景相同的四档规模进行对比：
    - small:  10-15 tasks,  3-5 agents/species, 3-4 species
    - medium: 20-30 tasks,  4-6 agents/species, 4-6 species
    - large:  40-50 tasks,  6-8 agents/species, 5-7 species
    - xlarge: 70-90 tasks,  8-10 agents/species, 6-8 species
    每档规模跑 num_envs_per_scale 次（不同 seed）
    """
    os.makedirs(save_dir, exist_ok=True)

    scale_configs = [
        {'tasks': (10, 15), 'agents_per_species': (3, 5), 'species': (3, 4), 'name': 'small'},
        {'tasks': (20, 30), 'agents_per_species': (4, 6), 'species': (4, 6), 'name': 'medium'},
        {'tasks': (40, 50), 'agents_per_species': (6, 8), 'species': (5, 7), 'name': 'large'},
        {'tasks': (70, 90), 'agents_per_species': (8, 10), 'species': (6, 8), 'name': 'xlarge'},
    ]

    model_path = './model/save_8/checkpoint.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return pd.DataFrame()

    tester = ModelTester(model_path, device=device)

    records = []

    for config in scale_configs:
        name = config['name']
        print(f"\n{'='*80}\nRunning scale: {name} | tasks={config['tasks']} | agents/species={config['agents_per_species']} | species={config['species']}\n{'='*80}")

        for i in range(num_envs_per_scale):
            seed = i  # 与 test.py 一致的增量 seed
            env = TaskEnv(
                prio_range=(0.3, 1.0),
                per_species_range=config['agents_per_species'],
                species_range=config['species'],
                tasks_range=config['tasks'],
                traits_dim=5,
                max_task_size=2,
                duration_scale=5,
                seed=seed
            )
            env.init_state()

            # 你的方法
            try:
                r = run_your_method_on_env(env, tester)
                records.append({
                    'scale': name,
                    'method': 'Your Method (RL)',
                    'success_rate': r.get('success_rate', 0.0),
                    'makespan': r.get('makespan', 0.0),
                    'reward': r.get('reward', 0.0),
                    'inference_time': r.get('inference_time', 0.0),
                    'env_id': f'{name}_{i}'
                })
                print(f"  Your Method: Success={records[-1]['success_rate']:.2%}, Time={records[-1]['inference_time']:.2f}s")
            except Exception as e:
                print(f"  Your Method failed: {e}")

            # 基线列表
            baselines = [
                HungarianBaseline,
                GreedyNearestBaseline,
                GreedyPriorityBaseline,
                GreedyCoalitionBaseline,
                CBBABaseline,
            ]
            for b in baselines:
                try:
                    m = run_baseline_on_env(env, b)
                    m['scale'] = name
                    m['env_id'] = f'{name}_{i}'
                    records.append(m)
                    print(f"  {m['method']}: Success={m['success_rate']:.2%}, Time={m['inference_time']:.2f}s")
                except Exception as e:
                    print(f"  {b.__name__} failed: {e}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(save_dir, 'baseline_raw_results.csv'), index=False)
    print(f"✅ Raw results saved to {save_dir}/baseline_raw_results.csv")

    # 统计并绘图（按规模与方法）
    grouped = df.groupby(['scale', 'method']).agg({
        'success_rate': ['mean', 'std'],
        'makespan': ['mean', 'std'],
        'inference_time': ['mean', 'std'],
        'reward': ['mean', 'std']
    })
    grouped.to_csv(os.path.join(save_dir, 'baseline_comparison_statistics.csv'))
    print(f"✅ Statistics saved to {save_dir}/baseline_comparison_statistics.csv")

    # 绘图：每档规模分别画一张对比图
    for name in df['scale'].unique():
        sdf = df[df['scale'] == name]
        methods = sdf['method'].unique()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Baseline Comparison ({name})', fontsize=16, weight='bold')

        # 成功率
        ax = axes[0, 0]
        means = sdf.groupby('method')['success_rate'].mean()
        stds = sdf.groupby('method')['success_rate'].std()
        ax.bar(methods, means.loc[methods].values, yerr=stds.loc[methods].values, capsize=5)
        ax.set_ylabel('Success Rate')
        ax.set_ylim([0, 1.1])
        ax.set_title('Success Rate')
        ax.tick_params(axis='x', rotation=30)

        # Makespan
        ax = axes[0, 1]
        means = sdf.groupby('method')['makespan'].mean()
        stds = sdf.groupby('method')['makespan'].std()
        ax.bar(methods, means.loc[methods].values, yerr=stds.loc[methods].values, capsize=5, color='#FF6B6B', alpha=0.8)
        ax.set_ylabel('Makespan')
        ax.set_title('Makespan')
        ax.tick_params(axis='x', rotation=30)

        # 推理时间
        ax = axes[1, 0]
        means = sdf.groupby('method')['inference_time'].mean()
        stds = sdf.groupby('method')['inference_time'].std()
        ax.bar(methods, means.loc[methods].values, yerr=stds.loc[methods].values, capsize=5, color='#45B7D1', alpha=0.8)
        ax.set_ylabel('Inference Time (s)')
        ax.set_title('Computational Efficiency')
        ax.tick_params(axis='x', rotation=30)

        # 奖励
        ax = axes[1, 1]
        means = sdf.groupby('method')['reward'].mean()
        stds = sdf.groupby('method')['reward'].std()
        ax.bar(methods, means.loc[methods].values, yerr=stds.loc[methods].values, capsize=5, color='#98D8C8', alpha=0.8)
        ax.set_ylabel('Reward')
        ax.set_title('Overall Reward')
        ax.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'baseline_comparison_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot saved: {save_dir}/baseline_comparison_{name}.png")

    return df


def main():
    # 每档规模跑的环境数量（与 test.py 的 num_trials 相同=10）
    num_envs_per_scale = 10
    device = 'cuda'  # 或 'cpu'

    print("\n=== Baseline Comparison (same scales as test.py) ===")
    df = compare_methods_same_scales(num_envs_per_scale=num_envs_per_scale, device=device, save_dir='./comparison_results')

    if df is None or df.empty:
        print("❌ No results produced.")
        return

    # 控制台打印摘要
    print("\n=== Summary (by scale & method) ===")
    summary = df.groupby(['scale', 'method']).agg(
        success_rate=('success_rate', 'mean'),
        makespan=('makespan', 'mean'),
        inference_time=('inference_time', 'mean'),
        reward=('reward', 'mean'),
    )
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()