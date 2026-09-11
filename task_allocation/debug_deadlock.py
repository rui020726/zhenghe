import torch
import numpy as np
import os
from attention import AutoregressiveCentralNet
from parameters import *
from env.task_env import TaskEnv
from driver import TaskPoolManager, generate_comprehensive_mask_with_pool
from torch.distributions import Categorical
import warnings
import json
from datetime import datetime
import sys
from io import StringIO

warnings.filterwarnings('ignore', category=RuntimeWarning)


class DeadlockDebugger:
    def __init__(self, model_path, device='cuda'):
        """初始化调试器"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"Loading model from {model_path}...")
        self.model = AutoregressiveCentralNet(
            TrainParams.AGENT_INPUT_DIM,
            TrainParams.TASK_INPUT_DIM,
            TrainParams.EMBEDDING_DIM
        ).to(self.device)

        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint['best_model'])
        self.model.eval()

        print(f"✅ Model loaded successfully!")

    def print_separator(self, title="", char="="):
        """打印分隔符"""
        print(f"\n{char * 80}")
        if title:
            print(f"{title}")
            print(f"{char * 80}")

    def debug_agent_state(self, env, agent_id):
        """调试单个智能体的状态"""
        agent = env.agent_dic[agent_id]
        print(f"\n  Agent {agent_id}:")
        print(f"    Current Task: {agent['current_task']}")
        print(f"    Next Decision: {agent['next_decision']:.2f}")
        print(f"    No Choice: {agent.get('no_choice', False)}")
        print(f"    Abilities: {agent['abilities']}")

    def debug_task_state(self, env, task_id):
        """调试单个任务的状态"""
        task = env.task_dic[task_id]
        print(f"\n  Task {task_id}:")
        print(f"    Finished: {task['finished']}")
        print(f"    Feasible Assignment: {task['feasible_assignment']}")
        print(f"    Requirements: {task['requirements']}")
        print(f"    Status (remaining): {task['status']}")
        print(f"    Priority: {task.get('search_priority', 0.5):.2f}")

    def debug_task_pool(self, task_pool_manager):
        """调试任务池状态"""
        print(f"\n  Task Pool Status:")
        print(f"    Active Tasks: {len(task_pool_manager.get_active_task_ids())}/5")
        print(f"    Active Task IDs: {task_pool_manager.get_active_task_ids()}")

        for task_id in task_pool_manager.get_active_task_ids():
            remaining = task_pool_manager.get_remaining_needs(task_id)
            assigned_agents = task_pool_manager.get_assigned_agents(task_id)
            print(f"      Task {task_id}:")
            print(f"        Remaining: {remaining}")
            print(f"        Assigned Agents: {assigned_agents}")

    def debug_mask(self, env, agent_id, task_pool_manager, max_tasks):
        """调试掩码生成"""
        mask = generate_comprehensive_mask_with_pool(
            env,
            agent_id,
            task_pool_manager,
            env.current_time,
            env.tasks_num,
            max_tasks
        )

        print(f"\n  Mask for Agent {agent_id}:")
        print(f"    Depot (0) blocked: {mask[0].item()}")

        available_tasks = []
        blocked_tasks = []

        for task_id in range(env.tasks_num):
            if mask[task_id + 1]:
                blocked_tasks.append(task_id)
            else:
                available_tasks.append(task_id)

        print(f"    Available Tasks ({len(available_tasks)}): {available_tasks[:10]}")
        print(f"    Blocked Tasks ({len(blocked_tasks)}): {blocked_tasks[:10]}")

        if len(available_tasks) == 0 and mask[0]:
            print(f"    ⚠️  ALL OPTIONS BLOCKED!")

        return mask

    def debug_next_decision_times(self, env):
        """调试所有智能体的下次决策时间"""
        print(f"\n  All Agent Next Decision Times:")

        decision_times = {}
        for agent_id, agent in env.agent_dic.items():
            next_time = agent['next_decision']
            if next_time not in decision_times:
                decision_times[next_time] = []
            decision_times[next_time].append(agent_id)

        sorted_times = sorted(decision_times.keys())
        print(f"    Current Time: {env.current_time:.2f}")
        print(f"    Unique Decision Times: {len(sorted_times)}")

        for i, time in enumerate(sorted_times[:10]):
            agents = decision_times[time]
            print(f"    {i + 1}. Time {time:.2f}: {len(agents)} agents {agents[:5]}")

        if len(sorted_times) > 0:
            next_time = sorted_times[0]
            time_jump = next_time - env.current_time
            print(f"\n    Next Decision Time: {next_time:.2f}")
            print(f"    Time Jump: {time_jump:.2f}")

            if time_jump > 50:
                print(f"    ⚠️  LARGE TIME JUMP DETECTED!")

    def test_single_episode(self, env, max_steps=200, verbose=False,
                            stop_on_deadlock=False, task_pool_manager=None):
        """测试单个回合，🔥 完全按照 test.py 的逻辑"""
        if task_pool_manager is None:
            task_pool_manager = TaskPoolManager(max_active_tasks=5, max_waiting_time=EnvParams.MAX_TIME)

        max_agents = EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1]
        max_tasks = EnvParams.TASKS_RANGE[1] + 1
        current_action_index = 0
        max_waiting = True  # 🔥 与 test.py 保持一致

        time_jump_counter = 0
        last_env_time = -1
        stuck_time_counter = 0
        env_warning_count = 0

        deadlock_detected = False
        deadlock_step = -1
        deadlock_reason = ""

        # 捕获环境的标准输出
        original_stdout = sys.stdout
        captured_output = StringIO()

        with torch.no_grad():
            while (not env.finished and env.current_time < EnvParams.MAX_TIME and current_action_index < max_steps):
                # 🔥 临时重定向stdout来捕获环境警告
                sys.stdout = captured_output

                try:
                    release_agents, next_decision_time = env.next_decision()
                    # 🔥 修复1：防止时间回退
                    if next_decision_time < env.current_time:
                        print(
                            f"  ⚠️  [Time Error] next_decision_time ({next_decision_time:. 2f}) < current_time ({env.current_time:.2f})")

                        # 找出所有 next_decision < current_time 的智能体并修正
                        for agent_id, agent in env.agent_dic.items():
                            if agent['next_decision'] < env.current_time:
                                print(
                                    f"     Fixing Agent {agent_id}: {agent['next_decision']:.2f} → {env.current_time:.2f}")
                                agent['next_decision'] = env.current_time

                        # 重新获取 next_decision_time
                        release_agents, next_decision_time = env.next_decision()

                    # 🔥 修复2：如果仍然小于当前时间，强制设为当前时间
                    if next_decision_time < env.current_time:
                        next_decision_time = env.current_time

                    env.current_time = next_decision_time


                finally:
                    sys.stdout = original_stdout

                # 🔥 检查捕获的输出中是否有警告
                output = captured_output.getvalue()
                warning_count_in_step = output.count("Time jump too large")

                if warning_count_in_step > 0:
                    env_warning_count += warning_count_in_step
                    if verbose and env_warning_count <= 3:
                        print(
                            f"  ⚠️  Step {current_action_index}: Environment warning detected ({env_warning_count} total)")

                # 清空缓冲区
                captured_output.truncate(0)
                captured_output.seek(0)

                time_jump = next_decision_time - env.current_time

                # 🔥 检测1：时间跳跃过大
                if time_jump > 50:
                    time_jump_counter += 1
                    if verbose:
                        print(
                            f"  ⚠️  Large time jump at step {current_action_index}: {env.current_time:.2f} → {next_decision_time:.2f}")

                    if time_jump_counter >= 3:
                        deadlock_detected = True
                        deadlock_step = current_action_index
                        deadlock_reason = f"Large time jump (>{time_jump:. 2f})"
                        if verbose:
                            print(f"  ❌ Deadlock: {deadlock_reason}")
                        if stop_on_deadlock:
                            break
                else:
                    time_jump_counter = 0

                # 🔥 检测2：时间卡死不动
                if abs(env.current_time - last_env_time) < 0.01:
                    stuck_time_counter += 1

                    if stuck_time_counter >= 5:
                        deadlock_detected = True
                        deadlock_step = current_action_index
                        deadlock_reason = f"Time stuck at {env.current_time:.2f} for {stuck_time_counter} steps"
                        if verbose:
                            print(f"  ❌ Deadlock: {deadlock_reason}")
                        if stop_on_deadlock:
                            break
                else:
                    stuck_time_counter = 0

                # 🔥 检测3：环境警告过多
                if env_warning_count >= 10:
                    deadlock_detected = True
                    deadlock_step = current_action_index
                    deadlock_reason = f"Too many environment warnings ({env_warning_count})"
                    if verbose:
                        print(f"  ❌ Deadlock: {deadlock_reason}")
                    if stop_on_deadlock:
                        break

                last_env_time = env.current_time
                env.current_time = next_decision_time

                # 获取待决策的智能体
                agent_ids = []
                for group in release_agents:
                    agent_ids.extend(group)

                if not agent_ids:
                    break

                # 🔥 与 test.py 完全一致的有效智能体筛选逻辑
                valid_agent_ids = []
                for agent_id in agent_ids:
                    agent = env.agent_dic[agent_id]
                    _, _, mask_check = env.agent_observe(agent_id, max_waiting)
                    block_flag = mask_check[0, 1:].all().item()
                    feasible_matrix = env.get_matrix(env.task_dic, 'feasible_assignment')

                    # 🔥 添加与 test.py 一致的检查逻辑
                    if block_flag and not np.all(feasible_matrix):
                        agent['no_choice'] = True
                        continue
                    elif block_flag and np.all(feasible_matrix) and agent['current_task'] < 0:
                        continue

                    valid_agent_ids.append(agent_id)

                if not valid_agent_ids:
                    continue

                # 获取全局状态
                global_agents, global_tasks = env.get_global_state()
                global_agents_tensor = torch.tensor(global_agents, dtype=torch.float32)
                global_tasks_tensor = torch.tensor(global_tasks, dtype=torch.float32)

                # Padding
                if global_agents_tensor.shape[0] < max_agents:
                    pad = torch.zeros(max_agents - global_agents_tensor.shape[0], 13)
                    global_agents_tensor = torch.cat([global_agents_tensor, pad], dim=0)

                if global_tasks_tensor.shape[0] < max_tasks:
                    pad = torch.zeros(max_tasks - global_tasks_tensor.shape[0], 18)
                    global_tasks_tensor = torch.cat([global_tasks_tensor, pad], dim=0)

                global_agents_tensor = global_agents_tensor.unsqueeze(0).to(self.device)
                global_tasks_tensor = global_tasks_tensor.unsqueeze(0).to(self.device)

                # 编码
                task_encoded, agent_encoded = self.model.encode(global_tasks_tensor, global_agents_tensor)

                # 🔥 自回归解码：逐个智能体选择动作并立即更新任务池
                allocations_final = []

                for agent_id in valid_agent_ids:
                    # 生成基于任务池的掩码
                    comprehensive_mask = generate_comprehensive_mask_with_pool(
                        env, agent_id, task_pool_manager,
                        env.current_time, env.tasks_num, max_tasks
                    ).unsqueeze(0).to(self.device)

                    # 任务分配标志
                    task_assigned_flags = torch.zeros(1, max_tasks, device=self.device)
                    for task_id in task_pool_manager.get_active_task_ids():
                        remaining = task_pool_manager.get_remaining_needs(task_id)
                        if remaining is not None:
                            original = env.task_dic[task_id]['requirements']
                            completion_ratio = 1.
                            0 - (remaining.sum() / (original.sum() + 1e-6))
                            task_assigned_flags[0, task_id + 1] = completion_ratio

                    agent_idx_tensor = torch.tensor([agent_id], dtype=torch.long, device=self.device)

                    probs, _ = self.model.decode_step(
                        agent_idx_tensor, task_encoded, agent_encoded,
                        task_assigned_flags, comprehensive_mask
                    )

                    # 🔥 检查是否所有任务都被屏蔽（与 test.py 一致）
                    if comprehensive_mask[0].all():
                        # 如果所有任务被屏蔽且仓库未被屏蔽，选择仓库
                        if not comprehensive_mask[0, 0]:
                            action = torch.tensor([0])
                        else:
                            # 所有选项都被屏蔽，跳过该智能体
                            continue
                    else:
                        dist = Categorical(probs)
                        action = dist.sample()[0]

                    allocations_final.append(action.item())

                    # 更新任务池
                    if action.item() > 0 and action.item() <= env.tasks_num:
                        task_idx = action.item() - 1
                        if not task_pool_manager.is_task_active(task_idx):
                            if task_pool_manager.can_open_new_task():
                                task_pool_manager.open_task(task_idx, env.task_dic[task_idx]['status'].copy())

                        try:
                            arrival_time = env.get_arrival_time(agent_id, task_idx)
                        except:
                            arrival_time = env.current_time

                        try:
                            task_pool_manager.assign_agent_to_task(
                                task_idx, agent_id, env.agent_dic[agent_id]['abilities'],
                                arrival_time, env.current_time
                            )
                        except:
                            pass

                # 执行动作
                for i_agent, agent_id in enumerate(valid_agent_ids[:len(allocations_final)]):
                    task_id = allocations_final[i_agent]
                    actual_task_id = task_id if task_id <= env.tasks_num else 0

                    try:
                        env.agent_step(agent_id, actual_task_id, 0)
                    except:
                        pass

                current_action_index += 1

        # 恢复stdout
        sys.stdout = original_stdout

        # 统计结果
        finished_tasks = sum(1 for t in env.task_dic.values() if t['finished'])
        total_tasks = env.tasks_num
        success_rate = finished_tasks / total_tasks if total_tasks > 0 else 0.
        0

        return {
            'deadlock': deadlock_detected,
            'deadlock_step': deadlock_step,
            'deadlock_reason': deadlock_reason,
            'env_warning_count': env_warning_count,
            'final_time': env.current_time,
            'finished_tasks': finished_tasks,
            'total_tasks': total_tasks,
            'success_rate': success_rate,
            'total_steps': current_action_index,
            'env': env if stop_on_deadlock else None,
            'task_pool_manager': task_pool_manager if stop_on_deadlock else None
        }

    def test_multiple_episodes_auto_stop(self, num_episodes=10, start_seed=0):
        """测试多个回合，遇到死锁自动停止并调试"""
        self.print_separator(f"🔬 AUTO-STOP ON DEADLOCK MODE ({num_episodes} episodes max)", "=")

        for episode in range(num_episodes):
            seed = start_seed + episode

            env = TaskEnv(
                prio_range=(0.3, 1.0),
                per_species_range=(4, 6),
                species_range=(4, 6),
                tasks_range=(20, 30),
                traits_dim=5,
                max_task_size=2,
                duration_scale=5,
                seed=seed
            )
            env.init_state()

            print(f"\n{'─' * 80}")
            print(f"Testing Episode {episode + 1}/{num_episodes} (seed={seed})")
            print(f"  Tasks: {env.tasks_num}, Agents: {env.agents_num}")

            task_pool_manager = TaskPoolManager(max_active_tasks=5, max_waiting_time=EnvParams.MAX_TIME)

            result = self.test_single_episode(
                env,
                max_steps=200,
                verbose=True,
                stop_on_deadlock=True,
                task_pool_manager=task_pool_manager
            )

            if result['deadlock']:
                print(f"\n{'🔴' * 40}")
                print(f"❌ DEADLOCK DETECTED in Episode {episode + 1}!")
                print(f"   Seed: {seed}")
                print(f"   Step: {result['deadlock_step']}")
                print(f"   Reason: {result['deadlock_reason']}")
                print(f"   Environment Warnings: {result['env_warning_count']}")
                print(f"   Time: {result['final_time']:.2f}")
                print(
                    f"   Success Rate: {result['success_rate']:.1%} ({result['finished_tasks']}/{result['total_tasks']})")
                print(f"{'🔴' * 40}")

                max_tasks = EnvParams.TASKS_RANGE[1] + 1
                self.deep_debug_deadlock(result['env'], result['task_pool_manager'], max_tasks)

                print(f"\n{'─' * 80}")
                choice = input("Continue to next episode? (y/n) [default=n]: ").strip().lower() or 'n'
                if choice != 'y':
                    print(f"\n✅ Stopped at Episode {episode + 1}")
                    print(f"   You can rerun with seed={seed} for detailed analysis")
                    return

            else:
                print(f"  ✅ Episode {episode + 1} completed normally")
                print(f"     Steps: {result['total_steps']}, Time: {result['final_time']:.2f}")
                print(f"     Success Rate: {result['success_rate']:.1%}")

        print(f"\n✅ All {num_episodes} episodes completed!")

    def test_multiple_episodes(self, num_episodes=10, start_seed=0, save_log=True):
        """测试多个回合并统计"""
        self.print_separator(f"🔬 MULTIPLE EPISODES TEST ({num_episodes} episodes)", "=")

        results = []
        deadlock_count = 0
        deadlock_episodes = []

        for episode in range(num_episodes):
            seed = start_seed + episode

            env = TaskEnv(
                prio_range=(0.3, 1.0),
                per_species_range=(4, 6),
                species_range=(4, 6),
                tasks_range=(20, 30),
                traits_dim=5,
                max_task_size=np.random.randint(2, 5),
                duration_scale=np.random.uniform(3, 6),
                seed=seed
            )
            env.init_state()

            print(f"\n{'─' * 80}")
            print(f"Episode {episode + 1}/{num_episodes} (seed={seed})")
            print(f"  Tasks: {env.tasks_num}, Agents: {env.agents_num}")

            result = self.test_single_episode(env, max_steps=200, verbose=False)
            result['episode'] = episode + 1
            result['seed'] = seed
            result['num_tasks'] = env.tasks_num
            result['num_agents'] = env.agents_num

            status = "❌ DEADLOCK" if result['deadlock'] else "✅ COMPLETED"
            print(f"  Status: {status}")
            print(f"  Steps: {result['total_steps']}, Time: {result['final_time']:.2f}")
            print(f"  Success Rate: {result['success_rate']:. 1%} ({result['finished_tasks']}/{result['total_tasks']})")

            if result['deadlock']:
                print(f"  Deadlock at step: {result['deadlock_step']}")
                print(f"  Reason: {result['deadlock_reason']}")
                deadlock_count += 1
                deadlock_episodes.append(episode + 1)

            results.append(result)

        self.print_separator("📊 SUMMARY", "=")
        print(f"\nTotal Episodes: {num_episodes}")
        print(f"Deadlock Episodes: {deadlock_count} ({deadlock_count / num_episodes * 100:.1f}%)")
        print(f"Normal Episodes: {num_episodes - deadlock_count}")

        if deadlock_episodes:
            print(f"\nDeadlock occurred in episodes: {deadlock_episodes}")
            print(f"Corresponding seeds: {[results[e - 1]['seed'] for e in deadlock_episodes]}")

        avg_success_rate = np.mean([r['success_rate'] for r in results])
        avg_steps = np.mean([r['total_steps'] for r in results])
        avg_time = np.mean([r['final_time'] for r in results])

        print(f"\nAverage Metrics:")
        print(f"  Success Rate: {avg_success_rate:.1%}")
        print(f"  Steps: {avg_steps:.1f}")
        print(f"  Time: {avg_time:. 2f}")

        if save_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"deadlock_test_{timestamp}.json"

            with open(log_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total_episodes': num_episodes,
                        'deadlock_count': deadlock_count,
                        'deadlock_rate': deadlock_count / num_episodes
                    },
                    'episodes': results
                }, f, indent=2)

            print(f"\n✅ Log saved to: {log_file}")

        return results

    def deep_debug_deadlock(self, env, task_pool_manager, max_tasks):
        """深度调试死锁状态"""
        self.print_separator("🔥 DEEP DEBUG: DEADLOCK STATE", "=")

        print(f"\n📊 Environment State:")
        print(f"  Current Time: {env.current_time:.2f}")
        print(f"  Finished: {env.finished}")
        print(f"  Total Tasks: {env.tasks_num}")
        print(f"  Total Agents: {env.agents_num}")

        finished_tasks = sum(1 for t in env.task_dic.values() if t['finished'])
        print(f"  Finished Tasks: {finished_tasks}/{env.tasks_num}")

        self.debug_task_pool(task_pool_manager)

        self.print_separator("📋 Sample Tasks (first 5)", "-")
        for task_id in range(min(5, env.tasks_num)):
            self.debug_task_state(env, task_id)

        self.print_separator("👥 Sample Agents (first 5)", "-")
        for agent_id in range(min(5, env.agents_num)):
            self.debug_agent_state(env, agent_id)

        self.debug_next_decision_times(env)


def main():
    """主调试函数"""
    MODEL_PATH = './model/save_7/checkpoint.pth'

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    debugger = DeadlockDebugger(MODEL_PATH, device='cuda')

    print("\n" + "=" * 80)
    print("🐛 DEADLOCK DEBUGGER")
    print("=" * 80)
    print("\nOptions:")
    print("  1.   Multiple episodes - AUTO-STOP on first deadlock (⭐ RECOMMENDED)")
    print("  2.  Multiple episodes - Test all and summarize")
    print("  3. Single episode - Debug specific seed")
    print("=" * 80)

    mode = input("\nSelect mode (1/2/3) [default=1]: ").strip() or '1'

    if mode == '1':
        num_episodes = input("Max episodes to test [default=50]: ").strip()
        num_episodes = int(num_episodes) if num_episodes else 50

        start_seed = input("Starting seed [default=0]: ").strip()
        start_seed = int(start_seed) if start_seed else 0

        debugger.test_multiple_episodes_auto_stop(num_episodes=num_episodes, start_seed=start_seed)

    elif mode == '2':
        num_episodes = input("Number of episodes [default=20]: ").strip()
        num_episodes = int(num_episodes) if num_episodes else 20

        start_seed = input("Starting seed [default=0]: ").strip()
        start_seed = int(start_seed) if start_seed else 0

        debugger.test_multiple_episodes(num_episodes=num_episodes, start_seed=start_seed)

    else:
        seed = input("Enter seed to debug: ").strip()
        if not seed:
            print("❌ Seed is required!")
            return
        seed = int(seed)

        env = TaskEnv(
            prio_range=(0.3, 1.0),
            per_species_range=(4, 6),
            species_range=(4, 6),
            tasks_range=(20, 30),
            traits_dim=5,
            max_task_size=np.random.randint(2, 5),
            duration_scale=np.random.uniform(3, 6),
            seed=seed
        )
        env.init_state()

        print(f"\n✅ Environment created (seed={seed})")

        task_pool_manager = TaskPoolManager(max_active_tasks=5, max_waiting_time=EnvParams.MAX_TIME)
        result = debugger.test_single_episode(
            env,
            max_steps=200,
            verbose=True,
            stop_on_deadlock=True,
            task_pool_manager=task_pool_manager
        )

        if result['deadlock']:
            max_tasks = EnvParams.TASKS_RANGE[1] + 1
            debugger.deep_debug_deadlock(result['env'], result['task_pool_manager'], max_tasks)
        else:
            print(f"\n✅ Episode completed without deadlock")
            print(f"   Success Rate: {result['success_rate']:.1%}")

    print("\n✅ Debug completed!")


if __name__ == "__main__":
    main()