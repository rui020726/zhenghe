# -*- coding: utf-8 -*-
"""
简化调试脚本
"""
from env.task_env import TaskEnv
import numpy as np
import time

env = TaskEnv(
    tasks_range=(10, 15),  # 减少任务数
    per_species_range=(2, 3),  # 减少智能体数
    comm_range=0.5,
    seed=42
)

for ep in range(5):
    print(f"\n{'=' * 50}")
    print(f"Episode {ep}")
    print('=' * 50)

    env.reset(seed=ep)
    step = 0

    while not env.finished and step < 50:
        env.update_communication_graph()

        ready_ids, next_time = env.next_decision()
        all_ids = ready_ids[0] + ready_ids[1]

        print(f"Step {step}:  current_time={env.current_time:.2f}, next_time={next_time:.2f}, agents={len(all_ids)}")

        if not all_ids:
            env.current_time = next_time
            step += 1
            time.sleep(0.01)  # 防止CPU占满
            continue

        env.current_time = next_time

        for aid in all_ids:
            s, n, t, m = env.get_local_obs_with_propagation(aid, max_tasks=35)

            # 🔥 修复：转换为布尔类型
            m_bool = m.astype(bool)

            if m_bool.all():
                m_bool[0] = False

            # 随机选择动作
            valid_actions = np.where(~m_bool)[0]
            if len(valid_actions) > 0:
                action = int(np.random.choice(valid_actions))
                try:
                    reward, _, _ = env.agent_step(aid, action, step)
                    print(f"  Agent {aid} -> Action {action}, reward={reward:.2f}")
                except Exception as e:
                    print(f"  ❌ Agent {aid} error: {e}")
            else:
                print(f"  ⚠️ Agent {aid} has no valid actions!")

        env.finished = env.check_finished()
        step += 1

        # 超时保护
        if step >= 50:
            print("⚠️ Max steps reached, breaking...")
            break

    # 在每个 episode 结束时
    try:
        reward, finished = env.get_episode_reward()
        if len(finished) > 0:
            success_rate = sum(finished) / len(finished)
        else:
            success_rate = 0.0

        # 🔥 使用 f-string 而不是 % 格式化
        print(f"Episode finished: reward={reward:.2f}, success={success_rate * 100:.1f}%")

    except Exception as e:
        print(f"Error calculating reward: {e}")
        import traceback

        traceback.print_exc()

print("\n✅ Debug script completed!")