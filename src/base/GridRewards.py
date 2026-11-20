from src.base.GridActions import GridActions


class GridRewardParams:
    def __init__(self):
        self.boundary_penalty = 1.0  # 边界惩罚
        self.empty_battery_penalty = 150.0  # 没电惩罚
        self.movement_penalty = 0.2  # 移动惩罚


class GridRewards:
    def __init__(self, stats):
        self.params = GridRewardParams()
        self.cumulative_reward: float = 0.0

        stats.add_log_data_callback('cumulative_reward', self.get_cumulative_reward)

    def get_cumulative_reward(self):  # 返回当前的累积奖励
        return self.cumulative_reward

    def calculate_motion_rewards(self, state, action: GridActions, next_state):
        reward = 0.0
        if not next_state.landed:
            # 移动惩罚
            reward -= self.params.movement_penalty

        # 如果代理的当前位置与下一个状态的位置相同，且它没有降落，且没有执行悬停动作
        # 代理将受到边界惩罚
        if state.position == next_state.position and not next_state.landed and not action == GridActions.HOVER:
            reward -= self.params.boundary_penalty

        # 惩罚电池耗尽
        if next_state.movement_budget == 0 and not next_state.landed:
            reward -= self.params.empty_battery_penalty

        return reward

    def reset(self):     # 重置累积奖励
        self.cumulative_reward = 0
