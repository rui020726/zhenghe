import copy
import tqdm
import distutils.util

from src.ModelStats import ModelStatsParams, ModelStats
from src.base.BaseDisplay import BaseDisplay
from src.base.GridActions import GridActions


class BaseEnvironmentParams:
    def __init__(self):
        self.model_stats_params = ModelStatsParams()


class BaseEnvironment:
    def __init__(self, params: BaseEnvironmentParams, display: BaseDisplay):
        self.stats = ModelStats(params.model_stats_params, display=display)
        self.trainer = None
        self.agent = None
        self.grid = None
        self.rewards = None
        self.physics = None
        self.display = display
        self.episode_count = 0
        self.step_count = 0

    # 经验填充
    def fill_replay_memory(self):

        while self.trainer.should_fill_replay_memory():

            state = copy.deepcopy(self.init_episode())
            while not state.terminal:
                next_state = self.step(state, random=self.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)

    # 训练一个完整的回合
    def train_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)  # 在回合开始时记录统计数据。
        while not state.is_terminal():   # 只要状态不是终止状态，就执行训练步骤。
            state = self.step(state)
            self.trainer.train_agent()  # 训练代理，通常涉及强化学习算法的更新。

        self.stats.on_episode_end(self.episode_count)    # 在回合结束时记录统计数据。
        self.stats.log_training_data(step=self.step_count)   # 记录训练数据。

        self.episode_count += 1

    # 启动训练过程
    def run(self):

        self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:  # 当训练步骤数小于最大步数时，执行训练过程。
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            # 每经过一定数量的回合进行一次测试。
            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode()

            self.stats.save_if_best()    # 如果当前训练结果是最好的，则保存模型

        self.stats.training_ended()

    def step(self, state, random=False):
        if random:
            action = self.agent.get_random_action()
        else:
            action = self.agent.act(state)
        next_state = self.physics.step(GridActions(action))  # 更新状态
        reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
        self.trainer.add_experience(state, action, reward, next_state)
        self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))
        self.step_count += 1
        return copy.deepcopy(next_state)
        # 返回 next_state，并对其进行深拷贝。这样做是为了防止对原始状态对象的修改影响到后续的状态更新。

    # 测试
    def test_episode(self, scenario=None):
        state = copy.deepcopy(self.init_episode(scenario))
        self.stats.on_episode_begin(self.episode_count)
        while not state.terminal:
            action = self.agent.get_exploitation_action_target(state)  # 确定性的选择动作
            next_state = self.physics.step(GridActions(action))
            reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
            self.stats.add_experience((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
            state = copy.deepcopy(next_state)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def test_scenario(self, scenario):
        state = copy.deepcopy(self.init_episode(scenario))
        while not state.terminal:
            action = self.agent.get_exploitation_action_target(state)
            state = self.physics.step(GridActions(action))

    # 初始化回合的状态
    def init_episode(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
        else:
            state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state

    # 用于评估代理在指定数量的 episodes（回合）上的表现。show 参数控制是否显示每一轮的可视化
    def eval(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.test_episode()
            self.step_count += 1  # Increase step count so that logging works properly

            # 可视化显示
            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    # 评估特定场景
    def eval_scenario(self, init_state):
        self.test_scenario(init_state)

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass
