import torch
import numpy as np
import ray
import os
# 1. 替换为集中式网络（关键）
from attention import CentralAttentionNet
# 2. 后续需替换为集中式 Worker（此处先定义适配接口，下文附 Worker 修改建议）
from worker import CentralWorker
from parameters import *
from env.task_env import TaskEnv


class Runner(object):
    """集中式 Runner：适配全局决策与搜救优先级，负责单智能体（Meta Agent）的训练/测试交互"""

    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        # 设备配置（与 Driver 对齐，支持 GPU/CPU）
        self.device = torch.device('cuda') if (TrainParams.USE_GPU and torch.cuda.is_available()) else torch.device(
            'cpu')

        # 3. 初始化集中式网络（替换原 AttentionNet）
        self.localNetwork = CentralAttentionNet(
            TrainParams.AGENT_INPUT_DIM,
            TrainParams.TASK_INPUT_DIM,
            TrainParams.EMBEDDING_DIM
        ).to(self.device)
        # 基线网络也用集中式结构（评估优势函数用）
        self.localBaseline = CentralAttentionNet(
            TrainParams.AGENT_INPUT_DIM,
            TrainParams.TASK_INPUT_DIM,
            TrainParams.EMBEDDING_DIM
        ).to(self.device)

    def get_weights(self):
        """获取集中式网络权重（与 Driver 权重同步用）"""
        return self.localNetwork.state_dict()

    def set_weights(self, weights):
        """加载全局网络权重（从 Driver 同步）"""
        self.localNetwork.load_state_dict(weights)

    def set_baseline_weights(self, weights):
        """加载基线网络权重（评估优势函数用）"""
        self.localBaseline.load_state_dict(weights)

    def training(self, global_weights, baseline_weights, curr_episode, env_params):
        """
        集中式训练：单 Meta Agent 处理一个环境的全局交互
        :param env_params: 环境参数（含搜救优先级范围，从 Driver 传递）
        """
        print(f"Starting episode {curr_episode} on metaAgent {self.metaAgentID}")
        # 同步全局权重到本地集中式网络
        self.set_weights(global_weights)
        self.set_baseline_weights(baseline_weights)

        # 4. 控制是否保存动画（与原逻辑一致，但适配集中式环境）
        save_img = False
        if SaverParams.SAVE_IMG and (curr_episode % SaverParams.SAVE_IMG_GAP == 0):
            save_img = True

        # 5. 初始化集中式 Worker（核心：替换原 Worker，处理全局决策）
        worker = CentralWorker(
            meta_agent_id=self.metaAgentID,
            central_net=self.localNetwork,  # 传入集中式网络
            baseline_net=self.localBaseline,
            episode=curr_episode,
            device=self.device,
            save_image=save_img,
            env_params=env_params  # 传递含搜救优先级的环境参数
        )

        # 6. 执行集中式环境交互（Worker 内实现全局观察→分配→奖励计算）
        worker.work(curr_episode)

        # 7. 收集集中式经验与性能指标（经验含全局状态、分配结果、优先级奖励）
        buffer = worker.experience  # 集中式经验：全局智能体/任务观察、分配动作、加权奖励等
        perf_metrics = worker.perf_metrics  # 含高优先级任务完成率等新增指标

        info = {
            "id": self.metaAgentID,
            "episode_number": curr_episode,
        }

        return buffer, perf_metrics, info

    def testing(self, seed=None, env_params=None):
        """
        集中式测试：无探索，贪心决策，评估模型性能
        :param env_params: 测试环境参数（含搜救优先级配置）
        """
        # 初始化测试用集中式 Worker（无探索，固定种子）
        worker = CentralWorker(
            meta_agent_id=self.metaAgentID,
            central_net=self.localNetwork,
            baseline_net=self.localBaseline,
            episode=0,
            device=self.device,
            save_img=False,
            seed=seed,  # 固定测试种子，保证结果可复现
            env_params=env_params,
            is_test=True  # 标记为测试模式（禁用采样，用贪心分配）
        )

        # 执行测试并返回关键指标（奖励、高优先级完成率等）
        total_reward, high_prio_success = worker.baseline_test()
        return total_reward, high_prio_success, seed, self.metaAgentID


# 7. Ray 远程封装（调整 GPU 分配逻辑，避免小数分配报错）
@ray.remote(
    num_cpus=1,
    num_gpus=1
)

class CentralRLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)


if __name__ == '__main__':
    # 本地测试：验证集中式 Runner 初始化与远程调用
    ray.init(ignore_reinit_error=True)
    # 测试用环境参数（含搜救优先级范围）
    test_env_params = {
        "per_species_agents": (3, 3),
        "species_num": (5, 5),
        "tasks_num": (20, 20),
        "prio_range": (0.3, 1.0)  # 搜救优先级：0.3~1.0
    }
    runner = CentralRLRunner.remote(0)
    # 测试训练接口
    dummy_weights = CentralAttentionNet(
        TrainParams.AGENT_INPUT_DIM,
        TrainParams.TASK_INPUT_DIM,
        TrainParams.EMBEDDING_DIM
    ).state_dict()
    job_id = runner.training.remote(dummy_weights, dummy_weights, 1, test_env_params)
    out = ray.get(job_id)
    print(f"Test Output: Episode {out[2]['episode_number']} finished")
    ray.shutdown()