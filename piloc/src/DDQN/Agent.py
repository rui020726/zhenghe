import configs
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import autocast
import os
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from src.DDQN.buffer import Buffer_for_PPO

from copy import deepcopy
# import pettingzoo #动态导入
import gymnasium as gym
import importlib
import argparse

## tricks
from normalization import Normalization,RewardScaling

## 其他

import time
import re
import pickle


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积模块，对于两个通道分别进行卷积
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1),  # 输入1通道，输出8通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 58x58 → 29x29
            nn.Conv2d(8, 32, kernel_size=5, stride=2),  # 29x29 → 13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13 → 6x6
        )

        # 卷积模块，对于两个通道分别进行卷积
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1),  # 输入1通道，输出8通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 58x58 → 29x29
            nn.Conv2d(8, 32, kernel_size=5, stride=2),  # 29x29 → 13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13 → 6x6
        )

        # 对信息素地图进行卷积
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1),  #
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=3, stride=1),  #
            nn.ReLU()
        )

        # 后续处理层
        self.fc1 = nn.Linear(36, 36)  # 输入36（6*6），输出32
        self.ln1 = nn.LayerNorm(36)  # 对最后一个维度（32）归一化
        self.fc2 = nn.Linear(32, 32)  # 输入32，输出36
        self.ln2 = nn.LayerNorm(32)
        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(49, 49)
        self.ln3 = nn.LayerNorm(49)
        self.fc4 = nn.Linear(32, 32)
        self.ln4 = nn.LayerNorm(32)

    def forward(self, x,pheromone_map_tensor, batch_size):
        # x 应该是形状为 [batch_size, 2, 60, 60] 的输入，即包含两个通道
        x1 = x[:, 0:1, :, :]  # 提取第一个通道（obstacle_map）
        x2 = x[:, 1:2, :, :]  # 提取第二个通道（explore_map）
        x3 = pheromone_map_tensor.unsqueeze(1)

        # 对两个通道分别进行卷积处理
        x1 = self.conv_block1(x1)  # [batch_size, 32, 6, 6]
        x2 = self.conv_block2(x2)  # [batch_size, 32, 6, 6]
        x3 = self.conv_block3(x3)  # [batch_size, 32, 7, 7]

        # 2. Reshape: [ batch_size,32, 6*6] → [ 32, 36]
        x1 = x1.reshape(batch_size, 32, 36)
        x2 = x2.reshape(batch_size, 32, 36)
        x3 = x3.reshape(batch_size, 32, 49)


        # 3. 第一个全连接层（输入32，输出32）
        x1 = self.fc1(x1.view(-1, 36))
        x1 = x1.view(batch_size, 32, 36)  # 重新调整为 [batch_size, 32, 36]
        x1 = self.ln1(x1)
        x1 = self.relu(x1)

        x2 = self.fc1(x2.view(-1, 36))
        x2 = x2.view(batch_size, 32, 36)  # 重新调整为 [batch_size, 32, 36]
        x2 = self.ln1(x2)
        x2 = self.relu(x2)

        x3 = self.fc3(x3.view(-1, 49))
        x3 = x3.view(batch_size, 32, 49)
        x3 = self.ln3(x3)
        x3 = self.relu(x3)

       # 4. 转置 → [batch_size,36, 32]
        x1 = x1.transpose(1, 2)  # 交换维度1和2
        x2 = x2.transpose(1, 2)  # 交换维度1和2
        x3 = x3.transpose(1, 2)  # 交换维度1和2

        # 5. 第二个全连接层（输入32，输出36）
        x1 = self.fc2(x1.reshape(-1, 32))  # [batch, 36, 32]
        x1 = x1.view(batch_size, 36, 32)  # 重新调整为 [batch_size, 36, 32]
        x1 = self.ln2(x1)
        x1 = self.relu(x1)
        x1 = x1.view(batch_size, -1)  # 展平后变为 [batch_size, 1152]

        x2 = self.fc2(x2.reshape(-1, 32))  # [batch, 36, 32]
        x2 = x2.view(batch_size, 36, 32)  # 重新调整为 [batch_size, 36, 32]
        x2 = self.ln2(x2)
        x2 = self.relu(x2)
        x2 = x2.view(batch_size, -1)  # 展平后变为 [batch_size, 1152]

        x3 = self.fc4(x3.reshape(-1, 32))  # [batch, 36, 32]
        x3 = x3.view(batch_size, 49, 32)  # 重新调整为 [batch_size, 36, 32]
        x3 = self.ln4(x3)
        x3 = self.relu(x3)
        x3 = x3.view(batch_size, -1)  # 展平后变为 [batch_size, 1152]

        x = torch.cat((x1, x2,x3), dim=1)  # [batch_size, 36 * 2]，即 [batch_size, 36]

        return x

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3872, 256),  # 输入3872维，输出256维
            nn.ReLU(),
            nn.Linear(256, 64),  # 输入256维，输出64维
            nn.ReLU(),
            nn.Linear(64, 4),  # 输出action_dim维
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.fc(state)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3872, 256),  # 输入3872维，输出256维
            nn.ReLU(),
            nn.Linear(256, 64),  # 输入256维，输出64维
            nn.ReLU(),
            nn.Linear(64, 1),  # 输出action_dim维
        )

    def forward(self, state):
        return self.fc(state)
"""
# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        # input: [batch_size x num_agents x input_dim]
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,
                                                                                         2)  # q_s: [batch_size x num_heads x num_agents x output_dim]
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,
                                                                                         2)  # k_s: [batch_size x num_heads x num_agents x output_dim]
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,
                                                                                         2)  # v_s: [batch_size x num_heads x num_agents x output_dim]

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        assert attn_mask.size(0) == batch_size, 'mask dim {} while batch size {}'.format(attn_mask.size(0), batch_size)

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads,
                                                             1)  # attn_mask : [batch_size x num_heads x num_agents x num_agents]
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        # context: [batch_size x num_heads x num_agents x output_dim]
        with autocast(enabled=False):
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (
                        self.output_dim ** 0.5)  # scores : [batch_size x n_heads x num_agents x num_agents]
            scores.masked_fill_(attn_mask, -1e9)  # 在掩码为 1 的位置，用指定的值（-1e9）填充自身张量的元素
            attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents,
                                                            self.num_heads * self.output_dim)  # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)

        return output  # output: [batch_size x num_agents x output_dim]


class CommBlock(nn.Module):
    def __init__(self, input_dim, output_dim=64, num_heads=configs.num_comm_heads, num_layers=configs.num_comm_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.self_attn = MultiHeadAttention(input_dim, output_dim, num_heads)

        self.update_cell = nn.GRUCell(output_dim, input_dim)

    def forward(self, latent, comm_mask):
        '''
        latent形状: batch_size x num_agents x latent_dim

        '''
        num_agents = latent.size(1)

        # 获取需要通信的智能体索引,
        # comm_mask 是形状 [batch_size, num_agents, num_agents] 的布尔矩阵
        update_mask = comm_mask.sum(dim=-1) > 1
        comm_idx = update_mask.nonzero(as_tuple=True)  # 所有 True 位置的索引

        # no agent use communication, return
        # comm_idx[0] 是 batch_size 维度的索引（即属于哪个 batch）。
        if len(comm_idx[0]) == 0:
            return latent

        if len(comm_idx) > 1:
            update_mask = update_mask.unsqueeze(2)
        # 反转comm_mask得到attn_mask，attn_mask中TURE表示不能通信
        attn_mask = comm_mask == False

        for _ in range(self.num_layers):

            info = self.self_attn(latent, attn_mask=attn_mask)
            # latent = attn_layer(latent, attn_mask=attn_mask)
            if len(comm_idx) == 1:

                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)
                # GRUCell 接收两个输入：info（计算出的新注意力特征）；latent（智能体的原始状态）
                latent[batch_idx, comm_idx[0]] = self.update_cell(info[batch_idx, comm_idx[0]],
                                                                  latent[batch_idx, comm_idx[0]])
            else:
                update_info = self.update_cell(info.view(-1, self.output_dim), latent.view(-1, self.input_dim)).view(
                    configs.batch_size, num_agents, self.input_dim)
                # update_mask = update_mask.unsqueeze(2)
                latent = torch.where(update_mask, update_info, latent)
                # latent[comm_idx] = self.update_cell(info[comm_idx], latent[comm_idx])
                # latent = self.update_cell(info, latent)

        return latent
"""
class Agent:
    def __init__(self, action_dim, actor_lr, critic_lr, device,max_comm_agents=configs.max_comm_agents,hidden_dim=configs.hidden_dim):
        super(Agent, self).__init__()

        # 将 FullModel、Actor 和 Critic 实例化
        self.device = device
        self.full_model = FullModel().to(device)
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.max_comm_agents = max_comm_agents
        # 定义优化器
        # 这里将 `FullModel` 和 `Actor`、`Critic` 的参数都包含在优化器中
        self.optimizer = optim.Adam([
            {'params': self.full_model.parameters(), 'lr': actor_lr},
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])
    def get_probs(self, state, pheromone_map_tensor, batch_size):
        """只算动作概率，不采样：不消耗 torch 采样 RNG。

        确定性验证、PPO 更新重算 log-prob 都必须走这里；采样一律由调用方
        在应用合法动作掩码后自行进行，保证训练/验证/更新三处动作分布一致。
        """
        state = state.to(self.device)
        pheromone_map_tensor = pheromone_map_tensor.to(self.device)
        feature = self.full_model(state, pheromone_map_tensor, batch_size)
        return self.actor(feature.view(batch_size, -1))

    def get_action(self, state,pheromone_map_tensor,batch_size = 256):
        """
        通过FullModel提取特征，使用Actor网络生成动作.这里的输入应该为时间戳地图
        """
        action_probs = self.get_probs(state, pheromone_map_tensor, batch_size)

        # 根据策略选择动作 (通过采样等方式)
        action = torch.multinomial(action_probs, 1).squeeze(1)  # 选择一个动作

        return action, action_probs

    def get_value(self, state,pheromone_map_tensor):
        """
               通过FullModel提取特征，使用Critic网络生成状态价值
               """
        state = state.to(self.device)  # 将输入状态移到设备上
        pheromone_map_tensor = pheromone_map_tensor.to(self.device)

        # 获取FullModel输出的特征
        feature = self.full_model(state, pheromone_map_tensor,batch_size=256)  # 传递修改后的state

        # 展平为(batch_size, 3600)
        feature = feature.view(feature.size(0), -1)  # 展平状态维度：60*60 = 3600

        # 使用Critic网络获取状态价值
        value = self.critic(feature)  # feature 传入Critic网络，输出是每个状态的价值

        return value
    def update_actor(self, loss):
        """
        更新Actor网络
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 防止梯度爆炸
        self.optimizer.step()

    def update_critic(self, loss):
        """
        更新Actor网络
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # 防止梯度爆炸
        self.optimizer.step()

    def update_ac(self, loss):
        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10)
        self.ac_optimizer.step()


# huber损失函数
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


class MAPPO:
    def __init__(self, dim_info, actor_lr, critic_lr, horizon, device, trick=None, is_continue = False,num_agent = configs.num_agents):
        self.agents = {}
        self.buffers = {}
        self.agents = Agent( 4, actor_lr, critic_lr, device,
                                      trick)
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.buffers[agent_id] = Buffer_for_PPO(horizon, obs_dim, act_dim=4 if is_continue else 1,
                                                        device=device)

        self.device = device
        self.is_continue = is_continue
        print('actor_type:continue') if self.is_continue else print('actor_type:discrete')

        self.horizon = int(horizon)

        self.trick = trick
        self.num_agents = num_agent # 智能体数量

        if self.trick['lr_decay']:
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr

    def select_action(self, obs):
        actions = {}
        action_log_pis = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue:
                mean, std = self.agents.actor(obs)
                dist = Normal(mean, std)
                action = dist.sample()
                action_log_pi = dist.log_prob(action)  # 1xaction_dim
            else:
                dist = Categorical(probs=self.agents.actor(obs))
                action = dist.sample()
                action_log_pi = dist.log_prob(action)
            # to 真实值
            actions[agent_id] = action.detach().cpu().numpy().squeeze(0)  # 1xaction_dim ->action_dim
            action_log_pis[agent_id] = action_log_pi.detach().cpu().numpy().squeeze(0)

        return actions, action_log_pis

    def evaluate_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue:
                mean, _ = self.agents.actor(obs)
                action = mean.detach().cpu().numpy().squeeze(0)
            else:
                a_prob = self.agents.actor(obs).detach().cpu().numpy().squeeze(0)
                action = np.argmax(a_prob)

            actions[agent_id] = action
        return actions

    ## buffer 相关
    def add(self, obs1,obs2, obs_pheromone,action, reward, next_obs1,next_obs2,next_obs_pheromone, done, action_log_pi, adv_dones, legal_masks):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs1[agent_id],obs2[agent_id],obs_pheromone[agent_id] ,action[agent_id],
                       reward[agent_id], next_obs1[agent_id], next_obs2[agent_id],next_obs_pheromone[agent_id],
                       done[agent_id],action_log_pi[agent_id], adv_dones[agent_id], legal_masks[agent_id])


    def all(self):
        obs1 = {}
        obs2 = {}
        obs_pheromone = {}
        action = {}
        reward = {}
        next_obs1 = {}
        next_obs2 = {}
        next_obs_pheromone = {}
        done = {}
        action_log_pi = {}
        adv_dones = {}
        legal_masks = {}
        for agent_id, buffer in self.buffers.items():
            obs1[agent_id],obs2[agent_id],obs_pheromone[agent_id], action[agent_id], reward[agent_id], next_obs1[agent_id],next_obs2[agent_id],next_obs_pheromone[agent_id],done[agent_id], action_log_pi[
                agent_id], adv_dones[agent_id], legal_masks[agent_id] = buffer.all()
        return obs1,obs2, obs_pheromone,action, reward, next_obs1,next_obs2,next_obs_pheromone, done, action_log_pi, adv_dones, legal_masks

    ## PPO算法相关
    def learn(self, minibatch_size, gamma, lmbda, clip_param, K_epochs, entropy_coefficient, huber_delta=None):
        # 多智能体特有-- 集中式训练critic:要用到所有智能体next状态和动作
        obs1,obs2, obs_pheromone,action, reward, next_obs1,next_obs2,next_obs_pheromone, done, action_log_pi, adv_dones, legal_masks = self.all()
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(self.horizon, self.num_agents)
            gae = 0
            vs = []
            vs_ = []
            for agent_id in self.buffers.keys():
                agent_obs1 = obs1[agent_id]  # 获取每个智能体的观测值
                agent_obs2 = obs2[agent_id]
                agent_obs_pheromone = obs_pheromone[agent_id]

                next_obs_agent1 = next_obs1[agent_id]
                next_obs_agent2 = next_obs2[agent_id]
                next_agent_obs_pheromone = next_obs_pheromone[agent_id]

                agent_obs = torch.stack((agent_obs1, agent_obs2), dim=1)
                next_obs_agent = torch.stack((next_obs_agent1, next_obs_agent2), dim=1)
                vs.append(self.agents.get_value(agent_obs,agent_obs_pheromone))  # batch_size x 1
                vs_.append(self.agents.get_value(next_obs_agent,next_agent_obs_pheromone))

            vs = torch.cat(vs, dim=1)  # batch_size x 3
            vs_ = torch.cat(vs_, dim=1)  # batch_size x 3

            reward = torch.cat(list(reward.values()), dim=1)  #
            done = torch.cat(list(done.values()), dim=1)
            adv_dones = torch.cat(list(adv_dones.values()), dim=1)

            td_delta = reward + gamma * (1.0 - done) * vs_ - vs  # 这里可能使用全局的reward

            for i in reversed(range(self.horizon)):
                gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i])
                adv[i] = gae

            adv = adv.to(self.device)  # batch_size x 3
            v_target = adv + vs  # batch_size x 3
            if self.trick['adv_norm']:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8))

        agent_id_list = list(self.buffers.keys())
        for agent_idx, agent_id in enumerate(agent_id_list):
            # Optimize policy for K epochs:
            for _ in range(K_epochs):
                # 随机打乱样本 并 生成小批量
                shuffled_indices = np.random.permutation(self.horizon)
                indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)]
                for index in indexes:
                    # 先更新actor
                    if self.is_continue:
                        mean, std = self.agents.actor(obs1[agent_id][index])
                        dist_now = Normal(mean, std)
                        dist_entropy = dist_now.entropy().sum(dim=1,
                                                              keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index])  # mini_batch_size x action_dim
                    else:
                        # 先更新actor
                        OBS1 = obs1[agent_id][index]
                        OBS2 = obs2[agent_id][index]
                        OBS_pheromone = obs_pheromone[agent_id][index]
                        OBS = torch.stack((OBS1, OBS2), dim=1)

                        # 重算 log-prob 必须用与采样时相同的合法动作掩码，
                        # 否则新旧策略分布不一致，PPO ratio 有偏。
                        probs = self.agents.get_probs(OBS, OBS_pheromone, batch_size=OBS.shape[0])
                        mask = legal_masks[agent_id][index]
                        masked_probs = probs * mask
                        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        dist_now = Categorical(masked_probs)
                        dist_entropy = dist_now.entropy().reshape(-1, 1)  # mini_batch_size  -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index].reshape(-1)).reshape(-1,
                                                                                                           1)  # mini_batch_size  -> mini_batch_size x 1
                    ratios = torch.exp(
                        action_log_pi_now.sum(dim=1, keepdim=True) - action_log_pi[agent_id][index].sum(dim=1,
                                                                                                        keepdim=True))  # shape(mini_batch_size X 1)
                    # 信用分配：一个 agent 的 ratio 只与它自己的 advantage 相乘。
                    # 旧实现把 (mb,1) 的 ratio 与 (mb,num_agents) 的 adv 广播相乘，
                    # 个体碰撞信号被队友的 advantage 稀释，碰撞学不到。
                    adv_own = adv[index][:, agent_idx].reshape(-1, 1)  # mini_batch_size x 1
                    surr1 = ratios * adv_own
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv_own
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean()
                    self.agents.update_actor(actor_loss)

                    # 再更新critic
                    obs_1 = {agent_id: obs1[agent_id][index] for agent_id in obs1.keys()}
                    obs_2 = {agent_id: obs2[agent_id][index] for agent_id in obs2.keys()}
                    obs_critic1 = obs_1[agent_id]
                    obs_critic2 = obs_2[agent_id]
                    obs_ph = obs_pheromone[agent_id][index]
                    obs_critic = torch.stack((obs_critic1, obs_critic2), dim=1)
                    v_s = self.agents.get_value(obs_critic,obs_ph)  # mini_batch_size x 1
                    v_s = v_s.repeat(1, self.num_agents)  # mini_batch_size x 3
                    v_target_ = v_target[index]
                    if self.trick['ValueClip']:
                        ''' 参考原mappo代码,原代码存储了return和value值,故实现上和如下有些许差异'''
                        v_target_clip = torch.clamp(v_target_, v_s - clip_param, v_s + clip_param)
                        if self.trick['huber_loss']:
                            critic_loss_clip = huber_loss(v_target_clip - v_s, huber_delta).mean()
                            critic_loss_original = huber_loss(v_target_ - v_s, huber_delta).mean()
                        else:
                            critic_loss_clip = F.mse_loss(v_target_clip, v_s)
                            critic_loss_original = F.mse_loss(v_target_, v_s)
                        critic_loss = torch.max(critic_loss_original, critic_loss_clip)
                    else:
                        if self.trick['huber_loss']:
                            critic_loss = huber_loss(v_target_ - v_s, huber_delta).mean()
                        else:
                            critic_loss = F.mse_loss(v_target_, v_s)
                    self.agents.update_critic(critic_loss)

        ## 清空buffer
        for buffer in self.buffers.values():
            buffer.clear()

    def lr_decay(self, episode_num, max_episodes):
        lr_a_now = self.actor_lr * (1 - episode_num / max_episodes)
        lr_c_now = self.critic_lr * (1 - episode_num / max_episodes)
        for agent in self.agents.values():
            for p in agent.actor_optimizer.param_groups:
                p['lr'] = lr_a_now
            for p in agent.critic_optimizer.param_groups:
                p['lr'] = lr_c_now

    def save(self, model_dir,episode_num, meta=None, trainer_state=None):
        # 保存共享的 full_model、actor、critic 的状态字典（仅需保存一次，无需区分智能体）
        os.makedirs(model_dir, exist_ok=True)
        state = {
            'full_model': self.agents.full_model.state_dict(),  # 假设 self.agents 是共享的单个 Agent 实例
            'actor': self.agents.actor.state_dict(),
            'critic': self.agents.critic.state_dict(),
            'optimizer': self.agents.optimizer.state_dict(),  # 可选：如需保存优化器状态
            'episode': episode_num,
            # 训练配置随权重走：一个 checkpoint 必须自带它是哪套奖励/折扣训出来的，
            # 否则事后无法解释，也无法安全地送去联合推理。
            'run_meta': dict(meta) if meta else None,
            # 训练状态随权重走（trainer_state.pkl 的内容合一进同一个文件）：
            # 权重与课程/RNG/global step/ buffer 分开覆盖时，恢复旧 checkpoint
            # 可能加载到更新一轮的其余状态，resume 不再与连续训练一致。
            'trainer_state': trainer_state,
        }
        torch.save(state, os.path.join(model_dir, f'shared_mappo_model_21_{episode_num}.pth'))
    # 21为修正通信丢失的模型
    # 20 获取观测时，可能会遇到噪声，观测半径减小为4.通信时可能会遇到问题
    # 19 将通信半径改为6
    # 18 将通信半径改为3
    #17 将通信半径改为4
    # shared_mappo_model_16 重新训练model14
    # shared_mappo_model_15 在10的基础上，取消掉阈值，看看效果
    # shared_mappo_model_14 训练两个；引入信息素奖励,奖励设置为 0.1*（v_ - v)/v_ + 0.1*（v_ - v) 相对加绝对,通信惩罚取消。
    # shared_mappo_model_13 训练两个；引入信息素奖励,会根据周围信息素浓度，受到浓度*0.1的惩罚。通信惩罚取消
    # shared_mappo_model_12 训练两个；引入信息素奖励,信息素奖励与浓度差相关，并且浓度差超过阈值才会获得奖励，阈值增加，阈值为2.浓度差越大，奖励越多;还引入了视野内的智能体会受到惩罚
    # shared_mappo_model_11 训练两个；去除信息素奖励;还引入了视野内的智能体会受到惩罚
    # shared_mappo_model_10 训练两个；引入信息素奖励,信息素奖励与浓度差相关，并且浓度差超过阈值才会获得奖励，阈值为0.5，浓度差越大，奖励越多;还引入了视野内的智能体会受到惩罚
    # shared_mappo_model_09 训练两个；引入信息素奖励,信息素奖励与浓度差相关，并且浓度差超过阈值才会获得奖励，阈值为0.5，浓度差越大，奖励越多
    # shared_mappo_model_08 训练两个；引入信息素奖励 使用相同的模型
    # shared_mappo_model_07 训练三个
    # shared_mappo_model_06 为了获得5000次训练的模型
    # shared_mappo_model_05 去除了信息素的奖励,去除前期探索奖励加成
    # shared_mappo_model_04 去除了信息素的奖励
    # MAPPO_change_Loss_and_model_03 修改了奖励函数，探索奖励的系数从0.4增加到0.5
    # MAPPO_change_Loss_and_model_02 修改了奖励函数，增加了奖励

    ## 加载模型
    @staticmethod
    def load(dim_info, is_continue, model_dir, trick=None):
        # 初始化一个共享参数的 MAPPO 实例（假设所有智能体共用一个 Agent 实例）
        policy = MAPPO(dim_info, actor_lr=0, critic_lr=0, horizon=0, device='cuda', trick=trick)

        # 加载共享的模型状态
        state = torch.load(os.path.join(model_dir, 'shared_mappo_model_21_9232.pth'))
        policy.agents.full_model.load_state_dict(state['full_model'])
        policy.agents.actor.load_state_dict(state['actor'])
        policy.agents.critic.load_state_dict(state['critic'])
        # 可选：如需加载优化器状态
        # policy.agents.optimizer.load_state_dict(state['optimizer'])

        return policy
