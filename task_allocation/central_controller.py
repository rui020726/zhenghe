# central_controller.py
import torch
import numpy as np
from torch.distributions import Categorical
from parameters import *  # 确保 TrainParams.AGENT_INPUT_DIM / TASK_INPUT_DIM / EnvParams 可用
import warnings

def _flatten_to_numbers(obj):
    """
    将可能的标量 / list / np.ndarray / torch.tensor 等展平成 float 的 Python 列表。
    忽略无法转为数值的项。
    """
    out = []
    if obj is None:
        return out
    # numpy scalar
    if isinstance(obj, (np.generic,)):
        try:
            return [float(obj)]
        except Exception:
            return []
    # torch scalar / tensor
    if isinstance(obj, torch.Tensor):
        try:
            arr = obj.detach().cpu().numpy()
        except Exception:
            try:
                return [float(obj.item())]
            except Exception:
                return []
        return _flatten_to_numbers(arr)
    # list/tuple/ndarray: 递归展开
    if isinstance(obj, (list, tuple, np.ndarray)):
        for e in obj:
            out.extend(_flatten_to_numbers(e))
        return out
    # python int/float
    if isinstance(obj, (int, float)):
        return [float(obj)]
    # 尝试把可转字符串转为 float
    try:
        v = float(obj)
        return [v]
    except Exception:
        return []

def _to_fixed_length_vector(items, target_len):
    """
    items: 任意可展平对象（标量 / list / np.array / tensor / 混合）
    target_len: 目标长度（int）
    返回 numpy 1D float32 向量（长度 == target_len）
    """
    flat = _flatten_to_numbers(items)
    if len(flat) >= target_len:
        vec = np.array(flat[:target_len], dtype=np.float32)
    else:
        vec = np.zeros(target_len, dtype=np.float32)
        if flat:
            vec[:len(flat)] = np.array(flat, dtype=np.float32)
    return vec

class CentralController:
    """
    集中式控制器：提取全局观测并（可选）调用 central_net 生成分配。
    本实现对 agent/task 混合字段做展平 + pad，保证输出张量形状规则。
    """
    def __init__(self, central_net=None, device='cpu'):
        self.central_net = central_net
        try:
            self.device = torch.device(device if isinstance(device, str) else str(device))
        except Exception:
            self.device = torch.device('cpu')

    def _build_agents_tensor(self, agent_iter, agent_dim):
        """
        agent_iter: iterable of agent entries (可能是 dict 或对象或像你示例的 list)
        agent_dim: 目标每个 agent 行长度
        返回 torch.Tensor shape [num_agents, agent_dim]（device=self.device）
        """
        rows = []
        for a in agent_iter:
            if isinstance(a, dict):
                # 优先使用 'trait' 或 'abilities' 或 'feat' 字段
                if 'trait' in a:
                    raw = a['trait']
                elif 'abilities' in a:
                    raw = a['abilities']
                elif 'feat' in a:
                    raw = a['feat']
                else:
                    # 否则把整个行（例如你示例行）展平成数值序列
                    # 若是像示例那样的列表，直接展平
                    raw = list(a.values()) if a else None
            else:
                # a 是对象：优先尝试属性 trait/abilities/feat
                raw = getattr(a, 'trait', None) or getattr(a, 'abilities', None) or getattr(a, 'feat', None) or a

            vec = _to_fixed_length_vector(raw, agent_dim)
            rows.append(vec)

        if len(rows) == 0:
            arr = np.zeros((0, agent_dim), dtype=np.float32)
        else:
            arr = np.stack(rows, axis=0).astype(np.float32)  # shape (num_agents, agent_dim)

        try:
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        except Exception:
            return torch.tensor(arr, dtype=torch.float32)

    def _build_tasks_tensor(self, task_iter, task_dim):
        rows = []
        for t in task_iter:
            if isinstance(t, dict):
                # 优先 'feat' 或由若干常用字段构造
                if 'feat' in t:
                    raw = t['feat']
                elif 'search_priority' in t:
                    # 构造从优先级、位置和时间等字段组成的列表（按需调整）
                    raw = [
                        1.0 if t.get('finished', False) else 0.0,
                        t.get('search_priority', 0.0),
                        t.get('time_start', 0.0) if t.get('time_start', -1) != -1 else 0.0,
                        t.get('time_finish', 0.0) if t.get('time_finish', -1) != -1 else 0.0,
                    ]
                    # 加入位置（若存在）
                    loc = t.get('location', None)
                    if loc is not None and isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
                        raw.extend([loc[0], loc[1]])
                else:
                    raw = list(t.values()) if t else None
            else:
                raw = getattr(t, 'feat', None) or t

            vec = _to_fixed_length_vector(raw, task_dim)
            rows.append(vec)

        if len(rows) == 0:
            arr = np.zeros((0, task_dim), dtype=np.float32)
        else:
            arr = np.stack(rows, axis=0).astype(np.float32)
        try:
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        except Exception:
            return torch.tensor(arr, dtype=torch.float32)

    def get_global_observation(self, env):
        """
        从 env 提取并返回：
          {'agents': Tensor[num_agents, AGENT_INPUT_DIM],
           'tasks' : Tensor[num_tasks, TASK_INPUT_DIM],
           'global': Tensor[1, global_dim]}  # 保持与 worker.py 原逻辑兼容
        任何缺失或混合字段均会被展平并 pad 到固定长度，避免 shape 错误。
        """
        agent_dim = getattr(TrainParams, "AGENT_INPUT_DIM", getattr(EnvParams, "TRAIT_DIM", 8))
        task_dim = getattr(TrainParams, "TASK_INPUT_DIM", getattr(EnvParams, "TASK_INPUT_DIM", 8))

        # 获取 agent iterable（兼容多种 env 接口）
        try:
            if hasattr(env, 'agent_dic'):
                agent_iter = list(env.agent_dic.values())
            elif hasattr(env, 'agents'):
                agent_iter = list(env.agents)
            elif hasattr(env, 'get_agents'):
                agent_iter = list(env.get_agents())
            else:
                agent_iter = []
        except Exception:
            agent_iter = []

        # 获取 task iterable
        try:
            if hasattr(env, 'task_dic'):
                task_iter = list(env.task_dic.values())
            elif hasattr(env, 'tasks'):
                task_iter = list(env.tasks)
            elif hasattr(env, 'get_tasks'):
                task_iter = list(env.get_tasks())
            else:
                task_iter = []
        except Exception:
            task_iter = []

        agents_tensor = self._build_agents_tensor(agent_iter, agent_dim)
        tasks_tensor = self._build_tasks_tensor(task_iter, task_dim)

        # 构造 global 信息（保持为 shape (1, N) 以兼容 worker 里对 global[:,1] 的索引）
        try:
            completed = float(np.sum([1 for t in env.task_dic.values() if t.get('finished', False)])) if hasattr(env, 'task_dic') else 0.0
        except Exception:
            completed = 0.0
        global_vec = np.array([
            float(getattr(env, 'current_time', 0.0)),
            1.0 if getattr(env, 'finished', False) else 0.0,
            completed,
            float(len(agent_iter)),
            float(len(task_iter))
        ], dtype=np.float32)
        global_tensor = torch.tensor(global_vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # shape (1, 5)

        return {
            'agents': agents_tensor,   # [num_agents, agent_dim]
            'tasks': tasks_tensor,     # [num_tasks, task_dim]
            'global': global_tensor    # [1, global_dim]
        }

    def decide_allocation(self, global_obs, is_train=True):
        """
        调用 central_net 生成 allocation 概率并采样 / 贪心决策。
        保持原来接口：返回 allocations dict 与 log_probs（训练时）。
        """
        if self.central_net is None:
            raise RuntimeError("central_net is not set in CentralController.")

        agent_input = global_obs['agents']       # [num_agents, agent_dim]
        task_input = global_obs['tasks']         # [num_tasks, task_dim]
        global_input = global_obs['global'].unsqueeze(0).unsqueeze(0)  # 如果网络期望更高维，可调整

        # 可能需要在调用前添加 batch dim；你的 local_net 在 worker 中被调用时使用了 unsqueeze(0)
        # 这里我们保持与 worker 中的调用一致性：central_net(task_input.unsqueeze(0), agent_input.unsqueeze(0), global_input)
        # 如果你的 central_net 接口不同，请按实际调整。
        try:
            allocation_probs, _ = self.central_net(
                task_input.unsqueeze(0), agent_input.unsqueeze(0), global_input
            )
            # allocation_probs: [1, max_agents, max_tasks(+1)?] 或 [1, max_agents, ...]
            allocation_probs = allocation_probs.squeeze(0)  # [max_agents, max_tasks]
        except Exception as e:
            raise RuntimeError(f"Calling central_net failed: {e}")

        num_agents = allocation_probs.shape[0]
        num_tasks = global_obs['tasks'].shape[0]

        if is_train:
            dist = Categorical(allocation_probs)
            alloc_tensor = dist.sample()
            log_probs = dist.log_prob(alloc_tensor)
        else:
            alloc_tensor = torch.argmax(allocation_probs, dim=1)
            log_probs = None

        allocations = {}
        for agent_id in range(num_agents):
            task_id = int(alloc_tensor[agent_id].item())
            allocations[agent_id] = task_id if task_id < num_tasks else -1

        return allocations, log_probs
