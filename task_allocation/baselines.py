import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class BaselineMethod:
    def __init__(self, env):
        self.env = env
        self.name = "Base"

    def assign_tasks(self, release_agent_ids):
        raise NotImplementedError

    def distance(self, agent, target):
        return np.linalg.norm(agent['location'] - target['location'])

    def contrib_vector(self, agent, task_status):
        return np.maximum(np.minimum(task_status, agent['abilities']), 0.0)

    def contrib_sum(self, agent, task_status):
        return self.contrib_vector(agent, task_status).sum()

    def can_contribute(self, agent, task):
        return self.contrib_sum(agent, task['status']) > 0


class HungarianBaseline(BaselineMethod):
    def __init__(self, env):
        super().__init__(env)
        self.name = "Hungarian"

    def assign_tasks(self, release_agent_ids):
        agents = release_agent_ids[:]
        tasks = [t for t in self.env.task_dic.values()
                 if (not t['finished']) and (not t['feasible_assignment'])]

        if len(agents) == 0:
            return {}
        if len(tasks) == 0:
            return {aid: 0 for aid in agents}

        cost = np.zeros((len(agents), len(tasks)))
        for i, aid in enumerate(agents):
            a = self.env.agent_dic[aid]
            for j, t in enumerate(tasks):
                cost[i, j] = self.distance(a, t) / max(a['velocity'], 1e-6)

        row_ind, col_ind = linear_sum_assignment(cost)

        assignment = {}
        for i, j in zip(row_ind, col_ind):
            assignment[agents[i]] = tasks[j]['ID'] + 1

        for aid in agents:
            if aid not in assignment:
                assignment[aid] = 0
        return assignment


class GreedyNearestBaseline(BaselineMethod):
    def __init__(self, env):
        super().__init__(env)
        self.name = "Greedy-Nearest"

    def assign_tasks(self, release_agent_ids):
        assignment = {}
        for aid in release_agent_ids:
            agent = self.env.agent_dic[aid]
            best_task_id = 0
            best_dist = float('inf')

            for tid, t in self.env.task_dic.items():
                if t['finished']:
                    continue
                if not self.can_contribute(agent, t):
                    continue

                d = self.distance(agent, t)
                if d < best_dist:
                    best_dist = d
                    best_task_id = tid + 1

            assignment[aid] = best_task_id
        return assignment


class GreedyPriorityBaseline(BaselineMethod):
    """
    基于deadline urgency的贪心：评分 = urgency * contrib / (1 + dist)
    - urgency = 1.5 - tanh(time_remaining / 50)，范围 [0.5, 1.5]，越紧急越高（与 env shaping 一致）
    - contrib = agent 能贡献的能力维度数
    - dist = 欧氏距离
    区别于 Greedy-Nearest：紧急任务即使稍远也优先选；区别于 CBBA：单任务单 agent，无 bundle
    """
    def __init__(self, env):
        super().__init__(env)
        self.name = "Greedy-Priority"

    def assign_tasks(self, release_agent_ids):
        assignment = {}
        for aid in release_agent_ids:
            agent = self.env.agent_dic[aid]
            best_task_id = 0
            best_score = -float('inf')

            for tid, t in self.env.task_dic.items():
                if t['finished']:
                    continue
                if not self.can_contribute(agent, t):
                    continue
                dist = self.distance(agent, t)
                contrib = self.contrib_sum(agent, t['status'])
                deadline = float(t.get('deadline', self.env.current_time + 999))
                time_remaining = max(0.0, deadline - self.env.current_time)
                urgency = 1.5 - float(np.tanh(time_remaining / 50.0))
                score = urgency * contrib / (1.0 + dist)
                if score > best_score:
                    best_score = score
                    best_task_id = tid + 1

            assignment[aid] = best_task_id
        return assignment


class GreedyCoalitionBaseline(BaselineMethod):
    """
    协作贪心（累积式）：
    - 按优先级排序任务
    - 对每个任务，按“边际贡献/距离”综合评分选择若干个agent（至少1个），即使本轮无法完全满足也分配，
      让成员逐步累积，后续轮次继续补齐，最终满足需求。
    - 评分：score = contrib_sum / (1 + dist)
    - 每任务最多分配 top_k 个（防止过度集中，默认 top_k=2），你可以根据需要调整。
    """
    def __init__(self, env, top_k=2):
        super().__init__(env)
        self.name = "Greedy-Coalition"
        self.top_k = max(1, int(top_k))

    def assign_tasks(self, release_agent_ids):
        # 默认全部回仓库
        assignment = {aid: 0 for aid in release_agent_ids}

        # 未完成任务列表
        tasks = [t for t in self.env.task_dic.values() if not t['finished']]
        if len(tasks) == 0 or len(release_agent_ids) == 0:
            return assignment

        # 按优先级降序处理任务
        tasks.sort(key=lambda x: x.get('search_priority', 0.5), reverse=True)

        used_agents = set()

        for t in tasks:
            remaining = t['status'].copy()

            # 对当前释放且未用的agent计算贡献评分
            candidates = []
            for aid in release_agent_ids:
                if aid in used_agents:
                    continue
                a = self.env.agent_dic[aid]
                contrib = self.contrib_vector(a, remaining)
                if contrib.sum() <= 0:
                    continue
                dist = self.distance(a, t)
                score = contrib.sum() / (1.0 + dist)
                candidates.append((aid, contrib, score))

            if not candidates:
                continue

            # 按评分排序，选择前 top_k
            candidates.sort(key=lambda x: x[2], reverse=True)
            selected = candidates[:self.top_k]

            # 累积分配：即使本轮还无法“全满足”，也把这些agent分到该任务
            for aid, contrib, score in selected:
                assignment[aid] = t['ID'] + 1
                used_agents.add(aid)
                remaining = remaining - contrib
                # 若已满足，可提前停止选择更多成员
                if remaining.sum() <= 0:
                    break

        return assignment

class CBBABaseline(BaselineMethod):
    """
    稳健版简化 CBBA：
    - 一轮捆绑：每个任务挑 top-K 个出价最高的 agent（默认 K=2）
    - 出价包含优先级、边际贡献与时间惩罚：bid = 100*priority + contrib_sum - travel_time
    - 无自适应上限、无多轮，以降低出错概率并保持稳定行为
    """
    def __init__(self, env, max_bundle_size=2, w_prio=100.0, w_marg=1.0, w_time=1.0):
        super().__init__(env)
        self.name = "CBBA"
        self.max_bundle_size = max_bundle_size
        self.w_prio = w_prio
        self.w_marg = w_marg
        self.w_time = w_time

    def assign_tasks(self, release_agent_ids):
        assignment = {aid: 0 for aid in release_agent_ids}
        tasks = [t for t in self.env.task_dic.values() if not t['finished']]
        if len(tasks) == 0 or len(release_agent_ids) == 0:
            return assignment

        # 出价表：{task_id: [(aid, bid, contrib_sum)]}
        bids = defaultdict(list)
        for aid in release_agent_ids:
            a = self.env.agent_dic[aid]
            for t in tasks:
                status = t['status']
                contrib = self.contrib_vector(a, status)
                if contrib.sum() <= 0:
                    continue
                dist = self.distance(a, t)
                travel_time = dist / max(a['velocity'], 1e-6)
                prio = t.get('search_priority', 0.5)
                bid = self.w_prio * prio + self.w_marg * contrib.sum() - self.w_time * travel_time
                bids[t['ID']].append((aid, bid, contrib.sum()))

        # 每个任务选择 top-K winners
        for tid, offers in bids.items():
            if not offers:
                continue
            offers.sort(key=lambda x: x[1], reverse=True)
            winners = []
            for aid, _, _ in offers:
                if len(winners) >= self.max_bundle_size:
                    break
                winners.append(aid)
            for aid in winners:
                assignment[aid] = tid + 1

        return assignment


# ============================================================================
# 分布式 baseline（信息不对称公平对比）
# 每个 agent 只能用 self.env.agent_knowledge[aid]['known_tasks'] 局部知识，
# 不能访问 env.task_dic 全局。知识可能过期（task_current_status 可能滞后）。
# 潮汐任务只有 20% 前线 agent 一开始就知道，其余靠 gossip 传染。
# ============================================================================


class DistributedCBBABaseline(BaselineMethod):
    """
    分布式 CBBA：每个 agent 独立基于局部知识出价，无中心拍卖。
    - bid = w_prio * urgency + w_marg * contrib - w_time * travel_time
    - urgency 由 deadline 计算（与 env shaping 一致）
    - 不做 bundle 共享/共识（信息不对称的核心代价）
    - 多个 agent 可能选同一任务（无冲突解决）
    """
    def __init__(self, env, w_prio=1.0, w_marg=1.0, w_time=1.0):
        super().__init__(env)
        self.name = "Dist-CBBA"
        self.w_prio = w_prio
        self.w_marg = w_marg
        self.w_time = w_time

    def assign_tasks(self, release_agent_ids):
        assignment = {aid: 0 for aid in release_agent_ids}
        for aid in release_agent_ids:
            a = self.env.agent_dic[aid]
            knowledge = self.env.agent_knowledge.get(aid, {})
            known_tids = knowledge.get('known_tasks', set()) - knowledge.get('known_finished', set())

            best_tid = 0
            best_bid = -float('inf')
            for tid in known_tids:
                if tid not in self.env.task_dic:
                    continue
                t = self.env.task_dic[tid]
                if t['finished']:
                    continue
                # 用 knowledge 中可能过期的 status（信息不对称的代价）
                status = knowledge.get('task_current_status', {}).get(tid, t['status'])
                contrib = self.contrib_vector(a, status)
                if contrib.sum() <= 0:
                    continue
                loc = knowledge.get('task_locations', {}).get(tid, t['location'])
                dist = float(np.linalg.norm(a['location'] - loc))
                travel_time = dist / max(a['velocity'], 1e-6)
                deadline = float(knowledge.get('task_deadlines', {}).get(tid, t.get('deadline', self.env.current_time + 999)))
                time_remaining = max(0.0, deadline - self.env.current_time)
                urgency = 1.5 - float(np.tanh(time_remaining / 50.0))
                bid = self.w_prio * urgency + self.w_marg * contrib.sum() - self.w_time * travel_time
                if bid > best_bid:
                    best_bid = bid
                    best_tid = tid + 1
            assignment[aid] = best_tid
        return assignment


class DistributedGreedyCoalitionBaseline(BaselineMethod):
    """
    分布式 Greedy-Coalition：每个 agent 独立选最优任务，无中心协调。
    - score = urgency * contrib / (1 + dist)
    - 不做 top-K 选 agent（中心协调的福利）
    - 多个 agent 可能挤同一任务，无人协调联盟大小
    """
    def __init__(self, env):
        super().__init__(env)
        self.name = "Dist-Greedy-Coalition"

    def assign_tasks(self, release_agent_ids):
        assignment = {aid: 0 for aid in release_agent_ids}
        for aid in release_agent_ids:
            a = self.env.agent_dic[aid]
            knowledge = self.env.agent_knowledge.get(aid, {})
            known_tids = knowledge.get('known_tasks', set()) - knowledge.get('known_finished', set())

            best_tid = 0
            best_score = -float('inf')
            for tid in known_tids:
                if tid not in self.env.task_dic:
                    continue
                t = self.env.task_dic[tid]
                if t['finished']:
                    continue
                status = knowledge.get('task_current_status', {}).get(tid, t['status'])
                contrib = self.contrib_vector(a, status)
                if contrib.sum() <= 0:
                    continue
                loc = knowledge.get('task_locations', {}).get(tid, t['location'])
                dist = float(np.linalg.norm(a['location'] - loc))
                deadline = float(knowledge.get('task_deadlines', {}).get(tid, t.get('deadline', self.env.current_time + 999)))
                time_remaining = max(0.0, deadline - self.env.current_time)
                urgency = 1.5 - float(np.tanh(time_remaining / 50.0))
                score = urgency * contrib.sum() / (1.0 + dist)
                if score > best_score:
                    best_score = score
                    best_tid = tid + 1
            assignment[aid] = best_tid
        return assignment