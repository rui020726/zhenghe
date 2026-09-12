# -*- coding: utf-8 -*-
"""
时间驱动任务分配环境（CTDE范式 + 严格时序多阶段任务 + 潮汐式爆发 + 局部通信传染）
- dt=1 时间驱动推进
- 移动过程中持续通信/传播 (Gossip Protocol)
- ✅ [潮汐式爆发]：模拟无人机群周期性发现大量任务。
- ✅ [信息不对称]：新任务只通知 20% 的前线车辆，其余车辆必须通过通信网络相互传染得知！
- ✅ [性能极致优化版]：Numpy通信图 + 集合差集Gossip + 延迟同步 + 垃圾回收
- ✅ [强制值班制]：移除“任务做完即提前下班”机制，硬扛到时间耗尽以等待潮汐。
"""

import numpy as np
from collections import deque
import torch

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

EPS_ARRIVE = 1e-6


class TaskEnv:
    def __init__(
            self,
            deadline_tightness=(0.5, 2.0),
            per_species_range=(3, 5),
            species_range=(2, 3),
            tasks_range=(15, 25),
            traits_dim=5,
            max_task_size=2,
            duration_scale=5,
            seed=None,
            comm_range=0.3,
            dynamic_task_discovery=True,
            discovery_probability=0.03,
            max_dynamic_tasks=5,
            dt=1.0,
            shaping_crowd_penalty=0.15,
            shaping_lonely_bonus=0.10,
            shaping_redundant_penalty=0.30,
            shaping_distance_penalty=0.05,
            working_timeout_range=(50.0, 120.0),
            working_timeout_decay=0.7,
            working_timeout_min=15.0,
            working_timeout_penalty=-0.5,
            cooldown_duration=10.0,
            enable_tidal_tasks=False,
            tidal_interval=300.0,
            tidal_burst_range=(10, 20),
            tidal_notify_ratio=0.2,
            gossip_interval=2,
            enable_discovered_tasks=True,
            discovered_interval=350.0,
            discovered_burst_range=(3, 8),
            discovered_notify_ratio=0.1,
            discovered_cluster_radius=0.15,
            hard_expire=False,
            **kwargs,
    ):
        # 向后兼容旧版 prio_range 参数
        if "prio_range" in kwargs:
            deadline_tightness = kwargs.pop("prio_range")
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.ranges = {
            "deadline_tightness": deadline_tightness,
            "per_species": per_species_range,
            "species": species_range,
            "tasks": tasks_range,
        }
        self.traits_dim = traits_dim
        self.max_task_size = max_task_size
        self.duration_scale = duration_scale
        self.comm_range = comm_range

        self.dynamic_task_discovery = dynamic_task_discovery
        self.discovery_probability = discovery_probability
        self.max_dynamic_tasks = max_dynamic_tasks
        self.next_dynamic_task_id = 0
        self.task_source = {}

        self.dt = float(dt)

        self.shaping_crowd_penalty = shaping_crowd_penalty
        self.shaping_lonely_bonus = shaping_lonely_bonus
        self.shaping_redundant_penalty = shaping_redundant_penalty
        self.shaping_distance_penalty = shaping_distance_penalty

        self.working_timeout_range = working_timeout_range
        self.working_timeout_decay = working_timeout_decay
        self.working_timeout_min = working_timeout_min
        self.working_timeout_penalty = working_timeout_penalty
        self.cooldown_duration = cooldown_duration

        self.enable_tidal_tasks = enable_tidal_tasks
        self.tidal_interval = tidal_interval
        self.tidal_burst_range = tidal_burst_range
        self.tidal_notify_ratio = tidal_notify_ratio
        self.last_tidal_time = 0.0

        self.enable_discovered_tasks = enable_discovered_tasks
        self.discovered_interval = discovered_interval
        self.discovered_burst_range = discovered_burst_range
        self.discovered_notify_ratio = discovered_notify_ratio
        self.discovered_cluster_radius = discovered_cluster_radius
        self.last_discovered_time = 0.0
        self.hard_expire = hard_expire

        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.next_dynamic_task_id = self.tasks_num

        self.comm_graph = {}
        self.agent_knowledge = {}
        self.init_agent_memory()

        self.current_time = 0.0
        self.max_waiting_time = 200
        self.finished = False
        self._step_counter = 0  # gossip 降频计数器
        self.gossip_interval = gossip_interval

        for a in self.agent_dic.values():
            a["need_decision"] = True

        self._step_events = []
        self._inner_step_counter = 0
        self._active_agent_steps = 0
        self._total_agent_steps = 0

    def random_int(self, low, high, size=None):
        if low == high:
            return low if size is None else np.full(size, low, dtype=int)
        return self.rng.integers(low, high, size)

    def random_val(self, shape):
        return self.rng.random(shape)

    def random_choice(self, a, size=None, replace=True):
        return self.rng.choice(a, size, replace=replace)

    def _sample_working_timeout(self, agent_id):
        agent = self.agent_dic[agent_id]
        consecutive = agent.get("consecutive_timeouts", 0)
        base_lo, base_hi = self.working_timeout_range
        factor = self.working_timeout_decay ** consecutive
        lo = max(self.working_timeout_min, base_lo * factor)
        hi = max(lo + 5.0, base_hi * factor)
        return float(self.rng.uniform(lo, hi))

    def _release_agent(self, agent_id):
        if agent_id in self.agent_dic:
            agent = self.agent_dic[agent_id]
            agent.update({
                "assigned": False, "current_task": -1, "status": "idle", "need_decision": True,
                "consecutive_timeouts": 0, "working_start_time": None, "working_timeout_value": None,
                "target_type": None, "target_id": None, "immediate_replan": True
            })

    def generate_task(self, tasks_num):
        tasks_stages = {}
        scene_complexities = self.rng.uniform(0.3, 1.0, tasks_num)

        for i in range(tasks_num):
            num_stages = self.random_int(1, 4)
            stages = []
            base_time = 2.0
            complexity_scale = self.duration_scale * 2
            noise = self.rng.uniform(0, 1.5)
            total_time = base_time + scene_complexities[i] * complexity_scale + noise
            stage_time = max(2.0, total_time / num_stages)

            for s in range(num_stages):
                req = np.zeros(self.traits_dim, dtype=int)
                num_required = self.random_int(1, min(3, self.traits_dim + 1))
                dims = self.random_choice(self.traits_dim, size=num_required, replace=False)
                for d in dims: req[d] = 1

                stages.append({
                    "stage_id": s, "requirements": req, "status": req.copy(),
                    "time": float(stage_time), "time_left": float(stage_time), "finished": False
                })
            tasks_stages[i] = stages

        return tasks_stages, scene_complexities

    def generate_agent(self, species_num):
        agents_ini = np.zeros((species_num, self.traits_dim), dtype=int)
        for s in range(species_num):
            num_skills = self.random_int(1, 3)
            skill_indices = self.random_choice(self.traits_dim, size=num_skills, replace=False)
            for idx in skill_indices: agents_ini[s, idx] = 1

        for dim in range(self.traits_dim):
            if agents_ini[:, dim].sum() == 0:
                random_species = self.random_int(0, species_num)
                agents_ini[random_species, dim] = 1
        return agents_ini

    def _assign_deadlines(self, task_ids, base_time=0.0, task_dic=None, depot_dic=None):
        """为任务分配绝对截止时间"""
        td = task_dic if task_dic is not None else self.task_dic
        dd = depot_dic if depot_dic is not None else self.depot_dic
        tight_lo, tight_hi = self.ranges["deadline_tightness"]
        for tid in task_ids:
            task = td[tid]
            min_dist = min(
                np.linalg.norm(task["location"] - d["location"]) for d in dd.values()
            )
            travel_time = min_dist / 0.2
            total_work = sum(s["time"] for s in task["stages"])
            estimated = travel_time + total_work + 30.0 + 0.3 * total_work + self.rng.uniform(0, 50)
            tightness = self.rng.uniform(tight_lo, tight_hi)
            deadline = base_time + max(estimated * tightness, estimated * 0.95)
            task["deadline"] = float(deadline)

    def generate_env(self):
        n_tasks = self.random_int(self.ranges["tasks"][0], self.ranges["tasks"][1])
        n_species = self.random_int(self.ranges["species"][0], self.ranges["species"][1])
        agents_species_num = [self.random_int(self.ranges["per_species"][0], self.ranges["per_species"][1]) for _ in
                              range(n_species)]

        agents_ini = self.generate_agent(n_species)
        covered_abilities = agents_ini.sum(axis=0) > 0
        for dim in range(self.traits_dim):
            if not covered_abilities[dim]:
                species_idx = self.random_int(0, n_species)
                agents_ini[species_idx, dim] = 1

        tasks_stages, scene_complexities = self.generate_task(n_tasks)
        depot_loc = self.random_val((n_species, 2))
        tasks_loc = self.random_val((n_tasks, 2))

        task_dic, agent_dic, depot_dic = {}, {}, {}
        species_dict = {"abilities": agents_ini, "number": agents_species_num}

        for i in range(n_tasks):
            task_dic[i] = {
                "ID": i, "stages": tasks_stages[i], "current_stage_index": 0, "total_stages": len(tasks_stages[i]),
                "requirements": tasks_stages[i][0]["requirements"].copy(),
                "status": tasks_stages[i][0]["status"].copy(),
                "time": tasks_stages[i][0]["time"], "location": tasks_loc[i, :],
                "scene_complexity": float(scene_complexities[i]),
                "finished": False, "feasible_assignment": False, "members": [],
                "time_start": 0.0, "time_finish": 0.0,
                "deadline": 0.0, "expired": False,
            }
            self.task_source[i] = "initial"

        cnt = 0
        for s in range(n_species):
            depot_dic[s] = {"ID": -s - 1, "location": depot_loc[s], "members": []}
            for _ in range(agents_species_num[s]):
                agent_dic[cnt] = {
                    "ID": cnt, "species": s, "abilities": agents_ini[s, :],
                    "location": depot_loc[s].copy(), "depot": depot_loc[s].copy(), "velocity": 0.2,
                    "current_task": -1, "travel_dist": 0.0, "sum_waiting_time": 0.0, "route": [],
                    "arrival_time": [0.0], "assigned": False, "is_rushing": False,
                    "target_task_expected_state": None, "failed_tasks": [], "immediate_replan": False,
                    "need_decision": True, "status": "idle", "target_type": None, "target_id": None,
                    "target_pos": None, "working_start_time": None, "working_timeout_value": None,
                    "consecutive_timeouts": 0, "cooldown_tasks": {},
                }
                depot_dic[s]["members"].append(cnt)
                cnt += 1

        self._assign_deadlines(list(task_dic.keys()), base_time=0.0, task_dic=task_dic, depot_dic=depot_dic)

        return task_dic, agent_dic, depot_dic, species_dict

    def _future_requirements(self, task):
        """下一阶段的真实需求向量；已是最后阶段则为全零。"""
        stage_index = min(int(task.get("current_stage_index", 0)), len(task["stages"]) - 1)
        if stage_index + 1 < len(task["stages"]):
            return task["stages"][stage_index + 1]["requirements"].copy()
        return np.zeros(self.traits_dim, dtype=int)

    def init_agent_memory(self):
        self.agent_knowledge = {
            aid: {
                "known_tasks": set(self.task_dic.keys()),
                "task_locations": {tid: task["location"].copy() for tid, task in self.task_dic.items()},
                "task_requirements": {tid: task["requirements"].copy() for tid, task in self.task_dic.items()},
                "task_deadlines": {tid: task["deadline"] for tid, task in self.task_dic.items()},
                "dynamic_tasks": set(), "known_finished": set(), "known_assigned": set(),
                "task_current_status": {tid: task["status"].copy() for tid, task in self.task_dic.items()},
                "task_members": {tid: [] for tid in self.task_dic.keys()},
                "task_last_update_time": {tid: 0.0 for tid in self.task_dic.keys()},
                # 决策输入（observation/mask）只读这些 belief 副本：断联期间真值推进
                # 不会改变该机器人的观察，只有新消息到达才会更新。
                "task_stage_index": {tid: int(task.get("current_stage_index", 0)) for tid, task in self.task_dic.items()},
                "task_future_requirements": {tid: self._future_requirements(task) for tid, task in self.task_dic.items()},
                "task_scene_complexity": {tid: float(task.get("scene_complexity", 0.5)) for tid, task in self.task_dic.items()},
            }
            for aid in self.agent_dic.keys()
        }

    def learn_task_view(self, agent_id, view):
        """把一条任务视图（通信消息或视距感知到的真值）写进一台机器人的 belief。

        view 是集成层与本项目共享的 belief 字段：task_id、location、deadline、
        status、requirements、members、update_time、dynamic，以及阶段信息
        stage_index、future_requirements、scene_complexity。调用方决定这条消息
        是否比机器人已知的更新；本方法只负责按统一模式落盘。
        """
        tid = int(view["task_id"])
        knowledge = self.agent_knowledge[agent_id]
        knowledge["known_tasks"].add(tid)
        knowledge["task_locations"][tid] = np.asarray(view["location"], dtype=float).copy()
        knowledge["task_deadlines"][tid] = float(view["deadline"])
        knowledge["task_current_status"][tid] = np.asarray(view["status"], dtype=int).copy()
        knowledge["task_requirements"][tid] = np.asarray(view["requirements"], dtype=int).copy()
        knowledge["task_members"][tid] = list(view["members"])
        knowledge["task_last_update_time"][tid] = float(view["update_time"])
        knowledge["task_stage_index"][tid] = int(view.get("stage_index", 0))
        future_req = np.asarray(
            view.get("future_requirements", np.zeros(self.traits_dim, dtype=int)), dtype=int
        )
        if future_req.size == 0:
            # 最后阶段（或消息未携带未来需求）： belief 里记为全零。
            future_req = np.zeros(self.traits_dim, dtype=int)
        knowledge["task_future_requirements"][tid] = future_req.copy()
        knowledge["task_scene_complexity"][tid] = float(view.get("scene_complexity", 0.5))
        if view.get("dynamic"):
            knowledge["dynamic_tasks"].add(tid)

    def refresh_task_belief(self, agent_id, tid, update_time=None):
        """视距感知/到场核实：机器人用现场真值刷新自己对任务 tid 的 belief。"""
        task = self.task_dic[tid]
        stage_index = min(int(task.get("current_stage_index", 0)), len(task["stages"]) - 1)
        self.learn_task_view(
            agent_id,
            {
                "task_id": tid,
                "location": task["location"],
                "deadline": task["deadline"],
                "status": task["status"],
                "requirements": task["requirements"],
                "members": task.get("members", []),
                "update_time": self.current_time if update_time is None else update_time,
                "stage_index": stage_index,
                "future_requirements": self._future_requirements(task),
                "scene_complexity": float(task.get("scene_complexity", 0.5)),
                "dynamic": self.task_source.get(tid) in ("discovered", "tidal"),
            },
        )
        if task.get("finished", False):
            self.agent_knowledge[agent_id]["known_finished"].add(tid)

    def _copy_knowledge(self, knowledge):
        """tick 开始时刻的知识快照，gossip 只能基于它传播，避免一个 tick 多跳。"""
        return {
            "known_tasks": set(knowledge["known_tasks"]),
            "task_locations": {tid: value.copy() for tid, value in knowledge["task_locations"].items()},
            "task_requirements": {tid: value.copy() for tid, value in knowledge["task_requirements"].items()},
            "task_deadlines": dict(knowledge["task_deadlines"]),
            "dynamic_tasks": set(knowledge["dynamic_tasks"]),
            "known_finished": set(knowledge["known_finished"]),
            "known_assigned": set(knowledge["known_assigned"]),
            "task_current_status": {tid: value.copy() for tid, value in knowledge["task_current_status"].items()},
            "task_members": {tid: list(value) for tid, value in knowledge["task_members"].items()},
            "task_last_update_time": dict(knowledge["task_last_update_time"]),
            "task_stage_index": dict(knowledge["task_stage_index"]),
            "task_future_requirements": {
                tid: value.copy() for tid, value in knowledge["task_future_requirements"].items()
            },
            "task_scene_complexity": dict(knowledge["task_scene_complexity"]),
        }

    def _share_task_belief(self, sender, tid, receiver):
        """把发送方 belief 中的任务 tid 合并进接收方（读快照，写真实 knowledge）。"""
        task = self.task_dic[tid]
        receiver["known_tasks"].add(tid)
        receiver["task_locations"][tid] = sender["task_locations"].get(tid, task["location"]).copy()
        receiver["task_deadlines"][tid] = sender["task_deadlines"].get(tid, task["deadline"])
        receiver["task_current_status"][tid] = sender["task_current_status"].get(tid, task["status"]).copy()
        receiver["task_requirements"][tid] = sender["task_requirements"].get(tid, task["requirements"]).copy()
        receiver["task_members"][tid] = list(sender["task_members"].get(tid, task.get("members", [])))
        receiver["task_last_update_time"][tid] = sender["task_last_update_time"].get(tid, 0.0)
        receiver["task_stage_index"][tid] = int(
            sender["task_stage_index"].get(tid, min(int(task.get("current_stage_index", 0)), len(task["stages"]) - 1))
        )
        receiver["task_future_requirements"][tid] = sender["task_future_requirements"].get(
            tid, self._future_requirements(task)
        ).copy()
        receiver["task_scene_complexity"][tid] = float(
            sender["task_scene_complexity"].get(tid, task.get("scene_complexity", 0.5))
        )
        if tid in sender["dynamic_tasks"]:
            receiver["dynamic_tasks"].add(tid)

    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.task_source = {}
        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.tasks_num, self.agents_num, self.next_dynamic_task_id = len(self.task_dic), len(self.agent_dic), len(
            self.task_dic)
        self.init_agent_memory()
        self.current_time, self.last_tidal_time, self.finished = 0.0, 0.0, False
        self.last_discovered_time = 0.0
        self.comm_graph, self._step_events = {}, []
        self._step_counter = 0

        self._inner_step_counter = 0
        self._active_agent_steps = 0
        self._total_agent_steps = 0
        return self.get_global_state()

    def get_resource_utilization(self):
        if self._total_agent_steps <= 0:
            return 0.0
        return float(self._active_agent_steps) / float(self._total_agent_steps)

    def update_communication_graph(self):
        self.comm_graph = {aid: [] for aid in self.agent_dic.keys()}
        agent_ids = list(self.agent_dic.keys())
        if not agent_ids: return

        locs = np.array([self.agent_dic[aid]["location"] for aid in agent_ids])
        diff = locs[:, np.newaxis, :] - locs[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        adj_matrix = dists <= self.comm_range
        np.fill_diagonal(adj_matrix, False)

        for i, aid_i in enumerate(agent_ids):
            self.comm_graph[aid_i] = [agent_ids[j] for j in np.where(adj_matrix[i])[0]]

    def propagate_task_completion(self):
        # 第一部分：自身视距感知（批量向量化）
        agent_ids = list(self.agent_dic.keys())
        task_ids = list(self.task_dic.keys())
        if agent_ids and task_ids:
            agent_locs = np.array([self.agent_dic[aid]["location"] for aid in agent_ids])
            task_locs = np.array([self.task_dic[tid]["location"] for tid in task_ids])
            diffs = agent_locs[:, np.newaxis, :] - task_locs[np.newaxis, :, :]
            in_range = np.linalg.norm(diffs, axis=-1) <= self.comm_range

            for i, aid in enumerate(agent_ids):
                for j, tid in enumerate(task_ids):
                    if not in_range[i, j]:
                        continue
                    # 视距感知：机器人就在现场，用真值刷新自己的 belief。
                    self.refresh_task_belief(aid, tid, update_time=self.current_time)

        # 第二部分：一跳通信传播。本轮传播只读取 tick 开始时的知识快照，写入的才是
        # 各机器人真实的 knowledge：一次 gossip 最多传播一跳，A 的新知识不能因为
        # 原地修改集合而在同一个 tick 内经 B 直达 C。
        snapshots = {aid: self._copy_knowledge(k) for aid, k in self.agent_knowledge.items()}
        for i, aid in enumerate(agent_ids):
            knowledge = snapshots[aid]
            for nid in self.comm_graph.get(aid, []):
                if aid >= nid: continue

                nk = snapshots[nid]

                # 1. 增量交换互相不知道的新任务
                for tid in (knowledge["known_tasks"] - nk["known_tasks"]):
                    if tid in self.task_dic:
                        self._share_task_belief(knowledge, tid, self.agent_knowledge[nid])

                for tid in (nk["known_tasks"] - knowledge["known_tasks"]):
                    if tid in self.task_dic:
                        self._share_task_belief(nk, tid, self.agent_knowledge[aid])

                # 2. 传递已完成信息
                for tid in (knowledge["known_finished"] - nk["known_finished"]):
                    if tid in nk["known_tasks"]:
                        self.agent_knowledge[nid]["known_finished"].add(tid)
                        self.agent_knowledge[nid]["task_last_update_time"][tid] = self.current_time

                for tid in (nk["known_finished"] - knowledge["known_finished"]):
                    if tid in knowledge["known_tasks"]:
                        self.agent_knowledge[aid]["known_finished"].add(tid)
                        self.agent_knowledge[aid]["task_last_update_time"][tid] = self.current_time

                # 3. 双方都知道的任务，看谁的时间戳更新，直接覆盖
                for tid in (knowledge["known_tasks"] & nk["known_tasks"]):
                    t_a, t_n = knowledge["task_last_update_time"].get(tid, 0.0), nk["task_last_update_time"].get(tid, 0.0)
                    if t_a > t_n:
                        self._share_task_belief(knowledge, tid, self.agent_knowledge[nid])
                    elif t_n > t_a:
                        self._share_task_belief(nk, tid, self.agent_knowledge[aid])

        # 第三部分在途重规划检查
        for aid, agent in self.agent_dic.items():
            if agent.get("is_rushing", False) and agent.get("target_type") == "task" and agent.get(
                    "target_id") is not None:
                if self.check_agent_enroute_replan(aid)[0]:
                    agent.update(
                        {"immediate_replan": True, "need_decision": True, "status": "idle", "target_type": None,
                         "target_id": None, "target_pos": None, "is_rushing": False, "current_task": -1})

    def _spawn_new_task(self, tid, location=None, source="tidal", discoverer=None):
        if location is None: location = np.clip(self.rng.random(2), 0, 1)

        num_stages = self.random_int(1, 4)
        stages = []
        base_time, complexity = 2.0, float(self.rng.uniform(0.5, 0.9))
        stage_time = (base_time + complexity * self.duration_scale * 2 + self.rng.uniform(0, 1.5)) / num_stages

        for s in range(num_stages):
            req = np.zeros(self.traits_dim, dtype=int)
            num_required = self.random_int(1, min(3, self.traits_dim + 1))
            dims = self.random_choice(self.traits_dim, size=num_required, replace=False)
            for d in dims: req[d] = 1
            stages.append({"stage_id": s, "requirements": req, "status": req.copy(), "time": float(stage_time),
                           "time_left": float(stage_time), "finished": False})

        self.task_dic[tid] = {
            "ID": tid, "stages": stages, "current_stage_index": 0, "total_stages": len(stages),
            "requirements": stages[0]["requirements"].copy(), "status": stages[0]["status"].copy(),
            "time": stages[0]["time"], "location": location,
            "scene_complexity": complexity, "finished": False, "feasible_assignment": False,
            "members": [], "time_start": self.current_time, "time_finish": 0.0,
            "deadline": 0.0, "expired": False,
            "discovered_at": self.current_time, "discovered_by": discoverer,
        }
        self.task_source[tid] = source
        self._assign_deadlines([tid], base_time=self.current_time)

    def trigger_tidal_burst(self):
        if not self.enable_tidal_tasks or self.current_time - self.last_tidal_time < self.tidal_interval: return
        self.last_tidal_time = self.current_time
        num_new = self.random_int(self.tidal_burst_range[0], self.tidal_burst_range[1] + 1)
        new_tids = []

        for _ in range(num_new):
            tid = self.next_dynamic_task_id
            self.next_dynamic_task_id += 1
            self._spawn_new_task(tid, source="tidal", discoverer="drone_swarm")
            new_tids.append(tid)

        agent_ids = list(self.agent_dic.keys())
        num_notified = max(1, int(len(agent_ids) * self.tidal_notify_ratio))
        notified_agents = self.rng.choice(agent_ids, size=num_notified, replace=False)

        for aid in notified_agents:
            for tid in new_tids:
                self.refresh_task_belief(aid, tid, update_time=self.current_time)

    def trigger_discovered_burst(self):
        """周期性 discovered 任务爆发：模拟巡逻 agent 在某区域发现任务簇。
        与 tidal 的差异：
        - 更小批量、更长间隔（与 tidal 错峰）
        - 任务位置聚集在一个簇内（模拟局部发现）
        - 仅通知少量前线 agent，其余靠 gossip 传染
        """
        if not self.enable_discovered_tasks or self.current_time - self.last_discovered_time < self.discovered_interval:
            return
        self.last_discovered_time = self.current_time

        num_new = self.random_int(self.discovered_burst_range[0], self.discovered_burst_range[1] + 1)
        center = np.clip(self.rng.random(2), 0.1, 0.9)
        radius = float(self.discovered_cluster_radius)

        new_tids = []
        for _ in range(num_new):
            tid = self.next_dynamic_task_id
            self.next_dynamic_task_id += 1
            offset = self.rng.normal(0, radius, 2)
            location = np.clip(center + offset, 0, 1)
            self._spawn_new_task(tid, location=location, source="discovered", discoverer="agent_patrol")
            new_tids.append(tid)

        agent_ids = list(self.agent_dic.keys())
        if not agent_ids: return
        num_notified = max(1, int(len(agent_ids) * self.discovered_notify_ratio))
        notified_agents = self.rng.choice(agent_ids, size=num_notified, replace=False)

        for aid in notified_agents:
            for tid in new_tids:
                self.refresh_task_belief(aid, tid, update_time=self.current_time)

    def _count_enroute_and_working(self, task_id, exclude_agent_id=None):
        count = 0
        for aid, agent in self.agent_dic.items():
            if aid == exclude_agent_id: continue
            if (agent.get("target_type") == "task" and agent.get("target_id") == task_id) or (
                    agent.get("status") == "working" and agent.get("current_task") == task_id):
                count += 1
        return count

    def _compute_shaping_reward(self, agent_id, task_id):
        """
        即时 shaping reward（在 apply_action 时触发）
        修改：
        1. 引入优先级加权，高优先级任务的奖惩幅度更大
        2. 距离惩罚改为软惩罚（近距离几乎无惩罚，远距离才显著）
        3. 完全不匹配时惩罚降低（从 -0.30 降到 -0.10），减少对探索的抑制
        """
        agent, task = self.agent_dic[agent_id], self.task_dic[task_id]
        reward = 0.0
        deadline = float(task.get("deadline", self.current_time + 999))
        time_remaining = max(0.0, deadline - self.current_time)
        urgency = float(np.tanh(time_remaining / 50.0))  # 1.0=充裕, 0.0=紧急
        current_status = task["status"]
        remaining_needs = current_status.sum()
        total_others = self._count_enroute_and_working(task_id, exclude_agent_id=agent_id) + len(task["members"])

        # --- 拥挤惩罚 ---
        if remaining_needs <= 0:
            # 任务需求已满足，去了也没用
            reward -= self.shaping_crowd_penalty * (total_others + 1)
        elif total_others > 0:
            reward -= self.shaping_crowd_penalty * max(0, total_others - remaining_needs)

        # --- 孤独奖励：能填补空缺需求 ---
        if remaining_needs > 0:
            enroute_abilities = np.zeros(self.traits_dim)
            for other_aid, other_agent in self.agent_dic.items():
                if other_aid != agent_id and (
                        (other_agent.get("target_type") == "task" and other_agent.get("target_id") == task_id) or
                        (other_agent.get("status") == "working" and other_agent.get("current_task") == task_id)):
                    enroute_abilities += other_agent["abilities"]

            still_needed = np.maximum(
                task["requirements"] - (enroute_abilities + self.get_abilities(task["members"])), 0
            )
            my_fill = np.minimum(still_needed, agent["abilities"]).sum()

            if my_fill > 0:
                # ✅ 修改：优先级加权，高优先级任务奖励更大
                reward += self.shaping_lonely_bonus * float(my_fill) * (1.5 - urgency)
            elif still_needed.sum() <= 0 and total_others > 0:
                reward -= self.shaping_crowd_penalty * 0.5

        # --- 距离惩罚：改为软惩罚，近距离几乎无惩罚 ---
        dist_norm = float(np.linalg.norm(agent["location"] - task["location"])) / 1.414
        # ✅ 修改：用平方使近距离惩罚极小，远距离才显著惩罚
        reward -= self.shaping_distance_penalty * (dist_norm ** 2)

        # --- 能力匹配奖励 ---
        my_contribution = np.minimum(current_status, agent["abilities"]).sum()
        if my_contribution > 0:
            reward += 0.05 * float(my_contribution) * (1.5 - urgency)
        else:
            reward -= self.shaping_redundant_penalty * 0.33

        # --- 过期任务惩罚（恢复 5/25 版本）---
        if task.get("expired", False):
            reward -= 0.5

        return float(reward) * 1.0   # A方案：原 ×0.25 → ×1.0（信号强 4 倍）

    def get_stage_completion_reward(self):
        """
        ✅ 新增：阶段完成中间奖励，在 task_update 完成阶段时调用
        解决多阶段任务信用分配困难的问题
        返回 {agent_id: reward} 字典
        """
        rewards = {}
        for task in self.task_dic.values():
            if task["finished"]:
                continue
            stage_idx = task["current_stage_index"]
            if stage_idx == 0:
                continue
            # 检查上一阶段是否刚完成（通过 finished 标记）
            prev_stage = task["stages"][stage_idx - 1]
            if not prev_stage.get("finished", False) or prev_stage.get("reward_given", False):
                continue
            # 标记已发放，避免重复
            prev_stage["reward_given"] = True
            deadline = float(task.get("deadline", self.current_time + 999))
            urgency = 1.5 - float(np.tanh(max(0.0, deadline - self.current_time) / 50.0))
            stage_reward = 5.0 * urgency  # 阶段完成奖励，越紧急权重越高
            for mid in task["members"]:
                rewards[mid] = rewards.get(mid, 0.0) + stage_reward
        return rewards

    def get_episode_reward(self, max_time=100.0):
        """
        Deadline-based episode reward:
        - R_slack: 早于deadline完成的slack奖励
        - R_expired: 过期任务重罚
        - 所有任务等权（无priority加权）
        """
        task_ids = list(self.task_dic.keys())
        if not task_ids:
            return 0.0, []

        T_norm = float(max(1.0, max_time, self.current_time))
        n_total = len(task_ids)

        # 分类统计：按时完成 / 过期完成 / 过期未完成 / 未完成
        n_finished_on_time = sum(1 for tid in task_ids
                                 if self.task_dic[tid].get("finished", False)
                                 and not self.task_dic[tid].get("expired", False))
        n_finished_late = sum(1 for tid in task_ids
                              if self.task_dic[tid].get("finished", False)
                              and self.task_dic[tid].get("expired", False))
        n_expired_unfinished = sum(1 for tid in task_ids
                                    if not self.task_dic[tid].get("finished", False)
                                    and self.task_dic[tid].get("expired", False))
        n_unfinished = n_total - n_finished_on_time - n_finished_late

        # ========== 1. 任务完成率主奖励 ==========
        A = 300.0
        completion_reward = A * (n_finished_on_time + 0.2 * n_finished_late) / n_total

        # ========== 2. 完成速度 & slack 奖励（仅准时完成享有）==========
        lam = 80.0
        speed_reward = 0.0
        slack_reward = 0.0
        for tid in task_ids:
            task = self.task_dic[tid]
            if task.get("finished", False) and not task.get("expired", False):
                C = float(task.get("time_finish", T_norm))
                speed_reward += lam * max(0.0, 1.0 - C / T_norm)
                slack = max(0.0, task["deadline"] - C)
                slack_reward += lam * min(1.0, slack / max(1.0, T_norm))
        speed_reward /= max(1.0, float(n_total))
        slack_reward /= max(1.0, float(n_total))

        # ========== 3. 未完成任务惩罚 ==========
        eta = 40.0
        unfinished_penalty = eta * (n_unfinished / max(1.0, float(n_total)))

        # ========== 4. 过期任务惩罚 ==========
        omega = 200.0
        n_expired = n_finished_late + n_expired_unfinished
        expired_penalty = omega * (n_expired / max(1.0, float(n_total)))

        finished_flags = np.array(
            [bool(self.task_dic[tid].get("finished", False)) for tid in task_ids], dtype=bool
        )

        # ========== 5. 阶段完成率软奖励 ==========
        gamma_stage = 20.0
        total_stages = sum(task.get("total_stages", 1) for task in self.task_dic.values())
        completed_stages = 0.0
        for task in self.task_dic.values():
            ts = task.get("total_stages", 1)
            if task["finished"]:
                completed_stages += ts
            else:
                completed_stages += task.get("current_stage_index", 0)
        stage_reward = gamma_stage * (completed_stages / max(1.0, float(total_stages)))

        # ========== 6. failed_tasks 惩罚 ==========
        phi = 1.0
        failed_penalty = phi * float(
            sum(len(a.get("failed_tasks", [])) for a in self.agent_dic.values())
        )

        # ========== 7. makespan 惩罚 ==========
        mu = 15.0
        makespan = max(
            (self.task_dic[tid].get("time_finish", 0.0) for tid in task_ids
             if self.task_dic[tid].get("finished", False)),
            default=float(self.current_time)
        )
        makespan_penalty = mu * (makespan / T_norm)

        total_reward = (
                completion_reward
                + speed_reward
                + slack_reward
                + stage_reward
                - unfinished_penalty
                - expired_penalty
                - makespan_penalty
                - failed_penalty
        )

        return float(total_reward), finished_flags.tolist()

    def _check_working_timeouts(self):
        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "working" or agent.get("working_start_time") is None or agent.get(
                "working_timeout_value") is None: continue
            if self.current_time - agent["working_start_time"] < agent["working_timeout_value"]: continue

            task_id = agent.get("current_task", -1)
            if task_id >= 0 and task_id in self.task_dic:
                task = self.task_dic[task_id]
                if aid in task["members"]: task["members"].remove(aid)
                stage = task["stages"][task["current_stage_index"]]
                stage["status"] = np.maximum(stage["requirements"] - self.get_abilities(task["members"]), 0)
                task["status"] = stage["status"].copy()
                agent["cooldown_tasks"][task_id] = self.current_time + self.cooldown_duration
                agent["failed_tasks"].append(
                    {"task_id": task_id, "time": float(self.current_time), "reason": "Timeout"})
            agent["consecutive_timeouts"] = agent.get("consecutive_timeouts", 0) + 1
            self._release_agent(aid)

    def _cleanup_cooldowns(self):
        for agent in self.agent_dic.values():
            if agent.get("cooldown_tasks"):
                expired = [tid for tid, end_time in agent["cooldown_tasks"].items() if self.current_time >= end_time]
                for tid in expired: del agent["cooldown_tasks"][tid]

    def get_hard_mask(self, agent_id):
        # 不变量 9：mask 只能来自该机器人自己的 belief。阶段索引与未来阶段需求
        # 读 belief 副本，断联期间真值推进不得改变这台机器人的合法动作集合。
        agent, knowledge, mask = self.agent_dic[agent_id], self.agent_knowledge[agent_id], [False]
        pending_tasks = knowledge["known_tasks"] - knowledge["known_finished"]
        for tid in sorted(pending_tasks):
            if tid not in self.task_dic: continue
            current_status = knowledge["task_current_status"].get(tid)
            if current_status is None:
                # belief 不完整（正常流程不会发生）：保守起见不可选。
                mask.append(True)
                continue
            future_req = knowledge["task_future_requirements"].get(tid)
            if future_req is None:
                future_req = np.zeros(self.traits_dim)
            if np.minimum(current_status, agent["abilities"]).sum() <= 0 and np.minimum(
                    future_req, agent["abilities"]).sum() <= 0:
                mask.append(True);
                continue
            if tid in agent.get("cooldown_tasks", {}) and self.current_time < agent["cooldown_tasks"][tid]:
                mask.append(True);
                continue
            mask.append(False)
        return np.array(mask, dtype=bool)

    def get_soft_mask(self, agent_id):
        return self.get_hard_mask(agent_id)

    def get_ctde_mask(self, agent_id, is_training=True):
        return self.get_soft_mask(agent_id) if is_training else self.get_hard_mask(agent_id)

    def get_local_obs(self, agent_id, max_tasks=None):
        agent, knowledge, LOC_NORM = self.agent_dic[agent_id], self.agent_knowledge[agent_id], 1.414
        self_state = np.hstack(
            [agent["abilities"], agent["location"], float(agent["current_task"]), float(agent["travel_dist"]),
             float(agent.get("sum_waiting_time", 0))]).astype(np.float32)

        neighbors = []
        for nid in self.comm_graph.get(agent_id, []):
            nb = self.agent_dic[nid]
            neighbors.append(np.hstack(
                [nb["abilities"], nb["location"], float(nb["current_task"]), float(nb["assigned"]),
                 float(nb["travel_dist"])]))
        neighbors = np.array(neighbors, dtype=np.float32) if neighbors else np.zeros((1, 10), dtype=np.float32)
        neighbors = np.vstack([neighbors, np.zeros((20 - neighbors.shape[0], 10), dtype=np.float32)]) if \
        neighbors.shape[0] < 20 else neighbors[:20]

        tasks, action_map = [np.hstack(
            [np.zeros(5), np.zeros(5), (self.depot_dic[agent["species"]]["location"] - agent["location"]) / LOC_NORM,
             [0.0] * 8]).astype(np.float32)], [("depot", None)]

        pending_tasks = knowledge["known_tasks"] - knowledge["known_finished"]
        for tid in sorted(pending_tasks):
            if tid not in self.task_dic: continue
            # 不变量 9：观察特征全部来自该机器人的 belief 副本，绝不回退读 task_dic
            # 真值（阶段索引、未来需求、位置、截止时间、复杂度都是消息到达时的快照）。
            status = knowledge["task_current_status"].get(tid)
            location = knowledge["task_locations"].get(tid)
            if status is None or location is None:
                continue
            status_feat = np.asarray(status, dtype=np.float32).flatten()
            future_req = np.asarray(
                knowledge["task_future_requirements"].get(tid, np.zeros(self.traits_dim, dtype=int)),
                dtype=np.float32,
            )
            num_members = float(len(knowledge["task_members"].get(tid, []))) / 10.0

            deadline_val = float(knowledge["task_deadlines"].get(tid, self.current_time))
            time_until_deadline = max(0.0, deadline_val - self.current_time)
            deadline_urgency = float(np.tanh(time_until_deadline / 50.0))
            base_feat = np.hstack([status_feat, future_req.flatten(),
                                   ((location - agent["location"]) / LOC_NORM).flatten(), [deadline_urgency],
                                   [float(knowledge["task_scene_complexity"].get(tid, 0.5))], [num_members]])
            info_feat = np.array([float(tid in knowledge["known_finished"]), float(tid in knowledge["known_assigned"]),
                                  float(tid in knowledge["dynamic_tasks"]), np.tanh(
                    float(self.current_time - float(knowledge["task_last_update_time"].get(tid, 0.0))) / 50.0),
                                  min(time_until_deadline / 100.0, 1.0)], dtype=np.float32)

            feat = np.hstack([base_feat, info_feat]).astype(np.float32)
            tasks.append(feat if feat.shape[0] == 20 else np.zeros(20, dtype=np.float32))
            action_map.append(("task", int(tid)))
            if max_tasks is not None and len(tasks) >= max_tasks: break

        tasks_array = np.array(tasks, dtype=np.float32)
        mask = self.get_ctde_mask(agent_id, is_training=True)[: len(action_map)]

        if max_tasks is not None and tasks_array.shape[0] < max_tasks:
            tasks_array = np.vstack([tasks_array, np.zeros((max_tasks - tasks_array.shape[0], 20), dtype=np.float32)])
            mask = np.concatenate([mask, np.ones((max_tasks - len(mask),), dtype=bool)])

        if mask.all(): mask[0] = False
        return self_state, neighbors, tasks_array, mask.astype(np.bool_), action_map

    def get_global_state(self):
        LOC_NORM = 1.0
        g_agents = [np.hstack(
            [a["abilities"], a["location"] / LOC_NORM, float(a["current_task"]), float(a["assigned"]), 0.0, 0.0,
             float(a["travel_dist"]), float(a.get("sum_waiting_time", 0.0))]) for a in self.agent_dic.values()]
        depot0 = self.depot_dic[0] if 0 in self.depot_dic else {"location": np.array([0.5, 0.5])}
        g_tasks = [np.hstack(
            [np.zeros(5), np.zeros(5), -np.ones(5), depot0["location"] / LOC_NORM, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

        for task in self.task_dic.values():
            if task.get("finished", False): continue
            future_req = task["stages"][task["current_stage_index"] + 1]["requirements"] if task[
                                                                                                "current_stage_index"] + 1 < \
                                                                                            task[
                                                                                                "total_stages"] else np.zeros(
                self.traits_dim)
            deadline_norm = float(task.get("deadline", self.current_time + 999)) / max(1.0, self.current_time + 100)
            g_tasks.append(np.hstack([task["status"], future_req, task["requirements"], task["location"] / LOC_NORM,
                                      float(task["feasible_assignment"]), deadline_norm,
                                      float(task.get("scene_complexity", 0.5)), float(task.get("time_start", 0.0)),
                                      float(task.get("time_finish", 0.0)), float(len(task.get("members", [])))]))

        return np.array(g_agents, dtype=np.float32), np.array(g_tasks, dtype=np.float32)

    def verify_task_on_arrival(self, agent_id, task_id):
        if task_id < 0: return True, "Depot", False
        task, agent, knowledge = self.task_dic[task_id], self.agent_dic[agent_id], self.agent_knowledge[agent_id]
        # 到场核实：机器人就在任务点，用现场真值刷新 belief（含阶段信息）。
        self.refresh_task_belief(agent_id, task_id, update_time=self.current_time)

        if task["finished"]:
            knowledge["known_finished"].add(task_id)
            return False, "Task already finished", True

        future_req = task["stages"][task["current_stage_index"] + 1]["requirements"] if task[
                                                                                            "current_stage_index"] + 1 < \
                                                                                        task[
                                                                                            "total_stages"] else np.zeros(
            self.traits_dim)
        if np.minimum(task["status"], agent["abilities"]).sum() <= 0 and np.minimum(future_req,
                                                                                    agent["abilities"]).sum() <= 0:
            return False, "Cannot contribute", True
        return True, "Valid", False

    def apply_action(self, agent_id, action_index, action_map):
        agent = self.agent_dic[agent_id]
        kind, tid = action_map[action_index] if 0 <= action_index < len(action_map) else ("depot", None)
        agent["immediate_replan"] = False

        if kind == "depot":
            agent.update({"target_type": "depot", "target_id": None,
                          "target_pos": self.depot_dic[agent["species"]]["location"].copy(), "status": "moving",
                          "is_rushing": False, "current_task": -1, "need_decision": False,
                          "target_task_expected_state": None, "working_start_time": None,
                          "working_timeout_value": None})
            return 0.0, False, []

        task_id = int(tid)
        if task_id not in self.task_dic:
            agent.update({"need_decision": True, "status": "idle"})
            return -1.0, True, []

        agent.update(
            {"target_type": "task", "target_id": task_id, "target_pos": self.task_dic[task_id]["location"].copy(),
             "status": "moving", "is_rushing": True, "current_task": task_id, "need_decision": False,
             "target_task_expected_state": self.task_dic[task_id]["status"].copy(), "working_start_time": None,
             "working_timeout_value": None, "consecutive_timeouts": 0})
        return self._compute_shaping_reward(agent_id, task_id), False, []

    def step(self, dt=None):
        dt = float(dt) if dt is not None else self.dt

        self._inner_step_counter += 1

        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "moving" or agent.get("target_pos") is None: continue
            dist = float(np.linalg.norm(agent["target_pos"] - agent["location"]))
            if dist <= EPS_ARRIVE: agent["location"] = agent["target_pos"].copy(); continue
            move = min(agent["velocity"] * dt, dist)
            agent["location"] += (agent["target_pos"] - agent["location"]) / dist * move
            agent["travel_dist"] += move

        self.current_time += dt

        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "moving" or agent.get("target_pos") is None or float(
                np.linalg.norm(agent["location"] - agent["target_pos"])) > EPS_ARRIVE: continue

            if agent.get("target_type") == "depot":
                agent.update({"status": "idle", "need_decision": True, "target_type": None, "target_id": None,
                              "target_pos": None, "is_rushing": False})
                agent["route"].append(self.depot_dic[agent["species"]]["ID"])
                agent["arrival_time"].append(float(self.current_time))
                continue

            if agent.get("target_type") == "task" and agent.get("target_id") is not None:
                task_id = int(agent["target_id"])
                agent["route"].append(task_id)
                agent["arrival_time"].append(float(self.current_time))

                is_valid, reason, _ = self.verify_task_on_arrival(aid, task_id)
                if not is_valid:
                    agent["failed_tasks"].append(
                        {"task_id": task_id, "time": float(self.current_time), "reason": reason})
                    self._release_agent(aid)
                    continue

                if aid not in self.task_dic[task_id]["members"]: self.task_dic[task_id]["members"].append(aid)
                agent.update({"assigned": True, "status": "working", "need_decision": False, "is_rushing": False,
                              "target_type": None, "target_id": None, "target_pos": None,
                              "working_start_time": self.current_time,
                              "working_timeout_value": self._sample_working_timeout(aid)})

        self.update_communication_graph()
        self.trigger_tidal_burst()
        self.trigger_discovered_burst()
        self.task_update()

        # gossip 降频：每 gossip_interval 步执行1次传播（减少CPU开销~40%）
        self._step_counter += 1
        if self._step_counter % self.gossip_interval == 0:
            self.propagate_task_completion()

        self._check_working_timeouts()
        self._cleanup_cooldowns()

        active_count = 0
        for agent in self.agent_dic.values():
            if agent.get("status") != "idle":
                active_count += 1
        self._active_agent_steps += active_count
        self._total_agent_steps += max(1, len(self.agent_dic))

        # ✅ 修复：取消“提前下班”机制（强制值班制），把结束权交给 driver 里的时间限制
        self.finished = False
        return self.finished

    def task_update(self):
        f_task = []
        stage_rewards = {}

        # deadline 过期检测（软约束：过期不删除，不释放成员，不标记完成）
        for tid, task in self.task_dic.items():
            if task.get("expired") or task.get("finished"):
                continue
            if self.current_time > task["deadline"]:
                if self.hard_expire:
                    # 消融：硬过期——任务直接 finished，从可选池移除
                    task["finished"] = True
                    task["expired"] = True
                else:
                    task["expired"] = True

        for task in self.task_dic.values():
            if task["finished"]: continue

            stage = task["stages"][task["current_stage_index"]]
            stage["status"] = np.maximum(stage["requirements"] - self.get_abilities(task["members"]), 0)
            task["status"], task["requirements"], task["time"] = (
                stage["status"].copy(), stage["requirements"].copy(), stage["time"]
            )

            if (stage["status"] <= 0).all() and len(task["members"]) > 0:
                task["feasible_assignment"] = True
                stage["time_left"] -= self.dt

                if stage["time_left"] <= 0:
                    stage["finished"] = True
                    task["current_stage_index"] += 1
                    task["feasible_assignment"] = False

                    if task["current_stage_index"] >= task["total_stages"]:
                        task["finished"] = True
                        task["time_finish"] = self.current_time
                        f_task.append(task["ID"])
                        for m in list(task["members"]): self._release_agent(m)
                        task["members"] = []
                    else:
                        # ✅ 新增：阶段完成，给当前成员发放中间奖励
                        deadline = float(task.get("deadline", self.current_time + 999))
                        urgency = 1.5 - float(np.tanh(max(0.0, deadline - self.current_time) / 50.0))
                        for m in task["members"]:
                            stage_rewards[m] = stage_rewards.get(m, 0.0) + 5.0 * urgency

                        next_req = task["stages"][task["current_stage_index"]]["requirements"]
                        task["status"] = next_req.copy()
                        task["requirements"] = next_req.copy()
                        task["time"] = task["stages"][task["current_stage_index"]]["time"]
                        keep_members = [
                            m for m in list(task["members"])
                            if m in self.agent_dic and np.minimum(next_req, self.agent_dic[m]["abilities"]).sum() > 0
                        ]
                        for m in list(task["members"]):
                            if m not in keep_members: self._release_agent(m)
                        task["members"] = keep_members
            else:
                task["feasible_assignment"] = False
                stage["time_left"] = stage["time"]

        # ✅ 把阶段奖励挂到 _step_events，供 driver 读取
        if stage_rewards:
            self._step_events.append({"type": "stage_complete", "rewards": stage_rewards})

        return f_task

    def check_agent_enroute_replan(self, agent_id):
        agent = self.agent_dic[agent_id]
        if agent.get("target_type") != "task" or agent.get("target_id") is None: return False, "Not enroute"
        task_id = int(agent["target_id"])
        if task_id not in self.task_dic: return True, "Target no longer exists"
        task = self.task_dic[task_id]
        if task_id in self.agent_knowledge[agent_id]["known_finished"] or task["finished"]: return True, "Task finished"
        if np.minimum(task["status"], agent["abilities"]).sum() <= 0: return True, "Cannot contribute"
        return False, "OK"


    def get_abilities(self, members):
        if not members: return np.zeros(self.traits_dim)
        return np.sum([self.agent_dic[m]["abilities"] for m in members if m in self.agent_dic], axis=0)

    def cleanup_failed_tasks_memory(self):
        for agent in self.agent_dic.values():
            agent["failed_tasks"] = [r for r in agent["failed_tasks"] if
                                     self.current_time - r["time"] <= 10.0 and r["task_id"] in self.task_dic and not
                                     self.task_dic[r["task_id"]]["finished"]]