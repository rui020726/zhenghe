# -*- coding: utf-8 -*-
"""
时间驱动任务分配环境（CTDE范式 + 动态任务发现）
- dt=1 时间驱动推进
- 移动过程中持续通信/传播
- 动作(action) ↔ 任务(task_id) 使用 action_map 显式映射（彻底修复错位）
- depot 固定为 action=0，支持：仓库出发→执行任务→返回仓库

新增（去聚堆）：
- 局部可见意图锁定（reservation，不泄露全局）
- 拥挤门控(cap)掩码
- 扑空后冷却(cooldown)掩码（局部传播）
- 无效到达即时惩罚（env.last_step_reward）
"""

import numpy as np
import torch

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
# deterministic 已移除（拖慢 30-50%，且本代码不需要严格确定性）

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
        # ===== anti-crowding =====
        crowding_cap_factor=1.0,
        crowding_min_cap=1,
        cooldown_time=15.0,
        invalid_arrival_penalty=-2.0,
        **kwargs,
    ):
        tightness = deadline_tightness
        if "prio_range" in kwargs:
            tightness = kwargs.pop("prio_range")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.ranges = {
            "deadline_tightness": tightness,
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

        # anti-crowding params
        self.crowding_cap_factor = float(crowding_cap_factor)
        self.crowding_min_cap = int(crowding_min_cap)
        self.cooldown_time = float(cooldown_time)
        self.invalid_arrival_penalty = float(invalid_arrival_penalty)

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

        for a in self.agent_dic.values():
            a["need_decision"] = True

        self._step_events = []
        self.last_step_reward = 0.0  # step内即时奖励缓存

    # ─────────────────────────────────────────────────────────────
    # RNG helpers
    # ─────────────────────────────────────────────────────────────
    def random_int(self, low, high, size=None):
        if low == high:
            return low if size is None else np.full(size, low, dtype=int)
        return self.rng.integers(low, high, size)

    def random_val(self, shape):
        return self.rng.random(shape)

    def random_choice(self, a, size=None, replace=True):
        return self.rng.choice(a, size, replace=replace)

    # ─────────────────────────────────────────────────────────────
    # Env generation
    # ─────────────────────────────────────────────────────────────
    def generate_task(self, tasks_num):
        tasks_ini = np.zeros((tasks_num, self.traits_dim), dtype=int)

        for i in range(tasks_num):
            num_required_dims = self.random_int(1, min(4, self.traits_dim + 1))
            required_dims = self.random_choice(self.traits_dim, size=num_required_dims, replace=False)
            for dim in required_dims:
                tasks_ini[i, dim] = 1

        for i in range(tasks_num):
            if np.sum(tasks_ini[i, :]) == 0:
                dim = self.random_int(0, self.traits_dim)
                tasks_ini[i, dim] = 1

        scene_complexities = self.rng.uniform(0.2, 1.0, tasks_num)

        base_time = 2.0
        complexity_scale = self.duration_scale * 2
        noise = self.rng.uniform(0, 1.5, tasks_num)
        tasks_time = base_time + scene_complexities * complexity_scale + noise

        return tasks_ini, scene_complexities, tasks_time

    def generate_agent(self, species_num):
        agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))
        while not np.all(np.sum(agents_ini, axis=1) != 0):
            agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))

        while np.unique(agents_ini, axis=0).shape[0] != species_num:
            agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))

        for dim in range(self.traits_dim):
            if agents_ini[:, dim].sum() == 0:
                random_species = self.random_int(0, species_num)
                agents_ini[random_species, dim] = 1
        return agents_ini

    def generate_env(self):
        n_tasks = self.random_int(self.ranges["tasks"][0], self.ranges["tasks"][1])
        n_species = self.random_int(self.ranges["species"][0], self.ranges["species"][1])
        agents_species_num = [
            self.random_int(self.ranges["per_species"][0], self.ranges["per_species"][1]) for _ in range(n_species)
        ]

        agents_ini = self.generate_agent(n_species)

        covered_abilities = agents_ini.sum(axis=0) > 0
        for dim in range(self.traits_dim):
            if not covered_abilities[dim]:
                species_idx = self.random_int(0, n_species)
                agents_ini[species_idx, dim] = 1

        covered_abilities = agents_ini.sum(axis=0) > 0
        if not covered_abilities.all():
            return self.generate_env()

        tasks_ini, scene_complexities, tasks_time = self.generate_task(n_tasks)

        depot_loc = self.random_val((n_species, 2))
        tasks_loc = self.random_val((n_tasks, 2))

        task_dic = {}
        agent_dic = {}
        depot_dic = {}
        species_dict = {"abilities": agents_ini, "number": agents_species_num}

        for i in range(n_tasks):
            task_dic[i] = {
                "ID": i,
                "requirements": tasks_ini[i, :],
                "status": tasks_ini[i, :].copy(),
                "location": tasks_loc[i, :],
                "time": float(tasks_time[i]),
                "scene_complexity": float(scene_complexities[i]),
                "finished": False,
                "feasible_assignment": False,
                "members": [],
                "time_start": 0.0,
                "time_finish": 0.0,
                "deadline": float("inf"),
                "expired": False,
            }
            self.task_source[i] = "initial"

        cnt = 0
        for s in range(n_species):
            depot_dic[s] = {"ID": -s - 1, "location": depot_loc[s], "members": []}
            for _ in range(agents_species_num[s]):
                agent_dic[cnt] = {
                    "ID": cnt,
                    "species": s,
                    "abilities": agents_ini[s, :],
                    "location": depot_loc[s].copy(),
                    "depot": depot_loc[s].copy(),
                    "velocity": 0.2,
                    "current_task": -1,
                    "travel_dist": 0.0,
                    "sum_waiting_time": 0.0,
                    "route": [],
                    "arrival_time": [0.0],
                    "assigned": False,
                    "is_rushing": False,
                    "target_task_expected_state": None,
                    "failed_tasks": [],
                    "immediate_replan": False,

                    "need_decision": True,
                    "status": "idle",       # idle / moving / working
                    "target_type": None,    # None / "task" / "depot"
                    "target_id": None,      # task_id if target_type=="task"
                    "target_pos": None,     # np.array([x,y])
                }
                depot_dic[s]["members"].append(cnt)
                cnt += 1

        self._assign_deadlines(task_dic, depot_dic)

        return task_dic, agent_dic, depot_dic, species_dict

    def _assign_deadlines(self, task_dic, depot_dic):
        tight_min, tight_max = self.ranges["deadline_tightness"]
        for tid, task in task_dic.items():
            if task.get("finished"):
                continue
            min_travel = min(
                np.linalg.norm(task["location"] - d["location"]) / 0.2
                for d in depot_dic.values()
            )
            work_time = float(task["requirements"].sum()) * self.duration_scale * 2
            base_time = task.get("discovered_at", 0.0)
            estimated = min_travel + work_time + 10.0
            tightness = self.rng.uniform(tight_min, tight_max)
            deadline = base_time + max(estimated * tightness, work_time * 1.1)
            task["deadline"] = float(deadline)

    def _release_agent_from_task(self, aid):
        agent = self.agent_dic[aid]
        agent["status"] = "idle"
        agent["current_task"] = -1
        agent["target_type"] = None
        agent["target_id"] = None
        agent["target_pos"] = None
        agent["assigned"] = False
        agent["is_rushing"] = False
        agent["need_decision"] = True

    # ─────────────────────────────────────────────────────────────
    # Knowledge base
    # ─────────────────────────────────────────────────────────────
    def init_agent_memory(self):
        self.agent_knowledge = {
            aid: {
                "known_tasks": set(self.task_dic.keys()),
                "task_locations": {tid: task["location"].copy() for tid, task in self.task_dic.items()},
                "task_requirements": {tid: task["requirements"].copy() for tid, task in self.task_dic.items()},
                "dynamic_tasks": set(),
                "known_finished": set(),
                "known_assigned": set(),
                "task_current_status": {tid: task["status"].copy() for tid, task in self.task_dic.items()},
                "task_members": {tid: [] for tid in self.task_dic.keys()},
                "task_reserved_members": {tid: [] for tid in self.task_dic.keys()},
                "task_last_update_time": {tid: 0.0 for tid in self.task_dic.keys()},
                "task_cooldown_until": {},  # tid -> time
            }
            for aid in self.agent_dic.keys()
        }

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.next_dynamic_task_id = self.tasks_num
        self.init_agent_memory()

        self.current_time = 0.0
        self.finished = False
        self.comm_graph = {}
        self._step_events = []
        self.last_step_reward = 0.0

        for a in self.agent_dic.values():
            a["need_decision"] = True
            a["status"] = "idle"
            a["target_type"] = None
            a["target_id"] = None
            a["target_pos"] = None
            a["is_rushing"] = False
            a["immediate_replan"] = False
            a["current_task"] = -1

        return self.get_global_state()

    # ─────────────────────────────────────────────────────────────
    # Reservation / cooldown helpers (local visibility only)
    # ─────────────────────────────────────────────────────────────
    def _ensure_task_memory_fields(self, aid, tid):
        k = self.agent_knowledge[aid]
        if tid not in k["task_reserved_members"]:
            k["task_reserved_members"][tid] = []
        if tid not in k["task_members"]:
            k["task_members"][tid] = []
        if tid not in k["task_current_status"]:
            if tid in self.task_dic:
                k["task_current_status"][tid] = self.task_dic[tid]["status"].copy()
            else:
                k["task_current_status"][tid] = np.zeros(self.traits_dim)
        if tid not in k["task_last_update_time"]:
            k["task_last_update_time"][tid] = self.current_time

    def get_known_reserved_abilities(self, observer_id, task_id):
        k = self.agent_knowledge[observer_id]
        reserved = k.get("task_reserved_members", {}).get(task_id, [])
        valid = [rid for rid in reserved if rid in self.agent_dic]
        if len(valid) == 0:
            return np.zeros(self.traits_dim, dtype=float)
        return np.sum([self.agent_dic[rid]["abilities"] for rid in valid], axis=0)

    def get_task_crowding_cap(self, task_id):
        if task_id not in self.task_dic:
            return self.crowding_min_cap
        req = self.task_dic[task_id]["requirements"]
        base = int(np.sum(req > 0))
        cap = int(max(self.crowding_min_cap, np.ceil(base * self.crowding_cap_factor)))
        return cap

    def set_task_cooldown(self, agent_id, task_id, until_time):
        if agent_id not in self.agent_knowledge:
            return
        k = self.agent_knowledge[agent_id]
        if "task_cooldown_until" not in k:
            k["task_cooldown_until"] = {}
        k["task_cooldown_until"][int(task_id)] = float(until_time)

    def propagate_cooldown_local(self, source_agent_id, task_id, until_time):
        self.set_task_cooldown(source_agent_id, task_id, until_time)
        for nid in self.comm_graph.get(source_agent_id, []):
            self.set_task_cooldown(nid, task_id, until_time)

    def broadcast_intention_local(self, agent_id, task_id):
        if task_id not in self.task_dic:
            return

        my_k = self.agent_knowledge[agent_id]
        self._ensure_task_memory_fields(agent_id, task_id)
        if agent_id not in my_k["task_reserved_members"][task_id]:
            my_k["task_reserved_members"][task_id].append(agent_id)
        my_k["task_last_update_time"][task_id] = self.current_time

        for nid in self.comm_graph.get(agent_id, []):
            nk = self.agent_knowledge[nid]
            if task_id not in nk["known_tasks"]:
                continue
            self._ensure_task_memory_fields(nid, task_id)
            if agent_id not in nk["task_reserved_members"][task_id]:
                nk["task_reserved_members"][task_id].append(agent_id)
            nk["task_last_update_time"][task_id] = self.current_time

    def remove_reservation_local(self, agent_id, task_id):
        if task_id is None:
            return

        if agent_id in self.agent_knowledge:
            k = self.agent_knowledge[agent_id]
            self._ensure_task_memory_fields(agent_id, task_id)
            if agent_id in k["task_reserved_members"][task_id]:
                k["task_reserved_members"][task_id].remove(agent_id)
            k["task_last_update_time"][task_id] = self.current_time

        for nid in self.comm_graph.get(agent_id, []):
            nk = self.agent_knowledge[nid]
            if task_id not in nk.get("known_tasks", set()):
                continue
            self._ensure_task_memory_fields(nid, task_id)
            if agent_id in nk["task_reserved_members"][task_id]:
                nk["task_reserved_members"][task_id].remove(agent_id)
            nk["task_last_update_time"][task_id] = self.current_time

    # ─────────────────────────────────────────────────────────────
    # Communication + propagation
    # ─────────────────────────────────────────────────────────────
    def update_communication_graph(self):
        self.comm_graph = {aid: [] for aid in self.agent_dic.keys()}
        agent_ids = list(self.agent_dic.keys())

        for i, aid_i in enumerate(agent_ids):
            loc_i = self.agent_dic[aid_i]["location"]
            for j, aid_j in enumerate(agent_ids):
                if i == j:
                    continue
                loc_j = self.agent_dic[aid_j]["location"]
                if np.linalg.norm(loc_i - loc_j) <= self.comm_range:
                    self.comm_graph[aid_i].append(aid_j)

    def propagate_task_completion(self):
        # 视距感知
        for aid in self.agent_dic.keys():
            agent = self.agent_dic[aid]
            knowledge = self.agent_knowledge[aid]

            for tid, task in self.task_dic.items():
                distance = np.linalg.norm(agent["location"] - task["location"])
                if distance <= self.comm_range:
                    if tid not in knowledge["known_tasks"]:
                        knowledge["known_tasks"].add(tid)
                        knowledge["task_locations"][tid] = task["location"].copy()
                        knowledge["task_requirements"][tid] = task["requirements"].copy()
                        knowledge["task_current_status"][tid] = task["status"].copy()
                        knowledge["task_members"][tid] = []
                        knowledge["task_reserved_members"][tid] = []
                        knowledge["task_last_update_time"][tid] = self.current_time

                        if self.task_source.get(tid) == "discovered":
                            knowledge["dynamic_tasks"].add(tid)

                    if task["finished"]:
                        knowledge["known_finished"].add(tid)
                        knowledge["task_last_update_time"][tid] = self.current_time

                    if task["feasible_assignment"]:
                        knowledge["known_assigned"].add(tid)
                        knowledge["task_last_update_time"][tid] = self.current_time

                    knowledge["task_current_status"][tid] = task["status"].copy()
                    knowledge["task_members"][tid] = task["members"].copy()

                    reserved = knowledge.get("task_reserved_members", {}).get(tid, [])
                    knowledge["task_reserved_members"][tid] = [r for r in reserved if r not in task["members"]]
                    knowledge["task_last_update_time"][tid] = self.current_time

        # 一跳传播
        for aid in self.agent_dic.keys():
            neighbors = self.comm_graph.get(aid, [])
            knowledge = self.agent_knowledge[aid]

            for nid in neighbors:
                nk = self.agent_knowledge[nid]

                for tid in knowledge["known_tasks"]:
                    if tid not in nk["known_tasks"]:
                        if tid in self.task_dic:
                            nk["known_tasks"].add(tid)
                            nk["task_locations"][tid] = knowledge["task_locations"][tid].copy()
                            nk["task_requirements"][tid] = knowledge["task_requirements"][tid].copy()
                            nk["task_current_status"][tid] = knowledge["task_current_status"][tid].copy()
                            nk["task_members"][tid] = knowledge["task_members"][tid].copy()
                            nk["task_reserved_members"][tid] = knowledge.get("task_reserved_members", {}).get(tid, []).copy()
                            nk["task_last_update_time"][tid] = knowledge["task_last_update_time"][tid]
                            if tid in knowledge["dynamic_tasks"]:
                                nk["dynamic_tasks"].add(tid)

                for tid in knowledge["known_finished"]:
                    if tid in nk["known_tasks"] and tid not in nk["known_finished"]:
                        nk["known_finished"].add(tid)
                        nk["task_last_update_time"][tid] = knowledge["task_last_update_time"].get(tid, self.current_time)

                for tid in knowledge["known_assigned"]:
                    if tid in nk["known_tasks"] and tid not in nk["known_assigned"]:
                        nk["known_assigned"].add(tid)
                        nk["task_last_update_time"][tid] = knowledge["task_last_update_time"].get(tid, self.current_time)

                for tid in knowledge["known_tasks"]:
                    if tid not in nk["known_tasks"]:
                        continue
                    my_t = knowledge["task_last_update_time"].get(tid, 0.0)
                    nb_t = nk["task_last_update_time"].get(tid, 0.0)
                    if my_t > nb_t:
                        nk["task_current_status"][tid] = knowledge["task_current_status"][tid].copy()
                        nk["task_members"][tid] = knowledge["task_members"][tid].copy()
                        nk["task_reserved_members"][tid] = knowledge.get("task_reserved_members", {}).get(tid, []).copy()
                        nk["task_last_update_time"][tid] = my_t

                    # cooldown同步（更晚截止时间覆盖更早）
                    my_cd = knowledge.get("task_cooldown_until", {}).get(tid, -1.0)
                    nb_cd = nk.get("task_cooldown_until", {}).get(tid, -1.0)
                    if my_cd > nb_cd:
                        nk.setdefault("task_cooldown_until", {})[tid] = my_cd

        # 在途重规划
        for aid, agent in self.agent_dic.items():
            if agent.get("is_rushing", False) and agent.get("target_type") == "task" and agent.get("target_id") is not None:
                need_replan, _reason = self.check_agent_enroute_replan(aid)
                if need_replan:
                    self.remove_reservation_local(aid, int(agent["target_id"]))
                    agent["immediate_replan"] = True
                    agent["need_decision"] = True
                    agent["status"] = "idle"
                    agent["target_type"] = None
                    agent["target_id"] = None
                    agent["target_pos"] = None
                    agent["is_rushing"] = False
                    agent["current_task"] = -1

    # ─────────────────────────────────────────────────────────────
    # Dynamic task discovery
    # ─────────────────────────────────────────────────────────────
    def discover_dynamic_task(self, agent_id):
        if not self.dynamic_task_discovery:
            return None

        current_dynamic_count = sum(1 for source in self.task_source.values() if source == "discovered")
        if current_dynamic_count >= self.max_dynamic_tasks:
            return None

        agent = self.agent_dic[agent_id]
        if agent.get("target_type") != "task" and agent.get("current_task", -1) < 0:
            return None

        if self.rng.random() > self.discovery_probability:
            return None

        new_task_id = self.next_dynamic_task_id
        self.next_dynamic_task_id += 1

        offset = self.rng.uniform(-0.1, 0.1, size=2)
        new_location = np.clip(agent["location"] + offset, 0, 1)

        new_requirements = self.rng.integers(0, 2, size=self.traits_dim)
        while new_requirements.sum() == 0:
            new_requirements = self.rng.integers(0, 2, size=self.traits_dim)

        new_time = self.rng.uniform(2.0, 5.0)

        self.task_dic[new_task_id] = {
            "ID": new_task_id,
            "requirements": new_requirements,
            "status": new_requirements.copy(),
            "location": new_location,
            "time": float(new_time),
            "scene_complexity": float(self.rng.uniform(0.5, 0.9)),
            "finished": False,
            "feasible_assignment": False,
            "members": [],
            "time_start": 0.0,
            "time_finish": 0.0,
            "deadline": float("inf"),
            "expired": False,
            "discovered_at": self.current_time,
            "discovered_by": agent_id,
        }
        self._assign_deadlines({new_task_id: self.task_dic[new_task_id]}, self.depot_dic)
        self.task_source[new_task_id] = "discovered"

        k = self.agent_knowledge[agent_id]
        k["known_tasks"].add(new_task_id)
        k["dynamic_tasks"].add(new_task_id)
        k["task_locations"][new_task_id] = new_location.copy()
        k["task_requirements"][new_task_id] = new_requirements.copy()
        k["task_current_status"][new_task_id] = new_requirements.copy()
        k["task_members"][new_task_id] = []
        k["task_reserved_members"][new_task_id] = []
        k["task_last_update_time"][new_task_id] = self.current_time

        for nid in self.comm_graph.get(agent_id, []):
            nk = self.agent_knowledge[nid]
            nk["known_tasks"].add(new_task_id)
            nk["dynamic_tasks"].add(new_task_id)
            nk["task_locations"][new_task_id] = new_location.copy()
            nk["task_requirements"][new_task_id] = new_requirements.copy()
            nk["task_current_status"][new_task_id] = new_requirements.copy()
            nk["task_members"][new_task_id] = []
            nk["task_reserved_members"][new_task_id] = []
            nk["task_last_update_time"][new_task_id] = self.current_time

        return new_task_id

    # ─────────────────────────────────────────────────────────────
    # Masks (True = masked/illegal)
    # ─────────────────────────────────────────────────────────────
    def get_hard_mask(self, agent_id):
        _, _, _, mask, _ = self.get_local_obs(agent_id)
        return mask

    def get_soft_mask(self, agent_id):
        return self.get_hard_mask(agent_id)

    def get_ctde_mask(self, agent_id, is_training=True):
        return self.get_hard_mask(agent_id)

    # ─────────────────────────────────────────────────────────────
    # Local obs + action_map
    # ─────────────────────────────────────────────────────────────
    def get_local_obs(self, agent_id, max_tasks=None):
        agent = self.agent_dic[agent_id]
        knowledge = self.agent_knowledge[agent_id]

        LOC_NORM = 1.414

        self_state = np.hstack([
            agent["abilities"],
            agent["location"],
            float(agent["current_task"]),
            float(agent["travel_dist"]),
            float(agent.get("sum_waiting_time", 0))
        ]).astype(np.float32)

        neighbors = []
        for nid in self.comm_graph.get(agent_id, []):
            nb = self.agent_dic[nid]
            feat = np.hstack([
                nb["abilities"],
                nb["location"],
                float(nb["current_task"]),
                float(nb["assigned"]),
                float(nb["travel_dist"]),
            ])
            neighbors.append(feat)

        if len(neighbors) == 0:
            neighbors = np.zeros((1, 10), dtype=np.float32)
        else:
            neighbors = np.array(neighbors, dtype=np.float32)

        max_neighbors = 20
        if neighbors.shape[0] < max_neighbors:
            pad = np.zeros((max_neighbors - neighbors.shape[0], 10), dtype=np.float32)
            neighbors = np.vstack([neighbors, pad])
        else:
            neighbors = neighbors[:max_neighbors]

        tasks = []
        action_map = []
        mask = []

        depot = self.depot_dic[agent["species"]]
        depot_feat = np.hstack([
            np.zeros(5),
            (depot["location"] - agent["location"]) / LOC_NORM,
            [0.0], [0.0], [0.0],
            [0.0], [0.0], [0.0],
            [0.0], [0.0],
        ]).astype(np.float32)
        tasks.append(depot_feat)
        action_map.append(("depot", None))
        mask.append(False)

        for tid in sorted(knowledge["known_tasks"]):
            if tid not in self.task_dic:
                continue

            task = self.task_dic[tid]
            location = knowledge["task_locations"].get(tid, task["location"])
            current_status = knowledge["task_current_status"].get(
                tid, knowledge["task_requirements"].get(tid, task["requirements"])
            )

            # 1) cooldown先判
            cd_until = float(knowledge.get("task_cooldown_until", {}).get(tid, -1.0))
            if self.current_time < cd_until:
                is_masked = True
            # 2) finished
            elif tid in knowledge["known_finished"]:
                is_masked = True
            else:
                # 3) 拥挤门控
                reserved_ids = knowledge.get("task_reserved_members", {}).get(tid, [])
                reserved_ids = [rid for rid in reserved_ids if rid in self.agent_dic]
                known_members = knowledge.get("task_members", {}).get(tid, [])
                known_members = [mid for mid in known_members if mid in self.agent_dic]

                cap = self.get_task_crowding_cap(tid)
                known_count = len(set(known_members).union(set(reserved_ids)))
                if known_count >= cap:
                    is_masked = True
                else:
                    reserved_ability = self.get_known_reserved_abilities(agent_id, tid)
                    effective_need = np.maximum(current_status - reserved_ability, 0)
                    my_contribution = np.minimum(effective_need, agent["abilities"]).sum()
                    is_masked = (my_contribution <= 0)

            deadline = float(task.get("deadline", float("inf")))
            time_remaining = max(0.0, deadline - self.current_time)
            deadline_urgency = float(np.tanh(time_remaining / 50.0))
            rel = (location - agent["location"]) / LOC_NORM
            last_t = float(knowledge["task_last_update_time"].get(tid, 0.0))
            time_since_norm = np.tanh((self.current_time - last_t) / 50.0)
            num_members_known = float(len(knowledge["task_members"].get(tid, []))) / 10.0

            base_feat = np.hstack([
                current_status.astype(np.float32).flatten(),
                rel.flatten(),
                [deadline_urgency],
                [float(task.get("scene_complexity", 0.5))],
                [float(len(task.get("members", []))) / 10.0],
            ])
            info_feat = np.array([
                float(tid in knowledge["known_finished"]),
                float(tid in knowledge["known_assigned"]),
                float(tid in knowledge["dynamic_tasks"]),
                float(time_since_norm),
                float(num_members_known),
            ], dtype=np.float32)

            feat = np.hstack([base_feat, info_feat]).astype(np.float32)
            if feat.shape[0] != 15:
                feat = np.zeros(15, dtype=np.float32)

            tasks.append(feat)
            action_map.append(("task", int(tid)))
            mask.append(bool(is_masked))

            if max_tasks is not None and len(tasks) >= max_tasks:
                break

        tasks_array = np.array(tasks, dtype=np.float32)
        mask_arr = np.array(mask, dtype=bool)

        if mask_arr.all():
            mask_arr[0] = False

        assert tasks_array.shape[0] == len(action_map) == len(mask_arr), (
            f"对齐错误: tasks={tasks_array.shape[0]}, action_map={len(action_map)}, mask={len(mask_arr)}"
        )

        if max_tasks is not None and tasks_array.shape[0] < max_tasks:
            pad_t = np.zeros((max_tasks - tasks_array.shape[0], 15), dtype=np.float32)
            tasks_array = np.vstack([tasks_array, pad_t])
            pad_m = np.ones(max_tasks - len(mask_arr), dtype=bool)
            mask_arr = np.concatenate([mask_arr, pad_m])

        return self_state, neighbors, tasks_array, mask_arr.astype(np.bool_), action_map

    # ────────────────────────────────────────────���────────────────
    # Global state (critic)
    # ─────────────────────────────────────────────────────────────
    def get_global_state(self):
        LOC_NORM = 1.0

        g_agents = []
        for agent in self.agent_dic.values():
            travel_time = 0.0
            waiting_time = 0.0
            agent_feature = np.hstack(
                [
                    agent["abilities"],
                    agent["location"] / LOC_NORM,
                    float(agent["current_task"]),
                    float(agent["assigned"]),
                    float(travel_time),
                    float(waiting_time),
                    float(agent["travel_dist"]),
                    float(agent.get("sum_waiting_time", 0.0)),
                ]
            )
            g_agents.append(agent_feature)

        g_tasks = []
        depot0 = self.depot_dic[0] if 0 in self.depot_dic else {"location": np.array([0.5, 0.5])}
        depot_feature = np.hstack(
            [
                np.zeros(self.traits_dim),
                -np.ones(self.traits_dim),
                depot0["location"] / LOC_NORM,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        )
        g_tasks.append(depot_feature)

        for task in self.task_dic.values():
            task_feature = np.hstack(
                [
                    task["status"],
                    task["requirements"],
                    task["location"] / LOC_NORM,
                    float(task["feasible_assignment"]),
                    float(task.get("deadline", float("inf"))),
                    float(task.get("scene_complexity", 0.5)),
                    float(task.get("time_start", 0.0)),
                    float(task.get("time_finish", 0.0)),
                    float(len(task.get("members", []))),
                ]
            )
            g_tasks.append(task_feature)

        return np.array(g_agents, dtype=np.float32), np.array(g_tasks, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────────────────────────
    def calculate_eulidean_distance(self, entity1, entity2):
        return float(np.linalg.norm(entity1["location"] - entity2["location"]))

    def get_abilities(self, members):
        if len(members) == 0:
            return np.zeros(self.traits_dim)
        return np.sum([self.agent_dic[m]["abilities"] for m in members], axis=0)

    # ─────────────────────────────────────────────────────────────
    # Arrival/verify
    # ─────────────────────────────────────────────────────────────
    def verify_task_on_arrival(self, agent_id, task_id):
        if task_id < 0:
            return True, "Depot", False

        task = self.task_dic[task_id]
        agent = self.agent_dic[agent_id]
        knowledge = self.agent_knowledge[agent_id]

        knowledge["task_current_status"][task_id] = task["status"].copy()
        knowledge["task_members"][task_id] = task["members"].copy()
        knowledge["task_last_update_time"][task_id] = self.current_time

        if task["finished"]:
            knowledge["known_finished"].add(task_id)
            knowledge["task_last_update_time"][task_id] = self.current_time
            return False, "Task already finished", True

        if task["feasible_assignment"]:
            current_ability = self.get_abilities(task["members"])
            remaining_need = np.maximum(task["requirements"] - current_ability, 0)
            my_contribution = np.minimum(remaining_need, agent["abilities"]).sum()
            if my_contribution <= 0:
                knowledge["known_assigned"].add(task_id)
                knowledge["task_last_update_time"][task_id] = self.current_time
                return False, "Task already sufficiently staffed", True

        current_status = task["status"]
        my_contribution = np.minimum(current_status, agent["abilities"]).sum()
        if my_contribution <= 0:
            return False, "Cannot contribute to current task status", False

        if agent.get("target_task_expected_state") is not None:
            expected = agent["target_task_expected_state"]
            actual = task["status"]
            if np.sum(actual) < np.sum(expected) * 0.5:
                return False, "Task state changed significantly", True

        return True, "Valid", False

    # ─────────────────────────────────────────────────────────────
    # Apply action
    # ─────────────────────────────────────────────────────────────
    def apply_action(self, agent_id, action_index, action_map):
        agent = self.agent_dic[agent_id]

        if action_index < 0 or action_index >= len(action_map):
            kind, tid = ("depot", None)
        else:
            kind, tid = action_map[action_index]

        agent["immediate_replan"] = False

        if kind == "depot":
            depot = self.depot_dic[agent["species"]]
            if agent.get("target_type") == "task" and agent.get("target_id") is not None:
                self.remove_reservation_local(agent_id, int(agent["target_id"]))

            agent["target_type"] = "depot"
            agent["target_id"] = None
            agent["target_pos"] = depot["location"].copy()
            agent["status"] = "moving"
            agent["is_rushing"] = False
            agent["current_task"] = -1
            agent["need_decision"] = False
            agent["target_task_expected_state"] = None
            return 0.0, False, []

        task_id = int(tid)
        if task_id not in self.task_dic:
            agent["need_decision"] = True
            agent["status"] = "idle"
            return -1.0, True, []

        if agent.get("target_type") == "task" and agent.get("target_id") is not None and int(agent["target_id"]) != task_id:
            self.remove_reservation_local(agent_id, int(agent["target_id"]))

        task = self.task_dic[task_id]
        agent["target_type"] = "task"
        agent["target_id"] = task_id
        agent["target_pos"] = task["location"].copy()
        agent["status"] = "moving"
        agent["is_rushing"] = True
        agent["current_task"] = task_id
        agent["need_decision"] = False
        agent["target_task_expected_state"] = task["status"].copy()

        # 局部可见意图锁定
        self.broadcast_intention_local(agent_id, task_id)

        return 0.0, False, []

    # ─────────────────────────────────────────────────────────────
    # Time-driven step(dt)
    # ─────────────────────────────────────────────────────────────
    def step(self, dt=None):
        if dt is None:
            dt = self.dt
        dt = float(dt)

        self.last_step_reward = 0.0

        # 1) 移动
        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "moving" or agent.get("target_pos") is None:
                continue

            pos = agent["location"]
            tgt = agent["target_pos"]
            dist = float(np.linalg.norm(tgt - pos))

            if dist <= EPS_ARRIVE:
                agent["location"] = tgt.copy()
                continue

            step_len = agent["velocity"] * dt
            move = min(step_len, dist)
            direction = (tgt - pos) / dist
            new_pos = pos + direction * move
            agent["location"] = new_pos
            agent["travel_dist"] += move

        # 2) 时间推进
        self.current_time += dt

        # 3) 到达事件
        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "moving" or agent.get("target_pos") is None:
                continue

            if float(np.linalg.norm(agent["location"] - agent["target_pos"])) > EPS_ARRIVE:
                continue

            if agent.get("target_type") == "depot":
                if agent.get("current_task", -1) >= 0:
                    self.remove_reservation_local(aid, int(agent["current_task"]))

                agent["status"] = "idle"
                agent["need_decision"] = True
                agent["route"].append(self.depot_dic[agent["species"]]["ID"])
                agent["arrival_time"].append(float(self.current_time))
                agent["target_type"] = None
                agent["target_id"] = None
                agent["target_pos"] = None
                agent["is_rushing"] = False
                continue

            if agent.get("target_type") == "task" and agent.get("target_id") is not None:
                task_id = int(agent["target_id"])

                agent["route"].append(task_id)
                agent["arrival_time"].append(float(self.current_time))

                is_valid, reason, need_reselect = self.verify_task_on_arrival(aid, task_id)
                if not is_valid:
                    self.remove_reservation_local(aid, task_id)

                    # 即时惩罚 + 局部冷却传播
                    self.last_step_reward += float(self.invalid_arrival_penalty)
                    self.propagate_cooldown_local(aid, task_id, self.current_time + self.cooldown_time)

                    agent["failed_tasks"].append(
                        {
                            "task_id": task_id,
                            "time": float(self.current_time),
                            "reason": reason,
                            "task_status_snapshot": self.task_dic[task_id]["status"].copy()
                            if task_id in self.task_dic else None,
                        }
                    )
                    agent["status"] = "idle"
                    agent["need_decision"] = True
                    agent["immediate_replan"] = bool(need_reselect)
                    agent["target_type"] = None
                    agent["target_id"] = None
                    agent["target_pos"] = None
                    agent["is_rushing"] = False
                    agent["current_task"] = -1
                    continue

                task = self.task_dic[task_id]
                if aid not in task["members"]:
                    task["members"].append(aid)

                self.remove_reservation_local(aid, task_id)

                agent["assigned"] = True
                agent["status"] = "working"
                agent["need_decision"] = False
                agent["target_type"] = None
                agent["target_id"] = None
                agent["target_pos"] = None
                agent["is_rushing"] = False

        # 4) 通信图 + 发现 + 更新 + 传播
        self.update_communication_graph()

        if self.dynamic_task_discovery:
            for aid in self.agent_dic.keys():
                self.discover_dynamic_task(aid)

        self.task_update()
        self.propagate_task_completion()

        # 5) 结束
        self.finished = self.check_finished()
        return self.finished

    # ─────────────────────────────────────────────────────────────
    # Task update
    # ─────────────────────────────────────────────────────────────
    def get_arrival_time(self, agent_id, task_id):
        arrival_time = self.agent_dic[agent_id]["arrival_time"]
        route = np.array(self.agent_dic[agent_id]["route"])
        idx = np.where(route == task_id)[0]
        if len(idx) == 0:
            return float(self.current_time)
        pos = int(idx[-1] + 1)
        return float(arrival_time[pos]) if pos < len(arrival_time) else float(self.current_time)

    def task_update(self):
        f_task = []

        for task in self.task_dic.values():
            if task.get("expired") or task["finished"]:
                continue
            if self.current_time > task.get("deadline", float("inf")):
                task["expired"] = True
                task["finished"] = True
                for mid in list(task.get("members", [])):
                    if mid in self.agent_dic:
                        self._release_agent_from_task(mid)
                continue

        for task in self.task_dic.values():
            if task["finished"]:
                continue

            if not task["feasible_assignment"]:
                abilities = self.get_abilities(task["members"])
                arrival = np.array([self.get_arrival_time(member, task["ID"]) for member in task["members"]], dtype=float)

                task["status"] = np.maximum(task["requirements"] - abilities, 0)

                if (task["status"] <= 0).all():
                    if len(arrival) > 0:
                        max_arrival = float(np.max(arrival))
                        min_arrival = float(np.min(arrival))

                        if max_arrival - min_arrival <= self.max_waiting_time:
                            task["time_start"] = max_arrival
                            task["time_finish"] = max_arrival + float(task["time"])
                            task["feasible_assignment"] = True
                            f_task.append(task["ID"])
                        else:
                            keep = [
                                m for m in task["members"]
                                if self.get_arrival_time(m, task["ID"]) >= max_arrival - self.max_waiting_time
                            ]
                            task["members"] = keep
                            abilities = self.get_abilities(keep)
                            task["status"] = np.maximum(task["requirements"] - abilities, 0)

                            if keep and (task["status"] <= 0).all():
                                arrival_new = np.array([self.get_arrival_time(m, task["ID"]) for m in keep], dtype=float)
                                task["time_start"] = float(np.max(arrival_new))
                                task["time_finish"] = float(np.max(arrival_new) + float(task["time"]))
                                task["feasible_assignment"] = True
                                f_task.append(task["ID"])
                    else:
                        if (task["requirements"] <= 0).all():
                            task["finished"] = True
                            task["feasible_assignment"] = True
                            task["time_finish"] = float(self.current_time)
                            f_task.append(task["ID"])
            else:
                if self.current_time >= float(task["time_finish"]):
                    task["finished"] = True
                    for member_id in task["members"]:
                        if member_id in self.agent_dic:
                            member = self.agent_dic[member_id]
                            if member["status"] == "working":
                                member["assigned"] = False
                                member["current_task"] = -1
                                member["status"] = "idle"
                                member["need_decision"] = True
        return f_task

    def check_agent_enroute_replan(self, agent_id):
        agent = self.agent_dic[agent_id]
        if agent.get("target_type") != "task" or agent.get("target_id") is None:
            return False, "Not enroute"

        task_id = int(agent["target_id"])
        if task_id not in self.task_dic:
            return True, "Target no longer exists"

        task = self.task_dic[task_id]
        knowledge = self.agent_knowledge[agent_id]

        if task_id in knowledge["known_finished"] or task["finished"]:
            return True, "Task finished"

        if task_id in knowledge["known_assigned"] or task["feasible_assignment"]:
            current_ability = self.get_abilities(task["members"])
            remaining_need = np.maximum(task["requirements"] - current_ability, 0)
            my_contribution = np.minimum(remaining_need, agent["abilities"]).sum()
            if my_contribution <= 0:
                return True, "Already sufficiently staffed"

        current_status = task["status"]
        my_contribution = np.minimum(current_status, agent["abilities"]).sum()
        if my_contribution <= 0:
            return True, "Cannot contribute"

        return False, "OK"

    # ─────────────────────────────────────────────────────────────
    # Finished condition
    # ─────────────────────────────────────────────────────────────
    def check_finished(self):
        return len([t for t in self.task_dic.values() if not t["finished"]]) == 0

    # ─────────────────────────────────────────────────────────────
    # Episode reward
    # ─────────────────────────────────────────────────────────────
    def calculate_waiting_time(self):
        for agent in self.agent_dic.values():
            agent["sum_waiting_time"] = 0.0

        for task in self.task_dic.values():
            arrival = np.array([self.get_arrival_time(member, task["ID"]) for member in task["members"]], dtype=float)
            if len(arrival) != 0:
                if task["feasible_assignment"]:
                    task["sum_waiting_time"] = float(np.sum(np.max(arrival) - arrival))
                else:
                    task["sum_waiting_time"] = float(np.sum(self.current_time - arrival))

            for member in task["members"]:
                if task["feasible_assignment"]:
                    self.agent_dic[member]["sum_waiting_time"] += float(np.max(arrival) - self.get_arrival_time(member, task["ID"]))
                else:
                    self.agent_dic[member]["sum_waiting_time"] += float(max(0.0, self.current_time - self.get_arrival_time(member, task["ID"])))

    def get_episode_reward(self, max_time=100.0):
        task_ids = list(self.task_dic.keys())
        if len(task_ids) == 0:
            return 0.0, []

        finished_flags = np.array([bool(self.task_dic[tid].get("finished", False)) for tid in task_ids], dtype=bool)
        num_finished = int(finished_flags.sum())
        num_total = int(len(task_ids))
        success_rate = float(num_finished) / float(num_total)

        T_norm = float(max(1.0, max_time, self.current_time))

        A = 200.0
        lam = 60.0
        mu = 40.0
        eta = 40.0
        omega = 80.0
        phi = 8.0
        kappa = 2.0

        R_success = A * success_rate
        slack_sum = 0.0
        slack_count = 0
        makespan = 0.0
        num_unfinished = 0
        num_expired = 0

        for tid in task_ids:
            task = self.task_dic[tid]
            expired = bool(task.get("expired", False))
            if expired:
                num_expired += 1
            if task.get("finished", False) and not expired:
                C = float(task.get("time_finish", self.current_time))
                C = max(0.0, min(C, max(self.current_time, T_norm)))
                deadline = float(task.get("deadline", float("inf")))
                slack = max(0.0, deadline - C) / max(1.0, T_norm)
                slack_sum += min(1.0, slack)
                slack_count += 1
                makespan = max(makespan, C)
            elif not task.get("finished", False):
                num_unfinished += 1

        avg_slack = (slack_sum / max(1.0, float(slack_count))) if slack_count > 0 else 0.0
        R_slack = lam * avg_slack

        if makespan <= 0.0:
            makespan = float(self.current_time)
        R_makespan = -mu * (makespan / T_norm)

        R_unfinished = -eta * (float(num_unfinished) / max(1.0, float(num_total)))
        R_expired = -omega * (float(num_expired) / max(1.0, float(num_total)))

        failed_count = sum(len(a.get("failed_tasks", [])) for a in self.agent_dic.values())
        R_failed = -phi * float(failed_count)

        total_travel = sum(float(a.get("travel_dist", 0.0)) for a in self.agent_dic.values())
        D_norm = float(max(1.0, len(self.agent_dic) * 5.0))
        R_travel = -kappa * (total_travel / D_norm)

        reward = R_success + R_slack + R_makespan + R_unfinished + R_expired + R_failed + R_travel
        return float(reward), finished_flags.tolist()

    def cleanup_failed_tasks_memory(self):
        memory_window = 10.0
        for aid, agent in self.agent_dic.items():
            new_failed = []
            for record in agent["failed_tasks"]:
                task_id = record["task_id"]
                fail_time = record["time"]
                if self.current_time - fail_time > memory_window:
                    continue
                if task_id in self.task_dic:
                    task = self.task_dic[task_id]
                    if task["finished"]:
                        continue
                    if "task_status_snapshot" in record and record["task_status_snapshot"] is not None:
                        if not np.array_equal(record["task_status_snapshot"], task["status"]):
                            continue
                new_failed.append(record)
            agent["failed_tasks"] = new_failed

            # 清理过期cooldown
            k = self.agent_knowledge.get(aid, None)
            if k is not None and "task_cooldown_until" in k:
                cd = k["task_cooldown_until"]
                expired = [tid for tid, t_until in cd.items() if self.current_time >= float(t_until)]
                for tid in expired:
                    cd.pop(tid, None)