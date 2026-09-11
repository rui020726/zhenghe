# -*- coding: utf-8 -*-
"""
时间驱动任务分配环境（消融实验版 Ablation Universal Env）
- ✅ 支持一键切换预判机制 (pre_positioning=True/False)
- ✅ 自动根据开关切换特征维度 (15/18 维 ↔ 20/23 维)
- ✅ 内置 Stage Gap Time 计时器，用于证明预判机制能缩短等待时间
"""

import numpy as np
from collections import deque
import torch

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

EPS_ARRIVE = 1e-6


class TaskEnv:
    def __init__(
            self,
            prio_range=(0.3, 1.0),
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
            pre_positioning=True,  # ✅ 消融实验核心开关
    ):
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.ranges = {
            "prio": prio_range,
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

        self.pre_positioning = pre_positioning  # ✅ 保存开关状态

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
            agent["assigned"] = False
            agent["current_task"] = -1
            agent["status"] = "idle"
            agent["need_decision"] = True
            agent["consecutive_timeouts"] = 0
            agent["working_start_time"] = None
            agent["working_timeout_value"] = None
            agent["target_type"] = None
            agent["target_id"] = None
            agent["immediate_replan"] = True

    def generate_task(self, tasks_num):
        tasks_stages = {}
        search_priorities = self.rng.uniform(self.ranges["prio"][0], self.ranges["prio"][1], tasks_num)
        scene_complexities = np.clip(search_priorities * 0.8 + self.rng.uniform(0, 0.2, tasks_num), 0.0, 1.0)

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
                for d in dims:
                    req[d] = 1

                if req.sum() == 0:
                    req[self.random_int(0, self.traits_dim)] = 1

                stages.append({
                    "stage_id": s,
                    "requirements": req,
                    "status": req.copy(),
                    "time": float(stage_time),
                    "time_left": float(stage_time),
                    "finished": False
                })
            tasks_stages[i] = stages

        return tasks_stages, search_priorities, scene_complexities

    def generate_agent(self, species_num):
        agents_ini = np.zeros((species_num, self.traits_dim), dtype=int)
        # 强制设定：每个兵种（物种）只能有 1 到 2 种技能（稀疏化），而不是全能超人
        for s in range(species_num):
            num_skills = self.random_int(1, 3)  # 随机分配 1 或 2 项技能
            skill_indices = self.random_choice(self.traits_dim, size=num_skills, replace=False)
            for idx in skill_indices:
                agents_ini[s, idx] = 1
        # 确保所有 5 个维度在所有兵种加起来至少被覆盖一次
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

        tasks_stages, search_priorities, scene_complexities = self.generate_task(n_tasks)
        depot_loc = self.random_val((n_species, 2))
        tasks_loc = self.random_val((n_tasks, 2))

        task_dic = {}
        agent_dic = {}
        depot_dic = {}
        species_dict = {"abilities": agents_ini, "number": agents_species_num}

        for i in range(n_tasks):
            task_dic[i] = {
                "ID": i,
                "stages": tasks_stages[i],
                "current_stage_index": 0,
                "total_stages": len(tasks_stages[i]),
                "requirements": tasks_stages[i][0]["requirements"].copy(),
                "status": tasks_stages[i][0]["status"].copy(),
                "time": tasks_stages[i][0]["time"],
                "location": tasks_loc[i, :],
                "search_priority": float(search_priorities[i]),
                "scene_complexity": float(scene_complexities[i]),
                "finished": False,
                "feasible_assignment": False,
                "members": [],
                "time_start": 0.0,
                "time_finish": 0.0,

                # ✅ 新增：空窗期统计字段
                "last_stage_finish_time": 0.0,
                "waiting_for_next_stage": False,
                "gap_times": [],
            }
            self.task_source[i] = "initial"

        cnt = 0
        for s in range(n_species):
            depot_dic[s] = {"ID": -s - 1, "location": depot_loc[s], "members": []}
            for _ in range(agents_species_num[s]):
                agent_dic[cnt] = {
                    "ID": cnt, "species": s, "abilities": agents_ini[s, :],
                    "location": depot_loc[s].copy(), "depot": depot_loc[s].copy(),
                    "velocity": 0.2, "current_task": -1, "travel_dist": 0.0,
                    "sum_waiting_time": 0.0, "route": [], "arrival_time": [0.0],
                    "assigned": False, "is_rushing": False, "target_task_expected_state": None,
                    "failed_tasks": [], "immediate_replan": False, "need_decision": True,
                    "status": "idle", "target_type": None, "target_id": None, "target_pos": None,
                    "working_start_time": None, "working_timeout_value": None,
                    "consecutive_timeouts": 0, "cooldown_tasks": {},
                }
                depot_dic[s]["members"].append(cnt)
                cnt += 1

        return task_dic, agent_dic, depot_dic, species_dict

    def init_agent_memory(self):
        self.agent_knowledge = {
            aid: {
                "known_tasks": set(self.task_dic.keys()),
                "task_locations": {tid: task["location"].copy() for tid, task in self.task_dic.items()},
                "task_requirements": {tid: task["requirements"].copy() for tid, task in self.task_dic.items()},
                "task_priorities": {tid: task["search_priority"] for tid, task in self.task_dic.items()},
                "dynamic_tasks": set(),
                "known_finished": set(),
                "known_assigned": set(),
                "task_current_status": {tid: task["status"].copy() for tid, task in self.task_dic.items()},
                "task_members": {tid: [] for tid in self.task_dic.keys()},
                "task_last_update_time": {tid: 0.0 for tid in self.task_dic.keys()},
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

        return self.get_global_state()

    def update_communication_graph(self):
        self.comm_graph = {aid: [] for aid in self.agent_dic.keys()}
        agent_ids = list(self.agent_dic.keys())
        for i, aid_i in enumerate(agent_ids):
            loc_i = self.agent_dic[aid_i]["location"]
            for j, aid_j in enumerate(agent_ids):
                if i == j: continue
                loc_j = self.agent_dic[aid_j]["location"]
                if np.linalg.norm(loc_i - loc_j) <= self.comm_range:
                    self.comm_graph[aid_i].append(aid_j)

    def propagate_task_completion(self):
        for aid in self.agent_dic.keys():
            agent = self.agent_dic[aid]
            knowledge = self.agent_knowledge[aid]

            for tid, task in self.task_dic.items():
                distance = np.linalg.norm(agent["location"] - task["location"])

                if distance <= self.comm_range:
                    if tid not in knowledge["known_tasks"]:
                        knowledge["known_tasks"].add(tid)
                        knowledge["task_locations"][tid] = task["location"].copy()
                        knowledge["task_priorities"][tid] = task["search_priority"]
                        if self.task_source.get(tid) == "discovered":
                            knowledge["dynamic_tasks"].add(tid)

                    if task["finished"]:
                        knowledge["known_finished"].add(tid)

                    knowledge["task_current_status"][tid] = task["status"].copy()
                    knowledge["task_requirements"][tid] = task["requirements"].copy()
                    knowledge["task_members"][tid] = task["members"].copy()
                    knowledge["task_last_update_time"][tid] = self.current_time

        for aid in self.agent_dic.keys():
            neighbors = self.comm_graph.get(aid, [])
            knowledge = self.agent_knowledge[aid]

            for nid in neighbors:
                nk = self.agent_knowledge[nid]

                for tid in knowledge["known_tasks"]:
                    if tid not in nk["known_tasks"]:
                        if tid in self.task_dic:
                            nk["known_tasks"].add(tid)
                            nk["task_locations"][tid] = knowledge["task_locations"].get(tid, self.task_dic[tid][
                                "location"]).copy()
                            nk["task_priorities"][tid] = knowledge["task_priorities"].get(tid, self.task_dic[tid][
                                "search_priority"])
                            nk["task_current_status"][tid] = knowledge["task_current_status"].get(tid,
                                                                                                  self.task_dic[tid][
                                                                                                      "status"]).copy()
                            nk["task_requirements"][tid] = knowledge["task_requirements"].get(tid, self.task_dic[tid][
                                "requirements"]).copy()
                            nk["task_members"][tid] = knowledge["task_members"].get(tid, []).copy()
                            nk["task_last_update_time"][tid] = knowledge["task_last_update_time"].get(tid,
                                                                                                      self.current_time)

                            if tid in knowledge["dynamic_tasks"]:
                                nk["dynamic_tasks"].add(tid)

                for tid in knowledge["known_finished"]:
                    if tid in nk["known_tasks"] and tid not in nk["known_finished"]:
                        nk["known_finished"].add(tid)
                        nk["task_last_update_time"][tid] = knowledge["task_last_update_time"].get(tid,
                                                                                                  self.current_time)

                for tid in knowledge["known_tasks"]:
                    if tid not in nk["known_tasks"]: continue
                    my_t = knowledge["task_last_update_time"].get(tid, 0.0)
                    nb_t = nk["task_last_update_time"].get(tid, 0.0)
                    if my_t > nb_t:
                        nk["task_current_status"][tid] = knowledge["task_current_status"].get(tid, self.task_dic[tid][
                            "status"]).copy()
                        nk["task_requirements"][tid] = knowledge["task_requirements"].get(tid, self.task_dic[tid][
                            "requirements"]).copy()
                        nk["task_members"][tid] = knowledge["task_members"].get(tid, []).copy()
                        nk["task_last_update_time"][tid] = my_t

        for aid, agent in self.agent_dic.items():
            if agent.get("is_rushing", False) and agent.get("target_type") == "task" and agent.get(
                    "target_id") is not None:
                need_replan, _reason = self.check_agent_enroute_replan(aid)
                if need_replan:
                    agent["immediate_replan"] = True
                    agent["need_decision"] = True
                    agent["status"] = "idle"
                    agent["target_type"] = None
                    agent["target_id"] = None
                    agent["target_pos"] = None
                    agent["is_rushing"] = False
                    agent["current_task"] = -1

    def discover_dynamic_task(self, agent_id):
        if not self.dynamic_task_discovery: return None
        current_dynamic_count = sum(1 for source in self.task_source.values() if source == "discovered")
        if current_dynamic_count >= self.max_dynamic_tasks: return None
        agent = self.agent_dic[agent_id]
        if agent.get("target_type") != "task" and agent.get("current_task", -1) < 0: return None
        if self.rng.random() > self.discovery_probability: return None

        new_task_id = self.next_dynamic_task_id
        self.next_dynamic_task_id += 1

        offset = self.rng.uniform(-0.1, 0.1, size=2)
        new_location = np.clip(agent["location"] + offset, 0, 1)

        num_stages = self.random_int(1, 4)
        stages = []
        base_time = 2.0
        complexity = float(self.rng.uniform(0.5, 0.9))
        stage_time = (base_time + complexity * self.duration_scale * 2 + self.rng.uniform(0, 1.5)) / num_stages

        for s in range(num_stages):
            req = np.zeros(self.traits_dim, dtype=int)
            num_required = self.random_int(1, min(3, self.traits_dim + 1))
            dims = self.random_choice(self.traits_dim, size=num_required, replace=False)
            for d in dims: req[d] = 1
            if req.sum() == 0: req[self.random_int(0, self.traits_dim)] = 1

            stages.append({
                "stage_id": s, "requirements": req, "status": req.copy(),
                "time": float(stage_time), "time_left": float(stage_time), "finished": False
            })

        new_priority = self.rng.uniform(0.7, 1.0)

        self.task_dic[new_task_id] = {
            "ID": new_task_id,
            "stages": stages,
            "current_stage_index": 0,
            "total_stages": len(stages),
            "requirements": stages[0]["requirements"].copy(),
            "status": stages[0]["status"].copy(),
            "time": stages[0]["time"],
            "location": new_location,
            "search_priority": float(new_priority),
            "scene_complexity": complexity,
            "finished": False,
            "feasible_assignment": False,
            "members": [],
            "time_start": 0.0,
            "time_finish": 0.0,
            "discovered_at": self.current_time,
            "discovered_by": agent_id,
            "last_stage_finish_time": 0.0,
            "waiting_for_next_stage": False,
            "gap_times": [],
        }
        self.task_source[new_task_id] = "discovered"
        return new_task_id

    def _count_enroute_and_working(self, task_id, exclude_agent_id=None):
        count = 0
        for aid, agent in self.agent_dic.items():
            if aid == exclude_agent_id: continue
            if agent.get("target_type") == "task" and agent.get("target_id") == task_id:
                count += 1
            elif agent.get("status") == "working" and agent.get("current_task") == task_id:
                count += 1
        return count

    def _compute_shaping_reward(self, agent_id, task_id):
        agent = self.agent_dic[agent_id]
        task = self.task_dic[task_id]
        reward = 0.0

        current_status = task["status"]
        remaining_needs = current_status.sum()

        crowd_count = self._count_enroute_and_working(task_id, exclude_agent_id=agent_id)
        existing_members = len(task["members"])
        total_others = crowd_count + existing_members

        if remaining_needs <= 0:
            reward -= self.shaping_crowd_penalty * (total_others + 1)
        elif total_others > 0:
            excess = max(0, total_others - remaining_needs)
            reward -= self.shaping_crowd_penalty * excess

        if remaining_needs > 0:
            enroute_abilities = np.zeros(self.traits_dim)
            for other_aid, other_agent in self.agent_dic.items():
                if other_aid == agent_id: continue
                is_heading_here = (other_agent.get("target_type") == "task" and other_agent.get("target_id") == task_id)
                is_working_here = (
                            other_agent.get("status") == "working" and other_agent.get("current_task") == task_id)
                if is_heading_here or is_working_here:
                    enroute_abilities += other_agent["abilities"]

            member_abilities = self.get_abilities(task["members"])
            total_committed_abilities = enroute_abilities + member_abilities
            still_needed = np.maximum(task["requirements"] - total_committed_abilities, 0)
            still_needed_sum = still_needed.sum()
            my_fill = np.minimum(still_needed, agent["abilities"]).sum()

            if my_fill > 0:
                priority = float(task.get("search_priority", 0.5))
                reward += self.shaping_lonely_bonus * float(my_fill) * (0.5 + priority)
            elif still_needed_sum <= 0 and total_others > 0:
                reward -= self.shaping_crowd_penalty * 0.5

        dist = float(np.linalg.norm(agent["location"] - task["location"]))
        dist_norm = dist / 1.414
        reward -= self.shaping_distance_penalty * dist_norm

        my_contribution = np.minimum(current_status, agent["abilities"]).sum()
        if my_contribution > 0:
            reward += 0.05 * float(my_contribution)
        else:
            reward -= self.shaping_redundant_penalty

        return float(reward)

    def _check_working_timeouts(self):
        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "working": continue

            start_time = agent.get("working_start_time")
            timeout_val = agent.get("working_timeout_value")
            if start_time is None or timeout_val is None: continue

            elapsed = self.current_time - start_time
            if elapsed < timeout_val: continue

            task_id = agent.get("current_task", -1)
            if task_id >= 0 and task_id in self.task_dic:
                task = self.task_dic[task_id]
                if aid in task["members"]:
                    task["members"].remove(aid)
                abilities = self.get_abilities(task["members"])
                stage = task["stages"][task["current_stage_index"]]
                stage["status"] = np.maximum(stage["requirements"] - abilities, 0)
                task["status"] = stage["status"].copy()

                agent["cooldown_tasks"][task_id] = self.current_time + self.cooldown_duration
                agent["failed_tasks"].append({
                    "task_id": task_id, "time": float(self.current_time),
                    "reason": f"Working timeout",
                    "task_status_snapshot": task["status"].copy(),
                })

            agent["consecutive_timeouts"] = agent.get("consecutive_timeouts", 0) + 1
            self._release_agent(aid)

    def _cleanup_cooldowns(self):
        for agent in self.agent_dic.values():
            if not agent.get("cooldown_tasks"): continue
            expired = [tid for tid, end_time in agent["cooldown_tasks"].items() if self.current_time >= end_time]
            for tid in expired: del agent["cooldown_tasks"][tid]

    # ✅ 核心消融逻辑 1：掩码控制
    def get_hard_mask(self, agent_id):
        agent = self.agent_dic[agent_id]
        knowledge = self.agent_knowledge[agent_id]
        mask = [False]

        for tid in sorted(knowledge["known_tasks"]):
            if tid not in self.task_dic:
                mask.append(True);
                continue

            if tid in knowledge["known_finished"]:
                mask.append(True);
                continue

            task = self.task_dic[tid]
            current_status = knowledge["task_current_status"].get(tid, task["status"])
            my_contribution_now = np.minimum(current_status, agent["abilities"]).sum()

            # --- Ablation Toggle ---
            if self.pre_positioning:
                future_status = np.zeros(self.traits_dim)
                if task["current_stage_index"] + 1 < task["total_stages"]:
                    future_status = task["stages"][task["current_stage_index"] + 1]["requirements"]
                my_contribution_future = np.minimum(future_status, agent["abilities"]).sum()

                if my_contribution_now <= 0 and my_contribution_future <= 0:
                    mask.append(True);
                    continue
            else:
                if my_contribution_now <= 0:
                    mask.append(True);
                    continue
            # -----------------------

            if tid in agent.get("cooldown_tasks", {}):
                if self.current_time < agent["cooldown_tasks"][tid]:
                    mask.append(True);
                    continue

            mask.append(False)

        return np.array(mask, dtype=bool)

    def get_soft_mask(self, agent_id):
        return self.get_hard_mask(agent_id)

    def get_ctde_mask(self, agent_id, is_training=True):
        return self.get_soft_mask(agent_id) if is_training else self.get_hard_mask(agent_id)

    # ✅ 核心消融逻辑 2：动态维度切换 (15 vs 20)
    def get_local_obs(self, agent_id, max_tasks=None):
        agent = self.agent_dic[agent_id]
        knowledge = self.agent_knowledge[agent_id]

        LOC_NORM = 1.414
        PRIO_MIN, PRIO_MAX = self.ranges["prio"]

        self_state = np.hstack([
            agent["abilities"], agent["location"], float(agent["current_task"]),
            float(agent["travel_dist"]), float(agent.get("sum_waiting_time", 0))
        ]).astype(np.float32)

        neighbors = []
        for nid in self.comm_graph.get(agent_id, []):
            nb = self.agent_dic[nid]
            neighbors.append(np.hstack(
                [nb["abilities"], nb["location"], float(nb["current_task"]), float(nb["assigned"]),
                 float(nb["travel_dist"])]))

        neighbors = np.array(neighbors, dtype=np.float32) if neighbors else np.zeros((1, 10), dtype=np.float32)
        if neighbors.shape[0] < 20:
            neighbors = np.vstack([neighbors, np.zeros((20 - neighbors.shape[0], 10), dtype=np.float32)])
        else:
            neighbors = neighbors[:20]

        tasks = []
        action_map = []

        # --- Ablation Toggle ---
        rel_loc = (self.depot_dic[agent["species"]]["location"] - agent["location"]) / LOC_NORM
        if self.pre_positioning:
            depot_feat = np.hstack([np.zeros(5), np.zeros(5), rel_loc, [0.0] * 8]).astype(np.float32)  # 20维
        else:
            depot_feat = np.hstack([np.zeros(5), rel_loc, [0.0] * 8]).astype(np.float32)  # 15维
        # -----------------------

        tasks.append(depot_feat)
        action_map.append(("depot", None))

        for tid in sorted(knowledge["known_tasks"]):
            if tid not in self.task_dic: continue
            task = self.task_dic[task_id := tid]
            location = knowledge["task_locations"].get(tid, task["location"])
            current_status = knowledge["task_current_status"].get(tid, knowledge["task_requirements"].get(tid, task[
                "requirements"]))

            priority_norm = (float(knowledge["task_priorities"].get(tid, task["search_priority"])) - PRIO_MIN) / (
                        PRIO_MAX - PRIO_MIN + 1e-8)
            rel = (location - agent["location"]) / LOC_NORM
            time_since_norm = np.tanh(
                float(self.current_time - float(knowledge["task_last_update_time"].get(tid, 0.0))) / 50.0)
            num_members_known = float(len(knowledge["task_members"].get(tid, []))) / 10.0

            # --- Ablation Toggle ---
            if self.pre_positioning:
                future_status = np.zeros(self.traits_dim, dtype=np.float32)
                if task["current_stage_index"] + 1 < task["total_stages"]:
                    future_status = task["stages"][task["current_stage_index"] + 1]["requirements"].astype(np.float32)
                base_feat = np.hstack(
                    [current_status.astype(np.float32).flatten(), future_status.flatten(), rel.flatten(),
                     [priority_norm], [float(task.get("scene_complexity", 0.5))], [num_members_known]])
            else:
                base_feat = np.hstack([current_status.astype(np.float32).flatten(), rel.flatten(), [priority_norm],
                                       [float(task.get("scene_complexity", 0.5))], [num_members_known]])
            # -----------------------

            info_feat = np.array([float(tid in knowledge["known_finished"]), float(tid in knowledge["known_assigned"]),
                                  float(tid in knowledge["dynamic_tasks"]), time_since_norm, num_members_known],
                                 dtype=np.float32)

            feat = np.hstack([base_feat, info_feat]).astype(np.float32)

            target_dim = 20 if self.pre_positioning else 15
            if feat.shape[0] != target_dim: feat = np.zeros(target_dim, dtype=np.float32)

            tasks.append(feat)
            action_map.append(("task", int(tid)))
            if max_tasks is not None and len(tasks) >= max_tasks: break

        tasks_array = np.array(tasks, dtype=np.float32)
        mask = self.get_ctde_mask(agent_id, is_training=True)[: len(action_map)]

        if max_tasks is not None and tasks_array.shape[0] < max_tasks:
            target_dim = 20 if self.pre_positioning else 15
            tasks_array = np.vstack(
                [tasks_array, np.zeros((max_tasks - tasks_array.shape[0], target_dim), dtype=np.float32)])
            mask = np.concatenate([mask, np.ones((max_tasks - len(mask),), dtype=bool)])

        if mask.all(): mask[0] = False
        return self_state, neighbors, tasks_array, mask.astype(np.bool_), action_map

    # ✅ 核心消融逻辑 3：全局状态维度动态切换 (18 vs 23)
    def get_global_state(self):
        LOC_NORM = 1.0
        g_agents = []
        for agent in self.agent_dic.values():
            g_agents.append(np.hstack([agent["abilities"], agent["location"] / LOC_NORM, float(agent["current_task"]),
                                       float(agent["assigned"]), 0.0, 0.0, float(agent["travel_dist"]),
                                       float(agent.get("sum_waiting_time", 0.0))]))

        g_tasks = []
        depot0 = self.depot_dic[0] if 0 in self.depot_dic else {"location": np.array([0.5, 0.5])}

        # --- Ablation Toggle ---
        if self.pre_positioning:
            g_tasks.append(np.hstack([np.zeros(self.traits_dim), np.zeros(self.traits_dim), -np.ones(self.traits_dim),
                                      depot0["location"] / LOC_NORM, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))  # 23
        else:
            g_tasks.append(np.hstack(
                [np.zeros(self.traits_dim), -np.ones(self.traits_dim), depot0["location"] / LOC_NORM, 1.0, 0.0, 0.0,
                 0.0, 0.0, 0.0]))  # 18
        # -----------------------

        for task in self.task_dic.values():
            # --- Ablation Toggle ---
            if self.pre_positioning:
                future_status = np.zeros(self.traits_dim)
                if task["current_stage_index"] + 1 < task["total_stages"]:
                    future_status = task["stages"][task["current_stage_index"] + 1]["requirements"]
                g_tasks.append(np.hstack(
                    [task["status"], future_status, task["requirements"], task["location"] / LOC_NORM,
                     float(task["feasible_assignment"]), float(task.get("search_priority", 0.5)),
                     float(task.get("scene_complexity", 0.5)), float(task.get("time_start", 0.0)),
                     float(task.get("time_finish", 0.0)), float(len(task.get("members", [])))]))
            else:
                g_tasks.append(np.hstack([task["status"], task["requirements"], task["location"] / LOC_NORM,
                                          float(task["feasible_assignment"]), float(task.get("search_priority", 0.5)),
                                          float(task.get("scene_complexity", 0.5)), float(task.get("time_start", 0.0)),
                                          float(task.get("time_finish", 0.0)), float(len(task.get("members", [])))]))
            # -----------------------

        return np.array(g_agents, dtype=np.float32), np.array(g_tasks, dtype=np.float32)

    def calculate_eulidean_distance(self, entity1, entity2):
        return float(np.linalg.norm(entity1["location"] - entity2["location"]))

    def get_abilities(self, members):
        if len(members) == 0: return np.zeros(self.traits_dim)
        return np.sum([self.agent_dic[m]["abilities"] for m in members if m in self.agent_dic], axis=0)

    # ✅ 核心消融逻辑 4：到达校验
    def verify_task_on_arrival(self, agent_id, task_id):
        if task_id < 0: return True, "Depot", False
        task = self.task_dic[task_id]
        agent = self.agent_dic[agent_id]
        knowledge = self.agent_knowledge[agent_id]

        knowledge["task_current_status"][task_id] = task["status"].copy()
        knowledge["task_requirements"][task_id] = task["requirements"].copy()
        knowledge["task_members"][task_id] = task["members"].copy()
        knowledge["task_last_update_time"][task_id] = self.current_time

        if task["finished"]:
            knowledge["known_finished"].add(task_id)
            return False, "Task already finished", True

        current_status = task["status"]
        my_contribution_now = np.minimum(current_status, agent["abilities"]).sum()

        # --- Ablation Toggle ---
        if self.pre_positioning:
            future_status = np.zeros(self.traits_dim)
            if task["current_stage_index"] + 1 < task["total_stages"]:
                future_status = task["stages"][task["current_stage_index"] + 1]["requirements"]
            my_contribution_future = np.minimum(future_status, agent["abilities"]).sum()

            if my_contribution_now <= 0 and my_contribution_future <= 0:
                return False, "Cannot contribute to current or next stage", True
        else:
            if my_contribution_now <= 0:
                return False, "Cannot contribute to CURRENT task stage", True
        # -----------------------

        if agent.get("target_task_expected_state") is not None:
            expected = agent["target_task_expected_state"]
            if np.sum(current_status) < np.sum(expected) * 0.5:
                return False, "Task stage/state changed significantly", True

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

        task = self.task_dic[task_id]
        agent.update(
            {"target_type": "task", "target_id": task_id, "target_pos": task["location"].copy(), "status": "moving",
             "is_rushing": True, "current_task": task_id, "need_decision": False,
             "target_task_expected_state": task["status"].copy(), "working_start_time": None,
             "working_timeout_value": None})
        agent["consecutive_timeouts"] = 0

        return self._compute_shaping_reward(agent_id, task_id), False, []

    def step(self, dt=None):
        dt = float(dt) if dt is not None else self.dt

        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "moving" or agent.get("target_pos") is None: continue
            pos = agent["location"]
            tgt = agent["target_pos"]
            dist = float(np.linalg.norm(tgt - pos))
            if dist <= EPS_ARRIVE:
                agent["location"] = tgt.copy()
                continue
            move = min(agent["velocity"] * dt, dist)
            agent["location"] = pos + (tgt - pos) / dist * move
            agent["travel_dist"] += move

        self.current_time += dt

        for aid, agent in self.agent_dic.items():
            if agent.get("status") != "moving" or agent.get("target_pos") is None: continue
            if float(np.linalg.norm(agent["location"] - agent["target_pos"])) > EPS_ARRIVE: continue

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

                is_valid, reason, need_reselect = self.verify_task_on_arrival(aid, task_id)
                if not is_valid:
                    agent["failed_tasks"].append(
                        {"task_id": task_id, "time": float(self.current_time), "reason": reason,
                         "task_status_snapshot": self.task_dic[task_id][
                             "status"].copy() if task_id in self.task_dic else None})
                    self._release_agent(aid)
                    continue

                task = self.task_dic[task_id]
                if aid not in task["members"]: task["members"].append(aid)
                agent.update({"assigned": True, "status": "working", "need_decision": False, "is_rushing": False,
                              "target_type": None, "target_id": None, "target_pos": None,
                              "working_start_time": self.current_time,
                              "working_timeout_value": self._sample_working_timeout(aid)})

        self.update_communication_graph()
        if self.dynamic_task_discovery:
            for aid in self.agent_dic.keys(): self.discover_dynamic_task(aid)

        self.task_update()
        self.propagate_task_completion()
        self._check_working_timeouts()
        self._cleanup_cooldowns()

        self.finished = len([t for t in self.task_dic.values() if not t["finished"]]) == 0
        return self.finished

    # ✅ 核心机制：阶段推进与空窗期统计
    def task_update(self):
        f_task = []
        for task in self.task_dic.values():
            if task["finished"]: continue

            curr_idx = task["current_stage_index"]
            stage = task["stages"][curr_idx]

            abilities = self.get_abilities(task["members"])
            stage["status"] = np.maximum(stage["requirements"] - abilities, 0)

            task["status"] = stage["status"].copy()
            task["requirements"] = stage["requirements"].copy()
            task["time"] = stage["time"]

            if (stage["status"] <= 0).all() and len(task["members"]) > 0:

                # ✅ 计时器：结束等待，计算这段 Gap Time
                if task.get("waiting_for_next_stage", False):
                    gap = self.current_time - task["last_stage_finish_time"]
                    task["gap_times"].append(gap)
                    task["waiting_for_next_stage"] = False

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
                        for m in list(task["members"]):
                            self._release_agent(m)
                        task["members"] = []
                    else:
                        next_req = task["stages"][task["current_stage_index"]]["requirements"]
                        task["status"] = next_req.copy()
                        task["requirements"] = next_req.copy()
                        task["time"] = task["stages"][task["current_stage_index"]]["time"]

                        # ✅ 计时器：开始等待下一个阶段的人员到达
                        task["last_stage_finish_time"] = self.current_time
                        task["waiting_for_next_stage"] = True

                        keep_members = []
                        for m in list(task["members"]):
                            if m in self.agent_dic:
                                ag = self.agent_dic[m]
                                if np.minimum(next_req, ag["abilities"]).sum() > 0:
                                    keep_members.append(m)
                                else:
                                    self._release_agent(m)

                        task["members"] = keep_members
            else:
                task["feasible_assignment"] = False
                stage["time_left"] = stage["time"]

        return f_task

    def check_agent_enroute_replan(self, agent_id):
        agent = self.agent_dic[agent_id]
        if agent.get("target_type") != "task" or agent.get("target_id") is None: return False, "Not enroute"
        task_id = int(agent["target_id"])
        if task_id not in self.task_dic: return True, "Target no longer exists"
        task = self.task_dic[task_id]
        knowledge = self.agent_knowledge[agent_id]

        if task_id in knowledge["known_finished"] or task["finished"]: return True, "Task finished"

        current_status = task["status"]
        my_contribution = np.minimum(current_status, agent["abilities"]).sum()
        if my_contribution <= 0: return True, "Cannot contribute to current stage"

        return False, "OK"

    def calculate_waiting_time(self):
        pass

    # ✅ 新增：外部调用接口，一键获取全局平均 Gap Time
    def get_avg_stage_gap(self):
        """计算本局游戏中，所有阶段切换的平均空窗期"""
        all_gaps = []
        for task in self.task_dic.values():
            all_gaps.extend(task.get("gap_times", []))
        if len(all_gaps) == 0:
            return 0.0
        return float(np.mean(all_gaps))

    def get_episode_reward(self, max_time=100.0):
        task_ids = list(self.task_dic.keys())
        if len(task_ids) == 0: return 0.0, []

        finished_flags = np.array([bool(self.task_dic[tid].get("finished", False)) for tid in task_ids], dtype=bool)
        num_finished = int(finished_flags.sum())
        num_total = int(len(task_ids))
        success_rate = float(num_finished) / float(num_total)

        T_norm = float(max(1.0, max_time, self.current_time))
        A, lam, mu, eta, phi, kappa = 200.0, 60.0, 40.0, 40.0, 8.0, 2.0
        R_success = A * success_rate

        weighted_C_sum, weighted_w_sum, makespan, unfinished_priority_sum = 0.0, 0.0, 0.0, 0.0

        for tid in task_ids:
            task = self.task_dic[tid]
            w = float(0.1 + task.get("search_priority", 0.5))
            if task.get("finished", False):
                C = float(task.get("time_finish", self.current_time))
                C = max(0.0, min(C, max(self.current_time, T_norm)))
                weighted_C_sum += w * (C / T_norm)
                weighted_w_sum += w
                makespan = max(makespan, C)
            else:
                unfinished_priority_sum += w

        avg_weighted_C = weighted_C_sum / weighted_w_sum if weighted_w_sum > 1e-9 else 1.0
        R_prio_early = -lam * avg_weighted_C
        if makespan <= 0.0: makespan = float(self.current_time)
        R_makespan = -mu * (makespan / T_norm)
        R_unfinished = -eta * (unfinished_priority_sum / max(1.0, float(num_total)))
        failed_count = sum(len(a.get("failed_tasks", [])) for a in self.agent_dic.values())
        R_failed = -phi * float(failed_count)
        total_travel = sum(float(a.get("travel_dist", 0.0)) for a in self.agent_dic.values())
        R_travel = -kappa * (total_travel / float(max(1.0, len(self.agent_dic) * 5.0)))

        return float(
            R_success + R_prio_early + R_makespan + R_unfinished + R_failed + R_travel), finished_flags.tolist()

    def cleanup_failed_tasks_memory(self):
        memory_window = 10.0
        for agent in self.agent_dic.values():
            new_failed = []
            for record in agent["failed_tasks"]:
                task_id = record["task_id"]
                if self.current_time - record["time"] > memory_window: continue
                if task_id in self.task_dic:
                    task = self.task_dic[task_id]
                    if task["finished"]: continue
                    if record.get("task_status_snapshot") is not None:
                        if not np.array_equal(record["task_status_snapshot"], task["status"]): continue
                new_failed.append(record)
            agent["failed_tasks"] = new_failed