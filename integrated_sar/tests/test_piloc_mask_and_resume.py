"""PILOC 合法动作掩码四路一致性与断点恢复一致性回归测试。

外部评审要求的两条硬性保证：

1. 部署一致性：训练采样、PPO log-prob 重算、确定性验证、``piloc/test.py``
   评估与集成 SAR runtime 的部署推理必须共用同一份 ``timemap.legal_action_mask``；
   即使非法方向持有最高原始概率（部署时最危险的失败模式——训练中非法方向
   几乎无梯度），生产推理最终也必须选择合法动作。
2. 恢复一致性：checkpoint 把权重、优化器、课程、RNG 与 PPO buffer 残留窗口
   嵌在同一个文件里，恢复后继续训练必须与连续训练逐位一致
   （N+M ≡ N+resume M）。``learn`` 通过 ``all()`` 消费全部 capacity 行，
   所以 buffer 的 state_dict 必须全量存取——只存前 ``_size`` 行会把残留
   数据清零，恢复后的梯度更新就与连续训练分岔。

真的加载 ``piloc/train.py`` 与 ``piloc/test.py``（它们在模块作用域导入
torch/gym/Pillow/matplotlib/imageio/pandas），因此整个文件按依赖可用性门控。
"""

from __future__ import annotations

import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOC_DIR = _REPO_ROOT / "piloc"


def _missing_dependency() -> str | None:
    for name in ("torch", "gym", "PIL", "matplotlib", "imageio", "pandas"):
        if importlib.util.find_spec(name) is None:
            return f"{name} is not installed"
    return None


_SKIP_REASON = _missing_dependency()
_MODULES: dict[str, Any] = {}


def _load_piloc_module(stem: str, filename: str) -> Any:
    """按路径导入 ``piloc`` 顶层脚本，并让 ``configs``/``src.*`` 可导入。"""
    if stem in _MODULES:
        return _MODULES[stem]
    for entry in (_PILOC_DIR, _PILOC_DIR / "src" / "DDQN"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    spec = importlib.util.spec_from_file_location(stem, _PILOC_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULES[stem] = module
    return module


def _load_train_module() -> Any:
    return _load_piloc_module("piloc_train_under_test", "train.py")


def _load_eval_module() -> Any:
    return _load_piloc_module("piloc_test_under_test", "test.py")


class _MaskEnv:
    """``legal_action_mask`` 只读 ``map``/``map_size``/``agents_pos``（外加信息素窗口）。"""

    def __init__(self, grid: np.ndarray, positions: list) -> None:
        self.map = np.asarray(grid, dtype=np.int32)
        self.map_size = self.map.shape
        self.agents_pos = [list(position) for position in positions]
        self.agents = [f"agent_{index}" for index in range(len(positions))]
        self.global_pheromone = np.zeros(self.map.shape, dtype=np.float32)


# action_list = [[-1, 0], [1, 0], [0, -1], [0, 1]]：0=上(x-1) 1=下(x+1) 2=左(y-1) 3=右(y+1)
_OPEN_5X5 = np.zeros((5, 5), dtype=np.int32)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class LegalActionMaskTests(unittest.TestCase):
    """边界、障碍与全包围退化情形——四条路径共用的这份实现必须各自正确。"""

    def setUp(self) -> None:
        self.mask_fn = _load_train_module().legal_action_mask

    def test_an_interior_cell_keeps_all_four_directions(self) -> None:
        env = _MaskEnv(_OPEN_5X5, [(2, 2)])
        np.testing.assert_array_equal(self.mask_fn(env, 0), np.ones(4, dtype=np.float32))

    def test_a_corner_masks_out_of_bounds_directions(self) -> None:
        env = _MaskEnv(_OPEN_5X5, [(0, 0)])
        np.testing.assert_array_equal(
            self.mask_fn(env, 0), np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        )

    def test_an_adjacent_obstacle_masks_that_direction(self) -> None:
        grid = _OPEN_5X5.copy()
        grid[3, 2] = 1  # 下方向 (x+1) 是墙
        env = _MaskEnv(grid, [(2, 2)])
        mask = self.mask_fn(env, 0)
        self.assertEqual(float(mask[1]), 0.0)
        self.assertEqual(float(mask.sum()), 3.0)

    def test_a_non_obstacle_special_cell_stays_legal(self) -> None:
        grid = _OPEN_5X5.copy()
        grid[2, 3] = 2  # 例如目标格：非 1 的特殊值不得被当成墙
        env = _MaskEnv(grid, [(2, 2)])
        self.assertEqual(float(self.mask_fn(env, 0)[3]), 1.0)

    def test_a_surrounded_agent_falls_back_to_all_legal(self) -> None:
        grid = _OPEN_5X5.copy()
        for row, col in ((1, 2), (3, 2), (2, 1), (2, 3)):
            grid[row, col] = 1
        env = _MaskEnv(grid, [(2, 2)])
        # 全 0 掩码会让 softmax 产生 NaN，退化成全 1 由碰撞惩罚兜底。
        np.testing.assert_array_equal(self.mask_fn(env, 0), np.ones(4, dtype=np.float32))


class _ScriptedAgents:
    """``get_probs`` 返回预设概率的假策略网络（部署测试只需要这一个接口）。"""

    def __init__(self, probs: list[float]) -> None:
        self._probs = np.asarray(probs, dtype=np.float32)

    def get_probs(self, obs, pheromone, batch_size=1):
        import torch

        return torch.as_tensor(
            np.tile(self._probs, (int(batch_size), 1)), dtype=torch.float32
        )


class _ScriptedPolicy:
    def __init__(self, probs: list[float]) -> None:
        self.agents = _ScriptedAgents(probs)


def _two_channelmap() -> dict[str, dict[str, np.ndarray]]:
    zeros = np.zeros((60, 60), dtype=np.float32)
    return {"agent_0": {"obstacle_map": zeros.copy(), "explore_map": zeros.copy()}}


def _wall_below_env() -> _MaskEnv:
    """60x60 开阔地图，agent_0 在 (2, 2)，其下方向 (3, 2) 是墙。"""
    grid = np.zeros((60, 60), dtype=np.int32)
    grid[3, 2] = 1
    return _MaskEnv(grid, [(2, 2)])


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class DeploymentMaskRegressionTests(unittest.TestCase):
    """评审要求：非法方向持有最高原始概率时，生产推理仍必须选合法动作。"""

    def setUp(self) -> None:
        self.train = _load_train_module()
        self.device = self.train.torch.device("cpu")
        self.env = _wall_below_env()
        # 概率质量几乎全在非法动作 1 上——掩码失效时的部署撞墙场景。
        self.policy = _ScriptedPolicy([0.05, 0.90, 0.03, 0.02])
        self.two_channelmap = _two_channelmap()

    def test_all_four_paths_share_one_function_object(self) -> None:
        evaluation = _load_eval_module()
        self.assertIs(self.train.legal_action_mask, evaluation.legal_action_mask)

    def test_training_sampling_never_picks_the_illegal_action(self) -> None:
        self.train.torch.manual_seed(0)
        for _ in range(100):
            actions, _, masks = self.train.select_action(
                self.two_channelmap, self.policy, self.env, ["agent_0"], self.device
            )
            self.assertNotEqual(actions["agent_0"], 1)
            self.assertEqual(float(masks["agent_0"][1]), 0.0)

    def test_deterministic_validation_never_picks_the_illegal_action(self) -> None:
        actions = self.train.select_action_deterministic(
            self.two_channelmap, self.policy, self.env, ["agent_0"], self.device
        )
        self.assertNotEqual(actions["agent_0"], 1)

    def test_evaluation_sampling_never_picks_the_illegal_action(self) -> None:
        # The final-eval CLI replaced the old ``select_action`` with
        # ``policy_actions(two_channelmap, policy, env, agents, device, sampled)``;
        # the masked-sampling path must stay legal, and so must the deterministic
        # argmax it defaults to.
        evaluation = _load_eval_module()
        self.train.torch.manual_seed(0)
        for _ in range(100):
            actions = evaluation.policy_actions(
                self.two_channelmap, self.policy, self.env, ["agent_0"], self.device, True
            )
            self.assertNotEqual(actions["agent_0"], 1)
        actions = evaluation.policy_actions(
            self.two_channelmap, self.policy, self.env, ["agent_0"], self.device, False
        )
        self.assertNotEqual(actions["agent_0"], 1)

    def test_runtime_argmax_never_picks_the_illegal_action(self) -> None:
        from integrated_sar import piloc_runtime

        provider = piloc_runtime.TrainedPilocPolicy(
            self.policy,
            device="cpu",
            view_radius=5,
            legal_action_mask=self.train.legal_action_mask,
        )
        actions = provider(self.env, {}, self.two_channelmap)
        self.assertNotEqual(actions["agent_0"], 1)

    def test_the_negative_control_does_pick_the_illegal_action(self) -> None:
        """负对照：掩码失效（恒全 1）时 argmax 就会撞墙——证明本测试有区分力。"""
        from integrated_sar import piloc_runtime

        bypass = piloc_runtime.TrainedPilocPolicy(
            self.policy,
            device="cpu",
            view_radius=5,
            legal_action_mask=lambda env, index: np.ones(4, dtype=np.float64),
        )
        actions = bypass(self.env, {}, self.two_channelmap)
        self.assertEqual(actions["agent_0"], 1)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class BufferFullArrayRoundtripTests(unittest.TestCase):
    """``state_dict`` 必须全量存取：残留行参与 learn 的梯度更新，不能丢。"""

    def setUp(self) -> None:
        _load_train_module()  # 顺带把 piloc 挂上 sys.path
        from src.DDQN.buffer import Buffer_for_PPO

        self.Buffer_for_PPO = Buffer_for_PPO

    def _buffer(self) -> Any:
        return self.Buffer_for_PPO(capacity=8, obs_dim=9, act_dim=1, device="cpu")

    def _row(self, index: int, base: float) -> dict:
        rng = np.random.RandomState(1000 + index)
        return {
            "obs1": (rng.rand(60, 60) + base).astype(np.float32),
            "obs2": (rng.rand(60, 60) + base).astype(np.float32),
            "obs_pheromone": (rng.rand(11, 11) + base).astype(np.float32),
            "action": float(index % 4),
            "reward": float(np.float32(rng.rand())),
            "next_obs1": (rng.rand(60, 60) + base).astype(np.float32),
            "next_obs2": (rng.rand(60, 60) + base).astype(np.float32),
            "next_obs_pheromone": (rng.rand(11, 11) + base).astype(np.float32),
            "done": bool(index == 7),
            "adv_done": bool(index % 5 == 4),
        }

    def _add(self, buffer: Any, row: dict) -> None:
        buffer.add(
            row["obs1"], row["obs2"], row["obs_pheromone"], row["action"], row["reward"],
            row["next_obs1"], row["next_obs2"], row["next_obs_pheromone"], row["done"],
            np.zeros(buffer.action_log_probs[0].shape, dtype=np.float32),
            row["adv_done"], self._legal_mask(row["action"]),
        )

    @staticmethod
    def _legal_mask(action: float) -> np.ndarray:
        mask = np.ones(4, dtype=np.float32)
        mask[(int(action) + 1) % 4] = 0.0
        return mask

    def test_state_dict_keeps_residual_rows_from_the_previous_window(self) -> None:
        buffer = self._buffer()
        for index in range(8):  # 上一窗口：learn 之后 clear，数组内容仍在
            self._add(buffer, self._row(index, 0.0))
        buffer.clear()
        for index in range(3):  # 本窗口已写入 3 行
            self._add(buffer, self._row(index, 100.0))

        state = buffer.state_dict()
        self.assertEqual(state["size"], 3)
        self.assertEqual(state["index"], 3)
        np.testing.assert_array_equal(state["obs1"][0], self._row(0, 100.0)["obs1"])
        # 行 3..7 必须仍是上一窗口的残留数据，而不是零。
        np.testing.assert_array_equal(state["obs1"][3], self._row(3, 0.0)["obs1"])
        self.assertEqual(len(state["rewards"]), 8)
        self.assertEqual(len(state["dones"]), 8)

    def test_load_state_dict_restores_every_row_and_the_consumed_view(self) -> None:
        source = self._buffer()
        for index in range(8):
            self._add(source, self._row(index, 0.0))
        source.clear()
        for index in range(3):
            self._add(source, self._row(index, 100.0))

        restored = self._buffer()
        restored.load_state_dict(source.state_dict())
        self.assertEqual(restored._index, source._index)
        self.assertEqual(restored._size, source._size)
        fields = ("obs1", "obs2", "obs_pheromone", "actions", "rewards",
                  "next_obs1", "next_obs2", "next_obs_pheromone", "dones",
                  "action_log_probs", "adv_dones", "legal_masks")
        for name in fields:
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(source, name), err_msg=name
            )
        # learn() 实际消费的张量视图也必须逐位一致。
        for name, tensor_source, tensor_restored in zip(
            fields, source.all(), restored.all()
        ):
            self.assertTrue(self.train_torch().equal(tensor_source, tensor_restored), name)

    @staticmethod
    def train_torch():
        import torch

        return torch

    def test_load_state_dict_none_resets_to_an_empty_window(self) -> None:
        buffer = self._buffer()
        self._add(buffer, self._row(0, 0.0))
        buffer.load_state_dict(None)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer._index, 0)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class ResumeEquivalenceTests(unittest.TestCase):
    """N+M ≡ N+resume M：恢复后的 learn 必须与连续训练逐位一致。

    用真实的 ``HORIZON=256``：``Agent.get_value`` 硬编码 ``batch_size=256``，
    生产约束 horizon == minibatch == 256，等价性测试必须遵守同一约束。
    """

    HORIZON = 256

    def setUp(self) -> None:
        self.train = _load_train_module()
        from src.curriculum import CurriculumGate, default_stages

        self.CurriculumGate = CurriculumGate
        self.default_stages = default_stages

    def _policy(self) -> Any:
        return self.train.MAPPO(
            {"agent_0": [9, 5], "agent_1": [9, 5]},
            actor_lr=1e-3,
            critic_lr=1e-3,
            horizon=self.HORIZON,
            device="cpu",
            trick=dict(self.train.configs.TRICK),
            num_agent=2,
        )

    def _learn(self, policy: Any) -> None:
        policy.learn(
            minibatch_size=self.HORIZON, gamma=0.9, lmbda=0.95, clip_param=0.2,
            K_epochs=1, entropy_coefficient=0.01,
        )

    def _row(self, index: int, base: float) -> dict:
        # 局部 RandomState：构造数据不消耗全局 RNG，路径 A/B 的 learn 才能
        # 从同一个 RNG 状态出发产生同一个 permutation。float32 与落盘精度一致。
        rng = np.random.RandomState(2000 + index)
        return {
            "obs1": (rng.rand(60, 60) + base).astype(np.float32),
            "obs2": (rng.rand(60, 60) + base).astype(np.float32),
            "obs_pheromone": (rng.rand(11, 11) + base).astype(np.float32),
            "action": float(index % 4),
            "reward": float(np.float32(rng.rand())),
            "next_obs1": (rng.rand(60, 60) + base).astype(np.float32),
            "next_obs2": (rng.rand(60, 60) + base).astype(np.float32),
            "next_obs_pheromone": (rng.rand(11, 11) + base).astype(np.float32),
            "done": bool(index == self.HORIZON - 1),
            "adv_done": bool(index % 5 == 4),
        }

    def _add_row(self, policy: Any, row: dict) -> None:
        for buffer in policy.buffers.values():
            mask = np.ones(4, dtype=np.float32)
            mask[(int(row["action"]) + 1) % 4] = 0.0
            buffer.add(
                row["obs1"], row["obs2"], row["obs_pheromone"], row["action"], row["reward"],
                row["next_obs1"], row["next_obs2"], row["next_obs_pheromone"], row["done"],
                np.zeros(buffer.action_log_probs[0].shape, dtype=np.float32),
                row["adv_done"], mask,
            )

    def _gate(self) -> Any:
        return self.CurriculumGate(stages=self.default_stages([4, 8]), eval_every=2)

    def test_resume_matches_continuous_learning_bit_for_bit(self) -> None:
        torch = self.train.torch
        torch.manual_seed(7)
        np.random.seed(7)
        random.seed(7)

        # ---- 连续训练到检查点：learn#1（产生 Adam 状态与窗口残留）→ 中途存档 ----
        policy_a = self._policy()
        for index in range(self.HORIZON):
            self._add_row(policy_a, self._row(index, 0.0))
        self._learn(policy_a)
        for index in range(3):  # 检查点落在窗口中途：3 行新数据 + 253 行残留
            self._add_row(policy_a, self._row(index, 100.0))
        gate_a = self._gate()
        for _ in range(5):
            gate_a.register_episode()
        gate_a.submit_validation({"coverage_gain": 0.9, "repeat_rate": 0.95})
        state = self.train.build_trainer_state(
            episode_num=7,
            global_step=777,
            best_validation_score=0.123,
            gate=gate_a,
            early_stop_failures=1,
            buffers=policy_a.buffers,
        )
        rows_m = [self._row(index, 200.0) for index in range(self.HORIZON - 3)]

        with tempfile.TemporaryDirectory() as tmp:
            policy_a.save(tmp, 7, trainer_state=state)
            checkpoint = Path(tmp) / "shared_mappo_model_21_7.pth"
            self.assertTrue(checkpoint.is_file())

            # ---- 路径 A：连续训练收满窗口并再更新一轮 ----
            for row in rows_m:
                self._add_row(policy_a, row)
            self._learn(policy_a)

            # ---- 路径 B：故意污染全局 RNG，恢复必须把它扳回检查点时刻 ----
            torch.manual_seed(12345)
            np.random.seed(12345)
            random.seed(12345)
            policy_b = self._policy()
            gate_b = self._gate()
            episode, bundle = self.train.resume_policy(policy_b, checkpoint, torch.device("cpu"))
            self.assertEqual(episode, 7)
            self.assertIsNotNone(bundle)
            self.assertEqual(int(bundle["episode_num"]), 7)
            self.assertEqual(int(bundle["global_step"]), 777)
            self.assertEqual(int(bundle["early_stop_failures"]), 1)
            self.train.restore_trainer_state(bundle, gate_b)
            self.train.restore_buffers(policy_b, bundle)
            for row in rows_m:
                self._add_row(policy_b, row)
            self._learn(policy_b)

        # 课程与评估节奏也必须与存档时刻一致。
        self.assertEqual(gate_b.state_dict(), gate_a.state_dict())
        # 恢复后再训练的参数必须与连续训练逐位一致（权重/优化器/buffer/RNG 同源）。
        for (name, weight_a), (_, weight_b) in zip(
            policy_a.agents.full_model.state_dict().items(),
            policy_b.agents.full_model.state_dict().items(),
        ):
            self.assertTrue(torch.equal(weight_a, weight_b), f"resume diverged: {name}")

    def test_mismatched_embedded_episode_is_refused(self) -> None:
        """episode 一致性校验：权重与 trainer state 不同轮的 checkpoint 必须拒载。"""
        torch = self.train.torch
        torch.manual_seed(11)
        policy = self._policy()
        state = self.train.build_trainer_state(
            episode_num=8,
            global_step=0,
            best_validation_score=-float("inf"),
            gate=self._gate(),
            buffers=policy.buffers,
        )
        with tempfile.TemporaryDirectory() as tmp:
            policy.save(tmp, 7, trainer_state=state)  # 权重记 7，状态记 8
            checkpoint = Path(tmp) / "shared_mappo_model_21_7.pth"
            with self.assertRaises(ValueError):
                self.train.resume_policy(self._policy(), checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
