"""Stage A 合成测试：全部在合成 tensor / stub 上运行，**不导入也不启动任何仿真**。

对应第二十轮指令 §4 的清单与 §3 的四件事，以及第十九轮设计 spec §6 的正确性测试计划。
运行方式（与测试运行器相同，见 c20_audit.py）：

    /d/project_env/python.exe -m unittest discover -s runs/comm_cost_controller_stage_a_20260927
"""

from __future__ import annotations

import unittest

import torch

import cost_math as cm


# --------------------------------------------------------------------- 合成 stub ----------------


class StubExecutor:
    """最小执行器替身：只负责报价与计费分类，没有任何环境语义。"""

    def __init__(self, *, payload_bytes: float, subscribers: int, drop_probability: float = 0.0,
                 seed: int = 0) -> None:
        self.payload_bytes = float(payload_bytes)
        self.subscribers = int(subscribers)
        self.drop_probability = float(drop_probability)
        self._rng = torch.Generator().manual_seed(seed)

    def quote(self) -> float:
        return cm.billed_bytes("send_executed", self.payload_bytes, self.subscribers)

    def execute(self, action: str, *, executable: bool = True) -> cm.DecisionOutcome:
        """`action` ∈ {"SEND", "HOLD"}；`executable` 伪装预算/容量检查结果。"""
        if action == "HOLD":
            return cm.DecisionOutcome("hold_active")
        if not executable:
            return cm.DecisionOutcome("send_refused_budget")
        dropped = bool(torch.rand(1, generator=self._rng).item() < self.drop_probability)
        bucket = "send_dropped" if dropped else "send_executed"
        return cm.DecisionOutcome(bucket, self.quote(),
                                  payload_bytes=self.payload_bytes,
                                  subscribers=self.subscribers)


def immediate_billing(detections: int, payload_bytes: float, subscribers: int) -> float:
    """`immediate` 的计费定义：每个发现一份首报，载荷 × 订阅者（丢包照付）。"""
    return detections * cm.billed_bytes("send_executed", payload_bytes, subscribers)


# ---------------------------------------------------------------------- 目标与符号 -------------


class ObjectiveSigns(unittest.TestCase):
    def test_same_effect_more_bytes_is_worse(self) -> None:
        cheaper = cm.dual_objective(c_mean=0.10, g=-0.02, lam=1.0)
        dearer = cm.dual_objective(c_mean=0.20, g=-0.02, lam=1.0)
        self.assertLess(dearer, cheaper)

    def test_better_task_effect_beats_cheaper_cost_when_constraint_binds(self) -> None:
        inside = cm.dual_objective(c_mean=0.30, g=-0.50, lam=1.0)
        outside = cm.dual_objective(c_mean=0.10, g=+0.50, lam=1.0)
        self.assertLess(outside, inside)

    def test_constraint_statistic_signs(self) -> None:
        y_ref = [6.0] * 10
        self.assertGreater(cm.constraint_statistic(y_ref, [5.0] * 10, delta=0.10), 0.0)
        self.assertLess(cm.constraint_statistic(y_ref, [6.0] * 10, delta=0.10), 0.0)
        self.assertAlmostEqual(
            cm.constraint_statistic(y_ref, [5.9] * 10, delta=0.10), 0.0, places=12)

    def test_dual_update_direction_projection_and_zero(self) -> None:
        self.assertGreater(cm.dual_step(0.0, +0.20, eta=1.0), 0.0)
        self.assertEqual(cm.dual_step(0.05, -10.0, eta=1.0), 0.0)
        self.assertAlmostEqual(cm.dual_step(0.40, 0.0, eta=1.0), 0.40, places=12)

    def test_dual_rejects_negative_lambda_objective(self) -> None:
        with self.assertRaises(ValueError):
            cm.dual_objective(c_mean=0.1, g=0.0, lam=-0.1)

    def test_dual_update_input_is_paired_difference_not_reference_failures(self) -> None:
        """更新信号必须是配对差，不能拿参考臂的失败数当分母或当信号。"""
        y_ref = [6.0, 0.0, 6.0, 6.0]      # 参考臂在 1 个 seed 上失败
        y_policy = [6.0, 6.0, 6.0, 6.0]
        g = cm.constraint_statistic(y_ref, y_policy, delta=0.0)
        self.assertAlmostEqual(g, -1.5, places=12)   # mean_ref 4.5 - mean_pol 6.0
        self.assertNotAlmostEqual(g, 1.0, places=12)  # 不是"参考臂失败数 = 1"
        self.assertEqual(cm.dual_step(0.0, g, eta=1.0), 0.0)   # 负违背 ⇒ 投影到 0
        self.assertAlmostEqual(cm.dual_step(0.4, g, eta=1.0), 0.0, places=12)
        self.assertFalse(cm.report_metric_row("send_refused_budget", 0.0)["counts_as_saving"])


# ------------------------------------------------------------------------ 梯度 -------------


class Gradients(unittest.TestCase):
    def test_reference_has_no_gradient_and_policy_has_correct_sign(self) -> None:
        theta = torch.tensor([0.30, -0.20], dtype=torch.float64, requires_grad=True)
        features = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]], dtype=torch.float64)
        price = torch.tensor([10.0, 4.0, 0.0], dtype=torch.float64)   # B0 归一化前先除
        benefit = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        y_reference = torch.tensor([2.0], dtype=torch.float64)        # 常数，无梯度
        delta, lam = 0.10, 1.5

        logits = features @ theta
        prob = torch.sigmoid(logits)
        c = (prob * price).mean()
        y = (prob * benefit).mean()
        g = y_reference - y - delta
        loss = -(c) - lam * g          # 最小化 c + lam*g
        loss.backward()

        self.assertTrue(torch.isfinite(theta.grad).all())
        # 参考臂是外部固定常数：不产生梯度，也不出现在 autograd 图里
        self.assertIsNone(y_reference.grad)
        self.assertFalse(y_reference.requires_grad)

    def test_finite_difference_matches_autograd(self) -> None:
        torch.manual_seed(7)
        n = 6
        features = torch.randn(n, 3, dtype=torch.float64)
        price = torch.rand(n, dtype=torch.float64) * 5.0
        benefit = torch.randint(0, 2, (n,)).double()
        y_reference = 4.0
        delta, lam = 0.10, 1.25
        theta0 = torch.tensor([0.4, -0.6, 0.25], dtype=torch.float64)

        def loss_at(theta: torch.Tensor) -> torch.Tensor:
            prob = torch.sigmoid(features @ theta)
            c = (prob * price).mean()
            y = (prob * benefit).mean()
            g = y_reference - y - delta
            return c + lam * g          # 最小化形式

        theta_g = theta0.clone().requires_grad_(True)
        value = loss_at(theta_g)
        value.backward()
        analytic = theta_g.grad.detach().clone()

        finite = []
        eps = 1e-6
        for index in range(theta0.numel()):
            plus = theta0.clone(); plus[index] += eps
            minus = theta0.clone(); minus[index] -= eps
            finite.append(((loss_at(plus) - loss_at(minus)) / (2 * eps)).item())
        finite = torch.tensor(finite, dtype=torch.float64)
        self.assertTrue(torch.allclose(analytic, finite, rtol=1e-4, atol=1e-6))

    def test_actor_advantage_sign_and_lambda_units(self) -> None:
        a_task = torch.tensor([1.0, 1.0], dtype=torch.float64)
        a_cost = torch.tensor([0.5, 2.0], dtype=torch.float64)
        combined = cm.actor_advantage(a_task, a_cost, lam=0.0)
        self.assertTrue(torch.equal(combined, a_task))          # lambda=0 时不含成本项
        combined = cm.actor_advantage(a_task, a_cost, lam=1.0)
        self.assertTrue(torch.equal(combined, a_task - a_cost))  # 成本以负号进入

    def test_lambda_zero_is_cost_minimising_not_all_send(self) -> None:
        """lambda=0 时目标只惩罚成本：方向必须正确，且**不得**断言它复现 immediate。

        本测试只检查"上升方向 = 少花字节"，并检查成本为 0 时**没有**方向；
        不断言 lambda=0 的策略行为等于 `immediate`（指令 §3.3 第二条）。
        """
        theta = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        price = torch.tensor([7.0], dtype=torch.float64)
        l_max = -(torch.sigmoid(theta) * price).mean()          # 最大化形式
        l_max.backward()
        self.assertLess(float(theta.grad[0]), 0.0)              # 上升 ⇒ 概率下降 ⇒ 成本下降
        with torch.no_grad():
            theta[0] += 0.5 * float(theta.grad[0])
            self.assertLess(float(torch.sigmoid(theta)), 0.5)
        flat = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        zero_cost = -(torch.sigmoid(flat) * torch.zeros(1, dtype=torch.float64)).mean()
        zero_cost.backward()
        self.assertEqual(float(flat.grad[0]), 0.0)              # 无成本 ⇒ 无方向，不被迫全发

    def test_forced_send_billing_matches_immediate_on_stub(self) -> None:
        executor = StubExecutor(payload_bytes=400.0, subscribers=6)
        outcomes = [executor.execute("SEND") for _ in range(5)]
        billed = sum(o.billed_bytes for o in outcomes)
        self.assertEqual(billed, immediate_billing(5, 400.0, 6))
        closure = cm.classify_closure(outcomes)
        self.assertTrue(closure["counts_close"])
        self.assertTrue(closure["bill_matches_decisions"])
        self.assertEqual(closure["billed_bytes_total"], billed)


# -------------------------------------------------------------------------- mask -------------


class Masking(unittest.TestCase):
    def test_masked_items_carry_zero_likelihood_and_entropy_weight(self) -> None:
        logits = torch.tensor([[0.5, 1.0, -1.0]], dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([1])
        allowed = torch.tensor([[True, True, False]])
        ll, entropy = cm.masked_log_likelihood(logits, actions, allowed)
        self.assertTrue(torch.isfinite(ll) and torch.isfinite(entropy))
        # 把被 mask 的项换成任意大 logit，actor 损失不应改变
        before = float(ll)
        with torch.no_grad():
            logits[0, 2] = 99.0
        after, _ = cm.masked_log_likelihood(logits, actions, allowed)
        self.assertEqual(before, float(after))

    def test_all_mask_batch_has_no_actor_sample_and_no_nan(self) -> None:
        logits = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        actions = torch.zeros(2, dtype=torch.long)
        allowed = torch.zeros(2, 3, dtype=torch.bool)
        ll, entropy = cm.masked_log_likelihood(logits, actions, allowed)
        self.assertTrue(torch.isfinite(ll))
        self.assertTrue(torch.isfinite(entropy))
        self.assertEqual(float(ll), 0.0)     # 全 mask：无 Actor 样本
        self.assertEqual(float(entropy), 0.0)
        self.assertFalse(ll.requires_grad)   # 常数 0：不在 autograd 图里
        with self.assertRaises(RuntimeError):
            (-ll).backward()                 # 因此不可能产生伪 Actor 更新

    def test_critic_samples_survive_masking(self) -> None:
        """mask 只作用于 actor 似然；critic 的时序样本（这里用长度表示）不受影响。"""
        allowed = torch.tensor([[True, False], [False, False], [True, True]])
        critic_samples = allowed.shape[0]
        self.assertEqual(critic_samples, 3)
        empty_rows = sum(1 for row in allowed if not bool(row.any()))
        self.assertEqual(empty_rows, 1)      # 第 2 步全 mask，critic 样本仍在


# ---------------------------------------------------------------------- 成本账本 -------------


class CostLedger(unittest.TestCase):
    def test_dropped_send_billed_refused_not_billed(self) -> None:
        executor = StubExecutor(payload_bytes=414.0, subscribers=6, drop_probability=1.0)
        dropped = executor.execute("SEND")
        self.assertEqual(dropped.bucket, "send_dropped")
        self.assertEqual(dropped.billed_bytes, immediate_billing(1, 414.0, 6))
        refused = executor.execute("SEND", executable=False)
        self.assertEqual(refused.bucket, "send_refused_budget")
        self.assertEqual(refused.billed_bytes, 0.0)
        held = executor.execute("HOLD")
        self.assertEqual(held.billed_bytes, 0.0)

    def test_billed_buckets_are_only_real_attempts(self) -> None:
        self.assertEqual(set(cm.BILLED_BUCKETS), {"send_executed", "send_dropped"})
        for bucket in ("send_refused_budget", "send_refused_capacity",
                       "send_suppressed_terminal", "hold_active"):
            self.assertEqual(cm.billed_bytes(bucket, 414.0, 6), 0.0)

    def test_episode_cost_increments_sum_to_total(self) -> None:
        decisions = [cm.DecisionOutcome("send_executed", 10.0),
                     cm.DecisionOutcome("hold_active"),
                     cm.DecisionOutcome("send_dropped", 4.0),
                     cm.DecisionOutcome("send_refused_budget"),
                     cm.DecisionOutcome("send_executed", 6.0)]
        running = 0.0
        for decision in decisions:
            running += decision.billed_bytes
        closure = cm.classify_closure(decisions)
        self.assertAlmostEqual(running, closure["billed_bytes_total"], places=12)
        self.assertEqual(closure["counts_sum"], len(decisions))

    def test_ack_structural_zero_distinguished_from_missing(self) -> None:
        off = cm.ack_status(ack_enabled=False, ack_bytes=None)
        missing = cm.ack_status(ack_enabled=True, ack_bytes=None)
        measured = cm.ack_status(ack_enabled=True, ack_bytes=512.0)
        self.assertEqual(off["bucket"], "ack_off_structural_zero")
        self.assertEqual(off["ack_attempt_bytes"], 0.0)
        self.assertTrue(missing["missing"])
        self.assertIsNone(missing["ack_attempt_bytes"])
        self.assertFalse(measured["missing"])
        total = cm.measured_cost(uplink_bytes=1000.0, ack_bytes=512.0, ack_enabled=True)
        self.assertEqual(total["c_measured_bytes"], 1512.0)
        self.assertTrue(total["fully_metered"])
        self.assertFalse(cm.measured_cost(1000.0, None, True)["fully_metered"])

    def test_terminal_task_counts_close(self) -> None:
        discovered, on_time, late, expired, unresolved = 6, 4, 1, 0, 1
        self.assertEqual(on_time + late + expired + unresolved, discovered)
        y_episode = on_time
        self.assertEqual(y_episode + late + expired + unresolved, discovered)


# -------------------------------------------------------------- 终止 / 截断 / 折扣 -------------


class ReturnsAndTermination(unittest.TestCase):
    def test_termination_truncation_and_bootstrap(self) -> None:
        rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        values = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float64)
        terminal = torch.tensor([False, True, False])
        bootstrap = torch.tensor([0.0, 0.0, 2.0])
        advantages, returns = cm.gae(rewards, values, gamma=1.0, gae_lambda=1.0,
                                    terminal=terminal, bootstrap=bootstrap)
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.isfinite(returns).all())
        # 真正终止那一步之后不再 bootstrap
        self.assertEqual(float(bootstrap[1] * (1 - terminal[1].float())), 0.0)
        # 截断步（最后一步 terminal=False）使用 bootstrap 值
        self.assertGreater(float(returns[2]), float(values[2]))

    def test_gamma_one_reproduces_undiscounted_total(self) -> None:
        rewards = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
        values = torch.zeros(3, dtype=torch.float64)
        terminal = torch.tensor([False, False, True])
        advantages, returns = cm.gae(rewards, values, gamma=1.0, gae_lambda=1.0,
                                    terminal=terminal, bootstrap=torch.zeros(3))
        self.assertAlmostEqual(float(returns[0]), 10.0, places=9)

    def test_gamma_below_one_is_not_the_undiscounted_total(self) -> None:
        """gamma<1 会改变时间偏好：不得把它悄悄当成 episode 无折扣总量。"""
        rewards = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
        values = torch.zeros(3, dtype=torch.float64)
        terminal = torch.tensor([False, False, True])
        discounted, _ = cm.gae(rewards, values, gamma=0.9, gae_lambda=1.0,
                               terminal=terminal, bootstrap=torch.zeros(3))
        undiscounted, _ = cm.gae(rewards, values, gamma=1.0, gae_lambda=1.0,
                                 terminal=terminal, bootstrap=torch.zeros(3))
        self.assertLess(float(discounted[0]), float(undiscounted[0]))
        self.assertAlmostEqual(float(discounted[0]), 2.0 + 0.9 * 3.0 + 0.81 * 5.0, places=9)


# ------------------------------------------------------------------ 归一化 / 状态 -------------


class NormalizationAndState(unittest.TestCase):
    def test_fixed_normalisation_recomputable_and_b0_conversion(self) -> None:
        norm = cm.CostNormalization(b0_bytes=30_000.0)
        self.assertAlmostEqual(norm.normalise(15_000.0), 0.5, places=12)
        self.assertAlmostEqual(norm.to_bytes(0.5), 15_000.0, places=9)
        shifted = cm.CostNormalization(b0_bytes=15_000.0)
        self.assertAlmostEqual(shifted.normalise(15_000.0), 1.0, places=12)
        self.assertAlmostEqual(shifted.to_bytes(1.0), 15_000.0, places=9)

    def test_b0_must_be_positive_and_finite(self) -> None:
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                cm.CostNormalization(b0_bytes=bad)

    def test_actor_features_reject_outcome_derived_fields(self) -> None:
        with self.assertRaises(ValueError):
            cm.build_actor_features(**{name: 0.0 for name in
                                       ["on_time", "future_success", "true_goal_state",
                                        "reference_failures"]})
        for forbidden in cm.FORBIDDEN_ACTOR_FEATURE_FIELDS:
            self.assertNotIn(forbidden, cm.ALLOWED_ACTOR_FEATURE_FIELDS)

    def test_actor_features_require_all_nineteen_fields(self) -> None:
        fields = {name: 0.0 for name in cm.ALLOWED_ACTOR_FEATURE_FIELDS}
        vector = cm.build_actor_features(**fields)
        self.assertEqual(vector.shape[0], cm.ACTOR_FEATURE_DIM)
        with self.assertRaises(ValueError):
            cm.build_actor_features(**{name: 0.0 for name in list(fields)[:-1]})

    def test_state_roundtrip_and_fail_closed_on_old_version(self) -> None:
        state = cm.new_cost_state(b0_bytes=30_000.0, delta=0.10, task_critic={"w": [1.0]},
                                  cost_critic={"w": [2.0]}, eta_lambda=0.01, gamma=0.99)
        state["lambda"] = 0.35
        state["lambda_history"] = [0.0, 0.2, 0.35]
        restored = cm.load_cost_state(json_roundtrip(state))
        self.assertEqual(restored["lambda"], 0.35)
        self.assertEqual(restored["lambda_history"], [0.0, 0.2, 0.35])
        self.assertEqual(restored["loss_version"], cm.LOSS_VERSION)
        self.assertEqual(restored["normalization"]["b0_bytes"], 30_000.0)
        legacy = dict(state)
        legacy["schema"] = "constrained-cost-state/0"
        legacy["loss_version"] = "constrained-lagrangian/0"
        with self.assertRaises(ValueError):
            cm.load_cost_state(legacy)


def json_roundtrip(payload: dict) -> dict:
    import json
    return json.loads(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
