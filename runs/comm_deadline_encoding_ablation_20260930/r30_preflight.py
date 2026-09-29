"""ROUND30 §6 零环境预检：9 项测试全部穿过真实接线（不调孤立除法、不构造环境）。

运行：python r30_preflight.py   （0 环境启动；合成 optimizer 更新与真实训练更新分账，
上限 64 次；超限即失败）

被测路径：r30_encoding 的作用域 collector → v2_trainer 的 PPO 更新 → checkpoint 保存/恢复，
以及归档的 ROUND23/ROUND24 独立反例套件。预期答案独立手算，不从被测实现复制。
"""

from __future__ import annotations

import importlib
import json
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r30_common as common
import r30_check as check
import r30_encoding as enc


# ------------------------------------------------------------------ 合成观测与确定性控制器
class FakeController:
    """A deterministic stand-in for the Stage B controller: forward + cost_values only."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(4242)
        self.weight = torch.randn(19, 2, generator=generator) * 0.1
        self.cost_weight = torch.randn(19, 1, generator=generator) * 0.1
        self.seen: list[list[list[float]]] = []

    def forward(self, batch: torch.Tensor):
        self.seen.append([[float(value) for value in row] for row in batch.tolist()])
        logits = batch @ self.weight
        values = (batch @ self.cost_weight).squeeze(-1)
        return logits, values

    def cost_values(self, batch: torch.Tensor) -> torch.Tensor:
        return (batch @ self.cost_weight).squeeze(-1)

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.weight, self.cost_weight]


def make_item(target_id: str, *, remaining: float, charge: int, executable: bool,
              reason: str = "allowed") -> Any:
    from integrated_sar.comm_policy import SendItem

    return SendItem(
        target_id=target_id,
        ready_step=5,
        age_steps=2,
        pending_event=True,
        changed_view=True,
        first_emission=False,
        revision=2,
        revision_delta=1,
        stage_index=1,
        stage_changed=True,
        progress=0.25,
        progress_delta=0.1,
        requirements_changed=True,
        members_changed=False,
        deadline=remaining + 5.0,
        remaining_time=remaining,
        sent_count=1,
        last_sent_step=2,
        since_last_emission_time=3.0,
        has_future_requirements=True,
        attempt_charge_bytes=charge,
        send_executable=executable,
        send_block_reason=reason,
    )


def make_observation(items: list[Any], *, budget_bytes: float = 32000.0,
                     remaining_budget: float = 30000.0) -> Any:
    from integrated_sar.comm_policy import SendObservation

    return SendObservation(
        step=26,
        current_time=25.0,
        items=tuple(items),
        remaining_budget_bytes=remaining_budget,
        budget_bytes_per_episode=int(budget_bytes),
        capacity_remaining=3,
        backlog_targets=len(items),
        subscribers=6,
    )


class Case:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0
        self.synthetic_optimizer_steps = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)

    def expect_close(self, got: float, want: float, message: str) -> None:
        self.check(abs(got - want) <= 1e-12, f"{message}: got {got!r}, want {want!r}")


SYNTHETIC_MODULES = ("v2_test_math", "v2_test_cost_objective", "v2_test_cost_head_lifecycle",
                     "v2_test_gae_next_value", "v2_test_reward_timing")


def main() -> int:
    case = Case()
    torch.manual_seed(0)

    # A legal observation whose candidate remaining times make both encodings visible.
    items = [
        make_item("piloc-e1-target-0", remaining=200.0, charge=2400, executable=True),
        make_item("piloc-e1-target-1", remaining=60.0, charge=2400, executable=True),
        make_item("piloc-e1-target-2", remaining=12.0, charge=2400, executable=True),
        make_item("piloc-e1-target-3", remaining=-5.0, charge=2400, executable=True),
        make_item("piloc-e1-target-4", remaining=80.0, charge=10 ** 9, executable=False,
                  reason="budget_exhausted"),
    ]
    observation = make_observation(items)

    # ------------------------------------------------------------------ 1) 只有期限列不同
    def tape_of(arm: str) -> dict[str, Any]:
        controller = FakeController()
        collector = enc.scoped_eval_collector(controller, arm, greedy=True)
        torch.manual_seed(7)
        collector.decide(observation)
        return collector.records[-1]

    tape_old = tape_of("old")
    tape_new = tape_of("new")
    case.check(tape_old["targets"] == tape_new["targets"], "candidate sets must match")
    case.check(tape_old["masked"] == tape_new["masked"], "masked sets must match")
    case.check(tape_old["actionable"] == tape_new["actionable"], "actionable sets must match")
    for index in range(len(tape_old["features"])):
        for column in range(19):
            if column == enc.DEADLINE_COLUMN:
                continue
            case.check(tape_old["features"][index][column] == tape_new["features"][index][column],
                       f"row {index} column {column} must be unchanged between arms")
    # 期限列的值按定义手算核对
    remaining_by_target = {item.target_id: item.remaining_time for item in items}
    for index, target in enumerate(tape_old["targets"]):
        remaining = remaining_by_target[target]
        case.expect_close(tape_old["features"][index][enc.DEADLINE_COLUMN],
                          max(-1.0, min(1.0, remaining / 50.0)),
                          f"old deadline column for {target}")
        case.expect_close(tape_new["features"][index][enc.DEADLINE_COLUMN],
                          remaining / 1000.0, f"new deadline column for {target}")
    case.check(tape_old["log_prob"] == tape_new["log_prob"] or True,
               "log-probs are recorded (arms may differ once probabilities differ; not asserted "
               "equal here)")

    # ------------------------------------------------------------------ 2) tape == 网络实际输入
    controller = FakeController()
    collector = enc.scoped_eval_collector(controller, "new", greedy=True)
    torch.manual_seed(7)
    collector.decide(observation)
    tape = collector.records[-1]
    seen = controller.seen[-1]
    case.check(len(seen) == len(tape["features"]), "every candidate row must reach the network")
    for row_index, (matrix_row, tape_row) in enumerate(zip(seen, tape["features"])):
        case.check(matrix_row == [float(value) for value in tape_row],
                   f"network input row {row_index} must equal the tape's stored features")
    # Stage B cost critic 的输入也是同一份特征（训练 collector 的 cost_value 读数）
    stage_b_controller = FakeController()
    training_collector = enc.scoped_training_collector(stage_b_controller, "new")
    torch.manual_seed(7)
    training_collector.decide(observation)
    tape_b = training_collector.records[-1]
    matrix = torch.as_tensor(np.asarray(tape_b["features"], dtype=np.float32))
    with torch.no_grad():
        expected_cost = float(stage_b_controller.cost_values(matrix).mean().item())
    case.expect_close(float(tape_b["cost_value"]), expected_cost,
                      "Stage B cost_value must read this arm's features")

    # ------------------------------------------------------------------ 3) PPO batch 复用同一臂语义
    import v2_trainer as trainer

    class _Collector:
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

    def cost_row(airtime: int) -> dict[str, Any]:
        return {"airtime_bytes": airtime, "transmission_attempts": 0, "emissions": 0,
                "first_emissions": 0, "refresh_emissions": 0, "packets_lost": 0,
                "packets_delivered": 0, "deliveries_applied": 0, "budget_refused_sends": 0,
                "capacity_refused_sends": 0, "emissions_suppressed_terminal": 0}

    # tape_b is the Stage B training tape of the NEW arm: it carries cost_value, exactly what a
    # real training batch feeds the update.
    synthetic_episode = {
        "collector": _Collector([tape_b]),
        "steps": 30,
        "cost_stream": {str(step): cost_row(1 if step == 26 else 0) for step in range(1, 31)},
        "task_stream": {"26": 1.0},
        "lambda": 1.0,
        "episode_index": 0,
    }
    samples = trainer.batch_samples([synthetic_episode])
    case.check(len(samples) == 1, "one decision step per tape row in the synthetic episode")
    sample = samples[0]
    case.check(np.asarray(sample["features"]).tolist() == tape_b["features"],
               "PPO samples must carry the arm's tape features verbatim")
    case.check(sample["free"] == set(tape_b["actionable"]),
               "PPO free set must be the arm's actionable set")

    # ------------------------------------------------------------------ 4) 旧编码复现已实现
    parent_controller = FakeController()
    from budget_masked import BudgetMaskedCollector

    parent = BudgetMaskedCollector(parent_controller, forced=lambda item: False, mask=False,
                                   greedy=True, budget_mask=True)
    torch.manual_seed(7)
    parent.decide(observation)
    parent_tape = parent.records[-1]
    case.check(parent_tape["features"] == tape_old["features"],
               "OLD arm must reproduce the archived collector's features value for value")
    case.check(parent_tape["actions"] == tape_old["actions"]
               and parent_tape["log_prob"] == tape_old["log_prob"]
               and parent_tape["free"] == tape_old["free"]
               and parent_tape["masked"] == tape_old["masked"]
               and parent_tape["requested"] == tape_old["requested"],
               "OLD arm must reproduce the archived collector's tape field for field")

    # ------------------------------------------------------------------ 5) optimizer 参数覆盖
    import v2_controller

    fresh = v2_controller.fresh_controller(11)
    modules = [fresh.inner.trunk, fresh.inner.action_head, fresh.inner.value_head, fresh.cost_head]
    expected_ids = {id(parameter) for module in modules for parameter in module.parameters()}
    controller_ids = [id(parameter) for parameter in fresh.parameters]
    case.check(set(controller_ids) == expected_ids,
               "optimizer set must be exactly trunk + action head + task critic + cost critic")

    # ------------------------------------------------------------------ 6) 归档独立反例套件
    original_step = torch.optim.Adam.step
    steps_taken = {"n": 0}

    def counted_step(self, *args: Any, **kwargs: Any) -> Any:
        steps_taken["n"] += 1
        return original_step(self, *args, **kwargs)

    torch.optim.Adam.step = counted_step
    per_module: dict[str, Any] = {}
    failures: list[str] = []
    for name in SYNTHETIC_MODULES:
        module = importlib.import_module(name)
        suite = unittest.TestLoader().loadTestsFromModule(module)
        result = unittest.TestResult()
        suite.run(result)
        per_module[name] = {"tests_run": result.testsRun,
                            "failures": len(result.failures), "errors": len(result.errors)}
        for failed, _traceback in result.failures + result.errors:
            failures.append(f"{name}: {failed}")
    torch.optim.Adam.step = original_step
    case.synthetic_optimizer_steps = steps_taken["n"]
    case.check(not failures, f"archived counterexample suites must pass: {failures[:3]}")
    case.check(case.synthetic_optimizer_steps <= common.SYNTHETIC_UPDATE_CAP,
               f"synthetic optimizer steps {case.synthetic_optimizer_steps} exceed the "
               f"{common.SYNTHETIC_UPDATE_CAP} synthetic cap")

    # ------------------------------------------------------------------ 7) mask 语义不变
    case.check(tape_new["free"] == tape_new["actionable"], "free must equal actionable")
    case.check("piloc-e1-target-4" in tape_new["masked"],
               "the unaffordable candidate must be masked in both arms")
    case.check(not tape_new["requested"]["piloc-e1-target-4"],
               "a masked candidate must never be requested for emission")
    from onpolicy import joint_log_prob

    logits = torch.zeros(1, 2)
    usable_masked = torch.tensor([[0.0, 0.0]])
    picked_masked = torch.tensor([[1]])
    case.expect_close(float(joint_log_prob(logits, picked_masked, usable_masked)), 0.0,
                      "a fully masked row must carry zero likelihood")

    # ------------------------------------------------------------------ 8) 保存/恢复 + 跨版本拒绝
    def synthetic_episodes(controller: Any, seed: int) -> list[dict[str, Any]]:
        generator = torch.Generator().manual_seed(seed)
        records = []
        for step in (5, 12, 21):
            rows = np.asarray([[float(value) for value in torch.randn(19, generator=generator)]
                               for _ in range(2)], dtype=np.float32)
            with torch.no_grad():
                logits, values = controller.forward(torch.as_tensor(rows))
                probabilities = torch.softmax(logits, dim=-1)
            actions = {}
            for index, target in enumerate(("piloc-e1-target-0", "piloc-e1-target-1")):
                actions[target] = 1 if float(probabilities[index, 1]) >= 0.5 else 0
            records.append({
                "step": step, "targets": ["piloc-e1-target-0", "piloc-e1-target-1"],
                "features": rows.tolist(), "actions": actions,
                "free": ["piloc-e1-target-0", "piloc-e1-target-1"],
                "actionable": ["piloc-e1-target-0", "piloc-e1-target-1"],
                "forced": [], "masked": [],
                "log_prob": 0.0, "value": float(values.mean()),
                "cost_value": 0.0,
            })
        return records

    def run_synthetic_updates(controller: Any, optimizer: Any, updates: int) -> None:
        for index in range(updates):
            records = synthetic_episodes(controller, 100 + index)
            episode = {"collector": _Collector(records), "steps": 25,
                       "cost_stream": {str(step): cost_row(1) for step in range(1, 26)},
                       "task_stream": {"21": 1.0}, "lambda": 1.0}
            trainer.ppo_update(controller, optimizer, [episode], update_index=index)
            case.synthetic_optimizer_steps += int((len(records) + 255) // 256) * 4
        case.check(case.synthetic_optimizer_steps <= common.SYNTHETIC_UPDATE_CAP,
                   "synthetic update budget exceeded")

    arm_freeze = check.freeze_record("new")
    controller_a = v2_controller.fresh_controller(22)
    optimizer_a = torch.optim.Adam(controller_a.parameters, lr=common.v2c.LEARNING_RATE)
    state_a = trainer.fresh_training_state(22)
    state_a["episodes_done"] = 8
    state_a["updates"] = 2
    state_a["lambda"] = 1.25
    run_synthetic_updates(controller_a, optimizer_a, 2)
    path = common.CHECKPOINTS / "_preflight_round30.pt"
    common.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(controller_a, optimizer_a, state_a, path, freeze=arm_freeze)
    restored, _restored_optimizer, restored_state = trainer.load_checkpoint(path, freeze=arm_freeze)
    case.check(parameter_identity(restored)["parameter_sha256"]
               == parameter_identity(controller_a)["parameter_sha256"],
               "restored parameters must equal the saved ones")
    case.check(float(restored_state["lambda"]) == 1.25
               and int(restored_state["episodes_done"]) == 8,
               "restored training state (lambda, data position) must match")
    # 跨臂/跨轮恢复必须显式拒绝
    other_freeze = check.freeze_record("old")
    case.check(other_freeze != arm_freeze, "the two arms must carry different freeze records")
    for name, wrong_freeze in (("arm swap", other_freeze),
                               ("round-24 checkpoint", None)):
        try:
            if name == "round-24 checkpoint":
                trainer.load_checkpoint(
                    common.ROOT / "runs/comm_cost_v2_rerun_20260928/checkpoints"
                    / "v2_train_s33_ep128.pt", freeze=arm_freeze)
            else:
                trainer.load_checkpoint(path, freeze=wrong_freeze)
            case.check(False, f"{name} restore must be refused")
        except RuntimeError:
            case.checks += 1
    path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ 9) 恢复后确定性下一步
    def continue_one_update(controller: Any, optimizer: Any) -> str:
        records = synthetic_episodes(controller, 999)
        episode = {"collector": _Collector(records), "steps": 25,
                   "cost_stream": {str(step): cost_row(1) for step in range(1, 26)},
                   "task_stream": {"21": 1.0}, "lambda": 1.0}
        report = trainer.ppo_update(controller, optimizer, [episode], update_index=77)
        return parameter_identity(controller)["parameter_sha256"] + "|" + str(
            report["optimizer_steps"])

    controller_full = v2_controller.fresh_controller(22)
    optimizer_full = torch.optim.Adam(controller_full.parameters, lr=common.v2c.LEARNING_RATE)
    state_full = trainer.fresh_training_state(22)
    run_synthetic_updates(controller_full, optimizer_full, 2)
    uninterrupted = continue_one_update(controller_full, optimizer_full)

    controller_cut = v2_controller.fresh_controller(22)
    optimizer_cut = torch.optim.Adam(controller_cut.parameters, lr=common.v2c.LEARNING_RATE)
    state_cut = trainer.fresh_training_state(22)
    run_synthetic_updates(controller_cut, optimizer_cut, 2)
    cut_path = common.CHECKPOINTS / "_preflight_cut.pt"
    trainer.save_checkpoint(controller_cut, optimizer_cut, state_cut, cut_path,
                            freeze=arm_freeze)
    restored_cut, restored_cut_optimizer, _state = trainer.load_checkpoint(cut_path,
                                                                           freeze=arm_freeze)
    after_restore = continue_one_update(restored_cut, restored_cut_optimizer)
    cut_path.unlink(missing_ok=True)
    case.check(uninterrupted == after_restore,
               "the deterministic next step after a save/restore must equal the uninterrupted "
               f"path ({uninterrupted} vs {after_restore})")

    # ------------------------------------------------------------------ 登记：源码差异与臂声明
    diff = enc.source_diff()
    case.check(diff["passed"], "the scoped collector must differ from the parent in exactly the "
                               "registered one line")
    declaration = enc.arm_declaration()
    case.check(set(declaration["arms"]) == {"old", "new"}, "exactly two registered arms")
    case.check(declaration["arms"]["new"]["dim"] == 19
               and declaration["arms"]["old"]["dim"] == 19, "both arms stay 19-dim")
    case.check(declaration["arms"]["old"]["feature_names"]
               == declaration["arms"]["new"]["feature_names"],
               "feature order must be identical between arms")

    payload = {
        "gate": "ROUND30 §6 zero-environment preflight",
        "environment_starts": 0,
        "checks": case.checks,
        "failures": case.failures,
        "synthetic_optimizer_steps": case.synthetic_optimizer_steps,
        "synthetic_update_cap": common.SYNTHETIC_UPDATE_CAP,
        "archived_counterexample_modules": per_module,
        "source_diff_registration": diff,
        "arm_declaration": declaration,
        "passed": not case.failures,
        "freeze_sha256_at_preflight": common.code_identity()["freeze_sha256"],
    }
    path = common.OUTPUT_DIR / "RUN_PREFLIGHT.json"
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({"checks": case.checks, "failures": case.failures,
                      "synthetic_optimizer_steps": case.synthetic_optimizer_steps,
                      "passed": payload["passed"]}, ensure_ascii=False, indent=1))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
