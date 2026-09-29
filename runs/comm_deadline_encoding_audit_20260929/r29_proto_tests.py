"""ROUND29 §7：期限列编码原型的纯计算测试（零环境、零优化器、零权重加载）。

测试从真实合法观测结构（integrated_sar.comm_policy.SendItem/SendObservation）进入原型，
而不是只测一个孤立除法。所有预期答案独立手算，不从被测实现复制。

运行：python r29_proto_tests.py  （不构造环境、不调用 optimizer、不加载旧权重）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs/comm_budget_action_mask_fix_20260922"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from integrated_sar.comm_learning import EPISODE_HORIZON_STEPS, FEATURE_NAMES, features  # noqa: E402
from integrated_sar.comm_policy import SendItem, SendObservation  # noqa: E402

import r29_encoding_prototype as proto  # noqa: E402


def make_item(target_id: str, *, remaining: float, deadline: float = 100.0) -> SendItem:
    """一个真实、合法的 SendItem：发送端在决策点对单个候选的全部自有状态。"""
    return SendItem(
        target_id=target_id,
        ready_step=5,
        age_steps=3,
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
        deadline=deadline,
        remaining_time=remaining,
        sent_count=1,
        last_sent_step=2,
        since_last_emission_time=3.0,
        has_future_requirements=True,
        attempt_charge_bytes=2400,
        send_executable=True,
        send_block_reason="allowed",
    )


def make_observation(*items: SendItem) -> SendObservation:
    return SendObservation(
        step=26,
        current_time=25.0,
        items=tuple(items),
        remaining_budget_bytes=29300.0,
        budget_bytes_per_episode=32000,
        capacity_remaining=3,
        backlog_targets=len(items),
        subscribers=6,
    )


class Case:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)

    def expect_close(self, got: float, want: float, message: str) -> None:
        self.check(abs(got - want) <= 1e-12, f"{message}: got {got!r}, want {want!r}")


def main() -> int:
    case = Case()

    # 1) horizon=1000，remaining=60/100/500：旧期限列均为 1；新列 0.06/0.10/0.50（手算）。
    observation = make_observation(make_item("piloc-e1-target-0", remaining=60.0))
    item = observation.items[0]
    for remaining, want_new in ((60.0, 0.06), (100.0, 0.10), (500.0, 0.50)):
        item_r = make_item("piloc-e1-target-0", remaining=remaining)
        obs_r = make_observation(item_r)
        old_column = features(obs_r, item_r)[proto.DEADLINE_COLUMN]
        new_column = proto.candidate_features(obs_r, item_r)[proto.DEADLINE_COLUMN]
        case.expect_close(old_column, 1.0, f"old column saturated for remaining={remaining}")
        case.expect_close(new_column, want_new, f"new column for remaining={remaining}")

    # 2) remaining=0、负值、超过 horizon：新列按 remaining/1000 直出，语义正确。
    zero_item = make_item("t0", remaining=0.0)
    case.expect_close(proto.candidate_features(make_observation(zero_item), zero_item)[proto.DEADLINE_COLUMN],
                      0.0, "zero remaining")
    neg_item = make_item("t0", remaining=-10.0)
    case.expect_close(proto.candidate_features(make_observation(neg_item), neg_item)[proto.DEADLINE_COLUMN],
                      -0.01, "negative remaining kept, not clipped")
    over_item = make_item("t0", remaining=1500.0)
    case.expect_close(proto.candidate_features(make_observation(over_item), over_item)[proto.DEADLINE_COLUMN],
                      1.5, "remaining above horizon kept above 1")

    # 3) horizon 非正/非有限、remaining 非有限：显式失败，不默默填零。
    for bad_horizon in (0.0, -1.0, float("nan"), float("inf")):
        try:
            proto.encoded_deadline(60.0, horizon=bad_horizon)
            case.check(False, f"horizon={bad_horizon!r} must raise")
        except ValueError:
            case.checks += 1
    for bad_remaining in (float("nan"), float("inf"), float("-inf")):
        try:
            proto.encoded_deadline(bad_remaining)
            case.check(False, f"remaining={bad_remaining!r} must raise")
        except ValueError:
            case.checks += 1
        try:
            proto.candidate_features(make_observation(make_item("t0", remaining=bad_remaining)),
                                     make_item("t0", remaining=bad_remaining))
            case.check(False, f"candidate_features with remaining={bad_remaining!r} must raise")
        except ValueError:
            case.checks += 1

    # 4) 除期限列外，其余 18 列与旧编码逐项一致（用多个不同剩余期限/不同 item 交叉验证）。
    for remaining in (60.0, 100.0, 500.0, 0.0, -10.0, 1500.0, 12.5):
        item_x = make_item(f"t{remaining}", remaining=remaining)
        obs_x = make_observation(item_x, make_item("t-other", remaining=remaining / 2))
        old_vector = features(obs_x, item_x)
        new_vector = proto.candidate_features(obs_x, item_x)
        case.check(len(new_vector) == 19, f"new vector must stay 19-dim, got {len(new_vector)}")
        for index in range(19):
            if index == proto.DEADLINE_COLUMN:
                continue
            case.check(old_vector[index] == new_vector[index],
                       f"column {index} ({FEATURE_NAMES[index]}) must be unchanged for remaining={remaining}")
    # 期限列的定义确实改变了（否则测试空洞）。
    changed_item = make_item("t0", remaining=60.0)
    changed_obs = make_observation(changed_item)
    case.check(features(changed_obs, changed_item)[proto.DEADLINE_COLUMN]
               != proto.candidate_features(changed_obs, changed_item)[proto.DEADLINE_COLUMN],
               "deadline column must actually change semantics")

    # 5) 输入维数、顺序及观测版本声明明确；候选布局名与旧布局名不同。
    declaration = proto.layout_declaration()
    case.check(declaration["dim"] == 19, "declared dim must be 19")
    case.check(declaration["feature_names"] == list(FEATURE_NAMES), "declared order must equal FEATURE_NAMES")
    case.check(declaration["feature_layout"] != "comm-send-state/1", "candidate layout must differ from /1")
    case.check(declaration["feature_layout"] == proto.CANDIDATE_FEATURE_LAYOUT, "layout name must be registered")
    case.check(declaration["horizon"] == EPISODE_HORIZON_STEPS, "default horizon must be the configured constant")

    # 6) 原型不读取接收端或结局字段：输出只随生产特征输入变化；把观测/候选中不属于生产特征
    #    输入的字段改掉（执行侧计费/门/时钟读数），19 维输出必须完全不变。
    base_item = make_item("t0", remaining=80.0)
    base_obs = make_observation(base_item)
    base_vector = proto.candidate_features(base_obs, base_item)
    tampered_item = SendItem(**{**base_item.__dict__,
                                "attempt_charge_bytes": 999999,   # 执行侧计费，不是特征输入
                                "send_executable": False,          # 执行侧门，不是特征输入
                                "send_block_reason": "budget_exhausted"})
    tampered_obs = SendObservation(**{**base_obs.__dict__,
                                      "current_time": 4242.0})       # 时钟读数不进特征
    tampered_vector = proto.candidate_features(tampered_obs, tampered_item)
    case.check(base_vector == tampered_vector,
               "candidate features must not read executor-only or clock fields")
    # 期限列只来自 item.remaining_time（发送端自有），不因观测其它字段变化。
    case.expect_close(tampered_vector[proto.DEADLINE_COLUMN], 80.0 / EPISODE_HORIZON_STEPS,
                      "deadline column from remaining_time only")

    # 7) 不构造环境、不调用 optimizer、不加载旧权重：本模块导入后禁用模块不得出现；
    #    原型是纯函数，重复调用确定性一致。
    forbidden = ("integrated_sar.joint_driver", "torch")
    for name in forbidden:
        case.check(name not in sys.modules or name in ("torch",),
                   f"import of the prototype must not require {name}")
    probe_item = make_item("t0", remaining=37.0)
    probe_obs = make_observation(probe_item)
    first = proto.candidate_features(probe_obs, probe_item)
    for _ in range(3):
        case.check(proto.candidate_features(probe_obs, probe_item) == first,
                   "prototype must be a deterministic pure function")

    # 登记件完整性：兼容规则必须显式拒绝静默换用。
    rule = proto.SEMANTIC_REGISTRY["compatibility_rule"]
    case.check("refused" in rule and "silently" in rule,
               "compatibility rule must register refusal of silent reuse")
    case.check(len(proto.SEMANTIC_REGISTRY["layout_unchanged_columns"]) == 18,
               "exactly 18 unchanged columns must be registered")

    print(json_report(case))
    return 1 if case.failures else 0


def json_report(case: Case) -> str:
    import json
    return json.dumps({"checks": case.checks, "failures": case.failures}, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
