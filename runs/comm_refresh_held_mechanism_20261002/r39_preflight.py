"""Pure ROUND39 refresh-held seam tests. No SAR environment or optimizer is constructed."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "runs" / "comm_cost_v2_rerun_20260928"),
    str(ROOT / "runs" / "comm_cost_stage_b_20260927"),
    str(ROOT / "runs" / "comm_budget_action_mask_fix_20260922"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(HERE),
]

from budget_masked import BudgetMaskedCollector, HOLD, SEND  # noqa: E402
from integrated_sar.comm import CommRelay  # noqa: E402
from integrated_sar.comm_budget import CommBudget  # noqa: E402
from integrated_sar.comm_policy import SendItem, SendObservation  # noqa: E402
from integrated_sar.contracts import (  # noqa: E402
    DetectionEvent,
    GridPosition,
    TaskSnapshot,
    TaskStageSpec,
    TaskStatus,
    TargetProfile,
    WorldPosition,
)
from r39_collector import RefreshHeldCollector  # noqa: E402


class FixedController:
    def __init__(self, action: int) -> None:
        self.action = int(action)

    def forward(self, batch: torch.Tensor):
        rows = int(batch.shape[0])
        logits = torch.zeros((rows, 2), dtype=torch.float32)
        logits[:, self.action] = 2.0
        logits[:, 1 - self.action] = 0.0
        return logits, torch.zeros((rows, 1), dtype=torch.float32)


def make_item(
    target: str,
    *,
    first: bool = True,
    charge: int = 4,
    revision: int = 1,
    send_executable: bool = True,
) -> SendItem:
    return SendItem(
        target_id=target,
        ready_step=0,
        age_steps=1,
        pending_event=first,
        changed_view=True,
        first_emission=first,
        revision=revision,
        revision_delta=revision,
        stage_index=0,
        stage_changed=False,
        progress=0.2,
        progress_delta=0.2,
        requirements_changed=False,
        members_changed=False,
        deadline=300.0,
        remaining_time=250.0,
        sent_count=0 if first else 1,
        last_sent_step=None if first else 1,
        since_last_emission_time=None if first else 1.0,
        has_future_requirements=False,
        attempt_charge_bytes=int(charge),
        send_executable=bool(send_executable),
        send_block_reason="allowed" if send_executable else "budget_exhausted",
    )


def observation(items: list[SendItem], credit: int, *, step: int = 3) -> SendObservation:
    return SendObservation(
        step=step,
        current_time=float(step),
        items=tuple(items),
        remaining_budget_bytes=float(credit),
        budget_bytes_per_episode=credit,
        capacity_remaining=None,
        backlog_targets=len(items),
        subscribers=1,
    )


def collector(controller: FixedController, *, enabled: bool = True) -> RefreshHeldCollector:
    return RefreshHeldCollector(
        controller,
        forced=lambda _item: False,
        mask=False,
        refresh_held=enabled,
    )


def relay_snapshot(revision: int = 0) -> TaskSnapshot:
    position = WorldPosition(0.25, 0.5)
    stages = (TaskStageSpec((1, 0), 10.0),)
    return TaskSnapshot(
        task_id="target-relay",
        source_event_id="event-relay",
        position=position,
        stages=stages,
        current_stage=0,
        remaining_requirements=(1, 0),
        progress=0.0 if revision == 0 else 0.1,
        status=TaskStatus.DETECTED,
        deadline=500.0,
        confidence=1.0,
        severity=0.5,
        discovered_at=0.0,
        updated_at=float(revision),
        revision=revision,
        members=(),
    )


def relay_event() -> DetectionEvent:
    return DetectionEvent(
        event_id="event-relay",
        target_id="target-relay",
        grid_position=GridPosition(1, 1),
        world_position=WorldPosition(0.25, 0.5),
        detected_at=0.0,
        discovered_by="searcher-0",
        confidence=1.0,
        profile=TargetProfile(
            stages=(TaskStageSpec((1, 0), 10.0),),
            deadline_offset=500.0,
            severity=0.5,
        ),
    )


def test_a_disabled_is_base_identity() -> dict[str, Any]:
    obs = observation([make_item("first-a"), make_item("first-b")], 6)
    baseline = BudgetMaskedCollector(
        FixedController(SEND), forced=lambda _item: False, mask=False,
        greedy=True, budget_mask=True,
    )
    wrapper = collector(FixedController(SEND), enabled=False)
    rng_before = torch.get_rng_state().clone()
    expected = baseline.choose(obs)
    actual = wrapper.choose(obs)
    rng_after = torch.get_rng_state().clone()
    assert actual == expected
    assert torch.equal(rng_before, rng_after), "collector changed Torch RNG state"
    return {"same_return_tuple_and_tape": True, "torch_rng_unchanged": True,
            "asked": expected[0], "actions": expected[1]}


def test_b_first_emission_unmodified() -> dict[str, Any]:
    out = {}
    for action in (HOLD, SEND):
        result = collector(FixedController(action)).choose(
            observation([make_item("first")], 10)
        )
        tape = result[-1]
        assert not tape["intervened"]
        assert tape["effective_actions"]["first"] == action
        assert tape["raw_model_actions"]["first"] == action
        out["SEND" if action == SEND else "HOLD"] = tape["effective_actions"]["first"]
    return {"first_report_actions_preserved": out}


def test_c_refresh_holds_without_shadow_charge() -> dict[str, Any]:
    obs = observation([make_item("already-sent", first=False, charge=4),
                       make_item("new-target", first=True, charge=4)], 6)
    result = collector(FixedController(SEND)).choose(obs)
    tape = result[-1]
    refresh = tape["candidate_budget_audit"]["already-sent"]["effective_run"]
    first = tape["candidate_budget_audit"]["new-target"]["effective_run"]
    assert tape["effective_actions"] == {"already-sent": HOLD, "new-target": SEND}
    assert refresh["budget_before"] == refresh["budget_after"] == 6.0
    assert refresh["charge_applied"] is False
    assert first["budget_before"] == 6.0 and first["budget_after"] == 2.0
    return {"refresh_budget_before_after": [6, 6], "first_report_budget_after": 2,
            "asked": result[0]}


def test_d_sequential_hand_calculation() -> dict[str, Any]:
    obs = observation([make_item("refresh", first=False, charge=4),
                       make_item("first-report", first=True, charge=4)], 6)
    baseline = BudgetMaskedCollector(
        FixedController(SEND), forced=lambda _item: False, mask=False,
        greedy=True, budget_mask=True,
    ).choose(obs)
    held = collector(FixedController(SEND)).choose(obs)
    assert baseline[1] == {"refresh": SEND, "first-report": HOLD}
    assert held[1] == {"refresh": HOLD, "first-report": SEND}
    return {"credit": 6, "quotes": [4, 4], "baseline_actions": baseline[1],
            "refresh_held_actions": held[1]}


def test_e_authoritative_relay_first_emission() -> dict[str, Any]:
    snapshots = {"target-relay": relay_snapshot(0)}
    relay = CommRelay(
        subscribers=("receiver-0",),
        writer=lambda *_args, **_kwargs: None,
        snapshot_of=lambda target: snapshots.get(target),
        delivery_probability=0.0,
        delay_ticks=0,
        seed=9,
        budget=CommBudget(bytes_per_episode=100_000),
        timing_mode="policy_owned_timing",
    )
    relay.handle_detection(relay_event())
    first_item = relay._observation().items[0]
    assert first_item.first_emission is True
    relay._emit(first_item, first=True)
    assert relay.counters["first_emissions"] == 1
    assert relay.counters["packets_lost"] == 1
    after_loss = relay._observation().items[0]
    assert after_loss.first_emission is False
    assert after_loss.sent_count == 1

    snapshots["target-relay"] = relay_snapshot(1)
    revised = relay._observation().items[0]
    assert revised.revision == 1 and revised.first_emission is False
    revised_result = collector(FixedController(SEND)).choose(
        SendObservation(
            step=relay._tick,
            current_time=0.0,
            items=(revised,),
            remaining_budget_bytes=relay._budget.remaining_bytes,
            budget_bytes_per_episode=relay._budget.bytes_per_episode,
            capacity_remaining=None,
            subscribers=1,
        )
    )
    assert revised_result[-1]["intervened"] == ["target-relay"]

    refused_snapshots = {"target-relay": relay_snapshot(0)}
    refused_relay = CommRelay(
        subscribers=("receiver-0",),
        writer=lambda *_args, **_kwargs: None,
        snapshot_of=lambda target: refused_snapshots.get(target),
        delivery_probability=0.0,
        delay_ticks=0,
        seed=9,
        budget=CommBudget(bytes_per_episode=1),
        timing_mode="policy_owned_timing",
    )
    refused_relay.handle_detection(relay_event())
    refused_first = refused_relay._observation().items[0]
    assert refused_first.first_emission is True
    assert refused_first.send_executable is False
    refused_relay._emit(refused_first, first=True)
    still_first = refused_relay._observation().items[0]
    assert still_first.first_emission is True
    refused_result = collector(FixedController(SEND)).choose(
        SendObservation(
            step=refused_relay._tick,
            current_time=0.0,
            items=(still_first,),
            remaining_budget_bytes=refused_relay._budget.remaining_bytes,
            budget_bytes_per_episode=refused_relay._budget.bytes_per_episode,
            capacity_remaining=None,
            subscribers=1,
        )
    )
    assert not refused_result[-1]["intervened"]
    assert refused_relay.counters["emissions"] == 0
    return {
        "dropped_first_emission_stays_first_sent": True,
        "new_revision_does_not_restore_first_send": True,
        "budget_refusal_does_not_count_as_first_send": True,
        "relay_cost_charged_for_lost_first_packet": relay._budget.spent_bytes,
    }


def test_f_always_hold_has_no_behavior_change() -> dict[str, Any]:
    obs = observation([make_item("first", first=True),
                       make_item("refresh", first=False)], 10)
    baseline = BudgetMaskedCollector(
        FixedController(HOLD), forced=lambda _item: False, mask=False,
        greedy=True, budget_mask=True,
    ).choose(obs)
    held = collector(FixedController(HOLD)).choose(obs)
    assert baseline[0] == held[0]
    assert baseline[1] == held[1]
    assert set(held[1].values()) == {HOLD}
    assert held[-1]["shadow_budget_after"] == baseline[-1]["shadow_budget_after"]
    assert held[-1]["raw_model_actions"]["refresh"] == HOLD
    return {"baseline_asked": baseline[0], "intervention_asked": held[0],
            "actions_equal": baseline[1] == held[1],
            "bytes_and_shadow_equal": True,
            "refresh_send_suppressed": False}


def test_g_new_episode_has_no_carried_target_state() -> dict[str, Any]:
    wrapper = collector(FixedController(SEND))
    old_episode = wrapper.choose(
        observation([make_item("same-target", first=False)], 10, step=9)
    )[-1]
    next_episode = wrapper.choose(
        observation([make_item("same-target", first=True)], 10, step=0)
    )[-1]
    assert old_episode["intervened"] == ["same-target"]
    assert next_episode["intervened"] == []
    assert next_episode["effective_actions"]["same-target"] == SEND
    return {"same_target_old_episode_held": True, "new_episode_first_send_untouched": True}


def test_h_no_free_or_duplicate_charges() -> dict[str, Any]:
    obs = observation([make_item("first-a", first=True, charge=4),
                       make_item("refresh-b", first=False, charge=4),
                       make_item("first-c", first=True, charge=4)], 8)
    result = collector(FixedController(SEND)).choose(obs)
    tape = result[-1]
    assert result[0] == ["first-a", "first-c"]
    assert tape["shadow_budget_after"] == 0.0
    charged = [
        target for target in tape["targets"]
        if tape["candidate_budget_audit"][target]["effective_run"]["charge_applied"]
    ]
    assert charged == ["first-a", "first-c"]
    assert all(
        entry["effective_run"]["budget_after"] >= 0
        for entry in tape["candidate_budget_audit"].values()
    )
    assert sum(tape["attempt_charge_bytes"][target] for target in result[0]) == 8
    try:
        collector(FixedController(SEND)).choose(
            observation([make_item("duplicate"), make_item("duplicate")], 8)
        )
    except RuntimeError as error:
        assert "duplicate target ids" in str(error)
    else:
        raise AssertionError("duplicate target id was charged twice")
    return {"charged_target_ids": charged, "charged_bytes": 8,
            "remaining_budget": 0, "duplicate_target_rejected": True}


def main() -> dict[str, Any]:
    tests = {
        "A_disabled_identity": test_a_disabled_is_base_identity(),
        "B_first_emission_preserved": test_b_first_emission_unmodified(),
        "C_refresh_hold_no_charge": test_c_refresh_holds_without_shadow_charge(),
        "D_ordered_budget_hand_case": test_d_sequential_hand_calculation(),
        "E_authoritative_relay_semantics": test_e_authoritative_relay_first_emission(),
        "F_all_hold_negative_case": test_f_always_hold_has_no_behavior_change(),
        "G_episode_state_reset": test_g_new_episode_has_no_carried_target_state(),
        "H_budget_and_charge_closure": test_h_no_free_or_duplicate_charges(),
    }
    result = {
        "schema": "round39-pure-preflight/1",
        "all_passed": True,
        "tests": tests,
        "environment_constructions": 0,
        "joint_runs": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
    }
    (HERE / "PRELAUNCH_GATE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
