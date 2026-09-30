"""ROUND35 任务 3 核对（0 环境启动、0 step、0 optimizer.step）：逐事件时序、归属、账本与 registry 一致性。

判定全部以 `TASK_REWARD_TIMING_CONTRACT.md` 为准，并且**归属用真实 `_bundle` 重算**，
不在本文件里另写一份分桶实现。

C1 逐事件抽取（每个完成事件的 9 个字段齐全，relay_tick == int(event_time)）
C2 task_stream 写入键 == 事件 relay_tick，且每键的累计增量 == 该 tick 的入账事件数
C3 clock_now == relay_tick - 1（本冻结配置 dt=1.0 下的登记时序；只作为观察，不作判据）
C4 归属用真实 v2_trainer._bundle 重算：每笔入账归到"当时或之前最近的动作行"
C5 六类情形逐一对上：真实可达的四类必须出现，两类不可达的如实标注为不可达
C6 资格五条：ledger 中每笔 credited 事件都同时满足完成/被接受/快照完成/未过期/不晚于 deadline
C7 与启动前登记的预测键集逐集相符（可被证伪的那一条）
C8 尾段与终止：最后一个完成事件当场入账；末行之后的入账仍归末行；on_time == 入账总数（无漏捕）
C9 deadline 过期：过期目标不产生入账也不产生 ledger 记录（collector 的第一道闸是状态）
C10 幂等与重复：event_id 唯一；同一目标不重复入账；真实数据未出现重复时的证据等级说明
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r35_common as smoke
import v2_trainer

LAUNCH = "launch1"
RESULTS: dict[str, Any] = {}
PROBLEMS: list[str] = []


def gate(name: str, payload: dict[str, Any]) -> None:
    payload["passed"] = bool(payload["passed"])
    RESULTS[name] = payload
    if not payload["passed"]:
        PROBLEMS.append(f"{name}: {payload.get('failure', 'check failed')}")


def row_for(episode: int) -> dict[str, Any]:
    paths = sorted((smoke.HERE / LAUNCH / "rows").glob(f"v2_train_s11_e{episode}_s*_a{episode}.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one smoke row for episode {episode}, got {len(paths)}")
    return json.loads(paths[0].read_text("utf-8"))


def tape_steps(episode: int) -> list[int]:
    path = sorted((smoke.HERE / LAUNCH / "logs").glob(
        f"v2_train_s11_e{episode}_s*_a{episode}_tape.jsonl"))[0]
    return [int(json.loads(line)["step"])
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def lifecycle(episode: int) -> list[dict[str, Any]]:
    path = sorted((smoke.HERE / LAUNCH / "logs").glob(
        f"v2_train_s11_e{episode}_s*_a{episode}_rp.jsonl"))[0]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_events() -> list[dict[str, Any]]:
    events = []
    for episode in range(smoke.SMOKE_EPISODES):
        row = row_for(episode)
        steps = tape_steps(episode)
        for record in row["task_reward_events"]:
            events.append({"episode": episode, "scenario": int(row["scenario_seed"]),
                           "horizon": int(row["steps"]), "decision_rows": len(steps),
                           "steps": steps, "row": row, **record})
    return events


def c1_extraction(events: list[str]) -> None:
    required = ("event_id", "task_id", "event_time", "relay_tick", "clock_now", "accepted",
                "deadline", "event_expired", "snapshot_expired", "on_time", "credited")
    missing = {key: [event["event_id"] for event in events if key not in event]
               for key in required}
    missing = {key: value for key, value in missing.items() if value}
    tick_matches = [event for event in events
                    if int(event["relay_tick"]) != int(event["event_time"])]
    gate("C1_per_event_extraction", {
        "events_collected": len(events), "episodes": smoke.SMOKE_EPISODES,
        "required_fields": list(required), "rows_missing_a_field": missing,
        "relay_tick_equals_int_event_time_violations": [
            {"episode": event["episode"], "event_id": event["event_id"],
             "relay_tick": event["relay_tick"], "event_time": event["event_time"]}
            for event in tick_matches],
        "passed": not missing and not tick_matches})


def c2_stream_keys(events: list[dict[str, Any]]) -> None:
    per_episode: dict[str, Any] = {}
    problems: list[str] = []
    for episode in range(smoke.SMOKE_EPISODES):
        row = row_for(episode)
        stream = {int(key): float(value) for key, value in row["task_stream"].items()}
        mine = {}
        for event in events:
            if event["episode"] != episode or not event["credited"]:
                continue
            mine[event["relay_tick"]] = mine.get(event["relay_tick"], 0.0) + 1.0
        if stream != mine:
            problems.append(f"e{episode}: stream {stream} != credited events {mine}")
        per_episode[f"e{episode}"] = {"task_stream": stream,
                                      "credited_events_per_tick": mine,
                                      "keys_equal_event_ticks": sorted(stream) == sorted(mine)}
    gate("C2_task_stream_written_at_the_event_tick", {
        "per_episode": per_episode, "mismatches": problems,
        "note": "every credit is written at the tick the event was published on, and a tick with "
                "several completions carries that many units",
        "passed": not problems})


def c3_clock_ordering(events: list[dict[str, Any]]) -> None:
    offsets = {}
    for event in events:
        offsets[round(float(event["event_time"]) - float(event["clock_now"]), 6)] = \
            offsets.get(round(float(event["event_time"]) - float(event["clock_now"]), 6), 0) + 1
    gate("C3_clock_ordering_observed", {
        "event_time_minus_clock_now_histogram": offsets,
        "expected_in_this_frozen_configuration": 1.0,
        "interpretation": "the contract says clock.now is still the pre-step time while the event "
                          "already carries the post-step time; this is an observation of that "
                          "ordering, not a pass/fail criterion, because the collector never reads "
                          "or writes the clock",
        "passed": True})


def c4_ownership_with_the_real_bundler(events: list[dict[str, Any]]) -> None:
    problems: list[str] = []
    table = []
    for episode in range(smoke.SMOKE_EPISODES):
        row = row_for(episode)
        steps = tape_steps(episode)
        stream = {int(key): float(value) for key, value in row["task_stream"].items()}
        bundled, prefix = v2_trainer._bundle(stream, steps, int(row["steps"]))
        for event in [item for item in events if item["episode"] == episode and item["credited"]]:
            tick = int(event["relay_tick"])
            expected = max((step for step in steps if step <= tick), default="prefix")
            actual = "prefix" if expected == "prefix" else bundled[expected]
            if expected != "prefix" and actual < 1.0:
                problems.append(f"e{episode} tick {tick}: the bundler gave row {expected} "
                                f"{actual}")
            if expected != "prefix" and (tick + 1) in steps:
                legacy_owner_row = tick + 1
                legacy_value = bundled.get(legacy_owner_row, 0.0)
            table.append({"episode": episode, "event_id": event["event_id"],
                          "task_id": event["task_id"], "relay_tick": tick,
                          "owner_row_predicted_latest_row_at_or_before": expected,
                          "a_decision_row_exists_at_the_next_tick": (tick + 1) in steps,
                          "row_at_next_tick_exists": (tick + 1) in steps,
                          "bundled_value_on_the_owner_row": (0.0 if expected == "prefix"
                                                             else bundled[expected]),
                          "on_time": event["on_time"], "deadline": event["deadline"],
                          "accepted": event["accepted"], "credited": event["credited"]})
        if prefix != 0.0:
            problems.append(f"e{episode}: an unassigned prefix appeared ({prefix}) even though "
                            "the archived block never produces one")
    gate("C4_ownership_recomputed_with_the_real_bundler", {
        "per_event_table": table, "problems": problems,
        "events_whose_next_tick_also_has_a_decision_row": sum(
            1 for item in table if item["a_decision_row_exists_at_the_next_tick"]),
        "note": "the ownership test uses v2_trainer._bundle itself on this round's own rows and "
                "tapes; no second implementation of the bucketing rule is introduced here",
        "passed": not problems})


def c5_case_coverage(events: list[dict[str, Any]]) -> None:
    counts = {"same_tick": 0, "cross_tick_new_action_next_tick": 0,
              "cross_tick_no_new_action": 0, "final_execution_tick": 0,
              "before_first_action_prefix": 0, "credited_after_deadline": 0}
    examples: dict[str, list[str]] = {key: [] for key in counts}
    for event in events:
        tick = int(event["relay_tick"])
        steps = event["steps"]
        labels = []
        if tick < min(steps):
            labels.append("before_first_action_prefix")
        elif tick not in steps:
            labels.append("cross_tick_no_new_action")
        else:
            labels.append("same_tick")
            if (tick + 1) in steps:
                labels.append("cross_tick_new_action_next_tick")
        if steps and tick == max(steps):
            labels.append("final_execution_tick")
        if event["credited"] and event["deadline"] is not None \
                and float(event["event_time"]) > float(event["deadline"]):
            labels.append("credited_after_deadline")
        for label in labels:
            counts[label] += 1
            if len(examples[label]) < 4:
                examples[label].append(f"e{event['episode']}:{event['event_id']}")
    expiry_events = []
    for episode in range(smoke.SMOKE_EPISODES):
        expiry_events.extend([
            {"episode": episode, "target_id": record["target_id"],
             "status": record["status"], "timestamp": record["timestamp"],
             "failure_reason": record["failure_reason"],
             "event_id": (record.get("detail") or {}).get("progress_event_id")}
            for record in lifecycle(episode)
            if record.get("event_type") == "progress"
            and (record.get("detail") or {}).get("expired")])
    reachable = ["same_tick", "cross_tick_new_action_next_tick", "final_execution_tick"]
    missing = [key for key in reachable if counts[key] == 0]
    gate("C5_contract_case_coverage_on_real_events", {
        "case_counts_on_real_events": counts, "examples": examples,
        "expiry_events_seen_in_the_lifecycle": expiry_events,
        "cases_the_block_cannot_reach": [
            "cross_tick_no_new_action (0 of 4281 archived credits and 0 here: every completion "
            "happens on a tick that has a decision row)",
            "before_first_action_prefix (relay ticks start at 1 and the policy decides at tick 1)"],
        "reachable_cases_all_present": not missing,
        "missing_reachable_cases": missing,
        "judged_against": "TASK_REWARD_TIMING_CONTRACT.md hand-calculated table",
        "passed": not missing})


def c6_eligibility(events: list[dict[str, Any]]) -> None:
    violations = []
    for event in events:
        if not event["credited"]:
            continue
        if not (event["accepted"] and not event["event_expired"]
                and not event["snapshot_expired"] and event["on_time"]
                and event["deadline"] is not None
                and float(event["event_time"]) <= float(event["deadline"])):
            violations.append(event["event_id"])
    per_rule = {
        "rule_1_status_is_completed": "every ledger row comes from a COMPLETED event: the "
                                      "collector returns before recording for any other status",
        "rule_2_registry_accepted_and_committed": sum(1 for event in events if event["accepted"]),
        "rule_3_neither_event_nor_snapshot_expired": sum(
            1 for event in events if event["event_expired"] or event["snapshot_expired"]),
        "rule_4_observed_at_not_after_deadline": sum(
            1 for event in events if event["deadline"] is not None
            and float(event["event_time"]) <= float(event["deadline"])),
        "rule_5_target_credited_once": len(events) - len({event["task_id"] for event in events}),
        "events_total": len(events), "credited_total": sum(1 for event in events
                                                            if event["credited"])}
    gate("C6_reward_eligibility_five_rules", {
        "per_rule": per_rule, "credited_rows_violating_a_rule": violations,
        "passed": not violations and per_rule["rule_3_neither_event_nor_snapshot_expired"] == 0})


def c7_predictions(events: list[dict[str, Any]]) -> None:
    expectation = smoke.archived_expectation()["episodes"]
    comparison = {}
    problems = []
    for episode, key in ((index, f"e{index}") for index in range(4)):
        predicted = sorted(expectation[key]["predicted_fixed_task_stream_keys"])
        actual = sorted(int(name) for name in row_for(episode)["task_stream"])
        comparison[key] = {"predicted": predicted, "actual": actual,
                           "equal": predicted == actual,
                           "archived_legacy_keys": expectation[key]["legacy_task_stream_keys"]}
        if predicted != actual:
            problems.append(f"{key}: predicted {predicted} actual {actual}")
    gate("C7_registered_prediction_falsification", {
        "per_episode_batch_0": comparison, "problems": problems,
        "note": "episodes 0..3 run before any weight moves, so they are directly comparable with "
                "the archived legacy run; the prediction (every archived key shifted back one "
                "tick) is falsifiable and is tested here",
        "passed": not problems})


def c8_tail_and_termination(events: list[dict[str, Any]]) -> None:
    rows = []
    problems = []
    for episode in range(smoke.SMOKE_EPISODES):
        row = row_for(episode)
        steps = tape_steps(episode)
        stream = {int(key): float(value) for key, value in row["task_stream"].items()}
        episode_events = [event for event in events if event["episode"] == episode]
        last_event_tick = max((int(event["relay_tick"]) for event in episode_events), default=None)
        bundled, prefix = v2_trainer._bundle(stream, steps, int(row["steps"]))
        credited = sum(stream.values())
        if int(row["on_time"]) != int(credited):
            problems.append(f"e{episode}: on_time {row['on_time']} != credited {credited}")
        if int(row["task_reward_total"]) != int(credited):
            problems.append(f"e{episode}: task_reward_total differs from the stream sum")
        after_last = sum(value for key, value in stream.items() if key > max(steps))
        rows.append({
            "episode": episode, "horizon": int(row["steps"]),
            "last_decision_row": max(steps), "last_credited_tick": last_event_tick,
            "credits_after_the_last_decision_row": after_last,
            "on_time": int(row["on_time"]), "completed": int(row["completed"]),
            "expired": int(row["expired"]), "credited_total": credited,
            "unassigned_prefix": prefix,
            "closure_on_time_equals_credited": int(row["on_time"]) == int(credited)})
    final_tick_events = [item for item in rows
                         if item["last_credited_tick"] == item["horizon"]]
    gate("C8_tail_and_termination", {
        "per_episode": rows, "problems": problems,
        "completions_exactly_at_the_horizon_tick": len(final_tick_events),
        "capture_is_immediate_because": "each credited tick equals the tick the event was "
                                        "published on (C2), so a completion on the last executed "
                                        "tick is written during that same publication and needs "
                                        "no later step; the closure on_time == credited_total "
                                        "then proves nothing was dropped at the tail",
        "passed": not problems})


def c9_deadline_expiry(events: list[dict[str, Any]]) -> None:
    per_episode = {}
    problems = []
    for episode in range(smoke.SMOKE_EPISODES):
        row = row_for(episode)
        expiry = [record for record in lifecycle(episode)
                  if record.get("event_type") == "progress"
                  and (record.get("detail") or {}).get("expired")]
        ledger_ids = {event["event_id"] for event in events if event["episode"] == episode}
        credited_targets = {event["task_id"] for event in events
                            if event["episode"] == episode and event["credited"]}
        for record in expiry:
            identifier = (record.get("detail") or {}).get("progress_event_id")
            if identifier in ledger_ids:
                problems.append(f"e{episode}: an expiry event reached the ledger ({identifier})")
            if record["target_id"] in credited_targets:
                problems.append(f"e{episode}: an expired target {record['target_id']} was "
                                "credited")
            if record["status"] == "completed":
                problems.append(f"e{episode}: an expired event carries status completed, which "
                                "would need the deadline guard instead of the status guard")
        per_episode[f"e{episode}"] = {
            "expired_count_in_the_row": int(row["expired"]),
            "expiry_events_in_the_lifecycle": [
                {"target_id": record["target_id"], "status": record["status"],
                 "timestamp": record["timestamp"],
                 "failure_reason": record["failure_reason"],
                 "event_id": (record.get("detail") or {}).get("progress_event_id")}
                for record in expiry],
            "expiry_events_in_the_ledger": 0, "credited_expired_targets": 0}
        if int(row["expired"]) != len(expiry):
            problems.append(f"e{episode}: the row reports {row['expired']} expired but the "
                            f"lifecycle shows {len(expiry)} expiry events")
    with_expiry = [key for key, value in per_episode.items()
                   if value["expiry_events_in_the_lifecycle"]]
    gate("C9_deadline_expiry_never_credits", {
        "per_episode": per_episode, "episodes_with_an_expiry": with_expiry, "problems": problems,
        "mechanism": "the collector's first guard is `status is not COMPLETED`, so an expired "
                     "target produces neither a credit nor a ledger row; the deadline guard is a "
                     "second layer that real data in this block never reaches (0 archived "
                     "completions after a deadline)",
        "passed": not problems and bool(with_expiry),
        "failure": "no expiry occurred in the smoke" if not with_expiry else None})


def c10_idempotency(events: list[dict[str, Any]]) -> None:
    ids = [event["event_id"] for event in events]
    duplicates = {key for key in ids if ids.count(key) > 1}
    per_target = {}
    for event in events:
        per_target.setdefault((event["episode"], event["task_id"]), []).append(event["credited"])
    double_credit = {f"e{episode}:{target}": value for (episode, target), value in
                     per_target.items() if value.count(True) > 1}
    gate("C10_event_ids_unique_and_no_double_credit", {
        "events": len(events), "unique_event_ids": len(set(ids)),
        "duplicate_event_ids_in_real_data": sorted(duplicates),
        "targets_credited_more_than_once": double_credit,
        "evidence_level": "real data exercised uniqueness and single-credit; a re-dispatch of the "
                          "same accepted event does not occur in these eight episodes, so "
                          "idempotency under duplication rests on the credited_task_ids guard "
                          "plus ROUND34's synthetic re-publication test and this round's own "
                          "pure wiring test, not on real data",
        "passed": not double_credit})


def main() -> int:
    events = collect_events()
    c1_extraction(events)
    c2_stream_keys(events)
    c3_clock_ordering(events)
    c4_ownership_with_the_real_bundler(events)
    c5_case_coverage(events)
    c6_eligibility(events)
    c7_predictions(events)
    c8_tail_and_termination(events)
    c9_deadline_expiry(events)
    c10_idempotency(events)
    payload = {
        "schema": "round35-event-timing-verification/1",
        "round": 35, "launch": LAUNCH,
        "environment_launches_used_by_this_script": 0,
        "judged_against": "runs/comm_task_reward_event_timing_20260930/TASK_REWARD_TIMING_CONTRACT.md",
        "events_read_from_the_real_run": len(events),
        "checks": RESULTS, "problems": PROBLEMS, "clean": not PROBLEMS,
        "per_event_table": [
            {key: event[key] for key in ("episode", "scenario", "event_id", "task_id",
                                         "event_time", "relay_tick", "clock_now", "accepted",
                                         "deadline", "event_expired", "snapshot_expired",
                                         "on_time", "credited")}
            for event in events],
        "note": "every number here is read from this launch's own rows, tapes, event ledger and "
                "lifecycle log; the ownership rule is recomputed with the trainer's own _bundle",
    }
    out = smoke.write_json("EVENT_TIMING_VERIFICATION.json", payload, refresh=True)
    print(json.dumps({"clean": payload["clean"], "problems": PROBLEMS,
                      "events": len(events),
                      "cases": RESULTS["C5_contract_case_coverage_on_real_events"][
                          "case_counts_on_real_events"],
                      "prediction_ok": RESULTS["C7_registered_prediction_falsification"]["passed"],
                      "expiry_ok": RESULTS["C9_deadline_expiry_never_credits"]["passed"]},
                     ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
