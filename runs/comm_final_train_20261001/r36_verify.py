"""ROUND36 训练后复核（0 环境启动、0 optimizer.step）：版本字段、事件账本、批 0 逐字节被动性、
训练量一致性与"旧件未被改写"对账。

所有断言都读真实产物：本轮 `seed<NN>/{rows,logs,checkpoints}` 与 ROUND32/ROUND34/ROUND35 的归档
件。报告里出现的每个数字都来自这些文件，不允许手打。

    /d/project_env/python.exe r36_verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import r36_common as r36
import v2_common
from v2_reward_timing import REWARD_TIMING_VERSION, TRAINING_VERSION

TAPE_FIELDS = ("features", "actions", "targets", "actionable", "log_prob", "step")
EVENT_REQUIRED = ("event_id", "task_id", "event_time", "relay_tick", "clock_now", "accepted",
                  "deadline", "event_expired", "snapshot_expired", "on_time", "credited")
BATCH_ZERO_EPISODES = (0, 1, 2, 3)
LEGACY_PASSED_TO_ROUND35 = r36.ROUND35 / "CHECKPOINT_CONTENT_LAYER.json"


def seed_rows(seed: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((r36.HERE / f"seed{seed}" / "rows").glob("v2_train_s*_e*_s*_a*.json")):
        row = r36.read_json(path)
        rows[int(row["episode_index"])] = row
    return rows


def tape_rows(seed: int, episode: int, scenario: int) -> list[dict[str, Any]]:
    path = (r36.HERE / f"seed{seed}" / "logs" /
            f"v2_train_s{seed}_e{episode}_s{scenario}_a{episode}_tape.jsonl")
    return list(v2_common.iter_jsonl(path))


def archived_row(arm: str, seed: int, episode: int) -> dict[str, Any]:
    matches = sorted((r36.ROUND32 / "rows").glob(
        f"r32_train_{arm}_s{seed}_e{episode}_s*_a{episode}.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one archived {arm} row for s{seed} e{episode}, got {matches}")
    return r36.read_json(matches[0])


def archived_tape(arm: str, seed: int, episode: int) -> list[dict[str, Any]]:
    matches = sorted((r36.ROUND32 / "rows").glob(
        f"r32_train_{arm}_s{seed}_e{episode}_s*_a{episode}.json"))
    tape = r36.ROUND32 / "logs" / (matches[0].stem + "_tape.jsonl")
    return list(v2_common.iter_jsonl(tape))


# ------------------------------------------------------------------- V1 版本字段 -------------
def v1_version_fields(seeds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed, artefacts in seeds.items():
        checkpoints = {}
        for episodes in (0, 64, 128):
            path = r36.seed_directory(seed) / "checkpoints" / f"v2_train_s{seed}_ep{episodes}.pt"
            checkpoints[f"ep{episodes}"] = r36.checkpoint_version_fields(path)
        checkpoints["rolling"] = r36.checkpoint_version_fields(
            r36.seed_directory(seed) / "checkpoints" / f"v2_train_s{seed}.pt")
        complete = all(item["both_fields_complete"] and
                       item["training_version"] == TRAINING_VERSION and
                       item["reward_timing_version"] == REWARD_TIMING_VERSION
                       for item in checkpoints.values())
        state_matches = all(item["in_training_state_training_version"] == TRAINING_VERSION
                            and item["in_training_state_reward_timing_version"]
                            == REWARD_TIMING_VERSION for item in checkpoints.values())
        rows_with_versions = sum(1 for row in artefacts["rows"].values()
                                 if row.get("training_version") == TRAINING_VERSION
                                 and row.get("reward_timing_version") == REWARD_TIMING_VERSION)
        batches_with_updates = len(artefacts["batches"])
        per_seed[f"s{seed}"] = {
            "checkpoints": checkpoints,
            "all_four_files_carry_both_fields_in_payload_and_state": bool(complete and
                                                                          state_matches),
            "rows_carrying_both_fields": f"{rows_with_versions}/{len(artefacts['rows'])}",
            "milestone_state_fields": {
                name: {key: item.get(key) for key in
                       ("episodes_done", "updates", "optimizer_steps", "lambda")}
                for name, item in checkpoints.items()},
            "batch_records": batches_with_updates}
    return {"per_seed": per_seed,
            "required": {"training_version": TRAINING_VERSION,
                         "reward_timing_version": REWARD_TIMING_VERSION},
            "passed": all(item["all_four_files_carry_both_fields_in_payload_and_state"]
                          and item["rows_carrying_both_fields"].split("/")[0]
                          == item["rows_carrying_both_fields"].split("/")[1]
                          for item in per_seed.values())}


# ------------------------------------------------------------------- V2 事件账本 -------------
def v2_event_ledgers(seeds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Per-episode integrity of the event ledger, recomputed from the rows this round wrote."""
    per_seed: dict[str, Any] = {}
    cases = {"credit_at_a_tick_that_has_a_decision_row": 0,
             "credit_at_a_tick_with_no_decision_row": 0,
             "credit_owned_by_the_final_decision_row": 0,
             "credit_before_the_first_decision_row": 0,
             "completed_event_recorded_after_its_deadline": 0,
             "duplicate_dispatch_of_the_same_event_id": 0,
             "same_target_credited_twice": 0}
    violations: list[str] = []
    for seed, artefacts in seeds.items():
        events_total = credited_total = fields_missing = 0
        closure: list[str] = []
        tick_mismatch: list[str] = []
        duplicate_ids: list[str] = []
        episodes_with_expiry: list[str] = []
        for episode, row in sorted(artefacts["rows"].items()):
            ledger = row.get("task_reward_events") or []
            ids = [item["event_id"] for item in ledger]
            if len(ids) != len(set(ids)):
                duplicate_ids.append(f"s{seed}e{episode}")
                cases["duplicate_dispatch_of_the_same_event_id"] += len(ids) - len(set(ids))
            steps = tape_steps(artefacts, episode)
            events_total += len(ledger)
            credited_here = [item for item in ledger if item["credited"]]
            credited_total += len(credited_here)
            for item in ledger:
                fields_missing += len([field for field in EVENT_REQUIRED if field not in item])
                if item["relay_tick"] != int(round(item["event_time"])):
                    tick_mismatch.append(f"s{seed}e{episode}:{item['event_id']}")
                if not item["on_time"]:
                    cases["completed_event_recorded_after_its_deadline"] += 1
            targets = [item["task_id"] for item in credited_here]
            cases["same_target_credited_twice"] += len(targets) - len(set(targets))
            for item in credited_here:
                tick = int(item["relay_tick"])
                if tick in steps:
                    cases["credit_at_a_tick_that_has_a_decision_row"] += 1
                    if tick == max(steps):
                        cases["credit_owned_by_the_final_decision_row"] += 1
                else:
                    cases["credit_at_a_tick_with_no_decision_row"] += 1
                if steps and tick < min(steps):
                    cases["credit_before_the_first_decision_row"] += 1
            stream_sum = int(round(float(sum(row["task_stream"].values()))))
            if not (int(row["on_time"]) == stream_sum == len(credited_here)
                    == int(row["task_reward_total"])):
                closure.append(f"s{seed}e{episode}: on_time={row['on_time']} "
                               f"task_stream_sum={stream_sum} credited={len(credited_here)} "
                               f"task_reward_total={row['task_reward_total']}")
            if int(row["expired"]) > 0:
                episodes_with_expiry.append(f"s{seed}e{episode}")
        prefix_task = round(sum(float(item["update"]["unassigned_prefix_task_reward"])
                                for item in artefacts["batches"]), 6)
        violations.extend(closure + tick_mismatch + duplicate_ids)
        per_seed[f"s{seed}"] = {
            "rows": len(artefacts["rows"]),
            "event_ledger_rows": events_total,
            "credited_events": credited_total,
            "episodes_whose_written_keys_are_crediting_event_ticks": sum(
                1 for row in artefacts["rows"].values()
                if set(int(key) for key in row["task_stream"])
                == {int(item["relay_tick"]) for item in (row["task_reward_events"] or [])
                    if item["credited"]}),
            "event_rows_missing_a_required_field": fields_missing,
            "relay_tick_not_equal_to_rounded_event_time": tick_mismatch,
            "closure_failures": closure,
            "episodes_with_duplicate_event_ids": duplicate_ids,
            "episodes_with_an_expired_target": episodes_with_expiry,
            "unassigned_prefix_task_reward": prefix_task}
    return {"per_seed": per_seed, "case_counts": cases, "violations": violations,
            "note": "closure means: for every episode lifecycle on_time == sum(task_stream) == "
                    "credited ledger rows == task_reward_total; every written task_stream key is "
                    "the relay tick of a crediting event; a duplicate event id inside one "
                    "episode would be an idempotency failure. The collector only ever sees "
                    "COMPLETED events, so an expiry shows up as a non-completed ProgressEvent "
                    "that is not in the ledger at all (ROUND35 C9 same reading)",
            "passed": not violations and all(item["event_rows_missing_a_required_field"] == 0
                                             and item["unassigned_prefix_task_reward"] == 0.0
                                             for item in per_seed.values())
                      and cases["credit_before_the_first_decision_row"] == 0
                      and cases["same_target_credited_twice"] == 0}


def tape_steps(artefacts: dict[str, Any], episode: int) -> list[int]:
    """The decision rows of one episode, cached per verification run."""
    cache = artefacts.setdefault("_tape_steps", {})
    if str(episode) not in cache:
        row = artefacts["rows"][episode]
        cache[str(episode)] = [int(item["step"]) for item in
                               tape_rows(int(artefacts["seed"]), episode,
                                         int(row["scenario_seed"]))]
    return cache[str(episode)]


# ------------------------------------------------------------------- V3 批 0 逐字节 -------------
def v3_batch_zero_byte_identity() -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in r36.SEEDS:
        rows = seed_rows(seed)
        episodes: dict[str, Any] = {}
        for episode in BATCH_ZERO_EPISODES:
            row = rows[episode]
            old = archived_row("ALL_ROWS", seed, episode)
            same, differing = {}, []
            for field in ("on_time", "completed", "expired", "discovered", "active",
                          "emissions", "spent_bytes", "airtime_tap_total", "steps",
                          "behaviour_sha256", "relay_counters", "cost_stream",
                          "emission_sequence", "refusal_sequence", "transport_parameters",
                          "budget_bytes", "b0_bytes", "scenario_seed"):
                equal = row.get(field) == old.get(field)
                same[field] = equal
                if not equal:
                    differing.append(field)
            new_tape = tape_rows(seed, episode, int(row["scenario_seed"]))
            old_tape = archived_tape("ALL_ROWS", seed, episode)
            tape_equal, gaps = True, []
            if len(new_tape) != len(old_tape):
                tape_equal = False
            else:
                for left, right in zip(new_tape, old_tape):
                    for field in TAPE_FIELDS:
                        if field == "features":
                            if not np.array_equal(np.asarray(left[field], dtype=np.float32),
                                                  np.asarray(right[field], dtype=np.float32)):
                                tape_equal = False
                        elif left[field] != right[field]:
                            tape_equal = False
                    gaps.append(abs(float(left["log_prob"]) - float(right["log_prob"])))
            legacy_keys = sorted(int(key) for key in old["task_stream"])
            fixed_keys = sorted(int(key) for key in row["task_stream"])
            old_steps = [int(item["step"]) for item in old_tape]

            def owner(tick: int) -> Any:
                earlier = [step for step in old_steps if step <= tick]
                return max(earlier) if earlier else "prefix"

            moved = [key for key in legacy_keys if owner(key) != owner(key - 1)]
            episodes[f"e{episode}"] = {
                "scenario_matches_order": int(row["scenario_seed"])
                                          == v2_common.scenario_order()[episode],
                "fields_equal_to_the_archive": same,
                "fields_that_differ": differing,
                "tape_rows": {"new": len(new_tape), "archived": len(old_tape),
                              "all_recorded_fields_equal": tape_equal,
                              "max_abs_log_prob_gap": max(gaps) if gaps else 0.0},
                "archived_task_stream_keys": legacy_keys,
                "fixed_task_stream_keys": fixed_keys,
                "keys_are_exactly_archived_minus_one": fixed_keys == [key - 1 for key in
                                                                      legacy_keys],
                "credits_whose_owning_action_row_the_fix_moves": len(moved),
                "credits_total": len(legacy_keys),
            }
        per_seed[f"s{seed}"] = episodes
    checks = [item for episodes in per_seed.values() for item in episodes.values()]
    return {"per_seed": per_seed,
            "grade": "A 直接逐字节（批 0 在任何权重移动之前，初值四源相同已由 G9 证明）",
            "all_observable_fields_equal": all(not item["fields_that_differ"]
                                              for item in checks),
            "all_tapes_equal": all(item["tape_rows"]["all_recorded_fields_equal"]
                                   for item in checks),
            "all_log_prob_gaps_zero": all(item["tape_rows"]["max_abs_log_prob_gap"] == 0.0
                                          for item in checks),
            "all_keys_are_archived_minus_one":
                all(item["keys_are_exactly_archived_minus_one"] for item in checks),
            "config_hash_not_compared": "它覆盖本轮自己的输出路径，跨轮不等是预期（ROUND35 P6 "
                                        "同口径）",
            "passed": all(not item["fields_that_differ"] and
                          item["tape_rows"]["all_recorded_fields_equal"] and
                          item["tape_rows"]["max_abs_log_prob_gap"] == 0.0 and
                          item["keys_are_exactly_archived_minus_one"] and
                          item["scenario_matches_order"] for item in checks)}


# ------------------------------------------------------------------- V4 训练量 -------------
def v4_training_volume(seeds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    baseline = r36.archived_baseline()["per_seed"]
    per_seed: dict[str, Any] = {}
    for seed, artefacts in seeds.items():
        records = artefacts["batches"]
        archived = baseline[f"s{seed}"]
        per_seed[f"s{seed}"] = {
            "episodes": {"this_round": len(artefacts["rows"]), "archived": archived["rows"],
                         "frozen_requirement": r36.EPISODES_PER_SEED},
            "batches": {"this_round": len(records), "archived": archived["batches"],
                        "frozen_requirement": r36.BATCHES_PER_SEED},
            "updates_in_state": {"this_round": artefacts["final_state"]["updates"],
                                 "archived": archived["batches"]},
            "optimizer_steps": {"this_round": artefacts["final_state"]["optimizer_steps"],
                                "archived": archived["optimizer_steps"],
                                "rule": "epochs(4) x ceil(sample rows / 256) per batch; the rule "
                                        "is the frozen one, the count is a consequence of how "
                                        "many decision rows the rollouts logged"},
            "samples": {"this_round": sum(int(item["update"]["samples"]) for item in records),
                        "archived": archived["samples"]},
            "per_batch_samples_this_round": [int(item["update"]["samples"]) for item in records],
            "per_batch_steps_this_round": [int(item["update"]["optimizer_steps"])
                                           for item in records],
            "batch0_sample_count_equals_the_archive":
                int(records[0]["update"]["samples"])
                == r36.read_json(sorted((r36.ROUND32 / "checkpoints").glob(
                    f"r32_batch_ALL_ROWS_s{seed}_b0.json"))[0])["update"]["samples"],
            "likelihood_alignment_all_passed": all(
                item["update"]["likelihood_alignment"]["passed"] for item in records),
            "controller_changed_in_every_batch": all(
                item["controller_parameters_before"]["parameter_sha256"]
                != item["controller_parameters_after"]["parameter_sha256"] for item in records),
            "epochs_and_minibatch_rule": {"epochs_per_batch": v2_common.EPOCHS_PER_BATCH,
                                          "minibatch_steps": v2_common.MINIBATCH_STEPS,
                                          "episodes_per_update": v2_common.EPISODES_PER_UPDATE},
            "lambda_first": records[0]["dual"]["lambda_next"],
            "lambda_last": records[-1]["dual"]["lambda_next"],
            "archived_lambda_last": archived["lambda_last"],
            "g_hat_never_negative_this_round": all(item["dual"]["g_hat"] >= 0
                                                   for item in records)}
    return {"per_seed": per_seed,
            "frozen_volume_requirement": {"episodes_per_seed": r36.EPISODES_PER_SEED,
                                          "batches_per_seed": r36.BATCHES_PER_SEED,
                                          "episodes_per_update": v2_common.EPISODES_PER_UPDATE},
            "passed": all(item["episodes"]["this_round"] == r36.EPISODES_PER_SEED
                          and item["batches"]["this_round"] == r36.BATCHES_PER_SEED
                          and item["likelihood_alignment_all_passed"]
                          and item["controller_changed_in_every_batch"]
                          for item in per_seed.values())}


# ------------------------------------------------------------------- V5 未改写旧件 -------------
def v5_old_artefacts_untouched() -> dict[str, Any]:
    """Re-hash the legacy artefacts ROUND35 recorded, to prove this round rewrote nothing.

    This round only ever imports the Stage C modules and reads archived rows; the check is the
    byte-level proof of that claim, taken after the launches rather than asserted in advance.
    """
    layer = r36.read_json(LEGACY_PASSED_TO_ROUND35)
    records = layer["checks"]["L_legacy_round32_checkpoints_content_layer"]["per_file_records"]
    per_file: dict[str, Any] = {}
    mismatched: list[str] = []
    for name, record in sorted(records.items()):
        candidates = [path for path in (list(r36.ROUND32.rglob(name))
                                        + list(r36.STAGE_C.rglob(name)))
                      if path.is_file()]
        if len(candidates) != 1:
            mismatched.append(f"{name}: located {len(candidates)} copies")
            continue
        now = r36.sha256_file(candidates[0])
        recorded = record["L1_file_sha256"]["round35_recomputed"]
        per_file[name] = {"path": str(candidates[0].relative_to(r36.ROOT)).replace("\\", "/"),
                          "recorded_by_round35": recorded, "now": now, "equal": now == recorded}
        if now != recorded:
            mismatched.append(name)
    anchors = r36.read_json(r36.ROUND35 / "ANCHORS_AND_ENVIRONMENT.json")
    round34_files: dict[str, Any] = {}
    not_located: list[str] = []
    for key, digest in sorted(anchors["round34_file_sha256"].items()):
        # ROUND35 recorded these as repo-relative paths with the platform's own separator, so
        # they are resolved against the repository root rather than glued onto the round dir.
        path = r36.ROOT / key
        if not path.exists():
            path = r36.ROUND34 / Path(key.replace("\\", "/")).name
        name = str(path.relative_to(r36.ROOT)).replace("\\", "/")
        if not path.exists():
            not_located.append(name)
            round34_files[name] = {"recorded_by_round35": digest, "now": None, "equal": False}
            continue
        current = r36.sha256_file(path)
        round34_files[name] = {"recorded_by_round35": digest, "now": current,
                               "equal": current == digest}
    round34_mismatched = sorted(name for name, item in round34_files.items()
                                if not item["equal"])
    stage_c = r36.code_identity()["stage_c_harness"]
    return {"legacy_round32_checkpoints_compared": len(per_file),
            "legacy_checkpoint_mismatches": mismatched,
            "per_legacy_file": per_file,
            "round34_deliverables_compared": len(round34_files),
            "round34_deliverable_mismatches": round34_mismatched,
            "round34_deliverables_not_located": not_located,
            "round34_per_file": round34_files,
            "stage_c_harness_hashes_at_verification": stage_c,
            "written_by_this_round_only_inside": str(r36.HERE.relative_to(r36.ROOT)),
            "passed": not mismatched and not round34_mismatched and len(per_file) == 18}


# ------------------------------------------------------------------- V6 轨迹分歧度 -------------
def v6_trajectory_vs_archive(seeds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """How far this round's runs actually diverge from the ROUND32 archive, counted per episode.

    The fix moves which action row owns a completion, so from the first update onward the
    weights, the sampled stream and therefore the byte content of later episodes are expected to
    differ. That is an intervention effect, not a defect and not a performance claim. Episode
    outcomes are reported only as counts with a denominator.
    """
    per_seed: dict[str, Any] = {}
    for seed, artefacts in seeds.items():
        same_on_time = same_bytes = same_row_count = 0
        behaviour_equal = []
        first_divergence: int | None = None
        rows_total = 0
        for episode in sorted(artefacts["rows"]):
            row = artefacts["rows"][episode]
            old = archived_row("ALL_ROWS", seed, episode)
            rows_total += 1
            same_on_time += int(int(row["on_time"]) == int(old["on_time"]))
            same_bytes += int(int(row["spent_bytes"]) == int(old["spent_bytes"]))
            equal_behaviour = row["behaviour_sha256"] == old["behaviour_sha256"]
            if equal_behaviour:
                behaviour_equal.append(episode)
            if not equal_behaviour and first_divergence is None:
                first_divergence = episode
            new_steps = tape_steps(artefacts, episode)
            old_steps = [int(item["step"]) for item in archived_tape("ALL_ROWS", seed, episode)]
            same_row_count += int(new_steps == old_steps)
        archived_batches = [r36.read_json(path) for path in sorted(
            (r36.ROUND32 / "checkpoints").glob(f"r32_batch_ALL_ROWS_s{seed}_b*.json"),
            key=lambda path: int(path.stem.split("_b")[-1]))]
        mine = [int(item["update"]["samples"]) for item in artefacts["batches"]]
        theirs = [int(item["update"]["samples"]) for item in archived_batches]
        common_prefix = 0
        for left, right in zip(mine, theirs):
            if left != right:
                break
            common_prefix += 1
        per_seed[f"s{seed}"] = {
            "episodes_compared": rows_total,
            "episodes_with_equal_on_time": same_on_time,
            "episodes_with_equal_spent_bytes": same_bytes,
            "episodes_with_equal_behaviour_hash": len(behaviour_equal),
            "episodes_with_equal_decision_row_sequence": same_row_count,
            "first_episode_whose_behaviour_hash_differs": first_divergence,
            "batches_with_equal_sample_count_prefix": common_prefix,
            "samples_this_round": sum(mine),
            "samples_archived": sum(theirs),
            "optimizer_steps_this_round": sum(int(item["update"]["optimizer_steps"])
                                              for item in artefacts["batches"]),
            "optimizer_steps_archived": sum(int(item["update"]["optimizer_steps"])
                                            for item in archived_batches),
            "parameter_hash_after_last_batch":
                artefacts["batches"][-1]["controller_parameters_after"]["parameter_sha256"],
            "parameter_hash_after_last_batch_archived":
                archived_batches[-1]["controller_parameters_after"]["parameter_sha256"],
            "lambda_after_last_batch": artefacts["batches"][-1]["dual"]["lambda_next"],
            "lambda_after_last_batch_archived": archived_batches[-1]["dual"]["lambda_next"],
            "batch_zero_update_effect": {
                "weights_before_batch0_equal_to_the_archive":
                    artefacts["batches"][0]["controller_parameters_before"]["parameter_sha256"]
                    == archived_batches[0]["controller_parameters_before"]["parameter_sha256"],
                "samples_equal_to_the_archive": mine[0] == theirs[0],
                "g_hat_equal_to_the_archive":
                    artefacts["batches"][0]["dual"]["g_hat"]
                    == archived_batches[0]["dual"]["g_hat"],
                "policy_surrogate_this_round": artefacts["batches"][0]["update"]["policy_surrogate"],
                "policy_surrogate_archived": archived_batches[0]["update"]["policy_surrogate"],
                "weights_after_batch0_equal_to_the_archive":
                    artefacts["batches"][0]["controller_parameters_after"]["parameter_sha256"]
                    == archived_batches[0]["controller_parameters_after"]["parameter_sha256"],
                "meaning": "same starting weights, same logged decision rows, same g_hat, but a "
                           "different surrogate and therefore different weights afterwards: that "
                           "is the intervention itself (credit moved to an earlier action row), "
                           "not a performance statement"}}
    return {"per_seed": per_seed,
            "interpretation": "the batch-0 episodes are byte-identical (V3); from the first "
                              "update onward the weights and the sampled stream differ, which is "
                              "the intervention itself. Equal outcome counts across episodes are "
                              "observations with a denominator, not evidence that the constraint "
                              "is better satisfied, and nothing here is a learning claim",
            "passed": all(item["episodes_compared"] == r36.EPISODES_PER_SEED
                          and item["parameter_hash_after_last_batch"]
                          != item["parameter_hash_after_last_batch_archived"]
                          for item in per_seed.values())}


def main(argv: list[str]) -> int:
    seeds: dict[int, dict[str, Any]] = {}
    for seed in r36.SEEDS:
        directory = r36.seed_directory(seed)
        if not directory.exists():
            raise SystemExit(f"seed {seed} has no artefact directory yet: {directory}")
        rows = seed_rows(seed)
        batches = sorted((directory / "checkpoints").glob(f"v2_batch_s{seed}_b*.json"),
                         key=lambda path: int(path.stem.split("_b")[-1]))
        sanity = r36.read_json(directory / "SANITY.json")
        seeds[seed] = {"seed": seed, "rows": rows,
                       "batches": [r36.read_json(path) for path in batches],
                       "final_state": sanity["real_loader_restore_of_ep128"]["training_state_fields"],
                       "sanity": sanity}
    checks = {
        "V1_milestone_row_and_checkpoint_version_fields": v1_version_fields(seeds),
        "V2_event_ledger_integrity": v2_event_ledgers(seeds),
        "V3_batch_zero_byte_identity_vs_round32": v3_batch_zero_byte_identity(),
        "V4_frozen_training_volume": v4_training_volume(seeds),
        "V5_old_artefacts_not_rewritten": v5_old_artefacts_untouched(),
        "V6_trajectory_divergence_vs_round32": v6_trajectory_vs_archive(seeds)}
    payload = {"kind": "round36-verification", "round": "36",
               "environment_launches_used": 0, "optimizer_steps_used": 0,
               "checks": checks, "all_passed": all(item["passed"] for item in checks.values()),
               "note": "these checks read only on-disk artefacts; the trajectory after batch 0 "
                       "is expected to differ from ROUND32 because the fix changed credit "
                       "assignment, and no check here treats any difference as a performance "
                       "claim"}
    r36.write_json("ROUND36_VERIFICATION.json", payload, refresh=True)
    print(json.dumps({key: value["passed"] for key, value in checks.items()},
                     ensure_ascii=False, indent=1))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
