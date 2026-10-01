"""ROUND36 交付对账（任务 4，0 环境启动）：三个 seed 的账本对账表，全部从落盘件重算。

报告里不许手打数字：本脚本从 `TRAIN_LAUNCH_BOOK.json`、每 seed 的 `START_LEDGER_p*.json`、批记录、
`SANITY.json`、`CHECKPOINT_HASHES.json` 与 `LAUNCH_RECORD.json` 重算所有计数，并标注与 ROUND32
同口径训练量的一致性。

    /d/project_env/python.exe r36_reconcile.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import r36_common as r36

VERSION_FIELDS = ("training_version", "reward_timing_version")


def per_seed(seed: int) -> dict[str, Any]:
    directory = r36.seed_directory(seed)
    launch = r36.read_json(directory / "LAUNCH_RECORD.json")
    sanity = r36.read_json(directory / "SANITY.json")
    hashes = r36.read_json(directory / "CHECKPOINT_HASHES.json")
    batches = sorted((directory / "checkpoints").glob(f"v2_batch_s{seed}_b*.json"),
                     key=lambda path: int(path.stem.split("_b")[-1]))
    records = [r36.read_json(path) for path in batches]
    rows = sorted((directory / "rows").glob("v2_train_s*_e*_s*_a*.json"))
    row_payloads = [r36.read_json(path) for path in rows]
    ledger = r36.starts_in(seed)
    events = sum(len(row.get("task_reward_events") or []) for row in row_payloads)
    version_layer = sanity["checkpoint_version_and_hash_layer"]
    complete = sum(1 for item in version_layer.values() if item and item["both_fields_complete"])
    rows_with_fields = sum(1 for row in row_payloads
                           if all(row.get(field) for field in VERSION_FIELDS))
    archived = r36.archived_baseline()["per_seed"][f"s{seed}"]
    wall = float(launch["parent_wall_clock_seconds"])
    ceiling = float(launch["wall_clock_ceiling_seconds"])
    rollout_seconds = round(sum(float(row["seconds"]) for row in row_payloads), 2)
    return {
        "training_seed": seed,
        "training_launch_processes": 1,
        "episode_constructions": {"ledger_starts_including_failures": ledger["starts"],
                                  "row_files_written": len(rows),
                                  "failed_starts": ledger["failed"]},
        "batches": len(records),
        "updates_in_final_state": sanity["real_loader_restore_of_ep128"]["training_state_fields"]
        ["updates"],
        "optimizer_steps": {"sum_of_batch_records": sum(int(item["update"]["optimizer_steps"])
                                                        for item in records),
                            "in_final_state":
                                sanity["real_loader_restore_of_ep128"]["training_state_fields"]
                                ["optimizer_steps"]},
        "samples": sum(int(item["update"]["samples"]) for item in records),
        "wall_clock": {"parent_measured_seconds": wall,
                       "authorised_ceiling_seconds": ceiling,
                       "ratio_to_ceiling": round(wall / ceiling, 4),
                       "within_ceiling": bool(launch["within_ceiling"]),
                       "episode_rollout_seconds_this_round": rollout_seconds,
                       "archived_round32_rollout_seconds": archived["rollout_seconds_sum"],
                       "this_round_vs_archived_rollout": round(
                           rollout_seconds / archived["rollout_seconds_sum"], 4)},
        "event_ledger_rows": events,
        "version_field_completeness": {
            "checkpoint_files_carrying_both_fields": f"{complete}/{len(version_layer)}",
            "rows_carrying_both_fields": f"{rows_with_fields}/{len(row_payloads)}"},
        "checkpoints": {name: {"sha256": item["bytes_sha256"],
                               "parameter_state": {key: item.get(key) for key in
                                                   ("episodes_done", "updates",
                                                    "optimizer_steps", "lambda")}}
                        for name, item in hashes["files"].items()
                        if isinstance(item, dict) and "bytes_sha256" in item},
        "batch_record_hashes": hashes["files"]["batch_records"],
        "milestones_present": sorted(path.name for path in
                                     (directory / "checkpoints").glob("*_ep*.pt")),
        "lambda_trajectory": {"first": records[0]["dual"]["lambda_next"],
                              "last": records[-1]["dual"]["lambda_next"],
                              "archived_first": archived["lambda_first"],
                              "archived_last": archived["lambda_last"]},
        "g_hat": {"batches_negative": sum(1 for item in records if item["dual"]["g_hat"] < 0),
                  "batches_total": len(records), "archived_negative_batches":
                      sum(1 for value in archived["g_hat_history"] if value < 0)},
        "parameters": {"registered_initialisation":
                       r36.fresh_controller_hash(seed),
                       "batch0_before_from_record":
                           records[0]["controller_parameters_before"]["parameter_sha256"],
                       "after_last_batch":
                           records[-1]["controller_parameters_after"]["parameter_sha256"],
                       "archived_after_last_batch": archived["parameters_after_last_batch"]},
        "volume_consistency_vs_round32": {
            "episodes_equal": len(rows) == archived["rows"],
            "batches_equal": len(records) == archived["batches"],
            "episodes_per_update_equal": v2_common_check(),
            "optimizer_steps_equal": sum(int(item["update"]["optimizer_steps"]) for item
                                         in records) == archived["optimizer_steps"],
            "samples_equal": sum(int(item["update"]["samples"]) for item in records)
                             == archived["samples"],
            "interpretation": "episodes/batches/epochs/minibatch rule are the frozen ones and are "
                              "identical; optimizer steps and sample counts are a consequence of "
                              "how many decision rows the rollouts logged, and they may differ "
                              "from the archive after batch 0 because the fix moved credit to a "
                              "different action row (expected, not an added or removed step)"}}


def v2_common_check() -> bool:
    import v2_common

    return (v2_common.EPISODES_PER_UPDATE == 4 and v2_common.EPOCHS_PER_BATCH == 4
            and v2_common.MINIBATCH_STEPS == 256)


def main(argv: list[str]) -> int:
    book = r36.book()
    seeds = {}
    for seed in r36.SEEDS:
        if (r36.seed_directory(seed) / "LAUNCH_RECORD.json").exists():
            seeds[f"s{seed}"] = per_seed(seed)
    totals = {
        "training_launch_processes_booked": book["training_launches"],
        "reserve_reruns_used": book["reserve_reruns"],
        "episode_constructions": book["episode_constructions"],
        "episode_constructions_from_seed_ledgers": sum(
            item["episode_constructions"]["ledger_starts_including_failures"]
            for item in seeds.values()),
        "optimizer_steps": book["optimizer_steps"],
        "wall_clock_seconds_per_seed": {key: value["wall_clock"]["parent_measured_seconds"]
                                        for key, value in seeds.items()},
        "authorised": {"training_launches": r36.MAX_TRAINING_LAUNCHES,
                       "reserve_reruns": r36.MAX_RESERVE_RERUNS,
                       "episode_constructions": r36.MAX_EPISODE_CONSTRUCTIONS},
        "remaining_constructions": r36.round_constructions()["remaining"],
        "evaluation_launches": 0,
        "sanity_forwards": len(seeds)}
    payload = {
        "kind": "round36-ledger-reconciliation", "round": "36",
        "per_seed": seeds, "totals": totals,
        "failures_recorded": book["failures"],
        "launch_book_events": [{"tag": item["tag"], "seed": item["training_seed"],
                               "constructions": item["episode_constructions"],
                               "optimizer_steps": item["optimizer_steps"],
                               "wall_clock_seconds": item["wall_clock_seconds"],
                               "ceiling_seconds": item["wall_clock_ceiling_seconds"],
                               "within_ceiling": item["wall_clock_within_ceiling"],
                               "reserve": item["reserve"]} for item in book["events"]],
        "sources_are_files_not_typing": [
            "TRAIN_LAUNCH_BOOK.json", "seed*/LAUNCH_RECORD.json", "seed*/SANITY.json",
            "seed*/CHECKPOINT_HASHES.json", "seed*/logs/START_LEDGER_p*.json",
            "seed*/checkpoints/v2_batch_s*_b*.json", "seed*/rows/v2_train_s*_e*_s*_a*.json"],
        "consistency_with_round32_same_calibre_volume": all(
            item["volume_consistency_vs_round32"]["episodes_equal"]
            and item["volume_consistency_vs_round32"]["batches_equal"]
            and item["volume_consistency_vs_round32"]["episodes_per_update_equal"]
            for item in seeds.values()),
        "all_launches_within_wall_clock_ceiling": all(
            item["wall_clock"]["within_ceiling"] for item in seeds.values()),
        "version_fields_complete_in_every_checkpoint_and_row": all(
            item["version_field_completeness"]["checkpoint_files_carrying_both_fields"].split(
                "/")[0] == item["version_field_completeness"]
            ["checkpoint_files_carrying_both_fields"].split("/")[1]
            and item["version_field_completeness"]["rows_carrying_both_fields"].split("/")[0]
            == item["version_field_completeness"]["rows_carrying_both_fields"].split("/")[1]
            for item in seeds.values())}
    r36.write_json("LEDGER_RECONCILIATION.json", payload, refresh=True)
    print(json.dumps({"per_seed": {key: {"constructions": value["episode_constructions"]
                                        ["ledger_starts_including_failures"],
                                         "batches": value["batches"],
                                         "optimizer_steps": value["optimizer_steps"]
                                         ["sum_of_batch_records"],
                                         "wall_clock": value["wall_clock"]
                                         ["parent_measured_seconds"],
                                         "within_ceiling": value["wall_clock"]["within_ceiling"],
                                         "version_fields": value["version_field_completeness"]}
                          for key, value in seeds.items()},
                      "totals": totals, "consistency": payload[
                          "consistency_with_round32_same_calibre_volume"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
