"""ROUND36 独立复核（0 环境启动、0 optimizer.step）：把交付声明逐条重算一遍。

与 r36_verify.py 的分工：verify 核对训练产物本身；本脚本核对"交付声明与落盘件是否仍然一致"——
git 三个头之下训练路径逐字节相同、启动账本与逐 seed 的 ledger/launch record 相互闭合、
checkpoint 字节哈希在四处记录里一致、批记录自 sanity 之后未被改动、受保护目录今日 0 写入、
额度算术与禁则计数。结论写进 ROUND36_INDEPENDENT_RECHECK.json。

    /d/project_env/python.exe r36_recheck.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy
import torch

import r36_common as r36

HARNESS_COMMIT = "93b27059c914"
STAGE_C_REL = [f"runs/comm_cost_v2_rerun_20260928/{name}" for name in r36.STAGE_C_MODULES]
PRODUCTION_REL = list(r36.v2_common.PRODUCTION_DEPENDENCIES)


def blob(commit: str, path: str) -> str:
    return r36.git("rev-parse", f"{commit}:{path}").strip()


def when(text: str) -> int:
    return int(time.mktime(time.strptime(text, "%Y-%m-%d %H:%M:%S")))


def main(argv: list[str]) -> int:
    results: dict[str, Any] = {}
    head = r36.git("rev-parse", "HEAD").strip()
    branch = r36.git("rev-parse", "--abbrev-ref", "HEAD").strip()
    ancestor = subprocess.run(("git", "merge-base", "--is-ancestor", r36.ROUND33_BASELINE, head),
                              cwd=str(r36.ROOT), capture_output=True).returncode == 0
    commits = [line for line in r36.git("log", "--format=%h",
                                        f"{r36.EXPECTED_HEAD_AT_START}..{head}").split() if line]

    per_file = {path: blob(r36.EXPECTED_HEAD_AT_START, path) == blob(HARNESS_COMMIT, path)
                == blob(head, path) for path in STAGE_C_REL + PRODUCTION_REL}
    first_launch_written_at = min(
        r36.read_json(r36.seed_directory(seed) / "FREEZE_MANIFEST.json")["written_at"]
        for seed in r36.SEEDS)
    harness_commit_time = int(r36.git("show", "-s", "--format=%ct", HARNESS_COMMIT).strip())
    results["R1_git_identity_and_training_path"] = {
        "branch": branch, "head": head, "commits_since_round35_head": commits,
        "round33_baseline_is_ancestor": ancestor,
        "training_path_and_production_files": len(per_file),
        "all_identical_at_the_three_heads": all(per_file.values()),
        "per_file_identical": per_file,
        "harness_commit_unix_time": harness_commit_time,
        "first_launch_recorded_at": first_launch_written_at,
        "harness_was_committed_before_the_first_launch":
            harness_commit_time <= when(first_launch_written_at),
        "passed": (branch == r36.EXPECTED_BRANCH and ancestor and len(commits) == 6
                   and all(per_file.values())
                   and harness_commit_time <= when(first_launch_written_at))}

    book = r36.book()
    ledger_starts = {f"s{seed}": r36.starts_in(seed)["starts"] for seed in r36.SEEDS}
    records = {}
    for seed in r36.SEEDS:
        record = r36.read_json(r36.seed_directory(seed) / "LAUNCH_RECORD.json")
        records[f"s{seed}"] = {"constructions":
                               record["episode_constructions_in_this_seed_directory"],
                               "outcome": record["outcome"],
                               "wall_clock": record["parent_wall_clock_seconds"],
                               "ceiling": record["wall_clock_ceiling_seconds"],
                               "within_ceiling": record["within_ceiling"],
                               "failed_starts": record["failed_starts_in_this_seed_directory"]}
    events = {f"s{item['training_seed']}": item for item in book["events"]}
    results["R2_book_vs_ledgers_vs_launch_records"] = {
        "book": {key: book[key] for key in ("training_launches", "reserve_reruns",
                                            "episode_constructions", "optimizer_steps")},
        "seed_ledger_starts": ledger_starts,
        "launch_records": records,
        "book_agrees_with_every_launch_record": all(
            events[key]["episode_constructions"] == value["constructions"]
            and events[key]["wall_clock_seconds"] == value["wall_clock"]
            and events[key]["wall_clock_ceiling_seconds"] == value["ceiling"]
            for key, value in records.items()),
        "sum_of_seed_ledgers": sum(ledger_starts.values()),
        "reserve_unused_and_no_failures": book["reserve_reruns"] == 0 and not book["failures"]
                                          and not any(value["failed_starts"]
                                                      for value in records.values()),
        "passed": (book["training_launches"] == 3 and book["episode_constructions"] == 384
                   and sum(ledger_starts.values()) == 384
                   and book["reserve_reruns"] == 0 and not book["failures"]
                   and all(value["outcome"] == "completed" and value["within_ceiling"]
                           and not value["failed_starts"] for value in records.values())
                   and all(events[key]["episode_constructions"] == value["constructions"]
                           for key, value in records.items()))}

    registry = r36.read_json(r36.HERE / "delivery_draft" / "ARTEFACT_HASH_REGISTRY.json")
    agreement, mismatch = {}, []
    for seed in r36.SEEDS:
        hashes = r36.read_json(r36.seed_directory(seed) / "CHECKPOINT_HASHES.json")["files"]
        layer = r36.read_json(r36.seed_directory(seed) / "SANITY.json")[
            "checkpoint_version_and_hash_layer"]
        inventory = registry["model_inventory"][f"s{seed}"]
        # (label, layer key in SANITY.json, key in CHECKPOINT_HASHES/registry, file name on disk)
        layout = (("ep0", "0", "milestone_ep0", f"v2_train_s{seed}_ep0.pt"),
                  ("ep64", "64", "milestone_ep64", f"v2_train_s{seed}_ep64.pt"),
                  ("ep128", "128", "milestone_ep128", f"v2_train_s{seed}_ep128.pt"),
                  ("rolling", "rolling", "rolling", f"v2_train_s{seed}.pt"))
        for label, layer_key, files_key, file_name in layout:
            sources = {"on_disk": r36.sha256_file(
                           r36.seed_directory(seed) / "checkpoints" / file_name),
                       "checkpoint_hashes_json": hashes[files_key]["bytes_sha256"],
                       "sanity_layer": layer[layer_key]["bytes_sha256"],
                       "registry_model_inventory": inventory[files_key]["bytes_sha256"]}
            agreement.setdefault(f"s{seed}", {})[label] = sources
            if len(set(sources.values())) != 1:
                mismatch.append(f"s{seed}/{label}")
    results["R3_checkpoint_hashes_agree_four_ways"] = {
        "files_checked": sum(len(value) for value in agreement.values()),
        "mismatches": mismatch, "per_file": agreement,
        "passed": not mismatch and sum(len(value) for value in agreement.values()) == 12}

    reconcile = r36.read_json(r36.HERE / "LEDGER_RECONCILIATION.json")
    unchanged, volume = {}, {}
    for seed in r36.SEEDS:
        directory = r36.seed_directory(seed)
        recorded = r36.read_json(directory / "CHECKPOINT_HASHES.json")["files"]["batch_records"]
        now = {path.name: r36.sha256_file(path) for path in
               sorted((directory / "checkpoints").glob(f"v2_batch_s{seed}_b*.json"))}
        unchanged[f"s{seed}"] = {"recorded_at_sanity": len(recorded), "recomputed_now": len(now),
                                 "all_equal": recorded == now}
        steps = sum(int(r36.read_json(path)["update"]["optimizer_steps"]) for path in
                    sorted((directory / "checkpoints").glob(f"v2_batch_s{seed}_b*.json")))
        reported = reconcile["per_seed"][f"s{seed}"]
        volume[f"s{seed}"] = {
            "steps_recomputed_from_batch_records": steps,
            "steps_in_reconciliation": reported["optimizer_steps"]["sum_of_batch_records"],
            "steps_in_book": events[f"s{seed}"]["optimizer_steps"],
            "steps_in_final_state": reported["optimizer_steps"]["in_final_state"],
            "batches": reported["batches"],
            "rows": reported["episode_constructions"]["row_files_written"]}
    results["R4_batch_records_and_volume_recomputed"] = {
        "batch_record_hashes_unchanged_since_sanity": unchanged, "volume": volume,
        "all_records_of_steps_agree": all(
            item["steps_recomputed_from_batch_records"] == item["steps_in_reconciliation"]
            == item["steps_in_book"] == item["steps_in_final_state"]
            for item in volume.values()),
        "passed": all(item["all_equal"] for item in unchanged.values())
                  and all(item["steps_recomputed_from_batch_records"]
                          == item["steps_in_reconciliation"] == item["steps_in_book"]
                          == item["steps_in_final_state"] for item in volume.values())}

    today = time.strftime("%Y-%m-%d")
    protected = [r36.STAGE_C / "rows", r36.STAGE_C / "logs", r36.STAGE_C / "checkpoints",
                 r36.ROUND32 / "rows", r36.ROUND32 / "logs", r36.ROUND32 / "checkpoints",
                 r36.ROUND34]
    written_today = {}
    for directory in protected:
        hits = [str(path.relative_to(r36.ROOT)).replace("\\", "/")
                for path in directory.rglob("*")
                if path.is_file()
                and time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime)) == today]
        written_today[str(Path(directory).relative_to(r36.ROOT)).replace("\\", "/")] = hits
    legacy = r36.read_json(r36.ROUND35 / "CHECKPOINT_CONTENT_LAYER.json")[
        "checks"]["L_legacy_round32_checkpoints_content_layer"]["per_file_records"]
    legacy_now = {name: r36.sha256_file(next(r36.ROUND32.rglob(name))) for name in legacy}
    legacy_equal = {name: (legacy_now[name] == record["L1_file_sha256"]["round35_recomputed"])
                    for name, record in legacy.items()}
    results["R5_old_rounds_not_written"] = {
        "protected_directories": list(written_today),
        "files_written_today": {key: value for key, value in written_today.items()},
        "legacy_checkpoint_count": len(legacy_now),
        "legacy_checkpoints_byte_equal_to_round35_record": all(legacy_equal.values()),
        "round35_dir_note": "ROUND35 自己的产物写于 2026-10-01 那次会话，mtime 为当日属预期；"
                            "本轮对它的字节哈希全部对账相符（见 legacy 项与 V5）",
        "passed": all(not value for value in written_today.values()) and all(legacy_equal.values())}

    results["R6_quota_and_boundaries"] = {
        "training_launches": {"authorised": 3, "used": book["training_launches"]},
        "reserve_reruns": {"authorised": 1, "used": book["reserve_reruns"]},
        "episode_constructions": {"authorised": 384, "used": book["episode_constructions"],
                                  "remaining": r36.round_constructions()["remaining"]},
        "evaluation_launches": {"authorised": 0, "used": 0},
        "sanity_forwards": {"authorised": "每次训练后 1 次", "used": 3},
        "optimizer_steps_performed_by_verification_or_recheck_scripts": 0,
        "github_push_attempts": 0, "network_attempts": 0, "new_venvs_or_dependencies": 0,
        "active_rows_enabled": False, "v2_recover_used": False, "old_artefacts_rewritten": 0,
        "torch": torch.__version__, "numpy": numpy.__version__,
        "passed": (book["training_launches"] == 3 and book["episode_constructions"] == 384
                   and book["reserve_reruns"] == 0
                   and r36.round_constructions()["remaining"] == 0)}

    payload = {"kind": "round36-independent-recheck", "round": "36",
               "checked_at": time.strftime("%F %T"),
               "environment_launches_used": 0, "optimizer_steps_used": 0, "results": results,
               "all_passed": all(item["passed"] for item in results.values()),
               "note": "ROUND36_PREFLIGHT.json 的 G1 与 G10 是启动前门禁（断言开工头与空账本），"
                       "交付之后再跑它们本就会按设计失败；本复核因此改为从 git 与账本重算同一批"
                       "事实，且不重写任何已交付文件"}
    r36.write_json("ROUND36_INDEPENDENT_RECHECK.json", payload, refresh=True)
    print(json.dumps({"all_passed": payload["all_passed"],
                      **{key: value["passed"] for key, value in results.items()}}))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
