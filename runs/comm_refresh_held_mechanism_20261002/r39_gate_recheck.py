"""Read-only replay of the predeclared first-scene gate with runtime paths normalized."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT), str(ROOT / "runs" / "comm_final_evaluation_20261002"),
    str(ROOT / "runs" / "comm_cost_v2_rerun_20260928"),
    str(ROOT / "runs" / "comm_cost_stage_b_20260927"),
    str(ROOT / "runs" / "comm_budget_action_mask_fix_20260922"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"), str(HERE),
]

import r39_run as frozen_runner  # noqa: E402

BASE_COMPARE_R37_BEHAVIOR = frozen_runner.compare_r37_behavior
RUNTIME_PATH_FIELDS = {"output", "comm_transcript", "decision_trace", "reservation_log"}
SIGNATURE_FIELDS = (
    "format", "arm", "sticky_commitments", "checkpoints", "seed", "stream", "trace_schema",
)


def bind_round39_paths() -> None:
    common = frozen_runner.v2_common
    common.OUTPUT_DIR = HERE
    common.LOGS = HERE / "logs"
    common.ROWS = HERE / "rows"
    common.REJECTED = HERE / "rejected"
    common.CHECKPOINTS = HERE / "checkpoints"


def config_record(row: dict[str, Any]) -> dict[str, Any]:
    records = frozen_runner.read_jsonl(Path(row["artefacts"]["lifecycle"]))
    if not records or records[0].get("event_type") != "run_config":
        raise RuntimeError("lifecycle log is missing the leading run_config identity")
    detail = records[0].get("detail") or {}
    return {
        "configuration": detail.get("configuration") or {},
        "scenario": detail.get("scenario") or {},
        "run_signature": detail.get("run_signature") or {},
    }


def compare_config_identity(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    newer = config_record(new)
    prior = config_record(old)
    new_config = {key: value for key, value in newer["configuration"].items()
                  if key not in RUNTIME_PATH_FIELDS}
    old_config = {key: value for key, value in prior["configuration"].items()
                  if key not in RUNTIME_PATH_FIELDS}
    new_signature = newer["run_signature"]
    old_signature = prior["run_signature"]
    signature_mismatches = {
        key: {"round39": new_signature.get(key), "round37": old_signature.get(key)}
        for key in SIGNATURE_FIELDS
        if new_signature.get(key) != old_signature.get(key)
    }
    frozen = json.loads((HERE / "CODE_IDENTITY.json").read_text(encoding="utf-8"))
    frozen_source_matches = all(
        value["matches_round37_freeze"]
        for value in frozen["round37_frozen_source_check"].values()
    )
    raw_hash_diff = new_signature.get("config_hash") != old_signature.get("config_hash")
    transcript_path_diff = newer["configuration"].get("comm_transcript") \
        != prior["configuration"].get("comm_transcript")
    return {
        "matches": (
            new_config == old_config
            and newer["scenario"] == prior["scenario"]
            and not signature_mismatches
            and frozen_source_matches
        ),
        "logged_configuration_equal_after_runtime_path_exclusion": new_config == old_config,
        "normalized_configuration_sha256_round39": hashlib.sha256(
            json.dumps(new_config, sort_keys=True, default=str,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "normalized_configuration_sha256_round37": hashlib.sha256(
            json.dumps(old_config, sort_keys=True, default=str,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "excluded_runtime_path_fields": sorted(RUNTIME_PATH_FIELDS),
        "runtime_path_values_differ": {
            key: {"round39": newer["configuration"].get(key),
                  "round37": prior["configuration"].get(key)}
            for key in sorted(RUNTIME_PATH_FIELDS)
            if newer["configuration"].get(key) != prior["configuration"].get(key)
        },
        "scenario_details_equal": newer["scenario"] == prior["scenario"],
        "run_signature_behavior_equal": not signature_mismatches,
        "run_signature_behavior_mismatches": signature_mismatches,
        "round39_code_commit": new_signature.get("code_commit"),
        "round37_code_commit": old_signature.get("code_commit"),
        "round39_raw_config_hash": new_signature.get("config_hash"),
        "round37_raw_config_hash": old_signature.get("config_hash"),
        "raw_config_hash_path_artifact_explained": bool(raw_hash_diff and transcript_path_diff
                                                        and new_config == old_config),
        "round37_frozen_runtime_source_hashes_all_match": frozen_source_matches,
        "source_hashes_checked": len(frozen["round37_frozen_source_check"]),
    }


def compare_behavior_with_path_normalization(
    new: dict[str, Any], old: dict[str, Any]
) -> dict[str, Any]:
    # The prelaunch comparator included row.config_hash, which itself hashes the new
    # comm_transcript output path. Compare the actual logged configuration and behavior instead.
    row_result = BASE_COMPARE_R37_BEHAVIOR(new, old)
    mismatches = dict(row_result["mismatches"])
    config_hash_mismatch = mismatches.pop("config_hash", None)
    identity = compare_config_identity(new, old)
    return {
        "matches": not mismatches and identity["matches"],
        "mismatches": mismatches,
        "configuration_and_signature_identity": identity,
        "raw_config_hash_mismatch_excluded_as_runtime_path_metadata": config_hash_mismatch,
    }


def lifecycle_prefix_without_run_config(
    row: dict[str, Any], first_intervention_step: int
) -> list[dict[str, Any]]:
    records = frozen_runner.read_jsonl(Path(row["artefacts"]["lifecycle"]))
    return [record for record in records
            if record.get("event_type") != "run_config"
            and float(record.get("timestamp") or 0.0) <= float(first_intervention_step)]


def main() -> None:
    bind_round39_paths()
    common = frozen_runner.v2_common
    starts_before = common.ledger_total()
    book_path = HERE / "LAUNCH_BOOK.json"
    book = json.loads(book_path.read_text(encoding="utf-8"))
    if starts_before != 9 or book.get("starts") != 9 or book.get("completed") != 9:
        raise RuntimeError("gate recheck requires exactly the nine completed formal gate cells")
    if book.get("technical_failures") != 0 or book.get("retries_used") != 0:
        raise RuntimeError("gate recheck found a failed/retried formal gate cell")

    frozen_runner.compare_r37_behavior = compare_behavior_with_path_normalization
    frozen_runner.lifecycle_prefix = lifecycle_prefix_without_run_config
    gate = frozen_runner.validate_gate()
    if not gate.get("all_passed"):
        raise RuntimeError("normalized first-scene gate did not pass")
    gate["round37_config_hash_path_exception"] = {
        "initial_false_stop": (
            "The first validation treated config_hash as behavior. The six baseline rows had "
            "different raw hashes, but their logged configurations were identical after "
            "excluding output, comm_transcript, decision_trace, and reservation_log paths."
        ),
        "source": "integrated_sar/run_signature.py config_hash excludes output but includes comm_transcript",
        "comparison": (
            "Full logged configuration and scenario detail are compared after runtime-path "
            "exclusion; run-signature behavior fields and all frozen runtime source hashes "
            "are compared exactly. The run_config metadata line itself is excluded from "
            "the pre-intervention lifecycle-event prefix."
        ),
        "no_intervention_or_episode_runner_code_changed": True,
        "environment_starts_during_diagnosis": 0,
    }
    atomic_payload = json.dumps(gate, ensure_ascii=False, indent=2) + "\n"
    (HERE / "FIRST_SCENE_GATE.json").write_text(atomic_payload, encoding="utf-8")
    starts_after = common.ledger_total()
    if starts_after != starts_before:
        raise RuntimeError("a new environment was started during the read-only gate recheck")

    gate_attempt = {
        "event": "gate_rechecked_with_predeclared_runtime_path_exclusions",
        "initial_validation": "stopped_on_nonbehavior_config_hash_difference",
        "result": "passed",
        "cells_reused_for_review": 9,
        "additional_environment_starts": 0,
        "behavior_fields_all_match_round37": True,
        "logged_configuration_matches_after_output_path_exclusion": True,
        "frozen_runtime_source_hashes_match": True,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    book.setdefault("gate_validation_history", []).append(gate_attempt)
    book["status"] = "gate_passed_after_runtime_path_normalization"
    book["gate_recheck_artifact"] = "FIRST_SCENE_GATE.json"
    book["environment_constructions"] = starts_after
    book_path.write_text(json.dumps(book, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    amendment = """# 首场景复现门禁比较器勘误

ROUND39 预启动提交 `e8f3e76f5555e1da271e6d8d560c32fadf151ba2` 后，首场景九格全部完成。第一次门禁执行在 seed11 原臂处停止：行为行仅 `config_hash` 与 ROUND37 不同。停止后没有再启动环境。

只读核查了六个原臂/规则对照（三个 original greedy、immediate、late_147、always_hold）：

- 排除 `config_hash` 后，行内模型身份、任务/通信行为、字节流、奖励流、`behaviour_sha256` 等字段全部相同；
- 从生命周期首行完整读取 `RunConfiguration` 和场景详情，排除 `output`、`comm_transcript`、`decision_trace`、`reservation_log` 后，配置与场景逐字段相同；
- `run_signature` 的 arm、checkpoint、seed、stream、schema 等行为身份相同；记录的 Git commit 因 ROUND39 冻结提交不同，且 17 个 ROUND37 冻结源文件哈希全部匹配；
- 原始 `config_hash` 差异来自每轮不同的 `comm_transcript` 输出路径。`integrated_sar/run_signature.py` 的哈希函数排除了 `output`，但没有排除 `comm_transcript`；它属于本轮输出路径元数据。

第一次门禁还把生命周期首条 `run_config` 记录纳入执行前缀比较，该记录包含输出路径和 Git commit。勘误后的前缀比较仅覆盖实际任务/生命周期事件，配置身份另按上述字段独立比较。

经预先授权的“排除本轮路径等非行为元数据”规则进行门禁复核后，六个历史对照全匹配；首发、预算顺序、权重不变、refresh 抑制和 s22 负对照门禁均通过。复核是只读的，环境启动仍为 9，技术失败与重试均为 0。未修改 `r39_collector.py`、`v2_run.py` 或生产目录中的行为代码；改动仅用于门禁比较和接续调度。

更新后的门禁结果见 `FIRST_SCENE_GATE.json`。剩余 135 格由独立接续脚本按同一已冻结任务清单和已有计数继续。
"""
    (HERE / "GATE_COMPARATOR_AMENDMENT.md").write_text(amendment, encoding="utf-8")
    print(json.dumps({
        "gate_passed": True,
        "baseline_arms_matching": sorted(gate["round37_reproduction"]),
        "new_environment_starts_during_recheck": 0,
        "total_starts": starts_after,
        "s11_bytes_reduced": gate["high_refresh_seed_gates"]["11"]["bytes_reduced"],
        "s33_bytes_reduced": gate["high_refresh_seed_gates"]["33"]["bytes_reduced"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
