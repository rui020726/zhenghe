"""Stage B 启动前冻结：CONFIG_AND_MODEL_MANIFEST.json + START_LEDGER.json（计划块）。

运行：/d/project_env/python.exe runs/comm_cost_stage_b_20260927/sb_freeze.py

只读仓库与磁盘；不启动任何环境。冻结后不再改 harness/生产依赖文件；纯报告修改另行提交并
登记，不伪装成相同代码身份。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sb_common as common

PRODUCTION_TREES = ("integrated_sar", "task_allocation", "piloc")
BRANCH_START = "5d0d71f4f779fb36982c468f34e7d801f1bc5de8"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=common.ROOT, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def production_diff() -> dict[str, Any]:
    diff = git("diff", "--name-only", BRANCH_START, "--", *PRODUCTION_TREES)
    files = [line for line in diff.splitlines() if line]
    return {"branch_start": BRANCH_START, "trees": list(PRODUCTION_TREES),
            "changed_files": files, "clean": not files}


def per_seed_configuration() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for seed in common.SEEDS:
        archive = common.d1_archive_path(seed)
        import sp_common as sc

        archived = sc.archived_configuration(archive)
        paths_payload = {}
        for path_name in common.PATHS:
            fields, substituted = common.stage_b_fields(path_name, seed)
            paths_payload[path_name] = {
                "comm_policy": fields["comm_policy"],
                "comm_budget_bytes": fields["comm_budget_bytes"],
                "comm_timing_mode": fields["comm_timing_mode"],
                "substituted_fields": substituted,
            }
        payload[str(seed)] = {
            "archived_a3_run_config_sha256": common.sha256_file(archive),
            "archived_scenario": sc.archived_scenario(archive),
            "inherited_behaviour_fields": {
                key: archived[key] for key in (
                    "comm_delivery_probability", "comm_delay_ticks", "comm_range",
                    "gossip_interval", "comm_ack", "comm_subscribers", "steps", "num_goals",
                    "deadline_tightness", "search_action_policy", "use_coordinator")
                if key in archived},
            "per_path": paths_payload,
        }
    return payload


def main() -> int:
    identity = common.code_identity()
    manifest = {
        "round": "21", "phase": "Stage B pre-launch freeze",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_at_freeze": git("rev-parse", "HEAD"),
        "branch_start": BRANCH_START,
        "production_diff": production_diff(),
        "code_identity": identity,
        "wiring_constants": {
            "B0_test_bytes": common.B0_TEST, "lambda_test": common.LAMBDA_TEST,
            "delta_test": common.DELTA_TEST, "eta_test": common.ETA_TEST,
            "gamma": common.GAMMA, "gae_lambda": common.GAE_LAMBDA,
            "actor_init_seed": common.ACTOR_INIT_SEED, "sampling_seed": common.SAMPLING_SEED,
            "log_prob_tolerance": common.LOG_PROB_TOLERANCE,
            "note": "wiring examples only; not approved training hyper-parameters or a task "
                    "loss tolerance",
        },
        "paths": {path: {"policy": {"R_send": "immediate", "I_send": "learned",
                                    "R_hold": "always_hold", "I_hold": "learned",
                                    "P_init": "learned", "P_repeat": "learned"}[path],
                            "decision_maker": {
                                "R_send": "native ImmediatePolicy",
                                "I_send": "LearnedSendPolicy(immediate_equivalent_fn)",
                                "R_hold": "native AlwaysHoldPolicy",
                                "I_hold": "LearnedSendPolicy(hold_equivalent_fn)",
                                "P_init": "StageBCollector(StageBController(ACTOR_INIT_SEED))",
                                "P_repeat": "same as P_init"}[path]}
                   for path in common.PATHS},
        "per_seed": per_seed_configuration(),
        "ledger_plan": {
            "normal_launches": common.NORMAL_LAUNCHES, "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "order": [f"{path} s{seed}" for seed in common.SEEDS for path in common.PATHS],
            "execution_rule": "seed 1100000's six paths run first; the remaining two seeds run "
                              "only after every wiring gate of the first seed passes",
            "counting_rule": "every environment start, technical failure, reference arm and "
                             "smoke counts; pure math unit tests do not",
        },
        "synthetic_tests": {"stage_b_suite": "test_stage_b_math.py",
                            "stage_a_suite": "runs/comm_cost_controller_stage_a_20260927/"
                                             "test_cost_math.py",
                            "launches": 0},
    }
    common.write_json("CONFIG_AND_MODEL_MANIFEST.json", manifest)

    ledger = {"plan": manifest["ledger_plan"], "live": {"used": common.ledger_total()}}
    common.write_json("START_LEDGER.json", ledger)
    print(json.dumps({"frozen_head": manifest["head_at_freeze"],
                      "production_clean": manifest["production_diff"]["clean"],
                      "ledger_used": ledger["live"]["used"]}, ensure_ascii=False))
    return 0 if manifest["production_diff"]["clean"] and ledger["live"]["used"] == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
