"""ROUND25 首启前冻结：配置差异、三臂实现、统计方案、启动账本与执行头。

冻结并本地提交（执行依据 §四）：本执行依据、完整执行头、模型 hash、配置允许差异、臂实现、
场景列表、代码依赖身份、启动账本、汇总器和统计方案——全部在首次环境启动前完成。
零环境启动：只读仓库与已核验件。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import r25_common as common
import r25_stats as stats


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(common.ROOT), capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def configuration_diff() -> dict[str, Any]:
    """The research configuration is inherited field by field; only registered differences stay."""
    sys.path.insert(0, str(common.ROUND24_DIR))
    import v2_freeze  # noqa: E402  (read-only: the same function ROUND24 froze with)

    recorded = json.loads((common.ROUND24_DIR / "CONFIG_AND_MODEL_MANIFEST.json").read_text(
        encoding="utf-8"))
    recorded_fields = recorded["inherited_behaviour_fields"]
    fields, substituted = common.stage_b_inherited_config(int(common.NEW_SCENARIOS[0]))
    identity = {"output", "comm_transcript", "info_probe", "search_probe", "decision_trace",
                "record_stream", "replay_stream", "seed", "comm_policy", "comm_policy_params"}
    live_fields = {key: value for key, value in fields.items() if key not in identity}
    field_diffs = [key for key in sorted(set(live_fields) | set(recorded_fields))
                   if live_fields.get(key) != recorded_fields.get(key)]
    v2_fields = v2_freeze.behaviour_fields()
    v2_field_diffs = [key for key in sorted(set(v2_fields) | set(recorded_fields))
                      if v2_fields.get(key) != recorded_fields.get(key)]
    config_hash = v2_freeze.configuration_hash(v2_fields)
    payload = {
        "round": "25",
        "section": "config diff vs the verified ROUND24 freeze (execution basis §3)",
        "behaviour_fields": {
            "recorded_freeze": recorded_fields,
            "live_inherited": live_fields,
            "differences": field_diffs,
            "identical": not field_diffs,
            "source_rule": "stage_b_inherited_config(scenario_seed): the archived D1 A3 record of "
                           "the configuration source (1100000) plus the Stage B frozen overrides, "
                           "only the scenario seed changed",
        },
        "round24_harness_function_agrees": {"differences": v2_field_diffs,
                                            "identical": not v2_field_diffs},
        "configuration_hash": {
            "function": "v2_freeze.configuration_hash(behaviour_fields()) -- the same function "
                        "ROUND24 froze with",
            "round24_recorded": recorded["configuration_hash"],
            "round25_recomputed": config_hash,
            "unchanged": config_hash == recorded["configuration_hash"],
            "inert_hyperparameters": "training hyperparameters inside the hash payload are inert "
                                     "this round: 0 training episodes, 0 optimizer steps",
        },
        "allowed_differences": [
            "output directory and artefact naming (r25_ prefix under "
            "runs/comm_cost_greedy_replication_20260928) -- registered, behaviour-neutral",
            "arm set: three fixed arms instead of the ROUND24 training/eval/rule blocks",
            "no training, no lambda bookkeeping, no data-order RNG use this round",
        ],
        "no_other_differences_allowed": True,
        "stage_b_override_table_matches_round24": common.STAGE_B_INHERITED_OVERRIDES
        == {key: recorded_fields[key] for key in common.STAGE_B_INHERITED_OVERRIDES
            if key in recorded_fields},
    }
    if field_diffs or v2_field_diffs or config_hash != recorded["configuration_hash"]:
        raise SystemExit(f"configuration drift from the ROUND24 freeze: {field_diffs}")
    return payload


def arm_freeze() -> dict[str, Any]:
    """The three fixed arms with their implementation identity."""
    identity = common.code_identity()
    learned_hash = common.sha256_file(common.LEARNED_CHECKPOINT)
    return {
        "round": "25",
        "section": "arm implementation freeze (execution basis §2)",
        "arms": {
            "learned_s33_greedy": {
                "behaviour": "ROUND24 seed33 ep128 final checkpoint, original greedy decode",
                "checkpoint": common._rel(common.LEARNED_CHECKPOINT),
                "checkpoint_sha256": learned_hash,
                "training_seed": 33,
                "episodes": 128,
                "decode": "SEND iff p(SEND) >= p(HOLD), ties to SEND",
                "collector": "budget_masked.BudgetMaskedCollector(forced=False, mask=False, "
                             "greedy=True, budget_mask=True) -- the same class and parameters as "
                             "the ROUND24 final_greedy block",
                "action_rng": f"sha256('{common.EVAL_RNG_LABEL}:33:<scene>:greedy')[:16], "
                              "inherited verbatim from the ROUND24 final_greedy block",
                "source_modules": ["runs/comm_budget_action_mask_fix_20260922/budget_masked.py",
                                   "runs/comm_cost_v2_rerun_20260928/v2_trainer.py",
                                   "runs/comm_cost_v2_rerun_20260928/v2_controller.py"],
                "selection_registration": "post-hoc fixed candidate chosen after seeing the 16 "
                                          "ROUND24 development scenes; not a randomly drawn "
                                          "training seed; generalisation can be tested, the "
                                          "selection fact cannot be removed",
            },
            "immediate": {
                "behaviour": "original reference implementation",
                "policy": "immediate",
                "implementation": "driver-native ImmediatePolicy (no injection)",
                "source_modules": ["integrated_sar/comm_policy.py"],
            },
            "late_discovery_gate_147": {
                "behaviour": "original threshold 147, original boundary and implementation",
                "policy": "late_discovery_gate",
                "params": {"cut_steps": 147.0},
                "implementation": "v2_rules.LateDiscoveryGatePolicy, strict greater-than on "
                                  "ready_step (the ROUND24 harness file, imported not rewritten)",
                "source_modules": ["runs/comm_cost_v2_rerun_20260928/v2_rules.py"],
            },
        },
        "not_evaluated_this_round": ["seed11", "seed22", "sampled mode", "always_hold",
                                     "periodic_20", "random_send", "event", "ep64 checkpoints"],
        "not_evaluated_note": "their ROUND24 results stay in the historical tables; no new "
                              "environment launches are made for them and no candidate swap "
                              "happens after seeing new results",
        "no_training": "0 training episodes, 0 optimizer steps, no distillation, no behaviour "
                       "cloning, no lambda/entropy/reward/network changes",
        "code_identity_freeze_sha256": identity["freeze_sha256"],
    }


def statistics_plan() -> dict[str, Any]:
    return {
        "round": "25",
        "section": "pre-registered statistics plan (frozen before the first launch)",
        "candidate_arm": common.CANDIDATE_ARM,
        "primary_cost_comparison": common.PRIMARY_COST_COMPARATOR,
        "task_comparators": list(common.TASK_COMPARATORS),
        "metrics": {
            "dY": "candidate on-time tasks minus comparator on-time tasks per scene (positive is better)",
            "dC": "candidate spent bytes minus comparator spent bytes per scene (negative is cheaper)",
        },
        "resampling": {
            "unit": "the 64 scenes are the paired sampling unit; 192 launches are not 192 "
                    "independent scenes",
            "generator": "numpy.random.Generator(numpy.random.PCG64(20260928))",
            "matrix": "integers(0, 64, size=(10000, 64)) generated exactly once",
            "shared_by": "every task and cost comparison of this round",
            "interval": "95% percentile bootstrap, quantiles with method='linear'",
            "matrix_sha256": stats.matrix_digest(stats.resample_matrix()),
        },
        "predicates": stats.PREDICATES,
        "verdict_fields": {
            "primary_cost_evidence": "lower_cost_supported only when the primary dC 95% CI upper "
                                     "bound < 0; otherwise lower_cost_not_established",
            "observed_task_count_status": "no_observed_count_loss only when dY >= 0 in every "
                                          "scene against BOTH comparators",
            "constraint_guarantee": "always not_established",
            "training_seed_stability": "always not_established",
        },
        "zero_event_rule": "when zero net task degradations are observed against a comparator, "
                           "attach 1-0.05^(1/64) (~4.57%) as the single-sided 95% upper bound on "
                           "the per-scenario probability of a degradation event under an i.i.d. "
                           "assumption; it is not an expected task-loss bound and not a "
                           "deployment guarantee",
        "stopping": "none of these criteria stops the batch mid-run; only technical health and "
                    "the launch quota decide stopping",
        "single_reader": "tables, judgement and verdict read the one machine-readable statistics "
                         "artefact (PAIRED_COMPARISON.json); no second sampling is performed",
    }


def start_ledger() -> dict[str, Any]:
    return {
        "round": "25",
        "plan": {
            "normal_launches": common.NORMAL_LAUNCHES,
            "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "cells": {"scenes": len(common.NEW_SCENARIOS), "arms": len(common.ARM_ORDER),
                      "total": len(common.all_cells()),
                      "gate": len(common.GATE_SCENARIOS) * len(common.ARM_ORDER)},
            "counting_rule": "every launch is booked before the driver is constructed; failures, "
                             "retries, smoke and reference launches all count; the per-process "
                             "ledger sum and the atomic claim-directory count must agree",
            "independent_budget": "does not borrow ROUND24's unused 28 launches",
        },
        "live": {"used": common.launch_count(), "per_process_total": common.ledger_total(),
                 "note": "refreshed as the batch runs; the plan block is frozen"},
    }


def execution_head() -> dict[str, Any]:
    return {
        "branch": "exp/comm-cost-greedy-replication",
        "head": _git("rev-parse", "HEAD"),
        "head_short": _git("rev-parse", "--short", "HEAD"),
        "branch_start": "f920c555b7bfe7d01296b9099635f673c2e5dcb6",
        "branch_start_note": "the verified ROUND24 execution head this branch was created from",
        "round24_execution_head": "f920c555b7bfe7d01296b9099635f673c2e5dcb6",
        "round24_branch": "exp/comm-cost-v2-rerun",
        "execution_basis": {
            "file": "runs/comm_cost_greedy_replication_20260928/EXECUTION_BASIS.md",
            "sha256": "cada6bca546bb2efef778b75c9e2ccafab0e61615b9933b0301bc224f8d817f6",
        },
        "publication": {"remote_verification": "NOT_PERFORMED",
                        "publication_status": "PENDING_NETWORK"},
    }


def freeze_all() -> dict[str, Any]:
    config = configuration_diff()
    common.write_json("CONFIG_DIFF.json", config)
    common.write_json("ARM_FREEZE.json", arm_freeze())
    common.write_json("STATISTICS_PLAN.json", statistics_plan())
    ledger = common.write_json("START_LEDGER.json", start_ledger())
    return {"config_diff": config["behaviour_fields"]["identical"],
            "arm_freeze": "written", "statistics_plan": "written",
            "start_ledger": ledger.name}


def update_start_identity() -> Path:
    """Record the lift of the evaluation block under the in-session execution basis."""
    path = common.OUTPUT_DIR / "START_IDENTITY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["round25_branch"] = {
        "branch": "exp/comm-cost-greedy-replication",
        "head": _git("rev-parse", "HEAD"),
        "commits_ahead_of_round24_head": None,
        "history_preserved": True,
        "note": "the existing branch and all its commits are kept; no rebuild, no rollback, no "
                "re-import of the old spec",
    }
    payload["execution_basis"] = {
        "file": "runs/comm_cost_greedy_replication_20260928/EXECUTION_BASIS.md",
        "sha256": "cada6bca546bb2efef778b75c9e2ccafab0e61615b9933b0301bc224f8d817f6",
        "supersedes": "the old spec text and the supplementary authorisation file as the "
                      "execution basis for the remainder of this round",
        "old_spec_status": "MISMATCH retained; the old file is not edited and not claimed verified",
    }
    payload["evaluation_authorised"] = True
    payload["evaluation_authorisation_basis"] = (
        "the user's in-session execution instruction (EXECUTION_BASIS.md), which lifts the "
        "document-delivery block and authorises the 192+16 launch evaluation under the fixed "
        "three-arm panel; publication stays blocked and is not a precondition")
    payload["evaluation_block_reason"] = None
    payload["updates"] = payload.get("updates", []) + [{
        "note": "evaluation block lifted by the in-session execution instruction; the spec "
                "offline hash check stays MISMATCH and is not claimed as verified; remote "
                "verification stays NOT_PERFORMED and publication PENDING_NETWORK; no network "
                "retries were made in this session",
        "old_spec_verification": "unchanged (MISMATCH)",
    }]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    if argv and argv[0] == "identity":
        update_start_identity()
        print("START_IDENTITY.json updated")
        return 0
    payload = freeze_all()
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
