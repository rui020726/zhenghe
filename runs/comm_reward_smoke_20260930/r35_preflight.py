"""ROUND35 启动前门禁（0 环境启动、0 环境构造、0 step、0 optimizer.step）。

只有全部 P1..P12 为真，才允许 `r35_launch.py` 启动真实 runner。任何一项不过：不启动、报告。

P5 是关键前提：本轮的"被动性直接比对"只有在 smoke 的第 0..3 集确实从**同一份登记初值**出发
才成立，所以先证明 fresh_controller(11) 的参数哈希与归档 ROUND32 更新前参数、与归档启动前配对件
（从 git 取原件）、与 ROUND24 ep0 里程碑恢复后的参数**四者相同**。

P9 把 ROUND34 的两套测试在有真实 torch 的解释器下重跑一遍：上一轮它们是在 torch stub 下通过的。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import torch

import r35_common as smoke
import v2_trainer
from v2_controller import fresh_controller, parameter_identity

RESULTS: dict[str, Any] = {}
PROBLEMS: list[str] = []


def gate(name: str, payload: dict[str, Any]) -> None:
    payload["passed"] = bool(payload["passed"])
    RESULTS[name] = payload
    if not payload["passed"]:
        PROBLEMS.append(f"{name}: {payload.get('failure', 'gate failed')}")


def p1_anchors() -> None:
    anchors = json.loads((smoke.HERE / "ANCHORS_AND_ENVIRONMENT.json").read_text("utf-8"))
    head = smoke.git("rev-parse", "HEAD").strip()
    branch = smoke.git("rev-parse", "--abbrev-ref", "HEAD").strip()
    ancestor = subprocess.run(("git", "merge-base", "--is-ancestor",
                               "c7d47d25e638cbe3728cf36e3a22b9cfc563a353", "HEAD"),
                              cwd=str(smoke.ROOT), capture_output=True).returncode == 0
    anchor_head = anchors["repository"]["head"]
    round35_only = [line for line in smoke.git("log", "--format=%h %s",
                                               f"{anchor_head}..HEAD").splitlines()
                    if "round-35" not in line]
    now = {path.name: smoke.sha256_file(path) for path in sorted(smoke.ROUND34.iterdir())
           if path.is_file()}
    recorded = {Path(name).name: value for name, value in
                anchors["round34_file_sha256"].items()}
    same = {name: value == now.get(name) for name, value in recorded.items()}
    gate("P1_anchors", {
        "branch": branch, "head": head, "head_recorded_at_the_opening_check": anchor_head,
        "branch_as_authorised": branch == "fix/comm-task-reward-event-timing",
        "round33_baseline_is_ancestor_of_head": ancestor,
        "opening_head_is_ancestor_of_head": subprocess.run(
            ("git", "merge-base", "--is-ancestor", anchor_head, "HEAD"),
            cwd=str(smoke.ROOT), capture_output=True).returncode == 0,
        "commits_since_the_opening_check_not_round35": round35_only,
        "round34_files_unchanged_since_the_opening_check": all(same.values()),
        "differing_round34_files": sorted(name for name, ok in same.items() if not ok),
        "passed": (branch == "fix/comm-task-reward-event-timing" and ancestor
                   and all(same.values()) and not round35_only)})


def p2_environment() -> None:
    gate("P2_environment", {
        "executable": sys.executable, "python": sys.version.split()[0],
        "torch": torch.__version__, "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "virtualenv_created": False, "dependencies_installed": False,
        "round34_pytorch_exemption_withdrawn": True,
        "passed": torch.__version__.startswith("2.")})


def p3_versions() -> None:
    state = v2_trainer.fresh_training_state(smoke.SMOKE_SEED)
    want_training = "constrained-cost-v2/published-completion-event-at-relay-tick/2"
    want_timing = "published-completion-event-at-relay-tick/2"
    gate("P3_training_versions", {
        "trainer_training_version": v2_trainer.TRAINING_VERSION,
        "trainer_reward_timing_version": v2_trainer.REWARD_TIMING_VERSION,
        "fresh_state_training_version": state["training_version"],
        "fresh_state_reward_timing_version": state["reward_timing_version"],
        "required_by_the_contract": {"training_version": want_training,
                                     "reward_timing_version": want_timing},
        "passed": (v2_trainer.TRAINING_VERSION == state["training_version"] == want_training
                   and v2_trainer.REWARD_TIMING_VERSION == state["reward_timing_version"]
                   == want_timing)})


def p4_identity() -> None:
    anchors = json.loads((smoke.HERE / "ANCHORS_AND_ENVIRONMENT.json").read_text("utf-8"))
    identity = smoke.code_identity()
    same = {name: identity["stage_c_harness"].get(name) == digest
            for name, digest in anchors["stage_c_modules_read_for_this_round_sha256"].items()}
    manifest = json.loads((smoke.STAGE_C / "CONFIG_AND_MODEL_MANIFEST.json").read_text("utf-8"))
    production_unchanged = manifest["code_identity"]["production"] == identity["production"]
    dirty = [line for line in smoke.git("status", "--porcelain",
                                         "runs/comm_cost_v2_rerun_20260928").splitlines()
             if line.strip() and not line.startswith("??")]
    other_session_untracked = [line for line in smoke.git(
        "status", "--porcelain", "runs/comm_cost_v2_rerun_20260928").splitlines()
        if line.startswith("??")]
    gate("P4_code_identity", {
        "stage_c_modules_unchanged_since_the_opening_check": all(same.values()),
        "differing": sorted(name for name, ok in same.items() if not ok),
        "production_files_unchanged_since_the_round24_freeze": production_unchanged,
        "tracked_modifications_under_stage_c_directory": dirty,
        "untracked_files_from_other_sessions_left_untouched": other_session_untracked,
        "round34_deliverables_hashed": len(identity["round34_deliverables"]),
        "round35_scripts_hashed": sorted(identity["round35_scripts"]),
        "passed": all(same.values()) and production_unchanged and not dirty})


def p5_initial_weights() -> None:
    now = parameter_identity(fresh_controller(smoke.SMOKE_SEED))["parameter_sha256"]
    rows = sorted((smoke.ROUND32 / "rows").glob("r32_train_ALL_ROWS_s11_e0_s*_a0.json"))
    archived_row = json.loads(rows[0].read_text("utf-8"))
    batch = json.loads((smoke.ROUND32 / "checkpoints" /
                        "r32_batch_ALL_ROWS_s11_b0.json").read_text("utf-8"))
    prelaunch_text = smoke.git("show",
                               "df176bdf8c3c265d8109494446073b23aee384a6:runs/"
                               "comm_actor_reduction_ablation_20261001/"
                               "INIT_PAIRING_ALL_ROWS_s11.json")
    prelaunch = json.loads(prelaunch_text)["controller_parameters"]["parameter_sha256"] \
        if prelaunch_text.strip() else None
    milestone = smoke.STAGE_C / "checkpoints" / "v2_train_s11_ep0.pt"
    restored_hash = None
    if milestone.exists():
        payload = torch.load(milestone, map_location="cpu", weights_only=False)
        restored = v2_trainer._restore_controller(payload, payload["training_state"])
        restored_hash = parameter_identity(restored)["parameter_sha256"]
    four = {"fresh_controller_this_machine": now,
            "archived_round32_batch0_parameters_before":
                batch["controller_parameters_before"]["parameter_sha256"],
            "archived_round32_prelaunch_pairing_from_git": prelaunch,
            "round24_ep0_milestone_restored_by_the_real_loader": restored_hash}
    equal = len({value for value in four.values()}) == 1 and all(
        value is not None for value in four.values())
    gate("P5_initial_weights", {
        "parameter_sha256_per_source": four,
        "all_four_identical": equal,
        "why_this_matters": "the passivity claim is a byte comparison against the archived "
                            "legacy run, and it is only legitimate if episode 0 starts from the "
                            "same registered weights that run used",
        "passed": equal, "failure": "initial weights differ from the archived run" if not equal
        else None})


def p6_reference_cache() -> None:
    """The paired references exist for all eight scenarios and are the same runs across rounds.

    `config_hash` covers the run's own output paths, so it is round-local and must not be used
    for a cross-round identity claim; the cross-round identity is carried by the behaviour hash
    (derived from the lifecycle records), the declared transport parameters, the emission
    sequence and the relay counters, all of which are compared here for real.
    """
    cache = smoke.reference_cache()
    order = smoke.v2_common.scenario_order()[:smoke.SMOKE_EPISODES]
    covered = sorted(scenario for scenario in order if scenario in cache["by_scenario"])
    round32 = {}
    for path in sorted((smoke.ROUND32 / "rows").glob("r32_reference_s*_a1.json")):
        row = json.loads(path.read_text("utf-8"))
        round32[int(row["scenario_seed"])] = row
    per_scenario = {}
    for scenario in order:
        ours, theirs = cache["by_scenario"].get(scenario), round32.get(scenario)
        if ours is None or theirs is None:
            per_scenario[scenario] = {"present_in_both_caches": False}
            continue
        per_scenario[scenario] = {
            "present_in_both_caches": True,
            "on_time_round24": int(ours["on_time"]), "on_time_round32": int(theirs["on_time"]),
            "behaviour_sha256_equal": ours["behaviour_sha256"] == theirs["behaviour_sha256"],
            "transport_parameters_equal": ours["transport_parameters"]
            == theirs["transport_parameters"],
            "emission_sequence_equal": ours["emission_sequence"] == theirs["emission_sequence"],
            "relay_counters_equal": ours["relay_counters"] == theirs["relay_counters"],
            "config_hash_equal": ours["config_hash"] == theirs["config_hash"]}
    identity_fields = all(
        value.get("behaviour_sha256_equal") and value.get("transport_parameters_equal")
        and value.get("emission_sequence_equal") and value.get("relay_counters_equal")
        and value.get("on_time_round24") == value.get("on_time_round32")
        for value in per_scenario.values())
    gate("P6_reference_cache", {
        "cached_references": cache["count"], "files_read_only": len(cache["files_read_only"]),
        "covers_the_eight_smoke_scenarios": covered == sorted(order),
        "per_scenario_cross_round_identity": per_scenario,
        "identity_fields_all_equal": identity_fields,
        "config_hash_is_round_local": not all(value.get("config_hash_equal")
                                             for value in per_scenario.values()),
        "why_that_is_expected": "config_hash covers this round's own lifecycle and transcript "
                                "paths, so two rounds running the same scenario differ in it by "
                                "construction; the passivity comparison therefore uses the "
                                "behaviour hash, the transport declaration, the emission sequence "
                                "and the relay counters",
        "new_reference_launches_required": 0,
        "passed": covered == sorted(order) and identity_fields})


def p7_order_and_predictions() -> None:
    expectation = smoke.archived_expectation()
    per = expectation["episodes"]
    gate("P7_order_and_registered_predictions", {
        "scenario_matches_the_frozen_order": {key: value["scenario_matches_order"]
                                              for key, value in per.items()},
        "legacy_keys": {key: value["legacy_task_stream_keys"] for key, value in per.items()},
        "predicted_fixed_keys": {key: value["predicted_fixed_task_stream_keys"]
                                 for key, value in per.items()},
        "keys_whose_owner_row_the_fix_moves":
            {key: value["legacy_keys_whose_owner_row_the_fix_moves"]
             for key, value in per.items()},
        "predictions_are_falsifiable": True,
        "note": "registered before the launch: the fixed run must write exactly the predicted "
                "keys in episodes 0..3, and must not move the credit to the next action row",
        "passed": all(value["scenario_matches_order"] for value in per.values())
                  and all(value["predicted_fixed_task_stream_keys"] for value in per.values())})


def p8_limits() -> None:
    paths = smoke.configure("preflight")
    cap = smoke.v2_common.LAUNCH_CAP
    gate("P8_limits_and_paths", {
        "OUTPUT_DIR": str(paths["OUTPUT_DIR"]), "ROWS": str(paths["ROWS"]),
        "rows_inside_this_round": str(paths["ROWS"]).startswith(str(smoke.HERE)),
        "nothing_bound_outside_this_round": all(
            not str(value).startswith(str(smoke.STAGE_C)) for value in paths.values()),
        "launch_cap_seen_by_the_real_ledger": cap,
        "authorised": {"invocations": smoke.MAX_LAUNCH_INVOCATIONS,
                       "episode_constructions": smoke.MAX_EPISODE_CONSTRUCTIONS,
                       "optimizer_steps": smoke.MAX_OPTIMIZER_STEPS,
                       "seed": smoke.SMOKE_SEED, "episodes": smoke.SMOKE_EPISODES},
        "passed": cap == smoke.MAX_EPISODE_CONSTRUCTIONS
                  and str(paths["ROWS"]).startswith(str(smoke.HERE))})


def p9_round34_tests() -> None:
    completed = subprocess.run((sys.executable, "-m", "unittest", "discover", "-s",
                                str(smoke.ROUND34), "-p", "test_*.py"),
                               cwd=str(smoke.ROOT), capture_output=True, text=True,
                               encoding="utf-8")
    lines = [line for line in (completed.stderr + completed.stdout).splitlines()
             if line.startswith(("Ran ", "OK", "FAILED", "ERROR"))]
    gate("P9_round34_tests_rerun_with_real_torch", {
        "returncode": completed.returncode, "summary": lines,
        "environments_constructed": 0,
        "note": "ROUND34 executed these suites with a torch stub; the stub branch is not taken "
                "on this machine because torch imports",
        "passed": completed.returncode == 0})


def p10_frozen_weights() -> None:
    batch = json.loads((smoke.ROUND32 / "checkpoints" /
                        "r32_batch_ALL_ROWS_s11_b0.json").read_text("utf-8"))
    current = {str(path.relative_to(smoke.ROOT)): smoke.sha256_file(path)
               for path in smoke.v2_common.CHECKPOINT_FILES}
    gate("P10_frozen_model_weights", {
        "archived_batch_record_declares": batch["frozen_checkpoints"],
        "current_files": current, "identical": batch["frozen_checkpoints"] == current,
        "passed": batch["frozen_checkpoints"] == current})


def p11_workspace() -> None:
    status = [line for line in smoke.git("status", "--porcelain").splitlines() if line.strip()]
    touching_old = [line for line in status if not line.startswith("??")
                    or "comm_reward_smoke" not in line]
    gate("P11_workspace", {
        "entries": status,
        "tracked_files_changed": [line for line in status if not line.startswith("??")],
        "old_round_artefacts_modified": [line for line in touching_old
                                         if line.startswith((" M", "M ", " D", "D "))],
        "passed": not [line for line in status
                       if line.startswith((" M", "M ", " D", "D "))
                       and "comm_reward_smoke" not in line]})


def p12_scope_statement() -> None:
    gate("P12_scope_and_case_reachability", {
        "single_question": "does the ROUND34 fix behave as the contract says on real runs, and "
                           "can the real checkpoint chain save and restore it?",
        "expected_reachable_contract_cases": list(smoke.EXPECTED_REACHABLE_CASES),
        "cases_not_reachable_in_this_scenario_block": list(smoke.UNREACHABLE_IN_THIS_BLOCK),
        "coverage_evidence": json.loads((smoke.HERE / "PRE_SMOKE_CASE_COVERAGE.json")
                                        .read_text("utf-8"))[
            "case_availability_in_archived_legacy_data"],
        "forbidden": ["hyper-parameter, network, objective, reward value or cost path change",
                      "grid search or result-selected extra training",
                      "hot start or silent continuation of any old checkpoint",
                      "rewriting ROUND32/33/34 artefacts", "pushing to GitHub",
                      "presenting smoke numbers as learning-quality evidence"],
        "passed": True})


def main() -> int:
    for step in (p1_anchors, p2_environment, p3_versions, p4_identity, p5_initial_weights,
                 p6_reference_cache, p7_order_and_predictions, p8_limits, p9_round34_tests,
                 p10_frozen_weights, p11_workspace, p12_scope_statement):
        step()
    payload = {
        "schema": "round35-preflight/1",
        "round": 35,
        "environment_launches_used_by_this_script": 0,
        "environments_constructed": 0,
        "environments_stepped": 0,
        "optimizer_steps_performed": 0,
        "gates": RESULTS,
        "problems": PROBLEMS,
        "clean": not PROBLEMS,
        "authorised_limits": {"launch_invocations": smoke.MAX_LAUNCH_INVOCATIONS,
                              "episode_constructions": smoke.MAX_EPISODE_CONSTRUCTIONS,
                              "optimizer_steps": smoke.MAX_OPTIMIZER_STEPS,
                              "wall_clock_per_launch_hours": 2,
                              "seed": smoke.SMOKE_SEED, "episodes": smoke.SMOKE_EPISODES},
        "decision": "launches may start only while clean is true",
    }
    out = smoke.write_json("ROUND35_PREFLIGHT.json", payload, refresh=True)
    print(json.dumps({"clean": payload["clean"], "problems": PROBLEMS,
                      "gates": {name: value["passed"] for name, value in RESULTS.items()}},
                     ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
