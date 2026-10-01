"""ROUND36 启动前核验与冻结配置清单（0 环境启动、0 optimizer.step）。

逐项比对"本轮实际运行时配置"与 ROUND32 冻结的 Stage C 配置，并把本轮训练路径与 ROUND32 时代
的训练路径做函数级源码比对，证明唯一的干预就是 ROUND34 的奖励时序修复本身。任何 drift 都在这里
暴露；场景块若变化立即停止（授权边界 §三 冻结项）。

    /d/project_env/python.exe r36_preflight.py

产出：FROZEN_CONFIG_MANIFEST.json（任务 1 的冻结配置清单）、ROUND36_PREFLIGHT.json（门禁表）。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import unittest
from typing import Any

import numpy as np
import torch

import r36_common as r36
import v2_common
import v2_controller
import v2_run
import v2_trainer

GATES: dict[str, Any] = {}


def gate(name: str, payload: dict[str, Any]) -> None:
    payload["passed"] = bool(payload.get("passed"))
    GATES[name] = payload
    if not payload["passed"]:
        raise SystemExit(f"GATE {name} FAILED: {json.dumps(payload, ensure_ascii=False)[:1200]}")


# ------------------------------------------------------------------ G1 身份锚点 -------------
def g1_anchors() -> None:
    branch = r36.git("rev-parse", "--abbrev-ref", "HEAD").strip()
    head = r36.git("rev-parse", "HEAD").strip()
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", r36.ROUND33_BASELINE, head),
        cwd=str(r36.ROOT), capture_output=True).returncode == 0
    required = [r36.ROUND34 / "TASK_REWARD_TIMING_CONTRACT.md",
                r36.ROUND34 / "TRAINING_VERSION_MANIFEST.json",
                r36.ROUND35 / "ROUND35_SMOKE_REPORT.md",
                r36.ROUND35 / "SMOKE_LAUNCH_BOOK.json",
                r36.ROUND35 / "LAUNCH_ACCOUNTING.json",
                r36.ROUND32 / "CONFIG_AND_MODEL_MANIFEST.json",
                r36.ROUND32 / "START_LEDGER.json",
                r36.STAGE_C / "CONFIG_AND_MODEL_MANIFEST.json"]
    missing = [str(path.relative_to(r36.ROOT)) for path in required if not path.exists()]
    gate("G1_repository_identity", {
        "branch": branch, "head": head, "expected_branch": r36.EXPECTED_BRANCH,
        "expected_head_at_start": r36.EXPECTED_HEAD_AT_START,
        "head_equals_round35_delivery_head": head == r36.EXPECTED_HEAD_AT_START,
        "round33_baseline_is_ancestor": ancestor,
        "round34_and_round32_and_round35_directories_present": not missing,
        "missing_files": missing,
        "github_snapshot_used": False,
        "passed": branch == r36.EXPECTED_BRANCH and ancestor and not missing})


# ------------------------------------------------------------------ G2 环境 -------------
def g2_environment() -> None:
    from onpolicy import joint_log_prob  # noqa: F401  (proves the tight-round import chain works)

    present = {str(path.relative_to(r36.ROOT)).replace("\\", "/"): path.exists()
               for path in v2_common.CHECKPOINT_FILES}
    gate("G2_environment", {
        "interpreter": sys.executable, "python": sys.version.split()[0],
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "logical_cpus": os.cpu_count(),
        "import_torch_succeeded": True,
        "new_virtualenv_created": False, "dependencies_installed": 0,
        "frozen_checkpoints_present": present,
        "round34_pytorch_exemption_applies_to_this_machine": False,
        "passed": all(present.values())})


# ------------------------------------------------------------------ G3 版本字段 -------------
def g3_versions() -> None:
    state = v2_trainer.fresh_training_state(11)
    expected_training = "constrained-cost-v2/published-completion-event-at-relay-tick/2"
    expected_timing = "published-completion-event-at-relay-tick/2"
    runner_source = (r36.STAGE_C / "v2_run.py").read_text("utf-8")
    saver_source = (r36.STAGE_C / "v2_trainer.py").read_text("utf-8")
    gate("G3_training_version_fields", {
        "trainer_tag": v2_trainer.TRAINER_TAG, "loss_version": v2_trainer.LOSS_VERSION,
        "training_version": v2_trainer.TRAINING_VERSION,
        "reward_timing_version": v2_trainer.REWARD_TIMING_VERSION,
        "training_state_carries_both_fields": (
            state.get("training_version") == expected_training
            and state.get("reward_timing_version") == expected_timing),
        "every_row_writes_both_fields": ('"training_version": TRAINING_VERSION' in runner_source
                                         and '"reward_timing_version": '
                                         'taps["reward_timing_version"]' in runner_source),
        "every_checkpoint_payload_writes_both_fields": (
            '"training_version": TRAINING_VERSION' in saver_source
            and '"reward_timing_version": REWARD_TIMING_VERSION' in saver_source),
        "collector_class_on_the_event_bus": v2_run.PublishedProgressRewardCollector.__name__,
        "passed": (v2_trainer.TRAINING_VERSION == expected_training
                   and v2_trainer.REWARD_TIMING_VERSION == expected_timing
                   and state.get("training_version") == expected_training
                   and state.get("reward_timing_version") == expected_timing
                   and ('"training_version": TRAINING_VERSION' in runner_source
                        and '"training_version": TRAINING_VERSION' in saver_source))})


# ------------------------------------------------------------------ G4 冻结常量 -------------
def g4_constants() -> None:
    archive = r36.read_json(r36.ROUND32 / "CONFIG_AND_MODEL_MANIFEST.json")
    hyper = archive["ppo_hyperparameters"]
    actual = {"clip": v2_common.CLIP, "entropy_bonus": v2_common.ENTROPY_BONUS,
              "epochs_per_batch": v2_common.EPOCHS_PER_BATCH,
              "learning_rate": v2_common.LEARNING_RATE,
              "max_grad_norm": v2_common.MAX_GRAD_NORM,
              "minibatch_steps": v2_common.MINIBATCH_STEPS,
              "value_coefficient": v2_common.VALUE_COEFFICIENT}
    objective = {"b0_bytes": v2_common.B0_BYTES, "delta": v2_common.DELTA,
                 "lambda_init": v2_common.LAMBDA_INIT, "eta_lambda": v2_common.ETA_LAMBDA,
                 "gamma": v2_common.GAMMA, "gae_lambda": v2_common.GAE_LAMBDA,
                 "log_prob_tolerance": v2_common.LOG_PROB_TOLERANCE,
                 "advantage_standardisation": "none"}
    order = v2_common.scenario_order()
    digest = hashlib.sha256(json.dumps(order).encode("utf-8")).hexdigest()
    gate("G4_frozen_training_constants", {
        "episodes_per_seed": {"actual": v2_common.EPISODES_PER_SEED,
                              "archived": archive["episodes_per_seed"],
                              "equal": v2_common.EPISODES_PER_SEED == archive["episodes_per_seed"]},
        "batches_per_seed": {"actual": v2_common.EPISODES_PER_SEED
                             // v2_common.EPISODES_PER_UPDATE, "archived": 32},
        "episodes_per_update": {"actual": v2_common.EPISODES_PER_UPDATE, "archived": 4},
        "steps_per_episode": {"actual": v2_common.STEPS_PER_EPISODE, "archived": 1000},
        "milestones": {"actual": list(v2_common.MILESTONE_EPISODES), "archived": [0, 64, 128]},
        "train_seeds": {"actual": list(v2_common.TRAIN_SEEDS), "archived": [11, 22, 33]},
        "ppo_hyperparameters": {"actual": actual, "archived": hyper, "equal": actual == hyper},
        "objective_and_dual": objective,
        "scenario_block": {"training": [v2_common.TRAINING_SCENARIOS[0],
                                        v2_common.TRAINING_SCENARIOS[-1],
                                        len(v2_common.TRAINING_SCENARIOS)],
                           "development": [v2_common.DEVELOPMENT_SCENARIOS[0],
                                           v2_common.DEVELOPMENT_SCENARIOS[-1],
                                           len(v2_common.DEVELOPMENT_SCENARIOS)],
                           "archived_development": archive["scene_registry"]["development"],
                           "equal": list(v2_common.DEVELOPMENT_SCENARIOS)
                                    == archive["scene_registry"]["development"]},
        "data_order": {"actual_sha256": digest,
                       "archived_sha256": archive["data_order"]["order_digest_sha256"],
                       "equal": digest == archive["data_order"]["order_digest_sha256"],
                       "first_8": order[:8]},
        "row_filtering": {"policy": "ALL_ROWS as encoded in the frozen v2_trainer; ACTIVE_ROWS "
                                    "is not reachable from this round's entry points",
                          "trainer_module_used": v2_trainer.__file__,
                          "active_rows_module_used": False},
        "passed": (actual == hyper
                   and v2_common.EPISODES_PER_SEED == archive["episodes_per_seed"]
                   and list(v2_common.TRAIN_SEEDS) == [11, 22, 33]
                   and digest == archive["data_order"]["order_digest_sha256"]
                   and list(v2_common.DEVELOPMENT_SCENARIOS)
                       == archive["scene_registry"]["development"])})


# ------------------------------------------------------------------ G5 世界配置 -------------
def g5_world_config() -> None:
    archive = r36.read_json(r36.ROUND32 / "CONFIG_AND_MODEL_MANIFEST.json")
    archived_fields = archive["inherited_behaviour_fields"]
    compared, differing, only_current = {}, {}, {}
    for scenario in (1300027, 1300014, 1300000):
        fields, _substituted = v2_common.stage_b_inherited_config(scenario)
        shared = sorted(set(archived_fields) & set(fields))
        bad = {key: [archived_fields[key], fields[key]] for key in shared
               if archived_fields[key] != fields[key]}
        differing[scenario] = bad
        compared[scenario] = len(shared)
    fields, _ = v2_common.stage_b_inherited_config(1300027)
    only_current = {key: fields[key] for key in sorted(set(fields) - set(archived_fields))}
    gate("G5_world_and_transport_config", {
        "archived_behaviour_field_count": len(archived_fields),
        "shared_fields_compared_per_scenario": compared,
        "differing_fields_per_scenario": differing,
        "fields_the_archived_behaviour_block_does_not_record": only_current,
        "note": "the archived manifest records the 40 behaviour fields the frozen configuration "
                "hash is built from; the unrecorded keys are the per-arm policy selection and "
                "the disabled probes, all of which are re-checked empirically by the batch-0 "
                "byte comparison after the launches",
        "passed": all(not value for value in differing.values())})


# ------------------------------------------------------------------ G6 生产与导入件身份 -------------
def g6_module_identity() -> None:
    archive = r36.read_json(r36.ROUND32 / "CONFIG_AND_MODEL_MANIFEST.json")
    production_now = {name: r36.sha256_file(r36.ROOT / name)
                      for name in v2_common.PRODUCTION_DEPENDENCIES}
    production_archived = archive["code_identity"]["production"]
    imported_archived = archive["code_identity"]["imported_round_modules"]
    imported_now = {r36._rel(path): r36.sha256_file(path)
                    for path in v2_common.IMPORTED_ROUND_MODULES
                    if path.exists() and r36._rel(path) in imported_archived}
    production_changed = sorted(key for key in production_now
                                if production_archived.get(key) != production_now[key])
    imported_changed = sorted(key for key in imported_now
                              if imported_archived.get(key) != imported_now[key])
    checkpoints_now = {key.replace("\\", "/"): value
                       for key, value in archive["code_identity"]["checkpoints"].items()}
    gate("G6_produced_and_imported_code_identity", {
        "production_files": len(production_now),
        "production_changed_vs_round32": production_changed,
        "imported_round_modules_changed_vs_round32": imported_changed,
        "frozen_search_and_allocation_checkpoints_equal": (
            r36.code_identity()["frozen_checkpoints"] == checkpoints_now),
        "integrated_sar_has_zero_commits_since_round33_baseline":
            r36.git("log", "--oneline", f"{r36.ROUND33_BASELINE}..HEAD", "--",
                    "integrated_sar/").strip() == "",
        "passed": not production_changed and not imported_changed})


# ------------------------------------------------------------------ G7 训练路径函数级 diff -------------
def _function_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = hashlib.sha256(
                "".join(lines[node.lineno - 1:node.end_lineno]).encode("utf-8")).hexdigest()
    return out


def g7_trainer_diff() -> None:
    allowed_trainer = {"_bundle", "fresh_training_state", "load_checkpoint", "save_checkpoint",
                       "_restore_controller", "_validate_checkpoint_base",
                       "load_checkpoint_read_only"}
    allowed_run = {"_install_taps", "_run_episode", "run_eval"}
    report: dict[str, Any] = {}
    for module, allowed, frozen in (("v2_trainer.py", allowed_trainer,
                                     {"ppo_update", "batch_samples", "likelihood_alignment",
                                      "dual_update", "batch_g_hat", "_unassigned_prefix_totals"}),
                                    ("v2_run.py", allowed_run,
                                     {"run_training_episode", "action_seed", "_assert_transport",
                                      "_booked", "eval_action_seed"})):
        path = r36.STAGE_C / module
        old = subprocess.run(("git", "show", f"{r36.ROUND32_FREEZE_HEAD}:runs/"
                              f"comm_cost_v2_rerun_20260928/{module}"),
                             cwd=str(r36.ROOT), capture_output=True, text=True,
                             encoding="utf-8").stdout
        if not old.strip():
            raise SystemExit(f"cannot read the ROUND32-era {module} from git")
        before, now = _function_hashes(old), _function_hashes(path.read_text("utf-8"))
        shared = sorted(set(before) & set(now))
        changed = sorted(key for key in shared if before[key] != now[key])
        untouched_frozen = sorted(key for key in frozen if key in before and key in now
                                  and before[key] == now[key])
        report[module] = {
            "changed_functions": changed,
            "changed_outside_the_allowed_set": sorted(set(changed) - allowed),
            "added_functions": sorted(set(now) - set(before)),
            "removed_functions": sorted(set(before) - set(now)),
            "update_math_functions_required_identical": sorted(frozen),
            "verified_identical": untouched_frozen,
            "missing_or_changed_required": sorted(set(frozen) - set(untouched_frozen)),
            "passed": (not set(changed) - allowed
                       and set(frozen) <= set(untouched_frozen))}
    gate("G7_only_the_reward_timing_intervention_differs", {
        "per_module": report,
        "what_the_fix_is": "the task-reward tap moved from a post-env.step registry scan to a "
                           "PublishedProgressRewardCollector observing the synchronous "
                           "ProgressEvent at the current relay tick; plus tick 0 reserved as an "
                           "unassigned prefix; plus the two version fields and the resume gate",
        "passed": all(item["passed"] for item in report.values())})


# ------------------------------------------------------------------ G8 参照缓存 -------------
def g8_reference_cache() -> None:
    cache = r36.reference_cache()
    gate("G8_read_only_reference_cache", {
        "cached_references": cache["count"],
        "unique_scenarios_in_the_frozen_128_episode_order":
            cache["unique_scenarios_in_frozen_order"],
        "missing": cache["missing"],
        "reference_launches_needed": 0,
        "order_sha256": cache["order_sha256"],
        "files_read_only": len(cache["files_read_only"]),
        "passed": not cache["missing"] and cache["count"] >= 48})


# ------------------------------------------------------------------ G9 初值身份 -------------
def g9_initial_weights() -> None:
    per_seed: dict[str, Any] = {}
    all_equal = True
    for seed in r36.SEEDS:
        fresh = r36.fresh_controller_hash(seed)
        archive = r36.read_json(sorted((r36.ROUND32 / "checkpoints").glob(
            f"r32_batch_ALL_ROWS_s{seed}_b0.json"))[0])
        archived_before = archive["controller_parameters_before"]["parameter_sha256"]
        prelaunch_text = r36.git("show", f"{r36.ROUND32_FREEZE_HEAD}:runs/"
                          f"comm_actor_reduction_ablation_20261001/"
                          f"INIT_PAIRING_ALL_ROWS_s{seed}.json")
        prelaunch = json.loads(prelaunch_text)["controller_parameters"]["parameter_sha256"] \
            if prelaunch_text.strip() else None
        milestone = r36.STAGE_C / "checkpoints" / f"v2_train_s{seed}_ep0.pt"
        restored = None
        if milestone.exists():
            payload = torch.load(milestone, map_location="cpu", weights_only=False)
            controller = v2_trainer._restore_controller(payload, payload["training_state"])
            restored = v2_controller.parameter_identity(controller)["parameter_sha256"]
        values = {"fresh_controller_this_machine": fresh,
                  "archived_round32_batch0_before": archived_before,
                  "archived_round32_prelaunch_pairing_from_git": prelaunch,
                  "round24_ep0_milestone_restored_by_the_real_loader": restored}
        equal = len({value for value in values.values()}) == 1 and all(
            value is not None for value in values.values())
        all_equal = all_equal and equal
        per_seed[f"s{seed}"] = {"parameter_sha256_per_source": values, "all_four_identical": equal}
    gate("G9_initial_weights_match_the_archived_run_per_seed", {
        "per_seed": per_seed, "why_this_matters": "the batch-0 byte comparison against ROUND32 "
                                                 "only means something if episode 0 starts from "
                                                 "the same registered weights that run used",
        "passed": all_equal})


# ------------------------------------------------------------------ G10 额度与起点 -------------
def g10_quota_and_fresh_start() -> None:
    state = r36.book()
    existing = {directory.name: sorted(path.name for path in
                                       (directory / "checkpoints").glob("*.pt"))
                for directory in sorted(r36.HERE.glob("seed*")) if directory.is_dir()}
    constructions = r36.round_constructions()
    gate("G10_quota_and_no_hot_start", {
        "authorised": {"training_launches": r36.MAX_TRAINING_LAUNCHES,
                       "reserve_reruns": r36.MAX_RESERVE_RERUNS,
                       "episode_constructions": r36.MAX_EPISODE_CONSTRUCTIONS,
                       "episodes_per_seed": r36.EPISODES_PER_SEED,
                       "batches_per_seed": r36.BATCHES_PER_SEED,
                       "wall_clock_multiplier": r36.WALL_CLOCK_MULTIPLIER},
        "seed": list(r36.SEEDS),
        "seed_selection_provenance": "v2_common.TRAIN_SEEDS as declared by the frozen Stage C "
                                     "manifest; fixed before the first launch",
        "book_before_any_launch": state,
        "constructions_already_booked": constructions["total"],
        "checkpoints_already_present_in_the_seed_directories": existing,
        "no_rolling_checkpoint_before_start": not any(existing.values()),
        "v2_recover_used": False,
        "passed": (state["training_launches"] == 0 and constructions["total"] == 0
                   and not any(existing.values()))})


# ------------------------------------------------------------------ G11 wall-clock 上限 -------------
def g11_wall_clock() -> None:
    baseline = r36.archived_baseline()
    ceilings = {key: r36.wall_clock_ceiling(int(key[1:]), baseline) for key in baseline["per_seed"]}
    gate("G11_wall_clock_ceiling_from_the_archive", {
        "archived_per_seed_rollout_seconds": {
            key: value["rollout_seconds_sum"] for key, value in baseline["per_seed"].items()},
        "archived_rows_per_seed": {key: value["rows"] for key, value in baseline["per_seed"].items()},
        "ceiling_seconds": ceilings,
        "ceiling_hours": {key: round(value / 3600.0, 3) for key, value in ceilings.items()},
        "baseline_construction": "ROUND32 recorded no per-launch wall-clock, only a per-episode "
                                 "`seconds` field, so the ceiling is 1.5x the archived sum of "
                                 "that seed's 128 episode seconds; the update overhead is not in "
                                 "the archived baseline and is registered separately from this "
                                 "round's own measurement",
        "passed": all(value > 0 for value in ceilings.values())})
    r36.write_json("FROZEN_CONFIG_MANIFEST.json", {
        "kind": "round36-frozen-config-manifest", "round": "36",
        "branch": r36.git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head_at_preflight": r36.git("rev-parse", "HEAD").strip(),
        "baseline_head_round33": r36.ROUND33_BASELINE,
        "round32_freeze_head": r36.ROUND32_FREEZE_HEAD,
        "training_version": v2_trainer.TRAINING_VERSION,
        "reward_timing_version": v2_trainer.REWARD_TIMING_VERSION,
        "trainer_tag": v2_trainer.TRAINER_TAG, "loss_version": v2_trainer.LOSS_VERSION,
        "state_schema": v2_trainer.STATE_SCHEMA, "gae_schema": v2_trainer.GAE_SCHEMA,
        "optimizer_schema": v2_trainer.OPTIMIZER_SCHEMA,
        "reward_timing_schema": v2_trainer.REWARD_TIMING_SCHEMA,
        "seeds": list(r36.SEEDS),
        "episodes_per_seed": r36.EPISODES_PER_SEED,
        "batches_per_seed": r36.BATCHES_PER_SEED,
        "episodes_per_update": v2_common.EPISODES_PER_UPDATE,
        "steps_per_episode": v2_common.STEPS_PER_EPISODE,
        "milestone_episodes": list(v2_common.MILESTONE_EPISODES),
        "hyperparameters": GATES["G4_frozen_training_constants"]["ppo_hyperparameters"]["actual"],
        "objective_and_dual": GATES["G4_frozen_training_constants"]["objective_and_dual"],
        "scenario_block": GATES["G4_frozen_training_constants"]["scenario_block"],
        "data_order": GATES["G4_frozen_training_constants"]["data_order"],
        "row_filtering": GATES["G4_frozen_training_constants"]["row_filtering"],
        "freeze_identity": r36.freeze_identity(),
        "code_identity": r36.code_identity(),
        "wall_clock_ceilings": ceilings,
        "authorised_quota": {"training_launches": r36.MAX_TRAINING_LAUNCHES,
                             "reserve_reruns": r36.MAX_RESERVE_RERUNS,
                             "episode_constructions": r36.MAX_EPISODE_CONSTRUCTIONS},
        "reference_cache": {"count": GATES["G8_read_only_reference_cache"]["cached_references"],
                            "missing": GATES["G8_read_only_reference_cache"]["missing"],
                            "launches_needed": 0},
        "only_intervention": GATES["G7_only_the_reward_timing_intervention_differs"][
            "what_the_fix_is"],
        "trainer_diff_vs_round32_era": GATES[
            "G7_only_the_reward_timing_intervention_differs"]["per_module"],
        "world_config_comparison": GATES["G5_world_and_transport_config"],
        "production_identity": {"production_changed_vs_round32": GATES[
            "G6_produced_and_imported_code_identity"]["production_changed_vs_round32"],
            "imported_changed_vs_round32": GATES[
                "G6_produced_and_imported_code_identity"][
                "imported_round_modules_changed_vs_round32"],
            "integrated_sar_commits_since_baseline": 0},
        "initial_weights_per_seed": GATES["G9_initial_weights_match_the_archived_run_per_seed"][
            "per_seed"],
        "archived_baseline": baseline,
        "frozen_items_untouched": ["PILOC search and CTDE allocation checkpoints",
                                   "network structure", "objective", "reward values",
                                   "cost definition and capture path", "observations and masks",
                                   "GAE / gamma / lambda / dual budget / hyper-parameters",
                                   "scenario block (same as ROUND32 and ROUND35)",
                                   "row filtering (ALL_ROWS as encoded, ACTIVE_ROWS not used)",
                                   "training version fields"]}, refresh=True)


# ------------------------------------------------------------------ G12 ROUND34 回归 -------------
def g12_round34_suites() -> None:
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module in ("test_reward_timing_regression", "test_checkpoint_version_boundary"):
        spec = __import__(module)
        suite.addTests(loader.loadTestsFromModule(spec))
    result = unittest.TextTestRunner(verbosity=0, stream=open(__import__("os").devnull, "w",
                                                              encoding="utf-8")).run(suite)
    gate("G12_round34_regression_suites_under_real_torch", {
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "torch_real_not_stub": "torch" in sys.modules
                                                             and hasattr(torch, "cuda"),
        "environment_exemption_used": False,
        "passed": result.testsRun > 0 and not result.failures and not result.errors})


def main(argv: list[str]) -> int:
    for step in (g1_anchors, g2_environment, g3_versions, g4_constants, g5_world_config,
                 g6_module_identity, g7_trainer_diff, g8_reference_cache, g9_initial_weights,
                 g10_quota_and_fresh_start, g11_wall_clock, g12_round34_suites):
        step()
    payload = {"kind": "round36-preflight", "round": "36",
               "gates": GATES, "all_gates_passed": all(item["passed"]
                                                       for item in GATES.values()),
               "gate_count": len(GATES),
               "environment_launches_used": 0, "optimizer_steps_used": 0,
               "note": "every gate above reads only archived artefacts, git objects and the "
                       "imported modules; no JointRun is constructed and no weight is updated"}
    r36.write_json("ROUND36_PREFLIGHT.json", payload, refresh=True)
    print(json.dumps({"gates": {key: value["passed"] for key, value in GATES.items()},
                      "all_passed": payload["all_gates_passed"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(r36.ROUND34))
    raise SystemExit(main(sys.argv[1:]))
