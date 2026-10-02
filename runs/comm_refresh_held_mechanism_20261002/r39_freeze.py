"""Build the ROUND39 prelaunch freeze from the local ROUND37/R38 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_HEAD = "233a44fc10048488a0c71bda9d88e977c99c800e"
BASE_BRANCH = "analysis/comm-research-recovery"
EXPERIMENT_BRANCH = "exp/comm-refresh-held-mechanism"
FAILED_SCENES = [1600000, 1600011, 1600018, 1600034, 1600036, 1600040,
                 1600049, 1600056, 1600060, 1600062]
CONTROL_SCENES = [1600001, 1600002, 1600003, 1600004, 1600005, 1600006]
SEEDS = [11, 22, 33]
ARMS = [
    "s11_original_greedy", "s11_refresh_held",
    "s22_original_greedy", "s22_refresh_held",
    "s33_original_greedy", "s33_refresh_held",
    "immediate", "late_discovery_gate_147", "always_hold",
]
R37 = ROOT / "runs" / "comm_final_evaluation_20261002"
R38 = ROOT / "runs" / "comm_research_recovery_20261002"
STAGE_C = ROOT / "runs" / "comm_cost_v2_rerun_20260928"
ATTACHMENT_SHA256 = "d6912912609a239eba255d0a89019a156a1c22b36f42916d13d3acd4f2966bf4"

sys.path[:0] = [
    str(ROOT), str(R37), str(STAGE_C),
    str(ROOT / "runs" / "comm_cost_stage_b_20260927"),
    str(ROOT / "runs" / "comm_budget_action_mask_fix_20260922"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"), str(HERE),
]


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True,
        text=True, encoding="utf-8",
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: Any) -> None:
    target = HERE / name
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == rendered:
            return
        raise RuntimeError(f"refusing to overwrite non-identical prelaunch freeze file {target}")
    target.write_text(rendered, encoding="utf-8")


def main() -> None:
    head = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    if head != BASE_HEAD or branch != EXPERIMENT_BRANCH:
        raise SystemExit(f"wrong freeze base: branch={branch} HEAD={head}")
    if run_git("rev-parse", f"{BASE_BRANCH}") != BASE_HEAD:
        raise SystemExit("the expected starting branch no longer points at the required base")
    modified = [name.replace("\\", "/") for name in run_git(
        "diff", "--name-only", "HEAD"
    ).splitlines() if name.strip()]
    if any(not name.startswith("runs/comm_refresh_held_mechanism_20261002/")
           for name in modified):
        raise SystemExit(f"tracked files outside the ROUND39 directory are modified: {modified}")
    if sha256_file(HERE / "EXECUTION_BASIS.md") != ATTACHMENT_SHA256:
        raise SystemExit("EXECUTION_BASIS.md is not the copied user prompt byte-for-byte")

    r37_plan = json_file(R37 / "FROZEN_EVALUATION_PLAN.json")
    absolute = json_file(R38 / "ABSOLUTE_PERFORMANCE.json")
    missing = json_file(R38 / "MISSING_TARGET_CHAINS.json")
    cases = missing["cases"]
    derived_failed = sorted({int(case["scenario"]) for case in cases})
    derived_pairs = sorted(
        {f"{int(case['scenario'])}|{case['target_id']}" for case in cases}
    )
    if len(cases) != 11 or derived_failed != FAILED_SCENES:
        raise SystemExit(
            f"R38 machine diff disagrees with the authorized failed scenes: {derived_failed}"
        )
    if any(
        not all(
            case["greedy_paths_by_training_seed"][str(seed)][
                "is_missing_from_immediate_on_time_set"
            ]
            for seed in SEEDS
        )
        for case in cases
    ):
        raise SystemExit("R38 does not mark all 11 occurrences missing in all greedy seeds")
    greedy_checks = absolute["greedy_target_set_checks_vs_always_hold"]
    if any(
        not greedy_checks[f"model_s{seed}_greedy"][
            "identical_to_always_hold_in_all_64_scenes"
        ]
        for seed in SEEDS
    ):
        raise SystemExit("R38 greedy/always-hold target-set check is not all-identical")
    controls = [scene for scene in range(1_600_000, 1_600_064)
                if scene not in set(derived_failed)][:6]
    if controls != CONTROL_SCENES:
        raise SystemExit(f"control scenes do not match the fixed first-six complement: {controls}")

    # Confirm each missing target in R38 is the exact target for each seed, and retain the
    # crosswalk rather than treating the three seeds as independent occurrences.
    target_crosswalk = []
    for case in cases:
        if set(case["greedy_paths_by_training_seed"]) != {"11", "22", "33"}:
            raise SystemExit("R38 missing-target case is missing a frozen training seed")
        target_crosswalk.append({
            "scene": int(case["scenario"]),
            "target_id": str(case["target_id"]),
            "missing_from_all_three_greedy_seeds": True,
            "paired_occurrence_id": case.get("paired_occurrence_id"),
        })

    expected_code_maps = dict(r37_plan["configuration"]["production_files"])
    expected_code_maps.update({
        f"runs/comm_cost_v2_rerun_20260928/{name}": digest
        for name, digest in r37_plan["configuration"]["stage_c_harness"].items()
    })
    frozen_code_check = {}
    for rel, expected in sorted(expected_code_maps.items()):
        actual = sha256_file(ROOT / rel)
        frozen_code_check[rel] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches_round37_freeze": actual == expected,
        }
        if actual != expected:
            raise SystemExit(f"ROUND37 frozen source drifted: {rel}")

    import r37_common  # noqa: E402
    import v2_common  # noqa: E402

    checkpoint_registry = r37_common.model_registry(verify_on_disk=True)
    for seed in SEEDS:
        expected = r37_plan["models"][f"s{seed}"]
        actual = checkpoint_registry["models"][f"s{seed}"]
        for key in ("bytes_sha256", "parameter_sha256", "module_digests", "episodes_done",
                    "training_seed_in_state", "updates", "optimizer_steps", "lambda_final"):
            if actual.get(key) != expected.get(key):
                raise SystemExit(
                    f"seed {seed} checkpoint {key} disagrees with ROUND37 freeze: "
                    f"{actual.get(key)!r} != {expected.get(key)!r}"
                )
        if not actual.get("matches_round36_registry"):
            raise SystemExit(f"seed {seed} checkpoint does not match the ROUND36 registry")

    inherited_config, substituted = v2_common.stage_b_inherited_config(FAILED_SCENES[0])
    python_env = {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
    }
    import torch  # noqa: E402
    import numpy  # noqa: E402
    python_env.update({"torch": torch.__version__, "numpy": numpy.__version__,
                       "device_for_evaluation": "cpu"})

    frozen_plan = {
        "schema": "round39-frozen-scene-arm-plan/1",
        "base": {"branch": BASE_BRANCH, "commit": BASE_HEAD,
                 "experiment_branch": EXPERIMENT_BRANCH},
        "round37_and_round38_sources": {
            "round37_plan": "runs/comm_final_evaluation_20261002/FROZEN_EVALUATION_PLAN.json",
            "round37_plan_sha256": sha256_file(R37 / "FROZEN_EVALUATION_PLAN.json"),
            "round38_absolute_performance_sha256": sha256_file(
                R38 / "ABSOLUTE_PERFORMANCE.json"),
            "round38_missing_target_chains_sha256": sha256_file(
                R38 / "MISSING_TARGET_CHAINS.json"),
            "round38_input_inventory_sha256": sha256_file(
                R38 / "INPUT_SOURCE_INVENTORY.json"),
        },
        "scene_scope": {
            "diagnostic_only": True,
            "failed_scenes": FAILED_SCENES,
            "control_scenes": CONTROL_SCENES,
            "all_scenes_sorted": sorted(FAILED_SCENES + CONTROL_SCENES),
            "r38_machine_diff_failed_scenes": derived_failed,
            "r38_unique_scene_target_occurrences": len(cases),
            "r38_failed_target_crosswalk": target_crosswalk,
            "no_independent_test_or_generalisation_claim": True,
            "no_scene_reselection": True,
        },
        "arms": ARMS,
        "seeds": SEEDS,
        "normal_cells": 144,
        "first_scene_gate": {"scene": 1600000, "cells": 9,
                              "included_in_normal_cells": True},
        "technical_retry_reserve": 9,
        "hard_environment_start_cap": 153,
        "configuration": {
            "round37_frozen_configuration": r37_plan["configuration"],
            "stage_c_inherited_fields_scene_1600000": inherited_config,
            "configuration_substitutions": substituted,
            "scene_variation": "only scenario seed changes across scenes",
            "runtime_path_fields_rebound": ["output", "comm_transcript"],
            "runtime_path_root": "runs/comm_refresh_held_mechanism_20261002",
            "arm_changes": {
                "learned_arms": "comm_policy=learned; frozen greedy model only",
                "immediate": "same native immediate policy as ROUND37",
                "late_discovery_gate_147": (
                    "ROUND37 late_discovery_gate rule with frozen cut_steps=147.0"
                ),
                "always_hold": "same ROUND37 always_hold rule",
                "refresh_held": (
                    "ROUND37 frozen greedy collector wrapped only after relay first_emission=false"
                ),
            },
        },
        "models": checkpoint_registry,
        "runtime": python_env,
        "launches": {"normal": 144, "technical_reserve": 9, "hard_cap": 153,
                     "training_launches": 0, "optimizer_steps": 0,
                     "checkpoint_writes": 0},
        "authorization": {"environment_start_before_prelaunch_commit": False,
                           "remote_push": False, "deployment": False},
    }
    write_json("FROZEN_SCENE_AND_ARM_PLAN.json", frozen_plan)
    write_json("CHECKPOINT_IDENTITY.json", {
        "schema": "round39-checkpoint-identity/1",
        "all_three_read_only_verified_against_round36_and_round37": bool(
            checkpoint_registry["all_match_round36_registry"]
        ),
        "models": checkpoint_registry["models"],
        "checks": {
            "file_sha256": "byte-level file digest",
            "parameter_sha256": "loaded model parameter digest",
            "module_digests": "loaded module parameter summaries",
            "training_seed_episode_update_lambda": "checkpoint state fields",
            "loader": "load_checkpoint_read_only; optimizer and RNG not restored",
        },
    })

    extra_sources = [
        "runs/comm_final_evaluation_20261002/r37_run.py",
        "runs/comm_final_evaluation_20261002/r37_common.py",
        "runs/comm_final_evaluation_20261002/r37_gate.py",
        "runs/comm_cost_v2_rerun_20260928/v2_run.py",
        "runs/comm_cost_v2_rerun_20260928/v2_common.py",
        "runs/comm_budget_action_mask_fix_20260922/budget_masked.py",
        "runs/comm_cost_stage_b_20260927/sb_collector.py",
        "runs/comm_tight_budget_20260919/onpolicy.py",
        "integrated_sar/comm.py",
        "integrated_sar/comm_budget.py",
        "integrated_sar/comm_policy.py",
        "integrated_sar/comm_learning.py",
        "integrated_sar/joint_driver.py",
        "integrated_sar/task_registry.py",
    ]
    own_sources = ["r39_collector.py", "r39_preflight.py", "r39_freeze.py",
                   "r39_run.py", "r39_analyze.py"]
    code_identity = {
        "schema": "round39-code-identity/1",
        "base_commit": BASE_HEAD,
        "round37_frozen_source_check": frozen_code_check,
        "additional_runner_and_relay_sources": {
            rel: sha256_file(ROOT / rel) for rel in extra_sources
        },
        "round39_experiment_sources": {
            rel: sha256_file(HERE / rel) for rel in own_sources
        },
        "auxiliary_document_sha256": {
            "EXECUTION_BASIS.md": sha256_file(HERE / "EXECUTION_BASIS.md"),
            "REPORT_ERRATA.md": sha256_file(HERE / "REPORT_ERRATA.md"),
        },
        "execution_path": [
            "r37_run.controller_for read-only ep128 loader",
            "v2_run.run_eval for original frozen greedy and real output taps",
            "v2_run.run_rule for immediate, late_discovery_gate, always_hold",
            "r39_collector.RefreshHeldCollector for the local refresh-held arm",
            "v2_run._run_episode for the same integrated SAR run and accounting path",
        ],
        "relay_first_emission_authority": {
            "source": "integrated_sar/comm.py",
            "candidate": "CommRelay._observation sets first_emission from last_sent is None",
            "commit": "CommRelay._emit updates last_sent only after budget can_send passes",
            "loss": "CommRelay._emit updates sender ledger after packet attempt even when lost",
        },
        "production_modules_modified": [],
    }
    write_json("CODE_IDENTITY.json", code_identity)

    all_scenes = sorted(FAILED_SCENES + CONTROL_SCENES)
    jobs = [
        {"index": scene_index * len(ARMS) + arm_index,
         "scene": scene, "arm": arm, "status": "not_started"}
        for scene_index, scene in enumerate(all_scenes)
        for arm_index, arm in enumerate(ARMS)
    ]
    write_json("LAUNCH_BOOK.json", {
        "schema": "round39-launch-book/1",
        "status": "frozen_before_first_environment_start",
        "planned_normal_starts": 144,
        "technical_retry_reserve": 9,
        "hard_cap": 153,
        "starts": 0,
        "completed": 0,
        "technical_failures": 0,
        "retries_used": 0,
        "environment_constructions": 0,
        "training_launches": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "serial_shared_book_writer": True,
        "per_cell_claim": "O_CREAT|O_EXCL before calling the existing v2_common launch ledger",
        "v2_common_launch_cap": 153,
        "jobs": jobs,
        "attempts": [],
    })

    print(json.dumps({
        "branch": branch,
        "head": head,
        "failed_scenes": derived_failed,
        "controls": controls,
        "r38_targets": derived_pairs,
        "checkpoint_registry_matches": checkpoint_registry["all_match_round36_registry"],
        "frozen_source_count": len(frozen_code_check),
        "prelaunch_files_written": [
            "FROZEN_SCENE_AND_ARM_PLAN.json", "CHECKPOINT_IDENTITY.json",
            "CODE_IDENTITY.json", "LAUNCH_BOOK.json",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
