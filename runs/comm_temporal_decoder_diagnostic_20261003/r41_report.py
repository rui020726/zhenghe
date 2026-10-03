"""Build read-only ROUND41 result, pairing, target-chain, and artifact manifests."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[1]
SEEDS = (11, 22, 33)
SCENARIOS = tuple(range(1_301_000, 1_301_016))
BASELINE_MODES = ("greedy", "sampled", "immediate", "late_discovery_gate", "always_hold")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def file_ref(path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": digest(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def target_ids(row: dict[str, Any]) -> set[str]:
    return {str(event["task_id"]) for event in row.get("task_reward_events", [])
            if event.get("credited") and event.get("on_time")}


def eligibility_counts(row: dict[str, Any]) -> dict[str, int]:
    summary = row.get("collector_summary", {})
    direct = summary.get("eligibility_counts", {})
    if not direct:
        direct = summary.get("timing", {}).get("first_report_only_counts", {})
    return {
        "free_first_reports": int(direct.get("free_first_reports", 0)),
        "budget_masked_first_reports": int(direct.get("budget_masked_first_reports", 0)),
        "capacity_masked_first_reports": int(direct.get("capacity_masked_first_reports", 0)),
        "requested_first_report_sends": int(direct.get("requested_first_report_sends", 0)),
    }


def validate_row(row: dict[str, Any], path: Path, role: str) -> dict[str, Any]:
    actual_role = row.get("round41_role")
    allowed_roles = {"hazard_median": {"hazard_median", "formal_hazard_median"},
                     "greedy_reproduction": {"greedy_reproduction"}}
    if actual_role not in allowed_roles.get(role, {role}):
        raise RuntimeError(f"wrong ROUND41 role in {path.name}: {row.get('round41_role')}")
    on_time_targets = target_ids(row)
    if int(row["on_time"]) != len(on_time_targets):
        raise RuntimeError(f"on-time target closure failed for {path.name}")
    emissions = row.get("emission_sequence", [])
    if int(row["emissions"]) != len(emissions):
        raise RuntimeError(f"emission count does not close for {path.name}")
    if int(row["spent_bytes"]) != sum(int(item[3]) for item in emissions):
        raise RuntimeError(f"emitted byte total does not close for {path.name}")
    if int(row["spent_bytes"]) != int(row.get("airtime_tap_total", -1)):
        raise RuntimeError(f"airtime tap does not close for {path.name}")
    if int(row["task_reward_total"]) != int(row["on_time"]):
        raise RuntimeError(f"task reward total does not close for {path.name}")
    refresh = int(row.get("relay_counters", {}).get("refresh_emissions", -1))
    if refresh != 0 or any(not bool(item[5]) for item in emissions):
        raise RuntimeError(f"refresh emission found in {path.name}")
    eligibility = eligibility_counts(row)
    refusals = row.get("refusal_sequence", [])
    return {
        "seed": int(row["training_seed"]),
        "scenario": int(row["scenario_seed"]),
        "mode": row.get("eval_mode"),
        "role": role,
        "row": file_ref(path, "result_row"),
        "on_time": int(row["on_time"]),
        "on_time_target_ids": sorted(on_time_targets),
        "spent_bytes": int(row["spent_bytes"]),
        "emissions": int(row["emissions"]),
        "first_emissions": int(row.get("relay_counters", {}).get("first_emissions", 0)),
        "refresh_emissions": refresh,
        "first_emission_steps": [int(item[0]) for item in emissions],
        "mean_first_emission_step": (statistics.fmean(int(item[0]) for item in emissions)
                                      if emissions else None),
        "budget_masked_first_reports": eligibility["budget_masked_first_reports"],
        "capacity_masked_first_reports": eligibility["capacity_masked_first_reports"],
        "free_first_reports": eligibility["free_first_reports"],
        "requested_first_report_sends": eligibility["requested_first_report_sends"],
        "executor_refusal_events": len(refusals),
        "budget_refused_sends": int(row.get("relay_counters", {}).get("budget_refused_sends", 0)),
        "capacity_refused_sends": int(row.get("relay_counters", {}).get("capacity_refused_sends", 0)),
        "packet_attempts": int(row.get("relay_counters", {}).get("transmission_attempts", 0)),
        "packets_lost": int(row.get("relay_counters", {}).get("packets_lost", 0)),
        "parameters_sha256": row.get("controller_parameters_after", {}).get("parameter_sha256"),
        "checkpoint_sha256": row.get("checkpoint_identity", {}).get("file_sha256"),
        "world_configuration_sha256": row.get("round41_world_configuration_sha256"),
        "behaviour_sha256": row.get("behaviour_sha256"),
        "action_rng_seed": row.get("action_rng_seed"),
        "artefacts": {
            name: file_ref(Path(value), name)
            for name, value in row.get("artefacts", {}).items()
            if name in ("lifecycle", "transcript", "tape")
        },
    }


def baseline_key(seed: int, mode: str, scene: int) -> str:
    if mode in ("greedy", "sampled"):
        return f"NEW/s{seed}/{mode}/{scene}"
    return f"{mode}/{scene}"


def changed_target_chain(seed: int, scene: int, target: str, direction: str,
                         hazard_row: dict[str, Any], baseline: dict[str, Any],
                         hazard_summary: dict[str, Any]) -> dict[str, Any]:
    artefacts = hazard_row["artefacts"]
    lifecycle_path = Path(artefacts["lifecycle"])
    transcript_path = Path(artefacts["transcript"])
    tape_path = Path(artefacts["tape"])
    lifecycle = [event for event in read_jsonl(lifecycle_path)
                 if str(event.get("target_id", "")) == target]
    transcript = [event for event in read_jsonl(transcript_path)
                  if str(event.get("target", "")) == target]
    tapes = read_jsonl(tape_path)
    decision_history = []
    for tape in tapes:
        if target not in tape.get("targets", []):
            continue
        candidate = tape.get("candidate_audit", {}).get(target, {})
        hazard = tape.get("hazard_audit", {}).get(target, {})
        decision_history.append({
            "relay_tick": int(tape["step"]),
            "first_emission_eligible": bool(candidate.get("first_emission", False)),
            "initial_send_executable": candidate.get("relay_initial_executable"),
            "initial_block_reason": candidate.get("relay_initial_block_reason"),
            "free_first_report": target in tape.get("free_first_reports", []),
            "masked": target in tape.get("masked", []),
            "p_send": tape.get("send_probabilities", {}).get(target),
            "H_before": hazard.get("H_before"),
            "H_after": hazard.get("H_after"),
            "H_updated": hazard.get("H_updated"),
            "threshold": hazard.get("threshold"),
            "threshold_reached": hazard.get("threshold_reached"),
            "effective_action": tape.get("effective_actions", {}).get(target),
            "requested": tape.get("requested", {}).get(target),
            "shadow_charge_applied": candidate.get("shadow_charge_applied"),
            "shadow_budget_before": candidate.get("shadow_budget_before"),
            "shadow_budget_after": candidate.get("shadow_budget_after"),
        })
    rewards = [event for event in hazard_row.get("task_reward_events", [])
               if str(event.get("task_id")) == target]
    return {
        "seed": seed,
        "scenario": scene,
        "target_id": target,
        "change_vs_greedy": direction,
        "hazard_outcome": "on_time" if target in target_ids(hazard_row) else "not_on_time",
        "greedy_baseline_outcome": "on_time" if target in set(baseline["on_time_target_ids"])
        else "not_on_time",
        "baseline_row": baseline["row"],
        "hazard_reward_events": rewards,
        "decoder_history": decision_history,
        "communication_receipts": transcript,
        "allocation_job_and_terminal_lifecycle": lifecycle,
        "evidence_files": {
            "row": file_ref(Path(artefacts["row"]), "hazard_result_row"),
            "lifecycle": file_ref(lifecycle_path, "target_lifecycle"),
            "transcript": file_ref(transcript_path, "send_receive_transcript"),
            "tape": file_ref(tape_path, "candidate_decoder_tape"),
        },
        "whole_cell_hazard_verification": hazard_summary,
    }


def main() -> None:
    frozen = read_json(RUN_DIR / "ROUND41_FROZEN_CONFIG.json")
    gate = read_json(RUN_DIR / "ROUND41_FIRST_SIX_GATE.json")
    progress = read_json(RUN_DIR / "ROUND41_EXECUTION_PROGRESS.json")
    if frozen.get("status") != "passed" or gate.get("status") != "passed":
        raise RuntimeError("frozen identity or first-six gate is not passed")
    if progress.get("phase") != "complete":
        raise RuntimeError("execution progress is incomplete")

    raw_rows = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((RUN_DIR / "rows").glob("*.json"))]
    if len(raw_rows) != 51:
        raise RuntimeError(f"expected 51 result rows, found {len(raw_rows)}")
    indexed: dict[tuple[int, int, str], tuple[dict[str, Any], Path]] = {}
    for path in sorted((RUN_DIR / "rows").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        raw_role = str(row.get("round41_role"))
        role = "hazard_median" if raw_role in ("hazard_median", "formal_hazard_median") else raw_role
        key = (int(row["training_seed"]), int(row["scenario_seed"]), role)
        if key in indexed:
            raise RuntimeError(f"duplicate result row: {key}")
        indexed[key] = (row, path)

    result_rows = []
    for key in sorted(indexed):
        row, path = indexed[key]
        expected_role = "greedy_reproduction" if key[2] == "greedy_reproduction" else "hazard_median"
        if expected_role == "hazard_median" and key[2] != "hazard_median":
            raise RuntimeError(f"unexpected decoder role {key[2]}")
        result_rows.append(validate_row(row, path, expected_role))
    if sum(record["role"] == "hazard_median" for record in result_rows) != 48:
        raise RuntimeError("expected 48 formal hazard_median rows")
    if sum(record["role"] == "greedy_reproduction" for record in result_rows) != 3:
        raise RuntimeError("expected 3 greedy reproduction rows")

    # Re-run only the artifact-level verifier. It reads tapes and transcripts; it does not
    # construct an environment, load a checkpoint, sample, or alter the saved artifacts.
    sys.path.insert(0, str(RUN_DIR))
    verifier = importlib.import_module("r41_run").verify_hazard_row
    verification: dict[str, dict[str, Any]] = {}
    seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        for scene in SCENARIOS:
            row, _ = indexed[(seed, scene, "hazard_median")]
            verification[f"s{seed}/{scene}"] = verifier(row)

    baselines = frozen["baseline_reuse"]["rows"]
    comparisons: dict[str, Any] = {}
    target_chains = []
    for seed in SEEDS:
        seed_outcomes: dict[str, Any] = {}
        hazard_rows = [indexed[(seed, scene, "hazard_median")][0] for scene in SCENARIOS]
        hazard_totals = {
            "on_time": sum(int(row["on_time"]) for row in hazard_rows),
            "spent_bytes": sum(int(row["spent_bytes"]) for row in hazard_rows),
            "first_emissions": sum(int(row["emissions"]) for row in hazard_rows),
            "budget_masked_first_reports": sum(eligibility_counts(row)["budget_masked_first_reports"]
                                                for row in hazard_rows),
            "capacity_masked_first_reports": sum(eligibility_counts(row)["capacity_masked_first_reports"]
                                                  for row in hazard_rows),
            "executor_refusal_events": sum(len(row.get("refusal_sequence", [])) for row in hazard_rows),
            "budget_refused_sends": sum(int(row.get("relay_counters", {}).get("budget_refused_sends", 0))
                                         for row in hazard_rows),
            "capacity_refused_sends": sum(int(row.get("relay_counters", {}).get("capacity_refused_sends", 0))
                                           for row in hazard_rows),
            "packet_attempts": sum(int(row.get("relay_counters", {}).get("transmission_attempts", 0))
                                    for row in hazard_rows),
            "packets_lost": sum(int(row.get("relay_counters", {}).get("packets_lost", 0))
                                 for row in hazard_rows),
        }
        times = [int(item[0]) for row in hazard_rows for item in row.get("emission_sequence", [])]
        hazard_totals["mean_first_emission_step"] = statistics.fmean(times) if times else None
        hazard_totals["median_first_emission_step"] = statistics.median(times) if times else None

        for mode in BASELINE_MODES:
            scene_records = []
            baseline_rows = [baselines[baseline_key(seed, mode, scene)] for scene in SCENARIOS]
            baseline_totals = {
                "on_time": sum(int(row["on_time"]) for row in baseline_rows),
                "spent_bytes": sum(int(row["spent_bytes"]) for row in baseline_rows),
                "emissions": sum(int(row.get("emissions", 0)) for row in baseline_rows),
            }
            gained_all: list[dict[str, Any]] = []
            lost_all: list[dict[str, Any]] = []
            for scene, hazard_row, baseline in zip(SCENARIOS, hazard_rows, baseline_rows):
                new_ids = target_ids(hazard_row)
                old_ids = set(str(x) for x in baseline["on_time_target_ids"])
                gained = sorted(new_ids - old_ids)
                lost = sorted(old_ids - new_ids)
                for target in gained:
                    gained_all.append({"scenario": scene, "target_id": target})
                    if mode == "greedy":
                        target_chains.append(changed_target_chain(seed, scene, target, "gained",
                                                                  hazard_row, baseline,
                                                                  verification[f"s{seed}/{scene}"]))
                for target in lost:
                    lost_all.append({"scenario": scene, "target_id": target})
                    if mode == "greedy":
                        target_chains.append(changed_target_chain(seed, scene, target, "lost",
                                                                  hazard_row, baseline,
                                                                  verification[f"s{seed}/{scene}"]))
                delta_tasks = int(hazard_row["on_time"]) - int(baseline["on_time"])
                delta_bytes = int(hazard_row["spent_bytes"]) - int(baseline["spent_bytes"])
                scene_records.append({
                    "scenario": scene,
                    "hazard_median": {"on_time": int(hazard_row["on_time"]),
                                      "on_time_target_ids": sorted(new_ids),
                                      "spent_bytes": int(hazard_row["spent_bytes"]),
                                      "first_emissions": int(hazard_row["emissions"]),
                                      "row_sha256": digest(Path(hazard_row["artefacts"]["row"]))},
                    "baseline": {"on_time": int(baseline["on_time"]),
                                 "on_time_target_ids": sorted(old_ids),
                                 "spent_bytes": int(baseline["spent_bytes"]),
                                 "emissions": int(baseline.get("emissions", 0)),
                                 "row_sha256": baseline["row"]["sha256"],
                                 "row_path": baseline["row"]["path"]},
                    "delta_hazard_minus_baseline": {
                        "on_time": delta_tasks,
                        "spent_bytes": delta_bytes,
                        "gained_targets": gained,
                        "lost_targets": lost,
                    },
                })
            comparisons[f"s{seed}/{mode}"] = {
                "seed": seed,
                "comparison": f"hazard_median - NEW {mode}" if mode in ("greedy", "sampled")
                else f"hazard_median - {mode}",
                "hazard_median_totals": hazard_totals,
                "baseline_totals": baseline_totals,
                "delta_totals": {"on_time": hazard_totals["on_time"] - baseline_totals["on_time"],
                                 "spent_bytes": hazard_totals["spent_bytes"] - baseline_totals["spent_bytes"]},
                "scenes_improved_on_time": sum(1 for x in scene_records
                                               if x["delta_hazard_minus_baseline"]["on_time"] > 0),
                "scenes_tied_on_time": sum(1 for x in scene_records
                                            if x["delta_hazard_minus_baseline"]["on_time"] == 0),
                "scenes_regressed_on_time": sum(1 for x in scene_records
                                                 if x["delta_hazard_minus_baseline"]["on_time"] < 0),
                "scenes_bytes_lower": sum(1 for x in scene_records
                                           if x["delta_hazard_minus_baseline"]["spent_bytes"] < 0),
                "scenes_bytes_equal": sum(1 for x in scene_records
                                           if x["delta_hazard_minus_baseline"]["spent_bytes"] == 0),
                "scenes_bytes_higher": sum(1 for x in scene_records
                                           if x["delta_hazard_minus_baseline"]["spent_bytes"] > 0),
                "gained_target_occurrences": gained_all,
                "lost_target_occurrences": lost_all,
                "target_swap_scenes": sum(1 for x in scene_records
                                           if x["delta_hazard_minus_baseline"]["gained_targets"]
                                           and x["delta_hazard_minus_baseline"]["lost_targets"]),
                "scenes": scene_records,
            }
        seed_outcomes["hazard_median_totals"] = hazard_totals
        seed_outcomes["comparison_deltas"] = {
            mode: comparisons[f"s{seed}/{mode}"]["delta_totals"] for mode in BASELINE_MODES
        }
        seed_outcomes["round40_greedy_reproduction"] = next(
            record for record in result_rows
            if record["seed"] == seed and record["scenario"] == SCENARIOS[0]
            and record["role"] == "greedy_reproduction")
        seed_outcomes["hazard_artifact_verifications"] = {
            str(scene): verification[f"s{seed}/{scene}"] for scene in SCENARIOS
        }
        seed_outcomes["hazard_median"] = {"totals": hazard_totals}
        seed_outcomes["round40_greedy_reproduction"] = {
            "on_time": seed_outcomes["round40_greedy_reproduction"]["on_time"],
            "spent_bytes": seed_outcomes["round40_greedy_reproduction"]["spent_bytes"],
            "row_sha256": seed_outcomes["round40_greedy_reproduction"]["row"]["sha256"],
        }
        seed_outcomes["hazard_median"]["by_scene"] = [
            {"scenario": scene, "on_time": int(indexed[(seed, scene, "hazard_median")][0]["on_time"]),
             "spent_bytes": int(indexed[(seed, scene, "hazard_median")][0]["spent_bytes"]),
             "first_emissions": int(indexed[(seed, scene, "hazard_median")][0]["emissions"])}
            for scene in SCENARIOS
        ]
        # Keep one seed-indexed object and one flat comparison table for compact downstream use.
        seed_results[str(seed)] = seed_outcomes

    progress_ledger = progress["ledger"]
    claims_dir = RUN_DIR / "launch_claims"
    slots_dir = RUN_DIR / "launch_slots"
    claims = [read_json(path) for path in sorted(claims_dir.glob("*/*.json"))]
    normal_slots = [read_json(path) for path in sorted((slots_dir / "normal").glob("*.json"))]
    reserve_slots = [read_json(path) for path in sorted((slots_dir / "reserve").glob("*.json"))]
    if len(claims) != 51 or len(normal_slots) != 51 or reserve_slots:
        raise RuntimeError("claim and atomic-slot files do not match the 51+5 quota")
    if any(claim.get("status") != "complete" or claim.get("attempt") != 1 for claim in claims):
        raise RuntimeError("a launch claim is incomplete or used a retry")
    if len({claim["logical_job"] for claim in claims}) != 51:
        raise RuntimeError("launch logical job identities are not unique")
    if any(slot.get("category") != "normal" for slot in normal_slots):
        raise RuntimeError("unexpected launch slot category")
    ledger = {
        "kind": "round41-execution-ledger/1",
        "status": "complete",
        "quota": {"normal_cap": 51, "technical_reserve_cap": 5, "hard_cap": 56},
        "counts": {
            "normal_used": len(normal_slots), "reserve_used": len(reserve_slots),
            "total_used": len(normal_slots) + len(reserve_slots),
            "environment_constructor_calls": sum(int(c.get("environment_constructor_calls", 0))
                                                  for c in claims),
            "simulation_started": sum(bool(c.get("simulation_started")) for c in claims),
            "technical_failed_claims": sum(c.get("status") == "failed_technical" for c in claims),
            "retry_claims": sum(int(c.get("attempt", 1)) > 1 for c in claims),
            "optimizer_step_calls": sum(int(c.get("optimizer_step_calls", 0)) for c in claims),
            "checkpoint_write_calls": sum(int(c.get("checkpoint_write_calls", 0)) for c in claims),
        },
        "progress_ledger_crosscheck": {
            key: progress_ledger[key] for key in (
                "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
                "simulation_started", "optimizer_step_calls", "checkpoint_write_calls", "hard_cap")
        },
        "normal_slots": normal_slots,
        "reserve_slots": reserve_slots,
        "claims": claims,
    }
    expected = {"normal_used": 51, "reserve_used": 0, "total_used": 51,
                "environment_constructor_calls": 51, "simulation_started": 51,
                "technical_failed_claims": 0, "retry_claims": 0,
                "optimizer_step_calls": 0, "checkpoint_write_calls": 0}
    if ledger["counts"] != expected:
        raise RuntimeError(f"launch ledger counts mismatch: {ledger['counts']}")

    for seed in SEEDS:
        model = frozen["models"][str(seed)]
        model_path = Path(model["path"])
        if digest(model_path) != model["sha256"]:
            raise RuntimeError(f"checkpoint changed after execution for seed {seed}")

    identity = {
        "round40_head": frozen["execution_identity"]["round40_head"],
        "round41_prelaunch_code_head": frozen["execution_identity"]["round41_source_head_before_freeze_commit"],
        "round41_branch": frozen["execution_identity"]["round41_branch"],
        "scenarios": list(SCENARIOS),
        "seeds": list(SEEDS),
        "decoder": frozen["method"],
        "models": frozen["models"],
        "world_configuration_hashes": frozen["world_and_scenarios"]["effective_world_config_sha256_by_scenario"],
        "baseline_reuse_counts": {k: frozen["baseline_reuse"][k] for k in
                                  ("new_greedy", "new_sampled", "immediate",
                                   "late_discovery_gate_cut_147", "always_hold")},
        "first_six_gate": gate,
        "round41_pure_compute_gate": read_json(RUN_DIR / "ROUND41_PURE_COMPUTE_GATE.json"),
        "postrun_checkpoint_rehash": {
            str(seed): {"path": frozen["models"][str(seed)]["path"],
                        "sha256_expected": frozen["models"][str(seed)]["sha256"],
                        "sha256_after": digest(Path(frozen["models"][str(seed)]["path"]))}
            for seed in SEEDS
        },
    }

    pair_output = {
        "kind": "round41-paired-results/1",
        "interpretation": "descriptive only; all scenes were previously used development scenes",
        "task_unit": "on-time target occurrences per one-episode scene",
        "cost_unit": "attempt-billed payload bytes per one-episode scene",
        "identity": identity,
        "seed_results": seed_results,
        "flat_comparisons": comparisons,
        "aggregate_hazard_verification": {
            "rows_verified": len(verification),
            "tape_rows": sum(value["tape_rows"] for value in verification.values()),
            "free_legal_opportunities": sum(value["free_legal_opportunities"] for value in verification.values()),
            "H_updates": sum(value["H_updates"] for value in verification.values()),
            "masked_opportunities": sum(value["masked_opportunities"] for value in verification.values()),
            "unique_target_tick_updates": sum(value["unique_target_tick_updates"] for value in verification.values()),
            "request_send_or_refusal_events": sum(value["request_send_or_refusal_events"] for value in verification.values()),
            "real_first_emissions": sum(value["real_first_emissions"] for value in verification.values()),
            "refresh_emissions": sum(value["refresh_emissions"] for value in verification.values()),
            "cells": verification,
        },
        "limits": [
            "The 16 scenes are the already-used ROUND40 development block.",
            "Do not interpret one sampled run per scene as full sampled-policy uncertainty.",
            "Matching scene seeds do not imply identical message-level packet losses.",
            "The fixed cumulative-risk decoder is not the exact median of the closed-loop sampled policy.",
            "No significance, generalization, independent confirmation, or overall noninferiority claim is made.",
        ],
    }

    result_rows_payload = {
        "kind": "round41-result-rows/1",
        "status": "complete",
        "row_count": len(result_rows),
        "formal_hazard_median_rows": 48,
        "greedy_reproduction_rows": 3,
        "rows": result_rows,
    }

    local_external_files = []
    for directory, role in ((RUN_DIR / "rows", "result_row"),
                            (RUN_DIR / "logs", "raw_log_or_tape"),
                            (RUN_DIR / "launch_claims", "launch_claim"),
                            (RUN_DIR / "launch_slots", "atomic_launch_slot")):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                local_external_files.append(file_ref(path, role))
    progress_path = RUN_DIR / "ROUND41_EXECUTION_PROGRESS.json"
    local_external_files.append(file_ref(progress_path, "execution_progress_checkpoint"))
    model_inputs = []
    for seed in SEEDS:
        item = frozen["models"][str(seed)]
        model_inputs.append({"seed": seed, "path": item["path"], "size_bytes": item["size_bytes"],
                             "sha256": item["sha256"], "role": "read_only_ep128_input"})
    external_manifest = {
        "kind": "round41-git-external-artifacts/1",
        "note": "Raw rows, lifecycle, transcript, tape, claim and slot files remain outside Git; hashes are recomputed from disk.",
        "local_generated_artifact_count": len(local_external_files),
        "local_generated_artifacts": local_external_files,
        "read_only_model_inputs": model_inputs,
        "frozen_round40_input_manifest": {
            "path": str((RUN_DIR / "ROUND41_FROZEN_CONFIG.json").resolve()),
            "note": "The frozen config records all 144 reused baseline row/lifecycle/transcript/tape identities and their source claims.",
        },
    }

    write_json(RUN_DIR / "ROUND41_EXECUTION_LEDGER.json", ledger)
    write_json(RUN_DIR / "ROUND41_PAIRED_RESULTS.json", pair_output)
    write_json(RUN_DIR / "R41_RESULT_ROWS.json", result_rows_payload)
    write_json(RUN_DIR / "R41_TARGET_CHAINS.json", {
        "kind": "round41-target-chains/1",
        "primary_comparison": "hazard_median - NEW greedy",
        "changed_target_occurrences": len(target_chains),
        "target_changes": target_chains,
    })
    write_json(RUN_DIR / "GIT_OUTSIDE_ARTIFACTS.json", external_manifest)
    print(json.dumps({
        "status": "passed",
        "rows": len(result_rows),
        "hazard_cells_verified": len(verification),
        "target_changes_vs_greedy": len(target_chains),
        "ledger": ledger["counts"],
        "external_artifact_count": len(local_external_files),
        "seed_totals": {
            seed: {mode: comparisons[f"s{seed}/{mode}"]["delta_totals"]
                   for mode in BASELINE_MODES}
            for seed in SEEDS
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
