"""Freeze the inputs of the action-geometry audit and prove what they can answer.

Run:  python audit_inputs.py

Writes COMM_ACTION_GEOMETRY_INPUT_MANIFEST.json into this directory. Reads the frozen round,
hashes every input, verifies the grid (16 streams x 9 arms x 4 layers), records the published
verdicts and their SHA256, states which audit questions the stored fields can answer, and
installs the environment tripwire whose (empty) call log is part of the manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import audit_common as A

CALLS = A.install_env_tripwire()
A.enter_source_round()

INPUT_ROOTS = ("rows", "streams", "runs")
AUDIT_QUESTIONS = (
    "A_concurrency_geometry",
    "B_quote_geometry",
    "C_budget_time_structure",
    "D_ledger_only_counterfactual",
)

#: Per-step candidate keys the g12 probe added. The three delivered layers do not have them in
#: their rows (extension protocol section 5), which is the asymmetry this manifest must state.
G12_ONLY_PROBE_KEYS = (
    "steps_with_one_candidate",
    "steps_with_multiple_candidates",
    "multi_all_payable_within_credit_steps",
    "multi_none_payable_steps",
    "multi_unclassifiable_steps",
    "multi_candidate_prices",
    "first_credit_exhaustion_facts",
    "competition_steps_with_unbounded_credit",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=A.ROOT, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def hash_inputs(refresh: bool = False) -> dict[str, Any]:
    """SHA256 every input. A size+mtime cache keeps re-running cheap during the round.

    The cache is keyed on (size, mtime_ns) as well as path, so a modified file re-hashes. The
    final delivery re-hashes everything with ``--refresh`` instead of trusting the cache.
    """
    cache_path = A.HERE / "_sha_cache.json"
    cache: dict[str, Any] = {} if refresh else (
        A.read_json(cache_path) if cache_path.exists() else {}
    )
    entries: dict[str, str] = {}
    total_bytes = 0
    fresh: dict[str, Any] = {}

    def digest(path: Path) -> str:
        stat = path.stat()
        key = str(path.relative_to(A.SOURCE_ROUND)).replace("\\", "/")
        stamp = [stat.st_size, stat.st_mtime_ns]
        if cache.get(key, {}).get("stamp") != stamp:
            fresh[key] = {"stamp": stamp, "sha256": A.sha256_file(path)}
        else:
            fresh[key] = {"stamp": stamp, "sha256": cache[key]["sha256"]}
        return fresh[key]["sha256"]

    for sub in INPUT_ROOTS:
        for path in sorted((A.SOURCE_ROUND / sub).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            key = str(path.relative_to(A.SOURCE_ROUND)).replace("\\", "/")
            entries[key] = digest(path)
            total_bytes += path.stat().st_size
    for pattern in ("*.json", "*.md", "*.py", "*.txt"):
        for path in sorted(A.SOURCE_ROUND.glob(pattern)):
            entries[path.name] = digest(path)
            total_bytes += path.stat().st_size
    A.write_json(cache_path, fresh)
    return {"file_count": len(entries), "total_bytes": total_bytes, "sha256": entries}


def grid(layer: str) -> dict[str, Any]:
    rows = A.rows_of(layer)
    seeds = sorted({int(row["seed"]) for row in rows})
    arms = sorted({str(row["arm"]) for row in rows})
    cells = {(str(row["arm"]), int(row["seed"])) for row in rows}
    expected = {(arm, seed) for arm in A.ARMS for seed in seeds}
    probe_keys = sorted({key for row in rows for key in row["probe"]})
    stamps: dict[str, int] = {}
    un_stamped = 0
    for row in rows:
        freeze = row.get("code_freeze")
        if freeze and freeze.get("episode_code_identity"):
            stamps[str(freeze["episode_code_identity"])] = stamps.get(str(freeze["episode_code_identity"]), 0) + 1
        else:
            un_stamped += 1
    # ``None`` is a real value here, not a gap: the ceiling arm runs with no byte cap at all, so
    # collapsing it to 0 or dropping it would erase the arm that L2 uses as its reference.
    def _distinct(field: str, cast: Any = None) -> list[Any]:
        values = {row.get(field) for row in rows}
        if cast is not None:
            values = {cast(value) if value is not None else None for value in values}
        return sorted(values, key=lambda value: (value is None, value))

    budgets = _distinct("budget_bytes", int)
    steps = _distinct("steps", int)
    ack = _distinct("ack_enabled", bool)
    delivery = _distinct("delivery_probability", float)
    timing = _distinct("timing_mode", str)
    goals = _distinct("goals", int)
    return {
        "streams": len(seeds),
        "seed_block": [seeds[0], seeds[-1]] if seeds else None,
        "seeds": seeds,
        "arms": arms,
        "arm_count": len(arms),
        "cells_found": len(cells),
        "cells_expected": len(expected),
        "missing_cells": sorted(f"{arm}/{seed}" for arm, seed in expected - cells),
        "duplicate_rows": len(rows) - len(cells),
        "steps_values": steps,
        "budget_bytes_values": budgets,
        "goals_values": goals,
        "ack_enabled_values": ack,
        "delivery_probability_values": delivery,
        "timing_mode_values": timing,
        "probe_keys": probe_keys,
        "g12_only_probe_keys_present": [k for k in G12_ONLY_PROBE_KEYS if k in probe_keys],
        "g12_only_probe_keys_absent": [k for k in G12_ONLY_PROBE_KEYS if k not in probe_keys],
        "row_stamp_identities": stamps,
        "row_cells_without_stamp": un_stamped,
        "row_files": sorted(
            str(p.relative_to(A.SOURCE_ROUND)).replace("\\", "/")
            for p in (A.SOURCE_ROUND / "rows" / layer).glob("*.jsonl")
        ),
        "transcript_files": sum(1 for p in (A.SOURCE_ROUND / "runs" / layer).glob("ct_*.jsonl")),
    }


def published(layer: str) -> dict[str, Any]:
    gates_path = A.SOURCE_ROUND / f"GATES_{layer}.json"
    gates = A.read_json(gates_path)
    return {
        "sha256_gates": A.sha256_file(gates_path),
        "sha256_rule_matrix": A.sha256_file(A.SOURCE_ROUND / f"RULE_MATRIX_{layer}.json"),
        "sha256_pressure_metrics": A.sha256_file(A.SOURCE_ROUND / f"PRESSURE_METRICS_{layer}.json"),
        "sha256_row_provenance": A.sha256_file(A.SOURCE_ROUND / f"ROW_PROVENANCE_{layer}.json"),
        "sha256_stream_manifest": A.sha256_file(A.SOURCE_ROUND / f"STREAM_MANIFEST_{layer}.json"),
        "gates_passed": {name: bool(gate["passed"]) for name, gate in gates["gates"].items()},
        "layer_passed": bool(gates["passed"]),
        "stop_reason": gates["stop_reason"],
        "l3_evidence": gates["gates"]["L3"]["evidence"],
        "best_fixed_rule": gates["gates"]["L3"]["evidence"]["best_fixed_rule"],
    }


def availability() -> dict[str, Any]:
    """Which audit question each stored field can answer, per layer -- stated before any analysis."""
    matrix: dict[str, Any] = {}
    for layer in A.LAYERS:
        rows = A.rows_of(layer)
        keys = {key for row in rows for key in row["probe"]}
        matrix[layer] = {
            "steps_with_two_or_more_candidates_exact": "steps_with_multiple_candidates" in keys,
            "strict_subset_step_count": "competition_points" in keys,
            "strict_subset_per_candidate_detail": "competition_sample" in keys,
            "none_payable_step_count": "multi_none_payable_steps" in keys,
            "all_payable_within_credit_step_count": "multi_all_payable_within_credit_steps" in keys,
            "joint_over_credit_step_count": "joint_only_points" in keys,
            "multi_candidate_price_summary": "multi_candidate_prices" in keys,
            "first_exhaustion_candidate_composition": "first_credit_exhaustion_facts" in keys,
            "per_event_refusal_charge_and_remaining": all(
                isinstance(row.get("refusal_sequence"), list) for row in rows
            ),
            "per_emission_charge_step_target_first_flag": all(
                isinstance(row.get("emission_sequence"), list) for row in rows
            ),
            "transcript_records_on_disk": all(
                A.transcript_path(layer, row["arm"], int(row["seed"])).exists() for row in rows
            ),
        }
    return matrix


def questions(availability_matrix: dict[str, Any]) -> dict[str, Any]:
    """Question-level answerability, including the parts only g12 can answer and how to cover them."""
    out: dict[str, Any] = {}
    out["A_concurrency_geometry"] = {
        "per_step_multi_candidate_count": {
            "g12": "exact (steps_with_multiple_candidates)",
            "g6_g8_g10": "bounded only: visible_total - steps_offered is an upper bound and "
                          "(visible_total - steps_offered) / (max_visible - 1) a lower bound, "
                          "because the delivered rows store sums, not a histogram",
        },
        "strict_subset_count": "exact for all four layers (competition_points)",
        "none_payable_count": {
            "g12": "exact (multi_none_payable_steps, visible >= 2)",
            "g6_g8_g10": "visible >= 1 only (credit_blocked_steps); the >= 2 split is missing",
        },
        "denominator_note": "a step with >= 1 payable candidate is exact for every layer via "
                            "probe.block_reasons['allowed']; the intersection of 'something "
                            "payable' and 'something blocked' is derived from block_reasons plus "
                            "steps_offered, not re-run",
    }
    out["B_quote_geometry"] = {
        "per_step_price_min_max_spread": "exact for g12 (multi_candidate_prices); for g6/g8/g10 "
                                         "only recoverable at the sampled strict-subset steps",
        "strict_subset_step_candidate_detail": "sampled, first 20 steps per cell "
                                               "(COMPETITION_SAMPLE_LIMIT = 20), all four layers",
        "transport_order_effect": "readable from the transcripts (send/refusal order), all layers",
    }
    out["C_budget_time_structure"] = {
        "first_budget_refusal_step": "exact for all layers (refusals.first_budget_refusal_step)",
        "first_none_payable_step": "exact for all layers (probe.first_credit_exhaustion_step, "
                                   "visible >= 1)",
        "first_strict_subset_step": "exact step number only where the cell has <= 20 points; "
                                    "otherwise the first sampled step is a bound, not the first",
        "exhaustion_candidate_composition": {
            "g12": "recorded (first_credit_exhaustion_facts)",
            "g6_g8_g10": "reconstructed from emission_sequence, then cross-validated against the "
                         "g12 cells where both are available; the audit reports that agreement "
                         "rate before trusting the reconstruction",
        },
        "credit_fraction_timestamps": "exact for all layers: emission_sequence carries per-send "
                                      "cost, so the spend curve is a sum over stored rows",
    }
    out["D_ledger_only_counterfactual"] = {
        "inputs": "emission_sequence + refusal_sequence per cell; no environment",
        "limitation": "the refusal log records the candidates the rule actually attempted and lost, "
                      "not every visible candidate, so an unblocked message can only be one that "
                      "was attempted -- this is stated in the verdict, not hidden",
        "cannot_claim": "task-outcome improvement; holding a refresh changes what the receivers "
                        "believe, which this round does not model",
    }
    return {key: out[key] for key in AUDIT_QUESTIONS}


def main(argv: list[str]) -> int:
    refresh = "--refresh" in argv
    started = time.time()
    frozen_before = hash_inputs(refresh=refresh)
    import pressure_common as common

    ledger = common.ledger_totals()
    layers: dict[str, Any] = {}
    for layer in A.LAYERS:
        layers[layer] = {
            "grid": grid(layer),
            "published": published(layer),
            "provenance": A.read_json(A.SOURCE_ROUND / f"ROW_PROVENANCE_{layer}.json"),
        }

    manifest = {
        "round": "comm_action_geometry_audit",
        "date": "2026-09-23",
        "status": "retrospective mechanism audit; not a pre-registered performance experiment",
        "authorisation": {
            "new_episodes": 0,
            "new_streams": 0,
            "optimizer_steps": 0,
            "checkpoint_loads": 0,
            "new_detection_runs": 0,
            "changed_reward_budget_ack_loss_topology_rules_or_defaults": False,
            "g14_g16_run": False,
            "h20_restored": False,
            "sealed_block_800000_800399_touched": False,
            "four_layer_result_files_modified": False,
        },
        "git": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "start_sha": "8e35a24e60a1542a13c38ed8f0c7f269c03d5374",
            "start_subject": git("log", "-1", "--format=%s", "8e35a24e60a1542a13c38ed8f0c7f269c03d5374"),
        },
        "source_round": str(A.SOURCE_ROUND.relative_to(A.ROOT)).replace("\\", "/"),
        "protocols": [
            "COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md",
            "COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md",
        ],
        "budget_bytes": A.BUDGET_BYTES,
        "definitions_frozen_by_the_source_round": {
            "payable": "a candidate whose send_executable is true against the credit left when "
                       "the step opened",
            "strict_subset_payable": "visible >= 2 and 1 <= payable < visible "
                                     "(= budget_competition_points, the only L3 reading)",
            "all_payable_within_credit": "visible >= 2 and payable == visible and "
                                         "sum(charge) <= remaining",
            "joint_over_credit": "visible >= 2 and payable == visible and sum(charge) > remaining "
                                 "(= joint_only_points; a cross-step choice, not a within-step one)",
            "none_payable": "visible >= 2 and payable == 0 (multi_none_payable_steps)",
            "not_a_competition_point": "a single-candidate refusal, and any step counted only "
                                       "because something was refused",
        },
        "inputs": {k: v for k, v in frozen_before.items() if k != "sha256"},
        "input_sha256": frozen_before["sha256"],
        "layers": layers,
        "field_availability": availability(),
        "question_answerability": questions(availability()),
        "launch_ledger_at_freeze": ledger,
        "environment_tripwire": {
            "installed": True,
            "wrapped_callables": [f"{m}.{a}" for m, a in A.FORBIDDEN_CALLS],
            "calls_so_far": list(CALLS),
            "meaning": "every episode-running entry point raises EnvironmentCallForbidden instead "
                       "of running; the log is re-checked at the end of the round and must be empty",
        },
        "generated_in_seconds": round(time.time() - started, 1),
    }
    A.write_json(A.HERE / "COMM_ACTION_GEOMETRY_INPUT_MANIFEST.json", manifest)
    print(json.dumps(
        {
            "files_hashed": frozen_before["file_count"],
            "gibibytes": round(frozen_before["total_bytes"] / 2**30, 3),
            "cells": {layer: layers[layer]["grid"]["cells_found"] for layer in A.LAYERS},
            "published": {layer: layers[layer]["published"]["gates_passed"] for layer in A.LAYERS},
            "tripwire_calls": len(CALLS),
            "seconds": round(time.time() - started, 1),
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
