"""ROUND55: narrow statement reconciliation + late-gate observability.

Read-only analysis: reuses frozen rows/tapes; no environment, simulation, new
flow, training, optimizer update, or checkpoint write.  The audited ROUND53/54
statistics code is never imported for evidence; numbers are rebuilt from raw
integers with the frozen bootstrap protocol.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R53 = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2" / "round53_critic_isolation_trial_readiness"
RAW = R53 / "raw"
R54 = REPO / "analysis" / "round54-r53-validation-and-cost-gap"
SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
RNG_SEED = 20261008
DRAWS = 10_000
THRESHOLD = 4838.25
LATE_CUT = 147.0
HORIZON = 1000.0

sys.path.insert(0, str(R53))
from r53_contracts import TRAINING_VERSION, sha256_file  # noqa: E402
from r53_execute import read_json, verify_baseline_reuse  # noqa: E402

ZERO_OPS = {"environment_constructor_calls": 0, "simulation_runs": 0, "new_flows": 0,
            "training_runs": 0, "optimizer_updates": 0, "checkpoint_writes": 0,
            "new_episode_rows": 0, "cpu_model_forwards": 0}


def reconcile_statement() -> dict[str, Any]:
    stats = read_json(R53 / "ROUND53_STATISTICS.json")
    cell = next(c for c in stats["comparisons"]
                if c["seed"] == 11 and c["mode"] == "greedy"
                and c["control_name"] == "late_dev" and c["metric"] == "spent_bytes")
    dec = read_json(R54 / "RESIDUAL_COST_DECOMPOSITION.json")
    per_scene = dec["seed_results"]["11"]["per_control"]["late"]["per_scene"]
    deltas = [int(per_scene[str(s)]["delta"]) for s in SCENES]
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    idx = rng.integers(0, 16, size=(DRAWS, 16))
    means = np.array([float(d) for d in deltas])[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], method="linear")
    zeros_fraction = float((means == 0.0).mean())
    p = 1.0 / 16.0
    analytic = {
        "P_zero_sends_in_16_draws": float((1 - p) ** 16),
        "P_k_le_3": float(sum(math.comb(16, k) * p ** k * (1 - p) ** (16 - k) for k in range(4))),
        "implied_lower_quantile_bytes": 0.0,
        "implied_upper_quantile_bytes": 3 * 157.125,
    }
    nonzero = [(SCENES[i], d) for i, d in enumerate(deltas) if d != 0]
    agrees = (float(lo) == cell["difference"]["ci95_linear"][0]
              and float(hi) == cell["difference"]["ci95_linear"][1])
    return {
        "kind": "round55-statement-reconciliation/1",
        "statement_under_audit": "s11 greedy vs late: significantly more expensive (+157.1*)",
        "audited_files": [
            "round53.../ROUND53_REPORT.md (contains 'seed 11 greedy significantly worse (+157.1)')",
            "analysis/round54.../ROUND54_REPORT.md (repeats 'relative to late inconsistent (s11 significantly worse)')",
            "analysis/round54.../ROUND54_CORRECTIONS.md (did NOT correct this wording; only rounding)",
        ],
        "machine_cell": cell,
        "recomputed": {
            "scene_delta_vector": deltas,
            "nonzero_scenes": nonzero,
            "frozen_index_ci95": [float(lo), float(hi)],
            "resample_mean_zero_fraction": zeros_fraction,
            "matches_frozen_cell": agrees,
        },
        "analytic_binomial_check": analytic,
        "distinct_claims": {
            "one_target_explains_whole_gap": {
                "verdict": "成立",
                "note": "single decision cell 1301014/target-5 (+2514) explains the whole net gap"},
            "only_one_nonzero_scene_difference": {
                "verdict": "成立",
                "note": "15 zeros and one +2514; but this is exactly why significance fails: (15/16)^16 ~ 35.6% of resamples miss the cell entirely"},
            "significant_at_95": {
                "verdict": "不成立",
                "note": "CI95 = [0.0, 471.375]; lower bound is exactly 0 -> contains 0; 'significantly more expensive' is not supported"},
        },
        "reconciled_statement": (
            "s11 greedy vs late: point +157.125 B/episode, frozen bootstrap 95% CI [0.0, 471.375] "
            "contains 0 -> NOT significant; the gap is explained by the single decision cell "
            "1301014/target-5"),
        "correction_status": "pending -> registered in this file; old reports preserved unchanged",
        "development_go_unchanged": False,
        "cost_means_unchanged": True,
    }


def recover_gate(features: list[float]) -> dict[str, Any]:
    """Pure recovery of the late-gate decision from one 19-dim actor input.

    Inputs: the float32 vector the actor actually received + frozen constants
    (LATE_CUT_STEPS=147.0, EPISODE_HORIZON_STEPS=1000.0).  No scene/target ids,
    no outcomes, no receiver knowledge, no fitted thresholds.
    """
    f32 = np.asarray(features, dtype=np.float32)
    if f32.shape != (19,) or not np.isfinite(f32).all():
        raise ValueError("feature row must be finite float32[19]")
    age = int(round(float(f32[0])))
    step = int(round(float(f32[18]) * float(HORIZON)))
    first_emission = bool(f32[5] == np.float32(1.0))
    ready_step = step - age
    send = bool(first_emission and ready_step > LATE_CUT)
    return {"age": age, "step": step, "ready_step": ready_step,
            "first_emission": first_emission, "late_sends": send}


def validate_gate_recovery(arms) -> dict[str, Any]:
    late_rows = arms[("late", 0, "greedy")]
    per_scene = {}
    total_decisions = 0
    agreement = 0
    mismatches = []
    collisions: dict[str, set] = {}
    focal = {}
    for seed in SEEDS:
        for scene in SCENES:
            row = read_json(RAW / "rows" / f"v2_r44_eval_new_s{seed}_greedy_s{scene}_a1.json")
            tape = Path(row["artefacts"]["tape"])
            steps = [json.loads(line) for line in tape.read_text(encoding="utf-8").strip().splitlines()]
            decisions = {}
            for t in steps:
                for target in t.get("free_one_shot_choices", []):
                    if target in decisions:
                        raise RuntimeError(f"target {target} decided twice: s{seed} scene {scene}")
                    idx = list(t["actions"].keys()).index(target)
                    vec = [float(v) for v in t["features"][idx]]
                    rec = recover_gate(vec)
                    decisions[target] = rec
                    key = json.dumps(vec, separators=(",", ":"))
                    collisions.setdefault(key, set()).add(rec["late_sends"])
            total_decisions += len(decisions)
            actual_late = set(late_rows[scene]["sent_bytes_by_target"])
            predicted_late = {t for t, rec in decisions.items() if rec["late_sends"]}
            ok = predicted_late == actual_late
            agreement += int(ok)
            if not ok:
                mismatches.append({"seed": seed, "scene": scene,
                                   "predicted_late": sorted(predicted_late),
                                   "actual_late": sorted(actual_late)})
            per_scene[f"s{seed}/{scene}"] = {
                "decisions": len(decisions),
                "predicted_matches_actual": ok,
                "recovered_ready_steps": {t: decisions[t]["ready_step"] for t in sorted(decisions)},
            }
            if scene == 1301014:
                focal[f"s{seed}/1301014/t5"] = {
                    "recovered": decisions.get("piloc-e1-target-5"),
                    "late_actually_sent": "piloc-e1-target-5" in actual_late}
            if scene == 1301010:
                focal[f"s{seed}/1301010/t5"] = {
                    "recovered": decisions.get("piloc-e1-target-5"),
                    "late_actually_sent": "piloc-e1-target-5" in actual_late}
    collision_cells = [k for k, labels in collisions.items() if len(labels) > 1]
    return {
        "kind": "round55-gate-observability-validation/1",
        "tapes_parsed": len(SEEDS) * len(SCENES),
        "first_emission_decisions": total_decisions,
        "scene_level_agreement": agreement,
        "scene_level_mismatches": mismatches,
        "byte_identical_vector_collisions_with_opposite_labels": len(collision_cells),
        "focal_cells": focal,
        "per_scene": per_scene,
    }


def main() -> dict[str, Any]:
    reconciliation = reconcile_statement()
    (HERE / "STATEMENT_RECONCILIATION.json").write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sys.path.insert(0, str(R54))
    import importlib.util
    spec = importlib.util.spec_from_file_location("r54_mod", R54 / "r54_validate.py")
    r54 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r54)
    baseline, arms, bc_ids = r54.collect_rows()

    validation = validate_gate_recovery(arms)
    observability = {
        "kind": "round55-gate-observability/1",
        "late_rule_identity": {
            "source": "zhenghe/runs/comm_cost_v2_rerun_20260928/v2_rules.py (LateDiscoveryGatePolicy)",
            "predicate": "SEND iff item.first_emission AND float(item.ready_step) > LATE_CUT_STEPS",
            "cut": 147.0,
            "comparison": "strictly greater (>)",
            "ready_step_definition": "relay-tick bookkeeping step of the target's first unsent view change (sender-visible SendItem field)",
            "units": "relay ticks (dt=1.0, integer scale)",
            "when_decided": "at the target's one-shot first-report decision point (candidate_audit.one_shot_state_before == 'undecided')",
            "policy_history_dependence": "none: pure function of SendItem.first_emission/ready_step",
            "budget_refusal": "retried next step; first_emission stays true",
        },
        "mapping_A_sender_raw_state": {
            "verdict": "成立",
            "note": "SendItem.first_emission/ready_step are fields of the legal pre-action SendObservation the rule policy reads (integrated_sar/comm_policy.py SendItem); decided before any send of the step",
        },
        "mapping_B_19dim_recovery": {
            "verdict": "成立",
            "function": "first_emission = (f[5]==1.0); step = round(f[18]*1000.0); ready_step = step - round(f[0]); late_sends = first_emission and ready_step > 147.0",
            "encoding_sources": {
                "f0": "age_steps = max(0, tick - ready_step) (integrated_sar/comm.py)",
                "f5": "first_emission flag (integrated_sar/comm_learning.py features())",
                "f18": "min(1.0, step / EPISODE_HORIZON_STEPS=1000.0)",
                "frozen_constants": {"LATE_CUT_STEPS": 147.0, "EPISODE_HORIZON_STEPS": 1000.0},
            },
            "float32_note": "age is an exact integer-valued float32; f18 = float32(step/1000) round-trips exactly via round(f18*1000) for all steps 0..1000 (see r55_zero_tests.py)",
        },
        "validation": validation,
        "r54_statement_update": (
            "corrects R54's 'the current input cannot explain late's skip': the gate quantity "
            "ready_step IS recoverable from the 19-dim vector (f0/f18 + frozen constants). The "
            "information is not missing; both learned models failed to use it at that off-path "
            "state (in the BC training distribution a ready_step~121 target is decided around "
            "step ~121, while the candidate's decision happened at step 140)."),
        "conclusion_category": "category 1: the gate quantity is recoverable from the 19-dim vector -> the next method should study how to USE the existing information, not change the schema",
        "zero_operations": ZERO_OPS,
    }
    (HERE / "GATE_OBSERVABILITY.json").write_text(
        json.dumps(observability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source_map = {
        "kind": "round55-source-input-map/1",
        "feature_layout_source": "zhenghe/integrated_sar/comm_learning.py features() (19-dim /1 layout)",
        "slots": [
            {"index": 0, "name": "age_steps", "role": "gate-recovery input"},
            {"index": 1, "name": "since_last_emission_time", "role": "other"},
            {"index": 2, "name": "sent_count", "role": "other"},
            {"index": 3, "name": "pending_event flag", "role": "other"},
            {"index": 4, "name": "changed_view flag", "role": "other"},
            {"index": 5, "name": "first_emission flag", "role": "gate-recovery input"},
            {"index": 6, "name": "stage_changed flag", "role": "other"},
            {"index": 7, "name": "requirements_changed flag", "role": "other"},
            {"index": 8, "name": "members_changed flag", "role": "other"},
            {"index": 9, "name": "abs(progress_delta)", "role": "other"},
            {"index": 10, "name": "revision_delta", "role": "other"},
            {"index": 11, "name": "remaining_time/horizon (clipped [-1,1])", "role": "other"},
            {"index": 12, "name": "urgent flag (0 <= remaining <= horizon/2)", "role": "other"},
            {"index": 13, "name": "has_future_requirements flag", "role": "other"},
            {"index": 14, "name": "backlog_targets", "role": "other"},
            {"index": 15, "name": "budget_fraction", "role": "other"},
            {"index": 16, "name": "capacity_fraction", "role": "other"},
            {"index": 17, "name": "subscribers", "role": "other"},
            {"index": 18, "name": "min(1, step/1000)", "role": "gate-recovery input"},
        ],
        "late_gate_inputs": {"slots": [0, 5, 18],
                             "frozen_constants": {"LATE_CUT_STEPS": 147.0,
                                                  "EPISODE_HORIZON_STEPS": 1000.0}},
        "evidence_files": {
            "rule": "zhenghe/runs/comm_cost_v2_rerun_20260928/v2_rules.py",
            "policy_interface": "zhenghe/integrated_sar/comm_policy.py (SendItem, SendObservation)",
            "feature_builder": "zhenghe/integrated_sar/comm_learning.py (features())",
            "collector": "zhenghe-round42/runs/comm_one_shot_full_trace_ablation_20261004/r42_one_shot.py",
            "age_definition": "zhenghe/integrated_sar/comm.py: age_steps=max(0, self._tick - int(ready_step))",
            "rule_runner": "zhenghe-round42/runs/comm_one_shot_full_trace_ablation_20261004/r42_run.py run_rule() -> v2_rules.build_rule('late_discovery_gate', scenario)",
        },
    }
    (HERE / "SOURCE_INPUT_MAP.json").write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "complete", "zero_ops": ZERO_OPS,
            "statement_matches_machine": reconciliation["recomputed"]["matches_frozen_cell"],
            "gate_scene_agreement": validation["scene_level_agreement"],
            "gate_scene_mismatches": len(validation["scene_level_mismatches"]),
            "collisions": validation["byte_identical_vector_collisions_with_opposite_labels"],
            "decisions_checked": validation["first_emission_decisions"],
            "focal_cells": validation["focal_cells"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
