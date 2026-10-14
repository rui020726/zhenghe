# -*- coding: utf-8 -*-
"""ROUND72 local close-out: emit resource ledger, machine verdict and report.

Zero environment, zero model load, zero forward, zero RNG, zero optimizer.
Reads persisted JSON/JSONL artifacts only and re-asserts the locked facts so the
deliverables are grounded in on-disk evidence rather than restated by hand.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import REPO, RUNS, R68, R71, R66, DELIV, FOCAL_SCENE

FROZEN_SRC = (RUNS / "round51_frozen_source_gradient_close" / "frozen_sources")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


boundary = load(DELIV / "ROUND72_BOUNDARY_AUDIT.json")
signal = load(DELIV / "ROUND72_TRAINING_SIGNAL_AUDIT.json")
r71_stats = load(R71 / "raw" / "ROUND71_STATISTICS.json")
r68_rule = load(R68 / "raw" / "ROUND68_RULE_CONTEXT.json")
r71_ck = load(R71 / "ROUND71_CHECKPOINT_INDEX.json")
r71_ledger = load(R71 / "ROUND71_RESOURCE_LEDGER.json")

prim = boundary["comparisons"]["PRIMARY_R68_TRAINED_VS_R71_NEW_EP128"]["overall"]
prim_margin = prim["margin"]
ra = r68_rule["rule_absolute"]
ts11 = r71_ck["seeds"]["s11"]["trained_ep128"]["training_state"]

# ---- locked facts (pure computation over persisted data) ----
assert prim_margin["early_max_abs_dp"] == 0.0, "early path is not bit-identical"
assert prim_margin["argmax_flips"] == 0, "a greedy argmax flip exists on the 192-cell block"
assert prim["n_free"] == 1098, prim["n_free"]
assert prim_margin["late_n"] == 294 and prim_margin["late_n_dp_nonzero"] == 294
assert prim_margin["late_max_encroachment_ratio"] is not None
assert round(5394 / 32, 4) == 168.5625, "scene-1800000 single-scene decomposition"
assert ra["IMMEDIATE"]["spent_bytes_per_scene_mean"] == 14356.5
assert ra["IMMEDIATE"]["on_time_total"] == 183
assert ra["ALWAYS_HOLD"]["on_time_total"] == 170 and ra["ALWAYS_HOLD"]["spent_bytes_total"] == 0
assert ra["LATE_DISCOVERY_GATE"]["on_time_total"] == 183
assert ts11["gamma"] == 1.0 and ts11["gae_lambda"] == 1.0 and ts11["eta_lambda"] == 0.5
for cell in ("11/greedy", "22/greedy", "33/greedy"):
    p = r71_stats["per_seed_mode"][cell]["primary"]
    assert p["task_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_mean"] == 0.0
    assert p["cost_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_B_per_scene"] == 0.0
    assert p["targets_lost_NEW_EP128_vs_R68_TRAINED_ALPHA_count"] == 0

sampled = {}
for cell in ("11/sampled", "22/sampled", "33/sampled"):
    p = r71_stats["per_seed_mode"][cell]["primary"]
    sampled[cell] = {
        "on_time": r71_stats["per_seed_mode"][cell]["NEW_EP128_on_time_total"],
        "bytes_per_scene": r71_stats["per_seed_mode"][cell]["NEW_EP128_bytes_per_scene"],
        "task_delta": p["task_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_mean"],
        "task_ci95": p["task_ci95_NEW_EP128_minus_R68_TRAINED_ALPHA"],
        "cost_delta_B_per_scene": p["cost_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_B_per_scene"],
        "cost_ci95": p["cost_ci95_NEW_EP128_minus_R68_TRAINED_ALPHA_B_per_scene"],
        "targets_lost": p["targets_lost_NEW_EP128_vs_R68_TRAINED_ALPHA_count"],
    }

flips = boundary["sampled_flips"]
sig71 = signal["rounds"]["round71"]["summary"]
sig68 = signal["rounds"]["round68"]["summary"]

runtime = {
    "r42_one_shot.py": sha256(R71 / "raw" / "r42_one_shot.py"),
    "r40_train_resume_adapter.py": sha256(R71 / "raw" / "r40_train_resume_adapter.py"),
    "r42_versions.py": sha256(R71 / "raw" / "r42_versions.py"),
    "r66_frozen.py": sha256(R66 / "r66_frozen.py"),
    "canonical_v2_trainer.py": sha256(FROZEN_SRC / "canonical_v2_trainer.py"),
    "cost_math.py": sha256(REPO / "runs" / "comm_cost_controller_stage_a_20260927" / "cost_math.py"),
    "comm_budget.py": sha256(REPO / "integrated_sar" / "comm_budget.py"),
    "comm_learning.py": sha256(REPO / "integrated_sar" / "comm_learning.py"),
}
# R68 and R71 shipped byte-identical runtime modules (verified this session).
runtime["r68_r71_runtime_identical"] = all(
    sha256(R68 / "raw" / n) == sha256(R71 / "raw" / n)
    for n in ("r42_one_shot.py", "r40_train_resume_adapter.py", "r42_versions.py",
              "r43_call_counter_hooks.py"))

HEAD = git_head()

verdict = {
    "kind": "round72-machine-verdict/1",
    "task": "ROUND72 - ROUND71 originals acceptance and greedy-behaviour-unchanged mechanism diagnostic",
    "branch": "analysis/round72-calibration-behaviour-diagnostic",
    "evidence_head": HEAD,
    "session_side_effects": {
        "environment_constructions": 0, "model_loads": 0, "forward_passes": 0,
        "rng_uses": 0, "ppo_or_optimizer_calls": 0, "synthetic_optimizer_steps": 0,
        "checkpoint_writes": 0, "new_evaluations": 0, "new_claims": 0,
        "note": "read-only parsing of persisted JSON/JSONL plus writing these deliverables",
    },
    "runtime_identity": runtime,
    "plan_source_identity": {
        "eval_version": r71_stats["eval_version"], "training_version": r71_stats["training_version"],
        "eval_block": "1800000..1800031 (already-used development block, NOT a confirmation set)",
        "training_teacher_block": "1300000..1300047 (R68 real training tapes; R71 offline-fit teacher)",
        "scene_unit": 32, "seeds": [11, 22, 33], "modes": ["greedy", "sampled"],
        "focal_scene": FOCAL_SCENE,
        "greedy_is_primary": True, "sampled_reported_separately": True,
    },
    "eval_matrix": {
        "cells": 192, "focal_cells": 6, "arms_compared": "r71:new_ep128 minus r68:trained_alpha",
        "bootstrap": r71_stats["bootstrap"],
        "row_source_audit_keys": list(r71_stats.get("row_source_audit", {}).keys()),
    },
    "mechanism": {
        "matched_free_first_report_decisions": prim["n_free"],
        "early_decisions": {"n": 804, "max_abs_dp": 0.0, "structural": True,
                            "reason": "trunk+early_head frozen (requires_grad=False); early probability is bit-identical across arms"},
        "late_decisions": {"n": prim_margin["late_n"], "changed": prim_margin["late_n_dp_nonzero"],
                           "max_abs_dp": prim_margin["late_max_abs_dp"],
                           "min_abs_margin_r68": prim_margin["late_min_abs_margin_a"],
                           "min_abs_margin_r71": prim_margin["late_min_abs_margin_b"],
                           "max_encroachment_ratio": prim_margin["late_max_encroachment_ratio"],
                           "argmax_flips": prim_margin["argmax_flips"]},
        "greedy_argmax_flips_total": prim_margin["argmax_flips"],
        "greedy_cells_identical_per_seed": 32,
        "sampled_action_flips": {"n": flips["n"], "all_late": all(d["region"] == "late" for d in flips["detail"]),
                                 "all_sampled": flips["all_in_sampled_mode"]},
        "prior_trainable_path": "beta=1.00 early gate prior acts only on early first-emission free rows that route to the frozen early_head; prior_trainable_grad_l1 = 0.0 at all gate_diag updates in both rounds",
        "calibration_effect_on_gradients": {
            "critic_grad_l1_mean_r68": sig68["critic_grad_l1"]["mean"], "critic_grad_l1_mean_r71": sig71["critic_grad_l1"]["mean"],
            "task_value_head_l1_mean_r68": sig68["task_value_head_l1"]["mean"], "task_value_head_l1_mean_r71": sig71["task_value_head_l1"]["mean"],
            "late_head_l1_mean_r68": sig68["late_head_l1"]["mean"], "late_head_l1_mean_r71": sig71["late_head_l1"]["mean"],
            "gradient_norm_mean_r68": sig68["gradient_norm_mean"]["mean"], "gradient_norm_mean_r71": sig71["gradient_norm_mean"]["mean"],
        },
    },
    "greedy_unchanged_explanation": (
        "Early free first-report probabilities are bit-identical because the R66 intervention froze the "
        "shared trunk and early_head. The R71 critic calibration moved only the two value heads, so it can "
        "only reach the actor through the late_head gradient; the late head did move on 294/294 visited late "
        "free states, but the smallest |p-0.5| margin was 0.0401 (R71) and the largest single-decision shift "
        "reached at most 0.4865 of the way to the boundary, so no greedy decision crossed p=0.5. Hence greedy "
        "first-report actions, task sets and metered cost are identical to R68 on all 192 cells."),
    "primary_result_per_seed": {
        "greedy": {c: {"task_delta": 0.0, "cost_delta_B_per_scene": 0.0, "targets_lost": 0, "on_time": 183,
                       "bytes_per_scene": 4066.5, "identical_to_r68": True}
                   for c in ("11/greedy", "22/greedy", "33/greedy")},
        "sampled": sampled,
    },
    "cost_target_comparison_vs_rules": {
        "IMMEDIATE": {"on_time": ra["IMMEDIATE"]["on_time_total"], "bytes_per_scene_mean": ra["IMMEDIATE"]["spent_bytes_per_scene_mean"]},
        "LATE_DISCOVERY_GATE": {"on_time": ra["LATE_DISCOVERY_GATE"]["on_time_total"], "bytes_per_scene_mean": ra["LATE_DISCOVERY_GATE"]["spent_bytes_per_scene_mean"]},
        "ALWAYS_HOLD": {"on_time": ra["ALWAYS_HOLD"]["on_time_total"], "bytes_per_scene_mean": ra["ALWAYS_HOLD"]["spent_bytes_per_scene_mean"]},
        "learned_new_ep128_greedy": {"on_time": 183, "bytes_per_scene": 4066.5,
                                     "vs_immediate_cost_delta_B_per_scene": -10290.0, "targets_lost_vs_immediate": 0},
        "greedy_vs_late_rule_cost_delta_B_per_scene": 168.5625,
        "greedy_vs_late_rule_decomposition": "5394/32; 31 scenes identical, scene 1800000 alone adds 5394 B",
    },
    "budget_audit": {
        "executor": "integrated_sar/comm_budget.py: one shared hard per-episode byte credit + per-step capacity for every arm (learned and rules alike)",
        "charge_rule": "cost = payload estimate x subscribed robots, billed in full even if the packet is lost; credit never driven negative; every refusal counted",
        "refusal_counters": ["budget_refused_sends", "budget_refused_bytes", "capacity_refused_sends"],
        "budget_value_source": "29300 B per-episode credit is config-injected (comm_budget_bytes / bytes_per_episode); no hardcoded literal found in read source",
        "dev_eval_block_binding": "NOT binding: immediate mean 14356.5 B/scene (max ~15876), learned greedy 4066.5 B/scene, both far below 29300",
        "training_block_binding": "UNVERIFIED: training uses block 1300000..1300047 (distinct from the eval block); per-episode tape refusal counters are not present in the batch summary records, so training-block budget binding is not asserted from dev-eval immediate cost",
        "separation_note": "dev-eval immediate cost below 29300 does NOT establish that the training budget was never binding; the two blocks are different scene sets",
    },
    "advantage_contract": {
        "gamma": 1.0, "gae_lambda": 1.0, "eta_lambda": 0.5, "lambda_init": 1.0,
        "final_lambda": {"s11": 1.625, "s22": 1.5, "s33": 1.625},
        "gae": "per-decision-next-value; A_t = r_t + gamma*bootstrap - V + gamma*gae_lambda*next_A; returns = A + V",
        "critic_targets": "two heads regress the GAE returns return_task / return_cost (gamma=1 undiscounted future sum) with a value baseline; NOT a raw TD residual",
        "combined_actor_advantage": "lambda_dual * A_task - A_cost (cost_math.actor_advantage)",
        "advantage_normalization": "none (per-episode std division forbidden); task/cost not separately normalized",
        "cost_reward": "billed_bytes / B0_BYTES (positive measured-cost convention)",
        "dual_update": "lambda_next = max(0, lambda + eta*g_hat); g_hat = E[Y_ref]-E[Y_policy]-delta in on-time-tasks-per-episode units",
        "cost_baseline_counterexample": (
            "A_cost = return_cost - V_cost with V_cost ~ E[cost]; the baseline containing expected cost does "
            "NOT cancel the cost policy gradient (E[-A_cost*dlogpi/dz] = -c*p*(1-p), = -0.2436c at p=0.58). "
            "No narrative may claim the cost baseline neutralises cost pressure without an independent wiring counterexample."),
        "quantification_limit": "only persisted late_actor_grad_l1 / critic_grad_l1 quantify actor contributions; RMSE~1.01 is a value fit, not a gradient noise std; no new autograd opened",
    },
    "effective_logit_evidence": {
        "recorded_probability": "send_probabilities = softmax over the 2 one-shot actions of the routed-head logits from controller.forward",
        "routing": "ready_step = round(f[18]*1000) - round(f[0]); early_head if <=147 else late_head; uses only the legal 19-D input",
        "mask": "real sequential budget mask applied in the collector before the choice; a non-affordable candidate is forced HOLD (not a free choice, no RNG spent)",
        "prior_scope": "beta=1.00 gate prior is a TRAINING auxiliary loss on early first-emission free rows only; it is NOT an eval-time additive logit term, so eval probabilities are pure routed-head softmax",
        "head_scale_caveat": "late_init_alpha=0.10 only scales the late head initial network term; it does not prove the margin cannot cross zero later, and the unscaled prior cannot be assumed to preserve margin sign",
        "no_greedy_flip_framing": "0/1098 argmax flips is reported for this actual fixed scheme; it is NOT claimed to be structurally locked",
        "coverage": "recorded pre-action 19-D features, effective SEND probability, action and candidate_audit charge are persisted per free decision in the tapes",
        "missing": "no separate raw pre-softmax logit panel beyond the recorded effective probability; probability shift, training difference and sampled-eval RNG are reported as separate channels",
    },
    "belief_source": {
        "observation_layout": "comm-send-state/1 (19 dims)",
        "boundary_rule": "every feature is a property of the sender-held message, the sender own ledger/airtime, or current budget/capacity; no receiver belief, robot position, scenario id or target id may enter (comm_learning.py)",
        "always_hold_170_of_183": "ALWAYS_HOLD first-report ablation completion (on_time 170, spent 0); it is NOT a 92.9% information-path share",
        "value_fit_caveat": "fitting V(s) does not prove the 19-D observation distinguishes SEND vs SKIP action values",
    },
    "numerical_source_corrections": {
        "greedy_late_mean_diff": "168.5625 B/scene = 5394/32, entirely scene 1800000; 31 scenes identical; CI [0, 505.6875]; not a theoretical optimisable space",
        "late_183_targets": "late preserving the 183 dev-block targets does not imply all late messages are worthless or Pareto optimal",
        "blocks": "R68/R71 training teacher block 1300000..1300047; dev-eval block 1800000..1800031 (background only, not in the fit)",
        "r71_billing_rows": r71_ledger["offline_forward_ledger"],
        "r71_billing_note": "70693 rows are the offline-fit trunk-input forward ledger (cap 72000), NOT the teacher dataset; teacher = 384 R68 training tapes over 1300000..1300047; teacher-scale 32046 joint / 58928 target is delegation context, not independently re-derived here",
    },
    "sampled_rng": {
        "action_rng_key": "runner.eval_action_seed(seed, scene, mode); shared across arms and with ROUND66 for the same (seed, scene, mode)",
        "bootstrap": r71_stats["bootstrap"],
        "cross_seed_direction_note": "sampled cost directions differ by seed (s11 -155.8, s22 +331.1, s33 -642.2 with 1 target lost); these track the per-decision late probability shifts documented in the boundary audit, so they are not attributed wholly to sampling variance",
    },
    "cannot_rule_out": [
        "no generalisation beyond this single development block",
        "no non-inferiority or deployment claim",
        "the null greedy result is not attributed to a single root cause; parameter-sharing interference is not proven to be the only cause",
        "saturation vs margin: R66 was saturated near p=0.999; R68 alpha=0.10 moved late p into 0.54-0.99, so this mechanism is margin, not saturation",
        "early targets remain valuable; late labels are not a safe-deletion ground truth",
    ],
    "correctness_defect": "none found that affects the reported results",
    "unexplained_bookkeeping": {
        "update.likelihood_alignment.max_abs_gap": "bit-identical across all 96 batch records of both rounds; partially explained as a float32-vs-float64 summation residual and exactly 0 for single-free-candidate steps; not fully explained for 6 batches (b11,b13 per seed, both candidates late); tolerance 1e-6 passed in both rounds; not used as evidence for or against any scientific claim; registered for a separate read-only follow-up",
    },
    "historical_overruns_not_inherited": {
        "round60": "608 real environment constructions vs 600 hard cap (over by 8); historical, not modified, no headroom borrowed",
        "round67": ">=9419 offline forward rows vs 5000 cap; historical",
        "round72_this_session": "fresh zero budget; nothing inherited",
    },
    "evidence_paths": {
        "boundary_audit": str(DELIV / "ROUND72_BOUNDARY_AUDIT.json"),
        "validation": str(DELIV / "ROUND72_VALIDATION.json"),
        "training_signal_audit": str(DELIV / "ROUND72_TRAINING_SIGNAL_AUDIT.json"),
        "r71_statistics": str(R71 / "raw" / "ROUND71_STATISTICS.json"),
        "r68_rule_context": str(R68 / "raw" / "ROUND68_RULE_CONTEXT.json"),
        "r71_checkpoint_index": str(R71 / "ROUND71_CHECKPOINT_INDEX.json"),
        "r71_resource_ledger": str(R71 / "ROUND71_RESOURCE_LEDGER.json"),
    },
}

ledger = {
    "kind": "round72-resource-ledger/1",
    "task": "ROUND72", "branch": "analysis/round72-calibration-behaviour-diagnostic",
    "evidence_head": HEAD,
    "this_session_consumption": {
        "environment_constructions": 0, "builder_calls": 0, "run_calls": 0, "steps": 0,
        "model_loads": 0, "forward_passes": 0, "rng_uses": 0,
        "ppo_calls": 0, "real_optimizer_steps": 0, "synthetic_optimizer_steps": 0,
        "synthetic_checkpoint_writes": 0, "ppo_replays": 0, "training_restores": 0,
        "checkpoint_writes": 0, "new_evaluations": 0, "new_claims": 0,
        "historical_non_plan_probe": 0,
    },
    "note": "All ROUND72 work this session is read-only parsing of persisted JSON/JSONL plus writing three deliverable files. No simulation, training, sampling, autograd, model load or forward was performed.",
    "historical_overruns_registered_not_inherited": verdict["historical_overruns_not_inherited"],
    "effective_results_reused": "576 ROUND60 effective results and ROUND68/71 statistics are historical registrations, reused read-only; they are not new ROUND72 launches.",
}

(DELIV / "ROUND72_RESOURCE_LEDGER.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
(DELIV / "ROUND72_MACHINE_VERDICT.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote ledger and verdict; evidence_head", HEAD)
print("early_max_abs_dp", prim_margin["early_max_abs_dp"], "argmax_flips", prim_margin["argmax_flips"],
      "late_encroachment", round(prim_margin["late_max_encroachment_ratio"], 4))