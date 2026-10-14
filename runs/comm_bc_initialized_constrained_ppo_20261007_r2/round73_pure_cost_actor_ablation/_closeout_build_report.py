# Assemble the ROUND73 read-only close-out JSON from parsed originals. stdlib only; writes ONE new file.
import json, os, sys, hashlib, time, subprocess
BASE = os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
R = os.path.join(BASE, "round73_pure_cost_actor_ablation")
RAW = os.path.join(R, "raw")
OUT = os.path.join(R, "ROUND73_READONLY_CLOSEOUT.json")
def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
def head():
    try:
        return subprocess.run(["git", "-C", os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42"),
                               "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception as exc:
        return "git unavailable: %s" % exc

claims = jload(os.path.join(RAW, "_closeout_claims_crosswalk.json"))
g96 = jload(os.path.join(RAW, "_closeout_greedy96.json"))
chain = jload(os.path.join(RAW, "_closeout_training_chain.json"))
off = jload(os.path.join(RAW, "_closeout_offline_accounting.json"))
arch = jload(os.path.join(RAW, "_closeout_archive_runs.json"))
mech = jload(os.path.join(RAW, "_closeout_lost_target_mechanism.json"))
lic = jload(os.path.join(R, "ROUND73_LICENCE.json"))
fr = jload(os.path.join(R, "ROUND73_FREEZE.json"))
audit = jload(os.path.join(RAW, "ROUND73_ACCOUNTING_AUDIT.json"))

HARD = lic["hard_cap"]; NORMAL = lic["normal_cap"]; RESERVE = lic["reserve_cap"]
booked = claims["claims"]["total"]
complete = claims["claims"]["statuses"].get("complete", 0)
interrupted = claims["claims"]["statuses"].get("claimed_before_environment", 0)
failed_attempts = jload(os.path.join(RAW, "runtime", "ROUND73_FAILED_ATTEMPT_EVIDENCE.json"))["attempts"]

rep = {
  "kind": "round73-readonly-closeout/1",
  "task_id": "ROUND73-PURE-COST-ACTOR-ABLATION",
  "branch": "exp/round73-pure-cost-actor-ablation",
  "head_at_this_closeout": head(),
  "nature": ("READ-ONLY CLOSE-OUT. No environment construction, no model load, no forward/backward, "
             "no optimizer step, no PPO replay, no RNG restore, no checkpoint write, no new claim, "
             "no test, no re-run. Existing files were parsed and hashed only; every number below is "
             "either read from a durable original (labelled ORIGINAL) or derived by stdlib parsing of "
             "those originals (labelled DERIVED)."),
  "generated_at_unix": time.time(),
}
# ---------------------------------------------------------------- 1. ERRATA
rep["errata_1_accounting_audit"] = {
  "target": "raw/ROUND73_ACCOUNTING_AUDIT.json",
  "target_sha256_unchanged": sha(os.path.join(RAW, "ROUND73_ACCOUNTING_AUDIT.json")),
  "original_preserved": True,
  "corrections": [
    {"field": "run_attribution.scratch_totals.runs", "was": 7, "now": 8,
     "why": ("the audit's own 9-run list contains 8 scratch runs (runs 1-7 plus run 9, the post-fix "
             "scratch verification); only run 8 is the REAL-ledger run. The run COUNT was miscounted; "
             "the calls and steps were not.")},
    {"field": "run_attribution.scratch_totals.ppo_update_calls", "was": 60, "now": 60,
     "why": "unchanged and re-derived from the archive: 6+6+8+8+8+8+8+8 = 60"},
    {"field": "run_attribution.scratch_totals.physical_optimizer_steps", "was": 240, "now": 240,
     "why": "unchanged and re-derived from the archive: 60 calls x 4 physical steps = 240"},
    {"field": "run_attribution.all_runs_including_scratch", "was": {"runs": 9, "ppo_update_calls": 68,
                                                                   "physical_optimizer_steps": 272},
     "now": {"runs": 9, "ppo_update_calls": 68, "physical_optimizer_steps": 272},
     "why": "unchanged and re-derived: 1 real run (8 calls / 32 steps) + 8 scratch runs (60 calls / 240 steps)"},
    {"field": "the_unit_defect.evidence", "was": "every one of the 68 archived gate journal lines carries optimizer_steps = 4",
     "now": ("CONFIRMED by direct re-parse: all 68 archived records carry line.optimizer_steps = 4 "
             "(the field is nested under 'line', not at the top level), so 68 x 4 = 272 physical steps.")}
  ],
  "attribution_limits": [
    "the 9-run grouping is INFERRED from archive order and the fixed alphabetical label sequence; it is not an independent log",
    "which ledger each of the 8 scratch runs charged is NOT verified (the real guard's 8 events prove only the real-ledger run charged it)",
    "wall-clock timestamps of the scratch runs are NOT recorded (journal lines carry no time)",
    "the archive is an append-only quarantine of diverted journal lines; it records what was diverted, not the guard charges of each run",
  ],
  "archive_measured_separately": {
    "file": "raw/synthetic_test_fixtures/diverted_real_journal_lines/ppo_update_calls.jsonl.synthetic.jsonl",
    "sha256_whole_file": arch["archive_total_bytes"] and off["archive"]["sha256_whole_file"],
    "bytes": arch["archive_total_bytes"],
    "lines": arch["archive_total_lines"],
    "optimizer_steps_field_values": arch["optimizer_steps_values"],
    "physical_steps_sum_from_the_field": arch["optimizer_steps_sum_all"],
    "prefix_20818_bytes": arch["prefix_20818"],
    "suffix_after_boundary": arch["suffix_after_boundary"],
    "note": ("the 20818-byte prefix sha256 950f3807... matches what the real run's own receipt recorded; "
             "it proves the byte boundary only, not that qualification passed"),
  },
}

# ---------------------------------------------------------------- 2. CLAIMS / 600 CAP
rep["consumption_and_cap_accounting"] = {
  "caps_from_licence": {"hard_cap": HARD, "normal_cap": NORMAL, "reserve_cap": RESERVE,
                        "offline_input_row_cap": lic["offline_input_row_cap"],
                        "synthetic_optimizer_step_cap": lic["synthetic_optimizer_step_cap"]},
  "booked": {
    "launch_slots": claims["slots"]["total"],
    "slot_categories": claims["slots"]["categories"],
    "slot_numbers_contiguous_1_to_max": claims["slots"]["contiguous_1_to_max"],
    "claim_attempt_files": claims["claims"]["total"],
    "distinct_logical_jobs": claims["claims"]["distinct_logical_jobs"],
    "statuses": claims["claims"]["statuses"],
    "scopes": claims["claims"]["scopes"],
    "booked_environment_constructor_calls": claims["constructor_accounting"]["booked_environment_constructor_calls"],
    "booked_joint_run_constructed_calls": claims["constructor_accounting"]["booked_joint_run_constructed_calls"],
  },
  "proven": {
    "claims_with_consumed_permit_and_constructor_returned": claims["permits"]["jobs_with_consumed_permit"],
    "claims_with_row_on_disk": claims["constructor_accounting"]["claims_with_row"],
    "claims_with_1000_completed_steps": claims["claims"]["field_completed_steps"]["distinct_values"].get("1000"),
    "claims_with_0_completed_steps": claims["claims"]["field_completed_steps"]["distinct_values"].get("0"),
    "completed_steps_sum": claims["claims"]["field_completed_steps"]["sum"],
    "note": ("booked (claim counters) and proven (permit consumed + constructor_returned + row + steps) "
             "agree at 574; every booked claim constructed an environment exactly once"),
  },
  "constructor_field_audit": {
    "environment_constructor_calls": claims["claims"]["field_environment_constructor_calls"],
    "joint_run_constructed_calls": claims["claims"]["field_joint_run_constructed_calls"],
    "run_invocation_calls": claims["claims"]["field_run_invocation_calls"],
    "run_return_calls": claims["claims"]["field_run_return_calls"],
    "run_failure_calls": claims["claims"]["field_run_failure_calls"],
    "joint_run_build_failures": claims["claims"]["field_joint_run_build_failures"],
    "completed_steps": claims["claims"]["field_completed_steps"],
  },
  "interrupted_cell": {
    "logical_job": "r73eval_model_new_ep128_s33_sampled_scene1800029",
    "launch_slot": 574,
    "status_label": "claimed_before_environment",
    "environment_constructor_calls": 1,
    "joint_run_constructed_calls": 1,
    "permit_state": "consumed", "constructor_returned": True,
    "constructor_called_at_unix": 1791961323.8945768,
    "completed_steps": 0, "simulation_started": False, "run_return_calls": 0,
    "refunded": False,
    "note": ("the status label 'claimed_before_environment' must NOT be read as zero consumption: the "
             "permit was consumed, the constructor returned, one environment construction is booked and "
             "never refunded; only the simulation never started, so no result exists for this cell"),
  },
  "failed_technical_attempts": [
    {"attempt": a.get("attempt"), "scope": a.get("scope"), "outcome": a.get("outcome"),
     "environment_constructions": a["consumption"]["environment_constructions"],
     "claims": a["consumption"]["claims"], "slots": a["consumption"]["slots"],
     "optimizer_steps": a["consumption"]["optimizer_steps"]}
    for a in failed_attempts],
  "historical_non_plan_probe": 0,
  "gate_phase_environment_constructions": 0,
  "hard_cap_accounting": {
    "environment_constructions_consumed": booked,
    "hard_cap": HARD,
    "hard_headroom": HARD - booked,
    "normal_cap": NORMAL, "normal_headroom": NORMAL - booked,
    "reserve_used": 0, "reserve_headroom": RESERVE,
    "within_cap": booked <= HARD,
    "attempts_including_the_two_zero_construction_failures": booked + len(failed_attempts),
  },
  "planned_but_not_run": {
    "model_eval_cells_without_result": 3,
    "focal_cells_without_result": 3,
    "detail": ("192 model_eval planned: 189 complete + 1 interrupted (slot 574) + 2 never claimed; "
               "3 focal (scene 1700027) never claimed; stats phase never run"),
  },
}
# ---------------------------------------------------------------- 3. OFFLINE TRUNK + SYNTHETIC STEPS
rep["offline_trunk_and_synthetic_steps"] = {
  "forward_ledger_durable_ORIGINAL": off["forward_ledgers"]["real"],
  "forward_ledger_scratch_copy_ORIGINAL": off["forward_ledgers"]["debug_scratch_copy"],
  "forward_ledger_cap_probe_fixture_ORIGINAL": off["forward_ledgers"]["cap_probe_fixture"],
  "billing_unit": ("19-D trunk input-vector rows handed to the trunk (ForwardLedger, R71 seam); "
                   "'calls' counts debit() invocations, 'rows' the charged rows, 'refusals' refused charges, "
                   "'joint_rows' informational only"),
  "itemised_charges_DERIVED_from_the_reconstructed_receipt": {
    "source": off["receipt_reconstructed"]["path"],
    "reconstructed": off["receipt_reconstructed"]["reconstructed_flag"],
    "charge_entries": off["receipt_reconstructed"]["charge_entries"],
    "charge_rows_sum": off["receipt_reconstructed"]["charge_rows_sum"],
    "recorded_single_row_forwards": off["receipt_reconstructed"]["single_row_forwards_recorded"],
    "recorded_single_row_rows": off["receipt_reconstructed"]["single_row_total"],
    "aggregate_precharge_entries": off["receipt_reconstructed"]["aggregate_precharges"],
    "aggregate_precharge_rows": off["receipt_reconstructed"]["aggregate_precharge_total"],
    "scratch_copy_receipt_same_figures": {
      "charge_entries": off["receipt_scratch_copy"]["charge_entries"],
      "charge_rows_sum": off["receipt_scratch_copy"]["charge_rows_sum"],
      "single_row_forwards_recorded": off["receipt_scratch_copy"]["single_row_forwards_recorded"],
      "aggregate_precharge_rows": off["receipt_scratch_copy"]["aggregate_precharge_total"],
    },
  },
  "unexplained_gap": {
    "durable_rows": off["forward_ledgers"]["real"]["rows"],
    "itemised_rows": off["receipt_reconstructed"]["charge_rows_sum"],
    "difference_rows": off["forward_ledgers"]["real"]["rows"] - off["receipt_reconstructed"]["charge_rows_sum"],
    "durable_calls": off["forward_ledgers"]["real"]["calls"],
    "itemised_calls": off["receipt_reconstructed"]["charge_entries"],
    "difference_calls": off["forward_ledgers"]["real"]["calls"] - off["receipt_reconstructed"]["charge_entries"],
    "status": "UNKNOWN attribution: 12 rows / 6 calls are charged to the persisted ledger but itemised in no surviving file",
  },
  "q5_charge_model": ("the per-update charge is an AGGREGATE PRE-CHARGE of 2 * n_records * EPOCHS_PER_BATCH rows "
                      "debited before the update runs; it is a formula, not a measured forward count"),
  "actual_measured_trunk_rows_and_calls": "UNKNOWN / bounds only - no durable record measures them",
  "bounds": {
    "lower_bound_rows": off["receipt_reconstructed"]["single_row_total"],
    "lower_bound_note": "the explicitly recorded single-row forwards, from a derived (reconstructed) receipt",
    "upper_bound_rows": off["forward_ledgers"]["real"]["rows"],
    "upper_bound_note": "everything ever charged to the persisted ledger, including aggregate pre-charges",
    "scratch_closure": ("the debug_scratch ledger copy reads 1018 rows / 124 calls, i.e. 12 rows / 6 calls "
                        "less than the durable 1030 / 130; per-run attribution between the real run and the "
                        "8 scratch runs is NOT recorded"),
  },
  "synthetic_optimizer_steps": {
    "guard_durable_ORIGINAL": off["synthetic_step_guards"]["real"],
    "charged_units": off["synthetic_step_guards"]["real"]["used"],
    "charged_units_meaning": "one unit per trainer.ppo_update call (the defect), NOT one unit per optimizer.step",
    "cap": lic["synthetic_optimizer_step_cap"],
    "physical_steps_real_ledger_run": 32,
    "physical_steps_all_archived_runs": arch["optimizer_steps_sum_all"],
    "compliance_against_cap_8_physical_steps": False,
    "overrun_real_run": 32 - lic["synthetic_optimizer_step_cap"],
    "overrun_all_runs": arch["optimizer_steps_sum_all"] - lic["synthetic_optimizer_step_cap"],
    "fixtures": {"cap_probe": off["synthetic_step_guards"]["cap_probe_fixture"],
                 "contention": off["synthetic_step_guards"]["contention_fixture"],
                 "debug_scratch_copy": off["synthetic_step_guards"]["debug_scratch_copy"]},
  },
  "freeze_and_licence_identity": {
    "note": "ROUND73_FREEZE.json and ROUND73_LICENCE.json live in the ROUND73 ROOT directory, not under raw/; the earlier _full_shas.py MISSING entries were a wrong-path lookup",
    "ROUND73_FREEZE.json_sha256": sha(os.path.join(R, "ROUND73_FREEZE.json")),
    "ROUND73_LICENCE.json_sha256": sha(os.path.join(R, "ROUND73_LICENCE.json")),
    "licence_source_freeze_sha256_expected": lic["source_freeze_sha256"],
    "licence_source_freeze_sha256_actual": sha(os.path.join(BASE, "round68_late_initial_confidence_ablation", "ROUND68_FREEZE.json")),
    "licence_input_identity_sha256_expected": lic["input_identity_sha256"],
    "licence_input_identity_sha256_actual": sha(os.path.join(BASE, "round71_critic_initialization_calibration", "ROUND71_CHECKPOINT_INDEX.json")),
  },
}

# ---------------------------------------------------------------- 4. GREEDY 96
ps = g96["per_seed_greedy"]
def seed_block(s):
    b = ps[s]
    return {
      "cells": b["cells"],
      "r73_on_time_total": b["r73_on_time_total"], "r71_on_time_total": b["r71_on_time_total"],
      "on_time_delta_total": b["on_time_delta_total"],
      "r73_bytes_mean": b["r73_bytes_mean"], "r71_bytes_mean": b["r71_bytes_mean"],
      "bytes_delta_mean": b["bytes_delta_mean"],
      "scenes_cost_down_equal_up": [b["scenes_with_cost_decrease"], b["scenes_with_cost_equal"], b["scenes_with_cost_increase"]],
      "scenes_with_task_loss": b["scenes_with_task_loss"], "scenes_with_task_gain": b["scenes_with_task_gain"],
      "targets_lost_vs_r71": b["targets_lost_vs_r71"], "targets_gained_vs_r71": b["targets_gained_vs_r71"],
      "lost_detail": b["lost_detail"], "gained_detail": b["gained_detail"],
      "send_skip_early_late": b["send_skip_early_late"], "send_skip_delta": b["send_skip_delta"],
      "bytes_delta_per_scene": b["bytes_delta_per_scene"],
      "lost_targets_were_late_sends_in_r71": mech["s" + s]["all_lost_were_late_sends_in_r71"],
      "lost_targets_are_late_skips_in_r73": mech["s" + s]["all_lost_are_late_skips_in_r73"],
      "training_chain": chain["seeds"]["s" + s],
    }
rep["greedy_96_paired_comparison"] = {
  "scope": "descriptive primary comparison only (NEW_EP128 - R71_EP128), greedy mode, same seed, same 32 scenes",
  "not_done": ["no bootstrap", "no new significance method", "no pooled seeds", "no overall verdict",
               "no mode selection", "sampled and focal cells NOT mixed in"],
  "identity_audit": g96["identity_audit"],
  "per_seed": {s: seed_block(s) for s in ("11", "22", "33")},
  "rule_arms_read_only": g96["rule_arms"],
  "cross_seed_observation": {
    "per_scene_results_identical_across_the_three_seeds_in_both_arms": True,
    "note": ("the three models are NOT identical (different parameter digests and different frozen-module "
             "digests), yet every per-scene on-time set and byte total coincides across seeds in BOTH arms; "
             "the three seeds therefore do not provide independent replication on this block"),
  },
  "discovered_ceiling": {
    "scenes_with_fewer_than_6_discovered": [1800003, 1800007, 1800008, 1800012, 1800013, 1800019, 1800022, 1800029],
    "identical_in_both_arms": True,
    "note": "in these 8 scenes the search module discovers 5 (or 4) goals, so the on-time ceiling is below 6 for BOTH arms",
  },
  "gaps": g96["gaps"],
}
# ---------------------------------------------------------------- 5. CLAIM<->ROW BIJECTION (re-derived)
# Independently re-derived from the claim attempt files and the row files themselves, because
# _closeout_claims_audit.json reports complete_claims_without_row = 573, which its own sibling
# _closeout_claims_crosswalk.json contradicts (573 matched). The audit field is a label-key
# artefact: it compared claim logical_job names against row labels, which never match because
# eval rows carry a per-(seed,mode) group label and train claims carry row_tag = None.
rep["claim_row_bijection_re_derived"] = {
  "method": ("regex over the 574 claim attempt files and the 573 row files; train claims matched on "
             "logical_job r73_new_training_(s\\d+)_e(\\d+) against row file name v2_r73_train_..._e(\\d+)_...; "
             "eval claims matched on logical_job r73eval_model_new_ep128_(s\\d+)_(greedy|sampled)_scene(\\d+) "
             "against row file name v2_r73eval_model_new_ep128_..._s(\\d+)_..."),
  "claims_total": 574,
  "claims_complete": 573,
  "claims_not_complete": 1,
  "rows_on_disk": 573,
  "complete_claims_without_a_row": 0,
  "rows_without_a_complete_claim": 0,
  "the_single_not_complete_claim": {"logical_job": "r73eval_model_new_ep128_s33_sampled_scene1800029",
                                    "status": "claimed_before_environment"},
  "train_rows_by_seed": {"s11": 128, "s22": 128, "s33": 128},
  "train_episodes_contiguous_0_127_per_seed": True,
  "eval_rows_by_seed_mode": {"s11|greedy": 32, "s11|sampled": 32, "s22|greedy": 32, "s22|sampled": 32,
                             "s33|greedy": 32, "s33|sampled": 29},
  "eval_scenes_contiguous_1800000_1800031": {"s11|greedy": True, "s11|sampled": True, "s22|greedy": True,
                                             "s22|sampled": True, "s33|greedy": True, "s33|sampled": False},
  "conclusion": ("a perfect 1:1 bijection between the 573 complete claims and the 573 result rows; "
                 "the audit field complete_claims_without_row = 573 and the crosswalk field "
                 "eval_coverage.row_scenes = 0 are both label-normalisation artefacts, NOT missing rows"),
}

# ---------------------------------------------------------------- 6. INTERRUPTED CELL, MEASURED
rep["interrupted_cell_measured_evidence"] = {
  "logical_job": "r73eval_model_new_ep128_s33_sampled_scene1800029",
  "launch_slot": 574,
  "booked_counters": {"environment_constructor_calls": 1, "joint_run_constructed_calls": 1,
                      "joint_run_method_calls": 1, "run_invocation_calls": 1, "run_return_calls": 0,
                      "run_failure_calls": 0, "joint_run_build_failures": 0, "completed_steps": 0},
  "status_label": "claimed_before_environment",
  "status_label_is_factually_wrong": True,
  "why": ("the label says 'before environment', but the constructor was called at unix 1791961323.8945768 "
          "and returned, the permit is consumed with constructor_returned = true, and one environment "
          "construction is booked and never refunded"),
  "partial_artefacts_on_disk": {
    "relay_policy_log": "raw/logs/v2_r73eval_model_new_ep128_s33_sampled_s1800029_a1_rp.jsonl",
    "relay_policy_log_lines": 972,
    "complete_s33_sampled_relay_policy_log_line_range": [1355, 1447],
    "controller_log_present": False,
    "tape_log_present": False,
    "result_row_present": False,
  },
  "interpretation": ("a partial episode was executed (972 relay-policy log lines versus 1355-1447 for the "
                     "29 complete s33-sampled cells) and then interrupted; no controller log, no tape and "
                     "no result row were produced. The booked completed_steps = 0 and simulation_started = "
                     "false flags were never updated because the run never returned, so they must NOT be "
                     "read as proof that zero simulation work happened"),
  "exact_simulation_step_count": "UNKNOWN - not recorded by any durable file",
  "consumption_charged": 1,
  "refunded": False,
}
# ---------------------------------------------------------------- 7. STALE FILES (append-only; NOT rewritten)
rep["stale_files_needing_correction"] = {
  "policy": ("these files are left EXACTLY as they are. They are pre-launch snapshots whose numbers were "
             "overtaken by the run; the claim files are the source of truth. Listing them here is the "
             "correction; overwriting them would destroy the visible difference"),
  "files": [
    {"path": "ROUND73_MACHINE_VERDICT.json",
     "sha256": sha(os.path.join(R, "ROUND73_MACHINE_VERDICT.json")),
     "says": "status 'fixed_training_and_eval_NOT_started'; every consumption field 0; scientific_conclusion DEFERRED",
     "actually": "384 training + 189 model_eval completed, 573 rows on disk, 574 booked",
     "note": "root-level verdict written before launch; never refreshed"},
    {"path": "ROUND73_LEDGER.json",
     "sha256": sha(os.path.join(R, "ROUND73_LEDGER.json")),
     "says": "every consumption field 0; headroom hard 600 / normal 579 / reserve 21",
     "actually": "574 booked; hard headroom 26, normal headroom 5, reserve 0",
     "note": "root-level pre-launch ledger"},
    {"path": "raw/runtime/ROUND73_LEDGER.json",
     "sha256": sha(os.path.join(RAW, "runtime", "ROUND73_LEDGER.json")),
     "says": "real_environment_constructions 384; headroom hard 216 / normal 195",
     "actually": "stale by 190: it was last written at the end of the training phase, before eval booked anything",
     "note": "the last written ledger file; the claim files supersede it"},
    {"path": "raw/runtime/ROUND73_EVAL_QUOTA_GUARD.json",
     "sha256": sha(os.path.join(RAW, "runtime", "ROUND73_EVAL_QUOTA_GUARD.json")),
     "says": "booked_normal 384, booked_total 384, hard_headroom 216",
     "actually": "booked_total is 574; hard_headroom 26",
     "note": "eval-phase guard snapshot taken at eval start"},
    {"path": "ROUND73_FREEZE.json",
     "sha256": sha(os.path.join(R, "ROUND73_FREEZE.json")),
     "says": "phases_completed train false, eval false, stats false",
     "actually": "train and eval ran (with 6 planned cells not run); stats never ran",
     "note": "the freeze identity fields themselves are correct and unchanged; only phases_completed is stale"},
  ],
  "numbers_that_must_not_be_quoted_as_current": [
    "600 hard headroom (actual 26)",
    "579 normal headroom (actual 5)",
    "21 reserve headroom (actual 0 used, 21 available)",
    "384 booked (actual 574)",
    "0 environment constructions (actual 574)",
    "8 synthetic optimizer steps respected (actual: 8 charged UNITS = 8 ppo_update calls; 32 physical optimizer steps in the real-ledger run and 272 across all 9 archived runs)",
  ],
}

# ---------------------------------------------------------------- 8. RULE-ARM PAIRING (descriptive)
ra = g96["rule_arms"]
rep["rule_arm_pairing_descriptive"] = {
  "scope": ("read-only reuse of the frozen R66 rule-context rows on the SAME 32 scenes 1800000..1800031; "
            "no new environment construction; descriptive only, no bootstrap, no significance claim"),
  "per_scene_on_time": {
    "IMMEDIATE": ra["IMMEDIATE"]["on_time_total"],
    "LATE_DISCOVERY_GATE": ra["LATE_DISCOVERY_GATE"]["on_time_total"],
    "ALWAYS_HOLD": ra["ALWAYS_HOLD"]["on_time_total"],
    "R71_EP128_control": g96["per_seed_greedy"]["11"]["r71_on_time_total"],
    "R73_EP128_new": g96["per_seed_greedy"]["11"]["r73_on_time_total"],
  },
  "billed_bytes_per_scene": {
    "IMMEDIATE": ra["IMMEDIATE"]["bytes_per_scene"],
    "LATE_DISCOVERY_GATE": ra["LATE_DISCOVERY_GATE"]["bytes_per_scene"],
    "ALWAYS_HOLD": ra["ALWAYS_HOLD"]["bytes_per_scene"],
    "R71_EP128_control": g96["per_seed_greedy"]["11"]["r71_bytes_mean"],
    "R73_EP128_new": g96["per_seed_greedy"]["11"]["r73_bytes_mean"],
  },
  "billed_bytes_total": {
    "IMMEDIATE": ra["IMMEDIATE"]["bytes_total"],
    "LATE_DISCOVERY_GATE": ra["LATE_DISCOVERY_GATE"]["bytes_total"],
    "ALWAYS_HOLD": ra["ALWAYS_HOLD"]["bytes_total"],
  },
  "duplicate_rows_dropped": {"IMMEDIATE": ra["IMMEDIATE"]["duplicate_rows_dropped"],
                             "LATE_DISCOVERY_GATE": ra["LATE_DISCOVERY_GATE"]["duplicate_rows_dropped"],
                             "ALWAYS_HOLD": ra["ALWAYS_HOLD"]["duplicate_rows_dropped"]},
  "reading": [
    "R73 matches ALWAYS_HOLD on task count (170 on-time) but is NOT zero-billing (168.5625 B/scene); it is 13 below the IMMEDIATE / LATE_DISCOVERY_GATE / R71 arms (183 on-time)",
    "R71 matches IMMEDIATE and LATE_DISCOVERY_GATE on task (183) at 4066.5 B/scene, i.e. 168.5625 B/scene above the LATE rule arm and 10290 B/scene below IMMEDIATE",
    "the 13-target loss is therefore a move from the 183-task regime to the 170-task regime, not a partial degradation inside one regime",
  ],
  "caution": ("ALWAYS_HOLD billing 0 B is a rule arm with a different action space, not a cost-optimal learned "
              "policy; equating R73 with ALWAYS_HOLD is a descriptive coincidence on this block only"),
}
# ---------------------------------------------------------------- 9. PER-SEED TRAINING-CHAIN ELIGIBILITY
def elig(s):
    b = chain["seeds"]["s" + s]
    return {
      "seed": int(s),
      "batch_records": b["batch_records"],
      "chain_contiguous_0_to_31": b["chain_contiguous_0_to_31"],
      "batch0_parameters_before_equals_registered_init": b["batch0_before_matches_registered_init"],
      "registered_init_param_sha256": b["registered_init_param_sha"],
      "batch31_parameters_after_equals_eval_model": b["batch31_after_matches_eval"],
      "eval_param_sha256": b["eval_param_sha"],
      "episodes_done_after_final": b["episodes_done_after_final"],
      "optimizer_steps_after_final": b["optimizer_steps_after_final"],
      "lambda_start": b["lambda_start"], "lambda_final": b["lambda_final"],
      "actor_objective": b["actor_objective"],
      "problems": b["problems"],
      "first_batch_gate": "passed",
      "init_file_sha256": fr["init_source_per_seed"]["s" + s]["file_sha256"],
      "ep128_file_sha256": sha(os.path.join(RAW, "checkpoints", "v2_train_s" + s + "_ep128.pt")),
    }
rep["per_seed_training_chain_eligibility"] = {
  "seeds": {s: elig(s) for s in ("11", "22", "33")},
  "verified_for_all_three_seeds": [
    "the INIT of every seed is the frozen R71 calibrated INIT, model-only (init file sha256 matches ROUND73_FREEZE.json)",
    "batch 0 parameters_before equals the registered INIT parameter digest",
    "the before/after parameter chain is contiguous across all 32 committed batch records",
    "batch 31 parameters_after equals the parameter digest of the ep128 model that was actually evaluated",
    "episodes_done_after_final = 128 for every seed",
    "the actor objective is pure-cost/1 on the live path (first-batch gate check actor_advantage_is_minus_a_cost)",
    "lambda starts at 1.0 and is updated and reported",
  ],
  "optimizer_steps_sum_across_seeds": sum(chain["seeds"]["s" + s]["optimizer_steps_after_final"] for s in ("11", "22", "33")),
  "real_ppo_journal": {"path": "raw/runtime/ppo_update_calls.jsonl", "lines": 96,
                       "optimizer_steps_sum": 892,
                       "sha256": sha(os.path.join(RAW, "runtime", "ppo_update_calls.jsonl"))},
  "real_optimizer_journal": {"path": "raw/runtime/optimizer_step_calls.jsonl", "lines": 892,
                             "sha256": sha(os.path.join(RAW, "runtime", "optimizer_step_calls.jsonl"))},
  "journal_matches_chain": True,
  "checkpoint_writes_journal": {
    "path": "raw/runtime/checkpoint_writes.jsonl", "bytes": 0, "lines": 0,
    "sha256": sha(os.path.join(RAW, "runtime", "checkpoint_writes.jsonl")),
    "note": ("the checkpoint-write journal is EMPTY although 201 checkpoint files exist under raw/checkpoints "
             "(67 per seed). The write events are evidenced by the files themselves and by the parameter "
             "chain, but NOT by this journal; the journal is a bookkeeping gap, not evidence of zero writes"),
  },
  "checkpoint_files_on_disk": {
    "total_pt_files": 204,
    "under_raw_checkpoints": 201,
    "per_seed": {"init": 1, "final": 1, "ep128": 1, "before_b": 32, "after_b": 32},
    "synthetic_pt_files_outside_checkpoints": 3,
    "synthetic_pt_all_under_synthetic_test_fixtures": True,
    "ep128_equals_after_b31_equals_final_on_disk": True,
    "file_level_after_b0_differs_from_before_b1": True,
    "file_level_note": ("after_b0 and before_b1 differ as FILE BYTES (the payload carries batch metadata), "
                        "while the parameter-level chain is verified contiguous by the batch records; "
                        "re-deriving the parameter equality would need a torch load, which this close-out "
                        "does not perform"),
  },
  "qualification_budget_compliance": False,
  "qualification_ruling": ("the decision-end ruled the synthetic optimizer-step cap is denominated in physical "
                           "optimizer.step calls, so run 32 (the real-ledger gate run) and the 272 archived "
                           "scratch steps are NOT retroactively approved. The three trained models and their "
                           "results are not discarded on that basis; the non-compliance is recorded, the "
                           "remaining 6 cells are not resumed and no new quota is granted"),
}

# ---------------------------------------------------------------- 10. HELD ACTIONS
rep["held_actions"] = [
  "the 6 remaining planned cells are NOT resumed: s33 sampled scenes 1800029 (interrupted, slot 574), 1800030 and 1800031, plus the 3 focal scene-1700027 cells",
  "--phase stats is NOT run; no bootstrap, no paired interval, no overall verdict",
  "no new claim, slot, permit or environment construction is created",
  "no model is loaded, no forward/backward, no optimizer step, no PPO replay, no RNG restore, no checkpoint write",
  "ROUND73_LEDGER.json, ROUND73_MACHINE_VERDICT.json and ROUND73_FREEZE.json are NOT rewritten; they stay visibly stale",
  "no re-tuning, no extra training, no seed selection, no checkpoint selection, no re-run for performance",
]

# ---------------------------------------------------------------- 11. CONCLUSIONS
rep["conclusions"] = {
  "may_be_kept": [
    "the pure-cost actor objective (A = -A_cost) was live on the real training path for all 384 episodes: the first-batch gate of every seed records actor_advantage_is_minus_a_cost and actor_objective pure-cost/1",
    "the three ep128 models come from a verified chain: frozen R71 INIT -> 32 contiguous committed batches -> 128 episodes -> the exact model that was evaluated",
    "on the 32-scene development block, greedy, the intervention moved the policy onto the no-report regime: on-time 183 -> 170 (-13) and billed bytes 4066.5 -> 168.5625 B/scene (-3897.9375), with 21 scenes cheaper, 11 equal and 0 dearer",
    "the task loss is fully mechanistically explained: every one of the 13 lost targets per seed was a LATE first report that R71 SENT and R73 SKIPPED; the early path is bit-identical (2 SEND / 132 SKIP in both arms)",
    "the three seeds produce identical per-scene results in both arms, so they do not provide independent replication on this block",
  ],
  "must_be_deferred": [
    "any claim that the synthetic optimizer-step cap was respected: it was charged in PPO-update-call units, not the physical optimizer.step units the guard and licence declare",
    "any claim that qualification budget compliance holds: run 32 and the 272 archived scratch steps are not approved",
    "the sampled-mode reading: 3 of 96 sampled cells have no result (s33 scenes 1800029/1800030/1800031), so the sampled arm is incomplete and is not reported as a paired comparison",
    "the 3 focal scene-1700027 cells: never claimed, no result",
    "any statistical inference: the stats phase never ran, so there is no interval, no significance statement and no non-inferiority claim",
    "any generalisation, deployment or production claim",
    "any attribution of the -13 task change to the actor objective alone rather than to the whole R73 configuration delta",
  ],
}
# ---------------------------------------------------------------- 12. ZERO SIDE-EFFECT COUNTS
rep["zero_side_effect_counts"] = {
  "scope": "counts for THIS read-only close-out only",
  "environment_constructions": 0,
  "environment_runs": 0,
  "simulation_steps": 0,
  "model_loads": 0,
  "cpu_forwards": 0,
  "backward_passes": 0,
  "real_optimizer_steps": 0,
  "synthetic_optimizer_steps": 0,
  "ppo_replays": 0,
  "rng_restores": 0,
  "checkpoint_writes": 0,
  "new_claims": 0,
  "new_slots": 0,
  "new_permits": 0,
  "tests_run": 0,
  "processes_started_for_the_experiment": 0,
  "files_written_inside_the_round_directory": [
    "ROUND73_READONLY_CLOSEOUT.json",
    "ROUND73_READONLY_CLOSEOUT.md",
  ],
  "originals_modified": 0,
  "originals_deleted": 0,
  "note": ("only stdlib JSON/text parsing and sha256 hashing of existing files was performed; "
           "no torch import, no model file was opened as a model, no ledger or journal was written"),
}

# ---------------------------------------------------------------- 13. KEY ORIGINALS + FULL HASHES
prot = os.path.join(BASE, "ROUND46_CURRENT_TASK.json")
prot2 = os.path.join(BASE, "ROUND46_R2_V6_STOP_STATE.json")
rep["key_originals_and_hashes"] = {
  "protected_round46_files": {
    "ROUND46_CURRENT_TASK.json": {"sha256": sha(prot),
                                  "expected": "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
                                  "match": sha(prot) == "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b"},
    "ROUND46_R2_V6_STOP_STATE.json": {"sha256": sha(prot2),
                                      "expected": "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18",
                                      "match": sha(prot2) == "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18"},
    "modified_or_staged": False,
  },
  "freeze_and_licence": {
    "ROUND73_FREEZE.json": sha(os.path.join(R, "ROUND73_FREEZE.json")),
    "ROUND73_LICENCE.json": sha(os.path.join(R, "ROUND73_LICENCE.json")),
    "ROUND73_EXECUTION_BASIS.md": sha(os.path.join(R, "ROUND73_EXECUTION_BASIS.md")),
  },
  "init_checkpoints": {"s11": sha(os.path.join(RAW, "checkpoints", "v2_train_s11_init.pt")),
                       "s22": sha(os.path.join(RAW, "checkpoints", "v2_train_s22_init.pt")),
                       "s33": sha(os.path.join(RAW, "checkpoints", "v2_train_s33_init.pt"))},
  "ep128_checkpoints": {"s11": sha(os.path.join(RAW, "checkpoints", "v2_train_s11_ep128.pt")),
                        "s22": sha(os.path.join(RAW, "checkpoints", "v2_train_s22_ep128.pt")),
                        "s33": sha(os.path.join(RAW, "checkpoints", "v2_train_s33_ep128.pt"))},
  "journals": {
    "raw/runtime/ppo_update_calls.jsonl": sha(os.path.join(RAW, "runtime", "ppo_update_calls.jsonl")),
    "raw/runtime/optimizer_step_calls.jsonl": sha(os.path.join(RAW, "runtime", "optimizer_step_calls.jsonl")),
    "raw/runtime/r73_forward_ledger.json": sha(os.path.join(RAW, "runtime", "r73_forward_ledger.json")),
    "raw/runtime/checkpoint_writes.jsonl": sha(os.path.join(RAW, "runtime", "checkpoint_writes.jsonl")),
  },
  "synthetic_gate_material": {
    "diverted_archive": sha(os.path.join(RAW, "synthetic_test_fixtures", "diverted_real_journal_lines",
                                         "ppo_update_calls.jsonl.synthetic.jsonl")),
    "diverted_archive_debug_scratch_copy": sha(os.path.join(RAW, "synthetic_test_fixtures", "debug_scratch",
                                                            "diverted_real_journal_lines",
                                                            "ppo_update_calls.jsonl.synthetic.jsonl")),
    "reconstructed_receipt": sha(os.path.join(RAW, "synthetic_test_fixtures",
                                              "R73_SYNTHETIC_PPO_JOURNAL_RECEIPT.json")),
    "preserved_scratch_receipt": sha(os.path.join(RAW, "synthetic_test_fixtures",
                                                  "R73_SYNTHETIC_PPO_JOURNAL_RECEIPT_SCRATCH_OVERWRITE.json")),
    "durable_synthetic_guard": sha(os.path.join(RAW, "runtime", "synthetic_gates",
                                                "r73_synthetic_optimizer_steps.json")),
  },
  "git": {
    "branch": "exp/round73-pure-cost-actor-ablation",
    "head_at_this_closeout": rep["head_at_this_closeout"],
    "base_head": fr["base_head"],
    "r71_science_start": fr["r71_science_start"],
    "pushed": False, "merged": False, "deployed": False,
  },
}



# ---------------------------------------------------------------- 11b. FROZEN MODULE IDENTITY ACROSS ARMS
def _gate_frozen(round_dir, seed):
    p = os.path.join(BASE, round_dir, "raw", "runtime", "r73_first_batch_gate_s%s.json" % seed)
    if not os.path.exists(p):
        p = os.path.join(BASE, round_dir, "raw", "runtime", "r71_first_batch_gate_s%s.json" % seed)
    if not os.path.exists(p):
        return None
    return jload(p).get("frozen_sha256")

rep["frozen_module_identity_across_arms"] = {
  "claim_under_test": "the early path (frozen trunk + early head) is untouched by the R73 intervention",
  "evidence_type": "sha256 of the frozen module set recorded by each round own first-batch gate",
  "per_seed": {s: {"r71_frozen_sha256": _gate_frozen("round71_critic_initialization_calibration", s),
                   "r73_frozen_sha256": _gate_frozen("round73_pure_cost_actor_ablation", s)}
               for s in ("11", "22", "33")},
  "observed_early_first_reports_identical_in_both_arms": True,
  "observed_early_send_skip": {"r71": {"early_send": 2, "early_skip": 132},
                               "r73": {"early_send": 2, "early_skip": 132}},
  "note": ("the frozen-module digests match per seed across the two arms and the measured early "
           "first-report behaviour is identical (2 SEND / 132 SKIP); together these support that the "
           "early path is unchanged, but they do not prove that no gradient ever reached a trainable "
           "module through the early head"),
}
for _s in ("11", "22", "33"):
    _b = rep["frozen_module_identity_across_arms"]["per_seed"][_s]
    _b["match"] = (_b["r71_frozen_sha256"] is not None and _b["r71_frozen_sha256"] == _b["r73_frozen_sha256"])
rep["frozen_module_identity_across_arms"]["all_three_seeds_match"] = all(
    rep["frozen_module_identity_across_arms"]["per_seed"][s]["match"] for s in ("11", "22", "33"))
# ---------------------------------------------------------------- WRITE
def _sk(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): _sk(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sk(v) for v in o]
    return o

with open(OUT, "wb") as f:
    f.write(json.dumps(_sk(rep), ensure_ascii=False, indent=1).encode("utf-8"))
print("wrote", OUT, os.path.getsize(OUT), "bytes")