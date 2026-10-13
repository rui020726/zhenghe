#!/usr/bin/env python
"""ROUND71 readiness machine-results generator (ZERO consumption).

Composes the machine JSON from the seam module's locked constants, the read-only offline input
count, and a fresh run of the pure-arithmetic readiness tests.  It opens no .pt file, restores
no RNG, fits no real data, runs no neural forward and writes no checkpoint.  The only "run" it
performs is the in-process pure-arithmetic test suite, which is explicitly not a real update.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r71_critic_calibration_seam as seam  # noqa: E402
import test_r71_readiness as tests  # noqa: E402


def run_tests():
    fns = [(k, v) for k, v in sorted(vars(tests).items())
           if k.startswith("test_") and callable(v)]
    passed, failed = [], []
    for name, fn in fns:
        try:
            fn()
            passed.append(name)
        except Exception as exc:                      # pragma: no cover - reported, not raised
            failed.append({"test": name, "error": f"{type(exc).__name__}: {exc}"})
    return passed, failed


def main():
    counts = json.loads((HERE / "ROUND71_OFFLINE_INPUT_COUNT.json").read_text(encoding="utf-8"))
    passed, failed = run_tests()

    out = {
        "kind": "round71-critic-calibration-readiness/1",
        "task_id": "ROUND71-CRITIC-CALIBRATION-ZERO-USE-READINESS",
        "branch": "analysis/round71-critic-calibration-readiness",
        "base_head": "92571c62b27b0c38d755b501a6d0153b630a08ed",
        "first_pass_commit": "53f9eb7d2319f295c7c087ca2edd6606cc3979f6",
        "stage": "ZERO_CONSUMPTION_PREPARATION_COMPLETE",
        "correction_pass": {
            "of_commit": "53f9eb7d2319f295c7c087ca2edd6606cc3979f6",
            "trigger": ("a delegated source review found 4 real gaps in the first-pass guard "
                        "code; all 4 were verified present in the committed blobs before "
                        "patching"),
            "consumption": 0,
        },
        "authorisation": {
            "experiment_licensed": False,
            "candidate_status": seam.CANDIDATE_PLAN["status"],
            "note": ("no human approval for a new experiment has been received; the candidate "
                     "plan is registered only and confers no run authorisation. Tests being "
                     "green does not advance the stage."),
        },
        "zero_consumption_this_round": {
            "environment_constructions": 0, "episodes": 0, "claims": 0,
            "optimizer_steps_real": 0, "optimizer_steps_synthetic": 0,
            "ppo_replays": 0, "neural_forwards": 0,
            "autograd_calls": 0, "real_data_statistical_fits": 0,
            "model_loads": 0, "pt_files_opened": 0, "rng_restores": 0,
            "checkpoint_writes": 0,
            "allowed_and_used": ("read of existing JSON batch records and tape paths for "
                                 "counting/hashing, pure in-memory arithmetic coefficient "
                                 "transform, in-memory spies, test-only private temporary "
                                 "counters in the writable scratch root"),
        },
        "candidate_plan": seam.CANDIDATE_PLAN,
        "offline_input_count": {
            "totals": counts["totals"],
            "per_seed": counts["per_seed"],
            "budget": counts["candidate_offline_budget"],
            "teacher_source": counts["teacher_data_source"],
        },
        "preparation_coverage": {
            "calibration_initialisation_seam": {
                "covered": ("exact closed-form transform of a dual-output ridge fitted on the "
                            "STANDARDISED embedding back to the raw 64-D embedding affine head "
                            "(B[j,:]=A[j,:]/sigma[j]; b=c-mu@B), plus expression of (B,b) as the "
                            "two real Linear(64,1) critic heads; only the two critic heads may "
                            "be replaced, and replacement is not in place"),
                "design_basis": ("the verified real critic input/affine-head path from ROUND70: "
                                 "value = head(mean over targets of trunk(features)), both real "
                                 "critic heads are Linear(64,1) on the frozen-trunk embedding, "
                                 "so the dual-output affine baseline is the same function class"),
                "not_done": "no real model opened, no real data fitted, no model weights produced",
            },
            "resource_and_entry_guard": {
                "covered": ("ForwardLedger debits the REAL 19-D input-vector rows BEFORE any "
                            "forward and refuses before exceeding the cap; billing unit is the "
                            "target-level input row (never the joint decision row, never the "
                            "call count); failures, retries and repeats are all charged; the "
                            "counter persists across instances so a restart does not zero it; "
                            "batches are grouped by real input rows with a 1024 cap and an "
                            "oversized single step is reported rather than silently split; "
                            "DuplicateOutputGuard refuses an implicit re-run over an existing "
                            "artefact; LicenseGate requires a VERIFIED RealLicense (a raw "
                            "string/dict token is rejected at construction, a test-only spy "
                            "verifier cannot unlock the real provider, and require refuses "
                            "both an absent and a test-only licence); CalibrationInitializer "
                            "bills ONLY the real row count at the trunk entry "
                            "(guarded_trunk_forward: 2-D (N,19) shape, N>0, N<=1024, atomic "
                            "debit of the real N, then the same matrix delegated exactly once) "
                            "and never pre-debits from estimated counts"),
                "hardening": ("persisted ledger state is validated on load (missing/corrupt/"
                              "non-int/negative/over-cap/wrong-cap/foreign-identity state "
                              "raises LedgerStateError and is never silently zeroed); persist "
                              "is same-dir temp + fsync + os.replace under a lock file that "
                              "fails closed on a residual lock unless an explicit recorded "
                              "recovery_basis is supplied (never auto-deleted); debit persists "
                              "the new state before committing in memory and records refusals "
                              "without refunding"),
                "fail_closed": ("LicenseGate refuses every gated action (real-data fit, model "
                                "forward, claim write, checkpoint write, episode call, "
                                "optimizer step, PPO replay, environment build) when no real "
                                "licence token is present, and the initialiser raises at the "
                                "FIRST gate so no collaborator runs and no charge occurs"),
                "not_claimed": ("the ledger is an in-process guard over a persisted counter; it "
                                "does NOT claim physical power-loss atomicity, and a spy/no-op "
                                "is not a real update"),
                "not_added": ("no invented conditions such as 'the receipt must live outside "
                              "the repository' or 'exactly N claim files'"),
            },
            "restore_source_legality": {
                "covered": ("validate_restore_source requires a complete identity block (seed, "
                            "role, structure, module_set, training_version, source_kind); "
                            "r68_ep0_model_only is the legal INIT source and MAY carry the "
                            "frozen INITIAL random state (not misclassified as trained RNG) "
                            "but no trained optimizer/lambda/advanced RNG; new_full_restore "
                            "may carry only THIS run's own trained state; rule_control_readonly "
                            "carries no trained state; a trained ep128, an old single-head / "
                            "four-module set, an empty module set, an incomplete identity, a "
                            "foreign structure/version, or a foreign run's trained state are "
                            "all refused"),
            },
            "read_only_input_count": counts["totals"],
        },
        "corrections": {
            "gaps_found_and_fixed": [
                ("LicenseGate.licensed was bool(self.token): any string/dict passed. Now a "
                 "verifier-accepted RealLicense is required; no verifier refuses everything; "
                 "test-only spy verifiers cannot unlock the real provider."),
                ("ForwardLedger loaded state via saved.get('rows', 0) with no identity/cap/"
                 "validation and persisted with a plain write_text. Now state is fully "
                 "validated on load (never silently zeroed), persisted atomically (same-dir "
                 "temp + fsync + os.replace) under a fail-closed lock, and debit persists "
                 "before committing in memory; refusals are recorded without refund."),
                ("CalibrationInitializer.run pre-debited all batches from ESTIMATED "
                 "target_counts and left the real matrices/call counts/batch sizes unguarded. "
                 "Now the only charge is the real row count at the trunk entry, which also "
                 "refuses wrong-shape and over-1024 real batches before any charge; estimated "
                 "counts are trace metadata only."),
                ("validate_restore_source trusted caller flags and let an EMPTY module list "
                 "pass. Now a complete identity block is required and roles/source kinds "
                 "distinguish r68_ep0_model_only from new_full_restore; future legitimate NEW "
                 "restores stay legal while foreign trained state is refused."),
            ],
            "superseded_test": ("test_legal_new_training_initialisation_is_accepted encoded the "
                                "old flag-only contract; it is superseded by "
                                "test_superseded_incomplete_initialisation_is_now_refused with "
                                "an explicit correction note, and the legal forms are covered "
                                "by test_legal_r68_ep0_model_only_source_is_accepted / "
                                "test_legal_new_full_restore_of_this_run_is_accepted"),
            "not_claimed": [
                "the ParameterContainer in the tests is a STAND-IN, not real R68 wiring; real "
                "5-module construction/install coverage is not yet executed",
                "the default trunk_delegate is an identity echo; no real trunk forward is "
                "wired or run",
                "no power-loss atomicity is claimed for the ledger",
                "0 .pt opens / 0 model loads / 0 RNG restores / 0 checkpoint writes / 0 real or "
                "synthetic optimizer updates in the correction pass",
            ],
        },
        "real_gates_not_yet_executed": [
            "real initialisation equivalence (calibrated heads vs the R68 ep0 critic heads on a "
            "real rollout input)",
            "real gradient gate (critic-to-trunk isolation still holds with the calibrated heads)",
            "real restore / resume gate (fresh optimizer, lambda and RNG from the R68 ep0 state)",
            "the actual offline ridge fit and the resulting coefficient transform on real data",
            "any real embedding extraction, model forward, claim, checkpoint write or episode",
            "first-batch gate, per-seed 128-episode training and the fixed-ep128 evaluations",
            "real 5-module model construction and head installation on actual R68 parameters "
            "(the tests' ParameterContainer is a stand-in only)",
        ],
        "tests": {
            "file": "test_r71_readiness.py",
            "total": len(passed) + len(failed),
            "passed": len(passed),
            "failed": failed,
            "exit_code": 0 if not failed else 1,
            "all_pass": not failed,
            "names": passed,
            "scope": ("pure arithmetic / static / spy only; no real or synthetic neural forward, "
                      "optimizer, autograd, real-data fit or checkpoint save"),
        },
        "protected_files": {
            "ROUND46_CURRENT_TASK.json":
                "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
            "ROUND46_R2_V6_STOP_STATE.json":
                "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18",
            "status": "byte-unchanged and unstaged",
        },
        "untracked_left_alone": "analysis/round59-communication-decision-density/r59_analysis_run.log",
    }
    dst = HERE / "ROUND71_READINESS.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"stage": out["stage"], "licensed": out["authorisation"]["experiment_licensed"],
                      "tests": out["tests"]["passed"], "failed": len(failed),
                      "exit": out["tests"]["exit_code"],
                      "offline_input_rows": counts["totals"]["target_level_19d_input_rows_AUTHORITATIVE"],
                      "within_offline_cap": counts["candidate_offline_budget"]["within_cap"]},
                     indent=1))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())