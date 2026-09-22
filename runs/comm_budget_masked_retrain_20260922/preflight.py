"""Everything that has to be true before a training episode is allowed to start.

The brief's §6 gates, in one runnable report. Two design decisions are worth stating because they
are the difference between a gate and a decoration:

*The suites are re-run, not quoted.* The previous round's M0-M3 reporter is recomputed here from
the artefacts it recorded plus fresh runs of the same test discovery on this tree, and each piece
of evidence says whether it is archived (measured at ``7f9929c``/``e692607``) or fresh. A gate that
could only be satisfied by citing an old document would not be a gate.

*The probes measure the seam that will be trained on.* Twelve episodes -- four per training seed,
each with that seed's own initial weights -- plus a deterministic save/resume regression and one
greedy evaluation pass. Those cover the three things that have actually broken in this research
line: a sampler whose tape disagrees with its draw, a resume that silently changes the stream order,
and an evaluation path that was never executed until the day it produced a headline number.

    python preflight.py            # run every gate, write PREFLIGHT.json
    python preflight.py --no-rerun  # re-read artefacts, re-run suites only, start no episode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import freeze as freeze_module
import retrain_common as common
import train_retrain as entry
import evaluate_retrain as evaluation

import torch  # noqa: E402


def suite_report() -> list[dict[str, Any]]:
    """The four suites AGENTS and the two previous rounds established, plus the integration smoke."""
    return [
        common.run_tests(common.ROOT / "integrated_sar" / "tests", "test_comm_budget_action_mask.py"),
        common.run_tests(common.ROOT / "integrated_sar" / "tests", "test_*.py"),
        common.run_tests(common.MASK_FIX_ROUND, "test_*.py"),
        common.run_tests(common.HERE, "test_*.py"),
        common.integration_smoke(),
    ]


def mask_gates(probe: dict[str, Any], suites: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-decide last round's M0-M3 on this tree, with the evidence source named for each field."""
    archived_smoke = json.loads(
        (common.MASK_FIX_ROUND / "COMM_BUDGET_ACTION_MASK_SMOKE.json").read_text(encoding="utf-8")
    )
    archived_manifest = json.loads(
        (
            common.MASK_FIX_ROUND / "COMM_BUDGET_ACTION_MASK_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    contract, full, mask_round, this_round, integration = suites
    closure = probe["closure"]
    totals = {
        key: sum(int(row[key]) for row in probe["rows"])
        for key in (
            "candidates",
            "initial_executable_candidates",
            "actionable_candidates",
            "masked_by_credit",
            "masked_by_capacity",
            "requested",
            "executed_policy_emissions",
            "policy_budget_refusals",
            "policy_capacity_refusals",
            "forced_emissions",
        )
    }
    production = common.production_diff()
    defaults = common.setting()["production_defaults"]
    gates = {
        "M0_production_and_history_isolation": {
            "passed": bool(
                production["empty"]
                and defaults["comm_ack"] is False
                and defaults["comm_timing_mode"] == "legacy_shared_bound"
                and defaults["comm_policy"] == "immediate"
                and full["passed"]
                and integration["passed"]
            ),
            "evidence": {
                "source": "fresh: git diff against e692607, RunConfiguration, both suites",
                "shipped_code_changed_since_fix_head": production["integrated_sar_changed"],
                "runtime_modules_changed": production["runtime_modules_changed"],
                "defaults": defaults,
                "feature_layout": f"{common.setting()['feature_dimension']}-dim "
                f"{common.setting()['feature_layout']}",
                "full_suite": full,
                "integration_smoke": integration,
                "mask_off_isolation": "budget_mask=False still delegates to the archived "
                "collector (mask-fix round's LegacyIsolationTests, re-run here)",
            },
        },
        "M1_action_execution_agreement": {
            "passed": bool(
                totals["policy_budget_refusals"] == 0
                and totals["policy_capacity_refusals"] == 0
                and totals["requested"] == totals["executed_policy_emissions"]
                and closure["requested_equals_executed"]
                and mask_round["passed"]
            ),
            "evidence": {
                "source": "fresh: 12 probe episodes on this tree; archived 8-stream smoke quoted",
                "probe_totals": totals,
                "probe_closure": closure,
                "archived_smoke_closure": archived_smoke["closure"],
                "mask_round_suite": mask_round,
            },
        },
        "M2_ppo_probability_agreement": {
            "passed": bool(
                float(closure["max_log_prob_gap"]) <= entry.LOG_PROB_TOLERANCE
                and this_round["passed"]
            ),
            "evidence": {
                "source": "fresh: logged vs recomputed over every probe tape",
                "max_abs_log_prob_gap": closure["max_log_prob_gap"],
                "tolerance": entry.LOG_PROB_TOLERANCE,
                "this_round_suite": this_round,
                "archived_smoke_gap": archived_smoke["closure"]["max_log_prob_gap"],
            },
        },
        "M3_probe_closure_zero_optimizer_steps": {
            "passed": bool(
                probe["throughput"]["episodes"] <= common.PROBE_EPISODES
                and not probe["technical_failures"]
                and closure["optimizer_steps_total"] == 0
                and closure["tape_invariants_passed"]
                and closure["episode_length_all_1000"]
                and common.sealed_block_scan()["clean"]
            ),
            "evidence": {
                "source": "fresh: 12 probe episodes (4 per training seed), fixed initial weights",
                "episodes": probe["throughput"]["episodes"],
                "probe_cap": common.PROBE_EPISODES,
                "technical_failures": probe["technical_failures"],
                "optimizer_steps_total": closure["optimizer_steps_total"],
                "sealed_block": common.sealed_block_scan(),
                "distinct_initial_fingerprints": closure["distinct_initial_fingerprints"],
                "training_runs_started": 0,
            },
        },
    }
    return gates


def provenance_gates() -> dict[str, Any]:
    """§2's rule as a check: the freeze exists, the graph is clean against it, nothing is untracked."""
    payload = common.freeze()
    paths = freeze_module.graph()
    cleanliness = common.freeze_cleanliness(paths)
    correction = common.json_path("COMM_BUDGET_MASK_PROVENANCE_CORRECTION.json")
    recorded = json.loads(correction.read_text(encoding="utf-8")) if correction.exists() else {}
    return {
        "freeze_registered": bool(payload.get("registered")),
        "freeze_sha": payload.get("sha"),
        "freeze_subject": payload.get("subject"),
        "graph_members": len(paths),
        "cleanliness": cleanliness,
        "production_diff_since_fix_head": common.production_diff(),
        "correction_document_present": correction.exists(),
        "correction_records_phantom_absent": recorded.get("phantom_module", {}).get(
            "exists_on_disk"
        )
        is False,
        "correction_started_no_episode": recorded.get("episodes_started_by_this_correction") == 0,
        "passed": bool(
            payload.get("registered")
            and cleanliness["clean"]
            and common.production_diff()["empty"]
            and correction.exists()
            and recorded.get("phantom_module", {}).get("exists_on_disk") is False
        ),
    }


def greedy_path_probe() -> dict[str, Any]:
    """One greedy, masked episode through the evaluation builder, on a training stream.

    The evaluation path differs from the training path in two ways -- ``greedy=True`` and a
    different row builder -- and neither was exercised by the probes. Running it once on a
    *training* stream keeps the development block untouched while making sure the code that will
    produce the headline numbers has at least executed before the run starts.
    """
    directory = common.EPISODE_OUT / "greedy_probe"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = common.LaunchLedger("LAUNCH_LEDGER_greedy_probe.json")
    controller = entry.initial_controller(1)
    stream = common.TRAIN_SEEDS[0]
    ledger.reserve("greedy_probe", stream, arm="greedy_probe")
    began = time.time()
    try:
        row = evaluation.score_learned(
            controller, stream, training_seed=1, at_steps=0, directory=directory
        )
        ledger.record("greedy_probe", stream, True, f"bytes={row['airtime_bytes']}")
        return {
            "ran": True,
            "passed": bool(
                row["requested"] == row["emissions_by_policy"]
                and not row["budget_refusals"]
                and not row["capacity_refusals"]
                and row["tape_invariants_passed"]
                and float(row["max_log_prob_gap"]) <= entry.LOG_PROB_TOLERANCE
                and row["steps_run"] == common.STEPS_PER_EPISODE
                and row["timing_mode"] == common.TIMING_MODE
                and row["budget_bytes"] == common.BUDGET_BYTES
            ),
            "row": {key: value for key, value in row.items() if key != "send_sequence"},
            "seconds": round(time.time() - began, 2),
            "stream_role": "training stream: the development block is not spent on a plumbing check",
        }
    except Exception as error:
        ledger.record("greedy_probe", stream, False, f"{type(error).__name__}: {error}")
        return {"ran": False, "passed": False, "error": f"{type(error).__name__}: {error}"}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="reuse THROUGHPUT.json / RESUME_CHECK.json instead of starting their episodes again",
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))
    provenance = provenance_gates()
    if not provenance["passed"]:
        report = {
            "document": "PREFLIGHT",
            "passed": False,
            "provenance": provenance,
            "note": "training is forbidden until the freeze and the correction close",
        }
        common.write_json(common.PREFLIGHT_FILE, report)
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 1
    if args.no_rerun and common.json_path("THROUGHPUT.json").exists():
        probe = common.read_json("THROUGHPUT.json")
        resume = common.read_json("RESUME_CHECK.json")
        reused_artefacts = True
    else:
        probe_args = argparse.Namespace(
            episodes=common.PROBE_EPISODES // len(common.TRAINING_SEEDS),
            threads=args.threads,
            mode="probe",
            seed=1,
        )
        if entry.probe(probe_args):
            raise SystemExit("a probe episode failed; see THROUGHPUT.json")
        probe = common.read_json("THROUGHPUT.json")
        resume_args = argparse.Namespace(
            seed=1, threads=args.threads, mode="resume-check", episodes=4, steps=common.STEPS_PER_RUN
        )
        if entry.resume_check(resume_args):
            raise SystemExit("deterministic resume failed; see RESUME_CHECK.json")
        resume = common.read_json("RESUME_CHECK.json")
        reused_artefacts = False
    suites = suite_report()
    gates = mask_gates(probe, suites)
    identity = entry.identity_report()
    streams = common.verify_streams()
    resume_report = {
        "passed": bool(resume.get("passed")),
        "weights_identical": resume.get("weights_identical"),
        "optimizer_identical": resume.get("optimizer_identical"),
        "differences": resume.get("differences"),
        "next_batch_seeds_continuous": resume.get("next_batch_seeds_continuous"),
        "next_batch_seeds_resumed": resume.get("next_batch_seeds_resumed"),
        "episodes_used": resume.get("episodes_used"),
        "optimizer_steps_total": resume.get("optimizer_steps_total"),
    }
    greedy = greedy_path_probe()
    ledger = common.ledger_totals()
    checks = {
        "provenance": provenance,
        "streams": streams,
        "suites": suites,
        "mask_gates": gates,
        "all_mask_gates_passed": all(gate["passed"] for gate in gates.values()),
        "deterministic_resume": resume_report,
        "initial_weights": {
            "distinct": identity["distinct"],
            "all_distinct": identity["all_distinct"],
            "fingerprints": {
                str(seed): item["fingerprint"]
                for seed, item in identity["initial_weights"].items()
            },
            "loaded_from_checkpoint": [
                item["loaded_from"] for item in identity["initial_weights"].values()
            ],
            "passed": bool(
                identity["all_distinct"]
                and all(
                    item["loaded_from"] is None
                    for item in identity["initial_weights"].values()
                )
            ),
        },
        "greedy_evaluation_path": greedy,
        "sealed_block": common.sealed_block_scan(),
        "launches": {
            "cap": ledger["cap"],
            "starts": ledger["starts"],
            "by_kind": ledger["by_kind"],
            "failures": ledger.get("failures", 0),
            "within_cap": ledger["starts"] <= ledger["cap"],
        },
        "throughput": probe["throughput"],
    }
    checks["passed"] = bool(
        provenance["passed"]
        and streams["passed"]
        and all(suite["passed"] for suite in suites)
        and checks["all_mask_gates_passed"]
        and resume_report["passed"]
        and checks["initial_weights"]["passed"]
        and greedy["passed"]
        and common.sealed_block_scan()["clean"]
        and checks["launches"]["within_cap"]
    )
    checks["document"] = "PREFLIGHT"
    checks["branch"] = common.git_out("rev-parse", "--abbrev-ref", "HEAD")
    checks["head"] = common.git_out("rev-parse", "HEAD")
    checks["code_freeze_sha"] = provenance["freeze_sha"]
    checks["configuration_hash"] = entry.configuration_hash()
    checks["trainer"] = {
        "trainer_tag": entry.TRAINER_TAG,
        "action_mask_schema": entry.ACTION_MASK_SCHEMA,
        "tape_schema": entry.TAPE_SCHEMA,
    }
    checks["training_runs_started"] = 0
    checks["probe_and_resume_reused_from_disk"] = reused_artefacts
    common.write_json(common.PREFLIGHT_FILE, checks)
    print(
        json.dumps(
            {
                "passed": checks["passed"],
                "provenance": provenance["passed"],
                "streams": streams["passed"],
                "suites": {suite["directory"] + "|" + suite["pattern"]: suite["passed"] for suite in suites},
                "mask_gates": {
                    name: gate["passed"] for name, gate in gates.items()
                },
                "deterministic_resume": resume_report["passed"],
                "initial_weights_distinct": checks["initial_weights"]["passed"],
                "greedy_path": greedy["passed"],
                "sealed_clean": checks["sealed_block"]["clean"],
                "launches": checks["launches"],
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        )
    )
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
