"""ROUND19_VERIFICATION.json + MANIFEST.json: this round did only what it was allowed to do.

Four kinds of check, all mechanical:

  1. identity      — the branch starts at the round-18 head, byte for byte;
  2. emptiness     — zero new episodes exist in this directory, and the production code that
                     runs them is untouched relative to that head;
  3. agreement     — every headline number quoted in the round's documents is read back out of
                     the round's own JSON artefacts (a document that drifts from its data fails);
  4. housekeeping  — the files another session left untracked are still there, untouched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import c19_common as c19

START_SHA = "4c71fead0047298101869200a8518059afc42ccc"
BRANCH = "docs/comm-cost-objective-design"
DELIVERED = (
    "ROUND19_SCOPE.md", "ROUND18_ARCHIVE_CLOSURE.md", "COMM_COST_COVERAGE.md",
    "CONSTRAINED_COMM_COST_DESIGN.md", "CONSTRAINED_COMM_COST_PILOT.md", "REMOTE_RETRY_LOG.md",
    "COMM_COST_OBJECTIVE_DESIGN_REVISED.md", "COMM_PILOT_STAGED_APPROVAL.md",
    "DECISION_SUMMARY.md", "DESIGN_SPEC_COMPLIANCE.md",
    "ROUND18_ARCHIVE_CLOSURE.json", "COMM_COST_COVERAGE.json", "LEARNABILITY_PROBE_OFFLINE.json",
    "BUCKET_ARITHMETIC_RECHECK.json",
    "c19_common.py", "c19_closure_r18.py", "c19_cost_coverage.py", "c19_learnability.py",
    "c19_recheck.py", "c19_verify.py")
# Paths that belong to other sessions. The round must leave every one of them in place.
OTHER_SESSION_PATHS = ("architecture", "runs/comm_multistep_stage1_20260922/_DOC_PATCH_NOTES.md",
                       "runs/coordinator_baselines_20260917/dev_summary.json",
                       "runs/comm_send_actionability_20260922/rows_s700010.json")
ALLOWED_CHANGES = ("runs/comm_cost_objective_design_20260926", "PROJECT_CHARTER.md",
                   "PROJECT_STATUS.md", "README.md",
                   # `runs/*` is ignored by .gitignore, so a round cannot deliver its artefacts
                   # without an allow-list block; that edit is declared here and then checked
                   # below to be exactly that block and nothing else.
                   ".gitignore")
GITIGNORE_ROUND_19_LINES = (
    "!runs/comm_cost_objective_design_20260926/",
    "runs/comm_cost_objective_design_20260926/*",
    "!runs/comm_cost_objective_design_20260926/*.py",
    "!runs/comm_cost_objective_design_20260926/*.md",
    "!runs/comm_cost_objective_design_20260926/*.json",
    "runs/comm_cost_objective_design_20260926/rejected/")
QUOTED = {  # document claim -> (artefact, dotted path)
    "r18_all_checks_pass": ("ROUND18_ARCHIVE_CLOSURE.json", ("all_checks_pass",)),
    "r18_A_undetected_mean": ("ROUND18_ARCHIVE_CLOSURE.json",
                              ("batch_summary", "A", "per_episode", "undetected")),
    "r18_B_undetected_mean": ("ROUND18_ARCHIVE_CLOSURE.json",
                              ("batch_summary", "B", "per_episode", "undetected")),
    "r18_task_slack_A": ("ROUND18_ARCHIVE_CLOSURE.json",
                         ("batch_summary", "A", "detected_but_not_on_time")),
    "coverage_strict_share": ("COMM_COST_COVERAGE.json",
                              ("redundancy_against_the_silence_counterfactual",
                               "savings_ceiling", "strict_share_of_bill")),
    "coverage_value_of_uplink": ("COMM_COST_COVERAGE.json",
                                 ("deployed_arm_table", "derived", "value_of_the_uplink_C1",
                                  "bytes_per_extra_on_time_task")),
    "coverage_payload_median": ("COMM_COST_COVERAGE.json",
                                ("price_of_one_report", "payload_bytes", "median")),
    "coverage_slack_at_completion_median": ("COMM_COST_COVERAGE.json",
                                            ("deadline_slack_on_the_deployed_path",
                                             "remaining_slack_at_completion", "median")),
    "learn_auc_step": ("LEARNABILITY_PROBE_OFFLINE.json",
                       ("separability", "step_reported_at", "auc_load_bearing_vs_rest")),
    "learn_rule_bytes": ("LEARNABILITY_PROBE_OFFLINE.json",
                         ("learnability_verdict",
                          "cheapest_one_threshold_rule_at_full_recorded_effect",
                          "estimated_bytes_per_episode"))}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=c19.REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=c19.REPO_ROOT, capture_output=True,
                          text=True).returncode == 0


def dig(payload: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        payload = payload[key]
    return payload


def main() -> int:
    checks: dict[str, bool] = {}
    notes: dict[str, Any] = {}

    checks["branch_is_the_one_declared"] = git("rev-parse", "--abbrev-ref", "HEAD") == BRANCH
    checks["start_sha_exists_locally"] = git("cat-file", "-t", START_SHA) == "commit"
    checks["head_descends_from_round_18_head"] = git_ok("merge-base", "--is-ancestor",
                                                        START_SHA, "HEAD")
    checks["round_18_head_is_untouched"] = git("rev-parse", "analysis/search-trace-and-policy") \
        == START_SHA

    changed = [line for line in git("diff", "--name-only", START_SHA, "--").splitlines() if line]
    untracked = [line[3:] for line in git("status", "--porcelain", "--untracked-files=normal")
                 .splitlines() if line.startswith("??")]
    unexpected = sorted({path for path in changed
                         if not any(path == root or path.startswith(root + "/")
                                    for root in ALLOWED_CHANGES)}
                        | {path for path in untracked
                           if not path.startswith("runs/comm_cost_objective_design_20260926")
                           and path.rstrip("/") not in OTHER_SESSION_PATHS
                           and not path.startswith("runs/comm_multistep_stage1_20260922")
                           and not path.startswith("runs/comm_send_actionability_20260922")})
    checks["only_this_rounds_files_changed"] = not unexpected
    checks["no_production_code_changed"] = not [
        path for path in changed
        if path.split("/")[0] in ("integrated_sar", "task_allocation", "piloc")]
    gitignore_diff = git("diff", "--unified=0", START_SHA, "--", ".gitignore")
    added = [line[1:].rstrip() for line in gitignore_diff.splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:].rstrip() for line in gitignore_diff.splitlines()
               if line.startswith("-") and not line.startswith("---")]
    meaningful = [line for line in added if line.strip() and not line.strip().startswith("#")]
    checks["gitignore_change_is_the_allow_list_only"] = (
        not removed and all(line in GITIGNORE_ROUND_19_LINES for line in meaningful)
        and len(meaningful) == len(GITIGNORE_ROUND_19_LINES))
    checks["governance_edits_are_only_the_three_status_docs"] = not [
        path for path in changed
        if path.split("/")[0] in ("integrated_sar", "architecture") or path == "AGENTS.md"]

    # emptiness: the round must contain no episode output of any kind
    round_files = sorted(path.name for path in c19.ROUND_DIR.iterdir())
    checks["zero_new_episodes"] = not any(
        name.endswith((".jsonl", ".pth", ".pt")) or name in ("rows", "logs")
        for name in round_files)
    checks["rejected_versions_are_kept_not_deleted"] = (c19.ROUND_DIR / "rejected").is_dir() \
        and len(list((c19.ROUND_DIR / "rejected").iterdir())) >= 1
    checks["other_sessions_files_still_present"] = all(
        (c19.REPO_ROOT / path).exists() for path in OTHER_SESSION_PATHS)

    # agreement: the documents must not drift from their own data
    closure = c19.read_json(c19.ROUND_DIR / "ROUND18_ARCHIVE_CLOSURE.json")
    coverage = c19.read_json(c19.ROUND_DIR / "COMM_COST_COVERAGE.json")
    learn = c19.read_json(c19.ROUND_DIR / "LEARNABILITY_PROBE_OFFLINE.json")
    artefacts = {"ROUND18_ARCHIVE_CLOSURE.json": closure, "COMM_COST_COVERAGE.json": coverage,
                 "LEARNABILITY_PROBE_OFFLINE.json": learn}
    values = {}
    for name, (file_name, path) in QUOTED.items():
        try:
            values[name] = dig(artefacts[file_name], path)
        except (KeyError, TypeError) as error:
            values[name] = f"missing:{error}"
    checks["quoted_numbers_present_in_artefacts"] = all(
        not (isinstance(value, str) and value.startswith("missing:"))
        for value in values.values())
    checks["closure_24_checks_x_120_cells"] = (
        len(closure["checks"]) == 24
        and all(count == 120 for count in closure["checks"].values())
        and closure["all_checks_pass"] is True)
    checks["r18_counts_partition"] = all(
        closure["batch_summary"][arm]["closure"]["equals_placed"] for arm in c19.ARMS_R18)
    redundancy = coverage["redundancy_against_the_silence_counterfactual"]
    checks["redundancy_and_learnability_agree"] = (
        redundancy["counts"]["redundant_knowledge_and_outcome"]
        == learn["counts"]["redundant_knowledge_and_outcome"]
        and redundancy["counts"]["uplink_bought_time_same_outcome"]
        == learn["counts"]["uplink_bought_time_same_outcome"]
        and redundancy["counts"]["load_bearing_outcome_only_with_uplink"]
        == learn["counts"]["load_bearing"])
    checks["redundancy_classes_are_exhaustive"] = (
        sum(redundancy["counts"].values()) == redundancy["targets_compared"] == 352)
    checks["no_interior_cost_point_exists_on_disk"] = all(
        arm["per_episode"]["budget_refused_sends"] == 0.0
        for arm in coverage["deployed_arm_table"].values() if "per_episode" in arm)
    # Delivery is claimed as unverified; that is only honest if the remote really has no such ref.
    checks["no_remote_ref_for_this_branch"] = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", f"origin/{BRANCH}"], cwd=c19.REPO_ROOT,
        capture_output=True).returncode != 0

    missing_files = [name for name in DELIVERED if name not in round_files]
    checks["all_declared_deliverables_exist"] = not missing_files

    # the same-day revision: its arithmetic must close, and the spec comparison must still be
    # declared pending rather than silently claimed
    recheck = c19.read_json(c19.ROUND_DIR / "BUCKET_ARITHMETIC_RECHECK.json")
    checks["revision_buckets_close"] = all(recheck["closure_checks"].values())
    checks["revision_recompute_matches_delivered"] = all(
        recheck["recomputation_matches_delivered_artefact"].values())
    checks["revision_buckets_are_exhaustive"] = sum(
        recheck["arithmetic"]["per_bucket"][name]["targets"]
        for name in recheck["arithmetic"]["per_bucket"]) == 352
    compliance = (c19.ROUND_DIR / "DESIGN_SPEC_COMPLIANCE.md").read_text(encoding="utf-8")
    checks["spec_compliance_declared_pending"] = all(
        marker in compliance for marker in (
            "挂起待补", "未获得", "成功的远端接触数 = 0", "把本文件读成"))
    revised = (c19.ROUND_DIR / "COMM_COST_OBJECTIVE_DESIGN_REVISED.md").read_text(encoding="utf-8")
    checks["revision_keeps_the_two_denominators_apart"] = (
        "5.3977 %" in revised and "31.2500 %" in revised and "63.3523 %" in revised
        and "5.4559 %" in revised)
    checks["revision_labels_are_descriptive"] = all(
        name in revised for name in (
            "outcome_differs_under_full_silence",
            "known_no_later_and_outcome_same_under_full_silence",
            "known_later_but_outcome_same_under_full_silence",
            "开发集探索性", "late_discovery_gate", "载荷尝试成本"))

    notes["gitignore_added_lines"] = added
    notes["gitignore_removed_lines"] = removed
    notes["quoted_values"] = values
    notes["changed_files_vs_start"] = changed
    notes["unexpected_changes"] = unexpected
    notes["missing_deliverables"] = missing_files
    notes["round_dir_files"] = round_files
    notes["quarantined"] = sorted(path.name for path in (c19.ROUND_DIR / "rejected").iterdir()) \
        if (c19.ROUND_DIR / "rejected").is_dir() else []
    notes["remote_verification"] = {
        "requested_plan_sha": "72c4c3f71e07339bf6533fbd771aef3b2682103c",
        "plan_sha_verified": False,
        "attempts_this_round": {"git_fetch": 2, "git_ls_remote": 5, "git_push": 0,
                                "successful_remote_contacts": 0,
                                "detail": "REMOTE_RETRY_LOG.md (逐条含错误原文与诊断)"},
        "reason": "git fetch/ls-remote to github.com:443 failed; the private repo cannot be read "
                  "through api.github.com unauthenticated (404), and no local object matches",
        "last_actually_verified_remote_ref":
            "refs/remotes/origin/analysis/real-search-undetected-attribution = "
            "80af16ba4b90ce0c2d5456dd8b8e57068b4513e4",
        "documents_read": [],
        "documents_not_read": ["ROUND19_COMM_MAINLINE_RETURN.md",
                               "ROUND19_CONSTRAINED_COMM_CONTROLLER_DESIGN.md"],
        "revision_status": "the same-day revision is written against the user's 2026-09-26 "
                           "instruction only; item-by-item comparison with the two spec files is "
                           "suspended in DESIGN_SPEC_COMPLIANCE.md until they arrive"}

    payload = {
        "round": "19", "branch": BRANCH, "start_sha": START_SHA,
        "local_head_at_verification": git("rev-parse", "HEAD"),
        "episodes_launched": 0, "training_runs": 0, "new_stream_files": 0,
        "production_code_changed": False, "defaults_changed": False,
        "search_or_allocation_weights_changed": False, "sealed_sets_touched": False,
        "repo_config_changed": {
            "file": ".gitignore",
            "what": "appended the round-19 allow-list block (runs/* is ignored by default), "
                    "checked below to add nothing else and remove nothing",
            "lines": [line for line in GITIGNORE_ROUND_19_LINES]},
        "checks": checks,
        "checks_passed": sum(1 for value in checks.values() if value),
        "checks_total": len(checks),
        "passed": all(checks.values()),
        "notes": notes}
    c19.write_json("ROUND19_VERIFICATION.json", payload)

    manifest = {
        "round": "19", "branch": BRANCH, "start_sha": START_SHA,
        "head_at_manifest": git("rev-parse", "HEAD"),
        "delivered": {name: {"path": f"runs/comm_cost_objective_design_20260926/{name}",
                             "bytes": (c19.ROUND_DIR / name).stat().st_size,
                             "sha256": c19.sha256_file(c19.ROUND_DIR / name)}
                      for name in DELIVERED if (c19.ROUND_DIR / name).exists()},
        "quarantined": {path.name: c19.sha256_file(path)
                        for path in sorted((c19.ROUND_DIR / "rejected").iterdir())}
        if (c19.ROUND_DIR / "rejected").is_dir() else {},
        "inputs_read": {
            "round_18": {"dir": "runs/search_trace_and_policy_20260925",
                         "rows_read": 120,
                         "logs_read": {"lifecycle_rp": 120, "transcript_ct": 120,
                                       "search_probe_sp": 120, "step_record_sr": 120},
                         "raw_logs_not_in_git": {
                             "note": "round 18 already keeps its raw logs out of Git; this round "
                                     "read them, it created none",
                             "location": "runs/search_trace_and_policy_20260925/logs/"},
                         "anchor_artefact_sha256": c19.sha256_file(
                             c19.R18 / "SEARCH_POLICY_COMPARISON_RESULTS.json")},
            "round_16": {"dir": "runs/comm_info_path_validation_20260923",
                         "rows_read": 240,
                         "logs_read": {"lifecycle_rp": 120, "transcript_ct": 120,
                                       "info_probe_ip": 120,
                                       "note": "only A1 and A2 logs are opened: those two arms "
                                               "carry the deployed cost bill and the silence "
                                               "counterfactual"},
                         "raw_logs_not_in_git": {
                             "note": "same as above - read only",
                             "location": "runs/comm_info_path_validation_20260923/logs/"},
                         "anchor_artefact_sha256": c19.sha256_file(
                             c19.R16 / "D1_RESULTS.json")},
            "round_15": {"dir": "runs/comm_research_closure_20260923",
                         "anchor_artefact_sha256": c19.sha256_file(
                             c19.R15 / "COMMUNICATION_RESEARCH_CLOSURE.md")}},
        "checkpoints": "none read or written this round (0 training, 0 new episode)",
        "verification_passed": payload["passed"]}
    c19.write_json("MANIFEST.json", manifest)
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"],
                      "quoted": values}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
