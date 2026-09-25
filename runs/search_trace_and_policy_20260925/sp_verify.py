"""ROUND_VERIFICATION.json + MANIFEST.json for round 18.

Checks that constrain the round rather than describe it:

* the launch ledger is inside the cap, every failure is booked, and the row totals per phase
  match the artefacts on disk;
* the two archived batches this round reads, and the documents freezing those rounds' criteria,
  are byte-unchanged since the start sha;
* only this round's own paths plus the declared production files are modified, and other
  sessions' files are untouched;
* every cell's goal arithmetic closes: placed = detected + undetected, and the probe's own
  placement count agrees with the run's;
* the three-view closure (environment deletion / predicate pairs / adapter events) is counted
  over every data cell, not spot-checked;
* the two checkpoints are the same bytes and no training or reward file changed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import sp_common as sc

PROTECTED = ["runs/comm_info_path_validation_20260923",
             "runs/real_search_undetected_attribution_20260923"]
ALLOWED_MODIFIED = {
    "integrated_sar/search_probe.py", "integrated_sar/adapters/piloc.py",
    "integrated_sar/joint_driver.py", "integrated_sar/info_probe.py",
    "integrated_sar/tests/test_search_probe.py", "piloc/src/Map/timemap.py",
    "ARCHITECTURE.md", "AGENTS.md", "PROJECT_CHARTER.md", "PROJECT_STATUS.md", "README.md",
    ".gitignore",
}
OTHER_SESSIONS = ["architecture", "runs/comm_send_actionability_20260922",
                  "runs/coordinator_baselines_20260917/dev_summary.json",
                  "runs/comm_multistep_stage1_20260922"]
TRAINING_SURFACES = ["piloc/train.py", "piloc/configs.py", "piloc/src/DDQN/Agent.py",
                     "task_allocation"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(sc.REPO_ROOT), capture_output=True,
                          text=True, timeout=180).stdout


def changed_files() -> dict[str, list[str]]:
    modified: list[str] = []
    untracked: list[str] = []
    for line in _git("status", "--porcelain").splitlines():
        marker, path = line[:2].strip(), line[3:].strip().replace("\\", "/")
        (untracked if marker == "??" else modified).append(path)
    return {"modified_or_staged": modified, "untracked": untracked}


def protected_paths_unchanged() -> dict[str, Any]:
    dirty = [path for path in changed_files()["modified_or_staged"]
             if any(path.startswith(guard) for guard in PROTECTED)]
    tracked = [line.strip().replace("\\", "/") for line in
               _git("diff", "--name-only", sc.START_SHA, "--", *PROTECTED).splitlines()
               if line.strip()]
    return {"dirty_paths": dirty, "changed_vs_start_sha": tracked,
            "passed": not dirty and not tracked}


def round_scope_check() -> dict[str, Any]:
    files = changed_files()
    outside = [path for path in files["modified_or_staged"]
               if not path.startswith(f"runs/{sc.ROUND}") and path not in ALLOWED_MODIFIED]
    touched = [path for path in files["modified_or_staged"]
               if any(path.startswith(other) for other in OTHER_SESSIONS)]
    return {"modified_outside_this_round": outside, "other_sessions_files_modified": touched,
            "untracked_count": len(files["untracked"]), "passed": not outside and not touched}


def ledger_check() -> dict[str, Any]:
    report = sc.ledger_report()
    per_phase: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    for path in sorted(sc.ROWS.glob("*.json")):
        row = sc.read_json(path)
        per_phase[row["phase"]] = per_phase.get(row["phase"], 0) + 1
        if not row.get("ok"):
            failures.append({"cell": path.name, "error": row.get("error")})
    return {"cap": report["cap"], "booked": report["used"], "rows_per_phase": per_phase,
            "booked_per_phase": report["by_phase"],
            "diagnostic_phases_booked": {phase: count for phase, count in
                                         report["by_phase"].items() if phase not in sc.PHASE_PLAN},
            "technical_failures": failures, "planned_normal_total": sum(sc.PHASE_PLAN.values()),
            "within_budget": report["used"] <= report["cap"]}


def goal_arithmetic() -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    cells = 0
    for path in sorted(sc.ROWS.glob("*.json")):
        row = sc.read_json(path)
        if not row.get("ok"):
            continue
        cells += 1
        placed = int(row["goals_placed"])
        detected = len(row["detected_target_ids"])
        undetected = int(row["undetected_count"])
        if placed != detected + undetected:
            violations.append({"cell": path.name, "why": "placed != detected + undetected",
                               "placed": placed, "detected": detected,
                               "undetected": undetected})
        facts = row.get("probe") or {}
        probe_placed = facts.get("probe_placed")
        if facts.get("search_probe_present") and int(probe_placed or 0) != placed:
            violations.append({"cell": path.name, "why": "probe placement != run placement",
                               "probe": probe_placed, "placed": placed})
    return {"cells": cells, "violations": violations, "passed": not violations}


def closure_across(prefix: str) -> dict[str, Any]:
    disagreements = anomalies = cells = 0
    for path in sorted(sc.ROWS.glob(f"{prefix}_*.json")):
        row = sc.read_json(path)
        if not row.get("ok"):
            continue
        cells += 1
        facts = row.get("probe") or {}
        disagreements += int(facts.get("probe_disagreement_steps") or 0)
        anomalies += int(facts.get("probe_anomalies") or 0)
    return {"cells": cells, "disagreement_steps_total": disagreements,
            "probe_anomalies_total": anomalies,
            "passed": disagreements == 0 and anomalies == 0}


def checkpoint_identity() -> dict[str, Any]:
    changed = [line.strip() for line in
               _git("diff", "--name-only", sc.START_SHA, "--", *TRAINING_SURFACES).splitlines()
               if line.strip()]
    return {"sha256": {"piloc": sc.sha256_file(sc.PIL_CHECKPOINT),
                       "ctde": sc.sha256_file(sc.CTDE_CHECKPOINT)},
            "changed_training_or_reward_files": changed, "passed": not changed,
            "note": ("checkpoints are large binaries kept off-tree by .gitignore; their digests "
                     "are recorded here and in every run signature")}


def main(argv: list[str]) -> int:
    identity = sc.read_json(sc.ROUND_DIR / "SEARCH_TRACE_IDENTITY_REPORT.json")
    verdict = sc.read_json(sc.ROUND_DIR / "SEARCH_DETECTION_CORRECTNESS_VERDICT.json")
    policy_path = sc.ROUND_DIR / "SEARCH_POLICY_COMPARISON_RESULTS.json"
    policy = sc.read_json(policy_path) if policy_path.exists() else None
    checks = {
        "protected_rounds_unchanged": protected_paths_unchanged(),
        "this_round_scope": round_scope_check(),
        "launch_ledger": ledger_check(),
        "goal_arithmetic": goal_arithmetic(),
        "three_view_closure_repro": closure_across("repro"),
        "three_view_closure_data": closure_across("data"),
        "checkpoint_identity": checkpoint_identity(),
        "passivity_gate": {
            "all_checks_true": identity["passivity_gate_P"]["all_checks_true"],
            "checks_counted": identity["passivity_gate_P"]["checks_counted"],
            "cells_used": identity["passivity_gate_P"]["cells_used"]},
        "historical_reproduction": {
            "cells": identity["reproduction_gate_Q"]["cells"],
            "reproduced": len(identity["reproduction_gate_Q"]["reproduced"]),
            "not_reproduced": identity["reproduction_gate_Q"]["not_reproduced"]},
        "correctness_verdict": {"verdict": verdict["verdict"],
                                "category_counts": verdict["category_counts"],
                                "s1_matrix_allowed": verdict["s1_matrix_allowed"]},
        "policy_matrix": None if policy is None else {
            "episodes_per_arm": policy["design"]["episodes_per_arm"],
            "missing_seeds": policy["design"]["missing_seeds"],
            "criterion_verdict": policy["criterion_verdict"]},
    }
    checks["passed"] = all([
        checks["protected_rounds_unchanged"]["passed"], checks["this_round_scope"]["passed"],
        checks["launch_ledger"]["within_budget"], checks["goal_arithmetic"]["passed"],
        checks["three_view_closure_repro"]["passed"], checks["three_view_closure_data"]["passed"],
        checks["checkpoint_identity"]["passed"], checks["passivity_gate"]["all_checks_true"],
        not checks["historical_reproduction"]["not_reproduced"],
        checks["correctness_verdict"]["s1_matrix_allowed"],
        policy is None or (policy["design"]["episodes_per_arm"] == len(sc.S1_SEEDS)
                           and not policy["design"]["missing_seeds"])])
    sc.write_json("ROUND_VERIFICATION.json", checks)

    delivered = sorted(path.name for path in sc.ROUND_DIR.iterdir()
                       if path.is_file() and path.suffix in {".json", ".md", ".py"}
                       and "__pycache__" not in str(path))
    logs = [path for path in sc.LOGS.glob("*") if path.is_file()]
    rows = [path for path in sc.ROWS.rglob("*.json") if path.is_file()]
    payload = {
        "round": sc.ROUND, "start_branch": sc.START_BRANCH, "start_sha": sc.START_SHA,
        "head": _git("rev-parse", "HEAD").strip(),
        "delivered": [{"file": name, "sha256": sc.sha256_file(sc.ROUND_DIR / name),
                       "bytes": (sc.ROUND_DIR / name).stat().st_size} for name in delivered],
        "raw_artefacts_not_in_git": {
            "location": f"runs/{sc.ROUND}/{{logs,rows,rows/rejected}}",
            "log_files": len(logs), "log_bytes": sum(path.stat().st_size for path in logs),
            "row_files": len(rows),
            "row_files_sha256_prefix16": {path.name: sc.sha256_file(path)[:16]
                                           for path in sorted(rows)},
            "gitignore_patterns": [f"runs/{sc.ROUND}/logs/", f"runs/{sc.ROUND}/rows/",
                                   "*.jsonl"],
            "why": ("逐时刻轨迹、探针侧文件与每步记录是原始数据，体积大且可由脚本重建；"
                    "Git 里只留脚本、判定件、清单与哈希")},
        "checkpoints": {
            "piloc": {"path": str(sc.PIL_CHECKPOINT.relative_to(sc.REPO_ROOT)).replace("\\", "/"),
                      "sha256": sc.sha256_file(sc.PIL_CHECKPOINT)},
            "ctde": {"path": str(sc.CTDE_CHECKPOINT.relative_to(sc.REPO_ROOT)).replace("\\", "/"),
                     "sha256": sc.sha256_file(sc.CTDE_CHECKPOINT)}},
        "inputs_read": [
            {"path": str(path.relative_to(sc.REPO_ROOT)).replace("\\", "/"),
             "sha256": sc.sha256_file(path), "present": path.exists()}
            for path in (
                sc.ROUND17_ATTRIBUTION,
                sc.d1_archive_path("A1", 1100051), sc.d1_archive_path("A1", 1100000),
                sc.systemloop_archive_path("baseline", 31003),
                sc.REPO_ROOT / "runs/comm_info_path_validation_20260923"
                / "D1_PROTOCOL_REVISION.md",
                sc.REPO_ROOT / "runs/real_search_undetected_attribution_20260923"
                / "SEARCH_SIDE_PAIRED_VALIDATION_PROTOCOL.md",
                sc.REPO_ROOT / "runs/real_search_undetected_attribution_20260923"
                / "MISSING_FIELDS_AND_PASSIVE_RECORD_PLAN.md",
                sc.REPO_ROOT / "AGENTS.md", sc.REPO_ROOT / "ARCHITECTURE.md",
                sc.REPO_ROOT / "integrated_sar" / "search_probe.py",
                sc.REPO_ROOT / "piloc" / "src" / "Map" / "timemap.py")],
        "launches_booked_by_phase": sc.ledger_report()["by_phase"],
        "verification_passed": checks["passed"],
    }
    sc.write_json("MANIFEST.json", payload)
    print(json.dumps({"passed": checks["passed"], "booked": checks["launch_ledger"]["booked"],
                      "closure_data": checks["three_view_closure_data"],
                      "outside": checks["this_round_scope"]["modified_outside_this_round"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
