"""Boundary and internal-consistency checks for the undetected-goal attribution round.

Every statement here is re-derived at run time: the read-only scope from git, the
0-episode claim from the tripwire and the import list, the input set from its hashes,
and the bucket arithmetic from the two artefacts it is supposed to close against.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import rs_common as rc

rc.enter_repo()

ALLOWED_DOCUMENTS = {"PROJECT_CHARTER.md", "PROJECT_STATUS.md", "README.md", "ARCHITECTURE.md"}
#: The round must register its own artefact allow-list, which is the only `.gitignore`
#: change it is permitted to make; the block it adds names only this round's directory.
ALLOWED_OTHER = {".gitignore"}
ALLOWED_PREFIXES = (f"runs/{rc.ROUND}/",)
PROTECTED = ("integrated_sar/", "piloc/", "task_allocation/")
D1_DIR = "runs/comm_info_path_validation_20260923/"


def git(*args: str) -> str:
    result = subprocess.run(("git", *args), cwd=rc.REPO_ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    return (result.stdout or "").strip()


def scope() -> dict[str, Any]:
    changed = [line for line in git("diff", "--name-only", rc.START_SHA).splitlines() if line.strip()]
    allowed = ALLOWED_DOCUMENTS | ALLOWED_OTHER
    outside = sorted(path for path in changed
                     if path not in allowed and not path.startswith(ALLOWED_PREFIXES))
    additions = [line[1:].strip() for line in git("diff", rc.START_SHA, "--", ".gitignore").splitlines()
                 if line.startswith("+") and not line.startswith("+++")]
    return {
        "changed_vs_start_sha": sorted(changed),
        "outside_the_allowed_set": outside,
        "gitignore_additions_all_scope_this_round": all(
            line.startswith(f"runs/{rc.ROUND}") or line.startswith(f"!runs/{rc.ROUND}")
            or line.startswith("#") or not line
            for line in additions),
        "gitignore_added_lines": len(additions),
        "protected_paths_touched": sorted(path for path in changed if path.startswith(PROTECTED)),
        "d1_frozen_artefacts_changed": sorted(
            path for path in changed if path.startswith(D1_DIR)),
        "start_sha_resolvable": bool(git("rev-parse", "--verify", rc.START_SHA)),
    }


def rehash_inputs() -> dict[str, Any]:
    """Re-verify every file the three analysis scripts registered as an input."""
    entries: dict[str, str] = {}
    for name in ("BATCH_RECONCILIATION.json", "FIELD_INVENTORY.json", "SCENARIO_GEOMETRY.json"):
        payload = json.loads((rc.ROUND_DIR / name).read_text(encoding="utf-8"))
        for path, row in (payload.get("inputs_read") or {}).get("files", {}).items():
            entries[path] = row["sha256"]
    changed, missing = [], []
    for path, digest in sorted(entries.items()):
        full = rc.REPO_ROOT / path
        if not full.exists():
            missing.append(path)
        elif rc.sha256_file(full) != digest:
            changed.append(path)
    return {"inputs": len(entries), "changed": changed, "missing": missing}


def episode_freedom() -> dict[str, Any]:
    """In a fresh process: import the analysis modules, then try to start an episode.

    The attempt must be stopped by the tripwire. A tripwire that wrapped nothing would let
    the call through and report "no calls" truthfully, which is the worst failure mode here,
    so the number of resolved targets and the block message are both checked.
    """
    script = f"""
import json, sys
sys.path.insert(0, {str(rc.ROUND_DIR)!r}); sys.path.insert(1, {str(rc.REPO_ROOT)!r})
import rs_common as rc
rc.enter_repo()
resolved = rc.install_env_tripwire()
import rs_counts, rs_fields, rs_geometry, rs_attribute
calls_before_attempt = rc.tripwire_calls()
from integrated_sar.joint_driver import build_joint_run
attempt = None
try:
    build_joint_run(None)
except RuntimeError as exc:
    attempt = "blocked" if "read-only" in str(exc) else "raised_other"
except Exception as exc:
    attempt = "not_blocked:" + type(exc).__name__
print(json.dumps({{"tripwire_resolved": resolved,
                   "analysis_calls": calls_before_attempt,
                   "probe_calls_after_attempt": rc.tripwire_calls(),
                   "forbidden_imports": rc.forbidden_imports_seen(),
                   "episode_attempt": attempt}}))
"""
    result = subprocess.run((sys.executable, "-c", script), cwd=rc.REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload: Any = {}
    for line in (result.stdout or "").splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
    return {"probe_stderr_tail": (result.stderr or "")[-500:] if not payload else "", **payload}


def sealed_blocks() -> dict[str, Any]:
    hits = git("ls-files", "--others", "--exclude-standard", "runs").splitlines()
    sealed = [line for line in hits if "s800000" in line or "s800399" in line]
    return {"untracked_files_under_runs_matching_sealed_seeds": sorted(sealed)[:10],
            "sealed_seed_count_seen": len(sealed)}


def bucket_closure() -> dict[str, Any]:
    reconciliation = json.loads((rc.ROUND_DIR / "BATCH_RECONCILIATION.json").read_text(
        encoding="utf-8"))
    geometry = json.loads((rc.ROUND_DIR / "SCENARIO_GEOMETRY.json").read_text(encoding="utf-8"))
    attribution = json.loads((rc.ROUND_DIR / "UNDETECTED_GOAL_ATTRIBUTION.json").read_text(
        encoding="utf-8"))
    checks: dict[str, Any] = {}
    for key in reconciliation["batches"]:
        missed = reconciliation["batches"][key]["per_arm"][
            rc.BATCHES[key]["search_record_arm"]]["undetected"]
        rows = [row for row in attribution["rows"] if row["batch"] == key]
        counts = attribution["batches"][key]["bucket_counts"]
        checks[key] = {
            "rows_equal_misses": len(rows) == missed,
            "buckets_sum_equal_misses": sum(counts.values()) == missed,
            "every_row_certified": all(row["component_8_shared_with_an_agent"] is not None
                                       for row in rows),
            "all_seeds_certified": geometry["batches"][key]["seeds_certified_by_map_hash"]
            == geometry["batches"][key]["seeds_total"],
            "placements_and_starts_agree": (
                geometry["batches"][key]["logged_and_rebuilt_placements_agree"]
                and geometry["batches"][key]["logged_and_rebuilt_agent_starts_agree"]),
            "line_of_sight_disagreements": geometry["batches"][key][
                "line_of_sight_predicate_cross_check"]["disagreements"],
            "no_unreachable_claimed": counts["unreachable_or_scene_problem"] == 0,
            "machinery_consistency": attribution["batches"][key]["machinery_consistency"],
        }
    return checks


def run() -> int:
    checks: dict[str, Any] = {
        "round": rc.ROUND, "start_sha": rc.START_SHA,
        "scope": scope(), "inputs_unchanged": rehash_inputs(),
        "episode_freedom": episode_freedom(), "sealed": sealed_blocks(),
        "bucket_closure": bucket_closure(),
        "authorisation_boundaries": {
            "no_piloc_retraining": "no training entry point was imported or called; "
                                   "torch/piloc/task_allocation never appear in sys.modules",
            "no_communication_change": "the two batches' logged transport parameters are read "
                                       "as-is; this round writes no configuration",
            "no_added_goal_difficulty": "num_goals/steps/deadline are read from the archived "
                                        "run_config and never overridden",
            "next_step_documents_executed": False,
        },
    }
    checks["passed"] = bool(
        not checks["scope"]["outside_the_allowed_set"]
        and checks["scope"]["gitignore_additions_all_scope_this_round"]
        and not checks["scope"]["protected_paths_touched"]
        and not checks["scope"]["d1_frozen_artefacts_changed"]
        and not checks["inputs_unchanged"]["changed"]
        and not checks["inputs_unchanged"]["missing"]
        and checks["episode_freedom"].get("episode_attempt") == "blocked"
        and checks["episode_freedom"].get("analysis_calls") == []
        and checks["episode_freedom"].get("probe_calls_after_attempt")
        == ["integrated_sar.joint_driver.build_joint_run"]
        and len(checks["episode_freedom"].get("tripwire_resolved") or ()) == len(rc.FORBIDDEN_CALLS)
        and checks["episode_freedom"].get("forbidden_imports") == []
        and checks["sealed"]["sealed_seed_count_seen"] == 0
        and all(per["line_of_sight_disagreements"] == 0
                and per["all_seeds_certified"] and per["rows_equal_misses"]
                and per["buckets_sum_equal_misses"]
                and per["placements_and_starts_agree"]
                and per["no_unreachable_claimed"]
                and all(value for key, value in per["machinery_consistency"].items()
                        if isinstance(value, bool))
                for per in checks["bucket_closure"].values())
    )
    path = rc.write_json("ATTRIBUTION_VERIFICATION.json", checks)
    print(json.dumps({key: value for key, value in checks.items()
                      if key in ("passed", "scope", "inputs_unchanged", "sealed",
                                 "episode_freedom")}, ensure_ascii=False, indent=1)[:2200])
    print("bucket closure:", json.dumps(checks["bucket_closure"], ensure_ascii=False)[:600])
    print(f"wrote {path.name}")
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
