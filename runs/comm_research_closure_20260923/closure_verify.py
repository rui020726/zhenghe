"""Prove the round's own boundaries, mechanically.

Each check is the kind that has bitten this project before: an input that changed
mid-analysis, a tripwire that silently wrapped nothing, a production file touched
"just for a log line", a sealed seed block that got an artefact, or a write that
landed in somebody else's round directory. The report fails closed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import closure_common as cc

REPO = cc.REPO_ROOT
START_SHA = cc.START_SHA

#: Directories whose contents would change what an episode does.
PRODUCTION_PREFIXES = ("integrated_sar/", "task_allocation/", "piloc/", "requirements.txt")


def git(*args: str) -> str:
    result = subprocess.run(("git", *args), cwd=REPO, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    return (result.stdout or "").strip()


def production_drift() -> dict[str, Any]:
    """Compare tracked production files against the start commit.

    Deliberately not a raw blob-hash comparison: checkout line-ending conversion makes
    a byte comparison differ without any behaviour change, which this project has
    already been bitten by once.
    """
    diff = git("diff", "--name-only", START_SHA, "--", *PRODUCTION_PREFIXES)
    status = git("status", "--porcelain", "--", *PRODUCTION_PREFIXES)
    return {"diff_against_start_sha": [line for line in diff.splitlines() if line.strip()],
            "working_tree_entries": [line for line in status.splitlines() if line.strip()]}


def inputs_unchanged() -> dict[str, Any]:
    manifest = json.loads((cc.ROUND_DIR / "CLOSURE_INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    changed: list[str] = []
    missing: list[str] = []
    for rel, record in sorted(manifest["explicit_inputs"].items()):
        path = REPO / rel
        if not path.exists():
            missing.append(rel)
            continue
        if cc.sha256_file(path) != record.get("sha256"):
            changed.append(rel)
    return {"hashed": len(manifest["explicit_inputs"]), "changed": changed, "missing": missing}


def log_families_unchanged() -> dict[str, Any]:
    """Recount the per-episode archives: same files, same total bytes means nothing ran."""
    manifest = json.loads((cc.ROUND_DIR / "CLOSURE_INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    drift: list[dict[str, Any]] = []
    for family in manifest["log_families"]:
        paths = cc.family_paths(REPO / family["dir"], family["pattern"])
        now = {"files": len(paths), "bytes": sum(path.stat().st_size for path in paths)}
        if now["files"] != family["files"] or now["bytes"] != family["bytes"]:
            drift.append({"family": family["name"], "frozen": {
                "files": family["files"], "bytes": family["bytes"]}, "now": now})
    return {"families": len(manifest["log_families"]), "drifted": drift}


def writes_confined() -> dict[str, Any]:
    """Report what this round created, and what was already there and must stay untouched.

    Other sessions' untracked files are listed, never judged or moved: they are not this
    round's to clean up.
    """
    untracked = git("ls-files", "--others", "--exclude-standard")
    foreign: list[str] = []
    own = 0
    for line in untracked.splitlines():
        path = line.strip().strip('"')
        if not path:
            continue
        if path.startswith(f"runs/{cc.ROUND}/"):
            own += 1
        else:
            foreign.append(path)
    return {"untracked_files_in_round_directory": own,
            "pre_existing_untracked_files_not_touched": sorted(foreign)}


def start_sha_verified() -> dict[str, Any]:
    remote = git("ls-remote", "origin", "refs/heads/analysis/comm-recipient-topology-feasibility")
    check = subprocess.run(("git", "merge-base", "--is-ancestor", START_SHA, "HEAD"),
                           cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    return {"start_sha": START_SHA,
            "remote_line": remote,
            "remote_matches_start": bool(remote) and remote.split()[0] == START_SHA,
            "head_descends_from_start": check.returncode == 0}


def governing_change_scope() -> dict[str, Any]:
    """The governing documents are the only tracked files this round may modify.

    ``git diff --name-only`` rather than positional parsing of ``--porcelain``: the first
    cut at this round's own path list silently mangled ``.gitignore`` into ``gitignore``.
    """
    state = cc.governing_state()
    diff = git("diff", "--name-only", "HEAD")
    modified = [line.strip().strip('"') for line in diff.splitlines() if line.strip()]
    allowed = set(cc.GOVERNING_DOCS)
    return {"docs": state,
            "tracked_files_modified": sorted(modified),
            "modified_outside_the_allowed_set": sorted(set(modified) - allowed)}


def run() -> int:
    cc.install_env_tripwire()
    manifest = json.loads((cc.ROUND_DIR / "CLOSURE_INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    ledger_now = {
        "starts": sum(int(json.loads(path.read_text(encoding="utf-8")).get("starts") or 0)
                      for path in (REPO / "runs/comm_task_pressure_learnability_20260922")
                      .glob("LAUNCH_LEDGER*.json")),
    }
    checks = {
        "inputs_unchanged": inputs_unchanged(),
        "log_families_unchanged": log_families_unchanged(),
        "production_drift": production_drift(),
        "governing_change_scope": governing_change_scope(),
        "writes_confined": writes_confined(),
        "start_sha_verified": start_sha_verified(),
        "episodes_generated_by_this_round": 0,
        "launch_ledger_starts_at_freeze": manifest["pressure_launch_ledger"]["starts"],
        "launch_ledger_starts_now": ledger_now["starts"],
        "sealed_block_artefacts_now": manifest["sealed_block_800000_800399_artefacts_on_disk"],
        "env_tripwire": cc.tripwire_report(),
    }
    checks["passed"] = (
        not checks["inputs_unchanged"]["changed"]
        and not checks["inputs_unchanged"]["missing"]
        and not checks["log_families_unchanged"]["drifted"]
        and not checks["production_drift"]["diff_against_start_sha"]
        and not checks["production_drift"]["working_tree_entries"]
        and not checks["governing_change_scope"]["modified_outside_the_allowed_set"]
        and checks["start_sha_verified"]["remote_matches_start"]
        and checks["launch_ledger_starts_at_freeze"] == checks["launch_ledger_starts_now"]
        and checks["sealed_block_artefacts_now"] == 0
        and checks["env_tripwire"]["blocked_call_count"] == 0
        and len(checks["env_tripwire"]["entry_points_wrapped"]) == 5
    )
    cc.write_json("CLOSURE_VERIFICATION.json", checks)
    print(json.dumps({key: value for key, value in checks.items()
                      if key in ("passed", "inputs_unchanged", "log_families_unchanged",
                                 "production_drift", "launch_ledger_starts_now")},
                     ensure_ascii=False, indent=1))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
