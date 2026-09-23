"""What the two batches can and cannot answer, field by field, with file evidence.

The point of this script is that "the logs are not enough" has to be a checked statement
about named fields, not a sentence in a report. Every claim below is either a key that was
found in the archived records, or a key whose absence was searched for in all of them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import rs_common as rc

rc.enter_repo()
rc.install_env_tripwire()

#: Source locations whose behaviour the inventory depends on, quoted as of the audited
#: revision rather than paraphrased from memory.
CODE_EVIDENCE: tuple[tuple[str, str, int, int], ...] = (
    ("integrated_sar/adapters/piloc.py", "placement snapshot taken at reset", 95, 101),
    ("integrated_sar/adapters/piloc.py", "detection event carries the goal cell of this step",
     129, 137),
    ("piloc/src/Map/timemap.py", "targets take one random free-cell step per environment step",
     814, 845),
    ("piloc/src/Map/timemap.py", "found targets are deleted from goals_pos", 776, 782),
    ("piloc/src/Map/timemap.py", "the legal observation test (Chebyshev <= 5 and clear line)",
     577, 593),
)

#: Keys that would make the four-bucket attribution decidable.
WANTED = {
    "per_step_target_cells": ("target positions at every step", ("goal_positions", "targets_pos")),
    "per_step_search_agent_cells": ("search robot positions at every step",
                                    ("agents_pos", "agent_positions", "positions_history")),
    "search_end_positions": ("search robot positions at the episode cut", ("end_positions",)),
    "coverage_by_step": ("when a region was first swept", ("coverage", "seen", "visited")),
}


def json_keys(path: Path, depth_limit: int = 3) -> Counter:
    found: Counter = Counter()

    def walk(node, prefix: str, depth: int) -> None:
        if depth > depth_limit:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                found[prefix + str(key)] += 1
                walk(value, prefix + str(key) + ".", depth + 1)
        elif isinstance(node, list):
            for item in node[:3]:
                walk(item, prefix, depth + 1)

    for record in rc.iter_jsonl(path):
        walk(record, "", 1)
    return found


def git_show(path: str, start: int, end: int) -> str:
    result = subprocess.run(("git", "show", f"{rc.START_SHA}:{path}"), cwd=rc.REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (result.stdout or "").splitlines()
    return "\n".join(lines[start - 1:end])


def run() -> int:
    manifest: dict[str, dict[str, Any]] = {}
    out: dict[str, Any] = {"round": rc.ROUND, "start_sha": rc.START_SHA, "batches": {},
                           "wanted_fields": {}, "code_evidence": {}, "working_tree_drift": {}}

    for key, batch in rc.BATCHES.items():
        sample_arm = batch["search_record_arm"]
        seed = min(batch["seeds"])
        paths = rc.batch_paths(batch, sample_arm, seed)
        lifecycle = paths["lifecycle"]
        keys = json_keys(lifecycle)
        probe_keys: Counter = Counter()
        if paths.get("probe") and paths["probe"].exists():
            probe_keys = json_keys(paths["probe"])
        out["batches"][key] = {
            "sample_file": str(lifecycle.relative_to(rc.REPO_ROOT)).replace("\\", "/"),
            "lifecycle_keys": dict(sorted(keys.items())),
            "probe_keys": dict(sorted(probe_keys.items())) if probe_keys else None,
            "has_run_config_scenario": keys.get("detail.scenario.goal_positions", 0) > 0,
            "has_run_config_map_hash": keys.get("detail.scenario.map_sha256", 0) > 0,
        }

        # absence is searched over every file of the batch, not inferred from one sample.
        # A key that only appears in the run_config record is a snapshot, not a per-step
        # record, so this scan ignores that record on purpose.
        absent = {name: 0 for name, (_, _) in WANTED.items()}
        scanned = 0
        for arm in batch["arms"]:
            for episode_seed in batch["seeds"]:
                cell = rc.batch_paths(batch, arm, episode_seed)["lifecycle"]
                scanned += 1
                rc.register_input(cell, manifest)
                per_step_keys: set[str] = set()
                for record in rc.iter_jsonl(cell):
                    if record.get("event_type") == "run_config":
                        continue
                    per_step_keys |= set(record)
                    per_step_keys |= set((record.get("detail") or {}))
                for name, (_, candidates) in WANTED.items():
                    if per_step_keys & set(candidates):
                        absent[name] += 1
        if batch["probe"]:
            for arm in batch["arms"]:
                for episode_seed in batch["seeds"]:
                    rc.register_input(rc.batch_paths(batch, arm, episode_seed)["probe"], manifest)
        out["batches"][key]["files_scanned"] = scanned
        out["batches"][key]["per_step_files_holding_the_key"] = absent

    for name, (description, candidates) in WANTED.items():
        hits = {batch: row["per_step_files_holding_the_key"][name]
                for batch, row in out["batches"].items()}
        out["wanted_fields"][name] = {
            "description": description, "files_with_the_key": hits,
            "present_in_the_archived_logs": any(count > 0 for count in hits.values()),
        }

    for path, description, start, end in CODE_EVIDENCE:
        out["code_evidence"][f"{path}:{start}-{end}"] = {
            "what_it_fixes": description, "excerpt": git_show(path, start, end)}
        probe = subprocess.run(("git", "diff", "--name-only", rc.START_SHA, "--", path),
                               cwd=rc.REPO_ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        out["working_tree_drift"][path] = (probe.stdout or "").strip()

    out["inputs_read"] = {"count": len(manifest), "files": manifest}
    path = rc.write_json("FIELD_INVENTORY.json", out)

    for key, row in out["batches"].items():
        print(f"== {key}: scanned {row['files_scanned']} lifecycle files")
        for name, counts in row["per_step_files_holding_the_key"].items():
            print(f"   {name:<28} files where the key appears: {counts}")
    print("scenario block present in run_config:",
          {key: row["has_run_config_scenario"] for key, row in out["batches"].items()})
    print("working tree drift on cited code:", out["working_tree_drift"])
    print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
