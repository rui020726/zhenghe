"""This round's manifest: what code ran, what it produced, and where the rest lives.

Same discipline as the previous rounds, because the same failure modes apply: a manifest
regenerated *after* a commit records the delivered code rather than the code that ran, and a
raw-byte comparison reports false mismatches whenever the checkout writes CRLF for an LF blob.
So executed modules are judged by git blob identity, the streams and per-episode logs stay out
of git with their sha256 folded here, and the sealed learning block is checked rather than
asserted.

    python confirm_manifest.py --threads 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import confirm_common as common

#: Production modules this round's runs import. Judged against HEAD by blob identity.
EXECUTED_MODULES = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_ack.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/allocation_trace.py",
    "integrated_sar/contracts.py",
    "integrated_sar/task_registry.py",
    "integrated_sar/orchestrator.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/adapters/piloc.py",
    "integrated_sar/lifecycle_log.py",
    "integrated_sar/lifecycle_metrics.py",
    "integrated_sar/expired_attribution.py",
    "integrated_sar/tests/test_comm_ack_feedback.py",
)

#: Round scripts, and the ones inherited from earlier rounds rather than restated.
ROUND_SCRIPTS = (
    "confirm_common.py",
    "confirm_env.py",
    "confirm_matrix.py",
    "confirm_preflight.py",
    "confirm_manifest.py",
    "confirm_verdict.py",
    "render_results.py",
    "block_audit.py",
    "record_streams.py",
    "test_confirm_harness.py",
)

#: Harness defects found *after* the episodes ran, registered rather than quietly fixed. Each
#: entry names what was wrong, what it touched, and whether the recorded episodes had to be
#: regenerated. A fix that changes how an episode is produced invalidates its artefacts; a fix
#: that only changes how a summary is computed does not, and the difference is stated here
#: instead of being assumed.
POST_RUN_HARNESS_FIXES: tuple[dict[str, Any], ...] = (
    {
        "found_by": "confirm_matrix.py --mode report",
        "defect": (
            "the per-additional-target ratio resampled the paired vectors but compared a "
            "per-replicate mean against a per-seed mask, so the bootstrap crashed on the "
            "first comparison (ValueError: operands could not be broadcast)"
        ),
        "touches": "ratio_bootstrap, the report stage only",
        "episodes_invalidated": False,
        "action": (
            "mean per replicate first, then the undefined mask, so a replicate whose resampled "
            "denominator is not positive is dropped whole; the report was regenerated from the "
            "unchanged 3,000 rows"
        ),
    },
    {
        "found_by": "confirm_preflight.py",
        "defect": (
            "the ACK ledger was snapshotted before the episode ran, so every ACK-on row "
            "compared a zero ledger against the transcript bytes (60 of 75 probe episodes "
            "flagged); the preflight also called run_arm with the directory positionally, "
            "compared a JSON list against a configuration tuple, and read the locked budget "
            "from a module that does not export it"
        ),
        "touches": "confirm_env (ledger read timing), confirm_preflight (four checks)",
        "episodes_invalidated": False,
        "action": (
            "the ledger is now read after the episode; the preflight checks were corrected and "
            "all twelve pass. The 3,000 matrix rows were produced by run_arm and validate_row, "
            "neither of which changed"
        ),
    },
)

INHERITED_SCRIPTS = (
    "runs/comm_protocol_v2_20260920/protocol_common.py",
    "runs/comm_protocol_v2_20260920/protocol_env.py",
    "runs/comm_tight_budget_20260919/tight_common.py",
    "runs/comm_stability_20260919/attribution_metrics.py",
    "runs/comm_stability_20260919/data_split_audit.py",
    "runs/comm_budget_pilot_20260919/pilot_env.py",
    "runs/comm_budget_pilot_20260919/pilot_common.py",
    "runs/comm_budget_pilot_20260919/evaluate_pilot.py",
)

ROUND_DOCUMENTS = (
    "COMM_RULE_CONFIRMATION_PROTOCOL.md",
    "COMM_RULE_CONFIRMATION_RESULTS.md",
    "COMMUNICATION_LEARNING_VERDICT.md",
    "COMM_RULE_BLOCK_AUDIT.json",
    "COMM_RULE_STREAMS.json",
    "COMM_RULE_THROUGHPUT.json",
    "COMM_RULE_PREFLIGHT.json",
    "COMM_RULE_CONFIRMATION_RESULTS.json",
    "COMMUNICATION_LEARNING_VERDICT.json",
)

#: The sealed learning final test block. Registered, never generated, never read.
SEALED_BLOCK = (800000, 800400)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(common.ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def blob_of(relative: str) -> str:
    return git("hash-object", "--", relative)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executed_code() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    modules: dict[str, Any] = {}
    for relative in EXECUTED_MODULES:
        path = common.ROOT / relative
        working_blob = blob_of(relative)
        try:
            committed_blob = git("rev-parse", f"HEAD:{relative}")
        except subprocess.CalledProcessError:
            committed_blob = "<absent-at-head>"
        modules[relative] = {
            "sha256": sha256_of(path),
            "blob": working_blob,
            "blob_at_head": committed_blob,
            "matches_head": working_blob == committed_blob,
        }
    scripts: dict[str, Any] = {}
    for name in ROUND_SCRIPTS:
        path = common.HERE / name
        if not path.exists():
            continue
        relative = str(path.relative_to(common.ROOT)).replace("\\", "/")
        try:
            at_head = git("rev-parse", f"HEAD:{relative}")
        except subprocess.CalledProcessError:
            at_head = "<absent-at-head>"
        scripts[name] = {
            "sha256": sha256_of(path),
            "blob": blob_of(relative),
            "blob_at_head": at_head,
            "matches_head": blob_of(relative) == at_head,
        }
    inherited: dict[str, Any] = {}
    for relative in INHERITED_SCRIPTS:
        path = common.ROOT / relative
        if not path.exists():
            continue
        inherited[relative] = {"sha256": sha256_of(path), "blob": blob_of(relative)}
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": head,
        "comparison": "git blob identity (line-ending filters applied)",
        "dirty_modules": sorted(
            name for name, value in modules.items() if not value["matches_head"]
        ),
        "all_modules_match_head": all(value["matches_head"] for value in modules.values()),
        "dirty_round_scripts": sorted(
            name for name, value in scripts.items() if not value["matches_head"]
        ),
        "modules": modules,
        "round_scripts": scripts,
        "inherited_scripts": inherited,
    }


def fold(paths: list[Path]) -> dict[str, Any]:
    """A directory listing with one sha256 fold over the file contents, not the names."""
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        total_bytes += len(payload)
        digest.update(str(path.relative_to(path.parents[1])).encode("utf-8"))
        digest.update(payload)
    return {"files": len(paths), "bytes": total_bytes, "sha256_fold": digest.hexdigest()}


def artefacts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("streams", "evaluation", "preflight", "record_logs", "stream_reruns"):
        directory = common.HERE / name
        if not directory.exists():
            continue
        files = [path for path in directory.rglob("*") if path.is_file()]
        out[name] = fold(files)
    shards = [path for path in common.HERE.glob("matrix_shard*.json") if path.is_file()]
    if shards:
        out["matrix_shards"] = fold(shards)
    return out


def sealed_block_check() -> dict[str, Any]:
    """The sealed learning block must not exist as a recorded stream anywhere in this round."""
    low, high = SEALED_BLOCK
    roots = [common.EVALUATION_OUT, common.HERE / "preflight", common.STREAMS, common.HERE]
    leaks: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            for token in re.findall(r"\d{6}", path.name):
                if low <= int(token) <= high:
                    leaks.append(str(path.relative_to(common.HERE)))
                    break
    return {
        "block": [low, high - 1],
        "generated": False,
        "read": False,
        "artefact_leaks": sorted(set(leaks)),
        "clean": not leaks,
    }


def streams_summary() -> dict[str, Any]:
    path = common.HERE / "COMM_RULE_STREAMS.json"
    if not path.exists():
        return {"present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "block": payload["block"],
        "size": payload["size"],
        "fold": payload["fold"],
        "block_audit_clean": payload["block_audit"]["clean"],
        "recording_configuration": payload["recording_configuration"],
        "seeds_with_no_detections": payload["seeds_with_no_detections"],
    }


def results_summary() -> dict[str, Any]:
    path = common.HERE / "COMM_RULE_CONFIRMATION_RESULTS.json"
    if not path.exists():
        return {"present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "present": True,
        "episodes": payload["episodes"],
        "episodes_expected": payload["episodes_expected"],
        "incomplete_cells": {
            setting: value
            for setting, value in payload["incomplete_cells"].items()
            if value
        },
        "revalidated_rejections": payload["revalidated_rejections"],
        "settings": {},
    }
    for setting, block in payload["judgement"].items():
        summary["settings"][setting] = {
            "per_arm_on_time": {
                arm: payload["settings"][setting][arm]["on_time_per_episode"]
                for arm in payload["settings"][setting]
            },
            "per_arm_total_bytes": {
                arm: payload["settings"][setting][arm]["total_bytes"]
                for arm in payload["settings"][setting]
            },
            "frontier": block["pareto"]["frontier"],
            "dominated": block["pareto"]["dominated"],
        }
    return summary


def documents() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ROUND_DOCUMENTS:
        path = common.HERE / name
        if path.exists():
            out[name] = {"sha256": sha256_of(path)}
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    payload = {
        "round": "comm-rule-independent-confirmation",
        "brief": "COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md",
        "charter_version": "v1.9",
        "branch_start": {
            "document_commit": "2e341b1b321feef6c270b82cf81dd1f9d5c3b3bd",
            "code_base": "feat/comm-ack-feedback-gate@fbf96d8aa30fb2600fd710eab79a7b5f8121422a",
        },
        "executed_code": executed_code(),
        "post_run_harness_fixes": list(POST_RUN_HARNESS_FIXES),
        "documents": documents(),
        "streams": streams_summary(),
        "artefacts_not_in_git": artefacts(),
        "sealed_final_test_block": sealed_block_check(),
        "results": results_summary(),
        "training": {
            "permitted": False,
            "reason": (
                "the brief freezes the S3 learning verdict in this round: it confirms locked "
                "rules on fresh engineering streams and trains nothing"
            ),
            "runs": 0,
            "environment_steps": 0,
        },
        "defaults_unchanged": {
            "comm_policy": "immediate",
            "comm_budget_bytes": None,
            "comm_max_wait_steps": None,
            "comm_timing_mode": "legacy_shared_bound",
            "comm_ack": False,
            "comm_range": 0.3,
            "gossip_interval": 2,
        },
        "rerun": [
            "cd runs/comm_rule_confirm_20260921",
            "python block_audit.py",
            "python record_streams.py --shards 4 --shard 0",
            "python record_streams.py --mode merge",
            "python confirm_preflight.py --threads 1",
            "python confirm_matrix.py --mode probe",
            "python confirm_matrix.py --mode matrix --shards 8 --shard 0",
            "python confirm_matrix.py --mode report",
            "python render_results.py",
            "python confirm_verdict.py",
            "python -m unittest discover -s runs/comm_rule_confirm_20260921 -p \"test_*.py\"",
            "python confirm_manifest.py --threads 1",
        ],
    }
    common.write_json(common.json_path("COMM_RULE_CONFIRMATION_MANIFEST.json"), payload)
    print(
        json.dumps(
            {
                "executed_head": payload["executed_code"]["head"],
                "dirty_modules": payload["executed_code"]["dirty_modules"],
                "all_modules_match_head": payload["executed_code"]["all_modules_match_head"],
                "sealed_clean": payload["sealed_final_test_block"]["clean"],
                "episodes": payload["results"].get("episodes"),
                "manifest": "COMM_RULE_CONFIRMATION_MANIFEST.json",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
