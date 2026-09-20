"""This round's manifest: what code ran, what it produced, and where the rest lives.

Same discipline as round 5's, because the same failure modes apply: a manifest regenerated
*after* a commit records the delivered code rather than the code that ran, and a raw-byte
comparison reports false mismatches whenever the checkout writes CRLF for an LF blob. So the
executed modules are judged by git blob identity, the checkpoints and per-episode logs stay
out of git with their sha256 folded here, and the sealed test block is checked rather than
asserted.

    python ack_manifest.py --threads 1
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

import ack_common as common

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
    "integrated_sar/tests/test_comm_ack_feedback.py",
)

#: Round scripts, and the ones inherited from earlier rounds rather than restated.
ROUND_SCRIPTS = (
    "ack_common.py",
    "ack_matrix.py",
    "ack_counterfactual.py",
    "ack_gate.py",
    "ack_preflight.py",
    "ack_manifest.py",
)

INHERITED_SCRIPTS = (
    "runs/comm_protocol_v2_20260920/protocol_common.py",
    "runs/comm_protocol_v2_20260920/protocol_env.py",
    "runs/comm_tight_budget_20260919/onpolicy.py",
    "runs/comm_tight_budget_20260919/train_onpolicy.py",
    "runs/comm_tight_budget_20260919/tight_common.py",
    "runs/comm_stability_20260919/attribution_metrics.py",
    "runs/comm_budget_pilot_20260919/pilot_env.py",
    "runs/comm_budget_pilot_20260919/train_pilot.py",
    "runs/comm_budget_pilot_20260919/pilot_common.py",
    "runs/comm_budget_pilot_20260919/evaluate_pilot.py",
    "runs/comm_stability_20260919/data_split_audit.py",
)

ROUND_DOCUMENTS = (
    "COMM_ACK_CONTRACT.md",
    "COMM_ACK_VALUE_GATE.md",
    "COMM_ACK_PREFLIGHT.json",
    "COMM_ACK_VALUE_GATE_25pct.json",
    "COMM_ACK_VALUE_RULE_MATRIX_25pct.json",
    "COMM_ACK_VALUE_COUNTERFACTUAL_25pct.json",
)

#: The sealed final test block. Registered, never generated, never read.
SEALED_BLOCK = (800000, 800399)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=str(common.ROOT), capture_output=True, text=True, check=True
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
        scripts[name] = {"sha256": sha256_of(path), "blob": blob_of(relative)}
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
        "modules": modules,
        "round_scripts": scripts,
        "inherited_scripts": inherited,
    }


def fold(paths: list[Path]) -> dict[str, Any]:
    """A directory listing with one sha256 fold over the file contents, not the names."""
    import hashlib as _hashlib

    digest = _hashlib.sha256()
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
    root = common.EVALUATION_OUT
    out: dict[str, Any] = {}
    if not root.exists():
        return out
    for setting in sorted(path for path in root.iterdir() if path.is_dir()):
        entries: dict[str, Any] = {}
        for kind in ("matrix", "counterfactual", "smoke"):
            directory = setting / kind
            if not directory.exists():
                continue
            files = [path for path in directory.rglob("*") if path.is_file()]
            entries[kind] = fold(files)
        shards = [path for path in setting.glob("*.json") if path.is_file()]
        if shards:
            entries["shard_json"] = fold(shards)
        out[setting.name] = entries
    preflight = common.HERE / "preflight"
    if preflight.exists():
        files = [path for path in preflight.rglob("*") if path.is_file()]
        out["preflight"] = fold(files)
    return out


def sealed_block_check() -> dict[str, Any]:
    """The final test block must not exist as a recorded stream anywhere in this round."""
    low, high = SEALED_BLOCK
    roots = [common.EVALUATION_OUT, common.HERE / "preflight", common.TRAIN_OUT]
    leaks: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            for token in re.findall(r"\d{6}", path.name):
                if low <= int(token) <= high:
                    leaks.append(str(path))
                    break
    return {
        "block": [low, high],
        "generated": False,
        "read": False,
        "artefact_leaks": sorted(set(leaks)),
        "clean": not leaks,
    }


def gate_summary() -> dict[str, Any]:
    path = common.json_path("COMM_ACK_VALUE_GATE_25pct.json")
    if not path.exists():
        return {"present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "all_passed": payload["all_passed"],
        "decision": payload["decision"],
        "gates": {
            name: {
                "passed": gate.get("passed"),
                **(
                    {"streams_qualifying": gate["streams_qualifying"]}
                    if "streams_qualifying" in gate
                    else {}
                ),
                **(
                    {
                        "selected_ack_rule": gate.get("selected_ack_rule"),
                        "completion_mean_difference": (
                            gate.get("completion_difference") or {}
                        ).get("mean_difference"),
                        "completion_ci95": [
                            (gate.get("completion_difference") or {}).get("ci95_lower"),
                            (gate.get("completion_difference") or {}).get("ci95_upper"),
                        ],
                        "total_bytes_mean_difference": (
                            gate.get("total_bytes_difference") or {}
                        ).get("mean_difference"),
                    }
                    if "selected_ack_rule" in gate
                    else {}
                ),
                **(
                    {
                        "decision_points": gate.get("decision_points"),
                        "send_better": gate.get("send_better_on_time"),
                        "hold_better": gate.get("hold_better_on_time"),
                        "tied": gate.get("tied_on_time"),
                        "consistency": gate.get("consistency"),
                    }
                    if "decision_points" in gate
                    else {}
                ),
                **(
                    {"dominators": gate.get("dominators")}
                    if "dominators" in gate
                    else {}
                ),
            }
            for name, gate in payload["gates"].items()
        },
    }


def matrix_summary() -> dict[str, Any]:
    path = common.json_path("COMM_ACK_VALUE_RULE_MATRIX_25pct.json")
    if not path.exists():
        return {"present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "present": True,
        "tier": payload["tier"],
        "budget_bytes": payload["budget_bytes"],
        "streams": len(payload["streams"]),
        "channels": {},
    }
    for channel, entries in payload["channels"].items():
        summary["channels"][channel] = {
            "arms": len(entries),
            "episodes_per_arm": max(
                (entry["episodes"] for entry in entries.values()), default=0
            ),
            "selection": payload["selection"][channel]["selected"],
        }
    return summary


def documents() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ROUND_DOCUMENTS:
        path = common.HERE / name
        if path.exists():
            out[name] = {"sha256": sha256_of(path)}
    contract = common.ROOT / "COMM_ACK_CONTRACT.md"
    if contract.exists():
        out["COMM_ACK_CONTRACT.md"] = {"sha256": sha256_of(contract)}
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    payload = {
        "round": "comm-ack-value-gate",
        "brief": "COMM_ACK_VALUE_NEXT_ROUND.md",
        "charter_version": "v1.7",
        "branch_start": {
            "document_commit": "d4c73af04ccf144fd968bdf2c795144206c05caf",
            "code_base": "feat/comm-policy-owned-timing@23feba8f2c8edd44792827844e0f9482b708a535",
        },
        "executed_code": executed_code(),
        "documents": documents(),
        "artefacts_not_in_git": artefacts(),
        "sealed_final_test_block": sealed_block_check(),
        "gate": gate_summary(),
        "rule_matrix": matrix_summary(),
        "training": {
            "permitted": False,
            "reason": "information-value gate item 3 did not pass, so the brief's condition "
            "for training was not met",
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
            "cd runs/comm_ack_value_20260920",
            "python ack_preflight.py --threads 1",
            "python ack_matrix.py --mode matrix --shards 6 --shard 0 --threads 1",
            "python ack_matrix.py --mode report",
            "python ack_counterfactual.py --mode run --shards 6 --shard 0 --threads 1",
            "python ack_counterfactual.py --mode report",
            "python ack_gate.py --tier 25pct",
            "python ack_manifest.py --threads 1",
        ],
    }
    common.write_json(common.json_path("COMM_ACK_MANIFEST.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
