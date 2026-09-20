"""The round manifest: what ran, on which code, under which protocol, and what stays sealed.

Four facts the manifest exists to make checkable, in the order a reader needs them.

**Which protocol executed.** A run label says ``policy_owned_timing`` or nothing about it. The
manifest records the mode from the relay's own ``transport_parameters`` statement, per run, so a
table row can be traced to a protocol rather than to a name.

**Which code executed.** Commits and executed code are not the same object, so every module the
episodes import is hashed by content and compared against ``HEAD`` -- as a *blob id*, because a
raw byte comparison would report a false mismatch whenever the checkout writes CRLF for a file
whose blob is LF.

**Which weights produced which row.** Every checkpoint with its sha256, the steps recorded inside
it, its weight fingerprint, and whether the trainer that wrote it was this round's.

**What was not looked at.** The reserved final-test block, the locked byte credits, the stream
blocks, and the artefacts that deliberately stay out of Git.

    python protocol_manifest.py --threads 1
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

import numpy as np
import torch

import protocol_common as common

import train_pilot

#: Production modules whose contents the executed episodes depend on. ``comm.py`` and
#: ``comm_policy.py`` carry the protocol change, so they are here on purpose rather than assumed.
EXECUTED_MODULES = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/task03_runtime.py",
    "integrated_sar/allocation_trace.py",
    "integrated_sar/run_signature.py",
    "integrated_sar/expired_attribution.py",
    "integrated_sar/lifecycle_metrics.py",
    "integrated_sar/orchestrator.py",
)

#: Round scripts the executed episodes and evaluations depend on, hashed the same way.
ROUND_SCRIPTS = (
    "protocol_common.py",
    "protocol_env.py",
    "mask_audit.py",
    "rule_matrix.py",
    "train_protocol.py",
    "evaluate_protocol.py",
    "protocol_v2_preflight.py",
    "protocol_manifest.py",
)

#: Round-4 scripts this round imports rather than restates.
INHERITED_SCRIPTS = (
    "runs/comm_tight_budget_20260919/onpolicy.py",
    "runs/comm_tight_budget_20260919/train_onpolicy.py",
    "runs/comm_tight_budget_20260919/tight_common.py",
    "runs/comm_stability_20260919/attribution_metrics.py",
    "runs/comm_budget_pilot_20260919/pilot_env.py",
    "runs/comm_budget_pilot_20260919/train_pilot.py",
    "runs/comm_stability_20260919/data_split_audit.py",
)

PROTOCOL_DOCUMENTS = (
    "COMM_PROTOCOL_V2_PROTOCOL.md",
    "COMM_MASK_CAUSE_AUDIT.md",
    "COMM_PROTOCOL_V2_PREFLIGHT.json",
    "COMM_PROTOCOL_V2_EQUIVALENCE.json",
    "COMM_PROTOCOL_V2_RULE_SELECTION.json",
    "COMM_PROTOCOL_V2_THROUGHPUT.json",
    "COMM_MASK_CAUSE_AUDIT.json",
)

_UPDATE_LINE = re.compile(
    r"\[(?P<tier>\d+pct) seed (?P<seed>\d+)\] steps=\s*(?P<steps>\d+)\s+"
    r"on_time=(?P<on_time>\d+)/(?P<targets>\d+) bytes=(?P<bytes>\d+) "
    r"use=(?P<use>[-\d.]+) free=(?P<free>\d+)/(?P<candidates>\d+) "
    r"send_rate=(?P<send_rate>[-\d.]+) samples=(?P<samples>\d+) "
    r"opt=(?P<optimizer_steps>\d+) kl=(?P<kl>[-\d.e+]+) clip=(?P<clip>[-\d.e+]+) "
    r"ent=(?P<entropy>[-\d.e+]+) ent_free=(?P<ent_free>[-\d.e+]+).*?\((?P<elapsed>\d+)s\)"
)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=str(common.ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


def blob_of(relative: str) -> str:
    return git("hash-object", "--", relative)


def executed_code() -> dict[str, Any]:
    """Content identity of every module the runs import, judged the way git judges it.

    Raw bytes would report a false mismatch whenever the checkout writes CRLF for a file whose
    blob is LF -- a line-ending artefact of ``.gitattributes``, not a difference in executed code.
    ``git hash-object`` applies the same clean filter a commit would, so the comparison is between
    two blob ids and means "the code that ran is the code that is stored".
    """
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
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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
        scripts[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "blob": blob_of(relative),
        }
    inherited: dict[str, Any] = {}
    for relative in INHERITED_SCRIPTS:
        path = common.ROOT / relative
        if not path.exists():
            continue
        inherited[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "blob": blob_of(relative),
        }
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


def documents() -> dict[str, Any]:
    return {
        name: {"sha256": hashlib.sha256((common.HERE / name).read_bytes()).hexdigest()}
        for name in PROTOCOL_DOCUMENTS
        if (common.HERE / name).exists()
    }


def _controller_of(path: Path, training_seed: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    controller = train_pilot.Controller(training_seed)
    controller.load_state_dict(payload["controller"])
    return controller, payload


def checkpoints() -> dict[str, Any]:
    """Every checkpoint this round produced, with the steps recorded inside it."""
    found: dict[str, Any] = {}
    for tier in common.BUDGET_TIERS:
        for training_seed in (1, 2, 3):
            directory = common.TRAIN_OUT / f"{tier}_seed{training_seed}"
            if not directory.exists():
                continue
            for path in sorted(directory.glob(f"v2_{tier}_seed{training_seed}*.pt")):
                controller, payload = _controller_of(path, training_seed)
                state = payload.get("state") or {}
                found[f"{tier}/seed{training_seed}/{path.stem}"] = {
                    "path": str(path),
                    "sha256": common.sha256(path),
                    "trainer": payload.get("trainer"),
                    "timing_mode": payload.get("timing_mode"),
                    "environment_steps": int(state.get("environment_steps", -1)),
                    "episodes": int(state.get("episodes", -1)),
                    "updates": len(state.get("updates", [])),
                    "optimizer_steps": sum(
                        int(item.get("optimizer_steps", 0)) for item in state.get("updates", [])
                    ),
                    "decision_samples": sum(
                        int(item.get("samples", 0)) for item in state.get("updates", [])
                    ),
                    "free_decision_terms": sum(
                        int(item.get("free_candidate_samples", 0)) for item in state.get("updates", [])
                    ),
                    "budget_bytes": int(state.get("budget_bytes", -1)),
                    "initial_weights_fingerprint": state.get("initial_weights_fingerprint"),
                    "final_weights_fingerprint": common.weight_fingerprint(controller),
                    "likelihood_max_abs_gap": max(
                        (
                            float(item["likelihood_alignment"]["max_abs_gap"])
                            for item in state.get("updates", [])
                            if "likelihood_alignment" in item
                        ),
                        default=None,
                    ),
                    "selected_by_rule": state.get("selected_by_rule")
                    or _summary_selection(state.get("evaluations") or []),
                    "git_tracked": False,
                }
    return found


def _summary_selection(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not evaluations:
        return None
    best = max(evaluations, key=lambda item: (float(item["mean_on_time"]), -float(item["mean_airtime_bytes"])))
    return {"at_steps": best["at_steps"], "mean_on_time": best["mean_on_time"]}


def training_log_facts() -> dict[str, Any]:
    """Throughput, optimisation load and protocol facts read from the runs' own log lines."""
    per_run: dict[str, Any] = {}
    for tier in common.BUDGET_TIERS:
        for training_seed in (1, 2, 3):
            path = common.HERE / f"train_{tier}_seed{training_seed}.log"
            if not path.exists():
                continue
            updates = [
                match.groupdict()
                for match in (
                    _UPDATE_LINE.search(line) for line in path.read_text(encoding="utf-8").splitlines()
                )
                if match
            ]
            if not updates:
                continue
            elapsed = [int(item["elapsed"]) for item in updates]
            deltas = [second - first for first, second in zip(elapsed, elapsed[1:])]
            per_run[f"{tier}/seed{training_seed}"] = {
                "updates_logged": len(updates),
                "environment_steps": int(updates[-1]["steps"]),
                "decision_samples": sum(int(item["samples"]) for item in updates),
                "free_decision_terms": sum(int(item["free"]) for item in updates),
                "candidates_seen": sum(int(item["candidates"]) for item in updates),
                "free_candidates_per_decision": round(
                    sum(int(item["free"]) for item in updates)
                    / max(1, sum(int(item["candidates"]) for item in updates)),
                    4,
                ),
                "optimizer_steps": sum(int(item["optimizer_steps"]) for item in updates),
                "seconds_observed": elapsed[-1],
                "median_seconds_per_update": round(float(np.median(deltas)), 1) if deltas else None,
                "mean_seconds_per_update": round(float(np.mean(deltas)), 1) if deltas else None,
                "environment_steps_per_second": round(
                    int(updates[-1]["steps"]) / max(1.0, float(elapsed[-1])), 1
                ),
                "kl_last": float(updates[-1]["kl"]),
                "clip_last": float(updates[-1]["clip"]),
                # Two entropy readings, named for what they are: the per-step number is a sum over
                # free candidates and grows when the choice surface grows, which round 4 learned to
                # distrust; the per-candidate one is the comparable quantity.
                "entropy_last_per_step_sum": float(updates[-1]["entropy"]),
                "entropy_last_per_free_candidate": float(updates[-1]["ent_free"]),
                "drawn_send_rate_last": float(updates[-1]["send_rate"]),
                "batch_on_time_last": int(updates[-1]["on_time"]),
                "batch_bytes_last": int(updates[-1]["bytes"]),
            }
    rates = [value["mean_seconds_per_update"] for value in per_run.values() if value["mean_seconds_per_update"]]
    updates_per_run = 400_000 / (common.STREAM_STEPS * train_pilot.EPISODES_PER_UPDATE)
    return {
        "runs": per_run,
        "note": (
            "six runs in flight on one machine, torch pinned to one thread each, so the per-update "
            "seconds include contention: the honest wall-clock number for this batch rather than "
            "for one run alone"
        ),
        "runs_launched_in_parallel": max(1, len(per_run)),
        "updates_to_400k_per_run": int(updates_per_run),
        "projected_serial_hours_per_run_at_400k": (
            round(float(np.mean(rates)) * updates_per_run / 3600, 2) if rates else None
        ),
    }


def protocol_facts() -> dict[str, Any]:
    """The protocol each arm ran under, stated by the run and by the arm table."""
    locked = (
        json.loads(common.json_path("COMM_PROTOCOL_V2_RULE_SELECTION.json").read_text(encoding="utf-8"))
        if common.json_path("COMM_PROTOCOL_V2_RULE_SELECTION.json").exists()
        else {}
    )
    return {
        "timing_modes": {"legacy": common.LEGACY_MODE, "research": common.POLICY_MODE},
        "declared_bound": {
            "max_wait_steps": common.BOUND_MAX_WAIT_STEPS,
            "urgent_wait_steps": common.BOUND_URGENT_WAIT_STEPS,
            "urgency_time": common.BOUND_URGENCY_TIME,
        },
        "budget_tiers_bytes": {tier: common.tier_bytes(tier) for tier in common.BUDGET_TIERS},
        "comparator_locked_on_training_streams": {
            tier: block["selected_primary"] for tier, block in (locked.get("tiers") or {}).items()
        },
        "train_seeds": {"count": len(common.TRAIN_SEEDS), "first": int(common.TRAIN_SEEDS[0]), "last": int(common.TRAIN_SEEDS[-1])},
        "validation_seeds": {
            "count": len(common.VALIDATION_SEEDS),
            "first": int(common.VALIDATION_SEEDS[0]),
            "last": int(common.VALIDATION_SEEDS[-1]),
        },
        "gossip_interval_when_isolated": common.ISOLATED_GOSSIP_INTERVAL,
    }


def artefacts_not_in_git() -> dict[str, Any]:
    """Counts and folded hashes of the raw evidence, so nothing is lost by not committing it."""
    raw: dict[str, Any] = {}
    for label, pattern in (
        ("mask_audit_cells", "mask_audit/cell_*.json"),
        ("rule_matrix_rows", "rules/row_*.json"),
        ("equivalence_runs", "rules/owned_*"),
        ("evaluation_rows", "evaluation/*/rows_*.jsonl"),
        ("evaluation_curve_rows", "evaluation/*/rows_*_curve.jsonl"),
        ("evaluation_diagnostic_rows", "evaluation/*/rows_*_diag.jsonl"),
    ):
        files = sorted(common.HERE.glob(pattern))
        digest = hashlib.sha256()
        total = 0
        for path in files:
            if not path.is_file():
                continue
            total += path.stat().st_size
            digest.update(path.read_bytes())
        raw[label] = {
            "files": len(files),
            "bytes": total,
            "sha256_fold": digest.hexdigest() if files else None,
        }
    streams = sorted((common.pilot.STREAMS).glob("s*.json")) if common.pilot.STREAMS.exists() else []
    stream_digest = hashlib.sha256()
    for path in streams:
        stream_digest.update(path.read_bytes())
    raw["recorded_streams"] = {
        "files": len(streams),
        "sha256_fold": stream_digest.hexdigest() if streams else None,
        "note": "the recorded detection streams this round replays; committed upstream in the pilot round",
    }
    raw["checkpoints"] = {
        "files": len(list(common.TRAIN_OUT.glob("*/*.pt"))) if common.TRAIN_OUT.exists() else 0,
        "git_tracked": False,
    }
    return raw


def sealed() -> dict[str, Any]:
    import data_split_audit

    split = data_split_audit.audit(data_split_audit.CANDIDATE_BLOCK, data_split_audit.PREVIOUS_TEST_BLOCK)
    return {
        "reserved_final_test_block": split["replacement"]["block"],
        "reserved_final_test_clean": split["replacement"]["clean"],
        "previous_registered_block_executed_seeds": split["previously_registered_final_test"][
            "seeds_with_executed_episodes"
        ],
        "budget_tiers_are_locked_calibration_values": {
            tier: common.tier_bytes(tier) == common.locked_budget(tier) for tier in common.BUDGET_TIERS
        },
    }


def rerun() -> dict[str, Any]:
    return {
        "mask_audit": "python runs/comm_protocol_v2_20260920/mask_audit.py --mode run --learned --shards 6 --shard 0 --threads 1",
        "equivalence": "python runs/comm_protocol_v2_20260920/rule_matrix.py --mode pairs --threads 1",
        "rule_matrix": "python runs/comm_protocol_v2_20260920/rule_matrix.py --mode matrix --shards 6 --shard 0 --threads 1",
        "rule_lock": "python runs/comm_protocol_v2_20260920/rule_matrix.py --mode report",
        "gate": "python runs/comm_protocol_v2_20260920/protocol_v2_preflight.py --threads 1",
        "throughput": "python runs/comm_protocol_v2_20260920/train_protocol.py --mode throughput --tier 25pct --seed 1 --episodes 8 --threads 1 --record",
        "training": (
            "python runs/comm_protocol_v2_20260920/train_protocol.py --mode train --tier 25pct "
            "--seed 1 --steps 400000 --threads 1"
        ),
        "evaluation": (
            "python runs/comm_protocol_v2_20260920/evaluate_protocol.py --mode arms --shards 6 "
            "--shard 0 --threads 1 && python runs/comm_protocol_v2_20260920/evaluate_protocol.py "
            "--mode report"
        ),
        "manifest": "python runs/comm_protocol_v2_20260920/protocol_manifest.py --threads 1",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=common.json_path("COMM_PROTOCOL_V2_MANIFEST.json")
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))
    payload = {
        "round": "comm_protocol_v2",
        "date": "2026-09-20",
        "stage": "S3 protocol ablation: who decides to speak at the silence bound",
        "question": (
            "With the shared 3/10-step takeover removed from the transport, can a learned policy "
            "beat a strong rule that owns the same choice, under the same hard credit?"
        ),
        "executed_code": executed_code(),
        "protocol": protocol_facts(),
        "documents": documents(),
        "training": training_log_facts(),
        "checkpoints": checkpoints(),
        "artefacts_not_in_git": artefacts_not_in_git(),
        "sealed": sealed(),
        "rerun": rerun(),
    }
    payload["checks"] = {
        "all_executed_modules_match_head": payload["executed_code"]["all_modules_match_head"],
        "reserved_final_test_clean": payload["sealed"]["reserved_final_test_clean"],
        "budgets_are_locked": all(payload["sealed"]["budget_tiers_are_locked_calibration_values"].values()),
        "runs_reached_400k": all(
            value["environment_steps"] >= 400_000
            for value in payload["training"]["runs"].values()
        ) if payload["training"]["runs"] else False,
        "checkpoints_count": len(payload["checkpoints"]),
    }
    common.write_json(args.out, payload)
    print(json.dumps(payload["checks"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
