"""The round manifest: what ran, on which code, with which locked numbers, and what stays sealed.

Two facts the manifest exists to make checkable.

**Which code executed.** The commits and the code that ran the episodes are not the same object, so
every module the runs import is hashed by content and compared against ``HEAD`` -- a bare HEAD would
let a claim about a modified sampler rest on a commit that predates it.

**What was not looked at.** The reserved final-test block, the round's own protocol hash, and the
checkpoint inventory with weight fingerprints, so a table row can be traced to the file that made it.
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

import tight_common as common

import train_pilot

#: Modules whose contents the executed episodes depend on.
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
)
#: Round scripts whose behaviour the runs depend on, hashed the same way.
ROUND_SCRIPTS = (
    "onpolicy.py",
    "train_onpolicy.py",
    "tight_common.py",
    "budget_probe.py",
    "probability_audit.py",
    "evaluate_tight.py",
    "ledger_gap_reproduction.py",
    "ledger_fix_invariance.py",
    "tight_preflight.py",
)
PROTOCOL_DOCUMENTS = (
    "COMM_PPO_PROBABILITY_AUDIT.md",
    "COMM_TIGHT_BUDGET_PROTOCOL.md",
    "COMM_LEDGER_GAP.md",
    "COMM_PPO_PROBABILITY_AUDIT.json",
    "COMM_TIGHT_BUDGET_GATE.json",
    "COMM_LEDGER_GAP_REPRODUCTION.json",
    "COMM_LEDGER_FIX_INVARIANCE.json",
    "TIGHT_SMOKE.json",
)
_UPDATE_LINE = re.compile(
    r"\[(?P<tier>\d+pct) seed (?P<seed>\d+)\] steps=\s*(?P<steps>\d+).*?"
    r"on_time=(?P<on_time>\d+)/(?P<targets>\d+) bytes=(?P<bytes>\d+).*?"
    r"samples=(?P<samples>\d+) opt=(?P<optimizer_steps>\d+) kl=(?P<kl>[-\d.e+]+) "
    r"clip=(?P<clip>[-\d.e+]+) ent=(?P<entropy>[-\d.e+]+).*?\((?P<elapsed>\d+)s\)"
)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=str(common.ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


def executed_code() -> dict[str, Any]:
    """Content identity of every module the runs import, judged the way git judges it.

    Raw bytes would report a false mismatch whenever the checkout writes CRLF for a file whose
    blob is LF -- a line-ending artefact of ``.gitattributes``, not a difference in executed code.
    ``git hash-object`` applies the same clean filter a commit would, so the comparison is between
    two blob ids and means "the code that ran is the code that is stored".
    """
    head = git("rev-parse", "HEAD")
    dirty = {
        line
        for line in git("diff", "--name-only", "HEAD", "--", *EXECUTED_MODULES).splitlines()
        if line
    }
    modules: dict[str, Any] = {}
    for relative in EXECUTED_MODULES:
        path = common.ROOT / relative
        working_blob = git("hash-object", "--", relative)
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
    scripts = {}
    for name in ROUND_SCRIPTS:
        path = common.HERE / name
        if not path.exists():
            continue
        scripts[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "blob": git("hash-object", "--", str(path.relative_to(common.ROOT))),
        }
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": head,
        "dirty_modules": sorted(dirty),
        "comparison": "git blob identity (line-ending filters applied)",
        "modules": modules,
        "round_scripts": scripts,
    }


def documents() -> dict[str, Any]:
    return {
        name: {"sha256": hashlib.sha256((common.HERE / name).read_bytes()).hexdigest()}
        for name in PROTOCOL_DOCUMENTS
        if (common.HERE / name).exists()
    }


def checkpoints() -> dict[str, Any]:
    """Every checkpoint this round produced, with the steps recorded inside it."""
    found: dict[str, Any] = {}
    for tier in common.BUDGET_TIERS:
        for training_seed in (1, 2, 3):
            directory = common.TRAIN_OUT / f"{tier}_seed{training_seed}"
            for path in sorted(directory.glob(f"tight_{tier}_seed{training_seed}*.pt")):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                state = payload["state"]
                found[f"{tier}/seed{training_seed}/{path.stem}"] = {
                    "path": str(path),
                    "sha256": common.sha256(path),
                    "trainer": payload.get("trainer"),
                    "collector": state.get("collector"),
                    "environment_steps": int(state.get("environment_steps", -1)),
                    "episodes": int(state.get("episodes", -1)),
                    "updates": len(state.get("updates", [])),
                    "optimizer_steps": sum(
                        int(item.get("optimizer_steps", 0)) for item in state.get("updates", [])
                    ),
                    "decision_samples": sum(
                        int(item.get("samples", 0)) for item in state.get("updates", [])
                    ),
                    "budget_bytes": int(state.get("budget_bytes", -1)),
                    "initial_weights_fingerprint": state.get("initial_weights_fingerprint"),
                    "final_weights_fingerprint": common.weight_fingerprint(
                        _load(path, training_seed)
                    ),
                    "likelihood_max_abs_gap": max(
                        (
                            float(item["likelihood_alignment"]["max_abs_gap"])
                            for item in state.get("updates", [])
                            if "likelihood_alignment" in item
                        ),
                        default=None,
                    ),
                    "git_tracked": False,
                }
    return found


def _load(path: Path, training_seed: int) -> Any:
    controller = train_pilot.Controller(training_seed)
    controller.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["controller"])
    return controller


def training_log_facts() -> dict[str, Any]:
    """Throughput and schedule read back from the runs' own log lines, not from a promise."""
    per_run: dict[str, Any] = {}
    for tier in common.BUDGET_TIERS:
        for training_seed in (1, 2, 3):
            path = common.HERE / f"train_{tier}_seed{training_seed}.log"
            if not path.exists():
                continue
            updates = [
                match.groupdict()
                for match in (_UPDATE_LINE.search(line) for line in path.read_text(encoding="utf-8").splitlines())
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
                "optimizer_steps": sum(int(item["optimizer_steps"]) for item in updates),
                "seconds_observed": elapsed[-1],
                "median_seconds_per_update": round(float(np.median(deltas)), 1) if deltas else None,
                "mean_seconds_per_update": round(float(np.mean(deltas)), 1) if deltas else None,
                "environment_steps_per_second": round(
                    int(updates[-1]["steps"]) / max(1.0, float(elapsed[-1])), 1
                ),
                "kl_last": float(updates[-1]["kl"]),
                "clip_last": float(updates[-1]["clip"]),
                "entropy_last": float(updates[-1]["entropy"]),
            }
    concurrency = max(1, len(per_run))
    updates_per_run = 400_000 / (common.STREAM_STEPS * train_pilot.EPISODES_PER_UPDATE)
    rates = [value["mean_seconds_per_update"] for value in per_run.values() if value["mean_seconds_per_update"]]
    return {
        "runs": per_run,
        "note": (
            "measured with six runs in flight on one machine and torch pinned to one thread each; "
            "the per-update seconds therefore include contention, which is the honest number for a "
            "wall-clock estimate of this batch rather than for one run alone"
        ),
        "runs_launched_in_parallel": concurrency,
        "updates_to_400k_per_run": int(updates_per_run),
        "projected_serial_hours_per_run_at_400k": (
            round(float(np.mean(rates)) * updates_per_run / 3600, 2) if rates else None
        ),
    }


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
            tier: common.BUDGETS_BYTES[tier] == common.locked_budget(tier)
            for tier in common.BUDGET_TIERS
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="COMM_TIGHT_BUDGET_MANIFEST.json")
    args = parser.parse_args(argv)
    payload = {
        "round": "comm_ppo_probability_and_tight_budget",
        "date": "2026-09-19",
        "stage": "S3 (charter v1.3)",
        "question": "does a probability-correct PPO beat strong rules when the budget really binds?",
        "executed_code": executed_code(),
        "frozen_documents": documents(),
        "budgets_bytes": common.BUDGETS_BYTES,
        "data": sealed(),
        "training": training_log_facts(),
        "checkpoints": checkpoints(),
        "artefacts_not_in_git": [
            "runs/comm_tight_budget_20260919/train/**",
            "runs/comm_tight_budget_20260919/probe/**",
            "runs/comm_tight_budget_20260919/probe_after/**",
            "runs/comm_tight_budget_20260919/evaluation/**",
            "runs/comm_tight_budget_20260919/audit/**",
        ],
        "rerun": {
            "probability_audit": "python runs/comm_tight_budget_20260919/probability_audit.py --episodes 6 --threads 1 --controller-path runs/comm_stability_20260919/train/seed1/stability_seed1_at400000.pt",
            "ledger_reproduction": "python runs/comm_tight_budget_20260919/ledger_gap_reproduction.py",
            "budget_gate": "python runs/comm_tight_budget_20260919/budget_probe.py --mode aggregate --output-subdir probe_after",
            "train": "python runs/comm_tight_budget_20260919/train_onpolicy.py --mode train --tier {tier} --seed {seed} --steps 400000 --threads 1",
            "evaluate": "python runs/comm_tight_budget_20260919/evaluate_tight.py --mode curve --tier {tier}; "
            "python runs/comm_tight_budget_20260919/evaluate_tight.py --mode settings --tier {tier} "
            "--information isolated --delivery-probability 1.0",
            "preflight": "python runs/comm_tight_budget_20260919/tight_preflight.py --threads 1",
            "tests": "python -m unittest discover -s integrated_sar/tests",
        },
    }
    if Path(common.HERE / "COMM_TIGHT_BUDGET_RESULTS.json").exists():
        payload["results_present"] = True
    common.write_json(common.HERE / args.out, payload)
    print(json.dumps(
        {
            "head": payload["executed_code"]["head"],
            "dirty_modules": payload["executed_code"]["dirty_modules"],
            "all_modules_match_head": all(
                value["matches_head"] for value in payload["executed_code"]["modules"].values()
            ),
            "checkpoints": len(payload["checkpoints"]),
            "reserved_final_test_clean": payload["data"]["reserved_final_test_clean"],
            "runs": {key: value["environment_steps"] for key, value in payload["training"]["runs"].items()},
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
