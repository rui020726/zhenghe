"""Write the round manifest: what ran, on which code, with which locked numbers.

Reproducible facts belong in one machine-readable place: the commit the production code came
from, the checkpoint bytes, the seed blocks and stream hashes, the locked budget credits and
the rule that picked them, the smoke and gate results, the training curves, and what was
deliberately not committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import evaluate_pilot
import pilot_common as common
import pilot_env
import train_pilot

from integrated_sar.comm_learning import FEATURE_LAYOUT

ROWS = common.HERE / "rows_calibration.jsonl"


def _defaults() -> dict[str, Any]:
    import dataclasses

    from integrated_sar.joint_driver import RunConfiguration

    wanted = {
        "comm_policy", "comm_policy_params", "comm_budget_bytes", "comm_max_wait_steps",
        "comm_urgent_wait_steps", "comm_sends_per_tick", "comm_transcript", "comm_range",
        "gossip_interval", "comm_delivery_probability", "comm_delay_ticks",
        "comm_subscribers", "sticky_commitments", "release_redundant_members",
        "suppress_recommand_own_task", "check_release_source_stage",
        "task_membership_consistency", "coordinator_policy",
    }
    return {
        field.name: field.default
        for field in dataclasses.fields(RunConfiguration)
        if field.name in wanted
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(common.ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


def rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def curve(directory: Path, seed: int) -> dict[str, Any]:
    import torch

    payload = torch.load(directory / f"pilot_seed{seed}.pt", map_location="cpu", weights_only=False)
    state = payload["state"]
    return {
        "seed": seed,
        "environment_steps": state["environment_steps"],
        "episodes": state["episodes"],
        "updates": len(state["updates"]),
        "budget_bytes": state["budget_bytes"],
        "learning_curve": state["evaluations"],
        "final_update": state["updates"][-1] if state["updates"] else None,
        "batch_on_time_first_last": [
            state["updates"][0]["batch_on_time"],
            state["updates"][-1]["batch_on_time"],
        ] if state["updates"] else None,
        "parameters": sum(int(t.numel()) for t in train_pilot.Controller(0).parameters),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="*", default=["eval_main.json"])
    args = parser.parse_args(argv)

    calibration = json.loads((common.HERE / "COMM_BUDGET_CALIBRATION.json").read_text(encoding="utf-8"))
    locked = json.loads((common.HERE / "COMM_BUDGET_LOCKED.json").read_text(encoding="utf-8"))
    preflight = json.loads((common.HERE / "PILOT_PREFLIGHT.json").read_text(encoding="utf-8"))
    eval_files = {
        Path(path).name: json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.evaluations
        if Path(path).exists()
    }
    streams = sorted(common.STREAMS.glob("s*.json"))
    production_dirty = git("status", "--porcelain", "integrated_sar", "task_allocation")
    manifest = {
        "round": "learned_communication_budget_pilot_20260919",
        "date": "2026-09-19",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_at_report": git("rev-parse", "HEAD"),
        "production_head": git("log", "-1", "--format=%H", "--", "integrated_sar", "task_allocation"),
        "inherited_from": {
            "branch": "feat/communication-control-baselines",
            "head": "0794de9d2adb9438b753397cea8304616ddd0102",
            "corrected_environment": "task_membership_consistency is switched on for every episode of this round, asserted per run",
            "previous_freeze": common.FROZEN["code_commit"],
            "production_paths_dirty_at_report": bool(production_dirty),
        },
        "checkpoints": {
            name: {
                "path": str(Path(path).relative_to(common.ROOT)).replace("\\", "/"),
                "sha256": digest,
            }
            for (name, path), digest in zip(common.CHECKPOINTS.items(), common.CHECKPOINT_SHA256.values())
        },
        "scenario_frozen": {
            key: common.FROZEN[key]
            for key in (
                "steps",
                "dt",
                "grid_size",
                "obstacle_density",
                "detection_radius",
                "search_agents",
                "num_goals",
                "deadline_tightness",
                "residual_scale",
                "coordination_interval",
                "allocation_interval",
                "command_ttl",
            )
        },
        "allocation_arm_unchanged": preflight["expected_allocation"],
        "preflight_mismatches": preflight["mismatches"],
        "runtime_identity": preflight["runtime"],
        "seeds": {
            "train_streams_recorded_this_round": list(common.TRAIN_SEEDS),
            "validation_streams_recorded_this_round": list(common.VALIDATION_SEEDS),
            # The recorded streams are not committed, so their digests are folded here: one
            # digest over the ordered list identifies the exact bytes this round trained on.
            "stream_sha256_fold": hashlib.sha256(
                "".join(common.sha256(stream) for stream in streams).encode("ascii")
            ).hexdigest(),
            "stream_sha256": {stream.stem: common.sha256(stream) for stream in streams},
            "not_read_this_round": "500000..500399 (previous round's registered independent test set), 600000..600059",
            "previously_used_for_protocol_fitting": "30000..30011",
        },
        "budget": {
            "reference": locked["reference_mean_airtime_bytes"],
            "locked_credits_bytes": locked["budgets_bytes"],
            "selection_rule": locked["tier_selection_rule"],
            "selected_tier": calibration["tier_selection"]["selected_tier"],
            "selected_credit_bytes": locked["budgets_bytes"][
                calibration["tier_selection"]["selected_tier"]
            ],
            "charging": "payload bytes x subscribed robots per transmission attempt, lost packets included",
            "silence_bound_steps": {
                "normal": pilot_env.MAX_WAIT_STEPS,
                "urgent": pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
                "urgency_window": pilot_env.URGENCY_STEPS,
                "applied_to": "every arm, in the transport, not inside a policy",
            },
            "on_exhaustion": (
                "the emission does not happen, the change stays queued and keeps ageing, and the "
                "refusal is counted; no arm, urgent or not, can spend credit it does not have"
            ),
        },
        "calibration": calibration["arms"],
        "model": {
            "feature_layout": FEATURE_LAYOUT,
            "feature_names": list(train_pilot.FEATURE_NAMES),
            "actions": ["send", "hold"],
            "hidden": train_pilot.HIDDEN,
            "hyperparameters": {
                "gamma": train_pilot.GAMMA,
                "gae_lambda": train_pilot.GAE_LAMBDA,
                "clip": train_pilot.CLIP,
                "value_coefficient": train_pilot.VALUE_COEFFICIENT,
                "entropy_bonus": train_pilot.ENTROPY_BONUS,
                "learning_rate": train_pilot.LEARNING_RATE,
                "max_grad_norm": train_pilot.MAX_GRAD_NORM,
                "epochs_per_batch": train_pilot.EPOCHS_PER_BATCH,
                "minibatch_steps": train_pilot.MINIBATCH_STEPS,
                "episodes_per_update": train_pilot.EPISODES_PER_UPDATE,
                "exploration_epsilon": train_pilot.EXPLORATION_EPSILON,
                "pilot_environment_steps_cap": train_pilot.PILOT_ENVIRONMENT_STEPS,
                "evaluate_every_steps": train_pilot.EVALUATE_EVERY_STEPS,
            },
            "reward": "one credit per target completed on time, booked on the step it happened; no message penalty term -- the budget is an executor constraint",
        },
        "training": [curve(common.HERE / "train", seed) for seed in (1, 2, 3)],
        "throughput": {
            "episode_wall_clock_seconds_mean": round(
                float(np.mean([row.get("seconds", 0.0) for row in rows(ROWS)])), 3
            ),
            "episodes_measured": len(rows(ROWS)),
            "environment_steps_per_second_estimate": round(
                common.STREAM_STEPS
                / max(0.001, float(np.mean([row.get("seconds", 1.0) for row in rows(ROWS)]))),
                1,
            ),
        },
        "evaluations": {
            name: {key: value for key, value in payload.items() if key != "per_seed"}
            for name, payload in eval_files.items()
        },
        "per_seed_evaluations": {
            name: payload.get("rows", payload.get("per_seed"))
            for name, payload in eval_files.items()
        },
        "gates": {
            "bootstrap": {
                "replications": evaluate_pilot.BOOTSTRAP_REPLICATIONS,
                "rng_seed": evaluate_pilot.BOOTSTRAP_SEED,
                "scheme": "seed-level paired percentile bootstrap",
            },
            "calibration_technical_failures": calibration["technical_failures"],
            # Read off the built configuration class, not restated by hand: a changed default
            # fails this round's own claim instead of slipping past it.
            "configuration_defaults_as_built": _defaults(),
            "configuration_defaults_expected": {
                "comm_policy": "immediate",
                "comm_policy_params": "",
                "comm_budget_bytes": None,
                "comm_max_wait_steps": None,
                "comm_urgent_wait_steps": None,
                "comm_sends_per_tick": None,
                "comm_transcript": None,
                "comm_range": 0.3,
                "gossip_interval": 2,
                "comm_delivery_probability": 1.0,
                "comm_delay_ticks": 0,
                "comm_subscribers": "all",
                "sticky_commitments": False,
                "release_redundant_members": False,
                "suppress_recommand_own_task": False,
                "check_release_source_stage": False,
                "task_membership_consistency": False,
                "coordinator_policy": "heuristic",
            },
        },
        "artefacts": {
            "streams_recorded": len(streams),
            "stream_bytes": sum(stream.stat().st_size for stream in streams),
            "calibration_rows": len(rows(ROWS)),
            "documents": sorted(path.name for path in common.HERE.glob("*.md")),
            "not_committed": [
                "recorded streams (runs/comm_budget_pilot_20260919/streams)",
                "per-episode lifecycle logs, decision traces and transport transcripts",
                "torch checkpoints and optimiser state (runs/comm_budget_pilot_20260919/train)",
            ],
        },
        "hashes": {
            name: common.sha256(common.HERE / name)
            for name in (
                "COMM_BUDGET_CALIBRATION.json",
                "COMM_BUDGET_LOCKED.json",
                "PILOT_PREFLIGHT.json",
                "PILOT_SMOKE.json",
            )
            if (common.HERE / name).exists()
        },
    }
    expected = manifest["gates"]["configuration_defaults_expected"]
    found = manifest["gates"]["configuration_defaults_as_built"]
    manifest["gates"]["configuration_defaults_match"] = expected == found
    manifest["gates"]["configuration_defaults_differences"] = {
        key: [expected.get(key), found.get(key)]
        for key in set(expected) | set(found)
        if expected.get(key) != found.get(key)
    }
    path = common.write_json(common.HERE / "COMM_ROUND_MANIFEST.json", manifest)
    print(
        f"wrote {path.name}: episodes={len(rows(ROWS))} streams={len(streams)} "
        f"selected_tier={manifest['budget']['selected_tier']} "
        f"credit={manifest['budget']['selected_credit_bytes']}B"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
