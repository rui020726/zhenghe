"""Fixtures for the probability-and-budget round: two locked tiers, one frozen system.

Nothing here re-cuts a seed block or re-calibrates a budget. The 48 training streams and 24
validation streams keep the purpose round 2 gave them, the tier values come from the locked
calibration file, and the final test block stays registered-but-unread. What this round adds is
the *second* tier -- 25 % -- because round 3 established that at 75 % the credit never binds for
any arm on the frontier, and an unconstrained budget cannot answer a question about budgeting.

The arms list is the protocol's §4 set: the rules a learned controller has to beat, each run
through the same transport at each tier. Two of them are labelled for what they are:
``event_unbudgeted`` is a ceiling, and ``send_once_no_bound`` is a diagnostic with the shared
silence bound removed, so it is never quoted as a same-protocol comparison.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STABILITY_ROUND = ROOT / "runs" / "comm_stability_20260919"
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"

for _directory in (str(ROOT), str(PILOT_ROUND), str(STABILITY_ROUND), str(HERE)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

stability = importlib.import_module("stability_common")
pilot = stability.pilot

TRAIN_SEEDS = pilot.TRAIN_SEEDS
VALIDATION_SEEDS = pilot.VALIDATION_SEEDS
STREAM_STEPS = pilot.STREAM_STEPS
configuration_for = pilot.configuration_for
write_json = pilot.write_json
sha256 = pilot.sha256
metrics = pilot.metrics
locked_budget = stability.locked_budget

#: The two credits this round compares, both from the locked calibration file rather than typed
#: in, and both checked against what the calibration actually published.
BUDGET_TIERS: tuple[str, ...] = ("75pct", "25pct")
BUDGETS_BYTES: dict[str, int] = {tier: locked_budget(tier) for tier in BUDGET_TIERS}

TRAIN_OUT = HERE / "train"

#: §4's rule set. ``bound: none`` marks the arm that steps outside the shared silence protocol,
#: and ``ceiling`` the one that runs with no credit at all; neither is a competitor.
RULE_ARMS: tuple[dict[str, Any], ...] = (
    {"arm": "immediate", "policy": "immediate", "params": ""},
    {"arm": "hold_to_bound", "policy": "hold_to_bound", "params": ""},
    {"arm": "periodic_5", "policy": "periodic", "params": "interval=5"},
    {"arm": "event", "policy": "event", "params": ""},
    {"arm": "event_priority", "policy": "event_priority", "params": ""},
    {"arm": "send_once_no_bound", "policy": "immediate", "params": "", "bound": "none"},
    {"arm": "event_unbudgeted", "policy": "event", "params": "", "ceiling": True},
)

#: The learned arms, once training has produced checkpoints.
LEARNED_ARMS: tuple[str, ...] = tuple(
    f"tight_{tier}_seed{seed}" for tier in BUDGET_TIERS for seed in (1, 2, 3)
)


def json_path(name: str) -> Path:
    return HERE / name


def read_json(name: str) -> Any:
    return json.loads(json_path(name).read_text(encoding="utf-8"))


def tensors(payload: Any) -> list[Any]:
    """Every tensor in a possibly nested state dict, in a fixed order."""
    found: list[Any] = []
    for value in payload.values() if isinstance(payload, dict) else payload:
        if isinstance(value, dict):
            found.extend(tensors(value))
        else:
            found.append(value)
    return found


def weight_fingerprint(controller: Any) -> str:
    digest = hashlib.sha256()
    for tensor in tensors(controller.state_dict()):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def run_rule(spec: dict[str, Any], seed: int, *, budget_bytes: int | None, directory: Path) -> dict[str, Any]:
    """One rule arm on one stream, with the transport ledger and the transcript attached."""
    import pilot_env
    import attribution_metrics as M
    from integrated_sar.comm_policy import parse_parameters

    unbounded = spec.get("bound") == "none"
    row = pilot_env.run_episode(
        seed,
        directory=directory,
        label=spec["arm"],
        policy=spec["policy"],
        policy_params=parse_parameters(spec["params"]) or None,
        budget_bytes=None if spec.get("ceiling") else budget_bytes,
        information="isolated",
        delivery_probability=1.0,
        max_wait_steps=None if unbounded else pilot_env.MAX_WAIT_STEPS,
        urgent_wait_steps=None if unbounded else pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
    )
    records = metrics.load_transcript(row["artefacts"]["transcript"])
    rollup = M._per_reason_rollup(records)
    return {
        "arm": spec["arm"],
        "seed": int(seed),
        "budget_bytes": row["budget_bytes"],
        "row": row,
        "records": records,
        "rollup": rollup,
        "closure": M._closure(row["ledger"], rollup, spent_bytes=row["spent_bytes"]),
    }


def refusal_detail(records: list[dict[str, Any]], *, budget_bytes: int | None) -> dict[str, Any]:
    """What the credit actually did in one episode: when it first said no, and to whom.

    A tier that is below an arm's *average* cost still may not bind in a given episode, so this
    is read per episode and the distribution is what gets reported.
    """
    sends_before: dict[str, int] = {}
    refused: list[dict[str, Any]] = []
    for record in records:
        kind = record["kind"]
        if kind == "send":
            sends_before[str(record["target"])] = sends_before.get(str(record["target"]), 0) + 1
        elif kind == "refused":
            refused.append(
                {
                    "step": int(record["step"]),
                    "target": str(record["target"]),
                    "reason": str(record["reason"]),
                    "send_reason": str(record.get("send_reason", "policy")),
                    "cost_bytes": int(record["cost_bytes"]),
                    "remaining_bytes": record.get("remaining_bytes"),
                    "kind_of_message": "first_report" if sends_before.get(str(record["target"]), 0) == 0 else "refresh",
                }
            )
    first = min(refused, key=lambda item: item["step"]) if refused else None
    remaining = [
        int(record["remaining_bytes"])
        for record in refused
        if record.get("remaining_bytes") is not None
    ]
    return {
        "refusals": len(refused),
        "first_refusal_step": None if first is None else first["step"],
        "first_refusal_reason": None if first is None else first["reason"],
        "first_refusal_send_reason": None if first is None else first["send_reason"],
        "first_refusal_message_type": None if first is None else first["kind_of_message"],
        "refusal_classes": {
            reason: sum(1 for item in refused if item["reason"] == reason)
            for reason in sorted({item["reason"] for item in refused})
        },
        "refused_as": {
            reason: sum(1 for item in refused if item["send_reason"] == reason)
            for reason in sorted({item["send_reason"] for item in refused})
        },
        "refused_message_types": {
            kind: sum(1 for item in refused if item["kind_of_message"] == kind)
            for kind in sorted({item["kind_of_message"] for item in refused})
        },
        "bytes_refused": sum(int(item["cost_bytes"]) for item in refused),
        "min_remaining_bytes": min(remaining) if remaining else None,
        "credit_used_fraction": (
            round(1.0 - min(remaining) / budget_bytes, 4)
            if remaining and budget_bytes
            else None
        ),
    }


def episode_row(result: dict[str, Any]) -> dict[str, Any]:
    """One flat row per episode, every quantity named with its unit."""
    row, rollup, ledger = result["row"], result["rollup"], result["row"]["ledger"]
    emissions = int(ledger["emissions"])
    forced = int(ledger["emissions_forced_maxwait"])
    candidates = int(ledger["generated_items"])
    reasons = rollup["by_reason"]
    total_bytes = sum(int(bucket["bytes"]) for bucket in reasons.values())
    return {
        "arm": result["arm"],
        "seed": int(result["seed"]),
        "budget_bytes": result["budget_bytes"],
        "on_time_targets_per_episode": int(row["task"]["on_time"]),
        "detected_targets_this_episode": int(row["task"]["targets"]),
        "on_time_rate_within_detected_this_episode": round(
            int(row["task"]["on_time"]) / int(row["task"]["targets"]), 5
        )
        if int(row["task"]["targets"])
        else None,
        "expired_targets_per_episode": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": row["budget_used_fraction"],
        "emissions": emissions,
        "emissions_forced": forced,
        "optional_emissions_share": round((emissions - forced) / emissions, 4) if emissions else None,
        "autonomous_share_of_bytes": rollup["autonomous_share_of_bytes"],
        "autonomous_bytes": int(reasons["policy"]["bytes"]),
        "forced_bytes": int(reasons["forced_maxwait"]["bytes"]),
        "generated_items": candidates,
        "coalesced_changes": int(ledger["coalesced_changes"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "unknown_to_everyone_steps": float(row["belief"].get("unknown_to_everyone_steps") or 0.0),
        "generation_balance": int(result["closure"]["generated_balance"]["unspoken_or_uncounted_at_end"]),
        "residuals": {key: value for key, value in result["closure"]["residuals"].items() if value},
        "refusals": refusal_detail(result["records"], budget_bytes=result["budget_bytes"]),
        "seconds": float(row["seconds"]),
        "config_hash": row["config_hash"],
    }


def evaluate_controller(
    controller: Any,
    *,
    budget_bytes: int,
    episodes: int,
    directory: Path,
    label: str,
    information: str = "isolated",
    delivery_probability: float = 1.0,
) -> dict[str, Any]:
    """The frozen validation read: greedy deployment policy, the same streams, the same credit.

    Measured through the round-3 attribution row builder so a point on a learning curve and a
    row of the comparison table are the same kind of number.
    """
    import attribution_metrics as M
    import onpolicy
    import pilot_env
    import train_pilot

    seeds = list(VALIDATION_SEEDS)[:episodes]
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, float]] = []
    for seed in seeds:
        collector = onpolicy.OnPolicyCollector(controller, forced=train_pilot.forced, greedy=True)
        row = pilot_env.run_episode(
            seed,
            directory=directory,
            label=label,
            policy="learned",
            budget_bytes=budget_bytes,
            controller=collector.policy(),
            information=information,
            delivery_probability=delivery_probability,
        )
        records = metrics.load_transcript(row["artefacts"]["transcript"])
        rollup = M._per_reason_rollup(records)
        rows.append(
            {
                "on_time_targets_per_episode": int(row["task"]["on_time"]),
                "detected": int(row["task"]["targets"]),
                "rate": (
                    int(row["task"]["on_time"]) / int(row["task"]["targets"])
                    if int(row["task"]["targets"])
                    else None
                ),
                "airtime_bytes": int(row["spent_bytes"]),
                "emissions": int(row["ledger"]["emissions"]),
                "emissions_by_policy": int(row["ledger"]["emissions_policy"]),
                "emissions_forced": int(row["ledger"]["emissions_forced_maxwait"]),
                "refusals": sum(int(bucket["refused"]) for bucket in rollup["by_reason"].values()),
                "autonomous_share_of_bytes": float(rollup["autonomous_share_of_bytes"] or 0.0),
                "applied": int(row["ledger"]["deliveries_applied"]),
                "belief_age": float(row["belief"].get("belief_age_steps_mean") or 0.0),
                "seconds": float(row["seconds"]),
                "generation_balance": int(
                    M._closure(row["ledger"], rollup, spent_bytes=row["spent_bytes"])["generated_balance"][
                        "unspoken_or_uncounted_at_end"
                    ]
                ),
            }
        )
        timings.append(collector.timing())

    def mean(key: str) -> float:
        return round(float(np.mean([float(row[key] or 0.0) for row in rows])), 4)

    return {
        "episodes": len(rows),
        "information": information,
        "delivery_probability": delivery_probability,
        "budget_bytes": budget_bytes,
        "mean_on_time": mean("on_time_targets_per_episode"),
        "mean_on_time_rate_per_episode": mean("rate"),
        "mean_airtime_bytes": mean("airtime_bytes"),
        "mean_emissions": mean("emissions"),
        "mean_emissions_by_policy": mean("emissions_by_policy"),
        "mean_emissions_forced": mean("emissions_forced"),
        "mean_refusals": mean("refusals"),
        "autonomous_share_of_bytes": mean("autonomous_share_of_bytes"),
        "mean_applied": mean("applied"),
        "mean_belief_age": mean("belief_age"),
        "generation_balance_total": int(sum(int(row["generation_balance"]) for row in rows)),
        "inference_milliseconds_per_decision": round(
            1000.0 * sum(item["decide_milliseconds_per_call"] * item["decisions"] for item in timings)
            / max(1, sum(item["decisions"] for item in timings)),
            4,
        ),
        "drawn_send_rate": round(
            float(np.mean([item["voluntary_send_rate"] for item in timings])), 4
        ),
        "per_episode": rows,
    }


def checkpoint_path(tier: str, training_seed: int, at_steps: int | None = None) -> Path:
    stem = f"tight_{tier}_seed{training_seed}"
    directory = TRAIN_OUT / f"{tier}_seed{training_seed}"
    return directory / (f"{stem}.pt" if at_steps is None else f"{stem}_at{at_steps}.pt")


def paired_bootstrap(rows: list[dict[str, Any]], *, baseline: str, key: str) -> dict[str, Any]:
    """Round 2's seed-level paired bootstrap, reused rather than re-derived."""
    import evaluate_pilot

    return evaluate_pilot.paired(rows, baseline=baseline, extract=lambda row, key=key: float(row[key]))


__all__ = [
    "HERE",
    "ROOT",
    "TRAIN_SEEDS",
    "VALIDATION_SEEDS",
    "STREAM_STEPS",
    "BUDGET_TIERS",
    "BUDGETS_BYTES",
    "RULE_ARMS",
    "LEARNED_ARMS",
    "TRAIN_OUT",
    "locked_budget",
    "weight_fingerprint",
    "run_rule",
    "episode_row",
    "evaluate_controller",
    "checkpoint_path",
    "paired_bootstrap",
    "write_json",
    "json_path",
    "read_json",
    "sha256",
    "metrics",
]
