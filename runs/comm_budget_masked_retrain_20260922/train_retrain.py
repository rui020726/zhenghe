"""Retrain the budget-masked sender from initial weights: 3 seeds x 400,000 environment steps.

This is the run the mask fix existed to make possible. The sampler, the tape and the PPO loop are
imported from ``runs/comm_budget_action_mask_fix_20260922`` and ``runs/comm_tight_budget_20260919``
unchanged -- there is no second copy of the optimiser here, because two copies are how a later
round ends up comparing two trainers instead of finding a bug. What this file adds is the parts a
from-scratch run needs and a smoke test did not:

* every arm starts at ``Controller(training_seed)`` and nothing else; a checkpoint may only be
  resumed by a trainer that repeats this run's freeze SHA, configuration hash, trainer tag, mask
  schema, tape schema, timing mode, budget and step cap, and refuses one whose RNG is missing;
* the sample accounting §5 demands, in absolute counts: actionable candidates, actor terms, the
  per-episode distribution and the number of episodes that never got a decision, message-type and
  within-episode timing structure, credit-exhausted steps, and the optimiser diagnostics;
* a greedy replay of each batch's own tapes, so "what this policy would deploy" is read from the
  same weights without spending an extra episode on it;
* one ceiling for the whole round: every episode started -- probe, training or evaluation -- is
  booked in the shared launch ledger, and a technical failure spends the ceiling too.

    python train_retrain.py --mode identity
    python train_retrain.py --mode probe --episodes 4 --threads 1
    python train_retrain.py --mode resume-check --threads 1
    python train_retrain.py --mode train --seed 1 --steps 400000 --threads 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import retrain_common as common

import protocol_env  # noqa: E402  (imported for the rollout seam, and to prove it resolves)
import train_pilot  # noqa: E402
import train_protocol  # noqa: E402
import train_budget_masked as masked  # noqa: E402
from budget_masked import ACTION_MASK_SCHEMA, TAPE_SCHEMA, TRAINER_TAG  # noqa: E402
from integrated_sar.comm_budget import CommBudget  # noqa: E402

STEPS_PER_EPISODE = common.STEPS_PER_EPISODE
EPISODES_PER_UPDATE = common.EPISODES_PER_UPDATE
TRAIN_SEEDS = common.TRAIN_SEEDS
BUDGET_BYTES = common.BUDGET_BYTES
CHECKPOINT_STEPS = common.CHECKPOINT_STEPS
#: The likelihood agreement this round is held to. ``update`` already refuses 1e-4; the brief says
#: 1e-6, so the stricter line is enforced here rather than negotiated later.
LOG_PROB_TOLERANCE = 1e-6
SCHEMA = "comm-budget-masked-retrain/1"

__all__ = [
    "configuration_hash",
    "initial_controller",
    "rollout",
    "greedy_replay",
    "emission_types",
    "update_record",
    "save",
    "load_for_resume",
    "load_for_eval",
    "train",
    "probe",
    "resume_check",
    "identity_report",
]


# ------------------------------------------------------------------ identity
def configuration_hash() -> str:
    """A hash of the setting, so a checkpoint states the experiment it belongs to.

    Hashed rather than dumped: the setting is a dozen fields and the point of the check is to catch
    a run that quietly changed one of them, which an equality test on a dict of dicts would do just
    as well but far more verbosely in a log line.
    """
    payload = {
        "setting": common.setting(),
        "steps_per_run": common.STEPS_PER_RUN,
        "checkpoints": list(CHECKPOINT_STEPS),
        "episodes_per_update": EPISODES_PER_UPDATE,
        "budget_bytes": BUDGET_BYTES,
        "timing_mode": common.TIMING_MODE,
        "information": common.INFORMATION,
        "delivery_probability": common.DELIVERY_PROBABILITY,
        "ack": common.ACK,
        "trainer_tag": TRAINER_TAG,
        "action_mask_schema": ACTION_MASK_SCHEMA,
        "tape_schema": TAPE_SCHEMA,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def initial_controller(training_seed: int) -> Any:
    """``Controller(seed)`` and nothing else -- no state dict, no warm start, no optimizer.

    The old 400k checkpoints are products of an action space in which an unaffordable SEND was a
    free action; continuing their optimiser state would carry that shaping into a run that claims
    to have fixed it. This function is the only place weights come from, and the resume guard is
    what keeps it that way.
    """
    return train_pilot.Controller(int(training_seed))


# ------------------------------------------------------------------ one rollout
def rollout(
    controller: Any, stream_seed: int, *, training_seed: int, episode_index: int, directory: Path
) -> dict[str, Any]:
    """One policy-owned episode under the budget mask, with this round's artefact names.

    Only the label and the directory differ from the archived runner: the transcript writer names
    its files after the label and the stream, so two rounds sharing a label overwrite each other's
    evidence. The collector comes from ``train_budget_masked``, unmodified.
    """
    collector = masked.collector_for(controller)
    episode = protocol_env.rollout_episode(
        controller,
        stream_seed,
        collector=collector,
        budget_bytes=BUDGET_BYTES,
        directory=directory,
        label=f"maskretrain_s{training_seed}",
        timing_mode=train_protocol.TIMING_MODE,
    )
    episode["controller"] = controller
    episode["collector"] = collector
    episode["timing"] = collector.timing()
    episode["episode_index"] = int(episode_index)
    episode["stream_seed"] = int(stream_seed)
    return episode


def emission_types(row: dict[str, Any]) -> dict[str, int]:
    """First / refresh / retry, straight off the transport's own counters.

    The relay counts ``first_emissions`` and ``refresh_emissions`` (everything that was not a
    first report) and ``redundant_emissions`` (a letter whose view was already said), so the three
    classes close as ``first + refresh == emissions`` with retry a subset of refresh. The identity
    is asserted by the caller rather than assumed: an arm whose message mix does not add up is a
    measurement defect, not a finding.
    """
    ledger = row["ledger"]
    emissions = int(ledger["emissions"])
    first = int(ledger["first_emissions"])
    refresh = int(ledger["refresh_emissions"])
    redundant = int(ledger["redundant_emissions"])
    return {
        "emissions": emissions,
        "first": first,
        "refresh": refresh,
        "update": refresh - redundant,
        "retry_same_view": redundant,
    }


def tape_hash(tapes: list[dict[str, Any]]) -> str:
    """A digest of what the batch decided, so the resume test compares behaviour, not vibes."""
    summary = [
        [
            int(record["step"]),
            list(record["targets"]),
            {key: int(value) for key, value in sorted(record["actions"].items())},
            sorted(record["actionable"]),
            sorted(record["masked"]),
            round(float(record["log_prob"]), 9),
            round(float(record["value"]), 9),
        ]
        for record in tapes
    ]
    text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def per_episode_row(episode: dict[str, Any]) -> dict[str, Any]:
    """Everything §5 asks for at episode granularity, from the collector and the row."""
    collector = episode["collector"]
    row = episode["row"]
    timing = collector.timing()
    tapes = collector.records
    candidates = int(collector.candidates)
    actionable = int(timing["free_candidates"])
    send_steps = sorted({int(step) for step, _targets in collector.send_steps})
    steps_with_candidates = [int(record["step"]) for record in tapes if record["targets"]]
    masked_steps = [int(record["step"]) for record in tapes if record["masked"]]
    zero_actionable_with_candidates = sum(
        1 for record in tapes if record["targets"] and not record["actionable"]
    )
    types = emission_types(row)
    return {
        "stream_seed": int(episode["stream_seed"]),
        "episode_index": int(episode["episode_index"]),
        "decisions": int(collector.decisions),
        "candidates": candidates,
        "initial_executable_candidates": sum(
            1
            for record in tapes
            for flag in record["send_executable_initial"].values()
            if flag
        ),
        "actionable_candidates": actionable,
        "actor_terms": actionable,
        "masked_by_credit": int(timing["masked_by_credit"]),
        "masked_by_capacity": int(timing["masked_by_capacity"]),
        "steps_with_candidates": len(steps_with_candidates),
        "steps_with_zero_actionable": zero_actionable_with_candidates,
        "first_credit_masked_step": masked_steps[0] if masked_steps else None,
        "send_steps": send_steps,
        "send_steps_count": len(send_steps),
        "sampled_send": sum(
            1
            for record in tapes
            for target in record["actionable"]
            if record["actions"][target] == 1
        ),
        "requested": sum(1 for record in tapes for on in record["requested"].values() if on),
        "executed_emissions": int(row["ledger"]["emissions"]),
        "executed_policy_emissions": int(row["ledger"]["emissions_policy"]),
        "forced_emissions": int(row["ledger"]["emissions_forced_maxwait"]),
        "policy_budget_refusals": int(row["ledger"]["budget_refused_sends"]),
        "policy_capacity_refusals": int(row["ledger"]["capacity_refused_sends"]),
        **{f"message_{key}": value for key, value in types.items()},
        "credit_spent_bytes": int(row["spent_bytes"]),
        "on_time": int(row["task"]["on_time"]),
        "detected": int(row["task"]["targets"]),
        "expired": int(row["task"]["expired"]),
        "steps_run": train_protocol.actual_steps(row["artefacts"]["lifecycle_log"]),
        "seconds": float(row["seconds"]),
        "config_hash": row["config_hash"],
        "tape_hash": tape_hash(tapes),
    }


# ------------------------------------------------------------------ greedy read of the same weights
def greedy_replay(controller: Any, tapes: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-decide the batch's own observations greedily, against the same shadow credit.

    Why this exists rather than a greedy episode: §5 wants the greedy SEND rate and greedy
    exhaustion step beside the stochastic ones, and an extra evaluation episode per update would
    spend the launch ceiling to learn what the recorded candidates already say. The replay walks
    ``targets`` in the relay's order, gates each candidate with a fresh ``CommBudget`` seeded at the
    credit the observation reported, and charges only a chosen SEND -- the same arithmetic the
    collector performs, so a divergence between the two is a bug in one of them and shows up as a
    disagreement about the actionable set.
    """
    actionable = 0
    sends = 0
    agreement = 0
    masked_by_credit = 0
    entropy_terms = 0.0
    steps = 0
    zero_actionable_steps = 0
    first_masked_step: int | None = None
    for record in tapes:
        targets = record["targets"]
        if not targets:
            continue
        charges = record["attempt_charge_bytes"]
        gate = _gate_from_tape(record)
        with torch.no_grad():
            logits, _values = controller.forward(
                torch.as_tensor(np.asarray(record["features"], dtype=np.float32))
            )
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
        step_had_choice = False
        for index, target in enumerate(targets):
            if target in record["forced"]:
                continue
            charge = charges.get(target)
            if charge is None:
                continue
            if not gate.can_send(int(charge)):
                masked_by_credit += 1
                if first_masked_step is None:
                    first_masked_step = int(record["step"])
                continue
            actionable += 1
            step_had_choice = True
            greedy_send = int(
                1 if float(probabilities[index, 1].item()) >= float(probabilities[index, 0].item())
                else 0
            )
            sends += greedy_send
            agreement += int(greedy_send == int(record["actions"][target]))
            per_candidate = -(probabilities[index] * log_probabilities[index]).sum()
            entropy_terms += float(per_candidate.item())
            if greedy_send:
                gate.charge(int(charge))
        steps += 1
        if targets and not step_had_choice:
            zero_actionable_steps += 1
    return {
        "greedy_actionable_candidates": actionable,
        "greedy_send": sends,
        "greedy_send_rate": round(sends / max(1, actionable), 4),
        "greedy_entropy_per_actionable_candidate": round(entropy_terms / max(1, actionable), 5),
        "greedy_masked_by_credit": masked_by_credit,
        "greedy_steps_with_zero_actionable": zero_actionable_steps,
        "greedy_first_credit_masked_step": first_masked_step,
        "greedy_matches_sampled_share": round(agreement / max(1, actionable), 4),
        "tapes_replayed": steps,
    }


def _gate_from_tape(record: dict[str, Any]) -> CommBudget:
    """The shadow credit as the collector saw it at the start of this step.

    The tape keeps the remaining bytes and the remaining capacity, so the amount already spent is
    the frozen episode credit minus the remainder -- the same arithmetic ``budget_masked.shadow_gate``
    does from the live observation. A remainder outside ``[0, credit]`` is not a rounding question,
    it means the tape and the transport disagree about what was spent, so the replay refuses.
    """
    remaining = record.get("shadow_budget_before")
    if remaining is None:
        raise RuntimeError("a masked tape with no shadow budget cannot be replayed")
    gate = CommBudget(
        bytes_per_episode=BUDGET_BYTES,
        sends_per_step=record.get("shadow_capacity_before"),
    )
    spent = int(round(BUDGET_BYTES - float(remaining)))
    if not 0 <= spent <= BUDGET_BYTES:
        raise RuntimeError(
            f"tape reports {remaining!r} bytes remaining of a {BUDGET_BYTES} credit, which reads "
            f"as {spent} spent"
        )
    gate.spent_bytes = spent
    return gate


# ------------------------------------------------------------------ one optimiser pass
def update_record(
    controller: Any,
    optimizer: torch.optim.Optimizer,
    episodes: list[dict[str, Any]],
    *,
    at_steps: int,
    batch_seeds: list[int],
    elapsed: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect -> align -> optimise, with every §5 quantity read off the tapes and the rows."""
    details = [per_episode_row(episode) for episode in episodes]
    tapes = [record for episode in episodes for record in episode["collector"].records]
    greedy = greedy_replay(controller, tapes)
    stochastic_entropy = sum(
        _tape_candidate_entropy(controller, record) for record in tapes
    )
    stochastic_terms = sum(len(record["actionable"]) for record in tapes)
    report = masked.update(controller, optimizer, episodes)
    gap = float(report["likelihood_alignment"]["max_abs_gap"])
    if gap > LOG_PROB_TOLERANCE:
        raise RuntimeError(
            f"log-prob gap {gap} exceeds the {LOG_PROB_TOLERANCE:g} this round froze; the sampler "
            "and the likelihood no longer describe one distribution"
        )
    for name, value in (
        ("NaN or inf in the weights", _weights_finite(controller)),
        ("episode length", all(item["steps_run"] == STEPS_PER_EPISODE for item in details)),
        ("forced emission under policy-owned timing", not any(
            item["forced_emissions"] for item in details
        )),
        ("policy send refused by the executor", not any(
            item["policy_budget_refusals"] or item["policy_capacity_refusals"] for item in details
        )),
        ("requested sends honoured", all(
            item["requested"] == item["executed_policy_emissions"] for item in details
        )),
        ("emission classes close", all(
            item["message_first"] + item["message_refresh"] == item["message_emissions"]
            and item["message_retry_same_view"] <= item["message_refresh"]
            for item in details
        )),
        ("tape invariants", masked.tape_invariants(tapes)["passed"]),
    ):
        if not value:
            raise RuntimeError(f"correctness gate failed during training: {name} ({at_steps} steps)")

    def total(key: str) -> int:
        return sum(int(item[key]) for item in details)

    types = {
        f"message_{suffix}": sum(int(item[f"message_{suffix}"]) for item in details)
        for suffix in ("first", "update", "retry_same_view")
    }
    report.update(
        {
            "at_steps": at_steps,
            "batch_seeds": batch_seeds,
            "episodes": len(details),
            "candidates": total("candidates"),
            "initial_executable_candidates": total("initial_executable_candidates"),
            "actionable_candidates": total("actionable_candidates"),
            "actor_terms": total("actor_terms"),
            "masked_by_credit": total("masked_by_credit"),
            "masked_by_capacity": total("masked_by_capacity"),
            "steps_with_candidates": total("steps_with_candidates"),
            "steps_with_zero_actionable": total("steps_with_zero_actionable"),
            "episodes_without_actionable": sum(
                1 for item in details if item["actionable_candidates"] == 0
            ),
            "episodes_without_decisions": sum(
                1 for item in details if item["decisions"] == 0
            ),
            "sampled_send": total("sampled_send"),
            "requested_by_policy": total("requested"),
            "executed_policy_emissions": total("executed_policy_emissions"),
            "policy_budget_refusals": total("policy_budget_refusals"),
            "policy_capacity_refusals": total("policy_capacity_refusals"),
            "batch_on_time": total("on_time"),
            "batch_targets": total("detected"),
            "batch_expired": total("expired"),
            "batch_bytes": total("credit_spent_bytes"),
            "batch_budget_use": round(
                total("credit_spent_bytes") / (BUDGET_BYTES * len(details)), 4
            ),
            "stochastic_send_rate": round(
                total("sampled_send") / max(1, total("actionable_candidates")), 4
            ),
            "stochastic_entropy_per_actionable_candidate": round(
                stochastic_entropy / max(1, stochastic_terms), 5
            ),
            "seconds": round(sum(item["seconds"] for item in details), 2),
            "wall_seconds": round(elapsed, 2),
            "cumulative_seconds": None,
            "batch_tape_hash": tape_hash(tapes),
            "decisions": total("decisions"),
            **types,
            **greedy,
        }
    )
    report["nonfinite"] = _nonfinite_count(report)
    return report, details


def _tape_candidate_entropy(controller: Any, record: dict[str, Any]) -> float:
    """Sum of per-candidate binary entropies over the tape's actionable set, no gradient."""
    actionable = record["actionable"]
    if not actionable:
        return 0.0
    with torch.no_grad():
        logits, _values = controller.forward(
            torch.as_tensor(np.asarray(record["features"], dtype=np.float32))
        )
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        per_candidate = -(probabilities * log_probabilities).sum(dim=-1)
    positions = {target: index for index, target in enumerate(record["targets"])}
    return float(sum(float(per_candidate[positions[target]].item()) for target in actionable))


def _weights_finite(controller: Any) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in common.tensors(controller.state_dict()))


def _nonfinite_count(report: dict[str, Any]) -> int:
    """How many of the report's own numbers are NaN or inf -- reported, never silently dropped."""
    found = 0
    for key, value in report.items():
        if isinstance(value, float) and not np.isfinite(value):
            found += 1
    return found


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "total": round(float(np.sum(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "p25": round(float(np.percentile(values, 25)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def diagnostics_at(
    updates: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """The §5 block for one checkpoint: absolute counts, distributions, then the optimiser.

    Written per milestone rather than reconstructed at the end, because the reconstruction would
    have to read the checkpoint's own state -- and a report that can only be produced by loading
    the weights it is describing is a report that cannot be checked while the run is still going.
    """
    actionable = [int(row["actionable_candidates"]) for row in rows]
    send_steps = sorted(
        step for row in rows for step in row.get("send_steps") or []
    )
    return {
        "episodes": len(rows),
        "updates": len(updates),
        "environment_steps": int(max((int(item["at_steps"]) for item in updates), default=0)),
        "actionable_per_episode": _quantiles([float(value) for value in actionable]),
        "candidates_per_episode": _quantiles(
            [float(row["candidates"]) for row in rows]
        ),
        "initial_executable_total": sum(
            int(row["initial_executable_candidates"]) for row in rows
        ),
        "actionable_total": sum(actionable),
        "actor_terms_total": sum(int(row["actor_terms"]) for row in rows),
        "masked_by_credit_total": sum(int(row["masked_by_credit"]) for row in rows),
        "masked_by_capacity_total": sum(int(row["masked_by_capacity"]) for row in rows),
        "episodes_without_actionable": sum(1 for value in actionable if value == 0),
        "episodes_without_decisions": sum(
            1 for row in rows if int(row["decisions"]) == 0
        ),
        "steps_with_candidates": sum(int(row["steps_with_candidates"]) for row in rows),
        "steps_with_zero_actionable": sum(
            int(row["steps_with_zero_actionable"]) for row in rows
        ),
        "requested_total": sum(int(row["requested"]) for row in rows),
        "executed_policy_total": sum(int(row["executed_policy_emissions"]) for row in rows),
        "policy_refusals_total": sum(
            int(row["policy_budget_refusals"]) + int(row["policy_capacity_refusals"])
            for row in rows
        ),
        "message_types": {
            key: sum(int(row[f"message_{key}"]) for row in rows)
            for key in ("first", "update", "retry_same_view")
        },
        "send_step_percentiles": {
            str(percentile): int(np.percentile(send_steps, percentile))
            for percentile in (5, 25, 50, 75, 95)
        }
        if send_steps
        else {"note": "no emission step recorded in this window"},
        "credit_bytes_spent_total": sum(int(row["credit_spent_bytes"]) for row in rows),
        "credit_used_fraction_mean": round(
            float(np.mean([row["credit_spent_bytes"] / BUDGET_BYTES for row in rows])), 4
        )
        if rows
        else None,
        "optimiser": {
            key: _quantiles([float(item[key]) for item in updates if item.get(key) is not None])
            for key in (
                "policy_surrogate",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "gradient_norm_mean",
                "optimizer_steps",
                "samples",
            )
        },
        "stochastic_send_rate_last": updates[-1]["stochastic_send_rate"] if updates else None,
        "greedy_send_rate_last": updates[-1]["greedy_send_rate"] if updates else None,
        "stochastic_entropy_per_candidate_last": updates[-1][
            "stochastic_entropy_per_actionable_candidate"
        ]
        if updates
        else None,
        "greedy_entropy_per_candidate_last": updates[-1][
            "greedy_entropy_per_actionable_candidate"
        ]
        if updates
        else None,
        "value_explained_variance_last": updates[-1]["value_explained_variance"] if updates else None,
        "max_log_prob_gap": max(
            (float(item["likelihood_alignment"]["max_abs_gap"]) for item in updates), default=0.0
        ),
        "nonfinite_values_total": sum(int(item.get("nonfinite", 0)) for item in updates),
        "wall_seconds": round(sum(float(item.get("wall_seconds", 0.0)) for item in updates), 1),
    }


def _read_episode_rows(path: Path, *, up_to: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("at_steps", 0)) <= up_to:
            rows.append(row)
    return rows


# ------------------------------------------------------------------ checkpoints
def save(
    controller: Any,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    path: Path,
    *,
    freeze_sha: str,
) -> None:
    """A checkpoint that can be resumed by this round and refused by any other.

    §4's list, in one place: weights, optimiser, environment steps, episodes, stream cursor, the
    three RNG streams, the trainer/mask/tape identities, the feature layout, the protocol, the
    freeze SHA, the configuration hash and the initial-weight fingerprint. A field missing here is
    a resume that cannot be checked, which is the same as a resume that cannot be refused.
    """
    from integrated_sar.comm_learning import FEATURE_LAYOUT

    torch.save(
        {
            "controller": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state,
            "trainer_tag": TRAINER_TAG,
            "action_mask_schema": ACTION_MASK_SCHEMA,
            "tape_schema": TAPE_SCHEMA,
            "timing_mode": train_protocol.TIMING_MODE,
            "feature_layout": FEATURE_LAYOUT,
            "code_freeze_sha": freeze_sha,
            "configuration_hash": configuration_hash(),
            "budget_bytes": BUDGET_BYTES,
            "steps_cap": common.STEPS_PER_RUN,
            "schema": SCHEMA,
        },
        path,
    )


def load_for_resume(path: Path, *, training_seed: int, freeze_sha: str | None = None) -> dict[str, Any]:
    """Resume, or refuse. Every refusal names the field that disagreed.

    ``freeze_sha`` defaults to the round's registered freeze and is only passed explicitly by the
    tests, which must be able to check the guard before any freeze exists. A checkpoint from an
    earlier trainer is welcome for a forward pass -- ``load_forward_only`` in the mask-fix round
    does exactly that -- and useless here: its weights were shaped by gradients on actions the
    transport would not honour, so continuing its optimiser would put the defect inside the run
    that is meant to measure its absence.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "trainer_tag": TRAINER_TAG,
        "action_mask_schema": ACTION_MASK_SCHEMA,
        "tape_schema": TAPE_SCHEMA,
        "timing_mode": train_protocol.TIMING_MODE,
        "configuration_hash": configuration_hash(),
        "budget_bytes": BUDGET_BYTES,
        "steps_cap": common.STEPS_PER_RUN,
        "code_freeze_sha": freeze_sha or common.freeze_sha(),
    }
    problems: list[str] = []
    for field, want in expected.items():
        got = payload.get(field)
        if got != want:
            problems.append(f"{field}: checkpoint={got!r} this_run={want!r}")
    state = payload.get("state") or {}
    if int(state.get("seed", -1)) != int(training_seed):
        problems.append(f"seed: checkpoint={state.get('seed')} asked={training_seed}")
    if not state.get("rng"):
        problems.append("rng: a checkpoint without the three RNG streams cannot be resumed exactly")
    if problems:
        raise RuntimeError(
            f"{path} is not resumable by this round: " + "; ".join(problems)
        )
    return payload


def load_for_eval(path: Path, *, training_seed: int) -> Any:
    """Weights for a read-only forward pass: the optimiser state is not even opened.

    Evaluation must not be able to resume training by accident, so this reader checks the same
    identity fields a resume would -- trainer tag, mask and tape schema, timing mode, budget and
    the freeze that produced the run -- and then takes only the controller. The one field it cannot
    require is the RNG block, which a milestone copy may legitimately not carry.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "trainer_tag": TRAINER_TAG,
        "action_mask_schema": ACTION_MASK_SCHEMA,
        "tape_schema": TAPE_SCHEMA,
        "timing_mode": train_protocol.TIMING_MODE,
        "budget_bytes": BUDGET_BYTES,
    }
    problems = [
        f"{field}: checkpoint={payload.get(field)!r} this_round={want!r}"
        for field, want in expected.items()
        if payload.get(field) != want
    ]
    state = payload.get("state") or {}
    if int(state.get("seed", -1)) != int(training_seed):
        problems.append(f"seed: checkpoint={state.get('seed')} asked={training_seed}")
    if payload.get("code_freeze_sha") != state.get("code_freeze_sha"):
        problems.append(
            f"code_freeze_sha: file={payload.get('code_freeze_sha')} "
            f"state={state.get('code_freeze_sha')}"
        )
    if problems:
        raise RuntimeError(f"{path} is not a checkpoint of this round: " + "; ".join(problems))
    controller = initial_controller(training_seed)
    controller.load_state_dict(payload["controller"])
    return controller


# ------------------------------------------------------------------ the training loop
def train(args: argparse.Namespace) -> int:
    training_seed = int(args.seed)
    torch.set_num_threads(max(1, int(args.threads)))
    freeze = common.freeze_sha()
    cleanliness = common.freeze_cleanliness()
    if not cleanliness["clean"]:
        raise RuntimeError(
            f"the execution path is not clean against the freeze {freeze[:9]}: {cleanliness}"
        )
    directory = common.TRAIN_OUT / f"seed{training_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"maskretrain_seed{training_seed}.pt"
    random.seed(training_seed)
    np.random.seed(training_seed % 2**31)
    controller = initial_controller(training_seed)
    initial_fingerprint = common.weight_fingerprint(controller)
    optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
    state: dict[str, Any] = {
        "seed": training_seed,
        "budget_bytes": BUDGET_BYTES,
        "budget_tier": common.PRIMARY_TIER,
        "timing_mode": train_protocol.TIMING_MODE,
        "trainer_tag": TRAINER_TAG,
        "action_mask_schema": ACTION_MASK_SCHEMA,
        "tape_schema": TAPE_SCHEMA,
        "code_freeze_sha": freeze,
        "configuration_hash": configuration_hash(),
        "initial_weights_fingerprint": initial_fingerprint,
        "weights_loaded_from_checkpoint": None,
        "environment_steps": 0,
        "episodes": 0,
        "cursor": 0,
        "updates": [],
        "milestones": {},
        "scenario_order": list(TRAIN_SEEDS),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    resumed = False
    if path.exists():
        payload = load_for_resume(path, training_seed=training_seed)
        controller.load_state_dict(payload["controller"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["state"])
        state["rng_restored"] = True
        resumed = True
        print(
            f"[seed {training_seed}] resumed at {state['environment_steps']} steps "
            f"freeze={state['code_freeze_sha'][:9]}",
            flush=True,
        )
    else:
        from onpolicy import rng_state

        state["rng"] = rng_state()
        # The 0-step checkpoint is recorded as landed before the loop runs, so the first update
        # cannot overwrite it with a post-update copy under the same name -- a reader comparing
        # "what the model did untrained" against "at 100k" must be able to trust which file is which.
        state["milestones"]["0"] = 0
        save(controller, optimizer, state, path, freeze_sha=freeze)
        save(
            controller,
            optimizer,
            state,
            directory / f"maskretrain_seed{training_seed}_at0.pt",
            freeze_sha=freeze,
        )
        print(
            f"[seed {training_seed}] initial checkpoint at 0 steps, fingerprint "
            f"{initial_fingerprint}, freeze {freeze[:9]}",
            flush=True,
        )
    rows_path = directory / f"episodes_seed{training_seed}.jsonl"
    updates_path = directory / f"updates_seed{training_seed}.jsonl"
    if not path.exists():        # A fresh start that finds episode rows from an earlier attempt would otherwise append to
        # them, and a reader could not tell which rows belonged to which run. Quarantine, never
        # delete: the old rows are evidence about a failed attempt, not garbage.
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for stale in (rows_path, updates_path):
            if stale.exists():
                stale.rename(stale.with_name(f"{stale.stem}.quarantined_{stamp}{stale.suffix}"))
    # Each arm owns a ledger file: the three training processes run in parallel and the ledger is a
    # read-modify-write document, so a shared one would have them overwrite each other's counts.
    # The 1,500 ceiling is still global because ``reserve`` adds up every ledger file before it
    # lets an episode start.
    ledger = common.LaunchLedger(f"LAUNCH_LEDGER_train_seed{training_seed}.json")
    started = time.time()
    while state["environment_steps"] < args.steps:
        batch_seeds = [
            TRAIN_SEEDS[(state["cursor"] + index) % len(TRAIN_SEEDS)]
            for index in range(EPISODES_PER_UPDATE)
        ]
        state["cursor"] = (state["cursor"] + EPISODES_PER_UPDATE) % len(TRAIN_SEEDS)
        began = time.time()
        episodes: list[dict[str, Any]] = []
        for index, stream_seed in enumerate(batch_seeds):
            ledger.reserve("train", stream_seed, arm=f"seed{training_seed}")
            episode = rollout(
                controller,
                stream_seed,
                training_seed=training_seed,
                episode_index=int(state["episodes"]) + index,
                directory=directory / "rollout",
            )
            episodes.append(episode)
        at_steps = int(state["environment_steps"]) + STEPS_PER_EPISODE * len(batch_seeds)
        report, details = update_record(
            controller,
            optimizer,
            episodes,
            at_steps=at_steps,
            batch_seeds=batch_seeds,
            elapsed=time.time() - began,
        )
        for index, episode in enumerate(episodes):
            ledger.record(
                "train",
                int(batch_seeds[index]),
                True,
                f"on_time={details[index]['on_time']} bytes={details[index]['credit_spent_bytes']}",
                arm=f"seed{training_seed}",
            )
        with rows_path.open("a", encoding="utf-8") as handle:
            for item in details:
                handle.write(
                    json.dumps(
                        {**item, "training_seed": training_seed, "at_steps": at_steps},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        with updates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
        state["environment_steps"] = at_steps
        state["episodes"] += len(batch_seeds)
        state["updates"].append(report)
        from onpolicy import rng_state

        state["rng"] = rng_state()
        state["cumulative_seconds"] = round(time.time() - started, 1)
        report["cumulative_seconds"] = state["cumulative_seconds"]
        milestones = state["milestones"]
        for milestone in CHECKPOINT_STEPS:
            if at_steps >= milestone and str(milestone) not in milestones:
                milestones[str(milestone)] = at_steps
                checkpoint = directory / f"maskretrain_seed{training_seed}_at{milestone}.pt"
                save(controller, optimizer, state, checkpoint, freeze_sha=freeze)
                common.write_json(
                    directory / f"diagnostics_at{milestone}.json",
                    {
                        **diagnostics_at(
                            state["updates"],
                            _read_episode_rows(rows_path, up_to=at_steps),
                        ),
                        "milestone_requested": milestone,
                        "landed_at_steps": at_steps,
                        "checkpoint": checkpoint.name,
                        "weights_fingerprint": common.weight_fingerprint(controller),
                        "window": "everything up to this checkpoint, from initial weights",
                    },
                )
                print(
                    f"[seed {training_seed}] checkpoint {milestone} landed at {at_steps} steps",
                    flush=True,
                )
        save(controller, optimizer, state, path, freeze_sha=freeze)
        print(
            f"[seed {training_seed}] steps={state['environment_steps']:>7} "
            f"on_time={report['batch_on_time']}/{report['batch_targets']} "
            f"bytes={report['batch_bytes']} use={report['batch_budget_use']} "
            f"act={report['actionable_candidates']}/{report['candidates']} "
            f"mask={report['masked_by_credit']} send={report['stochastic_send_rate']}/"
            f"g{report['greedy_send_rate']} ent={report['stochastic_entropy_per_actionable_candidate']} "
            f"samples={report['samples']} opt={report['optimizer_steps']} "
            f"kl={report['approx_kl']} clip={report['clip_fraction']} "
            f"vloss={report['value_loss']} ev={report['value_explained_variance']} "
            f"gap={report['likelihood_alignment']['max_abs_gap']:.1e} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
    summary = {
        "seed": training_seed,
        "budget_bytes": BUDGET_BYTES,
        "timing_mode": train_protocol.TIMING_MODE,
        "trainer_tag": TRAINER_TAG,
        "action_mask_schema": ACTION_MASK_SCHEMA,
        "tape_schema": TAPE_SCHEMA,
        "code_freeze_sha": freeze,
        "configuration_hash": configuration_hash(),
        "environment_steps": state["environment_steps"],
        "episodes": state["episodes"],
        "updates": len(state["updates"]),
        "resumed": resumed,
        "initial_weights_fingerprint": initial_fingerprint,
        "final_weights_fingerprint": common.weight_fingerprint(controller),
        "optimizer_steps_cumulative": sum(
            int(item.get("optimizer_steps", 0)) for item in state["updates"]
        ),
        "decision_samples_recorded": sum(
            int(item.get("samples", 0)) for item in state["updates"]
        ),
        "actor_terms_total": sum(int(item.get("actor_terms", 0)) for item in state["updates"]),
        "actionable_total": sum(
            int(item.get("actionable_candidates", 0)) for item in state["updates"]
        ),
        "candidates_total": sum(int(item.get("candidates", 0)) for item in state["updates"]),
        "masked_by_credit_total": sum(
            int(item.get("masked_by_credit", 0)) for item in state["updates"]
        ),
        "episodes_without_actionable_total": sum(
            int(item.get("episodes_without_actionable", 0)) for item in state["updates"]
        ),
        "requested_total": sum(int(item.get("requested_by_policy", 0)) for item in state["updates"]),
        "executed_total": sum(
            int(item.get("executed_policy_emissions", 0)) for item in state["updates"]
        ),
        "policy_refusals_total": sum(
            int(item.get("policy_budget_refusals", 0)) + int(item.get("policy_capacity_refusals", 0))
            for item in state["updates"]
        ),
        "likelihood_max_abs_gap": max(
            float(item["likelihood_alignment"]["max_abs_gap"]) for item in state["updates"]
        ),
        "nonfinite_values_recorded": sum(int(item.get("nonfinite", 0)) for item in state["updates"]),
        "sampling_frequency": dict(
            sorted(
                Counter(
                    seed for item in state["updates"] for seed in item.get("batch_seeds", [])
                ).items()
            )
        ),
        "wall_seconds": round(time.time() - started, 1),
        "milestones": state["milestones"],
        "checkpoints": sorted(
            file.name for file in directory.glob("*.pt")
        ),
        "last_update": state["updates"][-1] if state["updates"] else None,
    }
    common.write_json(directory / f"maskretrain_seed{training_seed}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str), flush=True)
    return 0


# ------------------------------------------------------------------ probes (0 optimizer steps)
def probe(args: argparse.Namespace) -> int:
    """Throughput and identity on training streams, before any weight moves.

    Four episodes per training seed, each from that seed's own initial weights, measured against
    the real transport: how long an update costs, whether the requested sends are the executed
    ones, whether the credit ever refuses a policy send, and whether the logged likelihood still
    recomputes. These episodes do not train, and they are booked against the round's ceiling.
    """
    torch.set_num_threads(max(1, int(args.threads)))
    freeze = common.freeze_sha()
    streams = common.verify_streams()
    if not streams["passed"]:
        raise RuntimeError(f"the streams are not the recorded ones: {streams['mismatches']}")
    cleanliness = common.freeze_cleanliness()
    if not cleanliness["clean"]:
        raise RuntimeError(f"the execution path is not clean against the freeze: {cleanliness}")
    directory = common.EPISODE_OUT / "probe"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = common.LaunchLedger("LAUNCH_LEDGER_probe.json")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    per_seed = int(args.episodes)
    plan = [
        (training_seed, stream_seed)
        for training_seed in common.TRAINING_SEEDS
        for stream_seed in TRAIN_SEEDS[:per_seed]
    ]
    for training_seed, stream_seed in plan:
        if ledger.remaining() <= 0:
            break
        ledger.reserve("probe", stream_seed, arm=f"probe_seed{training_seed}")
        controller = initial_controller(training_seed)
        began = time.time()
        try:
            episode = rollout(
                controller,
                stream_seed,
                training_seed=training_seed,
                episode_index=len(rows),
                directory=directory,
            )
            collector = episode["collector"]
            tapes = collector.records
            detail = per_episode_row(episode)
            alignment = masked.drawn_versus_logged(controller, tapes)
            greedy = greedy_replay(controller, tapes)
            invariants = masked.tape_invariants(tapes)
            row = {
                **detail,
                "training_seed": int(training_seed),
                "weight_fingerprint": common.weight_fingerprint(controller),
                "optimizer_steps": 0,
                "likelihood_alignment": alignment,
                "tape_invariants_passed": invariants["passed"],
                "tape_invariant_violations": invariants["violations"],
                "greedy": greedy,
                "probe_seconds": round(time.time() - began, 2),
            }
            rows.append(row)
            ledger.record("probe", stream_seed, True, f"candidates={detail['candidates']}",
                          arm=f"probe_seed{training_seed}")
        except Exception as error:  # a failure spends the ceiling and says so
            failures.append(
                {"training_seed": training_seed, "stream_seed": stream_seed,
                 "error": f"{type(error).__name__}: {error}"}
            )
            ledger.record("probe", stream_seed, False, f"{type(error).__name__}: {error}",
                          arm=f"probe_seed{training_seed}")
    per_stream = per_seed * len(common.TRAINING_SEEDS)
    seconds = [float(row["seconds"]) for row in rows] or [0.0]
    payload = {
        "schema": SCHEMA,
        "date": common.git_out("log", "-1", "--format=%ad", "--date=short"),
        "plan": {
            "episodes": len(plan),
            "per_training_seed": per_seed,
            "training_seeds": list(common.TRAINING_SEEDS),
            "streams": [int(seed) for seed in TRAIN_SEEDS[:per_seed]],
            "budget_bytes": BUDGET_BYTES,
            "timing_mode": common.TIMING_MODE,
            "trainer_tag": TRAINER_TAG,
            "action_mask_schema": ACTION_MASK_SCHEMA,
            "tape_schema": TAPE_SCHEMA,
            "optimizer_steps": 0,
        },
        "rows": rows,
        "technical_failures": failures,
        "throughput": {
            "episodes": len(rows),
            "mean_seconds_per_episode": round(float(np.mean(seconds)), 2),
            "max_seconds_per_episode": round(float(np.max(seconds)), 2),
            "seconds_per_update_4_episodes": round(float(np.mean(seconds)) * EPISODES_PER_UPDATE, 1),
            "environment_steps": int(np.sum([row["steps_run"] for row in rows])) if rows else 0,
            "projection_one_run_hours": round(
                float(np.mean(seconds)) * (common.STEPS_PER_RUN // STEPS_PER_EPISODE) / 3600.0, 2
            ),
            "projection_three_runs_serial_hours": round(
                3
                * float(np.mean(seconds))
                * (common.STEPS_PER_RUN // STEPS_PER_EPISODE)
                / 3600.0,
                2,
            ),
            "note": (
                "measured with the budget mask active on initial weights; a trained arm that talks "
                "less spends fewer credit-constrained candidates, so this is an upper bound on the "
                "rollout cost, not a promise"
            ),
        },
        "closure": {
            "requested_equals_executed": all(
                row["requested"] == row["executed_policy_emissions"] for row in rows
            ),
            "policy_refusals_zero": all(
                row["policy_budget_refusals"] == 0 and row["policy_capacity_refusals"] == 0
                for row in rows
            ),
            "forced_emissions_zero": all(not row["forced_emissions"] for row in rows),
            "tape_invariants_passed": all(row["tape_invariants_passed"] for row in rows),
            "max_log_prob_gap": max(
                (float(row["likelihood_alignment"]["max_abs_gap"]) for row in rows), default=0.0
            ),
            "distinct_initial_fingerprints": len({row["weight_fingerprint"] for row in rows}),
            "optimizer_steps_total": 0,
            "episode_length_all_1000": all(
                row["steps_run"] == STEPS_PER_EPISODE for row in rows
            ),
        },
        "launches": common.ledger_totals()["starts"],
        "code_freeze_sha": freeze,
        "cleanliness": cleanliness,
    }
    common.write_json(common.json_path("THROUGHPUT.json"), payload)
    print(
        json.dumps(
            {
                "episodes": payload["throughput"]["episodes"],
                "throughput": payload["throughput"],
                "closure": payload["closure"],
                "failures": len(failures),
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        )
    )
    return 1 if failures else 0


# ------------------------------------------------------------------ deterministic resume
def _batch_signature(report: dict[str, Any]) -> dict[str, Any]:
    """What "the same run" has to agree on: order, decisions, likelihood, weights, optimiser."""
    return {
        "batch_seeds": list(report["batch_seeds"]),
        "at_steps": int(report["at_steps"]),
        "tape_hash": report["batch_tape_hash"],
        "actionable_candidates": int(report["actionable_candidates"]),
        "masked_by_credit": int(report["masked_by_credit"]),
        "optimizer_steps": int(report["optimizer_steps"]),
        "surrogate": report["policy_surrogate"],
        "value_loss": report["value_loss"],
        "entropy": report["entropy"],
        "approx_kl": report["approx_kl"],
        "clip_fraction": report["clip_fraction"],
        "gradient_norm_mean": report["gradient_norm_mean"],
    }


def _optimizer_signature(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Adam's state, reduced to one number per tensor so a comparison is possible."""
    out: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        for key, value in sorted(group.items()):
            if torch.is_tensor(value):
                out[f"group{index}.{key}"] = round(float(value.double().sum()), 12)
    for key, value in sorted(optimizer.state.items(), key=lambda item: str(item[0])):
        for name, tensor in value.items():
            if torch.is_tensor(tensor):
                out[f"state{key}.{name}"] = round(float(tensor.double().sum()), 12)
    return out


def resume_check(args: argparse.Namespace) -> int:
    """Two updates in a row against "save after one, restore, then run the second".

    The comparison is on the second update only: the first is the same code either way, so agreeing
    there would prove nothing. Everything that must be reproducible is listed -- the stream order
    the cursor picks next, the decisions the batch made, the reported losses, the weights and the
    optimiser's moments -- and floats are required to match exactly rather than within a hand-waved
    epsilon, because both branches run the same CPU kernels in the same order. If a future change
    breaks that, the test says which quantity moved instead of quietly tolerating it.
    """
    from onpolicy import rng_state

    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.EPISODE_OUT / "resume_check"
    directory.mkdir(parents=True, exist_ok=True)
    freeze = common.freeze_sha()
    ledger = common.LaunchLedger("LAUNCH_LEDGER_resume_check.json")
    training_seed = int(args.seed)

    def fresh() -> tuple[Any, Any, dict[str, Any]]:
        random.seed(training_seed)
        np.random.seed(training_seed % 2**31)
        controller = initial_controller(training_seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
        state = {
            "seed": training_seed,
            "environment_steps": 0,
            "episodes": 0,
            "cursor": 0,
            "updates": [],
            "rng": rng_state(),
            "code_freeze_sha": freeze,
        }
        return controller, optimizer, state

    def step(controller: Any, optimizer: Any, state: dict[str, Any]) -> dict[str, Any]:
        batch_seeds = [
            TRAIN_SEEDS[(state["cursor"] + index) % len(TRAIN_SEEDS)]
            for index in range(EPISODES_PER_UPDATE)
        ]
        state["cursor"] = (state["cursor"] + EPISODES_PER_UPDATE) % len(TRAIN_SEEDS)
        began = time.time()
        episodes = []
        for index, stream_seed in enumerate(batch_seeds):
            ledger.reserve("resume_check", stream_seed, arm="resume")
            episode = rollout(
                controller,
                stream_seed,
                training_seed=training_seed,
                episode_index=int(state["episodes"]) + index,
                directory=directory,
            )
            episodes.append(episode)
            ledger.record("resume_check", stream_seed, True, "ok", arm="resume")
        at_steps = int(state["environment_steps"]) + STEPS_PER_EPISODE * len(batch_seeds)
        report, _details = update_record(
            controller, optimizer, episodes, at_steps=at_steps, batch_seeds=batch_seeds,
            elapsed=time.time() - began,
        )
        state["environment_steps"] = at_steps
        state["episodes"] += len(batch_seeds)
        state["updates"].append(report)
        state["rng"] = rng_state()
        return report

    continuous_controller, continuous_optimizer, continuous_state = fresh()
    first = step(continuous_controller, continuous_optimizer, continuous_state)
    second = step(continuous_controller, continuous_optimizer, continuous_state)
    continuous = {
        "first": _batch_signature(first),
        "second": _batch_signature(second),
        "weights": common.weight_fingerprint(continuous_controller),
        "optimizer": _optimizer_signature(continuous_optimizer),
    }

    split_controller, split_optimizer, split_state = fresh()
    step(split_controller, split_optimizer, split_state)
    path = directory / "resume_check.pt"
    save(split_controller, split_optimizer, split_state, path, freeze_sha=freeze)
    payload = load_for_resume(path, training_seed=training_seed)
    restored_controller = initial_controller(training_seed)
    restored_optimizer = torch.optim.Adam(
        restored_controller.parameters, lr=train_pilot.LEARNING_RATE
    )
    restored_controller.load_state_dict(payload["controller"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    restored_state = dict(payload["state"])
    from onpolicy import restore_rng

    if not restore_rng(restored_state.get("rng")):
        raise RuntimeError("resume check could not restore the RNG streams")
    after = step(restored_controller, restored_optimizer, restored_state)
    restored = {
        "second": _batch_signature(after),
        "weights": common.weight_fingerprint(restored_controller),
        "optimizer": _optimizer_signature(restored_optimizer),
    }
    differences = {
        key: {"continuous": value, "resumed": restored["second"].get(key)}
        for key, value in continuous["second"].items()
        if restored["second"].get(key) != value
    }
    payload_result = {
        "schema": SCHEMA,
        "training_seed": training_seed,
        "episodes_used": 8,
        "optimizer_steps_total": int(first["optimizer_steps"]) + 2 * int(second["optimizer_steps"]),
        "continuous": continuous,
        "resumed_after_one_update": restored,
        "differences": differences,
        "weights_identical": continuous["weights"] == restored["weights"],
        "optimizer_identical": continuous["optimizer"] == restored["optimizer"],
        "next_batch_seeds_continuous": [
            TRAIN_SEEDS[(continuous_state["cursor"] + index) % len(TRAIN_SEEDS)]
            for index in range(EPISODES_PER_UPDATE)
        ],
        "next_batch_seeds_resumed": [
            TRAIN_SEEDS[(restored_state["cursor"] + index) % len(TRAIN_SEEDS)]
            for index in range(EPISODES_PER_UPDATE)
        ],
        "passed": (
            not differences
            and continuous["weights"] == restored["weights"]
            and continuous["optimizer"] == restored["optimizer"]
            and [
                TRAIN_SEEDS[(continuous_state["cursor"] + index) % len(TRAIN_SEEDS)]
                for index in range(EPISODES_PER_UPDATE)
            ]
            == [
                TRAIN_SEEDS[(restored_state["cursor"] + index) % len(TRAIN_SEEDS)]
                for index in range(EPISODES_PER_UPDATE)
            ]
        ),
        "launches_used_so_far": ledger.starts,
    }
    common.write_json(common.json_path("RESUME_CHECK.json"), payload_result)
    print(json.dumps(payload_result, ensure_ascii=False, indent=1, default=str))
    return 0 if payload_result["passed"] else 1


def identity_report() -> dict[str, Any]:
    """The three starting points, and the proof that none of them came from a checkpoint."""
    fingerprints = {}
    for seed in common.TRAINING_SEEDS:
        controller = initial_controller(seed)
        fingerprints[seed] = {
            "fingerprint": common.weight_fingerprint(controller),
            "parameters": sum(int(tensor.numel()) for tensor in controller.parameters),
            "loaded_from": None,
        }
    distinct = len({item["fingerprint"] for item in fingerprints.values()})
    return {
        "training_seeds": list(common.TRAINING_SEEDS),
        "initial_weights": fingerprints,
        "distinct": distinct,
        "all_distinct": distinct == len(common.TRAINING_SEEDS),
        "configuration_hash": configuration_hash(),
        "trainer": {
            "trainer_tag": TRAINER_TAG,
            "action_mask_schema": ACTION_MASK_SCHEMA,
            "tape_schema": TAPE_SCHEMA,
        },
        "warm_start_policy": (
            "no state dict is read anywhere in this module; the only construction path is "
            "Controller(training_seed), and load_for_resume refuses anything that does not "
            "restate this run's freeze SHA and configuration hash"
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=("train", "probe", "resume-check", "identity"), default="identity"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=common.STEPS_PER_RUN)
    parser.add_argument("--episodes", type=int, default=4, help="probe episodes per training seed")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.mode == "identity":
        print(json.dumps(identity_report(), ensure_ascii=False, indent=1))
        return 0
    if args.mode == "probe":
        if args.episodes * len(common.TRAINING_SEEDS) > common.PROBE_EPISODES:
            raise SystemExit(
                f"--episodes {args.episodes} x {len(common.TRAINING_SEEDS)} seeds exceeds the "
                f"{common.PROBE_EPISODES}-episode probe ceiling"
            )
        return probe(args)
    if args.steps > common.STEPS_PER_RUN:
        raise SystemExit(
            f"--steps {args.steps} exceeds the {common.STEPS_PER_RUN:,} per-run ceiling frozen in "
            "the protocol; this round does not extend itself"
        )
    if args.mode == "resume-check":
        return resume_check(args)
    if common.production_diff()["integrated_sar_changed"]:
        raise SystemExit(
            f"refusing to train: shipped code moved since e692607: "
            f"{common.production_diff()['integrated_sar_changed']}"
        )
    streams = common.verify_streams()
    if not streams["passed"]:
        raise SystemExit(
            f"refusing to train: the streams are not the recorded ones: "
            f"{ {key: streams[key] for key in ('absent', 'mismatches')} }"
        )
    return train(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
