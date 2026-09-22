"""Read the trained senders and the locked rule on the same 24 development streams.

Evaluation is greedy, masked, and speaks to the same transport the arms trained against: the
collector comes from ``train_budget_masked`` with ``greedy=True``, so the deployed decision is read
under exactly the credit rule the sampler was trained with. Two things the brief insists on are
wired in here rather than assumed:

* **The rule is re-played with this round's code, not quoted.** ``send_once`` runs again on all 24
  streams and is compared per stream against the archived round-5 rows *and* against the archived
  emission sequence recovered from its transcript. A rule that behaves differently under the same
  commit is an unexplained difference, and the protocol says stop and attribute rather than swap in
  an easier opponent.
* **No model is selected by a middle checkpoint.** 100k/200k/300k are read on eight fixed streams and
  are only ever drawn as a curve; the judgement is 0-step and 400k on all 24.

Jobs are sharded, and each shard appends to its own row file: two processes writing one JSONL can
interleave a half-line, and a reader should not have to guess which half it got.

    python evaluate_retrain.py --mode plan
    python evaluate_retrain.py --mode rules --threads 1
    python evaluate_retrain.py --mode eval --kind initial --shards 3 --shard 0 --threads 1
    python evaluate_retrain.py --mode eval --kind curve  --shards 3 --shard 0 --threads 1
    python evaluate_retrain.py --mode eval --kind final   --shards 3 --shard 0 --threads 1
    python evaluate_retrain.py --mode coverage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import retrain_common as common
import train_retrain as entry

import protocol_env  # noqa: E402
import train_protocol  # noqa: E402
import train_pilot  # noqa: E402
from integrated_sar.comm_policy import parse_parameters  # noqa: E402

EVAL_DIR = common.HERE / "evaluation"
SETTING = (common.INFORMATION, common.DELIVERY_PROBABILITY)
ARCHIVED_ROWS = (
    common.ROOT
    / "runs"
    / "comm_protocol_v2_20260920"
    / "evaluation"
    / "25pct"
    / "rows_25pct_isolated_1.0.jsonl"
)
#: What a send sequence has to agree on: which packet, when, at what revision, for what money.
SEQUENCE_FIELDS = ("step", "target", "revision", "cost_bytes", "subscribers")
#: The evaluation cells this round is allowed to fill, and how many streams each reads.
CELLS = {
    "initial": {"checkpoints": [0], "streams": "dev"},
    "curve": {"checkpoints": [100_000, 200_000, 300_000], "streams": "curve"},
    "final": {"checkpoints": [400_000], "streams": "dev"},
}


def checkpoint_path(training_seed: int, at_steps: int) -> Path:
    return (
        common.TRAIN_OUT
        / f"seed{training_seed}"
        / f"maskretrain_seed{training_seed}_at{at_steps}.pt"
    )


def streams_for(kind: str) -> tuple[int, ...]:
    if kind not in CELLS:
        raise ValueError(f"unknown evaluation cell {kind!r}; available {sorted(CELLS)}")
    return (
        common.VALIDATION_SEEDS
        if CELLS[kind]["streams"] == "dev"
        else common.CURVE_STREAMS
    )


def arm_label(kind: str, training_seed: int, at_steps: int) -> str:
    return f"masked_s{training_seed}_at{at_steps}"


def rows_path(name: str, shard: int | None) -> Path:
    suffix = "" if shard is None else f"_s{shard}"
    return EVAL_DIR / f"rows_{name}{suffix}.jsonl"


def load_rows(name: str) -> list[dict[str, Any]]:
    """Every row of one family, de-duplicated by cell so a rerun cannot double a stream's weight."""
    rows: list[dict[str, Any]] = []
    for path in sorted(EVAL_DIR.glob(f"rows_{name}*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        unique[(str(row["arm"]), int(row["seed"]))] = row
    return list(unique.values())


def emission_sequence(transcript: str) -> list[tuple[Any, ...]]:
    """The sends in order, as the transport recorded them -- the sequence a rule must reproduce."""
    path = Path(transcript)
    if not path.exists():
        return []
    sequence: list[tuple[Any, ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "send":
            continue
        sequence.append(tuple(record.get(field) for field in SEQUENCE_FIELDS))
    return sequence


# ------------------------------------------------------------------ row builders
def _common_row(row: dict[str, Any], collector: Any) -> dict[str, Any]:
    """One flat row per episode: task outcome, transport ledger, mask accounting, closure."""
    ledger = row["ledger"]
    timing = collector.timing()
    tapes = collector.records
    invariants = entry.masked.tape_invariants(tapes)
    alignment = entry.masked.drawn_versus_logged(collector.controller, tapes)
    requested = sum(1 for record in tapes for on in record["requested"].values() if on)
    greedy_send = sum(
        1
        for record in tapes
        for target in record["actionable"]
        if record["actions"][target] == 1
    )
    emissions = int(ledger["emissions"])
    first = int(ledger["first_emissions"])
    refresh = int(ledger["refresh_emissions"])
    redundant = int(ledger["redundant_emissions"])
    return {
        "on_time_targets_per_episode": int(row["task"]["on_time"]),
        "detected_targets_this_episode": int(row["task"]["targets"]),
        "expired_targets_per_episode": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": row["budget_used_fraction"],
        "emissions": emissions,
        "emissions_by_policy": int(ledger["emissions_policy"]),
        "emissions_forced": int(ledger["emissions_forced_maxwait"]),
        "redundant_emissions": redundant,
        "message_first": first,
        "message_update": refresh - redundant,
        "message_retry_same_view": redundant,
        "budget_refusals": int(ledger["budget_refused_sends"]),
        "capacity_refusals": int(ledger["capacity_refused_sends"]),
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "max_backlog_age": int(ledger["max_backlog_age"]),
        "generated_items": int(ledger["generated_items"]),
        "decisions": int(collector.decisions),
        "candidates": int(collector.candidates),
        "actionable_candidates": int(timing["free_candidates"]),
        "masked_by_credit": int(timing["masked_by_credit"]),
        "masked_by_capacity": int(timing["masked_by_capacity"]),
        "greedy_send": greedy_send,
        "greedy_send_rate": round(greedy_send / max(1, int(timing["free_candidates"])), 4),
        "requested": requested,
        "forced_sends_by_collector": int(collector.forced_sends),
        "tape_invariants_passed": bool(invariants["passed"]),
        "tape_violation_count": int(invariants["violation_count"]),
        "max_log_prob_gap": float(alignment.get("max_abs_gap", 0.0)),
        "steps_run": train_protocol.actual_steps(row["artefacts"]["lifecycle_log"]),
        "seconds": float(row["seconds"]),
        "config_hash": row["config_hash"],
        "timing_mode": row["timing_mode"],
        "budget_bytes": row["budget_bytes"],
        "transcript": row["artefacts"]["transcript"],
        "send_sequence": emission_sequence(row["artefacts"]["transcript"]),
    }


def score_learned(
    controller: Any, stream_seed: int, *, training_seed: int, at_steps: int, directory: Path
) -> dict[str, Any]:
    """One greedy, masked episode of a trained arm on one development stream."""
    collector = entry.masked.collector_for(controller, greedy=True)
    row = protocol_env.run_episode(
        stream_seed,
        directory=directory,
        label=arm_label("learned", training_seed, at_steps),
        policy="learned",
        budget_bytes=common.BUDGET_BYTES,
        controller=collector.policy(),
        information=common.INFORMATION,
        delivery_probability=common.DELIVERY_PROBABILITY,
        timing_mode=common.TIMING_MODE,
        max_wait_steps=None,
        urgent_wait_steps=None,
        ack=common.ACK,
    )
    record = {
        "arm": arm_label("learned", training_seed, at_steps),
        "kind": "learned",
        "training_seed": int(training_seed),
        "at_steps": int(at_steps),
        "greedy": True,
        "seed": int(stream_seed),
        "weight_fingerprint": common.weight_fingerprint(controller),
    }
    record.update(_common_row(row, collector))
    return record


def score_rule(stream_seed: int, *, directory: Path) -> dict[str, Any]:
    """``send_once`` under this round's code: the same transport, the same credit, no network.

    A rule arm has no controller, no tape and no likelihood, so its row is built by its own
    function rather than by the shared one: the columns that would carry sampled probabilities are
    absent, not zero, and a reader comparing the two kinds of row can see which quantities a rule
    never had.
    """
    row = protocol_env.run_episode(
        stream_seed,
        directory=directory,
        label=common.MAIN_RULE,
        policy=common.MAIN_RULE,
        policy_params=parse_parameters("") or None,
        budget_bytes=common.BUDGET_BYTES,
        information=common.INFORMATION,
        delivery_probability=common.DELIVERY_PROBABILITY,
        timing_mode=common.TIMING_MODE,
        max_wait_steps=None,
        urgent_wait_steps=None,
        ack=common.ACK,
    )
    record = {
        "arm": common.MAIN_RULE,
        "kind": "rule",
        "training_seed": None,
        "at_steps": None,
        "greedy": None,
        "seed": int(stream_seed),
        "weight_fingerprint": None,
    }
    record.update(_rule_row(row))
    return record


def _rule_row(row: dict[str, Any]) -> dict[str, Any]:
    ledger = row["ledger"]
    emissions = int(ledger["emissions"])
    first = int(ledger["first_emissions"])
    refresh = int(ledger["refresh_emissions"])
    redundant = int(ledger["redundant_emissions"])
    return {
        "on_time_targets_per_episode": int(row["task"]["on_time"]),
        "detected_targets_this_episode": int(row["task"]["targets"]),
        "expired_targets_per_episode": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": row["budget_used_fraction"],
        "emissions": emissions,
        "emissions_by_policy": int(ledger["emissions_policy"]),
        "emissions_forced": int(ledger["emissions_forced_maxwait"]),
        "redundant_emissions": redundant,
        "message_first": first,
        "message_update": refresh - redundant,
        "message_retry_same_view": redundant,
        "budget_refusals": int(ledger["budget_refused_sends"]),
        "capacity_refusals": int(ledger["capacity_refused_sends"]),
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "max_backlog_age": int(ledger["max_backlog_age"]),
        "generated_items": int(ledger["generated_items"]),
        "decisions": None,
        "candidates": None,
        "actionable_candidates": None,
        "masked_by_credit": None,
        "masked_by_capacity": None,
        "greedy_send": None,
        "greedy_send_rate": None,
        "requested": None,
        "forced_sends_by_collector": None,
        "tape_invariants_passed": None,
        "tape_violation_count": None,
        "max_log_prob_gap": None,
        "steps_run": train_protocol.actual_steps(row["artefacts"]["lifecycle_log"]),
        "seconds": float(row["seconds"]),
        "config_hash": row["config_hash"],
        "timing_mode": row["timing_mode"],
        "budget_bytes": row["budget_bytes"],
        "transcript": row["artefacts"]["transcript"],
        "send_sequence": emission_sequence(row["artefacts"]["transcript"]),
    }


# ------------------------------------------------------------------ jobs
def jobs(kind: str) -> list[tuple[int, int, int]]:
    return [
        (training_seed, at_steps, stream)
        for training_seed in common.TRAINING_SEEDS
        for at_steps in CELLS[kind]["checkpoints"]
        for stream in streams_for(kind)
    ]


def plan() -> dict[str, Any]:
    booked = common.ledger_totals()["starts"]
    return {
        "cells": {
            name: {
                "episodes": len(jobs(name)),
                "checkpoints": CELLS[name]["checkpoints"],
                "streams": len(streams_for(name)),
            }
            for name in CELLS
        },
        "rule_replay_episodes": len(common.VALIDATION_SEEDS),
        "booked_so_far": booked,
        "launch_cap": common.LAUNCH_CAP,
        "remaining": common.LAUNCH_CAP - booked,
        "checkpoint_files": {
            f"seed{seed}_at{at}": checkpoint_path(seed, at).exists()
            for seed in common.TRAINING_SEEDS
            for at in (0, 100_000, 200_000, 300_000, 400_000)
        },
        "greedy_with_mask": {
            "collector": entry.masked.collector_for(entry.initial_controller(1)).__class__.__name__,
            "budget_mask": True,
            "trainer_tag": entry.TRAINER_TAG,
        },
    }


def run_rules(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = EVAL_DIR / f"runs_rules_s{args.shard}"
    directory.mkdir(parents=True, exist_ok=True)
    # One ledger per shard for the same reason the trainers have one per seed: shards are processes,
    # and a shared read-modify-write counter would lose launches to whichever shard wrote last.
    ledger = common.LaunchLedger(f"LAUNCH_LEDGER_rules_s{args.shard}.json")
    done = {(str(row["arm"]), int(row["seed"])) for row in load_rows("rules")}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for stream in streams_for("initial")[args.shard :: max(1, int(args.shards))]:
        if ("send_once", int(stream)) in done and not args.overwrite:
            continue
        ledger.reserve("rule", stream, arm=common.MAIN_RULE)
        began = time.time()
        try:
            row = score_rule(stream, directory=directory)
            rows.append(row)
            ledger.record(
                "rule", stream, True, f"on_time={row['on_time_targets_per_episode']}",
                arm=common.MAIN_RULE,
            )
            print(
                f"[rule] s{stream}: onT={row['on_time_targets_per_episode']} "
                f"bytes={row['airtime_bytes']} emissions={row['emissions']} "
                f"({row['seconds']}s)",
                flush=True,
            )
        except Exception as error:
            failures.append({"seed": int(stream), "error": f"{type(error).__name__}: {error}"})
            ledger.record("rule", stream, False, f"{type(error).__name__}", arm=common.MAIN_RULE)
    path = rows_path("rules", args.shard if int(args.shards) > 1 else None)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"written": len(rows), "failures": failures, "path": path.name}, ensure_ascii=False
        )
    )
    return 1 if failures else 0


def run_eval(args: argparse.Namespace) -> int:
    kind = str(args.kind)
    torch.set_num_threads(max(1, int(args.threads)))
    directory = EVAL_DIR / f"runs_{kind}_s{args.shard}"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = common.LaunchLedger(f"LAUNCH_LEDGER_eval_{kind}_s{args.shard}.json")
    done = {(str(row["arm"]), int(row["seed"])) for row in load_rows(f"learned_{kind}")}
    written = 0
    failures: list[dict[str, Any]] = []
    for training_seed, at_steps, stream in jobs(kind)[args.shard :: max(1, int(args.shards))]:
        arm = arm_label("learned", training_seed, at_steps)
        if (arm, int(stream)) in done and not args.overwrite:
            continue
        path = checkpoint_path(training_seed, at_steps)
        if not path.exists():
            raise SystemExit(
                f"no checkpoint at {path}; the {at_steps}-step milestone was never landed, and a "
                "missing arm must not be skipped silently"
            )
        ledger.reserve(f"eval_{kind}", stream, arm=arm)
        try:
            controller = entry.load_for_eval(path, training_seed=training_seed)
            row = score_learned(
                controller, stream, training_seed=training_seed, at_steps=at_steps,
                directory=directory,
            )
            with rows_path(f"learned_{kind}", args.shard if int(args.shards) > 1 else None).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            ledger.record(
                f"eval_{kind}", stream, True, f"on_time={row['on_time_targets_per_episode']}",
                arm=arm,
            )
            print(
                f"[{kind}] {arm} s{stream}: onT={row['on_time_targets_per_episode']} "
                f"bytes={row['airtime_bytes']} act={row['actionable_candidates']} "
                f"send={row['greedy_send_rate']} ({row['seconds']}s)",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {"arm": arm, "seed": int(stream), "error": f"{type(error).__name__}: {error}"}
            )
            ledger.record(f"eval_{kind}", stream, False, f"{type(error).__name__}", arm=arm)
    print(
        json.dumps(
            {
                "kind": kind,
                "written": written,
                "expected": len(jobs(kind)),
                "failures": failures,
                "launches_booked": ledger.starts,
            },
            ensure_ascii=False,
        )
    )
    return 1 if failures else 0


def coverage() -> dict[str, Any]:
    """Which cells exist, and whether any of them is short -- a gap is a result, not a rounding."""
    report: dict[str, Any] = {}
    rules = load_rows("rules")
    expected_rule = len(common.VALIDATION_SEEDS)
    report["rules"] = {
        "rows": len(rules),
        "expected": expected_rule,
        "missing_streams": sorted(
            set(common.VALIDATION_SEEDS) - {int(row["seed"]) for row in rules}
        ),
        "complete": len(rules) == expected_rule
        and {int(row["seed"]) for row in rules} == set(common.VALIDATION_SEEDS),
    }
    cells: dict[str, Any] = {}
    for kind in CELLS:
        rows = load_rows(f"learned_{kind}")
        wanted = {(arm_label("learned", seed, at), stream) for seed, at, stream in jobs(kind)}
        have = {(str(row["arm"]), int(row["seed"])) for row in rows}
        cells[kind] = {
            "rows": len(rows),
            "expected": len(wanted),
            "missing": sorted(str(cell) for cell in wanted - have),
            "extra": sorted(str(cell) for cell in have - wanted),
            "complete": not (wanted - have),
        }
    report["cells"] = cells
    report["closure"] = {
        "requested_equals_executed": all(
            row["requested"] == row["emissions_by_policy"] for row in _learned_rows()
        ),
        "policy_refusals_zero": all(
            not row["budget_refusals"] and not row["capacity_refusals"] for row in _all_rows()
        ),
        "tape_invariants_passed": all(
            row["tape_invariants_passed"] for row in _learned_rows()
        ),
        "max_log_prob_gap": max(
            (float(row["max_log_prob_gap"]) for row in _learned_rows()), default=0.0
        ),
        "episode_length_all_1000": all(
            row["steps_run"] == common.STEPS_PER_EPISODE for row in _all_rows()
        ),
    }
    report["launches"] = common.ledger_totals()
    return report


def _learned_rows() -> list[dict[str, Any]]:
    return [row for kind in CELLS for row in load_rows(f"learned_{kind}")]


def _all_rows() -> list[dict[str, Any]]:
    return _learned_rows() + load_rows("rules")


def identity_field_explanation(stream_seed: int) -> dict[str, Any]:
    """Why ``config_hash`` cannot match across rounds, demonstrated instead of asserted.

    ``run_signature.config_hash`` drops ``output``, ``decision_trace`` and ``reservation_log`` from
    the hashed payload for exactly the reason stated there -- two arms differ in their paths by
    construction -- but it keeps ``comm_transcript``, and that path carries the round's directory
    and the arm's label. A replay of the same rule in a new directory therefore has to hash
    differently while behaving identically, and a check that reads that as a disagreement is
    measuring the file layout, not the experiment.

    The demonstration is two mutations of one configuration and nothing else: move the transcript
    path, the hash changes; move the output path, it does not.
    """
    import protocol_common as fixtures
    from integrated_sar.run_signature import config_hash

    values = fixtures.pilot.configuration_for(
        stream_seed,
        directory=common.EPISODE_OUT / "identity_probe",
        label=common.MAIN_RULE,
        stream=None,
        comm_policy=common.MAIN_RULE,
        comm_policy_params="",
        comm_budget_bytes=common.BUDGET_BYTES,
        comm_delivery_probability=common.DELIVERY_PROBABILITY,
        comm_max_wait_steps=None,
        comm_urgent_wait_steps=None,
        comm_timing_mode=common.TIMING_MODE,
        gossip_interval=fixtures.ISOLATED_GOSSIP_INTERVAL,
        quiet=True,
    )
    baseline = config_hash(values)
    transcript_moved = config_hash({**values, "comm_transcript": str(
        common.ROOT / "other_round" / "ct_send_once.jsonl")})
    output_moved = config_hash({**values, "output": str(
        common.ROOT / "other_round" / "rp_send_once.jsonl")})
    return {
        "hashed_payload_excludes": ["output", "decision_trace", "reservation_log"],
        "hashed_payload_still_includes": ["comm_transcript"],
        "transcript_path_alone_changes_config_hash": transcript_moved != baseline,
        "output_path_alone_leaves_config_hash_equal": output_moved == baseline,
        "this_round_transcript": str(values["comm_transcript"]),
        "conclusion": (
            "config_hash is a layout-sensitive identity, not a behavioural one: two runs of the same "
            "rule on the same stream from different directories must differ in it"
        ),
    }


def rule_replay_agreement() -> dict[str, Any]:
    """Compare this round's ``send_once`` with the archived one, per stream and per packet.

    The archived row carries the outcome, the bytes and the ``config_hash``; its transcript carries
    the emission sequence. Sequence equality is the stronger claim and the one the brief asks for --
    the same packets at the same steps for the same money -- and the task outcome would only
    rearrange after a sequence difference, so a difference in the aggregates without one would point
    at the measurement rather than at the behaviour.

    ``config_hash`` is reported separately and not as a failure: it hashes the transcript path,
    which names this round's directory, so it differs for every stream by construction. That is
    established by ``identity_field_explanation``, not by declaring it unimportant -- and had the
    explanation come out the other way, this check would stay red.
    """
    rows = {int(row["seed"]): row for row in load_rows("rules")}
    if not rows or not ARCHIVED_ROWS.exists():
        return {
            "available": bool(rows),
            "archived_rows_present": ARCHIVED_ROWS.exists(),
            "passed": False,
            "note": "the archived comparison rows were not found; the replay cannot be judged",
        }
    archived: dict[int, dict[str, Any]] = {}
    for line in ARCHIVED_ROWS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("arm")) == common.MAIN_RULE:
            archived[int(row["seed"])] = row
    compared = 0
    behavioural: dict[str, Any] = {}
    identity: dict[str, Any] = {}
    sequence_differences: dict[str, Any] = {}
    for seed, row in sorted(rows.items()):
        reference = archived.get(seed)
        if reference is None:
            behavioural[str(seed)] = {"reason": "no archived row for this stream"}
            continue
        compared += 1
        fields = {
            "on_time": (
                int(row["on_time_targets_per_episode"]),
                int(reference["on_time_targets_per_episode"]),
            ),
            "detected": (
                int(row["detected_targets_this_episode"]),
                int(reference["detected_targets_this_episode"]),
            ),
            "expired": (
                int(row["expired_targets_per_episode"]),
                int(reference["expired_targets_per_episode"]),
            ),
            "airtime_bytes": (int(row["airtime_bytes"]), int(reference["airtime_bytes"])),
            "emissions": (int(row["emissions"]), int(reference["emissions"])),
            "config_hash": (str(row["config_hash"]), str(reference["config_hash"])),
        }
        mismatched = {
            name: {"this_round": got, "archived": want}
            for name, (got, want) in fields.items()
            if got != want
        }
        behaviour_only = {
            name: value for name, value in mismatched.items() if name != "config_hash"
        }
        if "config_hash" in mismatched:
            identity[str(seed)] = mismatched["config_hash"]
        if behaviour_only:
            behavioural[str(seed)] = behaviour_only
        archived_sequence = emission_sequence(str(reference.get("transcript") or ""))
        new_sequence = [list(step) for step in row.get("send_sequence") or []]
        if archived_sequence and [list(step) for step in archived_sequence] != new_sequence:
            sequence_differences[str(seed)] = {
                "archived": len(archived_sequence),
                "this_round": len(new_sequence),
                "first_difference": next(
                    (
                        {"archived": list(a), "this_round": list(b)}
                        for a, b in zip(archived_sequence, new_sequence)
                        if list(a) != list(b)
                    ),
                    {"note": "same prefix, different length"},
                ),
            }
    aggregate = {
        key: {
            "this_round": round(
                float(np.mean([row[key] for row in rows.values()])), 5
            ),
            "archived": round(
                float(np.mean([reference[key] for reference in archived.values()])), 5
            ),
        }
        for key in ("on_time_targets_per_episode", "airtime_bytes", "emissions")
    }
    explanation = identity_field_explanation(sorted(rows)[0]) if identity else {}
    identity_explained = bool(
        not identity
        or (
            explanation.get("transcript_path_alone_changes_config_hash")
            and explanation.get("output_path_alone_leaves_config_hash_equal")
        )
    )
    return {
        "available": True,
        "archived_rows": str(ARCHIVED_ROWS.relative_to(common.ROOT).as_posix()),
        "streams_compared": compared,
        "streams_this_round": len(rows),
        "aggregate": aggregate,
        "behaviour_field_differences": behavioural,
        "emission_sequence_differences": sequence_differences,
        "identity_field_differences": {
            "config_hash_streams": sorted(identity),
            "count": len(identity),
            "explanation": explanation,
            "explained": identity_explained,
        },
        "field_differences": {**behavioural, **{
            seed: {"config_hash": value} for seed, value in identity.items()
        }},
        "matches_archived_reference": {
            "mean_on_time_within_rounding": abs(
                aggregate["on_time_targets_per_episode"]["this_round"]
                - common.ARCHIVED_SEND_ONCE_DEV["mean_on_time_per_episode"]
            )
            < 5e-5,
            "mean_airtime_equals_reference": int(
                aggregate["airtime_bytes"]["this_round"]
            )
            == int(common.ARCHIVED_SEND_ONCE_DEV["mean_airtime_bytes"]),
            "mean_emissions_equals_reference": int(aggregate["emissions"]["this_round"])
            == int(common.ARCHIVED_SEND_ONCE_DEV["mean_emissions"]),
        },
        "passed": bool(
            not behavioural
            and not sequence_differences
            and identity_explained
            and compared == len(rows) == len(archived)
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=("plan", "rules", "eval", "coverage", "replay-check"), default="plan"
    )
    parser.add_argument("--kind", default="initial", choices=tuple(CELLS))
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.shard >= args.shards:
        raise SystemExit(f"--shard {args.shard} is outside --shards {args.shards}")
    if args.mode == "plan":
        print(json.dumps(plan(), ensure_ascii=False, indent=1))
        return 0
    if args.mode == "coverage":
        report = coverage()
        common.write_json(common.json_path("EVALUATION_COVERAGE.json"), report)
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0
    if args.mode == "replay-check":
        report = rule_replay_agreement()
        common.write_json(common.json_path("RULE_REPLAY_AGREEMENT.json"), report)
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0 if report["passed"] else 1
    cleanliness = common.freeze_cleanliness()
    if not cleanliness.get("clean"):
        raise SystemExit(f"refusing to evaluate against an unfrozen tree: {cleanliness}")
    if args.mode == "rules":
        return run_rules(args)
    return run_eval(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
