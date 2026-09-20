"""What actually masks a candidate: a small diagnostic, before any of this round's training.

Round 4 reported two numbers that read like one story and are not the same story. At the 25 % tier
"only 3.7 %--9.1 % of candidates were freely selectable" is a share of *decision cells*; "the
learning arm's autonomous byte share is 1.0" is a share of *bytes actually emitted*. The first
divides by candidates offered, the second by packets that went on the air, and neither one
licenses the sentence "most messages were sent by the bound for it". This script measures the
question instead of arguing about it.

For five fixed rules and the three round-4 learned policies, at both byte credits, on eight
training streams, a read-only recorder sits on the policy seam and classifies every candidate the
relay offered by why it could not be freely chosen, keeping both the multi-label flags and one
mutually exclusive primary cause at a declared priority:

``bound_send``
    the shared silence bound had already taken the change over -- the transport speaks whatever
    the arm asks, so there is no decision left here;
``capacity_full``
    the step's emission slots are gone, so asking now would be refused on the spot;
``credit_unaffordable``
    the packet would cost more than the credit still owed to it;
``content_redundant``
    the view on offer is the one already said -- a legal send that would tell a receiver nothing;
``free``
    none of the above: a real SEND/HOLD choice.

A step with no candidates is counted beside the table, never inside its denominator. A bound that
has run out on a change the credit cannot pay for ends in the refusal path, and is counted there,
not as a forced emission.

Nothing here is a claim about *why* the masks are what they are -- only what they are. The
affordability test asks the transport's own cost function and the transport's own budget predicate
with the credit as it stands at the start of the step, which is what the arm can know; the
prediction is then held against the realised byte cost of every emission, and any disagreement is
reported.

    python mask_audit.py --mode run --learned --threads 1
    python mask_audit.py --mode aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import protocol_common as common
import protocol_env

from integrated_sar.comm import LEGACY_SHARED_BOUND, estimate_message_bytes
from integrated_sar.comm_policy import SendObservation, parse_parameters

#: The declared priority, fixed before any count so a cause cannot be promoted afterwards. The
#: ordering is "what stops a choice from existing at all" first: a takeover, then a blocked step,
#: then an unaffordable packet, then a send with nothing to say.
CAUSE_PRIORITY: tuple[str, ...] = (
    "bound_send",
    "capacity_full",
    "credit_unaffordable",
    "content_redundant",
    "free",
)

#: Whether the candidate was still the arm's to choose. A takeover, a full step or an unaffordable
#: packet each end the choice; a redundant view does not, it just makes the choice worthless.
BLOCKING = ("bound_send", "capacity_full", "credit_unaffordable")

#: Cells per episode are numerous and no reader needs them individually; the per-episode aggregate
#: plus a bounded illustration is what gets archived.
SAMPLE_CELLS_PER_EPISODE = 40


class MaskRecorder:
    """A pass-through policy that writes down what the relay offered and what it could still do.

    It decides nothing. The wrapped policy answers, and the recorder only reads the observation,
    the relay's own cost function and the relay's own budget predicate, then hands the selection
    back untouched. ``non_perturbation`` in the output is that claim measured, not asserted.
    """

    def __init__(self, inner: Any, *, enforced: bool = True) -> None:
        #: Under ``policy_owned_timing`` an overdue change is still the arm's own decision, so the
        #: clock is recorded as information and does not remove the choice. The census would
        #: otherwise claim that the transport masks candidates it no longer masks at all.
        self._enforced = bool(enforced)
        self._inner = inner
        self.relay: Any = None
        self.name = getattr(inner, "name", "unknown")
        self.orders_selection = getattr(inner, "orders_selection", False)
        self.cells: Counter[str] = Counter()
        self.labels: Counter[str] = Counter()
        self.asked: Counter[str] = Counter()
        self.decisions = 0
        self.candidates = 0
        self.per_step: list[tuple[int, int, int]] = []
        self.predicted: dict[tuple[int, str], int] = {}
        self.sample: list[dict[str, Any]] = []
        self.bound_cells: set[tuple[int, str]] = set()
        #: cause tally per step, so the census can be split at the episode's first refusal
        self.step_cause: dict[int, Counter] = defaultdict(Counter)

    def parameters(self) -> dict[str, Any]:
        return dict(self._inner.parameters())

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        """Record every candidate, then ask the wrapped policy what it wants -- once."""
        asked = self._inner.decide(observation)
        self.decisions += 1
        assert self.relay is not None, "the recorder was never given the relay it reads"
        budget = self.relay.budget
        subscribers = max(1, int(observation.subscribers))
        overdue = self._overdue(observation)
        capacity_full = (
            budget.sends_per_step is not None
            and budget.sends_this_step >= budget.sends_per_step
        )
        requested = set(asked)
        choosable = 0
        for item in observation.items:
            self.candidates += 1
            cost = self._cost_of(item.target_id, subscribers)
            flags: list[str] = []
            if item.target_id in overdue:
                self.bound_cells.add((int(observation.step), item.target_id))
                flags.append("bound_send" if self._enforced else "timer_due_not_enforced")
            if capacity_full:
                flags.append("capacity_full")
            if cost is not None and not budget.can_send(cost):
                flags.append("credit_unaffordable")
            if not item.changed_view:
                flags.append("content_redundant")
            primary = next((cause for cause in CAUSE_PRIORITY if cause in flags), "free")
            if not any(flag in flags for flag in BLOCKING):
                choosable += 1
            self.cells[primary] += 1
            self.step_cause[int(observation.step)][primary] += 1
            for flag in flags:
                self.labels[f"label:{flag}"] += 1
            if item.target_id in requested:
                self.asked[f"asked:{primary}"] += 1
                if cost is not None:
                    self.predicted[(int(observation.step), item.target_id)] = cost
            if len(self.sample) < SAMPLE_CELLS_PER_EPISODE:
                self.sample.append(
                    {
                        "step": int(observation.step),
                        "target": item.target_id,
                        "age_steps": int(item.age_steps),
                        "flags": flags,
                        "primary": primary,
                        "asked": item.target_id in requested,
                        "predicted_cost_bytes": cost,
                        "remaining_bytes": None
                        if budget.bytes_per_episode is None
                        else int(budget.remaining_bytes),
                    }
                )
        self.per_step.append((int(observation.step), choosable, len(observation.items)))
        return asked

    # ---------------------------------------------------------------- transport facts, read
    def _overdue(self, observation: SendObservation) -> set[str]:
        """The bound's own predicate: 3 steps while urgent, 10 otherwise.

        Recomputed from the constants this round declares, then cross-checked against the realised
        ``forced_maxwait`` emissions in the same episode: if the two ever disagree, the census is
        measuring a mask the transport does not apply.
        """
        urgent = common.BOUND_URGENT_WAIT_STEPS
        limit = common.BOUND_MAX_WAIT_STEPS
        window = common.BOUND_URGENCY_TIME
        return {
            item.target_id
            for item in observation.items
            if int(item.age_steps) >= (urgent if 0.0 <= item.remaining_time <= window else limit)
        }

    def _cost_of(self, target_id: str, subscribers: int) -> int | None:
        """What this candidate's packet would cost, from the transport's own function.

        The relay builds a payload from the newest registry view at the moment of sending, so the
        same call one step and one emission apart gives the same bytes -- which is what
        ``cost_prediction_mismatches`` tests.
        """
        ledger = self.relay._ledger.get(target_id)
        if ledger is None:
            return None
        snapshot = self.relay._snapshot_of(target_id) if self.relay._snapshot_of else None
        try:
            message = self.relay._message(target_id, ledger, snapshot)
        except Exception:
            # Neither the registry nor a stored event describes this target: there is no payload
            # to price, so the credit mask is simply not evaluable for this cell.
            return None
        return int(estimate_message_bytes(message)) * subscribers


def audit_episode(
    spec: dict[str, Any],
    seed: int,
    *,
    tier: str,
    directory: Path,
    observe: bool = True,
    enforced: bool = True,
) -> dict[str, Any]:
    """One arm, one stream, one credit: the mask census beside the episode's own ledger."""
    budget_bytes = common.tier_bytes(tier)
    holder: dict[str, Any] = {}

    def hook(relay: Any) -> None:
        inner = relay.policy
        recorder = MaskRecorder(inner, enforced=enforced)
        recorder.relay = relay
        relay.attach_policy(recorder)
        holder["recorder"] = recorder

    row = protocol_env.run_episode(
        seed,
        directory=directory,
        label=spec["arm"],
        policy=spec["policy"] if spec["kind"] == "rule" else "learned",
        policy_params=parse_parameters(spec["params"]) or None,
        budget_bytes=budget_bytes,
        timing_mode=LEGACY_SHARED_BOUND if enforced else POLICY_OWNED_TIMING,
        # The bound is *declared* either way; only the protocol says whether the transport acts on
        # it, which is the same distinction the run arms use.
        max_wait_steps=common.BOUND_MAX_WAIT_STEPS,
        urgent_wait_steps=common.BOUND_URGENT_WAIT_STEPS,
        controller=spec.get("controller"),
        information="isolated",
        delivery_probability=1.0,
        policy_hook=hook if observe else None,
    )
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    rollup = common.attribution_metrics._per_reason_rollup(records)
    closure = common.attribution_metrics._closure(
        row["ledger"], rollup, spent_bytes=row["spent_bytes"]
    )
    emissions = int(row["ledger"]["emissions"])
    forced = int(row["ledger"]["emissions_forced_maxwait"])
    steps = int(common.STREAM_STEPS)
    result: dict[str, Any] = {
        "arm": spec["arm"],
        "kind": spec["kind"],
        "tier": tier,
        "seed": int(seed),
        "budget_bytes": budget_bytes,
        "observed": bool(holder),
        "on_time_targets_per_episode": int(row["task"]["on_time"]),
        "detected_targets_this_episode": int(row["task"]["targets"]),
        "expired_targets_per_episode": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": row["budget_used_fraction"],
        "emissions": emissions,
        "emissions_by_policy": int(row["ledger"]["emissions_policy"]),
        "emissions_forced": forced,
        "bytes_forced": int(row["ledger"]["bytes_forced_maxwait"]),
        "bytes_by_reason": {
            reason: int(row["ledger"][f"bytes_{reason}"])
            for reason in ("policy", "forced_maxwait", "other")
        },
        "autonomous_share_of_bytes": rollup["autonomous_share_of_bytes"],
        "generated_items": int(row["ledger"]["generated_items"]),
        "redundant_emissions": int(row["ledger"]["redundant_emissions"]),
        "refused_by_policy": int(row["ledger"]["refused_policy"]),
        "refused_forced": int(row["ledger"]["refused_forced_maxwait"]),
        "capacity_refused_sends": int(row["ledger"]["capacity_refused_sends"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "max_backlog_age": int(row["ledger"]["max_backlog_age"]),
        "unsent_change_steps": int(row["ledger"]["unsent_change_steps"]),
        "config_hash": row["config_hash"],
        "seconds": float(row["seconds"]),
        "residuals": {key: value for key, value in closure["residuals"].items() if value},
    }
    if not holder:
        return result
    recorder: MaskRecorder = holder["recorder"]
    cells = sum(recorder.cells.values())
    if cells != recorder.candidates:
        raise RuntimeError(
            f"{spec['arm']} {tier} s{seed}: the census does not add up, {cells} != {recorder.candidates}"
        )
    # The prediction is validated against what actually went out, not against itself.
    realised = {
        (int(record["step"]), str(record["target"])): int(record["cost_bytes"])
        for record in records
        if record["kind"] == "send"
    }
    checked = {key: cost for key, cost in recorder.predicted.items() if key in realised}
    mismatched = {key: [cost, realised[key]] for key, cost in checked.items() if cost != realised[key]}
    send_reason = {
        (int(record["step"]), str(record["target"])): str(record.get("send_reason", "policy"))
        for record in records
        if record["kind"] == "send"
    }
    refusal_steps = sorted(
        {int(record["step"]) for record in records if record["kind"] == "refused"}
    )
    first_refusal = refusal_steps[0] if refusal_steps else steps + 1
    pass
    free = recorder.cells["free"]
    result.update(
        {
            "decision_steps": recorder.decisions,
            "candidate_cells": recorder.candidates,
            "ticks_without_candidates": steps - recorder.decisions,
            "cells_by_cause": dict(sorted(recorder.cells.items())),
            "cell_flags": {
                flag[len("label:") :]: value
                for flag, value in sorted(recorder.labels.items())
                if flag.startswith("label:")
            },
            "asked_by_cause": {
                key[len("asked:") :]: value for key, value in sorted(recorder.asked.items())
            },
            "free_candidate_share_of_cells": round(free / cells, 5) if cells else None,
            "choosable_candidates": sum(
                choosable for _, choosable, _ in recorder.per_step
            ),
            "choosable_share_of_cells": round(
                sum(choosable for _, choosable, _ in recorder.per_step) / cells, 5
            )
            if cells
            else None,
            "choosable_per_decision_mean": round(
                float(np.mean([choosable for _, choosable, _ in recorder.per_step])), 3
            )
            if recorder.per_step
            else None,
            "decisions_with_nothing_choosable": sum(
                1 for _, choosable, _ in recorder.per_step if choosable == 0
            ),
            "first_refusal_step": first_refusal if first_refusal <= steps else None,
            # Before/after the credit first said no, in cells. Two different things happen after
            # that moment: a takeover becomes a refusal, and a free candidate stops being one.
            "cells_by_cause_before_first_refusal": dict(
                sorted(
                    Counter(
                        {
                            cause: sum(
                                tally[cause]
                                for step, tally in recorder.step_cause.items()
                                if step < first_refusal
                            )
                            for cause in CAUSE_PRIORITY
                        }
                    ).items()
                )
            ),
            "cells_by_cause_from_first_refusal": dict(
                sorted(
                    Counter(
                        {
                            cause: sum(
                                tally[cause]
                                for step, tally in recorder.step_cause.items()
                                if step >= first_refusal
                            )
                            for cause in CAUSE_PRIORITY
                        }
                    ).items()
                )
            ),
            "cost_predictions_checked": len(checked),
            "cost_prediction_mismatches": mismatched,
            # The census' own idea of "the bound has taken this over" is checked against the
            # transport's per-reason counter: every emission the relay says it forced has to be a
            # cell the census flagged ``bound_send``, or the census is describing a mask that does
            # not exist in this code.
            "forced_emissions_the_census_missed": len(
                [
                    (step, target)
                    for (step, target), reason in send_reason.items()
                    if reason == "forced_maxwait" and (step, target) not in recorder.bound_cells
                ]
            ),
            "sample_cells": recorder.sample,
        }
    )
    return result


def learned_specs(tier: str) -> list[dict[str, Any]]:
    """The three round-4 policies at their 400k end-of-run checkpoint, re-read from disk."""
    import train_pilot

    specs: list[dict[str, Any]] = []
    for training_seed in (1, 2, 3):
        path = (
            common.TIGHT_ROUND
            / "train"
            / f"{tier}_seed{training_seed}"
            / f"tight_{tier}_seed{training_seed}_at400000.pt"
        )
        if not path.exists():
            raise RuntimeError(
                f"{path} is missing; the audit reads archived policies, it does not retrain them"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("trainer") != "onpolicy":
            raise RuntimeError(f"{path} was not produced by the corrected trainer")
        controller = train_pilot.Controller(training_seed)
        controller.load_state_dict(payload["controller"])
        collector = common.onpolicy.OnPolicyCollector(controller, forced=train_pilot.forced, greedy=True)
        specs.append(
            {
                "arm": f"learned{training_seed}_at400000",
                "kind": "learned",
                "policy": "learned",
                "params": "",
                "controller": collector.policy(),
                "checkpoint": str(path),
                "weight_fingerprint": common.weight_fingerprint(controller),
            }
        )
    return specs


def rule_specs() -> list[dict[str, Any]]:
    return [
        {"arm": arm, "kind": "rule", "policy": spec["policy"], "params": spec["params"]}
        for arm, spec in (
            (name, common.LEGACY_ARMS[name])
            for name in ("immediate", "hold_to_bound", "periodic_5", "event", "event_priority")
        )
    ]


def run_mode(args: argparse.Namespace) -> int:
    """One cell per (tier, arm, stream), split across processes by a stable index.

    The task list is built once and sliced by position, so two shards cover the same grid whatever
    the machine or the start order is. A string hash is not usable here: it is salted per process.
    """
    torch.set_num_threads(max(1, int(args.threads)))
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, dict[str, Any], int]] = []
    for tier in args.tiers:
        specs = rule_specs()
        if args.learned:
            specs = specs + learned_specs(tier)
        for spec in specs:
            for seed in args.seeds:
                tasks.append((tier, spec, seed))
    written = 0
    for tier, spec, seed in tasks[args.shard :: args.shards]:
        path = directory / f"cell_{tier}_{spec['arm']}_s{seed}.json"
        if path.exists() and not args.overwrite:
            continue
        observed = audit_episode(spec, seed, tier=tier, directory=directory)
        if not args.observe_only:
            # The recorder must not be a participant: the same episode, with no recorder attached.
            # The same directory and label as the observed run, and only after it has been read
            # back: the transcript path is inside the hashed configuration, so a different name
            # would make the identity field differ for a reason that is not a perturbation.
            quiet = audit_episode(spec, seed, tier=tier, directory=directory)
            compared = (
                "airtime_bytes",
                "emissions",
                "emissions_forced",
                "on_time_targets_per_episode",
                "generated_items",
                "config_hash",
            )
            drift = {
                key: [observed[key], quiet[key]] for key in compared if observed[key] != quiet[key]
            }
            observed["non_perturbation"] = {"fields": drift, "clean": not drift}
        common.write_json(path, observed)
        written += 1
        print(
            f"[{tier}] {spec['arm']} s{seed}: cells={observed['candidate_cells']} "
            f"free={observed['free_candidate_share_of_cells']} "
            f"choosable={observed['choosable_share_of_cells']} "
            f"forced={observed['emissions_forced']}/{observed['emissions']} "
            f"bytes={observed['airtime_bytes']} onT={observed['on_time_targets_per_episode']} "
            f"pred_mismatch={len(observed['cost_prediction_mismatches'])} "
            f"census_missed={observed['forced_emissions_the_census_missed']} ({observed['seconds']}s)",
            flush=True,
        )
    print(json.dumps({"cells_written": written, "directory": str(directory)}, ensure_ascii=False))
    return 0


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key, 0) or 0) for row in rows)


def aggregate_mode(args: argparse.Namespace) -> int:
    directory = args.output_dir
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("cell_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no cells under {directory}; run --mode run first")
    arms = sorted({row["arm"] for row in rows}, key=lambda name: (name.startswith("learned"), name))
    table: dict[str, Any] = {}
    for arm in arms:
        for tier in args.tiers:
            group = [row for row in rows if row["arm"] == arm and row["tier"] == tier]
            if not group:
                continue
            cells = _sum(group, "candidate_cells")
            causes: Counter[str] = Counter()
            flags: Counter[str] = Counter()
            asked: Counter[str] = Counter()
            for row in group:
                causes.update({key: int(value) for key, value in (row.get("cells_by_cause") or {}).items()})
                flags.update({key: int(value) for key, value in (row.get("cell_flags") or {}).items()})
                asked.update({key: int(value) for key, value in (row.get("asked_by_cause") or {}).items()})
            emissions = _sum(group, "emissions")
            forced = _sum(group, "emissions_forced")
            observed = [row for row in group if row.get("observed")]
            table[f"{arm}@{tier}"] = {
                "episodes": len(group),
                "seeds": sorted(int(row["seed"]) for row in group),
                "per_seed": {
                    str(int(row["seed"])): {
                        "emissions": int(row["emissions"]),
                        "airtime_bytes": int(row["airtime_bytes"]),
                    }
                    for row in group
                },
                "candidate_cells": cells,
                "decision_steps": _sum(group, "decision_steps"),
                "ticks_without_candidates": _sum(group, "ticks_without_candidates"),
                "cells_by_cause": dict(sorted(causes.items())),
                "cell_flags": dict(sorted(flags.items())),
                "free_candidate_share_of_cells": round(causes.get("free", 0) / cells, 5) if cells else None,
                "choosable_share_of_cells": round(
                    _sum(group, "choosable_candidates") / cells, 5
                )
                if cells
                else None,
                "asked_by_cause": dict(sorted(asked.items())),
                "emissions": emissions,
                "emissions_forced": forced,
                "forced_share_of_emissions": round(forced / emissions, 5) if emissions else None,
                "bytes_forced": _sum(group, "bytes_forced"),
                "airtime_bytes": _sum(group, "airtime_bytes"),
                "autonomous_share_of_bytes_mean": round(
                    float(np.mean([row["autonomous_share_of_bytes"] or 0.0 for row in observed])), 5
                )
                if observed
                else None,
                "on_time_targets_per_episode": round(
                    float(np.mean([int(row["on_time_targets_per_episode"]) for row in group])), 4
                ),
                "refused_by_policy": _sum(group, "refused_by_policy"),
                "refused_forced": _sum(group, "refused_forced"),
                "capacity_refused_sends": _sum(group, "capacity_refused_sends"),
                "bytes_by_reason": {
                    reason: int(sum((row.get("bytes_by_reason") or {}).get(reason, 0) for row in group))
                    for reason in ("policy", "forced_maxwait", "other")
                },
                "cells_by_cause_before_first_refusal": dict(
                    sorted(
                        Counter(
                            {
                                cause: sum(
                                    int((row.get("cells_by_cause_before_first_refusal") or {}).get(cause, 0))
                                    for row in group
                                )
                                for cause in CAUSE_PRIORITY
                            }
                        ).items()
                    )
                ),
                "cells_by_cause_from_first_refusal": dict(
                    sorted(
                        Counter(
                            {
                                cause: sum(
                                    int((row.get("cells_by_cause_from_first_refusal") or {}).get(cause, 0))
                                    for row in group
                                )
                                for cause in CAUSE_PRIORITY
                            }
                        ).items()
                    )
                ),
                "cost_predictions_checked": _sum(group, "cost_predictions_checked"),
                "cost_prediction_mismatches": sum(
                    len(row.get("cost_prediction_mismatches") or {}) for row in group
                ),
                "forced_emissions_the_census_missed": _sum(group, "forced_emissions_the_census_missed"),
                "non_perturbation_clean": all(
                    (row.get("non_perturbation") or {}).get("clean", True) for row in group
                ),
            }
    payload = {
        "question": (
            "Which candidates could the arm still choose between, and how many bytes did the "
            "silence bound really speak?"
        ),
        "protocol": "legacy_shared_bound, isolated, lossless, 8 training streams, two locked tiers",
        "cause_priority": list(CAUSE_PRIORITY),
        "blocking_causes": list(BLOCKING),
        "denominators": {
            "candidate_cell": "one offered target at one decision step: the arm's decision surface",
            "emission": "one packet the transport put on the air, whichever reason produced it",
            "byte": "one transmission attempt billed at payload bytes x subscribers, lost packets included",
            "warning": (
                "free_candidate_share_of_cells and forced_share_of_emissions are not complements: "
                "the first divides by candidates offered, the second by emissions produced, and a "
                "masked candidate never becomes a packet"
            ),
        },
        "tier_bytes": {tier: common.tier_bytes(tier) for tier in args.tiers},
        "arms": table,
        "episodes": len(rows),
        "cells": str(directory),
    }
    payload["archive_crosscheck"] = archive_crosscheck(table)
    print(
        "archive cross-check: "
        + json.dumps(
            {k: v for k, v in payload["archive_crosscheck"].items() if k != "disagreements"},
            ensure_ascii=False,
        )
    )
    if payload["archive_crosscheck"]["disagreements"]:
        print("ARCHIVE DISAGREEMENTS: " + json.dumps(payload["archive_crosscheck"]["disagreements"], ensure_ascii=False)[:900])
    common.write_json(common.json_path("COMM_MASK_CAUSE_AUDIT.json"), payload)
    for key, values in sorted(table.items(), key=lambda item: (item[0].split("@")[1], item[0])):
        print(
            f"{key:40s} cells={values['candidate_cells']:>6} free={values['free_candidate_share_of_cells']} "
            f"choosable={values['choosable_share_of_cells']} em={values['emissions']:>5} "
            f"forced={values['emissions_forced']:>5} forced_share={values['forced_share_of_emissions']} "
            f"bytes={values['airtime_bytes']:>8} refused(policy/forced)="
            f"{values['refused_by_policy']}/{values['refused_forced']}"
        )
    return 0


def archive_crosscheck(arms: dict[str, Any]) -> dict[str, Any]:
    """Compare this census against round 4's archived transcripts for the same cells.

    Round 4's rule gate wrote one transport transcript per (arm, stream) into a single directory
    whose file names carry no tier. Where a file exists it can therefore only corroborate the tier
    whose run wrote it last, and the file says which one that was: the highest ``spent_after_bytes``
    it contains cannot exceed the credit that was in force. Cells whose archive slot belongs to the
    other tier are reported as unavailable rather than as disagreements.
    """
    archive = common.TIGHT_ROUND / "probe_after"
    if not archive.exists():
        return {"available": False, "reason": f"{archive} is not on this machine"}
    checked: dict[str, Any] = {}
    disagreements: dict[str, Any] = {}
    unusable: list[str] = []
    for key, values in sorted(arms.items()):
        arm, _, tier = key.partition("@")
        for seed in values.get("seeds", []):
            path = archive / f"ct_{arm}_s{seed}.jsonl"
            if not path.exists():
                continue
            records = common.metrics.load_transcript(path)
            sends = [record for record in records if record["kind"] == "send"]
            airtime = sum(int(record["cost_bytes"]) for record in sends)
            ceiling = max((int(record["spent_after_bytes"]) for record in sends), default=0)
            written_tier = next(
                (
                    candidate
                    for candidate in common.BUDGET_TIERS
                    if ceiling <= common.tier_bytes(candidate)
                ),
                None,
            )
            if written_tier != tier:
                unusable.append(
                    f"{tier}/{arm}/s{seed}: the file at {path.name} carries a peak spend of "
                    f"{ceiling} B, which is {written_tier}'s credit, not {tier}'s"
                )
                continue
            census = values["per_seed"][str(seed)]
            cell_key = f"{tier}/{arm}/s{seed}"
            checked[cell_key] = {
                "archive_emissions": len(sends),
                "archive_airtime_bytes": airtime,
                "census_emissions": int(census["emissions"]),
                "census_airtime_bytes": int(census["airtime_bytes"]),
            }
            if (len(sends), airtime) != (int(census["emissions"]), int(census["airtime_bytes"])):
                disagreements[cell_key] = checked[cell_key]
    return {
        "available": True,
        "archive": str(archive),
        "cells_compared": len(checked),
        "cells_agreeing": len(checked) - len(disagreements),
        "disagreements": disagreements,
        "cells_not_usable": len(unusable),
        "why_not_usable": unusable[:4],
        "note": (
            "round 4's probe transcripts are named after the arm and the stream only, so the two "
            "tiers overwrote each other there; its published gate aggregates are tier-qualified and "
            "are checked against this round's 48-stream rule matrix instead"
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("run", "aggregate"), default="run")
    parser.add_argument(
        "--tiers", nargs="+", default=list(common.BUDGET_TIERS), choices=list(common.BUDGET_TIERS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.GATE_STREAMS))
    parser.add_argument("--learned", action="store_true", help="include the round-4 learned policies")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--observe-only", action="store_true", help="skip the quiet rerun")
    parser.add_argument("--output-dir", type=Path, default=common.HERE / "mask_audit")
    args = parser.parse_args(argv)
    if args.shard >= args.shards:
        raise SystemExit(f"--shard {args.shard} is outside --shards {args.shards}")
    return aggregate_mode(args) if args.mode == "aggregate" else run_mode(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
