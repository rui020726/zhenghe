"""The controlled counterfactual: does the ACK state ever create a real send/hold trade-off?

The rule matrix can show that an ACK rule beats a no-ACK rule. That is not yet evidence the
ACK state carries *learnable* information: a rule that always retries until confirmed also
beats ``send_once`` on a lossy channel, because it is simply a periodic policy with extra
steps. This diagnostic asks the sharper question.

At a training decision point where the arm may legally SEND or HOLD and the ACK state is
non-trivial, the full simulation state and the future counter-based channel keys are copied,
one step is forced to SEND or to HOLD, and the run then returns to the same fixed rule. The
task outcome and the total bytes of the two branches are compared. A decision point where
both branches are worth taking -- each better than the other on some stream -- is a real
trade-off; one where SEND is always better (or HOLD always is) is a rule, not a decision.

This produces **no labels for an actor** and is **not an oracle performance bound**: forcing
one step and returning to the rule measures the value of the choice, not the value of a
policy that could see the future. It answers gate item 3 only.

    python ack_counterfactual.py --mode run --shards 6 --shard 0 --threads 1
    python ack_counterfactual.py --mode report
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import ack_common as common

#: The rule the branches return to. It is the *fixed* rule of the diagnostic, not a
#: comparator: the point is what one forced step is worth relative to the rule's own choice,
#: so the rule must be the same in both branches and in the unforced baseline.
BASELINE_RULE = "event_retry_until_acked_5"

#: The minimum number of decision points each branch must reach, and the minimum number of
#: streams they must be spread over. Both are gate item 3's thresholds, written before the
#: numbers exist.
MIN_PER_BRANCH = 20
MIN_STREAMS_PER_BRANCH = 8


def snapshot_relay(relay: Any) -> dict[str, Any]:
    """The reverse channel's state, deep-copied so a branch cannot perturb the other."""
    return {
        "ack_pending": [
            (due, ordinal, subscriber, target, ack, cost)
            for due, ordinal, subscriber, target, ack, cost in relay._ack_pending
        ],
        "ack_ordinal": int(relay._ack_ordinal),
        "states": {
            key: (state.highest, state.last_arrival_step, state.reaffirmations)
            for key, state in relay.ack.states.items()
        },
        "ledger": relay.ack.ledger.usage(),
        "broadcast_attempts": dict(relay.ack.broadcast_attempts),
        "counters": dict(relay.counters),
    }


def restore_relay(relay: Any, payload: dict[str, Any]) -> None:
    from integrated_sar.comm_ack import AckState

    relay._ack_pending = list(payload["ack_pending"])
    relay._ack_ordinal = int(payload["ack_ordinal"])
    relay.ack.states = {
        key: AckState(highest=highest, last_arrival_step=step, reaffirmations=count)
        for key, (highest, step, count) in payload["states"].items()
    }
    relay.ack.broadcast_attempts = dict(payload["broadcast_attempts"])
    relay.counters = dict(payload["counters"])


class ForcingPolicy:
    """Wraps the baseline rule and forces one candidate's decision, then gets out of the way.

    The force applies to a single ``target`` on a single ``at_step`` and nowhere else: every
    other candidate keeps the wrapped rule's own answer. That is what makes the consistency
    check below meaningful -- forcing the target to the action the rule already chose must
    reproduce the unforced run byte for byte, and if it does not, the wrapper is perturbing
    more than the one decision it claims to.
    """

    name = "forcing"
    orders_selection = True

    def __init__(self, inner: Any, *, at_step: int, target: str, action: str) -> None:
        self._inner = inner
        self._at_step = int(at_step)
        self._target = str(target)
        self._action = str(action)
        self.applied = False

    def parameters(self) -> dict[str, Any]:
        return {
            "at_step": self._at_step,
            "target": self._target,
            "action": self._action,
        }

    def decide(self, observation: Any) -> tuple[str, ...]:
        chosen = list(self._inner.decide(observation))
        if int(observation.step) == self._at_step and not self.applied:
            self.applied = True
            known = {str(item.target_id) for item in observation.items}
            if self._target not in known:
                # The candidate is gone this step (the view moved on or the task closed):
                # there is nothing to force, and saying so beats forcing something else.
                return tuple(chosen)
            if self._action == "hold":
                return tuple(name for name in chosen if name != self._target)
            if self._target not in chosen:
                chosen.append(self._target)
        return tuple(dict.fromkeys(chosen))


def run_branch(
    seed: int,
    *,
    at_step: int,
    target: str,
    action: str,
    directory: Path,
    tier: str,
    budget_bytes: int | None = None,
) -> dict[str, Any]:
    """One episode where a single candidate's step is forced to SEND or HOLD, then the rule resumes."""
    from integrated_sar.comm_policy import build_comm_policy

    spec = common.arm_spec(BASELINE_RULE)
    inner = build_comm_policy(spec["policy"], spec["params"])
    forcing = ForcingPolicy(inner, at_step=at_step, target=target, action=action)
    row = common.protocol_env.run_episode(
        seed,
        directory=directory,
        label=f"cf_{action}",
        policy=spec["policy"],
        policy_params={
            key: value
            for key, value in (
                token.split("=", 1) for token in str(spec["params"]).split(",") if token
            )
        },
        budget_bytes=budget_bytes if budget_bytes is not None else common.tier_bytes(tier),
        information="isolated",
        delivery_probability=common.PRIMARY_DELIVERY_PROBABILITY,
        timing_mode=spec.get("mode", common.POLICY_MODE),
        controller=forcing,
        max_wait_steps=None,
        urgent_wait_steps=None,
        ack=True,
        ack_delay_ticks=common.ACK_DELAY_TICKS,
        ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
    )
    ledger = row["ledger"]
    return {
        "seed": int(seed),
        "at_step": int(at_step),
        "target": str(target),
        "action": action,
        "forced_applied": bool(forcing.applied),
        "on_time": int(row["task"]["on_time"]),
        "targets": int(row["task"]["targets"]),
        "uplink_bytes": int(ledger["airtime_bytes"]),
        "ack_bytes": int(ledger["ack_attempt_bytes"]),
        "total_bytes": int(ledger["airtime_bytes"]) + int(ledger["ack_attempt_bytes"]),
        "emissions": int(ledger["emissions"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
    }


def run_unforced(
    seed: int,
    *,
    directory: Path,
    tier: str,
    budget_bytes: int | None = None,
) -> dict[str, Any]:
    """The same episode with nothing forced: the rule's own choice, for reference."""
    spec = common.arm_spec(BASELINE_RULE)
    row = common.protocol_env.run_episode(
        seed,
        directory=directory,
        label="cf_free",
        policy=spec["policy"],
        policy_params={
            key: value
            for key, value in (
                token.split("=", 1) for token in str(spec["params"]).split(",") if token
            )
        },
        budget_bytes=budget_bytes if budget_bytes is not None else common.tier_bytes(tier),
        information="isolated",
        delivery_probability=common.PRIMARY_DELIVERY_PROBABILITY,
        timing_mode=spec.get("mode", common.POLICY_MODE),
        max_wait_steps=None,
        urgent_wait_steps=None,
        ack=True,
        ack_delay_ticks=common.ACK_DELAY_TICKS,
        ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
    )
    ledger = row["ledger"]
    return {
        "seed": int(seed),
        "on_time": int(row["task"]["on_time"]),
        "targets": int(row["task"]["targets"]),
        "uplink_bytes": int(ledger["airtime_bytes"]),
        "ack_bytes": int(ledger["ack_attempt_bytes"]),
        "total_bytes": int(ledger["airtime_bytes"]) + int(ledger["ack_attempt_bytes"]),
        "emissions": int(ledger["emissions"]),
    }


def legal_decision_points(seed: int, *, directory: Path, tier: str) -> list[dict[str, Any]]:
    """Steps where the baseline rule had a candidate it could legally have sent or held.

    "Legally" means the transport would have paid for it: the candidate was offered, and the
    step was not already saturated by the per-step capacity. A point where the rule itself
    chose to send is still a legal decision point -- the question is what the *other* choice
    would have been worth, so both kinds are collected.
    """
    from integrated_sar.comm_policy import build_comm_policy

    spec = common.arm_spec(BASELINE_RULE)
    offered: list[dict[str, Any]] = []

    class Recorder:
        name = "recorder"
        orders_selection = True

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def parameters(self) -> dict[str, Any]:
            return self._inner.parameters()

        def decide(self, observation: Any) -> tuple[str, ...]:
            # The inner rule's order is returned untouched. Turning it into a set first
            # would reorder an ``orders_selection`` rule, and under a short credit the order
            # decides who is paid for -- so the probe episode would stop being the same
            # episode the branches are measured against. The consistency check below is what
            # catches that, and it did.
            chosen = self._inner.decide(observation)
            selected = set(chosen)
            for item in observation.items:
                offered.append(
                    {
                        "step": int(observation.step),
                        "target": str(item.target_id),
                        "rule_sent": str(item.target_id) in selected,
                        "confirmed_fraction": float(item.ack_confirmed_fraction),
                        "unacked": int(item.ack_unacked_broadcasts),
                        "age_steps": int(item.age_steps),
                        "changed_view": bool(item.changed_view),
                    }
                )
            return tuple(chosen)

    recorder = Recorder(build_comm_policy(spec["policy"], spec["params"]))
    common.protocol_env.run_episode(
        seed,
        directory=directory,
        label="cf_probe",
        policy=spec["policy"],
        policy_params={
            key: value
            for key, value in (
                token.split("=", 1) for token in str(spec["params"]).split(",") if token
            )
        },
        budget_bytes=common.tier_bytes(tier),
        information="isolated",
        delivery_probability=common.PRIMARY_DELIVERY_PROBABILITY,
        timing_mode=spec.get("mode", common.POLICY_MODE),
        controller=recorder,
        max_wait_steps=None,
        urgent_wait_steps=None,
        ack=True,
        ack_delay_ticks=common.ACK_DELAY_TICKS,
        ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
        read_timelines=False,
    )
    return offered


#: How many decision points per stream the diagnostic measures. Ten, at deterministic
#: positions among the partial-confirmation points. This is a *sample size*, not a threshold:
#: the brief fixes the thresholds (>= 20 per branch, >= 8 streams) and this number only
#: decides how much evidence is collected before they are applied. Measuring two points per
#: stream cannot distinguish "there is no trade-off" from "there was not enough measurement",
#: so the sample is wide enough for the gate to be able to fail for the right reason.
POINTS_PER_STREAM = 10


def select_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The partial-confirmation points where a send/hold choice is a real one.

    A point is interesting only when the ACK state *differs* across receivers or from the
    sender's own view: ``0 < confirmed_fraction < 1`` means some receivers have the view and
    some do not, which is precisely the state a retry rule keys on and the state a learned
    sender would have to interpret. Full confirmation leaves nothing to decide and no
    confirmation at all is the state a first emission is in.
    """
    partial = [
        point
        for point in points
        if 0.0 < point["confirmed_fraction"] < 1.0 and point["changed_view"]
    ]
    if not partial:
        return []
    if len(partial) <= POINTS_PER_STREAM:
        return partial
    positions = [
        index * (len(partial) - 1) // (POINTS_PER_STREAM - 1)
        for index in range(POINTS_PER_STREAM)
    ]
    return [partial[position] for position in dict.fromkeys(positions)]


def run(args: argparse.Namespace) -> int:
    directory = common.evaluation_root(args.tier) / "counterfactual"
    directory.mkdir(parents=True, exist_ok=True)
    streams = list(common.GATE_STREAMS)
    streams = [
        seed for index, seed in enumerate(streams) if index % args.shards == args.shard
    ]
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, seed in enumerate(streams, start=1):
        points = legal_decision_points(seed, directory=directory, tier=args.tier)
        chosen = select_points(points)
        if not chosen:
            rows.append({"seed": int(seed), "status": "no_informative_point"})
            continue
        free = run_unforced(seed, directory=directory, tier=args.tier)
        for point in chosen:
            send = run_branch(
                seed,
                at_step=point["step"],
                target=point["target"],
                action="send",
                directory=directory,
                tier=args.tier,
            )
            hold = run_branch(
                seed,
                at_step=point["step"],
                target=point["target"],
                action="hold",
                directory=directory,
                tier=args.tier,
            )
            # The branch that agrees with the rule's own choice must reproduce the unforced
            # run exactly: if it does not, the forcing wrapper is perturbing more than the
            # one step it claims to, and the measurement is worthless.
            agreeing = send if point["rule_sent"] else hold
            consistent = (
                agreeing["on_time"] == free["on_time"]
                and agreeing["total_bytes"] == free["total_bytes"]
            )
            rows.append(
                {
                    "seed": int(seed),
                    "status": "measured",
                    "at_step": int(point["step"]),
                    "target": point["target"],
                    "rule_sent": bool(point["rule_sent"]),
                    "confirmed_fraction": float(point["confirmed_fraction"]),
                    "unacked": int(point["unacked"]),
                    "consistent_with_unforced": bool(consistent),
                    "free": free,
                    "send": send,
                    "hold": hold,
                }
            )
        if index % 8 == 0 or index == len(streams):
            print(
                f"[counterfactual shard {args.shard}] {index}/{len(streams)} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    payload = {
        "tier": args.tier,
        "budget_bytes": common.tier_bytes(args.tier),
        "baseline_rule": BASELINE_RULE,
        "channel": {
            "delivery_probability": common.PRIMARY_DELIVERY_PROBABILITY,
            "ack_delay_ticks": common.ACK_DELAY_TICKS,
            "ack_delivery_probability": common.ACK_DELIVERY_PROBABILITY,
        },
        "shard": args.shard,
        "shards": args.shards,
        "rows": rows,
        # Every legal decision point the probe recorded, per stream. Gate 1 reads these: it
        # needs to know whether an unconfirmed attempt was ever followed by a change in the
        # sender's confirmation state, which is the same evidence the branches were chosen
        # from rather than a second measurement.
        "probe_points": [
            {"seed": int(seed), "points": legal_decision_points(seed, directory=directory, tier=args.tier)}
            for seed in streams
        ],
    }
    common.write_json(
        common.evaluation_root(args.tier) / f"counterfactual_shard{args.shard}.json", payload
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "measured": sum(1 for r in rows if r.get("status") == "measured"),
                "seconds": round(time.time() - started, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def load_rows(tier: str) -> list[dict[str, Any]]:
    root = common.evaluation_root(tier)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("counterfactual_shard*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    return rows


def _folded_probe_points(tier: str) -> list[dict[str, Any]]:
    """The per-stream probe records from every shard, de-duplicated by stream.

    The shards each carry the probe for their own streams, so folding them is a concatenation
    with a per-stream guard: two shards must never claim the same stream, or a stream's
    decision points would be counted twice and gate 1's evidence would be inflated.
    """
    root = common.evaluation_root(tier)
    folded: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("counterfactual_shard*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("probe_points", []):
            seed = int(entry["seed"])
            if seed in folded:
                raise RuntimeError(
                    f"{path}: stream {seed} already recorded by another shard; a duplicated "
                    "probe would inflate gate 1's evidence"
                )
            folded[seed] = entry["points"]
    return [{"seed": seed, "points": folded[seed]} for seed in sorted(folded)]


def report(args: argparse.Namespace) -> int:
    rows = [row for row in load_rows(args.tier) if row.get("status") == "measured"]
    if not rows:
        raise SystemExit("no measured counterfactual rows; run --mode run first")
    send_better = [
        row for row in rows if row["send"]["on_time"] > row["hold"]["on_time"]
    ]
    hold_better = [
        row for row in rows if row["hold"]["on_time"] > row["send"]["on_time"]
    ]
    ties = [row for row in rows if row["send"]["on_time"] == row["hold"]["on_time"]]
    send_cheaper = [
        row for row in rows if row["send"]["total_bytes"] < row["hold"]["total_bytes"]
    ]
    hold_cheaper = [
        row for row in rows if row["hold"]["total_bytes"] < row["send"]["total_bytes"]
    ]
    payload = {
        "tier": args.tier,
        "baseline_rule": BASELINE_RULE,
        "channel": {
            "delivery_probability": common.PRIMARY_DELIVERY_PROBABILITY,
            "ack_delay_ticks": common.ACK_DELAY_TICKS,
            "ack_delivery_probability": common.ACK_DELIVERY_PROBABILITY,
        },
        "decision_points": len(rows),
        # The internal consistency check: the branch that agrees with the rule's own choice
        # must reproduce the unforced run. Any mismatch means the forcing wrapper perturbed
        # more than the one step it claims to, and every number below it is void.
        "consistency": {
            "checked": sum(1 for row in rows if "consistent_with_unforced" in row),
            "consistent": sum(
                1 for row in rows if row.get("consistent_with_unforced") is True
            ),
        },
        "send_better_on_time": {
            "count": len(send_better),
            "streams": sorted({row["seed"] for row in send_better}),
        },
        "hold_better_on_time": {
            "count": len(hold_better),
            "streams": sorted({row["seed"] for row in hold_better}),
        },
        "tied_on_time": {"count": len(ties)},
        "send_cheaper_total_bytes": {"count": len(send_cheaper)},
        "hold_cheaper_total_bytes": {"count": len(hold_cheaper)},
        "gate_item_3": {
            "threshold_per_branch": MIN_PER_BRANCH,
            "threshold_streams": MIN_STREAMS_PER_BRANCH,
            "send_ok": len(send_better) >= MIN_PER_BRANCH
            and len({row["seed"] for row in send_better}) >= MIN_STREAMS_PER_BRANCH,
            "hold_ok": len(hold_better) >= MIN_PER_BRANCH
            and len({row["seed"] for row in hold_better}) >= MIN_STREAMS_PER_BRANCH,
        },
        "per_stream": rows,
        # Every legal decision point the probe recorded, per stream, folded in from the
        # shards. Gate 1 reads these: it needs to know whether an unconfirmed attempt was
        # ever followed by a change in the sender's confirmation state, which is the same
        # evidence the branches were chosen from rather than a second measurement.
        "probe_points": _folded_probe_points(args.tier),
    }
    common.write_json(common.json_path(f"COMM_ACK_VALUE_COUNTERFACTUAL_{args.tier}.json"), payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_stream"}, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("run", "report"), default="run")
    parser.add_argument("--tier", default=common.PRIMARY_TIER)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.mode == "run":
        return run(args)
    return report(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
