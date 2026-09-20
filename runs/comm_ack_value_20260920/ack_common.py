"""Fixtures for the ACK information-value round: the reverse channel as the only new dial.

This round changes exactly one thing about the experiment -- whether the sender can see
acknowledgements -- so everything else is imported from the rounds that fixed it: the 48
training and 24 validation streams, the locked byte credits, the frozen allocation arm and
checkpoints, the transcript readers, the paired bootstrap and the corrected on-policy sampler.
Restating any of it here would be a second definition that could drift from the numbers the
archived results quote.

Two things are new and both live here. ``ACK_ARMS`` is the strong ACK-rule table the
information-value gate runs, with every parameter written down before any result exists. And
``channel_parameters`` is the single place the reverse channel's declared delay, loss and seed
are stated, so the rule matrix, the counterfactual diagnostic and any later training all read
the same numbers instead of three copies of them.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"
STABILITY_ROUND = ROOT / "runs" / "comm_stability_20260919"
TIGHT_ROUND = ROOT / "runs" / "comm_tight_budget_20260919"
PROTOCOL_ROUND = ROOT / "runs" / "comm_protocol_v2_20260920"

for _directory in (
    str(ROOT),
    str(PILOT_ROUND),
    str(STABILITY_ROUND),
    str(TIGHT_ROUND),
    str(PROTOCOL_ROUND),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

pilot = importlib.import_module("pilot_common")
stability = importlib.import_module("stability_common")
attribution_metrics = importlib.import_module("attribution_metrics")
# Rounds 3-5's fixtures and corrected trainer are imported, not restated: the weight
# fingerprint, the paired bootstrap, the byte tiers and the on-policy sampler are shared with
# the archived results this round annotates.
tight_common = importlib.import_module("tight_common")
onpolicy = importlib.import_module("onpolicy")
train_pilot = importlib.import_module("train_pilot")
pilot_env = importlib.import_module("pilot_env")
# Round 5's fixtures: the arm tables, the equivalence pairs and the protocol-v2 episode runner
# with its ``timing_mode`` dial.
protocol_common = importlib.import_module("protocol_common")
protocol_env = importlib.import_module("protocol_env")

TRAIN_SEEDS = pilot.TRAIN_SEEDS
VALIDATION_SEEDS = pilot.VALIDATION_SEEDS
STREAM_STEPS = pilot.STREAM_STEPS
sha256 = pilot.sha256
write_json = pilot.write_json
metrics = pilot.metrics
locked_budget = stability.locked_budget
weight_fingerprint = tight_common.weight_fingerprint
paired_bootstrap = tight_common.paired_bootstrap
tensors = tight_common.tensors

#: The primary setting the brief freezes: uplink delivery probability 0.7 (lossy), lossless 1.0
#: only as a negative control. Stated once so the rule matrix, the counterfactual and any
#: training cannot drift apart.
PRIMARY_DELIVERY_PROBABILITY = 0.7
LOSSLESS_DELIVERY_PROBABILITY = 1.0

#: The primary byte credit. The brief fixes 29,300 B because the previous round's lossy
#: diagnostic already showed retry and send-once reordering under a binding credit.
PRIMARY_TIER = protocol_common.PRIMARY_TIER
SENSITIVITY_TIER = protocol_common.SENSITIVITY_TIER
BUDGET_TIERS = protocol_common.BUDGET_TIERS
BUDGETS_BYTES = protocol_common.BUDGETS_BYTES

#: The declared reverse-channel parameters. Delay and loss default to the uplink's own, which
#: is the contract's rule: the reverse channel is not a friendlier second uplink.
ACK_DELAY_TICKS = 0
ACK_DELIVERY_PROBABILITY = PRIMARY_DELIVERY_PROBABILITY
ACK_SEED_OFFSET = 0

#: The streams the information-value gate runs on. 48 training streams, never a validation
#: stream: the gate decides whether training is allowed at all, and spending the validation
#: block on that decision would leave nothing to validate with.
GATE_STREAMS: tuple[int, ...] = tuple(TRAIN_SEEDS)

LEGACY_MODE = protocol_common.LEGACY_MODE
POLICY_MODE = protocol_common.POLICY_MODE
ISOLATED_GOSSIP_INTERVAL = protocol_common.ISOLATED_GOSSIP_INTERVAL

TRAIN_OUT = HERE / "train"
EVALUATION_OUT = HERE / "evaluation"


def json_path(name: str) -> Path:
    return HERE / name


def read_json(name: str) -> Any:
    return json.loads(json_path(name).read_text(encoding="utf-8"))


def tier_bytes(tier: str) -> int:
    return protocol_common.tier_bytes(tier)


def evaluation_root(tier: str) -> Path:
    """One directory per (tier, channel) pair.

    Round 4 learned the hard way that the lifecycle writer names files after the arm and the
    seed only, so two settings sharing a directory overwrite each other. This round has two
    channel settings on top of two tiers, so the root carries both.
    """
    return EVALUATION_OUT / f"{tier}_p{int(round(100 * PRIMARY_DELIVERY_PROBABILITY))}"


def channel_parameters(
    *, delivery_probability: float = PRIMARY_DELIVERY_PROBABILITY
) -> dict[str, Any]:
    """The declared channel, in the form an episode runner takes."""
    return {
        "delivery_probability": float(delivery_probability),
        "ack": True,
        "ack_delay_ticks": ACK_DELAY_TICKS,
        "ack_delivery_probability": ACK_DELIVERY_PROBABILITY,
    }


def ack_arm_specs() -> dict[str, dict[str, Any]]:
    """The strong ACK rules, with every parameter written down before any result exists.

    Every rule uses the same queue, credit, ACK merge and billing: no rule gets a priority
    the others do not have. ``retry_until_acked_k`` and ``event_retry_until_acked_k`` read
    the sender's own confirmation state and nothing else, which is the information whose
    value this round tests.
    """
    return {
        # The reference: one report per accepted detection, no refresh traffic at all. It
        # cannot use an ACK, so it is the no-ACK side of the protocol comparison.
        "send_once": {"policy": "send_once", "params": "", "mode": POLICY_MODE},
        "always_hold": {"policy": "always_hold", "params": "", "mode": POLICY_MODE},
        "event": {"policy": "event", "params": "", "mode": POLICY_MODE},
        "event_priority": {"policy": "event_priority", "params": "", "mode": POLICY_MODE},
        **{
            f"periodic_{k}": {
                "policy": "periodic",
                "params": f"interval={k}",
                "mode": POLICY_MODE,
            }
            for k in (2, 5, 10, 20)
        },
        # Retry until every subscriber has confirmed the view, at most k attempts per view.
        **{
            f"retry_until_acked_{k}": {
                "policy": "retry_until_acked",
                "params": f"max_attempts={k}",
                "mode": POLICY_MODE,
            }
            for k in (2, 5, 10, 20)
        },
        # The event triggers, plus a retry budget for the views they have not yet confirmed.
        **{
            f"event_retry_until_acked_{k}": {
                "policy": "event_retry_until_acked",
                "params": f"max_attempts={k}",
                "mode": POLICY_MODE,
            }
            for k in (2, 5, 10, 20)
        },
        # Exponential backoff, one interval per arm, capped.
        **{
            f"backoff_{interval}": {
                "policy": "exponential_backoff",
                "params": f"base_interval={interval},max_interval=32",
                "mode": POLICY_MODE,
            }
            for interval in (1, 2, 4, 8, 16)
        },
        # Explicit deadline-aware tightening: retry spacing shrinks as urgency rises.
        "deadline_tightening_retry": {
            "policy": "deadline_tightening_retry",
            "params": "max_attempts=10",
            "mode": POLICY_MODE,
        },
    }


def arm_spec(arm: str) -> dict[str, Any]:
    specs = ack_arm_specs()
    if arm not in specs:
        raise ValueError(
            f"unknown ACK arm {arm!r}; available: {sorted(specs)}"
        )
    return dict(specs[arm])


def selectable_ack_arms() -> tuple[str, ...]:
    """Arms that may be chosen as the primary ACK comparator.

    ``always_hold`` is a control, not an opponent; ``send_once`` is the no-ACK reference the
    protocol comparison needs. Everything else competes.
    """
    return tuple(
        arm
        for arm in ack_arm_specs()
        if arm not in ("always_hold", "send_once")
    )


__all__ = [
    "HERE",
    "ROOT",
    "PILOT_ROUND",
    "STABILITY_ROUND",
    "TIGHT_ROUND",
    "PROTOCOL_ROUND",
    "pilot",
    "stability",
    "attribution_metrics",
    "tight_common",
    "onpolicy",
    "train_pilot",
    "pilot_env",
    "protocol_common",
    "protocol_env",
    "TRAIN_SEEDS",
    "VALIDATION_SEEDS",
    "GATE_STREAMS",
    "STREAM_STEPS",
    "PRIMARY_DELIVERY_PROBABILITY",
    "LOSSLESS_DELIVERY_PROBABILITY",
    "PRIMARY_TIER",
    "SENSITIVITY_TIER",
    "BUDGET_TIERS",
    "BUDGETS_BYTES",
    "ACK_DELAY_TICKS",
    "ACK_DELIVERY_PROBABILITY",
    "LEGACY_MODE",
    "POLICY_MODE",
    "ISOLATED_GOSSIP_INTERVAL",
    "TRAIN_OUT",
    "EVALUATION_OUT",
    "sha256",
    "write_json",
    "json_path",
    "read_json",
    "tier_bytes",
    "evaluation_root",
    "channel_parameters",
    "ack_arm_specs",
    "arm_spec",
    "selectable_ack_arms",
    "locked_budget",
    "weight_fingerprint",
    "paired_bootstrap",
    "metrics",
    "tensors",
]
