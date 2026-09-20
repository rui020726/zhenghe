"""Fixtures for the protocol-v2 round: which arm asks for what, and on which protocol.

This round changes exactly one thing about the transport -- whether the silence bound is the
transport's own forced action or a request the arm makes -- so everything else is imported from
the rounds that fixed it rather than restated: the 48 training and 24 validation streams, the two
locked byte credits, the frozen allocation arm and checkpoints, the transcript readers and the
closed identities from round 3, and the on-policy sampler and likelihood helpers from round 4.

Two arm tables live here on purpose. ``LEGACY_ARMS`` is the round-4 set run under
``legacy_shared_bound``, where the transport speaks for everyone at 3/10 steps; ``V2_ARMS`` is the
same behaviour expressed as requests under ``policy_owned_timing``. The pairs with
``legacy_equivalent`` set are this round's equivalence claim: they must emit the same packets at
the same steps in the same order, for the same money, and leave the tasks in the same state.

The two bound-named rules take their 3/10 constants from ``BOUND`` below, which is also what
``pilot_env`` hands the legacy transport, so a name cannot drift away from the numbers it quotes.
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

for _directory in (
    str(ROOT),
    str(PILOT_ROUND),
    str(STABILITY_ROUND),
    str(TIGHT_ROUND),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

pilot = importlib.import_module("pilot_common")
stability = importlib.import_module("stability_common")
attribution_metrics = importlib.import_module("attribution_metrics")
# Round 4's fixtures and corrected trainer are imported, not restated: the weight fingerprint,
# the paired bootstrap and the byte tiers are shared with the archived results they annotate, and
# a re-implementation here could drift from the numbers those results quote.
tight_common = importlib.import_module("tight_common")
onpolicy = importlib.import_module("onpolicy")
train_pilot = importlib.import_module("train_pilot")
pilot_env = importlib.import_module("pilot_env")

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

#: The declared silence bound, as a constant of the protocol rather than a magic number in a
#: policy body. The legacy transport enforces these; the new protocol only lets an arm ask.
BOUND_MAX_WAIT_STEPS = int(pilot_env.MAX_WAIT_STEPS)
BOUND_URGENT_WAIT_STEPS = int(pilot_env.MAX_WAIT_WHEN_URGENT_STEPS)
BOUND_URGENCY_TIME = float(pilot_env.URGENCY_STEPS)
BOUND_PARAMS = (
    f"max_wait_steps={BOUND_MAX_WAIT_STEPS},"
    f"urgent_wait_steps={BOUND_URGENT_WAIT_STEPS},"
    f"urgency_time={BOUND_URGENCY_TIME}"
)

LEGACY_MODE = "legacy_shared_bound"
POLICY_MODE = "policy_owned_timing"

#: Primary budget first (29,300 B), the 75 % tier kept as a sensitivity control. Both are read
#: from the locked calibration file, never typed in.
PRIMARY_TIER = "25pct"
SENSITIVITY_TIER = "75pct"
BUDGET_TIERS: tuple[str, ...] = (PRIMARY_TIER, SENSITIVITY_TIER)
BUDGETS_BYTES: dict[str, int] = {tier: locked_budget(tier) for tier in BUDGET_TIERS}

#: Streams the small gates run on. 8 training streams, never a validation stream: the validation
#: block stays reserved for the frozen comparison, and an equivalence check on it would spend it.
GATE_STREAMS: tuple[int, ...] = tuple(TRAIN_SEEDS[:8])

#: The same isolation switch the pilot used: a gossip interval above the step count turns the
#: uplink into the only message path, so a send policy can be judged on what it actually delivers.
ISOLATED_GOSSIP_INTERVAL = int(pilot_env.ISOLATED_GOSSIP_INTERVAL)

TRAIN_OUT = HERE / "train"
EVALUATION_OUT = HERE / "evaluation"


def json_path(name: str) -> Path:
    return HERE / name


def read_json(name: str) -> Any:
    return json.loads(json_path(name).read_text(encoding="utf-8"))


def tier_bytes(tier: str) -> int:
    if tier not in BUDGET_TIERS:
        raise ValueError(
            f"unknown budget tier {tier!r}; this round compares {BUDGET_TIERS} and does not "
            "invent a third one"
        )
    return BUDGETS_BYTES[tier]


def evaluation_root(tier: str) -> Path:
    """One directory per tier. Round 4 learned this the hard way: the lifecycle writer names its
    files after the arm and the seed only, so two tiers sharing a directory overwrite each other.
    """
    return EVALUATION_OUT / tier


# ------------------------------------------------------------------ arm tables
# Every entry: arm label, the policy object it asks for, its parameters, and how the transport is
# configured. ``bound`` says whether the legacy transport still enforces the silence bound;
# ``ceiling`` says the arm runs with no byte credit at all, which makes it a reference and never
# a competitor. Both flags exist so an out-of-protocol arm cannot be pooled by accident.
LEGACY_ARMS: dict[str, dict[str, Any]] = {
    "immediate": {"policy": "immediate", "params": ""},
    "hold_to_bound": {"policy": "hold_to_bound", "params": ""},
    "send_once_no_bound": {
        "policy": "immediate",
        "params": "",
        "bound": "none",
        "note": "the historical relay: one emission per accepted detection, nothing else",
    },
    "periodic_2": {"policy": "periodic", "params": "interval=2"},
    "periodic_5": {"policy": "periodic", "params": "interval=5"},
    "periodic_10": {"policy": "periodic", "params": "interval=10"},
    "periodic_20": {"policy": "periodic", "params": "interval=20"},
    "event": {"policy": "event", "params": ""},
    "event_priority": {"policy": "event_priority", "params": ""},
    "random_send": {"policy": "random_send", "params": "send_probability=0.5,seed=7"},
}

V2_ARMS: dict[str, dict[str, Any]] = {
    # The two equivalence claims. ``first_send_plus_bound_3_10`` is what ``immediate`` did: it
    # asks for every accepted detection at once and, in a second band behind its own requests,
    # for anything the declared bound has run out on. ``bound_only_3_10`` is what
    # ``hold_to_bound`` did: it asks for nothing but the timer band.
    "first_send_plus_bound_3_10": {
        "policy": "first_send_plus_bound_3_10",
        "params": BOUND_PARAMS,
        "mode": POLICY_MODE,
        "legacy_equivalent": "immediate",
    },
    "bound_only_3_10": {
        "policy": "bound_only_3_10",
        "params": BOUND_PARAMS,
        "mode": POLICY_MODE,
        "legacy_equivalent": "hold_to_bound",
    },
    # The arm that was previously only expressible by stepping outside the protocol. It is a
    # legal, complete strategy here, and the brief requires it in the main comparison.
    "send_once": {
        "policy": "send_once",
        "params": "",
        "mode": POLICY_MODE,
        "legacy_equivalent": "send_once_no_bound",
    },
    **{
        f"periodic_{k}": {
            "policy": "periodic",
            "params": f"interval={k}",
            "mode": POLICY_MODE,
        }
        for k in (2, 5, 10, 20)
    },
    "event": {"policy": "event", "params": "", "mode": POLICY_MODE},
    "event_priority": {"policy": "event_priority", "params": "", "mode": POLICY_MODE},
    # Negative controls. ``always_hold`` is only a control because this protocol lets it be one:
    # under the legacy transport the bound speaks for it.
    "always_hold": {"policy": "always_hold", "params": "", "mode": POLICY_MODE},
    "random_send": {
        "policy": "random_send",
        "params": "send_probability=0.5,seed=7",
        "mode": POLICY_MODE,
    },
}

#: The out-of-protocol reference: the event rule with no credit. Reported, never selected.
CEILING_ARMS: dict[str, dict[str, Any]] = {
    "event_unbudgeted": {"policy": "event", "params": "", "mode": POLICY_MODE, "ceiling": True},
}

#: Comparators that may be chosen as the primary opponent (总纲 §8.1, 执行指令 §4). ``always_hold``
#: and ``random_send`` are controls, not opponents; ``event_unbudgeted`` is a ceiling.
SELECTABLE_V2_ARMS = tuple(
    arm
    for arm, spec in V2_ARMS.items()
    if arm not in ("always_hold", "random_send")
)


def arm_spec(table: str, arm: str) -> dict[str, Any]:
    sources = {"legacy": LEGACY_ARMS, "v2": V2_ARMS, "ceiling": CEILING_ARMS}
    if table not in sources:
        raise ValueError(f"unknown arm table {table!r}; available {tuple(sources)}")
    if arm not in sources[table]:
        raise ValueError(f"unknown arm {arm!r} in table {table!r}; available {sorted(sources[table])}")
    return dict(sources[table][arm])


__all__ = [
    "HERE",
    "ROOT",
    "PILOT_ROUND",
    "STABILITY_ROUND",
    "TIGHT_ROUND",
    "pilot",
    "stability",
    "attribution_metrics",
    "tight_common",
    "onpolicy",
    "train_pilot",
    "pilot_env",
    "TRAIN_SEEDS",
    "VALIDATION_SEEDS",
    "GATE_STREAMS",
    "STREAM_STEPS",
    "BOUND_MAX_WAIT_STEPS",
    "BOUND_URGENT_WAIT_STEPS",
    "BOUND_URGENCY_TIME",
    "BOUND_PARAMS",
    "LEGACY_MODE",
    "POLICY_MODE",
    "PRIMARY_TIER",
    "SENSITIVITY_TIER",
    "BUDGET_TIERS",
    "BUDGETS_BYTES",
    "LEGACY_ARMS",
    "V2_ARMS",
    "CEILING_ARMS",
    "SELECTABLE_V2_ARMS",
    "ISOLATED_GOSSIP_INTERVAL",
    "TRAIN_OUT",
    "EVALUATION_OUT",
    "arm_spec",
    "tier_bytes",
    "evaluation_root",
    "locked_budget",
    "weight_fingerprint",
    "paired_bootstrap",
    "metrics",
    "write_json",
    "json_path",
    "read_json",
    "sha256",
]
