"""Fixtures for the stability round: the same frozen system, read from the rounds that fixed it.

This round changes nothing about the environment, the allocation arm, the budget or the
transport. It imports the pilot round's fixtures -- which themselves refuse to load if a
checkpoint has drifted -- and adds only what round 3 needs: its own output directory, the list
of arms to attribute a gain to, and the checkpoint inventory it will resume from.

Seed blocks are inherited, not re-cut. The 48 training streams and the 24 validation streams
keep the purpose they were given in round 2, and the final test block stays untouched and
registered elsewhere (see ``data_split_audit.py``).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"

for _directory in (str(ROOT), str(PILOT_ROUND)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

# The pilot round's fixtures pin the frozen block, the checkpoints and the streams. Importing
# it is the verification step: a drifted checkpoint or a moved stream fails at import.
pilot = importlib.import_module("pilot_common")

TRAIN_SEEDS = pilot.TRAIN_SEEDS
VALIDATION_SEEDS = pilot.VALIDATION_SEEDS
STREAM_STEPS = pilot.STREAM_STEPS
configuration_for = pilot.configuration_for
write_json = pilot.write_json
sha256 = pilot.sha256
metrics = pilot.metrics
CHECKPOINTS = pilot.CHECKPOINTS
CHECKPOINT_SHA256 = pilot.CHECKPOINT_SHA256
RECORD_CONFIGURATION = pilot.RECORD_CONFIGURATION

#: The pilot's locked credit, unchanged. Re-read rather than copied so a re-calibration would
#: show up here as a difference against the recorded number instead of a silent drift.
def locked_budget(name: str | None = None) -> int:
    locked = json.loads((pilot.HERE / "COMM_BUDGET_LOCKED.json").read_text(encoding="utf-8"))
    if name is None:
        calibration = json.loads(
            (pilot.HERE / "COMM_BUDGET_CALIBRATION.json").read_text(encoding="utf-8")
        )
        name = calibration["tier_selection"]["selected_tier"]
    return int(locked["budgets_bytes"][name])


_calibration = json.loads(
    (pilot.HERE / "COMM_BUDGET_CALIBRATION.json").read_text(encoding="utf-8")
)
BUDGET_TIER = str(_calibration["tier_selection"]["selected_tier"])
BUDGET_BYTES = locked_budget(BUDGET_TIER)

#: Where the pilot's resumable state lives. Round 3 continues these three runs; it does not
#: start new ones, and it never overwrites a pilot checkpoint.
PILOT_CHECKPOINTS = {seed: PILOT_ROUND / "train" / f"pilot_seed{seed}.pt" for seed in (1, 2, 3)}
TRAIN_OUT = HERE / "train"

#: Every arm the attribution compares, in one place, so the cost-effect table cannot quietly
#: drop the inconvenient entry. ``kind`` says how to build it.
#:
#: The initialisation control is per training seed: the question round 3 asks is whether a
#: trained network sends better than *its own untrained weights* did, and one shared random
#: network could answer a different question and look like it answered this one. The weights are
#: rebuilt from ``Controller(seed)``, which seeds torch and builds the same unfitted MLP --
#: reproducible by rule, and not a replay of an archived file, because round 2 saved no
#: pre-training checkpoint.
ARMS: tuple[dict[str, str], ...] = (
    *(
        {"arm": name, "kind": "rule", "policy": policy, "params": params}
        for name, policy, params in (
            ("immediate", "immediate", ""),
            ("hold_to_bound", "hold_to_bound", ""),
            ("random_send", "random_send", "send_probability=0.5,seed=7"),
            ("periodic_5", "periodic", "interval=5"),
            ("event", "event", ""),
            ("event_priority", "event_priority", ""),
        )
    ),
    {"arm": "event_unbudgeted", "kind": "rule_ceiling", "policy": "event", "params": ""},
    # Not a competitor: "immediate" inside the pilot's protocol is not "send once", because the
    # shared bound makes it refresh anything that has sat unsaid for 10 steps. This is the
    # historical arm itself, with no bound and no credit, so the cost-effect curve has its left
    # end on it rather than on a number from a round that measured it differently.
    {"arm": "send_once_no_bound", "kind": "rule_ceiling", "policy": "immediate", "params": "", "bound": "none"},
    *(
        {"arm": f"init_seed{seed}", "kind": "init", "training_seed": str(seed), "params": ""}
        for seed in (1, 2, 3)
    ),
    *(
        {"arm": f"learned_seed{seed}", "kind": "checkpoint", "training_seed": str(seed), "params": ""}
        for seed in (1, 2, 3)
    ),
)

#: The untrained initialisation is reproducible by rule -- ``Controller(seed)`` seeds torch and
#: builds an unfitted MLP -- but it was never archived. Reconstruction is therefore labelled as
#: such wherever it appears; it is not a bit-for-bit replay of a saved file.
INIT_IS_RECONSTRUCTION = True


def transport_bounded_collector(controller: Any, **options: Any):
    """A collector whose returned selection is only what the model itself asked for.

    Round 2's collector appended the mask's must-send targets to the list it handed back, so
    those emissions were billed as policy decisions and -- because the learned arm supplies its
    own order while a rule's forced sends are appended at the back -- they were billed *before*
    the equivalent sends of every rule arm. Same set of packets either way, different queue
    position when the credit ran out, and a reason split that could not be trusted.

    Dropping them from the selection costs nothing in safety: the transport applies the bound to
    every arm and emits anything overdue, so a held change still cannot escape the episode.
    """
    import train_pilot

    class TransportBounded(train_pilot.Collector):
        def choose(self, observation: Any) -> Any:
            chosen, actions, free, log_prob, value = super().choose(observation)
            if not self.mask or not chosen:
                return chosen, actions, free, log_prob, value
            overdue = {item.target_id for item in observation.items if train_pilot.forced(item)}
            return [target for target in chosen if target not in overdue], actions, free, log_prob, value

    return TransportBounded(controller, **options)


def arm_rows() -> list[str]:
    return [spec["arm"] for spec in ARMS]


def json_path(name: str) -> Path:
    return HERE / name


def read_json(name: str) -> Any:
    return json.loads(json_path(name).read_text(encoding="utf-8"))


__all__ = [
    "HERE",
    "ROOT",
    "PILOT_ROUND",
    "pilot",
    "TRAIN_SEEDS",
    "VALIDATION_SEEDS",
    "STREAM_STEPS",
    "configuration_for",
    "write_json",
    "sha256",
    "metrics",
    "CHECKPOINTS",
    "CHECKPOINT_SHA256",
    "RECORD_CONFIGURATION",
    "locked_budget",
    "BUDGET_TIER",
    "BUDGET_BYTES",
    "PILOT_CHECKPOINTS",
    "TRAIN_OUT",
    "ARMS",
    "arm_rows",
    "transport_bounded_collector",
    "json_path",
    "read_json",
]
