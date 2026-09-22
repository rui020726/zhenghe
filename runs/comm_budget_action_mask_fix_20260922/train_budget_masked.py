"""The training entry for the budget-masked sampler: same PPO, honest action set.

Round 5's trainer (`runs/comm_protocol_v2_20260920/train_protocol.py`) already had the corrected
on-policy sampler and the policy-owned timing rollouts; what it lacked was any notion that an
unaffordable SEND is not an action. This module changes exactly one thing -- which collector is
built -- and reuses the archived ``flatten`` / ``update`` / ``drawn_versus_logged`` / ``gae``
verbatim rather than copying them, on the grounds that a second copy of PPO is how the next
round finds a discrepancy between two trainers instead of a bug.

The reuse works because the new tape keeps the archived key ``free`` meaning "the candidates that
really had two legal actions" (`actionable`), so the likelihood weighting the archived update
performs is already over the right set. The identity guard is the other half: a checkpoint may only
be resumed by something that says the same trainer tag, mask schema and tape schema, and an
older checkpoint is welcome for a forward pass but never for continuing optimisation.

    python train_budget_masked.py --mode identity-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _directory in (
    str(ROOT),
    str(ROOT / "runs" / "comm_budget_pilot_20260919"),
    str(ROOT / "runs" / "comm_stability_20260919"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(ROOT / "runs" / "comm_protocol_v2_20260920"),
    str(ROOT / "runs" / "comm_ack_value_20260920"),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import protocol_env  # noqa: E402
import train_pilot  # noqa: E402
import train_protocol  # noqa: E402
from train_onpolicy import drawn_versus_logged, flatten, update  # noqa: E402
from budget_masked import (  # noqa: E402
    ACTION_MASK_SCHEMA,
    TAPE_SCHEMA,
    TRAINER_TAG,
    BudgetMaskedCollector,
    identity,
)

#: Reused, not restated: the probability bookkeeping and the PPO loop stay in one place.
#: ``update`` weights the likelihood by the tape's ``free`` set, which the masked collector
#: defines to be exactly the actionable set -- that equality is what makes the reuse honest.
gae = train_pilot.gae

__all__ = [
    "TRAINER_TAG",
    "ACTION_MASK_SCHEMA",
    "TAPE_SCHEMA",
    "collector_for",
    "run_episode",
    "collect_batch",
    "episode_diagnostics",
    "tape_invariants",
    "resume_identity",
    "load_forward_only",
    "flatten",
    "update",
    "gae",
    "drawn_versus_logged",
]


def collector_for(controller: Any, *, greedy: bool = False) -> BudgetMaskedCollector:
    """The fixed sampler, under the same "nothing is the transport's decision" rule as round 5.

    ``mask=False``/``forced=nothing_forced`` is policy-owned timing's existing claim: the silence
    bound emits nothing on its own, so every candidate the relay offers is the arm's own choice.
    The point of this round is that "offered" and "affordable" are different questions, and the
    budget mask is what separates them.
    """
    return BudgetMaskedCollector(
        controller,
        forced=train_protocol.nothing_forced,
        mask=False,
        greedy=greedy,
        budget_mask=True,
    )


def run_episode(
    controller: Any, seed: int, *, budget_bytes: int, directory: Path, label: str
) -> dict[str, Any]:
    """One policy-owned rollout whose tape records what the executor could have honoured."""
    collector = collector_for(controller)
    episode = protocol_env.rollout_episode(
        controller,
        seed,
        collector=collector,
        budget_bytes=budget_bytes,
        directory=directory,
        label=label,
        timing_mode=train_protocol.TIMING_MODE,
    )
    episode["controller"] = controller
    episode["timing"] = collector.timing()
    episode["collector"] = collector
    return episode


def collect_batch(
    controller: Any,
    seeds: list[int],
    *,
    budget_bytes: int,
    directory: Path,
    training_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for seed in seeds:
        episode = run_episode(
            controller,
            seed,
            budget_bytes=budget_bytes,
            directory=directory,
            label=f"mask_train{training_seed}",
        )
        episodes.append(episode)
        details.append(episode_diagnostics(episode["collector"], episode["row"]))
    return episodes, details


def episode_diagnostics(collector: BudgetMaskedCollector, row: dict[str, Any]) -> dict[str, Any]:
    """The archived diagnostics, plus the two counts this round exists to make visible."""
    base = train_protocol.episode_diagnostics(collector, row)
    timing = collector.timing()
    ledger = row["ledger"]
    base.update(
        {
            "trainer_tag": TRAINER_TAG,
            "action_mask_schema": ACTION_MASK_SCHEMA,
            "tape_schema": TAPE_SCHEMA,
            "actionable_candidates": int(timing["free_candidates"]),
            "masked_by_credit": int(timing["masked_by_credit"]),
            "masked_by_capacity": int(timing["masked_by_capacity"]),
            "actionable_share": timing["actionable_share"],
            # Execution-side truth: with the mask working, a policy send is never refused for
            # credit or capacity, because such a send is never asked for.
            "policy_budget_refusals": int(ledger["budget_refused_sends"]),
            "policy_capacity_refusals": int(ledger["capacity_refused_sends"]),
            "policy_refused_total": int(ledger.get("refused_policy", 0)),
        }
    )
    return base


# ------------------------------------------------------------------ the tape's own promises
def tape_invariants(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the tape the way the update relies on it, before any gradient is computed.

    ``free == actionable`` is what lets the archived PPO loop be reused: it weights the
    likelihood by ``free``, and this function is the reason that is the honest set.
    """
    violations: list[str] = []
    checked = 0
    for record in records:
        checked += 1
        if "actionable" not in record:
            violations.append(f"step {record.get('step')}: tape has no actionable set")
            continue
        if sorted(record["free"]) != sorted(record["actionable"]):
            violations.append(
                f"step {record.get('step')}: free {sorted(record['free'])} != actionable "
                f"{sorted(record['actionable'])}"
            )
        classes = (
            set(record["actionable"]),
            set(record["forced"]),
            set(record["masked"]),
        )
        if (
            classes[0] & classes[1]
            or classes[0] & classes[2]
            or classes[1] & classes[2]
        ):
            violations.append(f"step {record.get('step')}: candidate classes overlap")
        total = sum(len(group) for group in classes)
        if total != len(record["targets"]):
            violations.append(
                f"step {record.get('step')}: {total} classified != {len(record['targets'])} candidates"
            )
        for target in set(record["forced"]) | set(record["masked"]):
            if record["requested"].get(target):
                violations.append(
                    f"step {record.get('step')}: non-actionable candidate {target} was requested"
                )
        for target in record["masked"]:
            if record["actions"].get(target) != 0:
                violations.append(
                    f"step {record.get('step')}: masked candidate {target} was not held"
                )
    return {
        "tapes": checked,
        "violations": violations[:20],
        "violation_count": len(violations),
        "passed": not violations,
    }


# ------------------------------------------------------------------ checkpoint identity
def resume_identity(payload: dict[str, Any]) -> dict[str, str]:
    """What a resumable checkpoint must say, and the reason each field is checked."""
    expected = identity()
    required = {
        "trainer_tag": expected["trainer_tag"],
        "action_mask_schema": expected["action_mask_schema"],
        "tape_schema": expected["tape_schema"],
    }
    problems: dict[str, str] = {}
    for field, want in required.items():
        got = payload.get(field)
        if got != want:
            problems[field] = f"checkpoint says {got!r}, this trainer is {want!r}"
    if payload.get("timing_mode") != train_protocol.TIMING_MODE:
        problems["timing_mode"] = (
            f"checkpoint ran {payload.get('timing_mode')!r}, not {train_protocol.TIMING_MODE!r}"
        )
    return problems


def load_forward_only(path: Path, training_seed: int) -> dict[str, Any]:
    """Old checkpoints: weights for a forward pass, never an optimiser to continue.

    A controller trained while unaffordable SENDs were free actions has already been shaped by
    gradients on actions the transport would not honour, so resuming its optimiser would carry the
    defect into the "fixed" run. Loading the state dict alone is enough to compare behaviour and
    to seed a fresh run from the same initial-weight recipe.
    """
    payload = train_pilot.torch.load(path, map_location="cpu", weights_only=False)
    tag = str(payload.get("trainer_tag") or payload.get("collector") or "")
    controller = train_pilot.Controller(training_seed)
    controller.load_state_dict(payload["controller"])
    return {
        "path": str(path),
        "loaded": True,
        "recorded_trainer_tag": tag or None,
        "optimizer_resumable": False,
        "reason": (
            "the optimiser state is deliberately not restored: a run whose action space included "
            "unaffordable sends must not be continued as if it had not"
        ),
        "parameters": sum(int(value.numel()) for value in controller.parameters),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity-check",), default="identity-check")
    args = parser.parse_args(argv)
    import json

    print(json.dumps(identity(), ensure_ascii=False, indent=1))
    print(json.dumps({"reused_trainer": "comm_protocol_v2_20260920/train_protocol.py"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
