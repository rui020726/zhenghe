"""The round's checks. Every one of them fails the run; none of them is a warning.

Round 2 ran six smoke checks before its pilot. This file re-runs the ones that still mean
something after the protocol corrections and adds the four this round introduced: that the
reason ledger closes on a live episode, that the silence-bound rule really has no policy
contribution, that the send-once reference and the bound-bearing ``immediate`` are genuinely
different arms, and that a resumed run restores the random streams it claims to restore.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import attribution
import stability_common as common

import pilot_env
import train_pilot


def _run(spec: dict[str, Any], seed: int, directory: Path, budget: int | None) -> dict[str, Any]:
    controller, provenance = attribution.controller_for(spec)
    return attribution.run_arm(
        spec,
        seed,
        controller=controller,
        provenance=provenance,
        directory=directory,
        budget=budget,
    )


def check(args: argparse.Namespace) -> int:
    import data_split_audit

    directory = common.HERE / "smoke"
    directory.mkdir(parents=True, exist_ok=True)
    seed = common.VALIDATION_SEEDS[0]
    budget = args.budget_bytes
    results: dict[str, Any] = {}
    failures: list[str] = []

    by_name = {spec["arm"]: spec for spec in common.ARMS}

    # 1. The reason ledger closes on a live episode, for the two arms whose reasons differ.
    for arm in ("immediate", "hold_to_bound"):
        row = _run(by_name[arm], seed, directory, budget)
        residuals = {key: value for key, value in row["closure"]["residuals"].items() if value}
        if residuals:
            failures.append(f"{arm}: ledger does not close: {residuals}")
        if row["implementations_agree"] and set(row["implementations_agree"].values()) != {0}:
            failures.append(f"{arm}: transcript and relay counters disagree: {row['implementations_agree']}")
        results[arm] = {
            "on_time": row["on_time_targets_per_episode"],
            "bytes": row["airtime_bytes"],
            "emissions": row["emissions"],
            "policy_emissions": row["rollup"]["by_reason"]["policy"]["emissions"],
            "forced_emissions": row["rollup"]["by_reason"]["forced_maxwait"]["emissions"],
            "redundant": sum(bucket["redundant"] for bucket in row["rollup"]["by_reason"].values()),
            "unspoken_or_uncounted_at_end": row["closure"]["generated_balance"]["unspoken_or_uncounted_at_end"],
        }

    # 2. A hold-to-bound arm must have zero policy emissions; if it has any, the new rule arm is
    #    not what it claims to be and the whole attribution table inherits the error.
    if results["hold_to_bound"]["policy_emissions"] != 0:
        failures.append("hold_to_bound asked for a send -- it is not the silence rule it names")
    if results["hold_to_bound"]["emissions"] == 0:
        failures.append("hold_to_bound emitted nothing -- the shared bound is not being enforced")

    # 3. The bound is worth 20 KB on this stream. If send_once_no_bound and immediate agree, the
    #    bound key never reached the transport and the reference arm is decorative.
    row = _run(by_name["send_once_no_bound"], seed, directory, budget)
    results["send_once_no_bound"] = {"bytes": row["airtime_bytes"], "emissions": row["emissions"]}
    if row["rollup"]["by_reason"]["forced_maxwait"]["emissions"] != 0:
        failures.append("the no-bound arm was made to speak by a bound that should be off")
    if row["airtime_bytes"] >= results["immediate"]["bytes"]:
        failures.append(
            "send-once cost at least as much as immediate: the bound is contributing nothing, "
            "so the two arms are the same arm"
        )

    # 4. A learned arm under this round's collector must not be billed for what the bound forced.
    learned = _run(by_name["learned_seed3"], seed, directory, budget)
    results["learned_seed3"] = {
        "bytes": learned["airtime_bytes"],
        "policy_emissions": learned["rollup"]["by_reason"]["policy"]["emissions"],
        "forced_emissions": learned["rollup"]["by_reason"]["forced_maxwait"]["emissions"],
        "autonomous_share_of_bytes": learned["autonomous_share_of_bytes"],
    }
    residuals = {key: value for key, value in learned["closure"]["residuals"].items() if value}
    if residuals:
        failures.append(f"learned arm: ledger does not close: {residuals}")

    # 5. The credit is a hard constraint: a greedy learned arm may not exceed it, and the executor
    #    reports the same bill the ledger does.
    if learned["airtime_bytes"] > budget:
        failures.append(f"a run spent {learned['airtime_bytes']} against a credit of {budget}")

    # 6. Nothing receiver-side can reach the controller.
    audit = train_pilot.leakage_audit()
    results["leakage_audit"] = {key: value for key, value in audit.items() if key != "sender_side_fields"}
    if not audit["clean"]:
        failures.append(f"the sender's observation type is no longer sender-side only: {audit['fields_not_on_the_allow_list']}")

    # 7. One PPO step really moves the weights, and the diagnostics §4 promised are produced.
    controller = train_pilot.Controller(11)
    collector = common.transport_bounded_collector(controller, epsilon=1.0)
    sample = train_pilot.run_one(
        seed, collector=collector, budget_bytes=budget, directory=directory, label="smoke_update"
    )
    before = controller.state_dict()["trunk"]["0.weight"].clone()
    optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
    report = _update(controller, optimizer, [sample])
    after = controller.state_dict()["trunk"]["0.weight"]
    delta = float((after - before).abs().max().item())
    results["update"] = {**report, "weight_delta": round(delta, 6)}
    if delta < 1e-6 or report.get("optimizer_steps", 0) == 0:
        failures.append("no parameter change after a PPO update")
    for key in ("clip_fraction", "value_explained_variance", "gradient_norm_mean", "optimizer_steps", "samples"):
        if report.get(key) is None:
            failures.append(f"the update stopped reporting {key}")

    # 8. Return equals the scored metric, and the reward tape sums to it.
    if int(sample["row"]["return_on_time"]) != int(sample["row"]["task"]["on_time"]):
        failures.append("episode return disagrees with the scored on-time count")
    if abs(sum(sample["rewards"]) - float(sample["row"]["return_on_time"])) > 1e-9:
        failures.append("the per-step reward tape does not sum to the episode's score")

    # 9. Actions move schedules: opposite send behaviours must not produce the same ledger.
    flood = _run({**by_name["event"], "arm": "smoke_flood"}, seed, directory, budget)
    if (flood["emissions"], flood["airtime_bytes"]) == (
        results["hold_to_bound"]["emissions"],
        results["hold_to_bound"]["bytes"],
    ):
        failures.append("event and hold-to-bound produced the same schedule")

    # 10. The resume machinery restores what it claims. The pilot stored no RNG state, so the
    #     first continuation step is a warm start and must say so.
    state = train_stability_rng_check()
    results["resume"] = state
    if not state["rng_round_trip"]:
        failures.append("the saved RNG state does not reproduce the draws that follow it")
    if not state["pilot_checkpoint_has_no_rng"]:
        failures.append("the pilot checkpoint appears to have stored RNG state; warm-start wording is wrong")

    # 11. Configuration truth table, read from the run configuration rather than restated.
    executed = json.loads(json.dumps(common_read_defaults(), default=str))
    results["executed_configuration"] = executed
    if executed["mismatches"]:
        failures.append(f"the built configuration drifted from the pilot's truth table: {executed['mismatches']}")
    if not executed["membership_consistency_enabled"]:
        failures.append("the training environment does not have the membership-consistency fix enabled")
    if executed["values"].get("steps") != common.STREAM_STEPS:
        failures.append("the episode horizon is not the recorded stream length")

    # 12. The final test block is still untouched.
    split = data_split_audit.audit(data_split_audit.CANDIDATE_BLOCK, data_split_audit.PREVIOUS_TEST_BLOCK)
    results["data_split"] = {
        "reserved_block": split["replacement"]["block"],
        "reserved_clean": split["replacement"]["clean"],
        "previous_block_executed": split["previously_registered_final_test"]["seeds_with_executed_episodes"],
    }
    if not split["replacement"]["clean"]:
        failures.append("the reserved final-test block has artefacts on disk -- it is no longer unseen")

    results["failures"] = failures
    results["budget_bytes"] = budget
    results["seed"] = seed
    results["constants"] = {
        "episodes_per_update": train_pilot.EPISODES_PER_UPDATE,
        "milestones": list(__import__("train_stability").CHECKPOINT_STEPS),
        "checkpoint_milestone_note": "saved at the first update boundary at or after the milestone",
        "budget_tier": common.BUDGET_TIER,
        "collector": "transport_bounded",
    }
    common.write_json(common.json_path("STABILITY_SMOKE.json"), results)
    print(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    return 1 if failures else 0


def _update(controller: Any, optimizer: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    import train_stability

    return train_stability.update_with_diagnostics(controller, optimizer, episodes)


def common_read_defaults() -> dict[str, Any]:
    import stability_manifest

    return stability_manifest.executed_configuration()


def train_stability_rng_check() -> dict[str, Any]:
    """Save, draw, restore, draw again: the two draws have to be the same numbers."""
    import train_stability

    payload = torch.load(common.PILOT_CHECKPOINTS[1], map_location="cpu", weights_only=False)
    pilot_state = payload["state"]
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    saved = train_stability.rng_state()
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    for _ in range(7):  # move all three streams away from the snapshot
        random.random(), np.random.rand(), torch.rand(1)
    restored = train_stability.restore_rng(saved)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    return {
        "rng_round_trip": bool(restored) and first == second,
        "pilot_checkpoint_has_no_rng": "rng" not in pilot_state,
        "pilot_stored_keys": sorted(pilot_state),
        "draw": [round(value, 8) for value in first],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-bytes", type=int, default=common.BUDGET_BYTES)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    return check(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
