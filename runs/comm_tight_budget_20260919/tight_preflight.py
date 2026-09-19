"""This round's checks. Every one fails the run; none of them is a warning.

The round makes two claims -- that the collector now logs the distribution it draws from, and that
a resumed run resumes the *whole* run -- and both are checkable in a couple of minutes, so they are
checked here rather than argued in prose.

One check is worth naming out loud: :func:`check` deliberately feeds the update a tape whose logged
probability does not describe the action, and requires the trainer to refuse. A guard nobody has
seen fire is a comment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tight_common as common

import onpolicy
import pilot_env
import train_onpolicy
import train_pilot

#: One update is 4 episodes x 1000 steps. Two updates is what the resume check needs.
UPDATES = 2
STEPS_PER_UPDATE = train_onpolicy.STEPS_PER_EPISODE * train_pilot.EPISODES_PER_UPDATE


def _tape_of(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return episode["decisions"]


def check(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "smoke"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed = common.TRAIN_SEEDS[0]
    budget = args.budget_bytes
    results: dict[str, Any] = {}
    failures: list[str] = []

    # 1. On a live rollout, the drawn actions agree with the logged distribution, and the tape
    #    recomputes to the number stored beside it.
    controller = train_pilot.Controller(1)
    episode = train_onpolicy.run_episode(
        controller, seed, budget_bytes=budget, directory=directory / "sampler", label="smoke_sampler"
    )
    measured = train_onpolicy.drawn_versus_logged(controller, _tape_of(episode))
    results["sampler"] = {"alignment": measured, "timing": episode["collector"].timing()}
    if measured["records"] == 0:
        failures.append("the rollout recorded no decisions, so the sampler check proved nothing")
    if measured["max_abs_gap"] > 1e-4:
        failures.append(f"the logged probability does not describe the draw: {measured}")

    # 2. The guard fires. Flip an action on a recorded tape without touching its logged
    #    probability -- exactly the mixture's signature -- and the update must refuse it.
    corrupted = json.loads(json.dumps(_tape_of(episode), default=str))
    flipped = 0
    for record in corrupted:
        for target, action in record["actions"].items():
            if target in record["free"]:
                record["actions"][target] = int(not action)
                flipped += 1
                break
        if flipped:
            break
    if not flipped:
        failures.append("the rollout had no free action to corrupt; the guard went untested")
    else:
        try:
            corrupt_controller = train_pilot.Controller(1)
            optimizer = torch.optim.Adam(corrupt_controller.parameters, lr=train_onpolicy.LEARNING_RATE)
            train_onpolicy.update(
                corrupt_controller, optimizer, [dict(episode, decisions=corrupted, controller=corrupt_controller)]
            )
            failures.append("a tape whose actions contradict its logged probability was accepted")
        except RuntimeError as error:
            results["guard_refused_tape"] = str(error)[:200]

    # 3. A hard credit at the tight tier: neither a rule nor a learned arm may exceed it, and the
    #    executor's bill and the ledger's airtime must be the same number.
    for name, spec in (("hold_to_bound", {"arm": "hold_to_bound", "policy": "hold_to_bound", "params": ""}),
                       ("immediate", {"arm": "immediate", "policy": "immediate", "params": ""})):
        row = pilot_env.run_episode(
            seed, directory=directory / "budget", label=name, policy=spec["policy"],
            budget_bytes=budget, information="isolated", delivery_probability=1.0,
        )
        results[f"budget_{name}"] = {
            "spent": int(row["spent_bytes"]),
            "credit": budget,
            "ledger_airtime": int(row["ledger"]["airtime_bytes"]),
            "refusals": int(row["ledger"]["budget_refused_sends"]),
        }
        if int(row["spent_bytes"]) > budget:
            failures.append(f"{name} spent {row['spent_bytes']} against a credit of {budget}")
        if int(row["spent_bytes"]) != int(row["ledger"]["airtime_bytes"]):
            failures.append(f"{name}: the bill and the ledger's airtime are different numbers")

    # 4. The bound is alive: a policy that never asks still gets voice, and the reason split closes.
    silent = onpolicy.OnPolicyCollector(None, forced=train_pilot.forced, never_send=True)
    silent_row = train_pilot.run_one(
        seed, collector=silent, budget_bytes=budget, directory=directory / "bound", label="smoke_bound"
    )
    ledger = silent_row["row"]["ledger"]
    results["bound"] = {
        "emissions": int(ledger["emissions"]),
        "forced": int(ledger["emissions_forced_maxwait"]),
        "policy": int(ledger["emissions_policy"]),
        "reason_split_closes": int(ledger["emissions_policy"]) + int(ledger["emissions_forced_maxwait"])
        + int(ledger["emissions_other"])
        == int(ledger["emissions"]),
    }
    if ledger["emissions_forced_maxwait"] == 0:
        failures.append("the silence bound never fired; the shared protocol is not being enforced")
    if ledger["emissions_policy"] != 0:
        failures.append("an arm that never asked for the air still got policy emissions")
    if not results["bound"]["reason_split_closes"]:
        failures.append(f"the reason split does not close: {results['bound']}")

    # 5. One update moves the actor, reports the promised diagnostics, and the reward tape still
    #    sums to the scored metric.
    trained = train_pilot.Controller(11)
    sample = train_onpolicy.run_episode(
        trained, seed, budget_bytes=budget, directory=directory / "update", label="smoke_update"
    )
    before = trained.state_dict()["action_head"]["weight"].clone()
    value_before = trained.state_dict()["value_head"]["weight"].clone()
    optimizer = torch.optim.Adam(trained.parameters, lr=train_onpolicy.LEARNING_RATE)
    report = train_onpolicy.update(trained, optimizer, [sample])
    after = trained.state_dict()["action_head"]["weight"]
    results["update"] = {
        **{key: value for key, value in report.items() if key != "likelihood_alignment"},
        "actor_delta": round(float((after - before).abs().max().item()), 6),
        "critic_delta": round(
            float((trained.state_dict()["value_head"]["weight"] - value_before).abs().max().item()), 6
        ),
    }
    if report.get("optimizer_steps", 0) == 0 or results["update"]["actor_delta"] < 1e-7:
        failures.append(f"a PPO update changed nothing: {results['update']}")
    for key in ("clip_fraction", "value_explained_variance", "gradient_norm_mean", "samples", "free_candidate_samples"):
        if report.get(key) is None:
            failures.append(f"the update stopped reporting {key}")
    if int(sample["row"]["return_on_time"]) != int(sample["row"]["task"]["on_time"]):
        failures.append("episode return disagrees with the scored on-time count")
    if abs(sum(sample["rewards"]) - float(sample["row"]["return_on_time"])) > 1e-9:
        failures.append("the reward tape does not sum to the episode's score")

    # 6. A step where the bound has taken everything over must leave the actor alone while the
    #    critic still learns from that tick's outcome.
    all_forced = train_pilot.Controller(13)
    forced_collector = onpolicy.OnPolicyCollector(
        all_forced, forced=lambda _item: True
    )
    observation = _an_observation(step=5, targets=("a", "b"))
    asked, _actions, _free, logged, _value, tape = forced_collector.choose(observation)
    actor_before = all_forced.state_dict()["action_head"]["weight"].clone()
    optimizer = torch.optim.Adam(all_forced.parameters, lr=train_onpolicy.LEARNING_RATE)
    train_onpolicy.update(
        all_forced, optimizer,
        [{"decisions": [tape], "rewards": [1.0], "controller": all_forced,
          "row": {"task": {"on_time": 1}}}],
    )
    results["all_forced_step"] = {
        "logged_probability": logged,
        "asked": asked,
        "actor_moved": bool(
            (all_forced.state_dict()["action_head"]["weight"] - actor_before).abs().max().item() > 1e-8
        ),
    }
    if logged != 0.0 or asked != []:
        failures.append(f"a fully forced step still logged a decision: {results['all_forced_step']}")
    if results["all_forced_step"]["actor_moved"]:
        failures.append("a fully forced step trained the actor on actions nobody chose")

    # 7. Resume restores the whole run: two updates in one go must be indistinguishable from one,
    #    a save, a reload and one more -- same weights, same reported statistics, same cursor.
    results["resume"] = _resume_check(args, directory)
    if not results["resume"]["identical"]:
        failures.append(f"a resumed run did not reproduce an uninterrupted one: {results['resume']}")

    # 8. Nothing receiver-side can reach the controller.
    audit = train_pilot.leakage_audit()
    results["leakage_audit"] = {key: value for key, value in audit.items() if key != "sender_side_fields"}
    if not audit["clean"]:
        failures.append(f"the sender's observation type is no longer sender-side only: {audit['fields_not_on_the_allow_list']}")

    # 9. The configuration that runs is the configuration this branch claims.
    import stability_manifest

    executed = json.loads(json.dumps(stability_manifest.executed_configuration(), default=str))
    results["executed_configuration"] = {
        "mismatches": executed["mismatches"],
        "membership_consistency_enabled": executed["membership_consistency_enabled"],
        "allocation_arm_is_C": executed["allocation_arm_is_C"],
        "steps": executed["values"].get("steps"),
    }
    if executed["mismatches"]:
        failures.append(f"the built configuration drifted from the pilot's truth table: {executed['mismatches']}")
    if not executed["membership_consistency_enabled"]:
        failures.append("the training environment does not have the membership-consistency fix enabled")
    if executed["values"].get("steps") != common.STREAM_STEPS:
        failures.append("the episode horizon is not the recorded stream length")

    # 10. The reserved final-test block is still untouched, and this round wrote nothing into it.
    import data_split_audit

    split = data_split_audit.audit(data_split_audit.CANDIDATE_BLOCK, data_split_audit.PREVIOUS_TEST_BLOCK)
    results["data_split"] = {
        "reserved_block": split["replacement"]["block"],
        "reserved_clean": split["replacement"]["clean"],
        "previous_block_executed": split["previously_registered_final_test"]["seeds_with_executed_episodes"],
    }
    if not split["replacement"]["clean"]:
        failures.append("the reserved final-test block has artefacts on disk -- it is no longer unseen")

    # 11. Milestones are exact this round. A fresh run advances 4,000 steps per update, so the
    #     100k/200k/300k/400k checkpoints land on the milestone rather than after it.
    results["milestones"] = {
        "steps_per_update": STEPS_PER_UPDATE,
        "exact": [milestone for milestone in train_onpolicy.CHECKPOINT_STEPS if milestone % STEPS_PER_UPDATE == 0],
        "rounded_up": [
            milestone for milestone in train_onpolicy.CHECKPOINT_STEPS if milestone % STEPS_PER_UPDATE
        ],
    }
    if len(results["milestones"]["exact"]) != len(train_onpolicy.CHECKPOINT_STEPS):
        failures.append(f"a milestone would not land exactly: {results['milestones']}")

    # 12. Both tiers are the calibration's own numbers, not a dial this round turned.
    results["tiers"] = {
        tier: {"bytes": common.BUDGETS_BYTES[tier], "locked": common.locked_budget(tier)}
        for tier in common.BUDGET_TIERS
    }
    if any(value["bytes"] != value["locked"] for value in results["tiers"].values()):
        failures.append("a budget tier this round uses is not the locked calibration value")

    results["failures"] = failures
    results["budget_bytes"] = budget
    results["seed"] = seed
    results["constants"] = {
        "episodes_per_update": train_pilot.EPISODES_PER_UPDATE,
        "epochs_per_batch": train_pilot.EPOCHS_PER_BATCH,
        "minibatch_steps": train_pilot.MINIBATCH_STEPS,
        "clip": train_pilot.CLIP,
        "entropy_bonus": train_pilot.ENTROPY_BONUS,
        "max_wait_steps": pilot_env.MAX_WAIT_STEPS,
        "urgent_wait_steps": pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
        "urgency_steps": pilot_env.URGENCY_STEPS,
        "collector": "onpolicy",
        "exploration_epsilon_used": None,
        "feature_layout": train_pilot.FEATURE_LAYOUT_NAME,
    }
    common.write_json(common.json_path("TIGHT_SMOKE.json"), results)
    print(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    return 1 if failures else 0


def _an_observation(*, step: int, targets: tuple[str, ...]) -> Any:
    from integrated_sar.comm_policy import SendItem, SendObservation

    items = tuple(
        SendItem(
            target_id=target, ready_step=0, age_steps=99, pending_event=True, changed_view=True,
            first_emission=True, revision=1, revision_delta=1, stage_index=0, stage_changed=False,
            progress=0.0, progress_delta=0.0, requirements_changed=False, members_changed=False,
            deadline=500.0, remaining_time=5.0, sent_count=0, last_sent_step=None,
            since_last_emission_time=None, has_future_requirements=False,
        )
        for target in targets
    )
    return SendObservation(
        step=step, current_time=float(step), items=items, remaining_budget_bytes=1000,
        budget_bytes_per_episode=1000, capacity_remaining=None, backlog_targets=len(items), subscribers=2,
    )


def _resume_check(args: argparse.Namespace, directory: Path) -> dict[str, Any]:
    """Uninterrupted versus saved-and-reloaded, over the same two updates."""
    import argparse as _argparse

    def run(label: str, *, stops: list[int]) -> dict[str, Any]:
        root = directory / "resume" / label
        for steps in stops:
            train_onpolicy.train(
                _argparse.Namespace(
                    seed=1, tier="75pct", budget_bytes=args.budget_bytes, steps=steps,
                    validation_episodes=0, threads=args.threads, output_root=str(root),
                )
            )
        checkpoint = train_onpolicy.load(root / "75pct_seed1" / "tight_75pct_seed1.pt", 1)
        updates = checkpoint["state"]["updates"]
        return {
            "updates": len(updates),
            "final_fingerprint": _fingerprint_of(checkpoint["controller"]),
            "last_report": {
                key: updates[-1][key]
                for key in ("at_steps", "samples", "optimizer_steps", "approx_kl", "clip_fraction",
                            "batch_on_time", "batch_bytes", "drawn_send_rate")
            },
            "cursor": checkpoint["state"]["cursor"],
        }

    whole = run("whole", stops=[STEPS_PER_UPDATE * UPDATES])
    split = run("split", stops=[STEPS_PER_UPDATE, STEPS_PER_UPDATE * UPDATES])
    return {
        "uninterrupted": whole,
        "resumed": split,
        "identical": bool(whole["final_fingerprint"] == split["final_fingerprint"])
        and whole["last_report"] == split["last_report"]
        and whole["cursor"] == split["cursor"]
        and whole["updates"] == split["updates"] == UPDATES,
    }


def _fingerprint_of(payload: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for tensor in common.tensors(payload):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-bytes", type=int, default=common.BUDGETS_BYTES["75pct"])
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    started = time.time()
    code = check(args)
    print(f"preflight finished in {time.time() - started:.0f}s with exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
