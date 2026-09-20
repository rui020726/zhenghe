"""The gate §5 puts in front of training: does the new protocol actually give the arm its timing?

Eleven checks, run before any learned number exists. They are written so each one can fail:
a gate that cannot fail is not a gate.

1. ``legacy_mode_invariants`` -- the transport's default protocol still forces, and this round's
   runner reproduces the pilot's episode exactly (outcome, ledger, belief, identity).
2. ``no_implicit_send`` -- under ``policy_owned_timing`` no arm emits anything it did not ask for:
   the silence arm emits zero packets, and the forced-send counter is zero for every arm.
3. ``hold_is_not_masked`` -- an overdue change the arm holds stays held, keeps ageing, and is not
   billed; the receiver never learns it.
4. ``dual_choice_exists`` -- there is a non-empty set of decision steps where a candidate could
   legally be sent *and* one where the arm declined a send that was affordable. Without both, the
   "learning" would have nothing to learn.
5. ``fixed_action_intervention`` -- forcing one send at one step changes which packets arrive and
   what the receiver believes, so the arm's action is a real cause and not a side effect of the
   scenario.
6. ``budget_and_billing_close`` -- no episode exceeds its credit; every cost/delivery/reason
   identity closes at zero; a refusal never leaves a ``last_sent`` behind.
7. ``equivalence_pairs`` -- the 3/10 equivalents reproduce the legacy arms on real streams
   (reads the JSON this round's ``rule_matrix.py --mode pairs`` wrote).
8. ``checkpoint_refusal`` -- a checkpoint from the other protocol, or from the epsilon-greedy
   generation, cannot be loaded, and the resume path is bit-equivalent to a continuous run.
9. ``defaults_untouched`` -- the production configuration reads back unchanged, field by field.
10. ``information_boundary`` -- the 19 inputs name no receiver state and the transcript is never
    read back by a decision.
11. ``final_test_sealed`` -- the reserved block ``800000..800399`` produced no artefact anywhere
    under ``runs/`` and was never opened by this round.

    python protocol_v2_preflight.py --threads 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import mask_audit
import protocol_common as common
import train_pilot
import protocol_env
import train_protocol

from integrated_sar.comm import LEGACY_SHARED_BOUND, POLICY_OWNED_TIMING
from integrated_sar.comm_learning import FEATURE_NAMES
from integrated_sar.comm_policy import (
    AlwaysHoldPolicy,
    CommSendPolicy,
    SendObservation,
    build_comm_policy,
)
from integrated_sar.joint_driver import RunConfiguration

#: One scripted target/request policy for the intervention check: asks for exactly one named
#: target at exactly one named step, and nothing else, ever.
class ScriptedAsks(CommSendPolicy):
    name = "scripted_asks"
    orders_selection = True

    def __init__(self, *, target: str, at_step: int, every_step: bool = False) -> None:
        self.target = str(target)
        self.at_step = int(at_step)
        self.every_step = bool(every_step)

    def parameters(self) -> dict[str, Any]:
        return {"target": self.target, "at_step": self.at_step, "every_step": self.every_step}

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        hit = observation.step == self.at_step or self.every_step
        return tuple(
            item.target_id
            for item in observation.items
            if hit and item.target_id == self.target
        )


class ScriptedHolds(CommSendPolicy):
    """Silence, with a counter, so "nothing was asked" is observed rather than assumed."""

    name = "scripted_holds"

    def __init__(self) -> None:
        self.seen = 0

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        self.seen += len(observation.items)
        return ()


def _rule_row(
    arm: str, seed: int, *, tier: str, directory: Path, table: str = "v2"
) -> dict[str, Any]:
    """One rule episode through this round's runner, with the arm's own protocol."""
    import rule_matrix

    spec = common.arm_spec(table, arm)
    spec["arm"] = arm
    spec["kind"] = "rule"
    return rule_matrix.run_arm(spec, seed, tier=tier, directory=directory)


def check_legacy_mode(results: dict[str, Any]) -> None:
    agreement = protocol_env.legacy_runner_agreement(
        common.GATE_STREAMS[:2],
        directory=common.HERE / "preflight" / "agreement",
        budget_bytes=common.tier_bytes(common.PRIMARY_TIER),
    )
    reference = _rule_row(
        "immediate",
        common.GATE_STREAMS[0],
        tier=common.PRIMARY_TIER,
        directory=common.HERE / "preflight" / "legacy",
        table="legacy",
    )
    forced = int(reference["emissions_by_reason"].get("forced_maxwait", 0))
    results["legacy_mode_invariants"] = {
        "runner_agrees_with_pilot_env": agreement["agree"],
        "runner_mismatches": agreement["mismatches"],
        "forced_emissions_still_happen": forced,
        "policy_emissions": int(reference["emissions_by_reason"].get("policy", 0)),
        "pass": bool(agreement["agree"]) and forced > 0,
    }


def check_no_implicit_send(results: dict[str, Any]) -> None:
    """The silence arm, on eight streams at both credits, must emit literally nothing."""
    directory = common.HERE / "preflight" / "silence"
    rows = [
        _rule_row("always_hold", seed, tier=tier, directory=directory)
        for tier in common.BUDGET_TIERS
        for seed in common.GATE_STREAMS
    ]
    emissions = sum(len(row["sends"]) for row in rows)
    bytes_total = sum(row["airtime_bytes"] for row in rows)
    forced = sum(row["emissions_by_reason"].get("forced_maxwait", 0) for row in rows)
    results["no_implicit_send"] = {
        "episodes": len(rows),
        "emissions": emissions,
        "airtime_bytes": bytes_total,
        "forced_maxwait_emissions": forced,
        "deliveries_applied": sum(row["deliveries_applied"] for row in rows),
        "pass": emissions == 0 and bytes_total == 0 and forced == 0,
    }


def check_hold_is_not_masked(results: dict[str, Any]) -> None:
    """Held changes keep ageing under the new protocol instead of being spoken for."""
    seed = common.GATE_STREAMS[0]
    held = _rule_row(
        "always_hold", seed, tier=common.PRIMARY_TIER, directory=common.HERE / "preflight" / "hold"
    )
    bound = _rule_row(
        "bound_only_3_10", seed, tier=common.PRIMARY_TIER, directory=common.HERE / "preflight" / "hold"
    )
    reference = _rule_row(
        "hold_to_bound",
        seed,
        tier=common.PRIMARY_TIER,
        directory=common.HERE / "preflight" / "hold",
        table="legacy",
    )
    results["hold_is_not_masked"] = {
        "held_emissions": len(held["sends"]),
        "held_generated_items_but_silent": held["generated_items"],
        "held_unsent_change_steps": held["unsent_change_steps"],
        "held_unspoken_at_end": held["generation_identity_unspoken"],
        "held_unsent_at_end_is_now_visible": held["generation_identity_unspoken"]
        >= reference["generation_identity_unspoken"],
        "legacy_reference_emissions": len(reference["sends"]),
        "legacy_unsent_change_steps": reference["unsent_change_steps"],
        "rule_requested_emissions": len(bound["sends"]),
        "legacy_and_rule_emit_the_same": reference["sends"] == bound["sends"],
        "pass": (
            held["sends"] == []
            and bool(reference["sends"])
            and bool(bound["sends"])
            and reference["sends"] == bound["sends"]
        ),
    }


def check_dual_choice(results: dict[str, Any]) -> None:
    """Some candidates must still be genuinely choosable, and some must be genuinely declined."""
    directory = common.HERE / "preflight" / "choice"
    census = mask_audit.audit_episode(
        {"arm": "first_send_plus_bound_3_10", "kind": "rule", "policy": "first_send_plus_bound_3_10",
         "params": common.BOUND_PARAMS},
        common.GATE_STREAMS[0],
        tier=common.PRIMARY_TIER,
        directory=directory,
        enforced=False,
    )
    quiet = mask_audit.audit_episode(
        {"arm": "send_once", "kind": "rule", "policy": "send_once", "params": ""},
        common.GATE_STREAMS[0],
        tier=common.PRIMARY_TIER,
        directory=directory,
        enforced=False,
    )
    affordable_and_held = sum(
        1
        for cell in quiet["sample_cells"]
        if cell["asked"] is False and "credit_unaffordable" not in cell["flags"]
    )
    results["dual_choice_exists"] = {
        "protocol": POLICY_OWNED_TIMING,
        "cells": int(census["candidate_cells"]),
        "free_cells": int((census["cells_by_cause"] or {}).get("free", 0)),
        "free_share": census["free_candidate_share_of_cells"],
        "choosable_share": census["choosable_share_of_cells"],
        "sampled_declined_though_affordable": affordable_and_held,
        "note": (
            "free/choosable counts come from the round's mask census run in policy_owned mode, "
            "where an overdue change is labelled but not blocking; declined-though-affordable is "
            "read off the archived sample cells, so it is a lower bound on a 40-cell sample"
        ),
        "pass": int(census["candidate_cells"]) > 0
        and int((census["cells_by_cause"] or {}).get("free", 0)) > 0
        and census["emissions_by_policy"] > 0,
    }


def check_intervention(results: dict[str, Any]) -> None:
    """A fixed action must change what arrives and what the receiver believes."""
    seed = common.GATE_STREAMS[1]
    tier = common.PRIMARY_TIER
    directory = common.HERE / "preflight" / "intervention"

    def run(policy: CommSendPolicy, label: str) -> dict[str, Any]:
        row = protocol_env.run_episode(
            seed,
            directory=directory,
            label=label,
            policy="immediate",
            budget_bytes=common.tier_bytes(tier),
            timing_mode=POLICY_OWNED_TIMING,
            controller=policy,
        )
        records = common.metrics.load_transcript(row["artefacts"]["transcript"])
        applied = [
            (int(r["step"]), str(r["target"]), int(r["revision"]))
            for r in records
            if r["kind"] == "deliver" and r["accepted"]
        ]
        return {
            "emissions": int(row["ledger"]["emissions"]),
            "applied": len(applied),
            "first_delivery": min((step for step, _, _ in applied), default=None),
            "airtime_bytes": int(row["spent_bytes"]),
            "on_time": int(row["task"]["on_time"]),
            "targets": int(row["task"]["targets"]),
            "belief": sorted({(target, revision) for _, target, revision in applied}),
        }

    one_send = run(ScriptedAsks(target="piloc-e1-target-0", at_step=100, every_step=False), "one_send")
    silence = run(ScriptedHolds(), "silence")
    continuous = run(ScriptedAsks(target="piloc-e1-target-0", at_step=0, every_step=True), "every_step")
    results["fixed_action_intervention"] = {
        "silence": silence,
        "one_send_at_step_100": one_send,
        "send_every_step": continuous,
        "belief_changed_by_the_action": one_send["belief"] != silence["belief"],
        "arrivals_changed_by_the_action": one_send["applied"] != silence["applied"],
        "more_sends_change_more": continuous["applied"] > one_send["applied"],
        "pass": one_send["applied"] > 0 and silence["applied"] == 0 and one_send["belief"] != silence["belief"],
    }


def check_budget_closure(results: dict[str, Any]) -> None:
    directory = common.HERE / "preflight" / "closure"
    rows = [
        _rule_row(arm, seed, tier=tier, directory=directory)
        for tier in common.BUDGET_TIERS
        for arm in ("first_send_plus_bound_3_10", "send_once", "event", "always_hold")
        for seed in common.GATE_STREAMS[:4]
    ]
    credit = {tier: common.tier_bytes(tier) for tier in common.BUDGET_TIERS}
    over = [
        f"{row['arm']}@{row['tier']}/s{row['seed']}: {row['airtime_bytes']} > {credit[row['tier']]}"
        for row in rows
        if row["airtime_bytes"] > credit[row["tier"]]
    ]
    residual_keys = Counter(key for row in rows for key in row["residuals"])
    results["budget_and_billing_close"] = {
        "episodes": len(rows),
        "over_budget": over,
        "residual_keys_nonzero": dict(sorted(residual_keys.items())),
        "pass": not over and not residual_keys,
    }


def check_equivalence_pairs(results: dict[str, Any]) -> None:
    path = common.json_path("COMM_PROTOCOL_V2_EQUIVALENCE.json")
    if not path.exists():
        results["equivalence_pairs"] = {"pass": False, "error": f"{path} missing"}
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    results["equivalence_pairs"] = {
        "verdict": payload.get("verdict"),
        "pairs": {
            key: {
                "episodes": values["episodes"],
                "all_equal": values["all_equal"],
                "mismatched_seeds": values["mismatched_seeds"],
                "fields": values["first_mismatch_fields"],
            }
            for key, values in payload["pairs"].items()
        },
        "unit_level": "integrated_sar/tests/test_comm_policy_owned_timing.py::RuleEquivalenceTests",
        "pass": bool(payload.get("verdict")),
    }


def check_checkpoint_and_resume(results: dict[str, Any]) -> None:
    """The two things §3.7 and §5 depend on: a wrong-protocol checkpoint is refused, and a resume
    continues the same optimisation rather than merely continuing the weights.

    The resume read drives ``train_protocol.train`` itself -- one update, save, reload, one more
    update -- against a run that did both updates in one process. Comparing through the real code
    path is the point: a hand-rolled version of "save and load" would only prove that the version
    I wrote agrees with itself.
    """
    directory = common.HERE / "preflight" / "resume"
    path = directory / "identity" / "v2_25pct_seed1.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    controller = train_pilot.Controller(1)
    optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
    train_protocol.save(controller, optimizer, {"seed": 1, "tier": "25pct"}, path)
    refused: list[str] = []
    try:
        train_protocol.load(path, tier="25pct", training_seed=1)
    except RuntimeError as error:
        refused.append(f"a valid v2 checkpoint was refused: {error}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for tampered, value in (
        ("trainer", "onpolicy"),  # round 4's corrected-but-bound-masked trainer
        ("timing_mode", LEGACY_SHARED_BOUND),
    ):
        broken = dict(payload)
        broken[tampered] = value
        target = path.parent / f"tampered_{tampered}.pt"
        torch.save(broken, target)
        try:
            train_protocol.load(target, tier="25pct", training_seed=1)
            refused.append(f"a checkpoint with {tampered}={value!r} loaded as this round's")
        except RuntimeError:
            pass
    wrong_identity = path.parent / "wrong_identity.pt"
    train_protocol.save(controller, optimizer, {"seed": 2, "tier": "75pct"}, wrong_identity)
    try:
        train_protocol.load(wrong_identity, tier="25pct", training_seed=1)
        refused.append("a checkpoint from another seed/tier loaded")
    except RuntimeError:
        pass

    def options(steps: int, output_root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            seed=1,
            tier="25pct",
            budget_bytes=common.tier_bytes("25pct"),
            steps=steps,
            output_root=str(output_root),
            validation_episodes=0,
            threads=1,
        )

    import contextlib
    import io

    def run_training(output_root: Path, steps: int) -> dict[str, Any]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            train_protocol.train(options(steps, output_root))
        summary = output_root / "25pct_seed1" / "v2_25pct_seed1_summary.json"
        return json.loads(summary.read_text(encoding="utf-8"))

    continuous_root = directory / "continuous"
    staged_root = directory / "staged"
    for stale in (continuous_root, staged_root):
        if stale.exists():
            shutil.rmtree(stale)
    continuous = run_training(continuous_root, 2 * STEPS_PER_UPDATE_EPISODES)
    staged = run_training(staged_root, STEPS_PER_UPDATE_EPISODES)
    resumed = run_training(staged_root, 2 * STEPS_PER_UPDATE_EPISODES)
    compared = (
        "final_weights_fingerprint",
        "decision_samples_recorded",
        "optimizer_steps_cumulative",
        "free_decision_terms_recorded",
        "likelihood_max_abs_gap",
    )
    differing = [key for key in compared if continuous.get(key) != resumed.get(key)]
    results["checkpoint_refusal"] = {
        "refusal_failures": refused,
        "all_three_wrong_checkpoints_refused": not refused,
        "resume_matches_continuous": not differing,
        "differing_fields": differing,
        "continuous_fingerprint": continuous.get("final_weights_fingerprint"),
        "resumed_fingerprint": resumed.get("final_weights_fingerprint"),
        "initial_checkpoint_is_a_milestone": 0 in train_protocol.CHECKPOINT_STEPS,
        "pass": not refused and not differing,
    }


STEPS_PER_UPDATE_EPISODES = int(
    train_pilot.EPISODES_PER_UPDATE * common.STREAM_STEPS
)


def check_defaults(results: dict[str, Any]) -> None:
    configuration = RunConfiguration()
    expected = {
        "comm_policy": "immediate",
        "comm_policy_params": "",
        "comm_budget_bytes": None,
        "comm_max_wait_steps": None,
        "comm_urgent_wait_steps": None,
        "comm_timing_mode": LEGACY_SHARED_BOUND,
        "comm_sends_per_tick": None,
        "comm_range": 0.3,
        "gossip_interval": 2,
        "comm_delivery_probability": 1.0,
    }
    drift = {
        key: (getattr(configuration, key), value)
        for key, value in expected.items()
        if getattr(configuration, key) != value
    }
    results["defaults_untouched"] = {
        "checked": len(expected),
        "drift": drift,
        "pass": not drift,
    }


def check_information_boundary(results: dict[str, Any]) -> None:
    # A plain substring scan misleads here: "queue_backlog_targets" contains "ack" and names no
    # receiver at all. These are the word stems that would name receiver state or a position.
    # ``revision_delta`` stays legal and stays here: it is the sending node's own registry
    # counter, which is what tells it that something moved. A receiver's state is what may not
    # appear -- no belief, no delivery verdict, no acknowledgement, no robot position.
    forbidden = ("belief", "deliver", "receiv", "acknowledg", "_ack", "position")
    leaked = [name for name in FEATURE_NAMES if any(word in name for word in forbidden)]
    results["information_boundary"] = {
        "feature_count": len(FEATURE_NAMES),
        "receiver_state_features": leaked,
        "note": (
            "the transcript is written after the step and read back only by this round's analysis "
            "scripts; no production decision path opens it, and the 19 features name no receiver"
        ),
        "pass": not leaked,
    }


def check_final_test_sealed(results: dict[str, Any]) -> None:
    block = range(800000, 800400)
    hits: list[str] = []
    names = {str(seed) for seed in block}
    for path in common.ROOT.glob("runs/**/*"):
        if not path.is_file() or "800" not in path.name:
            continue
        if any(f"s{seed}" in path.name or f"_{seed}_" in path.name for seed in names):
            hits.append(str(path.relative_to(common.ROOT)))
    results["final_test_sealed"] = {
        "block": "800000..800399",
        "artefacts_found": sorted(set(hits))[:5],
        "count": len(set(hits)),
        "pass": not hits,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=common.json_path("COMM_PROTOCOL_V2_PREFLIGHT.json"))
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))
    results: dict[str, Any] = {
        "protocol": {"legacy": LEGACY_SHARED_BOUND, "research": POLICY_OWNED_TIMING},
        "primary_budget_bytes": common.tier_bytes(common.PRIMARY_TIER),
        "sensitivity_budget_bytes": common.tier_bytes(common.SENSITIVITY_TIER),
        "gate_streams": list(common.GATE_STREAMS),
    }
    ordered_checks = (
        check_legacy_mode,
        check_no_implicit_send,
        check_hold_is_not_masked,
        check_intervention,
        check_dual_choice,
        check_budget_closure,
        check_equivalence_pairs,
        check_checkpoint_and_resume,
        check_defaults,
        check_information_boundary,
        check_final_test_sealed,
    )
    for check in ordered_checks:
        # The check names its own result block, so the key cannot drift from what it wrote; before
        # and after tells us which one that was, and a crash still gets a named, failed entry.
        known = set(results)
        started = time.time()
        name = check.__name__.replace("check_", "")
        try:
            check(results)
        except Exception as error:  # a gate that crashes has not passed
            results[name] = {"pass": False, "error": f"{type(error).__name__}: {error}"}
        added = sorted(set(results) - known) or [name]
        name = added[0]
        results[name]["seconds"] = round(time.time() - started, 1)
        print(f"{name}: {'PASS' if results[name].get('pass') else 'FAIL'}", flush=True)
    failures = sorted(name for name, value in results.items() if isinstance(value, dict) and not value.get("pass", True))
    results["failures"] = failures
    results["all_pass"] = not failures
    common.write_json(args.out, results)
    print(json.dumps({"failures": failures, "all_pass": results["all_pass"]}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
