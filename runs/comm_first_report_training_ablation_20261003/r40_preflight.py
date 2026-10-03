"""Pure-compute ROUND40 gate. It imports no runner entry and constructs no JointRun."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

import r40_common as r40

COMM_ACTION_ELIGIBILITY = "first-report-only/1"
REWARD_TIMING_VERSION = "published-completion-event-at-relay-tick/2"
TRAINING_VERSION = (
    "constrained-cost-v2/first-report-only/1/"
    "published-completion-event-at-relay-tick/2"
)
FirstReportOnlyCollector = None

BASE_HEAD = "4e989a9088111ce6d9171cb1a21629a4b2b1e861"
ROUND40_BASIS_COMMIT = "e8a1841cf997baafbf16cb77005c7a496ee09589"
REFERENCE_PHASE_FREEZE_COMMIT = "2ac0933bffb8f7883b8ae35df025ce29d0c8a15d"
OLD_TRAINING_VERSION = (
    "constrained-cost-v2/published-completion-event-at-relay-tick/2"
)
R40_CODE_FILES = (
    "r40_common.py", "r40_eligibility.py", "r40_versions.py", "r40_run.py",
    "r40_train.py", "r40_worker.py", "r40_analysis.py", "r40_preflight.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _send_item(target: str, *, first: bool, charge: int = 4,
               executable: bool = True, block_reason: str = "allowed",
               ready_step: int = 1):
    from integrated_sar.comm_policy import SendItem

    return SendItem(
        target_id=target,
        ready_step=ready_step,
        age_steps=0,
        pending_event=first,
        changed_view=True,
        first_emission=first,
        revision=1 if first else 2,
        revision_delta=1,
        stage_index=0,
        stage_changed=False,
        progress=0.0,
        progress_delta=0.0,
        requirements_changed=False,
        members_changed=False,
        deadline=500.0,
        remaining_time=400.0,
        sent_count=0 if first else 1,
        last_sent_step=None if first else 1,
        since_last_emission_time=None if first else 1.0,
        has_future_requirements=False,
        attempt_charge_bytes=charge,
        send_executable=executable,
        send_block_reason=block_reason,
    )


def _observation(items, *, remaining: int = 100, budget: int = 100,
                 capacity: int | None = None, step: int = 1):
    from integrated_sar.comm_policy import SendObservation

    return SendObservation(
        step=step,
        current_time=float(step),
        items=tuple(items),
        remaining_budget_bytes=float(remaining),
        budget_bytes_per_episode=budget,
        capacity_remaining=capacity,
        backlog_targets=len(items),
        subscribers=1,
    )


class _FixedController:
    def __init__(self, *, send_logit: float = 8.0) -> None:
        self.send_logit = float(send_logit)

    def forward(self, batch: torch.Tensor):
        logits = torch.tensor([0.0, self.send_logit], dtype=torch.float32).repeat(
            len(batch), 1
        )
        return logits, torch.zeros(len(batch), dtype=torch.float32)

    def cost_values(self, batch: torch.Tensor):
        return torch.zeros(len(batch), dtype=torch.float32)


def _collector_preflight(common: Any, trainer: Any, v2_controller: Any) -> dict[str, Any]:
    from budget_masked import BudgetMaskedCollector, HOLD, SEND
    from integrated_sar.comm_learning import features
    from integrated_sar.comm_policy import SendObservation
    from onpolicy import action_entropy, joint_log_prob
    from sb_collector import StageBCollector

    controller = v2_controller.fresh_controller(11)
    all_first = _observation([
        _send_item("target-a", first=True, charge=3),
        _send_item("target-b", first=True, charge=3),
    ], remaining=20, budget=20)
    old = StageBCollector(controller)
    new = FirstReportOnlyCollector(controller)
    torch.manual_seed(20261003)
    old.decide(all_first)
    torch.manual_seed(20261003)
    new.decide(all_first)
    old_tape, new_tape = old.records[0], new.records[0]
    _assert(old_tape["targets"] == new_tape["targets"], "candidate order changed on free rows")
    _assert(old_tape["features"] == new_tape["features"], "feature rows changed on free rows")
    _assert(old_tape["attempt_charge_bytes"] == new_tape["attempt_charge_bytes"],
            "relay quotes changed on free rows")
    _assert(old_tape["send_probabilities"] == new_tape["send_probabilities"],
            "probabilities changed on free rows")
    _assert(old_tape["actions"] == new_tape["actions"], "sampled actions changed on free rows")
    _assert(old_tape["actionable"] == new_tape["actionable"], "budget gate changed on free rows")
    _assert(old_tape["shadow_budget_after"] == new_tape["shadow_budget_after"],
            "shadow budget changed on free rows")
    _assert(old_tape["log_prob"] == new_tape["log_prob"],
            "sample likelihood changed on free rows")

    budget_counterexample = _observation([
        _send_item("a-refresh", first=False, charge=4),
        _send_item("b-first", first=True, charge=4),
    ], remaining=6, budget=6)
    old_gate = BudgetMaskedCollector(_FixedController(), forced=lambda _item: False,
                                     mask=False, greedy=True, budget_mask=True)
    new_gate = FirstReportOnlyCollector(_FixedController(), greedy=True)
    old_gate.decide(budget_counterexample)
    new_gate.decide(budget_counterexample)
    old_gate_tape, new_gate_tape = old_gate.records[0], new_gate.records[0]
    _assert(old_gate_tape["actionable"] == ["a-refresh"],
            "legacy gate counterexample did not spend quote 4 on the refresh")
    _assert(new_gate_tape["actions"]["a-refresh"] == HOLD,
            "refresh escaped its fixed HOLD action")
    _assert(new_gate_tape["actionable"] == ["b-first"],
            "suppressed refresh blocked the affordable first report")
    _assert(new_gate_tape["actions"]["b-first"] == SEND,
            "the affordable first report did not remain model-controlled")
    _assert(new_gate_tape["shadow_budget_after"] == 2.0,
            "only the first-report SEND should charge the four-byte shadow")

    mixed = _observation([
        _send_item("a-refresh", first=False, charge=4),
        _send_item("b-first", first=True, charge=4),
        _send_item("c-refresh", first=False, charge=4),
    ], remaining=100, budget=100)
    sampled = FirstReportOnlyCollector(controller, greedy=False)
    matrix = torch.as_tensor(np.asarray(
        [features(mixed, item) for item in mixed.items], dtype=np.float32))
    with torch.no_grad():
        expected_logits, _ = controller.forward(matrix)
        expected_probabilities = torch.softmax(expected_logits, dim=-1)
    torch.manual_seed(7781)
    torch.multinomial(expected_probabilities[1], num_samples=1)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(7781)
    sampled.decide(mixed)
    actual_rng = torch.get_rng_state().clone()
    _assert(torch.equal(expected_rng, actual_rng),
            "fixed-HOLD refreshes consumed action RNG draws")
    mixed_tape = sampled.records[0]
    _assert(mixed_tape["free"] == ["b-first"], "refresh entered the free action set")
    _assert(mixed_tape["fixed_hold_refresh"] == ["a-refresh", "c-refresh"],
            "refresh qualification did not use relay first_emission")
    _assert(mixed_tape["actions"]["a-refresh"] == HOLD
            and mixed_tape["actions"]["c-refresh"] == HOLD,
            "fixed refresh HOLD was not recorded")
    mixed_summary = sampled.tape_summary()
    _assert(mixed_summary["candidate_classes_close"]
            and mixed_summary["candidate_classes_disjoint"],
            "candidate classes do not close/disjoin")
    _assert(mixed_summary["refreshes_excluded_from_likelihood"],
            "refresh contributed to likelihood")

    # Send through the real batch_samples/likelihood/alignment and entropy helpers. The critic
    # still receives the whole row (all three original feature vectors).
    episode = {
        "collector": sampled,
        "steps": 3,
        "cost_stream": {1: {"airtime_bytes": 0}},
        "task_stream": {1: 1.0, 3: 2.0},
        "lambda": 1.0,
        "episode_index": 0,
        "scenario_seed": common.TRAINING_SCENARIOS[0],
        "on_time": 3,
    }
    samples = trainer.batch_samples([episode])
    alignment = trainer.likelihood_alignment(controller, samples)
    _assert(alignment["passed"], f"mixed-action likelihood mismatch: {alignment}")
    sample = samples[0]
    _assert(sample["targets"] == mixed_tape["targets"], "batching removed critic candidate rows")
    _assert(sample["features"].shape == (3, 19), "ALL_ROWS feature matrix changed")
    _assert(sample["free"] == {"b-first"}, "batching admitted a refresh into likelihood")
    logits, _ = controller.forward(torch.as_tensor(sample["features"], dtype=torch.float32))
    picked = torch.tensor([sample["actions"][target] for target in sample["targets"]])
    usable = torch.tensor([float(target in sample["free"]) for target in sample["targets"]])
    actual_joint = joint_log_prob(logits, picked, usable)
    index = sample["targets"].index("b-first")
    independently_recomputed = torch.log_softmax(logits, dim=-1)[index, picked[index]]
    _assert(torch.allclose(actual_joint, independently_recomputed, atol=1e-7, rtol=0),
            "refresh log probability was not zeroed in joint likelihood")
    entropy = action_entropy(logits, usable)
    independently_recomputed_entropy = -(
        torch.softmax(logits[index], dim=-1) * torch.log_softmax(logits[index], dim=-1)
    ).sum()
    _assert(torch.allclose(entropy, independently_recomputed_entropy, atol=1e-7, rtol=0),
            "refresh entropy was not excluded")

    # Entirely fixed rows remain critic samples, while the actual actor path has exactly zero
    # policy gradient because joint likelihood and entropy both multiply by the empty free mask.
    refresh_only = FirstReportOnlyCollector(controller, greedy=False)
    only_refresh = _observation([_send_item("only-refresh", first=False)], step=2)
    refresh_only.decide(only_refresh)
    only_episode = {
        "collector": refresh_only,
        "steps": 2,
        "cost_stream": {2: {"airtime_bytes": 50}},
        "task_stream": {2: 1.0},
        "lambda": 1.0,
        "episode_index": 1,
        "scenario_seed": common.TRAINING_SCENARIOS[1],
        "on_time": 1,
    }
    only_samples = trainer.batch_samples([only_episode])
    only_alignment = trainer.likelihood_alignment(controller, only_samples)
    _assert(only_alignment["passed"], f"all-fixed likelihood should be zero: {only_alignment}")
    _assert(len(only_samples) == 1 and only_samples[0]["free"] == set(),
            "all-fixed row was dropped or treated as actor-free")
    _assert(float(only_samples[0]["return_task"]) == 1.0,
            "task critic lost its all-fixed row target")
    only_features = torch.as_tensor(only_samples[0]["features"], dtype=torch.float32)
    only_logits, _ = controller.forward(only_features)
    only_picked = torch.tensor([only_samples[0]["actions"][target]
                                for target in only_samples[0]["targets"]])
    no_usable = torch.zeros(len(only_picked), dtype=torch.float32)
    new_log_prob = joint_log_prob(only_logits, only_picked, no_usable)
    old_log_prob = torch.tensor(only_samples[0]["log_prob"], dtype=torch.float32)
    ratio = torch.exp(new_log_prob - old_log_prob)
    advantage = torch.tensor(only_samples[0]["advantage"], dtype=torch.float32)
    surrogate = torch.minimum(ratio * advantage,
                              torch.clamp(ratio, 0.8, 1.2) * advantage)
    actor_loss = -surrogate - common.ENTROPY_BONUS * action_entropy(only_logits, no_usable)
    controller.inner.trunk.zero_grad(set_to_none=True)
    controller.inner.action_head.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_gradient = sum(float(p.grad.abs().sum()) for p in controller.inner.trunk.parameters()
                         if p.grad is not None)
    actor_gradient += sum(float(p.grad.abs().sum())
                          for p in controller.inner.action_head.parameters()
                          if p.grad is not None)
    _assert(actor_gradient == 0.0, f"all-fixed actor row has false gradient {actor_gradient}")

    return {
        "same_free_candidate_path": True,
        "refresh_quote4_before_first_quote4_balance6": {
            "legacy_actionable": old_gate_tape["actionable"],
            "round40_actionable": new_gate_tape["actionable"],
            "round40_actions": new_gate_tape["actions"],
            "round40_shadow_budget_after": new_gate_tape["shadow_budget_after"],
        },
        "refreshes_excluded_from_rng_likelihood_entropy": True,
        "batch_samples_all_rows_features_shape": list(sample["features"].shape),
        "mixed_likelihood_alignment": alignment,
        "all_fixed_critic_row_retained": True,
        "all_fixed_actor_gradient_l1": actor_gradient,
    }


def _relay_preflight() -> dict[str, Any]:
    from budget_masked import HOLD
    from integrated_sar.comm import CommRelay
    from integrated_sar.comm_policy import CommSendPolicy
    from integrated_sar.comm_budget import CommBudget
    from integrated_sar.tests.test_comm_budget_action_mask import Harness, detection, progress

    class CaptureDelegate(CommSendPolicy):
        name = "r40_test_capture"
        orders_selection = True

        def __init__(self, delegate: Any) -> None:
            self.delegate = delegate
            self.observations = []
            self.selections = []

        def decide(self, observation):
            self.observations.append(observation)
            selected = self.delegate.decide(observation)
            self.selections.append(tuple(selected))
            return selected

    dropped_collector = FirstReportOnlyCollector(_FixedController(), greedy=True)
    dropped_policy = CaptureDelegate(dropped_collector.policy())
    dropped = Harness(policy=dropped_policy)
    dropped.relay._delivery_probability = 0.0
    dropped.publish(detection("r40-drop"))
    dropped.tick()
    first_item = dropped_policy.observations[0].items[0]
    _assert(first_item.first_emission, "fresh relay target did not offer first-emission authority")
    _assert(dropped.relay.counters["emissions"] == 1
            and dropped.relay.counters["packets_lost"] > 0,
            "drop fixture did not make a real billed emission")
    _assert(dropped.relay._ledger["r40-drop"].last_sent is not None,
            "loss incorrectly erased the relay's successful-emission ledger")
    dropped.report(progress("r40-drop", observed_at=2.0))
    dropped.tick()
    revision = dropped_policy.observations[-1].items[0]
    _assert(revision.revision > first_item.revision and not revision.first_emission,
            "new revision incorrectly reset first-emission eligibility")
    _assert(dropped_collector.records[-1]["actions"]["r40-drop"] == HOLD
            and dropped_collector.records[-1]["free"] == [],
            "refresh after a dropped first emission was not fixed HOLD")

    refused_policy = type("AlwaysAsk", (), {
        "name": "r40_refusal_fixture",
        "orders_selection": True,
        "observations": [],
        "decide": lambda self, observation: (
            self.observations.append(observation)
            or tuple(item.target_id for item in observation.items)
        ),
    })()
    refused = Harness(policy=refused_policy, budget=CommBudget(bytes_per_episode=1))
    refused.publish(detection("r40-refused"))
    refused.tick()
    _assert(refused.relay.counters["budget_refused_sends"] == 1,
            "executor refusal fixture did not refuse the actual send")
    _assert(refused_policy.observations[-1].items[0].first_emission,
            "executor refusal consumed first-emission eligibility")
    refused.tick()
    _assert(refused_policy.observations[-1].items[0].first_emission,
            "unaccepted refusal did not retain first-report qualification")

    # A fresh relay is the episode boundary in the real runner; it has no inherited sent ledger.
    reset = Harness(policy=refused_policy)
    reset.publish(detection("r40-refused"))
    reset.tick()
    _assert(reset.relay._ledger["r40-refused"].last_sent is not None,
            "fresh-episode fixture did not emit")
    fresh_episode = Harness(policy=type("ObserveOnly", (), {
        "name": "r40_new_episode",
        "orders_selection": True,
        "observations": [],
        "decide": lambda self, observation: (
            self.observations.append(observation) or ()
        ),
    })())
    fresh_episode.publish(detection("r40-refused"))
    fresh_episode.tick()
    _assert(fresh_episode.policy.observations[-1].items[0].first_emission,
            "new episode inherited the previous episode's emission ledger")
    return {
        "relay_first_emission_is_authoritative": True,
        "real_emission_with_all_packets_lost_consumes_eligibility": True,
        "executor_budget_refusal_retains_eligibility": True,
        "new_revision_does_not_reset_eligibility": True,
        "new_episode_clears_emission_identity": True,
        "joint_run_or_simulator_constructed": False,
    }


def _objective_preflight(common: Any) -> dict[str, Any]:
    from cost_math import actor_advantage
    from onpolicy import joint_log_prob

    task = torch.tensor([1.0], dtype=torch.float64)
    cost = torch.tensor([1.0], dtype=torch.float64)
    at_zero = actor_advantage(task, cost, 0.0)
    at_high = actor_advantage(task, cost, 3.0)
    _assert(float(at_zero[0]) == -1.0 and float(at_high[0]) == 2.0,
            "registered combined-advantage signs changed")
    theta_low = torch.tensor(0.0, requires_grad=True)
    loss_low = -at_zero[0].float() * torch.log_softmax(
        torch.stack((torch.tensor(0.0), theta_low)), dim=0
    )[1]
    loss_low.backward()
    theta_high = torch.tensor(0.0, requires_grad=True)
    loss_high = -at_high[0].float() * torch.log_softmax(
        torch.stack((torch.tensor(0.0), theta_high)), dim=0
    )[1]
    loss_high.backward()
    _assert(float(theta_low.grad) > 0 and float(theta_high.grad) < 0,
            "positive-cost hand calculation has the wrong descent direction")
    return {
        "actor_advantage_lambda0": float(at_zero[0]),
        "actor_advantage_lambda3": float(at_high[0]),
        "send_logit_gradient_lambda0": float(theta_low.grad),
        "send_logit_gradient_lambda3": float(theta_high.grad),
        "meaning": "lambda=0 suppresses a positive-cost SEND; larger task multiplier favors it",
    }


def _bundle_preflight(trainer: Any) -> dict[str, Any]:
    bundled, prefix = trainer._bundle({0: 3.0, 2: 2.0, 5: 4.0, 8: 7.0}, [2, 5], 8)
    _assert(prefix == 3.0, "tick-zero reward did not remain a prefix")
    _assert(bundled == {2: 2.0, 5: 11.0},
            f"same-tick and terminal tick event credit changed: {bundled}")
    return {"stream": {"0": 3, "2": 2, "5": 4, "8": 7},
            "decision_steps": [2, 5], "prefix": prefix, "bundled": bundled,
            "same_tick_and_final_tick_captured": True}


def _optimizer_preflight(v2_controller: Any) -> dict[str, Any]:
    controller = v2_controller.fresh_controller(33)
    expected = [*controller.inner.trunk.parameters(), *controller.inner.action_head.parameters(),
                *controller.inner.value_head.parameters(), *controller.cost_head.parameters()]
    actual = list(controller.parameters)
    _assert({id(value) for value in actual} == {id(value) for value in expected},
            "optimizer parameter set does not exactly cover four communication modules")
    return {"parameter_count": len(actual), "modules": ["trunk", "action_head",
            "task_value_head", "cost_value_head"],
            "optimizer_parameter_ids_equal_only_these_four_modules": True,
            "frozen_search_allocation_parameters_in_optimizer": False}


def _archive_identities(root: Path, common: Any, trainer: Any,
                         v2_controller: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    source_round = root / "runs" / "comm_final_train_20261001"
    archived = _json(source_round / "FROZEN_CONFIG_MANIFEST.json")
    checkpoint_rows: dict[str, Any] = {}
    for seed in r40.TRAIN_SEEDS:
        seed_dir = source_round / f"seed{seed}"
        hash_index = _json(seed_dir / "CHECKPOINT_HASHES.json")["files"]
        record = _json(seed_dir / "TRAINING_RECORD.json")["result"]
        checkpoints: dict[str, Any] = {}
        for episodes in (0, 64, 128):
            name = f"v2_train_s{seed}_ep{episodes}.pt"
            path = seed_dir / "checkpoints" / name
            expected_index = hash_index[f"milestone_ep{episodes}"]
            actual_sha = _sha256(path)
            _assert(actual_sha == expected_index["bytes_sha256"],
                    f"ROUND36 file bytes mismatch for {path}: {actual_sha}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            controller, state, identity = trainer.load_checkpoint_read_only(
                path, freeze=payload["freeze"])
            parameter = controller.parameter_summary()
            expected_parameter = None
            if episodes == 0:
                expected_parameter = archived["initial_weights_per_seed"][f"s{seed}"][
                    "parameter_sha256_per_source"]["round24_ep0_milestone_restored_by_the_real_loader"]
            elif episodes == 64:
                batch = _json(seed_dir / "checkpoints" / f"v2_batch_s{seed}_b15.json")
                expected_parameter = batch["controller_parameters_after"]["parameter_sha256"]
            else:
                expected_parameter = record["controller_parameters"]["parameter_sha256"]
            _assert(parameter["parameter_sha256"] == expected_parameter,
                    f"ROUND36 parameter identity mismatch for seed {seed} ep{episodes}")
            _assert(int(state["episodes_done"]) == episodes,
                    f"ROUND36 state episode count mismatch seed {seed} ep{episodes}")
            _assert(int(state["optimizer_steps"]) == int(expected_index["optimizer_steps"]),
                    f"ROUND36 optimizer count mismatch seed {seed} ep{episodes}")
            _assert(float(state["lambda"]) == float(expected_index["lambda"]),
                    f"ROUND36 lambda mismatch seed {seed} ep{episodes}")
            checkpoints[f"ep{episodes}"] = {
                "path": str(path), "size_bytes": path.stat().st_size,
                "file_sha256": actual_sha, "file_sha256_matches_round36_index": True,
                "training_seed": seed, "episodes_done": int(state["episodes_done"]),
                "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda": float(state["lambda"]),
                "parameter_sha256": parameter["parameter_sha256"],
                "parameter_identity_matches_round36_record": True,
                "training_version": payload.get("training_version"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "read_only_identity": identity,
            }
        checkpoint_rows[str(seed)] = checkpoints

    source_order = common.scenario_order()
    order_digest = hashlib.sha256(json.dumps(source_order).encode("utf-8")).hexdigest()
    _assert(order_digest == archived["data_order"]["actual_sha256"],
            "ROUND40 scenario order diverged from ROUND36")
    frozen_inputs: dict[str, Any] = {}
    for path in common.CHECKPOINT_FILES:
        expected = archived["code_identity"]["frozen_checkpoints"][
            path.resolve().relative_to(root.resolve()).as_posix()]
        actual = _sha256(path)
        _assert(actual == expected, f"frozen search/allocation checkpoint changed: {path}")
        frozen_inputs[str(path)] = {"size_bytes": path.stat().st_size,
                                    "sha256": actual, "matches_round36": True}
    return archived, {
        "training_order": source_order,
        "training_order_sha256": order_digest,
        "development_scenarios": list(r40.DEV_SCENARIOS),
        "old_checkpoint_milestones": checkpoint_rows,
        "frozen_search_allocation_inputs": frozen_inputs,
    }


def _input_module_hashes(root: Path, archived: dict[str, Any]) -> dict[str, Any]:
    expected_groups = {
        "production": archived["code_identity"]["production"],
        "imported_training_modules": archived["code_identity"]["imported_training_modules"],
        "stage_c_harness": archived["code_identity"]["stage_c_harness"],
    }
    actual_groups: dict[str, Any] = {}
    for group, expected_files in expected_groups.items():
        actual_groups[group] = {}
        for relative, expected in expected_files.items():
            path = root / relative
            if not path.is_file() and group == "stage_c_harness":
                path = root / "runs" / "comm_cost_v2_rerun_20260928" / relative
            actual = _sha256(path)
            _assert(actual == expected, f"ROUND36 dependency hash mismatch for {relative}")
            actual_groups[group][relative] = {"sha256": actual, "matches_round36": True}
    return actual_groups


def _round40_freeze(root: Path, archived: dict[str, Any], identities: dict[str, Any],
                    dependency_hashes: dict[str, Any]) -> dict[str, Any]:
    code_hashes = {name: _sha256(r40.ROUND_DIR / name) for name in R40_CODE_FILES}
    freeze = {
        "kind": "round40-first-report-only-training-freeze/1",
        "base_head": BASE_HEAD,
        "training_version": TRAINING_VERSION,
        "reward_timing_version": REWARD_TIMING_VERSION,
        "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
        "round36_frozen_config_sha256": _sha256(
            root / "runs" / "comm_final_train_20261001" / "FROZEN_CONFIG_MANIFEST.json"),
        "scenario_order_sha256": identities["training_order_sha256"],
        "old_checkpoint_parameter_identities": {
            seed: checkpoints["ep128"]["parameter_sha256"]
            for seed, checkpoints in identities["old_checkpoint_milestones"].items()
        },
        "initial_parameter_identities": {
            seed: checkpoints["ep0"]["parameter_sha256"]
            for seed, checkpoints in identities["old_checkpoint_milestones"].items()
        },
        "round40_code_sha256": code_hashes,
        "round36_dependency_hashes": dependency_hashes,
        "output_directory_repo_relative": r40.ROUND_DIR.relative_to(
            Path(__file__).resolve().parents[2]).as_posix(),
    }
    return freeze


def _checkpoint_gate(root: Path, common: Any, trainer: Any, v2_controller: Any,
                     freeze: dict[str, Any]) -> dict[str, Any]:
    old_path = (root / "runs" / "comm_final_train_20261001" / "seed11" / "checkpoints"
                / "v2_train_s11_ep128.pt")
    rejected = False
    try:
        trainer.load_checkpoint(old_path, freeze=freeze)
    except RuntimeError as error:
        rejected = "action eligibility version mismatch" in str(error)
    _assert(rejected, "actual restore entry did not reject a ROUND36 checkpoint")
    _old_controller, _old_state, old_identity = trainer.load_checkpoint_read_only(old_path)
    _assert(old_identity["optimizer_restored"] is False
            and old_identity["rng_restored"] is False
            and old_identity["read_only_for_round40"] is True,
            "OLD read-only load restored training state or lost its legacy identity")

    controller = v2_controller.fresh_controller(22)
    optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
    state = trainer.fresh_training_state(22)
    _assert(state["comm_action_eligibility"] == COMM_ACTION_ELIGIBILITY,
            "fresh state is missing the action eligibility identity")
    _assert(not optimizer.state, "fresh optimizer did not start empty")
    before = controller.parameter_summary()["parameter_sha256"]
    with tempfile.TemporaryDirectory(prefix="r40_resume_probe_", dir=r40.ROUND_DIR) as directory:
        path = Path(directory) / "synthetic_same_version_resume.pt"
        trainer.save_checkpoint(controller, optimizer, state, path, freeze=freeze)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _assert(payload.get("comm_action_eligibility") == COMM_ACTION_ELIGIBILITY,
                "checkpoint payload is missing action eligibility")
        _assert(payload["training_state"].get("comm_action_eligibility")
                == COMM_ACTION_ELIGIBILITY, "training_state is missing action eligibility")
        restored, restored_optimizer, restored_state = trainer.load_checkpoint(path,
                                                                              freeze=freeze)
        _assert(restored.parameter_summary()["parameter_sha256"] == before,
                "same-version resume changed synthetic controller contents")
        _assert(not restored_optimizer.state and restored_state["episodes_done"] == 0,
                "same-version resume restored unexpected progress")
        _assert(int(restored_state["optimizer_steps"]) == 0,
                "same-version resume changed optimizer step count")
    return {
        "old_checkpoint_cross_version_training_restore": "rejected_at_load_checkpoint_entry",
        "old_checkpoint_read_only_evaluation": old_identity,
        "same_version_recovery_entry": "passed_with_empty_optimizer_and_zero_updates",
        "synthetic_optimizer_steps": 0,
        "old_checkpoint_written": False,
    }


def _preserved_preconstruction_checkpoints(root: Path, common: Any, trainer: Any,
                                           freeze: dict[str, Any], old_identities: dict[str, Any]
                                           ) -> list[dict[str, Any]]:
    path = common.CHECKPOINTS / "v2_train_s11_ep0.pt"
    if not path.is_file():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    controller, state, identity = trainer.load_checkpoint_read_only(path)
    old_initial = old_identities["old_checkpoint_milestones"]["11"]["ep0"][
        "parameter_sha256"]
    _assert(payload.get("freeze") != freeze,
            "preserved preconstruction checkpoint unexpectedly has the current freeze")
    _assert(controller.parameter_summary()["parameter_sha256"] == old_initial,
            "preserved preconstruction checkpoint does not match registered ROUND36 ep0")
    _assert(int(state["episodes_done"]) == 0 and int(state["updates"]) == 0
            and int(state["optimizer_steps"]) == 0,
            "preserved preconstruction checkpoint contains a training update")
    _assert(identity.get("load_mode") == "read_only"
            and identity.get("optimizer_restored") is False
            and identity.get("rng_restored") is False,
            "preserved preconstruction checkpoint was not inspected read-only")
    import r40_train

    r40_train.common = common
    successor = r40_train._milestone_path(11, 0, freeze)
    _assert(successor != path and not successor.exists(),
            "new freeze would overwrite the preserved preconstruction checkpoint")
    return [{
        "path": str(path), "file_sha256": _sha256(path),
        "previous_freeze_sha256": hashlib.sha256(json.dumps(
            payload["freeze"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "parameter_sha256": controller.parameter_summary()["parameter_sha256"],
        "episodes_done": int(state["episodes_done"]), "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "load_mode": identity["load_mode"], "optimizer_restored": False,
        "rng_restored": False, "preserved_without_writeback": True,
        "current_freeze_successor_path": str(successor),
        "current_freeze_successor_does_not_exist_at_preflight": True,
    }]


def _preflight_launch_boundary(common: Any) -> dict[str, Any]:
    """Allow initial freeze, or an exact 64-reference-only continuation boundary."""
    ledger = r40.ledger_snapshot()
    slots_path = r40.ROUND_DIR / "launch_slots"
    if ledger["total_used"] == 0:
        _assert(not slots_path.exists(), "empty launch ledger has stray launch slots")
        return {"prior_environment_launches": 0, "completed_reference_rows": []}

    expected_scenarios = sorted((*r40.TRAIN_SCENARIOS, *r40.DEV_SCENARIOS))
    expected_jobs = {f"reference_immediate_s{scenario}" for scenario in expected_scenarios}
    claims = ledger["claims"]
    actual_jobs = {str(row.get("logical_job")) for row in claims}
    _assert(ledger["normal_used"] == 64 and ledger["reserve_used"] == 0
            and ledger["total_used"] == 64,
            f"revalidation is only permitted after the exact 64-reference phase: {ledger}")
    _assert(actual_jobs == expected_jobs and len(claims) == 64,
            "preexisting launch claims are not exactly the 64 immediate references")
    _assert(all(row.get("attempt") == 1 and row.get("category") == "normal"
                and row.get("status") == "complete"
                and (row.get("metadata") or {}).get("kind") == "immediate_reference"
                for row in claims),
            "preexisting reference attempts are incomplete or contain another launch class")
    artifacts = []
    for scenario in expected_scenarios:
        claim = next(row for row in claims
                     if row.get("logical_job") == f"reference_immediate_s{scenario}")
        row_path = common.ROWS / f"v2_reference_s{scenario}_a{claim['attempt']}.json"
        _assert(row_path.is_file(), f"preexisting immediate row is missing: {row_path}")
        row = _json(row_path)
        _assert(row.get("reference_arm") == "immediate"
                and int(row.get("scenario_seed", -1)) == scenario
                and row.get("training_version") == TRAINING_VERSION
                and row.get("comm_action_eligibility") == COMM_ACTION_ELIGIBILITY,
                f"preexisting immediate row identity differs: {row_path}")
        artifacts.append({"scenario_seed": scenario, "path": str(row_path),
                          "sha256": _sha256(row_path), "attempt": int(claim["attempt"])})
    all_rows = sorted(common.ROWS.glob("*.json"))
    _assert(len(all_rows) == 64 and {path.resolve() for path in all_rows}
            == {Path(row["path"]).resolve() for row in artifacts},
            "the prior phase contains rows outside the immediate reference block")
    return {"prior_environment_launches": 64,
            "completed_reference_rows": artifacts,
            "training_or_evaluation_launches_before_revalidation": 0,
            "preexisting_reserve_launches": 0}


def run(input_root: str | Path) -> dict[str, Any]:
    root = Path(input_root).resolve()
    common = r40.bootstrap(root)
    from r40_eligibility import COMM_ACTION_ELIGIBILITY as imported_eligibility
    from r40_eligibility import FirstReportOnlyCollector as imported_collector
    from r40_versions import install

    global FirstReportOnlyCollector
    _assert(imported_eligibility == COMM_ACTION_ELIGIBILITY,
            "preflight and collector eligibility identity differ")
    FirstReportOnlyCollector = imported_collector
    runner = importlib.import_module("v2_run")
    trainer = importlib.import_module("v2_trainer")
    v2_controller = importlib.import_module("v2_controller")
    inherited_path = root / "runs" / "comm_final_train_20261001" / "FROZEN_CONFIG_MANIFEST.json"
    archived = _json(inherited_path)
    prior_freeze_path = r40.ROUND_DIR / "ROUND40_FROZEN_CONFIG.json"
    prior_freeze_doc = _json(prior_freeze_path) if prior_freeze_path.is_file() else None
    prior_freeze_sha256 = _sha256(prior_freeze_path) if prior_freeze_path.is_file() else None
    source_head = _git(root, "rev-parse", "HEAD")
    worktree = Path(__file__).resolve().parents[2]
    worktree_parent = _git(worktree, "rev-parse", "HEAD")
    _assert(source_head == BASE_HEAD, f"canonical input checkout HEAD {source_head} != {BASE_HEAD}")
    _assert(_git(worktree, "merge-base", "--is-ancestor", ROUND40_BASIS_COMMIT, "HEAD") == "",
            f"ROUND40 execution branch no longer descends from its basis commit {ROUND40_BASIS_COMMIT}")
    _assert(_git(root, "branch", "--show-current") == "exp/comm-refresh-held-mechanism",
            "canonical read-only source checkout is not on the registered ROUND39 branch")

    order = common.scenario_order()
    order_digest = hashlib.sha256(json.dumps(order).encode("utf-8")).hexdigest()
    _assert(order_digest == archived["data_order"]["actual_sha256"],
            "frozen scene order is not identical to ROUND36")
    identities: dict[str, Any] = {
        "training_order": order,
        "training_order_sha256": order_digest,
        "development_scenarios": list(r40.DEV_SCENARIOS),
    }
    # Checkpoint files are read-only inputs; only payload/controller identities are decoded.
    checkpoint_source = _archive_identities(root, common, trainer, v2_controller)[1]
    identities.update(checkpoint_source)
    dependency_hashes = _input_module_hashes(root, archived)
    freeze = _round40_freeze(root, archived, identities, dependency_hashes)
    install(trainer, runner, freeze=freeze)

    optimizer_steps_called = 0
    with patch.object(torch.optim.Adam, "step",
                      side_effect=AssertionError("preflight must never call optimizer.step")):
        collector_results = _collector_preflight(common, trainer, v2_controller)
        relay_results = _relay_preflight()
        objective_results = _objective_preflight(common)
        bundle_results = _bundle_preflight(trainer)
        optimizer_results = _optimizer_preflight(v2_controller)
        checkpoint_results = _checkpoint_gate(root, common, trainer, v2_controller, freeze)

    preserved_preconstruction_checkpoints = _preserved_preconstruction_checkpoints(
        root, common, trainer, freeze, checkpoint_source)

    # A non-constructive syntax/import guard on the R40 modules.
    for name in R40_CODE_FILES:
        source = r40.ROUND_DIR / name
        compile(source.read_text(encoding="utf-8"), str(source), "exec")

    prior_phase = _preflight_launch_boundary(common)
    prior_code = ((prior_freeze_doc or {}).get("round40", {}).get("freeze_identity", {})
                  .get("round40_code_sha256", {}))
    current_code = {name: _sha256(r40.ROUND_DIR / name) for name in R40_CODE_FILES}
    unchanged_method_code = {
        name: {"previous_sha256": prior_code[name], "current_sha256": current_code[name],
               "unchanged": prior_code[name] == current_code[name]}
        for name in ("r40_common.py", "r40_eligibility.py", "r40_versions.py", "r40_run.py")
        if name in prior_code
    }
    if prior_phase["prior_environment_launches"]:
        _assert(prior_freeze_doc is not None,
                "reference continuation lacks its previously committed freeze")
        _assert(_git(worktree, "merge-base", "--is-ancestor",
                      REFERENCE_PHASE_FREEZE_COMMIT, "HEAD") == "",
                "reference continuation does not descend from its registered freeze commit")
        changed_since_reference = set(filter(None, _git(
            worktree, "diff", "--name-only", REFERENCE_PHASE_FREEZE_COMMIT, "HEAD").splitlines()))
        allowed_driver_changes = {
            "runs/comm_first_report_training_ablation_20261003/r40_train.py",
            "runs/comm_first_report_training_ablation_20261003/r40_worker.py",
            "runs/comm_first_report_training_ablation_20261003/r40_preflight.py",
            "runs/comm_first_report_training_ablation_20261003/r40_analysis.py",
        }
        _assert(changed_since_reference <= allowed_driver_changes,
                f"unexpected files changed since completed reference phase: {changed_since_reference}")
        _assert(all(row["unchanged"] for row in unchanged_method_code.values())
                and len(unchanged_method_code) == 4,
                "collector, relay version, or runner behavior changed after references")
    code_hashes = current_code
    identities["round36_checkpoint_inputs"] = checkpoint_source["old_checkpoint_milestones"]
    identities["frozen_search_allocation_inputs"] = checkpoint_source[
        "frozen_search_allocation_inputs"]
    identities["round36_imported_dependency_hashes"] = dependency_hashes
    identities["round40_code_sha256"] = code_hashes

    inherited_config = {
        "source_manifest_sha256": _sha256(inherited_path),
        "source_manifest": archived,
    }
    round40_configuration = {
        "round": "40",
        "training_seeds": list(r40.TRAIN_SEEDS),
        "episodes_per_seed": r40.TRAIN_EPISODES,
        "episodes_per_update": r40.EPISODES_PER_UPDATE,
        "training_order_sha256": order_digest,
        "training_scenarios": [min(r40.TRAIN_SCENARIOS), max(r40.TRAIN_SCENARIOS),
                               len(r40.TRAIN_SCENARIOS)],
        "development_scenarios": [min(r40.DEV_SCENARIOS), max(r40.DEV_SCENARIOS),
                                  len(r40.DEV_SCENARIOS)],
        "normal_launch_plan": r40.plan()["normal"],
        "technical_reserve_max": r40.RESERVE_CAP,
        "hard_launch_cap": r40.HARD_CAP,
        "training_version": TRAINING_VERSION,
        "reward_timing_version": REWARD_TIMING_VERSION,
        "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
        "allowed_differences_vs_round36": [
            "training action eligibility: first-report-only/1",
            "training version and checkpoint/freeze/evaluation identity fields",
            "ROUND40 output and launch accounting",
            "fixed evaluation matrix specified by ROUND40 execution basis",
        ],
        "unexpected_research_configuration_differences": [],
        "phase_continuation": {
            "prior_freeze_file_sha256": prior_freeze_sha256,
            "prior_reference_phase_commit": (
                REFERENCE_PHASE_FREEZE_COMMIT if prior_phase["prior_environment_launches"]
                else None),
            "prior_phase": prior_phase,
            "preserved_preconstruction_checkpoints": preserved_preconstruction_checkpoints,
            "unchanged_method_code_since_reference_phase": unchanged_method_code,
            "code_revisions_are_driver_or_audit_only": True,
        },
        "objective": archived["objective_and_dual"],
        "ppo_hyperparameters": archived["hyperparameters"],
        "world_config_comparison": archived["world_config_comparison"],
        "steps_per_episode": archived["steps_per_episode"],
        "row_filtering": archived["row_filtering"],
        "reward_capture": {
            "version": REWARD_TIMING_VERSION,
            "implementation": "PublishedProgressRewardCollector",
            "runner_source_unchanged": True,
        },
        "freeze_identity": freeze,
    }
    report = {
        "kind": "round40-pure-compute-preflight/1",
        "status": "passed",
        "environment_constructed": False,
        "joint_run_constructed": False,
        "simulator_started": False,
        "optimizer_step_calls": optimizer_steps_called,
        "synthetic_optimizer_updates": 0,
        "canonical_input_root": str(root),
        "canonical_input_head": source_head,
        "round40_basis_commit": worktree_parent,
        "preexisting_environment_launches": prior_phase["prior_environment_launches"],
        "preexisting_reference_rows": prior_phase["completed_reference_rows"],
        "preserved_preconstruction_checkpoints": preserved_preconstruction_checkpoints,
        "python": sys.version,
        "torch": torch.__version__,
        "device": "cpu",
        "configuration_differences": round40_configuration["allowed_differences_vs_round36"],
        "checkpoint_identities": identities,
        "tests": {
            "collector_and_batch_path": collector_results,
            "relay_authoritative_eligibility": relay_results,
            "objective_and_positive_cost_sign": objective_results,
            "reward_bundle_prefix_same_tick_terminal": bundle_results,
            "optimizer_scope": optimizer_results,
            "checkpoint_version_boundary": checkpoint_results,
        },
        "prelaunch_gate": "passed; requires local commit of this amended freeze before the next environment",
    }
    (r40.ROUND_DIR / "ROUND40_FROZEN_CONFIG.json").write_text(
        json.dumps({"inherited_round36": inherited_config,
                    "round40": round40_configuration}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (r40.ROUND_DIR / "ROUND40_PREFLIGHT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (r40.ROUND_DIR / "SCENARIO_ORDER.json").write_text(
        json.dumps({"training_seed": common.DATA_ORDER_SEED, "order": order,
                    "sha256": order_digest, "development": list(r40.DEV_SCENARIOS)},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (r40.ROUND_DIR / "TRAINING_VERSION_MANIFEST.json").write_text(
        json.dumps({"training_version": TRAINING_VERSION,
                    "reward_timing_version": REWARD_TIMING_VERSION,
                    "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
                    "old_round36_training_version": OLD_TRAINING_VERSION,
                    "old_round36_action_space": "first_emission_and_refresh_candidates",
                    "old_checkpoint_training_restore": "rejected",
                    "old_checkpoint_read_only_eval": "allowed",
                    "round40_checkpoint_resume": "same training version, eligibility and freeze only",
                    "freeze_identity": freeze}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    args = parser.parse_args(argv)
    report = run(args.input_root)
    print(json.dumps({"status": report["status"],
                      "optimizer_step_calls": report["optimizer_step_calls"],
                      "scenario_order_sha256": report["checkpoint_identities"][
                          "training_order_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
