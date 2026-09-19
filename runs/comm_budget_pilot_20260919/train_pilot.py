"""A small PPO controller for the uplink, plus the checks that must pass before it trains.

The decision process is the one the design fixed: at every joint step the sender is shown the
targets it has something unsaid about and chooses *send* or *hold* for each. Airtime is
shared, so a step's action is the whole subset, and what will not fit is trimmed by the same
executor the rule arms use.

The safety bound is an action mask, not a reward term: a change held past ``MAX_WAIT_STEPS``
(less while a task is urgent) must go out, and if the credit cannot pay for it, it does not go
out and the refusal is counted. The same mask applies to every arm.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pilot_common as common
import pilot_env

from integrated_sar.comm_learning import FEATURE_NAMES, LearnedSendPolicy, features  # noqa: E402
from integrated_sar.comm_policy import SendObservation  # noqa: E402

FEATURE_DIM = len(FEATURE_NAMES)
HIDDEN = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_BONUS = 0.01
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5
EPOCHS_PER_BATCH = 4
MINIBATCH_STEPS = 256
PILOT_ENVIRONMENT_STEPS = 100_000
EVALUATE_EVERY_STEPS = 20_000
EPISODES_PER_UPDATE = 4
EXPLORATION_EPSILON = 0.10
FEATURE_LAYOUT_NAME = "comm-send-state/1"
MAX_WAIT_STEPS = pilot_env.MAX_WAIT_STEPS
MAX_WAIT_WHEN_URGENT_STEPS = pilot_env.MAX_WAIT_WHEN_URGENT_STEPS
URGENCY_STEPS = pilot_env.URGENCY_STEPS
SEND, HOLD = 1, 0


class Controller:
    """Shared trunk over candidates, a send/hold head, and a pooled value for the step."""

    def __init__(self, seed: int) -> None:
        torch.manual_seed(int(seed))
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_DIM, HIDDEN),
            torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, HIDDEN),
            torch.nn.Tanh(),
        )
        self.action_head = torch.nn.Linear(HIDDEN, 2)
        self.value_head = torch.nn.Linear(HIDDEN, 1)
        self.parameters = [
            *self.trunk.parameters(),
            *self.action_head.parameters(),
            *self.value_head.parameters(),
        ]

    def forward(self, batch: torch.Tensor):
        hidden = self.trunk(batch)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)

    def value_of(self, matrix: np.ndarray) -> float:
        with torch.no_grad():
            _, values = self.forward(torch.as_tensor(matrix, dtype=torch.float32))
        return float(values.mean().item())

    def state_dict(self) -> dict[str, Any]:
        return {
            "trunk": self.trunk.state_dict(),
            "action_head": self.action_head.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.trunk.load_state_dict(payload["trunk"])
        self.action_head.load_state_dict(payload["action_head"])
        self.value_head.load_state_dict(payload["value_head"])


def forced(item: Any) -> bool:
    """The safety mask: a change may not be held past the bound, urgent or not."""
    urgent = 0.0 <= float(item.remaining_time) <= URGENCY_STEPS
    limit = MAX_WAIT_WHEN_URGENT_STEPS if urgent else MAX_WAIT_STEPS
    return int(item.age_steps) >= limit


class Collector:
    """Adapts a decision function to the relay, records the tape, and counts its own cost.

    ``controller=None`` makes it a plain rule: ``always_send`` is the unbudgeted flood,
    ``never_send`` with ``mask=False`` is the silence check, ``never_send`` with the mask on
    is "say nothing until the bound forces it", which is the worst policy that is still legal.
    """

    def __init__(
        self,
        controller: Controller | None = None,
        *,
        greedy: bool = False,
        epsilon: float = 0.0,
        mask: bool = True,
        always_send: bool = False,
        never_send: bool = False,
    ) -> None:
        self.controller = controller
        self.greedy = bool(greedy)
        self.epsilon = float(epsilon)
        self.mask = bool(mask)
        self.always_send = bool(always_send)
        self.never_send = bool(never_send)
        self.records: list[dict[str, Any]] = []
        self.send_steps: list[tuple[int, tuple[str, ...]]] = []
        self.decide_seconds = 0.0
        self.decisions = 0
        self.candidates = 0
        self.forced_sends = 0

    def policy(self) -> LearnedSendPolicy:
        return LearnedSendPolicy(self.decide)

    def choose(self, observation: SendObservation) -> tuple[list[str], dict[str, int], list[str], float, float]:
        items = list(observation.items)
        must = [item.target_id for item in items if self.mask and forced(item)]
        self.forced_sends += len(must)
        if self.controller is None or self.always_send or self.never_send:
            if self.never_send:
                actions = {item.target_id: (SEND if item.target_id in must else HOLD) for item in items}
            else:
                actions = {item.target_id: SEND for item in items}
            return (
                [key for key, value in actions.items() if value == SEND],
                actions,
                [item.target_id for item in items if item.target_id not in must],
                0.0,
                0.0,
            )
        matrix = np.asarray([features(observation, item) for item in items], dtype=np.float32)
        batch = torch.as_tensor(matrix, dtype=torch.float32)
        logits, values = self.controller.forward(batch)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
        actions: dict[str, int] = {}
        log_prob = 0.0
        entropy = 0.0
        for index, item in enumerate(items):
            if item.target_id in must:
                actions[item.target_id] = SEND
                continue
            if self.greedy:
                action = SEND if probabilities[index, SEND] >= probabilities[index, HOLD] else HOLD
            elif random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                action = int(torch.multinomial(probabilities[index].clamp_min(1e-9), 1).item())
            actions[item.target_id] = action
            log_prob += float(log_probabilities[index, action].item())
            entropy += float(-(probabilities[index] * log_probabilities[index]).sum().item())
        chosen = [item.target_id for item in items if actions[item.target_id] == SEND]
        return chosen, actions, [item.target_id for item in items if item.target_id not in must], log_prob, (
            float(values.mean().item()) if items else 0.0
        )

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        started = time.perf_counter()
        self.decisions += 1
        self.candidates += len(observation.items)
        chosen, actions, free, log_prob, value = self.choose(observation)
        if self.controller is not None and observation.items:
            self.records.append(
                {
                    "step": int(observation.step),
                    "targets": [item.target_id for item in observation.items],
                    "features": [list(features(observation, item)) for item in observation.items],
                    "actions": actions,
                    "free": free,
                    "log_prob": log_prob,
                    "value": value,
                }
            )
        if chosen:
            self.send_steps.append((int(observation.step), tuple(chosen)))
        self.decide_seconds += time.perf_counter() - started
        return tuple(chosen)

    def timing(self) -> dict[str, float]:
        return {
            "decisions": self.decisions,
            "candidates_seen": self.candidates,
            "forced_sends": self.forced_sends,
            "decide_milliseconds_per_call": round(
                1000.0 * self.decide_seconds / max(1, self.decisions), 4
            ),
        }


def gae(rewards: list[float], values: list[float]) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    running = 0.0
    for index in reversed(range(len(rewards))):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + GAMMA * next_value - values[index]
        running = delta + GAMMA * GAE_LAMBDA * running
        advantages[index] = running
    return advantages, [advantage + value for advantage, value in zip(advantages, values)]


def run_one(
    seed: int,
    *,
    collector: Collector,
    budget_bytes: int | None,
    directory: Path,
    label: str = "learned",
    information: str = "isolated",
    delivery_probability: float = 1.0,
    policy: str = "learned",
    max_wait_steps: int | None = pilot_env.MAX_WAIT_STEPS,
    urgent_wait_steps: int | None = pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
) -> dict[str, Any]:
    rewards: dict[int, float] = {}

    def on_step(step: int, gained: float) -> None:
        rewards[step] = rewards.get(step, 0.0) + gained

    row = pilot_env.run_episode(
        seed,
        directory=directory,
        label=label,
        policy=policy,
        budget_bytes=budget_bytes,
        controller=collector.policy(),
        on_step=on_step,
        information=information,
        delivery_probability=delivery_probability,
        max_wait_steps=max_wait_steps,
        urgent_wait_steps=urgent_wait_steps,
    )
    by_step = {int(record["step"]): record for record in collector.records}
    decisions = sorted(by_step)
    total = float(sum(rewards.values()))
    if decisions:
        # Every reward belongs to the next decision that could still act on it; rewards after
        # the last decision are credited to that decision, so the tape sums to the score.
        aligned = []
        previous = 0
        for step in decisions:
            aligned.append(sum(value for key, value in rewards.items() if previous < key <= step))
            previous = step
        aligned[-1] += total - sum(aligned)
        assert abs(sum(aligned) - total) < 1e-9
    else:
        aligned = []
    return {
        "row": row,
        "decisions": [by_step[step] for step in decisions],
        "rewards": aligned,
        "steps": decisions,
    }


def update(controller: Controller, optimizer: torch.optim.Optimizer, episodes: list[dict[str, Any]]) -> dict[str, float]:
    flat: list[dict[str, Any]] = []
    for episode in episodes:
        values = [record["value"] for record in episode["decisions"]]
        rewards = episode["rewards"]
        if not values:
            continue
        advantages, returns = gae(rewards, values)
        scale = max(1e-6, float(np.std(advantages)))
        for record, advantage, target_return in zip(episode["decisions"], advantages, returns):
            flat.append(
                {
                    "features": np.asarray(record["features"], dtype=np.float32),
                    "targets": record["targets"],
                    "actions": record["actions"],
                    "free": set(record["free"]),
                    "log_prob": record["log_prob"],
                    "advantage": advantage / scale,
                    "return": target_return,
                }
            )
    if not flat:
        return {"minibatches": 0, "policy_surrogate": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    order = list(range(len(flat)))
    totals = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0}
    minibatches = 0
    for _ in range(EPOCHS_PER_BATCH):
        random.shuffle(order)
        for start in range(0, len(order), MINIBATCH_STEPS):
            chunk = [flat[index] for index in order[start : start + MINIBATCH_STEPS]]
            stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
            logits, candidate_values = controller.forward(stacked)
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            row = 0
            new_log_prob: list[torch.Tensor] = []
            entropy: list[torch.Tensor] = []
            pooled: list[torch.Tensor] = []
            for item in chunk:
                count = len(item["targets"])
                rows = torch.arange(row, row + count)
                row += count
                picked = torch.tensor(
                    [item["actions"][target] for target in item["targets"]], dtype=torch.long
                )
                usable = torch.tensor(
                    [target in item["free"] for target in item["targets"]], dtype=torch.float32
                )
                gathered = log_probabilities[rows, picked] * usable
                new_log_prob.append(gathered.sum())
                entropy.append((-(probabilities[rows] * log_probabilities[rows]).sum(dim=-1) * usable).sum())
                pooled.append(candidate_values[rows].mean())
            new_log_prob_t = torch.stack(new_log_prob)
            entropy_t = torch.stack(entropy)
            old_log_prob = torch.tensor([item["log_prob"] for item in chunk], dtype=torch.float32)
            advantage = torch.tensor([item["advantage"] for item in chunk], dtype=torch.float32)
            target_returns = torch.tensor([item["return"] for item in chunk], dtype=torch.float32)
            ratio = torch.exp(new_log_prob_t - old_log_prob)
            surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * advantage)
            value_loss = ((torch.stack(pooled) - target_returns) ** 2).mean()
            loss = -surrogate.mean() + VALUE_COEFFICIENT * value_loss - ENTROPY_BONUS * entropy_t.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters, MAX_GRAD_NORM)
            optimizer.step()
            totals["surrogate"] += float(surrogate.mean().item())
            totals["value"] += float(value_loss.item())
            totals["entropy"] += float(entropy_t.mean().item())
            totals["kl"] += float((old_log_prob - new_log_prob_t).mean().item())
            minibatches += 1
    return {
        "minibatches": minibatches,
        "policy_surrogate": round(totals["surrogate"] / minibatches, 5),
        "value_loss": round(totals["value"] / minibatches, 5),
        "entropy": round(totals["entropy"] / minibatches, 5),
        "approx_kl": round(totals["kl"] / minibatches, 6),
    }


def train_feature_dim() -> int:
    return FEATURE_DIM


def evaluate(
    controller: Controller | None,
    seeds: list[int],
    *,
    budget_bytes: int,
    directory: Path,
    information: str = "isolated",
    delivery_probability: float = 1.0,
) -> dict[str, Any]:
    rows = []
    timings = []
    forced = 0
    if controller is not None:
        # One throw-away forward: torch lazy-initialises on the first call, and that cost is
        # not the per-decision inference cost anyone would deploy.
        warmup = np.zeros((1, train_feature_dim()), dtype=np.float32)
        with torch.no_grad():
            controller.forward(torch.as_tensor(warmup))
    for seed in seeds:
        collector = Collector(controller, greedy=True)
        result = run_one(
            seed,
            collector=collector,
            budget_bytes=budget_bytes,
            directory=directory,
            label=f"learned_eval_{information}_{delivery_probability}",
            information=information,
            delivery_probability=delivery_probability,
        )
        rows.append(result["row"])
        timings.append(collector.timing())
        forced += collector.forced_sends
    def mean(key: Any) -> float:
        return round(float(np.mean([key(row) for row in rows])), 4)

    return {
        "episodes": len(rows),
        "information": information,
        "delivery_probability": delivery_probability,
        "mean_on_time": mean(lambda row: row["task"]["on_time"]),
        "mean_expired": mean(lambda row: row["task"]["expired"]),
        "mean_targets": mean(lambda row: row["task"]["targets"]),
        "mean_airtime_bytes": mean(lambda row: row["spent_bytes"]),
        "mean_attempts": mean(lambda row: row["ledger"]["transmission_attempts"]),
        "mean_applied": mean(lambda row: row["ledger"]["deliveries_applied"]),
        "mean_budget_refusals": mean(lambda row: row["ledger"]["budget_refused_sends"]),
        "mean_forced_sends": round(forced / max(1, len(rows)), 3),
        "mean_belief_age": mean(lambda row: row["belief"]["belief_age_steps_mean"] or 0.0),
        "mean_unknown_to_everyone": mean(lambda row: row["belief"]["unknown_to_everyone_steps"] or 0.0),
        "decide_milliseconds_per_call": round(
            1000.0 * sum(item["decide_milliseconds_per_call"] * item["decisions"] for item in timings)
            / max(1, sum(item["decisions"] for item in timings)),
            4,
        ),
    }


def train(
    *,
    seed: int,
    budget_bytes: int,
    train_seeds: list[int],
    validation_seeds: list[int],
    directory: Path,
    steps_cap: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % 2**31)
    controller = Controller(seed)
    optimizer = torch.optim.Adam(controller.parameters, lr=LEARNING_RATE)
    path = Path(directory) / f"pilot_seed{seed}.pt"
    directory.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "seed": seed,
        "budget_bytes": budget_bytes,
        "environment_steps": 0,
        "episodes": 0,
        "cursor": 0,
        "updates": [],
        "evaluations": [],
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        controller.load_state_dict(payload["controller"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["state"])
        print(f"[seed {seed}] resumed at {state['environment_steps']} environment steps", flush=True)
    # Each training seed owns its own scratch directory. Sharing one made three processes
    # append to the same lifecycle log and read each other's half-written lines.
    directory = Path(directory) / f"seed{seed}"
    while state["environment_steps"] < steps_cap:
        batch = [
            train_seeds[(state["cursor"] + index) % len(train_seeds)]
            for index in range(EPISODES_PER_UPDATE)
        ]
        state["cursor"] = (state["cursor"] + EPISODES_PER_UPDATE) % len(train_seeds)
        episodes = [
            run_one(
                seed_value,
                collector=Collector(controller, epsilon=EXPLORATION_EPSILON),
                budget_bytes=budget_bytes,
                directory=directory / "rollout",
                label=f"learned_train{seed}",
            )
            for seed_value in batch
        ]
        report = update(controller, optimizer, episodes)
        spent = sum(int(item["row"]["spent_bytes"]) for item in episodes)
        credit = budget_bytes * len(batch)
        report.update(
            {
                "at_steps": state["environment_steps"] + common.STREAM_STEPS * len(batch),
                "batch_on_time": sum(int(item["row"]["task"]["on_time"]) for item in episodes),
                "batch_targets": sum(int(item["row"]["task"]["targets"]) for item in episodes),
                "batch_budget_use": round(spent / credit, 4),
                "batch_credit_overrun": int(spent - credit) if spent > credit else 0,
            }
        )
        optimizer_step_cost = max(0.001, sum(item["row"]["seconds"] for item in episodes))
        state["environment_steps"] += common.STREAM_STEPS * len(batch)
        state["episodes"] += len(batch)
        state["updates"].append(report)
        torch.save(
            {"controller": controller.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, path
        )
        print(
            f"[seed {seed}] steps={state['environment_steps']:>6} episodes={state['episodes']:>4} "
            f"on_time={report['batch_on_time']}/{report['batch_targets']} "
            f"budget_use={report['batch_budget_use']} kl={report['approx_kl']} "
            f"entropy={report['entropy']} value={report['value_loss']} "
            f"({optimizer_step_cost:.1f}s)",
            flush=True,
        )
        crossed = (
            state["environment_steps"] // EVALUATE_EVERY_STEPS
            > (state["environment_steps"] - common.STREAM_STEPS * len(batch)) // EVALUATE_EVERY_STEPS
        )
        if crossed and validation_seeds:
            result = evaluate(controller, validation_seeds, budget_bytes=budget_bytes, directory=directory / "eval")
            result["at_steps"] = state["environment_steps"]
            state["evaluations"].append(result)
            torch.save(
                {"controller": controller.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, path
            )
            print(f"[seed {seed}] validation {json.dumps(result, ensure_ascii=False)}", flush=True)
    return state


SENDER_SIDE_FIELDS = {
    "SendItem": {
        "target_id", "ready_step", "age_steps", "pending_event", "changed_view",
        "first_emission", "revision", "revision_delta", "stage_index", "stage_changed",
        "progress", "progress_delta", "requirements_changed", "members_changed", "deadline",
        "remaining_time", "sent_count", "last_sent_step", "since_last_emission_time",
        "has_future_requirements",
    },
    "SendObservation": {
        "step", "current_time", "items", "remaining_budget_bytes", "budget_bytes_per_episode",
        "capacity_remaining", "backlog_targets", "subscribers",
    },
}


def leakage_audit() -> dict[str, Any]:
    """A controller can only be fed what the sender legitimately holds.

    The durable guard is the field list: ``features`` takes a SendObservation and a SendItem,
    and neither type may carry a receiver-side quantity -- no robot id, position, belief, or
    delivery outcome, because the uplink has no acknowledgement to carry one back. If someone
    adds such a field, this check fails and the addition has to be argued about instead of
    quietly becoming an observation.
    """
    from dataclasses import fields as record_fields

    from integrated_sar.comm_policy import SendItem, SendObservation

    found = {
        "SendItem": {field.name for field in record_fields(SendItem)},
        "SendObservation": {field.name for field in record_fields(SendObservation)},
    }
    unexpected = {
        name: sorted(found[name] - SENDER_SIDE_FIELDS[name]) for name in found if found[name] - SENDER_SIDE_FIELDS[name]
    }
    missing = {
        name: sorted(SENDER_SIDE_FIELDS[name] - found[name]) for name in found if SENDER_SIDE_FIELDS[name] - found[name]
    }
    return {
        "sender_side_fields": found,
        "fields_not_on_the_allow_list": unexpected,
        "allow_list_entries_that_no_longer_exist": missing,
        "clean": not unexpected and not missing,
    }


def smoke(args: argparse.Namespace) -> int:
    """Every check fails the run. None of them is a warning."""
    directory = Path(args.directory) / "smoke"
    directory.mkdir(parents=True, exist_ok=True)
    seed = common.TRAIN_SEEDS[0]
    budget = args.budget_bytes
    failures: list[str] = []
    results: dict[str, Any] = {}

    # 1. budget never exceeded, and airtime is billed even when nothing arrives.
    flooded = Collector(None, always_send=True)
    flood = run_one(seed, collector=flooded, budget_bytes=budget, directory=directory)
    results["always_send"] = {
        "on_time": flood["row"]["task"]["on_time"],
        "spent": flood["row"]["spent_bytes"],
        "attempts": flood["row"]["ledger"]["transmission_attempts"],
    }
    if flood["row"]["spent_bytes"] > budget:
        failures.append("the executor let a run spend more than its credit")

    # 2. silence is not secretly rewarded: with no safety mask, holding everything informs nobody.
    held = Collector(None, never_send=True, mask=False)
    hold = run_one(
        seed,
        collector=held,
        budget_bytes=budget,
        directory=directory,
        max_wait_steps=None,
        urgent_wait_steps=None,
    )
    results["hold_unmasked"] = {
        "on_time": hold["row"]["task"]["on_time"],
        "targets_known": hold["row"]["belief"]["known_robot_target_pairs"],
        "attempts": hold["row"]["ledger"]["transmission_attempts"],
    }
    if hold["row"]["task"]["on_time"] > 0 or hold["row"]["belief"]["known_robot_target_pairs"] > 0:
        failures.append(
            "an all-hold sender still informed someone or completed a target -- the reward is "
            "not grounded in the messages"
        )

    # 3. "send everything" must not be the only way to do well: if a flood spends the credit on
    # nothing, that is a result to report, not a wiring failure. What has to hold is that the
    # forced-safety policy really does send -- otherwise the mask is dead code.
    bounded = Collector(None, never_send=True, mask=True)
    bound = run_one(seed, collector=bounded, budget_bytes=budget, directory=directory)
    results["hold_to_bound"] = {
        "on_time": bound["row"]["task"]["on_time"],
        "spent": bound["row"]["spent_bytes"],
        "forced_sends": bounded.forced_sends,
    }
    if bounded.forced_sends == 0 or bound["row"]["ledger"]["emissions"] == 0:
        failures.append("the safety mask never forced a send -- the bound is not being enforced")

    # 4. actions really move deliveries: opposite policies must produce different schedules.
    if flood["row"]["ledger"]["transmission_attempts"] == bound["row"]["ledger"]["transmission_attempts"] and flood[
        "row"
    ]["ledger"]["emissions"] == bound["row"]["ledger"]["emissions"]:
        failures.append("always-send and hold-to-bound produced the same schedule")

    # 5. nothing receiver-side can reach the controller in the first place.
    audit = leakage_audit()
    results["leakage_audit"] = {
        key: value for key, value in audit.items() if key != "sender_side_fields"
    }
    if not audit["clean"]:
        failures.append(
            f"the sender's observation type is no longer sender-side only: {audit['fields_not_on_the_allow_list']}"
        )

    # 6. a parameter update happens, and the return equals the scored metric.
    controller = Controller(seed=7)
    collector = Collector(controller, epsilon=1.0)
    sample = run_one(seed, collector=collector, budget_bytes=budget, directory=directory)
    before = controller.state_dict()["trunk"]["0.weight"].clone()
    optimizer = torch.optim.Adam(controller.parameters, lr=LEARNING_RATE)
    report = update(controller, optimizer, [sample])
    after = controller.state_dict()["trunk"]["0.weight"]
    delta = float((after - before).abs().max().item())
    results["update"] = {**report, "weight_delta": round(delta, 6)}
    if delta < 1e-6 or report["minibatches"] == 0:
        failures.append("no parameter change after a PPO update")
    if int(sample["row"]["return_on_time"]) != int(sample["row"]["task"]["on_time"]):
        failures.append("episode return disagrees with the scored on-time count")
    if abs(sum(sample["rewards"]) - float(sample["row"]["return_on_time"])) > 1e-9:
        failures.append("the per-step reward tape does not sum to the episode's score")

    results["failures"] = failures
    results["budget_bytes"] = budget
    results["constants"] = {
        "max_wait_steps": MAX_WAIT_STEPS,
        "urgent_wait_steps": MAX_WAIT_WHEN_URGENT_STEPS,
        "urgency_steps": URGENCY_STEPS,
        "feature_layout": FEATURE_LAYOUT_NAME,
        "feature_names": list(FEATURE_NAMES),
    }
    common.write_json(common.HERE / "PILOT_SMOKE.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), default="smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tier", default=None, help="a locked tier name, e.g. 50pct")
    parser.add_argument("--budget-bytes", type=int, default=None)
    parser.add_argument("--directory", default=str(common.HERE / "train"))
    parser.add_argument("--steps", type=int, default=PILOT_ENVIRONMENT_STEPS)
    parser.add_argument("--validation-episodes", type=int, default=8)
    args = parser.parse_args(argv)
    budget = args.budget_bytes or locked_budget(args.tier)
    directory = Path(args.directory)
    if args.mode == "smoke":
        args.budget_bytes = budget
        return smoke(args)
    validation = list(common.VALIDATION_SEEDS)[: args.validation_episodes]
    if args.mode == "evaluate":
        path = directory / f"pilot_seed{args.seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        controller = Controller(args.seed)
        controller.load_state_dict(payload["controller"])
        print(json.dumps(evaluate(controller, validation, budget_bytes=budget, directory=directory / "final_eval"), ensure_ascii=False, indent=1))
        return 0
    state = train(
        seed=args.seed,
        budget_bytes=budget,
        train_seeds=list(common.TRAIN_SEEDS),
        validation_seeds=validation,
        directory=directory,
        steps_cap=args.steps,
    )
    summary = {key: state[key] for key in ("seed", "budget_bytes", "environment_steps", "episodes")}
    summary["last_evaluation"] = state["evaluations"][-1] if state["evaluations"] else None
    summary["final_update"] = state["updates"][-1] if state["updates"] else None
    common.write_json(directory / f"pilot_seed{args.seed}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def locked_budget(tier: str | None) -> int:
    locked = json.loads((common.HERE / "COMM_BUDGET_LOCKED.json").read_text(encoding="utf-8"))
    name = tier or json.loads((common.HERE / "COMM_BUDGET_CALIBRATION.json").read_text(encoding="utf-8"))["tier_selection"]["selected_tier"]
    return int(locked["budgets_bytes"][name])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
