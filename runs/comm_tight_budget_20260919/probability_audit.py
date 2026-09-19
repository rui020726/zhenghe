"""§2's evidence: measure what a collector draws against what it writes down.

The defect is not visible in the usual health checks. A rollout logged with the wrong behaviour
probability still has ratio 1 on the first minibatch, still moves weights, and still learns
*something* -- which is exactly why round 3 could notice it, write it down, and train on it
anyway. What exposes it is comparing the frequency of the actions taken with the frequency the
logged numbers describe.

Two collectors are measured on the same episodes and the same network:

* ``legacy`` -- round 2's, with the epsilon-greedy overlay, on real rollout episodes;
* ``fixed``  -- :class:`onpolicy.OnPolicyCollector`, same episodes.

For each, every free candidate's action is compared with the policy's own probability of it,
recomputed from the recorded features. Under a correct sampler the two agree up to sampling error
and the implied log-probability of the drawn action is the number in the tape. Under the mixture
the drawn actions are systematically flatter than ``pi`` -- the empirical send rate drifts toward
0.5 at a rate the logged ``log pi`` cannot describe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tight_common as common

import pilot_env
import train_pilot
import onpolicy
from onpolicy import joint_log_prob

#: What round 2 trained with. Read from the frozen module so this file reports the real constant
#: rather than a remembered one.
EPSILON = train_pilot.EXPLORATION_EPSILON
FEATURE_DIM = train_pilot.FEATURE_DIM


def _recompute(controller: Any, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Per-candidate ``[p(hold), p(send)]`` for a recorded decision, from its own features."""
    logits, _values = controller.forward(
        torch.as_tensor(np.asarray(record["features"], dtype=np.float32))
    )
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1).numpy()
    free = np.asarray([target in set(record["free"]) for target in record["targets"]], dtype=bool)
    actions = np.asarray([record["actions"][target] for target in record["targets"]], dtype=int)
    return probabilities, actions, free, logits


def _build(controller: Any, fixed: bool) -> Any:
    return (
        onpolicy.OnPolicyCollector(controller, forced=train_pilot.forced)
        if fixed
        else train_pilot.Collector(controller, epsilon=EPSILON)
    )


def collect_episodes(
    controller: Any,
    seeds: list[int],
    *,
    budget_bytes: int,
    directory: Path,
    label: str,
    fixed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run rollout episodes with one of the two collectors and keep their tapes.

    A fresh collector per episode, exactly as the trainer uses one: ``run_one`` rebuilds its tape
    from ``collector.records`` keyed by step, and steps repeat between episodes, so a shared
    collector would make every later episode re-report the earlier ones' decisions.
    """
    records: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for seed in seeds:
        collector = _build(controller, fixed)
        episode = train_pilot.run_one(
            seed, collector=collector, budget_bytes=budget_bytes, directory=directory, label=label
        )
        records.extend(episode["decisions"])
        timings.append(collector.timing())
    decisions = sum(int(item["decisions"]) for item in timings)
    aggregated = {
        "episodes": len(timings),
        "decisions": decisions,
        "candidates_seen": sum(int(item["candidates_seen"]) for item in timings),
        "forced_sends": sum(int(item["forced_sends"]) for item in timings),
        "decide_milliseconds_per_call": round(
            sum(float(item["decide_milliseconds_per_call"]) * int(item["decisions"]) for item in timings)
            / max(1, decisions),
            4,
        ),
    }
    free = [item.get("free_candidates") for item in timings]
    if all(value is not None for value in free):
        aggregated["free_candidates"] = sum(int(value) for value in free)
    return records, aggregated


def measure(controller: Any, records: list[dict[str, Any]], *, epsilon: float = 0.0) -> dict[str, Any]:
    """Compare drawn actions with the logged distribution, candidate by candidate."""
    free_rows = 0
    drawn_send = 0
    expected_send = 0.0
    joint_gaps: list[float] = []
    mixture_gaps: list[float] = []
    forced_rows = 0
    for record in records:
        probabilities, actions, free, logits = _recompute(controller, record)
        forced_rows += int((~free).sum())
        logged_probabilities = probabilities[free]
        logged_actions = actions[free]
        for index in range(logged_actions.size):
            free_rows += 1
            drawn_send += int(logged_actions[index] == onpolicy.SEND)
            expected_send += float(logged_probabilities[index, onpolicy.SEND])
        with torch.no_grad():
            recomputed = float(
                joint_log_prob(
                    logits,
                    torch.tensor(actions, dtype=torch.long),
                    torch.tensor(free, dtype=torch.float32),
                ).item()
            )
        joint_gaps.append(recomputed - float(record["log_prob"]))
        if epsilon > 0.0:
            # log mu(a|s) for the mixture the old sampler actually drew from, per free candidate.
            mixture = (1.0 - epsilon) * logged_probabilities + epsilon * 0.5
            true_behaviour = float(
                np.log(np.maximum(mixture[np.arange(logged_actions.size), logged_actions], 1e-12)).sum()
            )
            mixture_gaps.append(true_behaviour - float(record["log_prob"]))
    drawn_rate = drawn_send / max(1, free_rows)
    return {
        "tapes": len(records),
        "free_candidates": free_rows,
        "bound_taken_over_candidates": forced_rows,
        "drawn_send_rate": round(drawn_rate, 5),
        "logged_policy_send_rate": round(expected_send / max(1, free_rows), 5),
        "rate_gap": round(drawn_rate - expected_send / max(1, free_rows), 5),
        "standard_error_of_gap": round(
            float(np.sqrt((drawn_rate * (1 - drawn_rate)) / max(1, free_rows))), 5
        ),
        "recomputed_joint_log_prob_minus_logged": {
            "mean": round(float(np.mean(joint_gaps)), 8) if joint_gaps else None,
            "max_abs": round(float(np.max(np.abs(joint_gaps))), 6) if joint_gaps else None,
        },
        "log_mixture_minus_log_pi": {
            "mean": round(float(np.mean(mixture_gaps)), 5) if mixture_gaps else None,
            "mean_abs": round(float(np.mean(np.abs(mixture_gaps))), 5) if mixture_gaps else None,
            "min": round(float(np.min(mixture_gaps)), 5) if mixture_gaps else None,
            "max": round(float(np.max(mixture_gaps)), 5) if mixture_gaps else None,
        },
    }


def load_trained(path: Path, training_seed: int) -> Any:
    """A trained controller, checked to be the architecture this audit reads features for."""
    import evaluate_pilot

    controller = evaluate_pilot.load_controller(path, training_seed)
    if int(controller.trunk[0].in_features) != FEATURE_DIM:
        raise RuntimeError(f"{path} is not a {FEATURE_DIM}-dimensional pilot controller")
    return controller


def audit(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    seeds = list(common.TRAIN_SEEDS[: args.episodes])
    subjects: dict[str, Any] = {"untrained_initialisation": train_pilot.Controller(args.training_seed)}
    twin = train_pilot.Controller(args.training_seed)
    if common.weight_fingerprint(twin) != common.weight_fingerprint(subjects["untrained_initialisation"]):
        raise RuntimeError(
            "the initialisation is not reproducible, so the two collectors would not be measuring "
            "the same policy"
        )
    if args.controller_path:
        subjects["trained_checkpoint"] = load_trained(Path(args.controller_path), args.training_seed)
    payload: dict[str, Any] = {
        "protocol": {
            "epsilon": EPSILON,
            "training_seed": args.training_seed,
            "controller_fingerprints": {
                name: common.weight_fingerprint(controller) for name, controller in subjects.items()
            },
            "streams": seeds,
            "budget_bytes": args.budget_bytes,
            "information": "isolated",
            "trained_checkpoint": str(args.controller_path) if args.controller_path else None,
        },
        "note": (
            "the marginal send rate is deliberately not the headline: an epsilon mixture is a small "
            "perturbation when the policy is near uniform, which an untrained network is. The "
            "per-step log(mu) - log(pi) column is the quantity the PPO ratio actually absorbs."
        ),
        "subjects": {},
    }
    for name, trained in subjects.items():
        measured: dict[str, Any] = {}
        for collector_name, fixed in (("legacy_epsilon_greedy", False), ("fixed_on_policy", True)):
            started = time.time()
            records, timing = collect_episodes(
                trained, seeds, budget_bytes=args.budget_bytes,
                directory=directory / f"{name}_{collector_name}", label=f"audit_{name}_{collector_name}",
                fixed=fixed,
            )
            measured[collector_name] = {
                "timing": timing,
                "seconds": round(time.time() - started, 1),
                **measure(trained, records, epsilon=0.0 if fixed else EPSILON),
            }
            print(f"[{name}/{collector_name}] {json.dumps(measured[collector_name], ensure_ascii=False)}", flush=True)
        mixture = measured["legacy_epsilon_greedy"]["log_mixture_minus_log_pi"]
        payload["subjects"][name] = {
            **measured,
            "verdict": {
                "legacy_rate_gap_in_standard_errors": round(
                    measured["legacy_epsilon_greedy"]["rate_gap"]
                    / max(1e-9, measured["legacy_epsilon_greedy"]["standard_error_of_gap"]),
                    2,
                ),
                "fixed_rate_gap_in_standard_errors": round(
                    measured["fixed_on_policy"]["rate_gap"]
                    / max(1e-9, measured["fixed_on_policy"]["standard_error_of_gap"]),
                    2,
                ),
                "mean_log_mixture_minus_log_pi": mixture["mean"],
                "mean_abs_log_mixture_minus_log_pi": mixture["mean_abs"],
                "worst_abs_log_mixture_minus_log_pi": max(
                    abs(float(mixture["min"] or 0.0)), abs(float(mixture["max"] or 0.0))
                ),
            },
        }
        print(f"[{name}] verdict {json.dumps(payload['subjects'][name]['verdict'], ensure_ascii=False)}", flush=True)
    common.write_json(common.json_path("COMM_PPO_PROBABILITY_AUDIT.json"), payload)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--controller-path", default=None, help="a trained checkpoint to measure as well")
    parser.add_argument("--budget-bytes", type=int, default=common.BUDGETS_BYTES["75pct"])
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    return audit(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
