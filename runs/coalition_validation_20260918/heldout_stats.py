"""Held-out statistics and acceptance (brief section five).

Every choice here was fixed in HELDOUT_PROTOCOL.md before any held-out number existed:
unit = seed; primary = paired per-seed on-time completion rate over detected targets;
zero-detection seeds excluded by rule and reported by count; 95 % CI = seed-level
percentile bootstrap, B = 10 000, RNG seed 20260918 (a t-interval is reported for
reference only); correctness acceptance separate from the performance claim.

A completed target's `failure_reason` can carry an unrelated late string (for example
"Cannot contribute"), so on-time is decided by progress plus the *specific* expiry
reason, never by "reason is not None".

    D:/project_env/python.exe runs/coalition_validation_20260918/heldout_stats.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "heldout"
BOOTSTRAP_SEED = 20260918
BOOTSTRAP_RESAMPLES = 10_000
CEILING = 0.99
ARMS = ("baseline", "candidate")
SECONDARY_KEYS = (
    "redundant_working_ticks", "working_ticks", "coverage_resets", "lost_work",
    "rejected_assignments", "executed_override_unjustified", "recommands",
    "releases_planned", "replan_attempts", "accepted_reassignments",
    "replan_declined", "same_task_noop",
)


def load(arm: str, seeds: list[int]) -> dict[int, dict]:
    return {
        seed: json.loads((OUT / f"{arm}_s{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    }


def is_on_time(fact: dict) -> bool:
    return bool(fact["progress"] >= 1.0 and fact["reason"] != "deadline_expired")


def bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, differences.size, size=(BOOTSTRAP_RESAMPLES, differences.size))
    means = differences[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def t_interval(differences: np.ndarray) -> tuple[float, float]:
    mean = float(differences.mean())
    if differences.size < 2:
        return mean, mean
    se = float(differences.std(ddof=1) / math.sqrt(differences.size))
    return mean - 1.96 * se, mean + 1.96 * se


def rejections(arm: str, seeds: list[int]) -> dict[int, list[dict]]:
    found: dict[int, list[dict]] = {}
    for seed in seeds:
        rows = [
            {"target": row.get("target_id"), "reason": row.get("failure_reason"), "at": row.get("timestamp")}
            for line in (OUT / f"{arm}_rp_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line)).get("event_type") == "assignment_rejected"
        ]
        if rows:
            found[seed] = rows
    return found


def completion_latencies(arm: str, seeds: list[int]) -> list[float]:
    """Detection to the last state change of a target finished on time."""
    values: list[float] = []
    for seed in seeds:
        for line in (OUT / f"{arm}_rp_s{seed}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event_type") != "final":
                continue
            detail = row.get("detail") or {}
            if (row.get("progress") or 0.0) < 1.0 or row.get("failure_reason") == "deadline_expired":
                continue
            if "updated_at" in detail and "discovered_at" in detail:
                values.append(float(detail["updated_at"]) - float(detail["discovered_at"]))
    return values


def main(argv: list[str]) -> int:
    seeds = [int(seed) for seed in json.loads(
        (HERE / "HELDOUT_SEEDS.json").read_text(encoding="utf-8")
    )["seeds"]]
    data = {arm: load(arm, seeds) for arm in ARMS}
    missing = [seed for seed in seeds for arm in ARMS if seed not in data[arm]]
    if missing:
        print(f"MISSING RUNS: {len(missing)} {sorted(set(missing))[:10]}")
        return 1

    differences: list[float] = []
    zero_detection: list[int] = []
    denominator_mismatch: list[int] = []
    regressions: list[dict] = []
    rescued: list[dict] = []
    closure = {arm: {"detections": 0, "on_time": 0, "expired": 0, "active": 0} for arm in ARMS}
    secondary = {
        arm: {**{key: 0 for key in SECONDARY_KEYS}, "max_idle_streak": 0.0} for arm in ARMS
    }
    expired_by_seed = {arm: {} for arm in ARMS}

    for seed in seeds:
        rates: dict[str, float] = {}
        for arm in ARMS:
            metrics = data[arm][seed]
            targets = metrics["targets"]
            detected = len(targets)
            on_time = sum(1 for fact in targets.values() if is_on_time(fact))
            expired = sum(1 for fact in targets.values() if fact["reason"] == "deadline_expired")
            closure[arm]["detections"] += detected
            closure[arm]["on_time"] += on_time
            closure[arm]["expired"] += expired
            closure[arm]["active"] += detected - on_time - expired
            expired_by_seed[arm][seed] = expired
            for key in SECONDARY_KEYS:
                secondary[arm][key] += metrics[key]
            secondary[arm]["max_idle_streak"] = max(
                secondary[arm]["max_idle_streak"], metrics["longest_idle_streak"]
            )
            rates[arm] = on_time / detected if detected else 0.0
        if len(data["baseline"][seed]["targets"]) != len(data["candidate"][seed]["targets"]):
            denominator_mismatch.append(seed)
        if not data["baseline"][seed]["targets"]:
            zero_detection.append(seed)
            continue
        differences.append(rates["candidate"] - rates["baseline"])
        for target, fact in data["candidate"][seed]["targets"].items():
            base_fact = data["baseline"][seed]["targets"].get(target, {})
            if base_fact and is_on_time(base_fact) and not is_on_time(fact):
                regressions.append({"seed": seed, "target": target, "candidate": fact, "baseline": base_fact})
            if base_fact and is_on_time(fact) and not is_on_time(base_fact):
                rescued.append({"seed": seed, "target": target, "baseline": base_fact})

    array = np.asarray(differences, dtype=float)
    ci_low, ci_high = bootstrap_ci(array) if array.size else (0.0, 0.0)
    t_low, t_high = t_interval(array) if array.size else (0.0, 0.0)
    pooled = {
        arm: closure[arm]["on_time"] / max(1, closure[arm]["detections"]) for arm in ARMS
    }
    expired_differences = np.asarray(
        [
            expired_by_seed["candidate"][seed] - expired_by_seed["baseline"][seed]
            for seed in seeds
            if data["baseline"][seed]["targets"]
        ],
        dtype=float,
    )
    expired_low, expired_high = bootstrap_ci(expired_differences)
    rejection_rows = {arm: rejections(arm, seeds) for arm in ARMS}
    candidate_only_rejections = {
        str(seed): rows
        for seed, rows in rejection_rows["candidate"].items()
        if seed not in rejection_rows["baseline"]
    }
    latency = {arm: completion_latencies(arm, seeds) for arm in ARMS}

    if ci_low > 0.0:
        claim = "significant improvement (bootstrap CI lower bound above zero)"
    elif ci_high < 0.0:
        claim = "significant regression (bootstrap CI upper bound below zero)"
    else:
        claim = "no improvement demonstrated: the 95% interval crosses zero"

    verdict = {
        "seeds": len(seeds),
        "seeds_in_primary": int(array.size),
        "zero_detection_seeds_excluded_by_rule": len(zero_detection),
        "zero_detection_seed_list": [int(seed) for seed in zero_detection[:20]],
        "identical_denominators": not denominator_mismatch,
        "denominator_mismatch_seeds": [int(seed) for seed in denominator_mismatch[:20]],
        "primary": {
            "metric": "paired per-seed on-time completion rate (candidate - baseline)",
            "mean_difference": round(float(array.mean()), 6) if array.size else 0.0,
            "bootstrap_ci95": [round(ci_low, 6), round(ci_high, 6)],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_rng_seed": BOOTSTRAP_SEED,
            "t_interval_reference": [round(t_low, 6), round(t_high, 6)],
            "significant_improvement": bool(ci_low > 0.0),
            "claim": claim,
        },
        "secondary_expired_count": {
            "baseline": int(closure["baseline"]["expired"]),
            "candidate": int(closure["candidate"]["expired"]),
            "paired_mean_difference": round(float(expired_differences.mean()), 6) if expired_differences.size else 0.0,
            "bootstrap_ci95": [round(expired_low, 6), round(expired_high, 6)],
        },
        "completion_latency_detected_to_finish": {
            arm: {
                "n": len(values),
                "mean": round(float(np.mean(values)), 3) if values else None,
                "median": round(float(np.median(values)), 3) if values else None,
            }
            for arm, values in latency.items()
        },
        "ceiling_effect": bool(pooled["baseline"] >= CEILING and pooled["candidate"] >= CEILING),
        "closure": {
            **{arm: {key: int(value) for key, value in facts.items()} for arm, facts in closure.items()},
            "pooled_on_time_rate": {arm: round(pooled[arm], 6) for arm in ARMS},
            "runs_per_arm": {arm: len(seeds) for arm in ARMS},
        },
        "secondary": {
            arm: {
                key: (round(value, 3) if isinstance(value, float) else int(value))
                for key, value in facts.items()
            }
            for arm, facts in secondary.items()
        },
        "rescued_count": len(rescued),
        "rescued_targets_sample": rescued[:40],
        "candidate_new_failure_count": len(regressions),
        "candidate_new_failures": regressions[:40],
        "rejections": {
            "baseline_seeds": sorted(str(seed) for seed in rejection_rows["baseline"]),
            "candidate_seeds": sorted(str(seed) for seed in rejection_rows["candidate"]),
            "candidate_only": candidate_only_rejections,
        },
        "correctness": {
            "no_joint_release_coverage_loss": len(regressions) == 0,
            "no_candidate_only_rejections": len(candidate_only_rejections) == 0,
            "no_duplicate_own_task_commands": secondary["candidate"]["recommands"] == 0,
            "no_unjustified_overrides": secondary["candidate"]["executed_override_unjustified"] == 0,
            "no_excess_deadlock_streak": secondary["candidate"]["max_idle_streak"] < 20.0,
        },
    }
    (HERE / "HELDOUT_RESULTS.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {key: verdict[key] for key in (
            "seeds", "seeds_in_primary", "zero_detection_seeds_excluded_by_rule",
            "identical_denominators", "primary", "secondary_expired_count",
            "completion_latency_detected_to_finish", "ceiling_effect", "closure",
            "rescued_count", "candidate_new_failure_count", "correctness",
        )},
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
