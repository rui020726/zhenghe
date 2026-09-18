"""The statistics for the corrected-environment block, fixed before the block is read.

Primary metric: the per-seed paired difference in on-time rate (arm D minus arm C), where the
rate for one episode is its on-time targets over the targets it detected. Interval: a
seed-level paired percentile bootstrap, B = 10 000, RNG seeded 20260918, 95 % interval. Only
the primary metric carries an inferential claim; every other number is reported descriptively
with a nominal interval and labelled as such.

Declared in advance, so no result can decide them afterwards:

* **zero-detection seeds** -- an episode pair where neither arm detected a single target
  carries no information about the primary rate; it is excluded from the paired set and
  counted in the open, and the same rule is applied before any arm is looked at;
* **technical failures** -- a pair where either arm's episode did not finish, did not match
  the frozen commit, replayed a different stream digest than its partner, or left an
  incomplete lifecycle log is excluded and listed by seed, never silently dropped;
* **ceiling** -- if both arms complete every detected target on a seed, that seed contributes
  a difference of zero, which is a real measurement and not a missing one; the share of such
  seeds is reported so a narrow interval is read against the amount of signal available;
* **correctness first** -- any real false credit, dual on-site supply, release that lowered
  actual coverage, duplicate-command reset or illegal execution fails the acceptance of the
  block regardless of the performance interval. A higher completion rate never buys those off.

    D:/project_env/python.exe runs/corrected_env_final_20260918/stats_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arms  # noqa: E402
import corrected_block as block  # noqa: E402

B = 10_000
RNG_SEED = 20260918
PRIMARY = "on_time_rate"
SECONDARY = (
    "on_time",
    "detections",
    "expired",
    "deadline_expired_targets",
    "unresolved",
    "completed_env_tasks",
    "mean_completion_time",
    "travel_dist_total",
    "lost_work",
    "coverage_resets",
    "idle_robot_ticks",
    "longest_idle_streak",
    "replan_decisions",
    "releases_published",
    "releases_executed",
    "release_refusal_recovered",
    "release_refusal_unrecovered",
)
CORRECTNESS = block.CORRECTNESS_FIELDS
GATED = block.GATED_FIELDS
LOWER_IS_BETTER = {
    "expired",
    "unresolved",
    "mean_completion_time",
    "travel_dist_total",
    "lost_work",
    "coverage_resets",
    "idle_robot_ticks",
    "longest_idle_streak",
    "release_refusal_unrecovered",
}


def load() -> dict[str, dict[int, dict]]:
    by_arm: dict[str, dict[int, dict]] = {}
    for label in arms.ARMS:
        rows: dict[int, dict] = {}
        for path in sorted((block.RUN_DIR / label).glob(f"episode_{label}_s*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows[int(payload["seed"])] = payload
        by_arm[label] = rows
    return by_arm


def failures(by_arm: dict[str, dict[int, dict]], scheduled: list[int]) -> tuple[list[int], list[str]]:
    """Seeds excluded for technical reasons, with the reason, before any metric is read."""
    reasons: list[str] = []
    bad: set[int] = set()
    for label, rows in by_arm.items():
        for seed, row in rows.items():
            if int(row.get("run_steps", 0)) != arms.STEPS or float(row.get("run_time", 0.0)) < arms.STEPS - 0.5:
                reasons.append(
                    f"{label} s{seed}: episode did not reach the horizon "
                    f"(steps={row.get('run_steps')}, time={row.get('run_time')})"
                )
                bad.add(seed)
            # What has to match is the production tree, not the label: a documentation commit
            # made while the block ran is a different string over the same executing code.
            if not arms.same_production(str(row.get("code_commit"))):
                reasons.append(
                    f"{label} s{seed}: commit {row.get('code_commit')} does not match the freeze"
                )
                bad.add(seed)
            if row.get("detections", 0) == 0:
                continue
        for seed, row in rows.items():
            partner = by_arm["D" if label == "C" else "C"].get(seed)
            if partner is not None and partner["stream_sha256"] != row["stream_sha256"]:
                reasons.append(f"s{seed}: the two arms replayed different streams")
                bad.add(seed)
    for label, rows in by_arm.items():
        for seed in scheduled:
            if seed not in rows:
                reasons.append(f"{label} s{seed}: no episode artefact")
                bad.add(seed)
    empty = [
        seed
        for seed in set(by_arm["C"]) & set(by_arm["D"])
        if by_arm["C"][seed]["detections"] == 0 and by_arm["D"][seed]["detections"] == 0
    ]
    for seed in empty:
        reasons.append(f"s{seed}: neither arm detected a target (zero-detection rule)")
    return sorted(set(bad) | set(empty)), reasons


def paired_bootstrap(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    indices = rng.integers(0, differences.size, size=(B, differences.size))
    means = differences[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    smoke = "--smoke" in sys.argv
    if smoke:
        block.RUN_DIR = HERE / "block_smoke"
    by_arm = load()
    scheduled = [] if smoke else block.seeds()
    excluded, reasons = failures(by_arm, scheduled)
    paired = sorted(set(by_arm["C"]) & set(by_arm["D"]) - set(excluded))
    differences = np.array(
        [by_arm["D"][seed][PRIMARY] - by_arm["C"][seed][PRIMARY] for seed in paired],
        dtype=float,
    )
    lower, upper = (
        paired_bootstrap(differences) if differences.size else (0.0, 0.0)
    )
    discordant = int(np.count_nonzero(differences))
    worse = [seed for seed, value in zip(paired, differences.tolist()) if value < -1e-12]
    better = [seed for seed, value in zip(paired, differences.tolist()) if value > 1e-12]
    both_ceiling = sum(
        1
        for seed in paired
        if by_arm["C"][seed]["on_time"] == by_arm["C"][seed]["detections"]
        and by_arm["D"][seed]["on_time"] == by_arm["D"][seed]["detections"]
    )
    correctness = {
        label: {field: int(sum(row[field] for row in by_arm[label].values())) for field in CORRECTNESS}
        for label in arms.ARMS
    }
    correctness_pass = all(
        row[field] == 0 for row in correctness.values() for field in GATED
    )
    secondary = {}
    for field in SECONDARY:
        values = {
            label: np.array([by_arm[label][seed][field] for seed in paired], dtype=float)
            for label in arms.ARMS
        }
        deltas = values["D"] - values["C"]
        observed = deltas[np.isfinite(deltas)]
        secondary[field] = {
            "C": round(float(np.nanmean(values["C"])), 4),
            "D": round(float(np.nanmean(values["D"])), 4),
            "mean_paired_difference": round(float(observed.mean()), 4) if observed.size else None,
            # Direction is named, not read off a sign: a smaller completion time is an
            # improvement while a larger idle count is not, and calling both "better" would
            # be the kind of label that quietly inverts a result.
            "seeds_D_higher": int(np.nansum(deltas > 1e-12)),
            "seeds_D_lower": int(np.nansum(deltas < -1e-12)),
            "lower_is_better": field in LOWER_IS_BETTER,
        }
    payload = {
        "protocol": {
            "primary_metric": f"per-seed paired difference in {PRIMARY} (D minus C)",
            "bootstrap_resamples": B,
            "bootstrap_rng_seed": RNG_SEED,
            "interval": "percentile 95 %, one primary metric, no multiplicity adjustment",
            "zero_detection_rule": "excluded from the paired set when neither arm detected a target",
            "technical_failure_rule": "excluded and listed: unfinished, commit or stream mismatch, or missing artefact",
            "ceiling_rule": "both arms completing every detected target is a difference of zero, kept",
            "correctness_gate": "any non-zero correctness count fails the block regardless of performance",
        },
        "scheduled_seeds": len(block.seeds()),
        "episodes": {label: len(rows) for label, rows in by_arm.items()},
        "paired_seeds": len(paired),
        "excluded_seeds": len(excluded),
        "exclusion_reasons": reasons[:60],
        "totals": {
            label: {
                "detections": int(sum(row["detections"] for row in by_arm[label].values())),
                "on_time": int(sum(row["on_time"] for row in by_arm[label].values())),
                "expired_targets": int(
                    sum(row["deadline_expired_targets"] for row in by_arm[label].values())
                ),
                "unresolved_targets": int(
                    sum(row["unresolved"] for row in by_arm[label].values())
                ),
            }
            for label in arms.ARMS
        },
        "primary": {
            "mean_difference": round(float(differences.mean()), 6) if differences.size else None,
            "ci95": [round(lower, 6), round(upper, 6)],
            "seeds_d_better": len(better),
            "seeds_d_worse": len(worse),
            "seeds_discordant": discordant,
            "seeds_both_at_ceiling": both_ceiling,
            "share_of_paired_seeds": round(
                discordant / max(1, len(paired)), 4
            ),
        },
        "correctness": correctness,
        "correctness_gate_pass": correctness_pass,
        "candidate_new_failures": [
            {
                "seed": seed,
                "C": by_arm["C"][seed][PRIMARY],
                "D": by_arm["D"][seed][PRIMARY],
                "D_expired": by_arm["D"][seed]["expired"],
                "D_unresolved": by_arm["D"][seed]["unresolved"],
                "D_releases_executed": by_arm["D"][seed]["releases_executed"],
            }
            for seed in worse
        ],
        "secondary_descriptive": secondary,
    }
    # A smoke run must not write the file the block's verdict will be read from.
    destination = HERE / (
        "SMOKE_STATS.json" if "--smoke" in sys.argv else "CORRECTED_ENV_HELDOUT_STATS.json"
    )
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    import os

    os.replace(part, destination)
    print(json.dumps({key: payload[key] for key in (
        "scheduled_seeds", "episodes", "paired_seeds", "excluded_seeds", "totals", "primary",
        "correctness", "correctness_gate_pass")}, indent=1))
    print(f"candidate new failures: {len(worse)}; discordant seeds: {discordant}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
