"""The engineering confirmation: five locked arms, three settings, 200 fresh streams.

The learning rounds asked whether a network could beat a rule. This round asks a narrower,
engineering question on data nothing has ever seen: does the cost--effect relation round 6
measured for ``retry_until_acked_2`` on 48 training streams reproduce? Nothing is selected
here -- the arms and their parameters were frozen on the training block, and the settings,
axes and statistics were frozen in ``COMM_RULE_CONFIRMATION_PROTOCOL.md`` before any stream
was recorded.

Two accounting rules carry the argument, so they are enforced rather than described:

* every cost is ``total bytes = task uplink attempts + ACK reverse attempts``, with the two
  components reported beside it -- an arm that saves uplink and spends the reverse channel
  has not saved anything;
* the forward loss process is a per-episode seeded stream, so packets across arms are a
  same-distribution comparison, not the same events. The reverse channel is counter-based,
  so ACK-on arms meet it as common random numbers.

    python confirm_matrix.py --mode probe --threads 1
    python confirm_matrix.py --mode matrix --shards 6 --shard 0 --threads 1
    python confirm_matrix.py --mode report
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

import confirm_common as common
import confirm_env

#: The comparisons the protocol freezes, as (arm, baseline, what it measures).
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ack_retry_2", "noack_send_once", "the protocol's cost-effect difference as a whole"),
    ("ack_retry_2", "ack_send_once", "what the retry rule adds on top of the ACK information"),
    ("ack_periodic_20", "ack_retry_2", "frontier stability: periodic against the locked rule"),
    ("ack_backoff_16", "ack_retry_2", "frontier stability: backoff against the locked rule"),
)

#: The axes every comparison is reported on. ``total_bytes`` is the only cost axis a claim may
#: rest on; the components are there so a shift between channels cannot hide.
AXES: tuple[str, ...] = ("on_time", "total_bytes", "uplink_bytes", "ack_bytes")

#: The arms the lossless negative control checks, and the non-inferiority margin.
NEGATIVE_CONTROL_ARMS = ("ack_retry_2", "ack_periodic_20", "ack_backoff_16")
NON_INFERIORITY_MARGIN = -0.05
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260919

#: Traffic counters copied out so the report shows the traffic, not only the money.
TRAFFIC_COUNTERS = (
    "emissions",
    "first_emissions",
    "refresh_emissions",
    "redundant_emissions",
    "generated_items",
    "coalesced_changes",
    "transmission_attempts",
    "packets_lost",
    "packets_delivered",
    "deliveries_applied",
    "delivery_duplicate",
    "delivery_superseded",
    "budget_refused_sends",
    "capacity_refused_sends",
    "emissions_suppressed_terminal",
    "ack_emitted",
    "ack_attempts",
    "ack_attempt_bytes",
    "ack_delivered",
    "ack_lost",
    "ack_frames_advanced",
    "ack_frames_reaffirmed",
)


def parse_params(text: str | None) -> dict[str, str]:
    return {
        key: value
        for key, value in (
            token.split("=", 1) for token in str(text or "").split(",") if token
        )
    }


def run_arm(
    arm: str,
    seed: int,
    setting: str,
    *,
    directory: Path | None = None,
    stream: Path | None = None,
) -> dict[str, Any]:
    """One locked arm on one stream in one setting, with the whole ledger."""
    spec = common.ARMS[arm]
    channel = common.SETTINGS[setting]
    row = confirm_env.run_episode(
        seed,
        directory=directory or common.evaluation_root(setting),
        label=f"{setting}_{arm}",
        policy=spec["policy"],
        policy_params=parse_params(spec.get("params")),
        budget_bytes=common.BUDGET_BYTES,
        information=channel["information"],
        delivery_probability=channel["delivery_probability"],
        delay_ticks=0,
        max_wait_steps=None,
        urgent_wait_steps=None,
        timing_mode=common.POLICY_MODE,
        ack=bool(spec["ack"]),
        ack_delay_ticks=common.ACK_DELAY_TICKS,
        ack_delivery_probability=channel["delivery_probability"],
        stream=stream,
    )
    ledger = row["ledger"]
    uplink_bytes = int(ledger["airtime_bytes"])
    ack_bytes = int(ledger["ack_attempt_bytes"])
    targets = int(row["task"]["targets"])
    on_time = int(row["task"]["on_time"])
    return {
        "arm": arm,
        "setting": setting,
        "seed": int(seed),
        "policy": spec["policy"],
        "parameters": str(spec.get("params") or ""),
        "information": channel["information"],
        "delivery_probability": channel["delivery_probability"],
        "ack_enabled": bool(spec["ack"]),
        "ack_delivery_probability": (
            channel["delivery_probability"] if spec["ack"] else None
        ),
        "ack_delay_ticks": common.ACK_DELAY_TICKS if spec["ack"] else None,
        "timing_mode": common.POLICY_MODE,
        "budget_bytes": common.BUDGET_BYTES,
        "on_time": on_time,
        "targets": targets,
        "expired": int(row["task"]["expired"]),
        "unresolved": int(row["task"]["unresolved"]),
        "completion_rate_conditional": round(on_time / targets, 5) if targets else None,
        "completion_rate_all_goals": round(
            on_time / float(common.RECORD_CONFIGURATION["num_goals"]), 5
        ),
        "uplink_bytes": uplink_bytes,
        "ack_bytes": ack_bytes,
        "total_bytes": uplink_bytes + ack_bytes,
        "spent_bytes": int(row["spent_bytes"]),
        "budget_used_fraction": row["budget_used_fraction"],
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "latencies": row["latencies"],
        "targets_outcome": row["targets_outcome"],
        **{key: int(ledger.get(key, 0)) for key in TRAFFIC_COUNTERS},
        "ack_ledger": row.get("ack_ledger"),
        "arm_label": row["arm_label"],
        "config_hash": row["config_hash"],
        "stream_sha256": row["stream_sha256"],
        "seconds": float(row["seconds"]),
        "artefacts": row["artefacts"],
    }


# ------------------------------------------------------------------ per-file identity
def validate_row(
    row: dict[str, Any],
    registration: dict[int, str],
    *,
    require_registered_seed: bool = True,
) -> list[str]:
    """The aggregation preflight, run on every row before it can be pooled.

    The arm's switchable identity (policy, ACK switch, information path, delivery
    probabilities, credit), the code identity (configuration hash), the data identity (the
    replayed stream's hash) and the metric schema must all say what the row claims. Anything
    else is a mixed sample and fails closed.
    """
    spec = common.ARMS[row["arm"]]
    channel = common.SETTINGS[row["setting"]]
    problems: list[str] = []
    if row["policy"] != spec["policy"] or row["parameters"] != str(spec.get("params") or ""):
        problems.append("policy or parameters differ from the locked arm table")
    if bool(row["ack_enabled"]) != bool(spec["ack"]):
        problems.append("the ACK switch differs from the locked arm table")
    if row["information"] != channel["information"]:
        problems.append("the information path differs from the setting")
    if float(row["delivery_probability"]) != float(channel["delivery_probability"]):
        problems.append("the uplink delivery probability differs from the setting")
    if row["ack_enabled"]:
        if row["ack_delivery_probability"] is None:
            problems.append("an ACK-on run does not state its reverse-channel probability")
        elif float(row["ack_delivery_probability"]) != float(
            channel["delivery_probability"]
        ):
            problems.append("the reverse-channel probability differs from the setting")
    if int(row["budget_bytes"]) != int(common.BUDGET_BYTES):
        problems.append("the byte credit is not the locked 29,300 B")
    if int(row["spent_bytes"]) > int(common.BUDGET_BYTES):
        problems.append("the budget was breached")
    if row["timing_mode"] != common.POLICY_MODE:
        problems.append("the run is not on policy_owned_timing")
    if not str(row["config_hash"] or ""):
        problems.append("the run carries no configuration hash")
    if require_registered_seed:
        expected_stream = registration.get(int(row["seed"]))
        if expected_stream is None:
            problems.append("the seed is not part of the registered block")
        elif str(row["stream_sha256"]) != expected_stream:
            problems.append("the replayed stream is not the registered one")
    if not isinstance(row["targets_outcome"], dict) or not row["targets_outcome"]:
        problems.append("the metric schema produced no per-target outcome")
    if (
        int(row["on_time"]) + int(row["expired"]) + int(row["unresolved"])
        != int(row["targets"])
    ):
        problems.append("task outcomes do not close against the discovered targets")
    if not row["ack_enabled"] and int(row["ack_bytes"]) != 0:
        problems.append("a no-ACK run charged reverse bytes")
    ledger = row.get("ack_ledger")
    if row["ack_enabled"] and isinstance(ledger, dict):
        if int(ledger.get("attempt_bytes", -1)) != int(row["ack_bytes"]):
            problems.append("the reverse ledger disagrees with the transcript ACK bytes")
    return problems


# ------------------------------------------------------------------ the run
def job_list(seeds: tuple[int, ...]) -> list[tuple[str, str, int]]:
    """Every cell of the design: one (setting, arm, seed) per episode."""
    return [
        (setting, arm, seed)
        for setting in common.SETTING_LABELS
        for arm in common.ARM_LABELS
        for seed in seeds
    ]


def matrix(args: argparse.Namespace) -> int:
    registration = load_registration()
    scheduled = common.shard_of(job_list(common.BLOCK_SEEDS), args.shards, args.shard)
    rows: list[dict[str, Any]] = []
    problems: dict[str, list[str]] = {}
    started = time.time()
    for index, (setting, arm, seed) in enumerate(scheduled, start=1):
        row = run_arm(arm, seed, setting)
        issues = validate_row(row, registration)
        if issues:
            problems[f"{setting}/{arm}/{seed}"] = issues
            continue
        rows.append(row)
        if index % 50 == 0 or index == len(scheduled):
            print(
                f"[matrix shard {args.shard}] {index}/{len(scheduled)} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    payload = {
        "budget_bytes": common.BUDGET_BYTES,
        "settings": list(common.SETTING_LABELS),
        "arms": list(common.ARM_LABELS),
        "seeds": [int(seed) for seed in common.BLOCK_SEEDS],
        "shard": args.shard,
        "shards": args.shards,
        "rejected": problems,
        "rows": rows,
    }
    common.write_json(common.HERE / f"matrix_shard{args.shard}.json", payload)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "rejected": len(problems),
                "seconds": round(time.time() - started, 1),
                "output": f"matrix_shard{args.shard}.json",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if problems else 0


def probe(args: argparse.Namespace) -> int:
    """Measure throughput, disk and artefact completeness before choosing a shard count.

    Run on five seeds that are *not* in the new block -- the probe is not a result, and it
    must not spend any of the confirmation data. The estimate it replaces is round 6's, which
    ran a different mix of settings.
    """
    seeds = tuple(int(seed) for seed in (args.seeds or common.pilot.TRAIN_SEEDS[:5]))
    for seed in seeds:
        if common.BLOCK_START <= seed < common.BLOCK_STOP:
            raise SystemExit(f"probe seed {seed} is inside the new block; the probe may not use it")
    directory = common.EVALUATION_OUT / "probe"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for setting in common.SETTING_LABELS:
        for arm in common.ARM_LABELS:
            for seed in seeds:
                row = run_arm(
                    arm, seed, setting, directory=directory,
                    stream=common.pilot.stream_path(seed),
                )
                issues = validate_row(row, {}, require_registered_seed=False)
                rows.append({**row, "validation_problems": issues})
                print(
                    f"[probe {setting}/{arm} s{seed}] onT={row['on_time']} "
                    f"totalB={row['total_bytes']} (up{row['uplink_bytes']}+ack{row['ack_bytes']}) "
                    f"{row['seconds']}s{' ISSUES ' + '; '.join(issues) if issues else ''}",
                    flush=True,
                )
    serial = time.time() - started
    bytes_on_disk = sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    )
    seconds = [float(row["seconds"]) for row in rows]
    payload = {
        "probe_seeds": list(seeds),
        "episodes": len(rows),
        "settings": list(common.SETTING_LABELS),
        "arms": list(common.ARM_LABELS),
        "seconds_per_episode_mean": round(float(np.mean(seconds)), 3),
        "seconds_per_episode_max": round(max(seconds), 3),
        "serial_seconds": round(serial, 1),
        "projected_serial_hours_3000_episodes": round(serial / len(rows) * 3000 / 3600, 2),
        "bytes_per_episode": round(bytes_on_disk / len(rows)),
        "projected_disk_gb_3000_episodes": round(bytes_on_disk / len(rows) * 3000 / 1e9, 2),
        "rows_with_issues": [
            f"{row['setting']}/{row['arm']}/{row['seed']}"
            for row in rows
            if row["validation_problems"]
        ],
        "per_setting_seconds": {
            setting: round(
                float(np.mean([r["seconds"] for r in rows if r["setting"] == setting])), 3
            )
            for setting in common.SETTING_LABELS
        },
        "per_arm_total_bytes": {
            arm: round(float(np.mean([r["total_bytes"] for r in rows if r["arm"] == arm])), 1)
            for arm in common.ARM_LABELS
        },
    }
    common.write_json(common.json_path("COMM_RULE_THROUGHPUT.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 1 if payload["rows_with_issues"] else 0


def load_registration() -> dict[int, str]:
    return {
        int(row["seed"]): str(row["sha256"])
        for row in common.read_json("COMM_RULE_STREAMS.json")["streams"]
    }


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(common.HERE.glob("matrix_shard*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    return rows


# ------------------------------------------------------------------ statistics
def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def _difference(
    rows: list[dict[str, Any]], arm: str, baseline: str, key: str
) -> dict[str, Any]:
    """Round 2's seed-level paired bootstrap, reused so the CI is the same quantity."""
    result = common.paired_bootstrap(rows, baseline=baseline, key=key)
    entry = result.get("arms", {}).get(arm, {})
    ci = entry.get("ci95") or [None, None]
    return {
        "arm": arm,
        "baseline": baseline,
        "key": key,
        "pairs": int(entry.get("pairs", 0)),
        "mean_difference": entry.get("mean_difference"),
        "ci95_lower": ci[0],
        "ci95_upper": ci[1],
        "seeds_better": entry.get("seeds_better"),
        "seeds_worse": entry.get("seeds_worse"),
        "seeds_equal": entry.get("seeds_equal"),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-arm means, grouped by the row's own setting.

    The grouping is taken from each row rather than from the caller's pre-filtering, so a
    caller that forgets to scope its rows cannot silently pool two settings -- the protocol
    forbids pooling, and the aggregate is where that would happen.
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for arm, items in by_arm.items():
        settings = sorted({str(item["setting"]) for item in items})
        if len(settings) != 1:
            raise ValueError(
                f"arm {arm!r} aggregates rows from {settings}; the three settings are never "
                "pooled"
            )
        summary[arm] = {
            "episodes": len(items),
            "on_time_per_episode": _mean([float(r["on_time"]) for r in items]),
            "expired_per_episode": _mean([float(r["expired"]) for r in items]),
            "unresolved_per_episode": _mean([float(r["unresolved"]) for r in items]),
            "discovered_per_episode": _mean([float(r["targets"]) for r in items]),
            "completion_rate_conditional": _mean(
                [
                    float(r["completion_rate_conditional"])
                    for r in items
                    if r["completion_rate_conditional"] is not None
                ]
            ),
            "completion_rate_all_goals": _mean(
                [float(r["completion_rate_all_goals"]) for r in items]
            ),
            "uplink_bytes": _mean([float(r["uplink_bytes"]) for r in items]),
            "ack_bytes": _mean([float(r["ack_bytes"]) for r in items]),
            "total_bytes": _mean([float(r["total_bytes"]) for r in items]),
            "ack_share_of_total": _mean(
                [
                    float(r["ack_bytes"]) / float(r["total_bytes"])
                    for r in items
                    if int(r["total_bytes"]) > 0
                ]
            ),
            "budget_used_fraction": _mean(
                [float(r["budget_used_fraction"] or 0.0) for r in items]
            ),
            "belief_age_steps": _mean([float(r["belief_age_steps"]) for r in items]),
            **{
                key: _mean(
                    [
                        float(r["latencies"][key])
                        for r in items
                        if r["latencies"].get(key) is not None
                    ]
                )
                for key in (
                    "detection_to_assignment",
                    "detection_to_work",
                    "dispatch_to_work",
                    "detection_to_completion",
                )
            },
            **{key: _mean([float(r[key]) for r in items]) for key in TRAFFIC_COUNTERS},
        }
    return summary


def delta_vectors(
    rows: list[dict[str, Any]], arm: str, baseline: str
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Per-seed differences against the baseline, on every axis, in one shared seed order."""
    left = {int(row["seed"]): row for row in rows if row["arm"] == arm}
    right = {int(row["seed"]): row for row in rows if row["arm"] == baseline}
    seeds = sorted(set(left) & set(right))
    deltas = {
        key: np.array(
            [float(left[seed][key]) - float(right[seed][key]) for seed in seeds], dtype=float
        )
        for key in AXES
    }
    return seeds, deltas


def ratio_bootstrap(deltas: dict[str, np.ndarray]) -> dict[str, Any]:
    """Bytes per additional on-time target, with the protocol's undefined rule.

    The point estimate is mean numerator over mean denominator, both reported raw. The
    interval resamples the paired seed vectors with the frozen bootstrap RNG; a replicate whose
    denominator is not positive is undefined, and an undefined estimate stays undefined rather
    than being trimmed into a stable-looking number.
    """
    completion = deltas["on_time"]
    out: dict[str, Any] = {"denominator_mean_on_time": round(float(completion.mean()), 5)}
    for key in ("total_bytes", "uplink_bytes", "ack_bytes"):
        out[f"{key}_numerator_mean"] = round(float(deltas[key].mean()), 2)
    if float(completion.mean()) <= 0.0:
        out["defined"] = False
        out["note"] = (
            "the arm does not complete more on-time targets than its baseline on average, so "
            "'bytes per additional on-time target' has no positive denominator: undefined, and "
            "the byte difference stands on its own"
        )
        return out
    out["defined"] = True
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, completion.size, size=(BOOTSTRAP_REPLICATIONS, completion.size))
    # Mean per replicate first, then the undefined mask, so a replicate whose resampled
    # denominator is not positive is dropped as a whole instead of being averaged over the
    # seeds it happens to contain.
    replicate_completion = completion[indices].mean(axis=1)
    defined = replicate_completion > 0.0
    out["undefined_replicates"] = int(replicate_completion.size - defined.sum())
    for key in ("total_bytes", "uplink_bytes", "ack_bytes"):
        numerator = deltas[key]
        replicate_numerator = numerator[indices].mean(axis=1)
        ratios = replicate_numerator[defined] / replicate_completion[defined]
        out[f"{key}_ratio"] = round(float(numerator.mean() / completion.mean()), 1)
        out[f"{key}_ratio_ci95"] = (
            [
                round(float(np.percentile(ratios, 2.5)), 1),
                round(float(np.percentile(ratios, 97.5)), 1),
            ]
            if ratios.size
            else None
        )
    return out


def new_failures(
    rows: list[dict[str, Any]], arm: str, baseline: str
) -> dict[str, Any]:
    """Per-target accounting: which specific targets this arm lost against the baseline."""
    left = {int(row["seed"]): row for row in rows if row["arm"] == arm}
    right = {int(row["seed"]): row for row in rows if row["arm"] == baseline}
    lost: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    for seed in sorted(set(left) & set(right)):
        mine = left[seed]["targets_outcome"]
        theirs = right[seed]["targets_outcome"]
        for target_id in sorted(set(mine) & set(theirs)):
            if theirs[target_id] == "on_time" and mine[target_id] != "on_time":
                lost.append(
                    {"seed": seed, "target_id": target_id, "outcome": mine[target_id]}
                )
            elif mine[target_id] == "on_time" and theirs[target_id] != "on_time":
                recovered.append(
                    {"seed": seed, "target_id": target_id, "outcome": theirs[target_id]}
                )
    return {
        "arm": arm,
        "baseline": baseline,
        "new_failures": len(lost),
        "new_failure_cases": lost,
        "recovered": len(recovered),
        "recovered_cases": recovered,
    }


def pareto_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which locked arms survive on the fresh data, compared per stream on both axes."""
    by_stream: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_stream.setdefault(int(row["seed"]), {})[row["arm"]] = row
    arms = sorted({row["arm"] for row in rows})
    complete = [arm for arm in arms if all(arm in entries for entries in by_stream.values())]
    dominated: dict[str, list[str]] = {}
    for arm in complete:
        for other in complete:
            if other == arm:
                continue
            at_least_as_good = all(
                entries[other]["on_time"] >= entries[arm]["on_time"]
                and entries[other]["total_bytes"] <= entries[arm]["total_bytes"]
                for entries in by_stream.values()
            )
            strictly_better = any(
                entries[other]["on_time"] > entries[arm]["on_time"]
                or entries[other]["total_bytes"] < entries[arm]["total_bytes"]
                for entries in by_stream.values()
            )
            if at_least_as_good and strictly_better:
                dominated.setdefault(arm, []).append(other)
    return {
        "definition": (
            "an arm is dominated when another arm is per-stream no worse on on-time "
            "completions and no costlier in total bytes, and strictly better on at least one "
            "stream"
        ),
        "arms_compared": complete,
        "streams": len(by_stream),
        "frontier": [arm for arm in complete if arm not in dominated],
        "dominated": {arm: sorted(sources) for arm, sources in sorted(dominated.items())},
    }


# ------------------------------------------------------------------ the judgement
def judge(rows: list[dict[str, Any]], setting: str) -> dict[str, Any]:
    """The protocol's four pre-registered readings, applied mechanically."""
    baseline = "noack_send_once"
    verdict: dict[str, Any] = {"setting": setting}

    def difference(arm: str, versus: str, key: str) -> dict[str, Any]:
        scoped = [
            row for row in rows if row["arm"] in (arm, versus)
        ]
        return _difference(scoped, arm, versus, key)

    completion_retry = difference("ack_retry_2", baseline, "on_time")
    total_retry = difference("ack_retry_2", baseline, "total_bytes")
    retry_improves = bool(
        completion_retry["ci95_lower"] is not None and completion_retry["ci95_lower"] > 0.0
    )

    if setting == "isolated-lossless":
        checks: dict[str, Any] = {}
        for arm in NEGATIVE_CONTROL_ARMS:
            completion = difference(arm, baseline, "on_time")
            checks[arm] = {
                "completion": completion,
                "improves": bool(
                    completion["ci95_lower"] is not None and completion["ci95_lower"] > 0.0
                ),
                "total_bytes": difference(arm, baseline, "total_bytes"),
            }
        violations = [arm for arm, block in checks.items() if block["improves"]]
        verdict["negative_control"] = {
            "expectation": "no ACK rule should gain completions where nothing is ever lost",
            "arms": checks,
            "violations": violations,
            "passed": not violations,
            "action": (
                "pause and check whether the ACK or the rule changed behaviour beyond the "
                "feedback itself"
                if violations
                else "proceed: the negative control behaved as the protocol expects"
            ),
            "note": (
                "the ACK arms' higher total bytes on this setting are the protocol's cost, "
                "not a pass and not a failure"
            ),
        }
    if setting == "isolated-lossy":
        verdict["lossy_confirmation"] = {
            "completion": completion_retry,
            "total_bytes": total_retry,
            "reproduced": retry_improves,
            "action": (
                "round 6's cost-effect effect gains independent support on fresh streams; "
                "report the byte increase beside the task gain"
                if retry_improves
                else "the +0.875 targets/episode measured on the training streams does not "
                "gain independent support: stop the deployment discussion"
            ),
        }
    if setting == "normal-lossy":
        failures = new_failures(rows, "ack_retry_2", baseline)
        same_outcome = bool(
            completion_retry["mean_difference"] == 0.0
            and (completion_retry["seeds_worse"] or 0) == 0
        )
        verdict["normal_path"] = {
            "completion": completion_retry,
            "total_bytes": total_retry,
            "new_failures": failures["new_failures"],
            "new_failure_cases": failures["new_failure_cases"],
            "recovered": failures["recovered"],
            "proposed_as_optional_mode": retry_improves,
            "same_outcome_but_dearer": bool(
                same_outcome
                and total_retry["ci95_lower"] is not None
                and total_retry["ci95_lower"] > 0.0
            ),
            "action": (
                "propose ack_retry_2 as an optional high-reliability mode candidate only; the "
                "default stays unchanged and the user decides on the bytes"
                if retry_improves
                else "do not propose deployment: the real information path gives no completion "
                "gain to justify the bytes"
            ),
        }
    frontier_checks: dict[str, Any] = {}
    for arm in ("ack_periodic_20", "ack_backoff_16"):
        completion = difference(arm, "ack_retry_2", "on_time")
        frontier_checks[arm] = {
            "completion": completion,
            "beats_the_locked_rule": bool(
                completion["ci95_lower"] is not None and completion["ci95_lower"] > 0.0
            ),
        }
    unstable = [arm for arm, block in frontier_checks.items() if block["beats_the_locked_rule"]]
    verdict["frontier"] = {
        "checks": frontier_checks,
        "unstable": unstable,
        "action": (
            "report the rule choice as unstable on fresh data; do not re-tune or rename the "
            "locked primary rule"
            if unstable
            else "the locked rule stays the strongest completion rule of the set on this data"
        ),
    }
    verdict["pareto"] = pareto_frontier(rows)
    return verdict


def report(args: argparse.Namespace) -> int:
    rows = load_rows()
    if not rows:
        raise SystemExit("no matrix rows; run --mode matrix first")
    registration = load_registration()
    rejected = {
        f"{row['setting']}/{row['arm']}/{row['seed']}": issue
        for row in rows
        for issue in validate_row(row, registration)
    }
    payload: dict[str, Any] = {
        "block": [common.BLOCK_START, common.BLOCK_STOP - 1],
        "budget_bytes": common.BUDGET_BYTES,
        "settings": {},
        "comparisons": {},
        "ratios": {},
        "new_failures": {},
        "judgement": {},
        "episodes": len(rows),
        "episodes_expected": len(job_list(common.BLOCK_SEEDS)),
        "incomplete_cells": {},
        "revalidated_rejections": rejected,
        "statistics": {
            "method": "seed-level paired percentile bootstrap (evaluate_pilot.paired)",
            "replications": BOOTSTRAP_REPLICATIONS,
            "rng_seed": BOOTSTRAP_SEED,
            "settings_pooled": False,
        },
        "randomness": {
            "forward": (
                "per-episode np.random.default_rng(scenario_seed) stream, one draw per "
                "(subscriber, emission): same mechanism and seed for every arm, so a "
                "same-distribution comparison, not the same events"
            ),
            "reverse": (
                "counter-based (episode_seed, 'uplink_ack', subscriber, target, ordinal) plus "
                "the delivery tick: common random numbers between ACK-on arms"
            ),
        },
    }
    for setting in common.SETTING_LABELS:
        scoped = [row for row in rows if row["setting"] == setting]
        summary = aggregate(scoped)
        payload["settings"][setting] = summary
        payload["incomplete_cells"][setting] = {
            arm: int(len(common.BLOCK_SEEDS)) - int(summary[arm]["episodes"])
            for arm in summary
            if int(summary[arm]["episodes"]) != len(common.BLOCK_SEEDS)
        }
        payload["comparisons"][setting] = {}
        payload["ratios"][setting] = {}
        payload["new_failures"][setting] = {}
        for arm, versus, note in COMPARISONS:
            seeds, deltas = delta_vectors(scoped, arm, versus)
            payload["comparisons"][setting][f"{arm} - {versus}"] = {
                "note": note,
                "pairs": len(seeds),
                "axes": {key: _difference(scoped, arm, versus, key) for key in AXES},
            }
            payload["ratios"][setting][f"{arm} - {versus}"] = ratio_bootstrap(deltas)
            payload["new_failures"][setting][f"{arm} - {versus}"] = new_failures(
                scoped, arm, versus
            )
        payload["judgement"][setting] = judge(scoped, setting)
    common.write_json(common.json_path("COMM_RULE_CONFIRMATION_RESULTS.json"), payload)
    for setting in common.SETTING_LABELS:
        print(f"\n=== {setting} (uplink and reverse delivery "
              f"{common.SETTINGS[setting]['delivery_probability']}, "
              f"{common.SETTINGS[setting]['information']} information path) ===")
        summary = payload["settings"][setting]
        print(f"{'arm':<18}{'onT/ep':>8}{'uplink':>9}{'ack':>8}{'total':>9}{'used':>7}{'age':>8}")
        for arm in common.ARM_LABELS:
            if arm not in summary:
                continue
            entry = summary[arm]
            print(
                f"{arm:<18}{entry['on_time_per_episode']:>8.4f}"
                f"{entry['uplink_bytes']:>9.0f}{entry['ack_bytes']:>8.0f}"
                f"{entry['total_bytes']:>9.0f}{entry['budget_used_fraction']:>7.3f}"
                f"{entry['belief_age_steps']:>8.1f}"
            )
        for key, block in payload["comparisons"][setting].items():
            completion = block["axes"]["on_time"]
            total = block["axes"]["total_bytes"]
            print(
                f"  {key}: onT {completion['mean_difference']} "
                f"[{completion['ci95_lower']}, {completion['ci95_upper']}] "
                f"better/equal/worse {completion['seeds_better']}/"
                f"{completion['seeds_equal']}/{completion['seeds_worse']}"
                f" | totalB {total['mean_difference']} "
                f"[{total['ci95_lower']}, {total['ci95_upper']}]"
            )
        print(f"  frontier: {payload['judgement'][setting]['pareto']['frontier']}")
        print(f"  dominated: {payload['judgement'][setting]['pareto']['dominated']}")
    return 1 if rejected else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("matrix", "report", "probe"), default="matrix")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="probe only")
    args = parser.parse_args()
    if args.mode == "matrix":
        return matrix(args)
    if args.mode == "probe":
        return probe(args)
    return report(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
