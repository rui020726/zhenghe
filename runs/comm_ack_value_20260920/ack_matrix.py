"""The strong ACK-rule matrix: 48 training streams, both channel settings, every arm.

This is the first half of the information-value gate. It answers one question before any
network exists: with a real, costed, lossy reverse channel, is there an ACK rule that beats
the strongest no-ACK rule -- and is there *no* simple rule that beats all the others on both
axes at once? The second half is the counterfactual diagnostic (``ack_counterfactual.py``),
which asks whether the ACK state ever creates a genuine send/hold trade-off at all.

Every arm goes through the same episode runner with the same queue, credit, ACK merge and
billing; the only difference is which object decides what goes on the air. Two channel
settings run: the primary lossy one (uplink delivery 0.7) and lossless 1.0 as the negative
control the brief names. Costs are reported as three separate numbers -- uplink attempt
bytes, ACK attempt bytes, and their sum -- because "saved uplink bytes by spending them on
the reverse channel" is not a saving.

    python ack_matrix.py --mode matrix --shards 6 --shard 0 --threads 1
    python ack_matrix.py --mode report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import ack_common as common

TRAIN_STREAMS: tuple[int, ...] = common.GATE_STREAMS
PRIMARY_TIER = common.PRIMARY_TIER
CHANNEL_SETTINGS: dict[str, dict[str, Any]] = {
    # The primary setting the brief freezes.
    "lossy": {"delivery_probability": common.PRIMARY_DELIVERY_PROBABILITY},
    # The negative control: a lossless uplink. An ACK rule that only wins here has found a
    # way to spend bytes, not a way to use information.
    "lossless": {"delivery_probability": common.LOSSLESS_DELIVERY_PROBABILITY},
}


def run_arm(
    arm: str,
    seed: int,
    *,
    tier: str,
    channel: str,
    directory: Path,
    budget_bytes: int | None = None,
) -> dict[str, Any]:
    """One arm on one stream on one channel setting, with the whole transport ledger."""
    spec = common.arm_spec(arm)
    setting = CHANNEL_SETTINGS[channel]
    row = common.protocol_env.run_episode(
        seed,
        directory=directory,
        label=f"ack_{channel}_{arm}",
        policy=spec["policy"],
        policy_params=spec.get("params") and {
            key: value
            for key, value in (
                token.split("=", 1) for token in str(spec["params"]).split(",") if token
            )
        },
        budget_bytes=budget_bytes if budget_bytes is not None else common.tier_bytes(tier),
        information="isolated",
        delivery_probability=setting["delivery_probability"],
        delay_ticks=0,
        max_wait_steps=None,
        urgent_wait_steps=None,
        timing_mode=spec.get("mode", common.POLICY_MODE),
        ack=True,
        ack_delay_ticks=common.ACK_DELAY_TICKS,
        ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
    )
    ledger = row["ledger"]
    uplink_bytes = int(ledger["airtime_bytes"])
    ack_bytes = int(ledger["ack_attempt_bytes"])
    return {
        "arm": arm,
        "policy": spec["policy"],
        "parameters": spec.get("params"),
        "tier": tier,
        "channel": channel,
        "delivery_probability": setting["delivery_probability"],
        "seed": int(seed),
        "on_time": int(row["task"]["on_time"]),
        "targets": int(row["task"]["targets"]),
        "expired": int(row["task"]["expired"]),
        "unresolved": int(row["task"]["unresolved"]),
        "uplink_bytes": uplink_bytes,
        "ack_bytes": ack_bytes,
        "total_bytes": uplink_bytes + ack_bytes,
        "emissions": int(ledger["emissions"]),
        "transmission_attempts": int(ledger["transmission_attempts"]),
        "packets_lost": int(ledger["packets_lost"]),
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "ack_attempts": int(ledger["ack_attempts"]),
        "ack_lost": int(ledger["ack_lost"]),
        "ack_frames_advanced": int(ledger["ack_frames_advanced"]),
        "ack_frames_reaffirmed": int(ledger["ack_frames_reaffirmed"]),
        "budget_refused_sends": int(ledger["budget_refused_sends"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "arm_label": row["arm_label"],
        "config_hash": row["config_hash"],
        "seconds": float(row["seconds"]),
        "artefacts": row["artefacts"],
    }


def shard_of(items: list[Any], shards: int, shard: int) -> list[Any]:
    return [item for index, item in enumerate(items) if index % shards == shard]


def matrix(args: argparse.Namespace) -> int:
    arms = sorted(common.ack_arm_specs())
    directory = common.evaluation_root(args.tier) / "matrix"
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [
        (arm, seed, channel)
        for channel in CHANNEL_SETTINGS
        for arm in arms
        for seed in TRAIN_STREAMS
    ]
    jobs = shard_of(jobs, args.shards, args.shard)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, (arm, seed, channel) in enumerate(jobs, start=1):
        row = run_arm(
            arm, seed, tier=args.tier, channel=channel, directory=directory
        )
        rows.append(row)
        if index % 40 == 0 or index == len(jobs):
            print(
                f"[matrix shard {args.shard}] {index}/{len(jobs)} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    payload = {
        "tier": args.tier,
        "budget_bytes": common.tier_bytes(args.tier),
        "channels": CHANNEL_SETTINGS,
        "arms": arms,
        "streams": list(TRAIN_STREAMS),
        "shard": args.shard,
        "shards": args.shards,
        "rows": rows,
    }
    common.write_json(
        common.evaluation_root(args.tier) / f"matrix_shard{args.shard}.json", payload
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "seconds": round(time.time() - started, 1),
                "output": str(common.evaluation_root(args.tier) / f"matrix_shard{args.shard}.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def load_matrix_rows(tier: str) -> list[dict[str, Any]]:
    root = common.evaluation_root(tier)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("matrix_shard*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def aggregate(rows: list[dict[str, Any]], channel: str) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["channel"] != channel:
            continue
        by_arm.setdefault(row["arm"], []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for arm, items in by_arm.items():
        summary[arm] = {
            "episodes": len(items),
            "on_time_per_episode": _mean([float(r["on_time"]) for r in items]),
            "uplink_bytes": _mean([float(r["uplink_bytes"]) for r in items]),
            "ack_bytes": _mean([float(r["ack_bytes"]) for r in items]),
            "total_bytes": _mean([float(r["total_bytes"]) for r in items]),
            "emissions_per_episode": _mean([float(r["emissions"]) for r in items]),
            "transmission_attempts": _mean([float(r["transmission_attempts"]) for r in items]),
            "packets_lost": _mean([float(r["packets_lost"]) for r in items]),
            "deliveries_applied": _mean([float(r["deliveries_applied"]) for r in items]),
            "ack_attempts": _mean([float(r["ack_attempts"]) for r in items]),
            "ack_lost": _mean([float(r["ack_lost"]) for r in items]),
            "ack_frames_advanced": _mean([float(r["ack_frames_advanced"]) for r in items]),
            "ack_frames_reaffirmed": _mean([float(r["ack_frames_reaffirmed"]) for r in items]),
            "budget_refused_sends": _mean([float(r["budget_refused_sends"]) for r in items]),
            "belief_age_steps": _mean([float(r["belief_age_steps"]) for r in items]),
        }
    return summary


def _paired(
    rows: list[dict[str, Any]], channel: str, arm: str, baseline: str, key: str
) -> dict[str, Any]:
    """Per-stream paired difference and its percentile bootstrap, the frozen implementation."""
    left = {
        int(row["seed"]): float(row[key])
        for row in rows
        if row["channel"] == channel and row["arm"] == arm
    }
    right = {
        int(row["seed"]): float(row[key])
        for row in rows
        if row["channel"] == channel and row["arm"] == baseline
    }
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"arm": arm, "baseline": baseline, "key": key, "streams": 0}
    pairs = [(left[seed], right[seed]) for seed in shared]
    result = common.paired_bootstrap(
        pairs, baseline=0.0, key=lambda pair: pair[0] - pair[1]
    )
    result.update({"arm": arm, "baseline": baseline, "key": key, "streams": len(shared)})
    return result


def report(args: argparse.Namespace) -> int:
    rows = load_matrix_rows(args.tier)
    if not rows:
        raise SystemExit(f"no matrix rows under {common.evaluation_root(args.tier)}")
    payload: dict[str, Any] = {
        "tier": args.tier,
        "budget_bytes": common.tier_bytes(args.tier),
        "streams": list(TRAIN_STREAMS),
        "channels": {},
        "comparison": {},
        "selection": {},
    }
    for channel in CHANNEL_SETTINGS:
        summary = aggregate(rows, channel)
        payload["channels"][channel] = summary
        # The strongest no-ACK reference is ``send_once`` (round 5's locked comparator); the
        # strongest ACK rule is selected by the frozen rule below, on the training streams,
        # before any learned number exists.
        baseline = "send_once"
        candidates = [
            arm for arm in common.selectable_ack_arms() if arm in summary
        ]
        ranked = sorted(
            candidates,
            key=lambda arm: (
                -summary[arm]["on_time_per_episode"],
                summary[arm]["total_bytes"],
                arm,
            ),
        )
        payload["selection"][channel] = {
            "rule": "highest on-time per episode, ties by lower total bytes, then name",
            "baseline": baseline,
            "candidates": candidates,
            "ranked": [
                {
                    "arm": arm,
                    "on_time_per_episode": summary[arm]["on_time_per_episode"],
                    "total_bytes": summary[arm]["total_bytes"],
                    "uplink_bytes": summary[arm]["uplink_bytes"],
                    "ack_bytes": summary[arm]["ack_bytes"],
                }
                for arm in ranked
            ],
            "selected": ranked[0] if ranked else None,
        }
        payload["comparison"][channel] = {
            "vs_send_once": {
                arm: _paired(rows, channel, arm, baseline, "on_time")
                for arm in sorted(summary)
            },
            "vs_send_once_total_bytes": {
                arm: _paired(rows, channel, arm, baseline, "total_bytes")
                for arm in sorted(summary)
            },
        }
    common.write_json(
        common.json_path(f"COMM_ACK_VALUE_RULE_MATRIX_{args.tier}.json"), payload
    )
    for channel in CHANNEL_SETTINGS:
        summary = payload["channels"][channel]
        print(f"\n=== {channel} (uplink delivery "
              f"{CHANNEL_SETTINGS[channel]['delivery_probability']}) ===")
        header = f"{'arm':<32}{'onT/ep':>9}{'uplink B':>10}{'ack B':>9}{'total B':>10}{'emis/ep':>9}"
        print(header)
        for arm in sorted(summary, key=lambda a: (-summary[a]["on_time_per_episode"], summary[a]["total_bytes"])):
            entry = summary[arm]
            print(
                f"{arm:<32}{entry['on_time_per_episode']:>9.4f}"
                f"{entry['uplink_bytes']:>10.0f}{entry['ack_bytes']:>9.0f}"
                f"{entry['total_bytes']:>10.0f}{entry['emissions_per_episode']:>9.2f}"
            )
        selected = payload["selection"][channel]["selected"]
        if selected:
            print(f"selected strongest ACK rule: {selected}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("matrix", "report"), default="matrix")
    parser.add_argument("--tier", default=PRIMARY_TIER)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.mode == "matrix":
        return matrix(args)
    return report(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
