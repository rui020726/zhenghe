"""Freeze the stage-one point list from archived pre-state fields only.

    python select_points.py --mode digest
    python select_points.py --mode initial
    python select_points.py --mode selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import ms_common as common


def source_digest() -> dict[str, Any]:
    payload = common.read_json(common.COUNTERFACTUAL_JSON)
    probes = common.load_probe_points()
    eligible_counts = {
        str(seed): sum(1 for point in points if common.eligible(point))
        for seed, points in sorted(probes.items())
    }
    return {
        "path": "runs/comm_ack_value_20260920/COMM_ACK_VALUE_COUNTERFACTUAL_25pct.json",
        "baseline_rule": payload.get("baseline_rule"),
        "channel": payload.get("channel"),
        "streams": len(probes),
        "total_points": sum(len(points) for points in probes.values()),
        "eligible_points": sum(eligible_counts.values()),
        "eligible_per_stream": eligible_counts,
        "allowed_fields": [
            "step",
            "target",
            "confirmed_fraction",
            "unacked",
            "age_steps",
            "changed_view",
            "rule_sent",
        ],
    }


def select_initial() -> dict[str, Any]:
    probes = common.load_probe_points()
    streams = sorted(probes)
    if streams != sorted(common.TRAIN_SEEDS):
        raise RuntimeError("probe streams are not exactly the 48 training seeds")
    rng = np.random.default_rng(common.STREAM_SAMPLE_SEED)
    drawn = list(rng.permutation(streams)[: common.STREAM_SAMPLE_DRAWN])
    used = drawn[: common.STREAM_SAMPLE_USED]
    unused = drawn[common.STREAM_SAMPLE_USED :]
    point_rng = np.random.default_rng(common.POINT_SELECTION_SEED)
    rows: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    for seed in used:
        candidates = [
            common.strip_outcome_fields(point)
            for point in probes[seed]
            if common.eligible(point)
        ]
        if not candidates:
            shortfalls.append({"seed": int(seed), "reason": "no_informative_point"})
            rows.append(
                {
                    "seed": int(seed),
                    "status": "no_informative_point",
                    "eligible_count": 0,
                }
            )
            continue
        # Independent, documented draw among eligible pre-state points only.
        index = int(point_rng.integers(0, len(candidates)))
        chosen = dict(candidates[index])
        chosen.update(
            {
                "seed": int(seed),
                "status": "selected",
                "eligible_count": len(candidates),
                "choice_index": index,
            }
        )
        rows.append(chosen)
    payload = {
        "schema": "comm-multistep-points-initial/1",
        "stream_sample_seed": common.STREAM_SAMPLE_SEED,
        "stream_sample_drawn": drawn,
        "stream_sample_used": used,
        "stream_sample_unused": unused,
        "point_selection_seed": common.POINT_SELECTION_SEED,
        "baseline_rule": common.BASELINE_RULE,
        "horizons": list(common.HORIZONS),
        "max_simulation_starts": common.MAX_SIMULATION_STARTS,
        "rows": rows,
        "shortfalls": shortfalls,
    }
    return payload


def selftest() -> None:
    """Selection must not depend on outcome fields of the archived counterfactual."""
    probes = common.load_probe_points()
    mutated = {}
    for seed, points in probes.items():
        flipped = []
        for point in points:
            clone = dict(point)
            # Poison every outcome-looking field. Selection must ignore them.
            clone["send"] = {"on_time": -1}
            clone["hold"] = {"on_time": 99}
            clone["free"] = {"on_time": 0}
            clone["status"] = "poisoned"
            flipped.append(clone)
        mutated[seed] = flipped
    # Recompute selection on poisoned copies with the same code path's eligibility helper.
    streams = sorted(mutated)
    rng = np.random.default_rng(common.STREAM_SAMPLE_SEED)
    drawn = list(rng.permutation(streams)[: common.STREAM_SAMPLE_DRAWN])
    used = drawn[: common.STREAM_SAMPLE_USED]
    point_rng = np.random.default_rng(common.POINT_SELECTION_SEED)
    chosen_pure = []
    chosen_poison = []
    for seed in used:
        pure = [
            common.strip_outcome_fields(p) for p in probes[seed] if common.eligible(p)
        ]
        poison = [
            common.strip_outcome_fields(p) for p in mutated[seed] if common.eligible(p)
        ]
        if not pure:
            continue
        index_a = int(np.random.default_rng(common.POINT_SELECTION_SEED).integers(0, 1))
        # Same deterministic index stream as select_initial: rebuild identically.
        chosen_pure.append((seed, len(pure), pure[0]))
        chosen_poison.append((seed, len(poison), poison[0]))
        assert index_a in (0, 1)
    # Stronger check: eligibility and stripped fields are identical under poisoning.
    for seed in used:
        pure = [common.strip_outcome_fields(p) for p in probes[seed] if common.eligible(p)]
        poison = [
            common.strip_outcome_fields(p) for p in mutated[seed] if common.eligible(p)
        ]
        if pure != poison:
            raise AssertionError(f"selection inputs changed under outcome poisoning for {seed}")
        for row in pure:
            if any(key in row for key in ("send", "hold", "free", "status")):
                raise AssertionError(f"outcome field leaked into selection input: {row}")
    # Replay the exact select_initial draws on pure vs poisoned point lists.
    point_rng_a = np.random.default_rng(common.POINT_SELECTION_SEED)
    point_rng_b = np.random.default_rng(common.POINT_SELECTION_SEED)
    for seed in used:
        pure = [common.strip_outcome_fields(p) for p in probes[seed] if common.eligible(p)]
        poison = [
            common.strip_outcome_fields(p) for p in mutated[seed] if common.eligible(p)
        ]
        if not pure:
            continue
        index_a = int(point_rng_a.integers(0, len(pure)))
        index_b = int(point_rng_b.integers(0, len(poison)))
        if index_a != index_b or pure[index_a] != poison[index_b]:
            raise AssertionError(f"point choice depends on outcome fields for seed {seed}")
    print(json.dumps({"selftest": "passed", "streams": len(used)}, ensure_ascii=False))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("digest", "initial", "selftest"), required=True)
    args = parser.parse_args(argv)
    if args.mode == "digest":
        payload = source_digest()
        # Fix path key without relying on a typo-prone alias.
        payload["path"] = "runs/comm_ack_value_20260920/COMM_ACK_VALUE_COUNTERFACTUAL_25pct.json"
        common.write_json(common.SOURCE_CANDIDATES_DIGEST, payload)
        print(json.dumps({k: payload[k] for k in payload if k != "eligible_per_stream"}, indent=1))
        return 0
    if args.mode == "selftest":
        selftest()
        return 0
    payload = select_initial()
    common.write_json(common.POINTS_INITIAL, payload)
    selected = sum(1 for row in payload["rows"] if row.get("status") == "selected")
    print(
        json.dumps(
            {
                "used_streams": payload["stream_sample_used"],
                "selected_points": selected,
                "shortfalls": payload["shortfalls"],
            },
            ensure_ascii=False,
            indent=1,
            default=common._json_default,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
