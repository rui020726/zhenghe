"""§2: whose decision is the gain -- the learner's, the silence bound's, or its own weights'?

Round 2 reported one training seed that beat ``periodic_5`` for 34,768 fewer bytes at the same
completions, and two that did not. That is a result about a seed unless the alternatives are on
the table, so every arm here runs the same 24 validation streams, under the same locked credit,
through the same transport, and each one decides its own emissions:

* ``immediate`` -- one packet per accepted detection, never a refresh. Round 2 measured it at
  14.7 KB, a quarter of what the best learned model spent, so it is in the table whether or not
  it is convenient;
* ``hold_to_bound`` -- ask for nothing, ever. Whatever goes on the air is the shared 3/10-step
  silence rule speaking. A controller that only learned to hold has become this arm;
* ``random_send`` -- a seeded coin. Sparse-by-accident has to be beaten too;
* ``periodic_5`` / ``event`` / ``event_priority`` -- the pre-registered rules, including the
  urgency-and-staleness priority rule that is the strongest simple sender under a budget;
* ``event_unbudgeted`` -- the same rule with no credit, kept as a labelled ceiling, not a
  competitor;
* ``init_seed{1,2,3}`` -- each trained model's own untrained starting point, rebuilt from the
  recorded initialisation rule;
* ``learned_seed{1,2,3}`` -- round 2's three checkpoints at 100,000 environment steps.

Every attempt is then charged to one primary reason (``policy``, ``forced_maxwait``, ``other``)
and the ledger's identities are written out as residuals rather than asserted.
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

import attribution_metrics as M
import stability_common as common

import evaluate_pilot as pilot_eval  # the pilot round's statistic: same bootstrap, same CI method
import pilot_env
import train_pilot
from integrated_sar.comm_policy import parse_parameters

#: What each learned arm is compared against, and on which two metrics. Fixed before the first
#: episode was run, because the interesting number decides which comparator gets emphasised.
BASELINES = ("immediate", "hold_to_bound", "event", "event_priority", "periodic_5")
METRICS = ("on_time_targets_per_episode", "airtime_bytes")


def _tensors(payload: Any) -> list[Any]:
    """Every tensor in a (possibly nested) state dict, in a fixed order."""
    found: list[Any] = []
    for value in payload.values() if isinstance(payload, dict) else payload:
        if isinstance(value, dict):
            found.extend(_tensors(value))
        else:
            found.append(value)
    return found


def controller_for(spec: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The decision function for one arm, plus what a reader should know about its provenance."""
    if spec["kind"] in ("rule", "rule_ceiling"):
        return None, {"controller": spec["kind"]}
    training_seed = int(spec["training_seed"])
    if spec["kind"] == "init":
        controller = train_pilot.Controller(training_seed)
        twin = train_pilot.Controller(training_seed)
        identical = all(
            torch.equal(first, second) for first, second in zip(_tensors(controller.state_dict()), _tensors(twin.state_dict()))
        )
        return controller, {
            "controller": "reconstructed-initialisation",
            "reconstruction_is_deterministic": bool(identical),
            "archived_checkpoint": False,
            "weight_fingerprint": weight_fingerprint(controller),
        }
    path = common.PILOT_CHECKPOINTS[training_seed]
    loaded = pilot_eval.load_controller(path, training_seed)
    return loaded, {
        "controller": "pilot-checkpoint",
        "path": str(path),
        "sha256": common.sha256(path),
        "environment_steps": 100_000,
        "weight_fingerprint": weight_fingerprint(loaded),
    }


def weight_fingerprint(controller: Any) -> str:
    """Eight hex characters that identify these weights and no others."""
    import hashlib

    digest = hashlib.sha256()
    for tensor in _tensors(controller.state_dict()):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def run_arm(
    spec: dict[str, Any],
    seed: int,
    *,
    controller: Any,
    provenance: dict[str, Any],
    directory: Path,
    budget: int | None,
    information: str = "isolated",
    delivery_probability: float = 1.0,
    label: str | None = None,
) -> dict[str, Any]:
    collector = common.transport_bounded_collector(controller, greedy=True)
    unbounded = spec.get("bound") == "none"
    row = pilot_env.run_episode(
        seed,
        directory=directory,
        label=label or spec["arm"],
        policy=spec.get("policy") or "learned",
        policy_params=parse_parameters(spec.get("params") or "") or None,
        budget_bytes=budget,
        controller=collector.policy() if controller is not None else None,
        information=information,
        delivery_probability=delivery_probability,
        max_wait_steps=None if unbounded else pilot_env.MAX_WAIT_STEPS,
        urgent_wait_steps=None if unbounded else pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
    )
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    rollup = M._per_reason_rollup(records)
    ledger = row["ledger"]
    return {
        "arm": spec["arm"],
        "kind": spec["kind"],
        "collector": "transport_bounded",
        "information": row["information"],
        "delivery_probability": float(row["delivery_probability"]),
        "seed": int(seed),
        "budget_bytes": row["budget_bytes"],
        "arm_label": row["arm_label"],
        "config_hash": row["config_hash"],
        "provenance": provenance,
        "autonomous_share_of_bytes": rollup["autonomous_share_of_bytes"],
        "unexplained_verdicts": rollup["unexplained_verdicts"],
        "rollup": rollup,
        "closure": M._closure(ledger, rollup, spent_bytes=row["spent_bytes"]),
        "implementations_agree": M._implementations_agree(
            common.metrics.transport_accounting(records), ledger
        ),
        "inference_milliseconds_per_decision": round(
            1000.0 * collector.decide_seconds / max(1, collector.decisions), 4
        ),
        "decisions": int(collector.decisions),
        "candidates_seen": int(collector.candidates),
        **M._task_units(row, rollup["subscribers"]),
    }


def row_files(directory: Path) -> list[Path]:
    """Every shard file this round has written. One file per process, never one shared append."""
    return sorted(directory.glob("rows_attribution*.jsonl"))


def row_path(directory: Path, shard: str) -> Path:
    # A dot, not an underscore: ``rows_attribution_s0.jsonl`` reads as seed 0 to the data-split
    # auditor's filename pattern, and a shard label is not a scenario.
    return directory / f"rows_attribution{'.' + shard if shard else ''}.jsonl"


def done_keys(paths: list[Path]) -> set[tuple[str, int]]:
    """The global resume view: a cell done in any shard is done, wherever the next shard runs."""
    finished = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    finished.add((str(record["arm"]), int(record["seed"])))
    return finished


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[tuple[str, int]] = set()
    doubled = set()
    for row in rows:
        key = (str(row["arm"]), int(row["seed"]))
        if key in seen:
            doubled.add(key)
        seen.add(key)
    if doubled:
        raise SystemExit(f"duplicate rows, refusing to pool: {sorted(doubled)[:5]}")
    return rows


def run(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "attribution"
    directory.mkdir(parents=True, exist_ok=True)
    rows_path = row_path(directory, args.shard)
    seeds = list(common.VALIDATION_SEEDS)
    if args.seed_modulus:
        seeds = [seed for seed in seeds if seed % args.seed_modulus == args.seed_remainder]
    finished = done_keys(row_files(directory))
    started = time.time()
    written = 0
    for spec in common.ARMS:
        if args.arms and spec["arm"] not in args.arms:
            continue
        budget = None if spec["kind"] == "rule_ceiling" else args.budget_bytes
        controller, provenance = controller_for(spec)
        for seed in seeds:
            if (spec["arm"], int(seed)) in finished:
                continue
            row = run_arm(
                spec, seed, controller=controller, provenance=provenance, directory=directory, budget=budget
            )
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            residuals = {key: value for key, value in row["closure"]["residuals"].items() if value}
            disagreements = {key: value for key, value in row["implementations_agree"].items() if value}
            if residuals or disagreements:
                print(
                    f"!! {row['arm']} s{seed}: residuals={residuals} implementations={disagreements}",
                    flush=True,
                )
            print(
                f"[{row['arm']:<16} s{seed}] on_time={row['on_time_targets_per_episode']}/"
                f"{row['detected_targets_this_episode']} bytes={row['airtime_bytes']:>6} "
                f"emissions={row['emissions']:>4} "
                f"autonomous={round(100 * (row['autonomous_share_of_bytes'] or 0)):>3}% "
                f"({row['seconds']}s) [{written} new, {time.time() - started:.0f}s]",
                flush=True,
            )
    print(f"shard finished: {written} new episodes", flush=True)
    return 0


def paired_vs_baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({str(row["arm"]) for row in rows})
    comparison: dict[str, Any] = {}
    for baseline in BASELINES:
        if baseline not in arms:
            continue
        comparison[baseline] = {}
        for metric in METRICS:
            comparison[baseline][metric] = pilot_eval.paired(
                rows, baseline=baseline, extract=lambda row, metric=metric: float(row[metric])
            )["arms"]
    return comparison


def attribution_metrics_short(row: dict[str, Any]) -> int:
    return M._generation_balance_shortfall(row["closure"])


def print_table(table: dict[str, Any]) -> None:
    header = (
        f"{'arm':<18}{'eps':>4}{'onT/ep':>8}{'rate/ep':>9}{'pooled':>8}{'bytes':>9}"
        f"{'auton%':>8}{'batchB':>8}{'pktB':>7}{'age':>7}{'unknow':>8}"
    )
    print(header)
    print("-" * len(header))
    for arm, values in table.items():
        reasons = values["by_reason"]
        total = sum(int(bucket["bytes"]) for bucket in reasons.values())
        print(
            f"{arm:<18}{values['episodes']:>4}"
            f"{values['on_time_targets_per_episode_mean']:>8}"
            f"{values['on_time_rate_per_episode_mean']:>9}"
            f"{values['pooled_on_time_over_detected_targets']:>8}"
            f"{values['airtime_bytes_mean']:>9}"
            f"{round(100 * int(reasons['policy']['bytes']) / total) if total else 0:>7}%"
            f"{values['bytes_per_emission_batch_mean']:>8}"
            f"{values['bytes_per_receiver_packet_mean']:>7}"
            f"{values['belief_age_steps_mean']:>7}"
            f"{values['unknown_to_everyone_steps_mean']:>8}"
        )


def aggregate(args: argparse.Namespace) -> int:
    rows = read_rows(row_files(common.HERE / "attribution"))
    table = M.summarise(rows)
    residuals = {
        key: value
        for row in rows
        for key, value in row["closure"]["residuals"].items()
        if value
    }
    disagreements = {
        key: value
        for row in rows
        for key, value in row["implementations_agree"].items()
        if value
    }
    payload = {
        "setting": {"information": "isolated", "delivery_probability": 1.0, "budget_bytes": args.budget_bytes},
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "episodes": len(rows),
        "arms": sorted({str(row["arm"]) for row in rows}),
        "units_note": {
            "on_time_targets_per_episode_mean": "targets completed before their deadline, per episode, then averaged",
            "on_time_rate_per_episode_mean": "per-episode on_time / that episode's detected targets, in [0, 1]",
            "pooled_on_time_over_detected_targets": "total on_time / total detected targets; a different number from either mean above",
            "bytes_per_emission_batch": "airtime / emissions -- what one send opportunity cost",
            "bytes_per_receiver_packet": "airtime / transmission attempts -- what one packet to one robot cost",
        },
        "per_arm": table,
        "paired": paired_vs_baselines(rows),
        "ledger": {
            "residuals_found": residuals,
            "transcript_and_relay_disagreements": disagreements,
            "unexplained_verdicts": sorted({str(verdict) for row in rows for verdict in row["unexplained_verdicts"]}),
            # Sized, not smoothed over: how many episodes fell short of the generation identity,
            # by how much, and on which arms. See attribution_metrics' note on the blind spot.
            "generation_balance_shortfall": {
                "episodes": sum(
                    1 for row in rows if attribution_metrics_short(row) < 0
                ),
                "total_units": sum(attribution_metrics_short(row) for row in rows),
                "per_arm": {
                    arm: sum(
                        attribution_metrics_short(row)
                        for row in rows
                        if row["arm"] == arm
                    )
                    for arm in sorted({str(row["arm"]) for row in rows})
                },
                "meaning": (
                    "negative units are second changes spotted by polling the registry while an "
                    "older one is still queued, which coalesced_changes does not count; no cost, "
                    "delivery or outcome identity is affected"
                ),
            },
            "generated_balance_mean_per_arm": {
                arm: {
                    key: round(float(np.mean([
                        row["closure"]["generated_balance"][key]
                        for row in rows
                        if row["arm"] == arm
                    ])), 2)
                    for key in (
                        "generated_items", "emissions", "redundant_emissions", "coalesced_changes",
                        "suppressed_terminal", "unspoken_or_uncounted_at_end", "unsent_change_steps",
                    )
                }
                for arm in sorted({str(row["arm"]) for row in rows})
            },
        },
        "provenance": {
            arm: next(row["provenance"] for row in rows if row["arm"] == arm)
            for arm in sorted({str(row["arm"]) for row in rows if row["kind"] != "rule"})
        },
    }
    common.write_json(common.json_path("COMM_POLICY_ATTRIBUTION.json"), payload)
    print_table(table)
    print(json.dumps({"ledger": payload["ledger"]["residuals_found"], "disagreements": payload["ledger"]["transcript_and_relay_disagreements"]}, ensure_ascii=False))
    print(json.dumps(payload["paired"], ensure_ascii=False, indent=1))
    return 0


def collector_delta(args: argparse.Namespace) -> int:
    """What the collector correction is worth, measured on the same 24 streams.

    Round 2 let the learned arm's mask put its forced sends inside its own priority order; every
    rule arm has them appended behind whatever it asked for. The set of packets a *free* policy
    wants is unchanged, so the only thing this can move is who gets the last of a exhausted
    credit. That is worth a number and not a paragraph.
    """
    import train_pilot

    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "collector_delta"
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in common.ARMS:
        if spec["kind"] != "checkpoint":
            continue
        controller, provenance = controller_for(spec)
        for seed in common.VALIDATION_SEEDS:
            collector = train_pilot.Collector(controller, greedy=True)
            row = pilot_env.run_episode(
                seed,
                directory=directory,
                label=f"pilot_collector_{spec['arm']}",
                policy="learned",
                budget_bytes=args.budget_bytes,
                controller=collector.policy(),
                information="isolated",
                delivery_probability=1.0,
            )
            rows.append(
                {
                    "arm": spec["arm"],
                    "seed": int(seed),
                    "on_time": int(row["task"]["on_time"]),
                    "bytes": int(row["spent_bytes"]),
                    "emissions": int(row["ledger"]["emissions"]),
                    "refusals": int(row["ledger"]["budget_refused_sends"]),
                }
            )
    with (directory / "rows_collector_delta.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    current = {
        (row["arm"], row["seed"]): row
        for row in read_rows(row_files(common.HERE / "attribution"))
    }
    comparison: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        pairs = [
            (before, current[(arm, before["seed"])])
            for before in rows
            if before["arm"] == arm and (arm, before["seed"]) in current
        ]
        comparison[arm] = {
            "episodes_compared": len(pairs),
            "same_bytes": sum(1 for before, after in pairs if before["bytes"] == after["airtime_bytes"]),
            "same_on_time": sum(
                1 for before, after in pairs if before["on_time"] == after["on_time_targets_per_episode"]
            ),
            "mean_bytes_pilot_collector": _mean(before["bytes"] for before, _ in pairs),
            "mean_bytes_transport_bounded": _mean(after["airtime_bytes"] for _, after in pairs),
            "mean_on_time_pilot_collector": _mean(before["on_time"] for before, _ in pairs),
            "mean_on_time_transport_bounded": _mean(
                after["on_time_targets_per_episode"] for _, after in pairs
            ),
        }
    common.write_json(common.json_path("COMM_COLLECTOR_CORRECTION.json"), {"provenance": provenance, "per_arm": comparison})
    print(json.dumps(comparison, ensure_ascii=False, indent=1))
    return 0


def _mean(values: Any) -> float:
    values = list(values)
    return round(sum(values) / len(values), 2) if values else 0.0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "aggregate", "collector-delta"), default="run")
    parser.add_argument("--arms", nargs="*", default=[])
    parser.add_argument("--shard", default="", help="suffix for this process's row file")
    parser.add_argument("--budget-bytes", type=int, default=common.BUDGET_BYTES)
    parser.add_argument("--seed-modulus", type=int, default=0)
    parser.add_argument("--seed-remainder", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.mode == "aggregate":
        return aggregate(args)
    if args.mode == "collector-delta":
        return collector_delta(args)
    return run(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
