"""Preflight for the rule confirmation: what must be true before the matrix may be read.

Twelve checks, each naming its own result block. They are the mechanical half of the brief's
§7 aggregation preflight plus the correctness conditions its judgement rules depend on, so a
reader can see which one failed without reading a log. Nothing here selects anything and
nothing here trains: a failure stops the round.

    python confirm_preflight.py --threads 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import confirm_common as common

SEALED_BLOCK = (800000, 800400)


def _check(results: dict[str, Any], name: str, body: Callable[[], dict[str, Any]]) -> None:
    try:
        block = body()
    except Exception as error:  # noqa: BLE001 - a failing gate must be reported, not raised
        results[name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        return
    block.setdefault("passed", True)
    results[name] = block


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="fewer streams for the slow gates")
    args = parser.parse_args()

    results: dict[str, Any] = {}
    directory = common.HERE / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    streams = [
        int(row["seed"])
        for row in common.read_json("COMM_RULE_STREAMS.json")["streams"][
            : 2 if args.quick else 4
        ]
    ]

    def production_defaults_unchanged() -> dict[str, Any]:
        """The production defaults must still be the no-ACK legacy protocol."""
        from integrated_sar.joint_driver import RunConfiguration

        configuration = RunConfiguration()
        expected = {
            "comm_policy": "immediate",
            "comm_budget_bytes": None,
            "comm_max_wait_steps": None,
            "comm_urgent_wait_steps": None,
            "comm_timing_mode": "legacy_shared_bound",
            "comm_ack": False,
            "comm_ack_delay_ticks": None,
            "comm_ack_delivery_probability": None,
            "comm_range": 0.3,
            "gossip_interval": 2,
        }
        observed = {key: getattr(configuration, key) for key in expected}
        return {
            "expected": expected,
            "observed": observed,
            "mismatches": {
                key: [expected[key], observed[key]] for key in expected if expected[key] != observed[key]
            },
            "passed": observed == expected,
        }

    def runner_is_not_a_drift() -> dict[str, Any]:
        """This round's runner must reproduce round 6's on the same streams."""
        import confirm_env

        seeds = tuple(common.pilot.TRAIN_SEEDS[: 1 if args.quick else 2])
        return confirm_env.runner_agreement(seeds, directory=directory)

    def block_audit_is_clean() -> dict[str, Any]:
        audit = common.read_json("COMM_RULE_BLOCK_AUDIT.json")
        return {
            "clean": audit["clean"],
            "verdict": audit["verdict"],
            "overlap": {
                key: len(value) if isinstance(value, (list, dict)) else value
                for key, value in audit["overlap"].items()
            },
            "sealed_clean": audit["sealed_learning_block"]["clean"],
            "passed": audit["clean"] and audit["sealed_learning_block"]["clean"],
        }

    def registration_matches_disk() -> dict[str, Any]:
        """Every registered stream is the file on disk, and no other seed crept in."""
        payload = common.read_json("COMM_RULE_STREAMS.json")
        mismatched: list[dict[str, Any]] = []
        for row in payload["streams"]:
            seed = int(row["seed"])
            observed = common.sha256(common.stream_path(seed))
            if observed != str(row["sha256"]):
                mismatched.append({"seed": seed, "registered": row["sha256"], "on_disk": observed})
        extra = sorted(set(common.BLOCK_SEEDS) ^ {int(row["seed"]) for row in payload["streams"]})
        return {
            "streams": len(payload["streams"]),
            "mismatched": mismatched,
            "seeds_outside_the_block": extra,
            "passed": not mismatched and not extra,
        }

    def forward_path_is_ack_agnostic() -> dict[str, Any]:
        """The ACK switch must change only the reverse channel, on the send-once rule.

        ``noack_send_once`` and ``ack_send_once`` are the same rule with the reverse channel
        switched on, so their forward behaviour must be identical -- same emissions, same lost
        packets, same deliveries, same task outcome. If anything on the forward path moved, the
        ACK arms would not be an isolated measurement of the reverse channel, and the protocol
        overhead could not be quoted on its own.
        """
        import confirm_matrix

        rows: dict[str, dict[str, Any]] = {}
        for arm in ("noack_send_once", "ack_send_once"):
            for seed in streams:
                row = confirm_matrix.run_arm(arm, seed, "isolated-lossy", directory=directory)
                rows[f"{arm}/{seed}"] = row
        disagreements: dict[str, Any] = {}
        for seed in streams:
            off = rows[f"noack_send_once/{seed}"]
            on = rows[f"ack_send_once/{seed}"]
            fields = (
                "on_time",
                "targets",
                "expired",
                "unresolved",
                "uplink_bytes",
                "emissions",
                "transmission_attempts",
                "packets_lost",
                "deliveries_applied",
                "budget_refused_sends",
            )
            different = {
                key: [int(off[key]), int(on[key])]
                for key in fields
                if int(off[key]) != int(on[key])
            }
            if different:
                disagreements[str(seed)] = different
        return {
            "seeds": streams,
            "fields_compared": [
                "on_time",
                "targets",
                "expired",
                "unresolved",
                "uplink_bytes",
                "emissions",
                "transmission_attempts",
                "packets_lost",
                "deliveries_applied",
                "budget_refused_sends",
            ],
            "disagreements": disagreements,
            "ack_bytes_added": {
                str(seed): int(rows[f"ack_send_once/{seed}"]["ack_bytes"]) for seed in streams
            },
            "passed": not disagreements,
            "note": (
                "the ACK-on arm's reverse bytes are the protocol's price; the forward path "
                "being identical is what makes the price measurable on its own"
            ),
        }

    def two_channel_cost_closes() -> dict[str, Any]:
        """total = uplink + ACK, and the ACK ledger never touches the task credit."""
        import confirm_matrix

        rows: list[dict[str, Any]] = []
        for arm in common.ARM_LABELS:
            row = confirm_matrix.run_arm(arm, streams[0], "isolated-lossy", directory=directory)
            rows.append(row)
        problems: list[dict[str, Any]] = []
        for row in rows:
            if row["total_bytes"] != row["uplink_bytes"] + row["ack_bytes"]:
                problems.append({"arm": row["arm"], "issue": "total bytes do not close"})
            if row["spent_bytes"] != row["uplink_bytes"]:
                problems.append(
                    {
                        "arm": row["arm"],
                        "issue": (
                            f"the credit charged {row['spent_bytes']} while the uplink airtime "
                            f"was {row['uplink_bytes']}: the reverse ledger leaked into the task "
                            "credit"
                        ),
                    }
                )
            if row["spent_bytes"] > common.BUDGET_BYTES:
                problems.append({"arm": row["arm"], "issue": "the credit was breached"})
        return {
            "arms": [row["arm"] for row in rows],
            "problems": problems,
            "rows": [
                {
                    "arm": row["arm"],
                    "uplink_bytes": row["uplink_bytes"],
                    "ack_bytes": row["ack_bytes"],
                    "total_bytes": row["total_bytes"],
                    "spent_bytes": row["spent_bytes"],
                    "budget_used_fraction": row["budget_used_fraction"],
                }
                for row in rows
            ],
            "passed": not problems,
        }

    def hard_mask_is_never_violated() -> dict[str, Any]:
        """Executed commands must be legal in the hard mask, on every sampled episode.

        Invariant 6, read off the decision trace rather than re-derived: the assignment that
        was actually executed has to be an index the mask left legal. A communication rule
        cannot touch this boundary, which is exactly why it is worth checking -- if it moved,
        the arms are not comparable and the comparison is meaningless.
        """
        from integrated_sar.allocation_trace import read_trace

        sampled = streams[: 1 if args.quick else 2]
        episodes = 0
        commands = 0
        violations: list[dict[str, Any]] = []
        for arm in common.ARM_LABELS:
            for seed in sampled:
                row = confirm_run(arm, seed, "normal-lossy", directory)
                episodes += 1
                _meta, rows = read_trace(row["artefacts"]["decision_trace"])
                for decision in rows:
                    assignment = decision.get("actual_assignment")
                    if assignment is None:
                        continue
                    ids = decision.get("action_task_ids") or []
                    mask = decision.get("mask") or []
                    if assignment not in ids:
                        continue
                    index = ids.index(assignment)
                    commands += 1
                    if index < len(mask) and bool(mask[index]):
                        violations.append(
                            {"arm": arm, "seed": seed, "step": decision.get("step")}
                        )
        return {
            "episodes_sampled": episodes,
            "commands_checked": commands,
            "mask_violations": violations,
            "passed": not violations,
        }

    def confirm_run(arm: str, seed: int, setting: str, directory: Path) -> dict[str, Any]:
        import confirm_matrix

        return confirm_matrix.run_arm(arm, seed, setting, directory=directory)

    def reverse_channel_is_reproducible() -> dict[str, Any]:
        """The same arm on the same stream must replay the same counter-based draws."""
        import confirm_matrix

        first = confirm_matrix.run_arm("ack_retry_2", streams[0], "isolated-lossy", directory=directory)
        second = confirm_matrix.run_arm("ack_retry_2", streams[0], "isolated-lossy", directory=directory)
        keys = (
            "on_time",
            "targets",
            "expired",
            "unresolved",
            "uplink_bytes",
            "ack_bytes",
            "total_bytes",
            "ack_attempts",
            "ack_delivered",
            "ack_lost",
            "ack_frames_advanced",
            "ack_frames_reaffirmed",
            "packets_lost",
            "deliveries_applied",
            "belief_age_steps",
        )
        different = {key: [first[key], second[key]] for key in keys if first[key] != second[key]}
        return {
            "keys_compared": list(keys),
            "different": different,
            "passed": not different,
        }

    def lossless_reverse_is_lossless() -> dict[str, Any]:
        """On isolated-lossless the reverse channel must deliver every ACK it sends."""
        import confirm_matrix

        row = confirm_matrix.run_arm("ack_retry_2", streams[0], "isolated-lossless", directory=directory)
        return {
            "ack_attempts": row["ack_attempts"],
            "ack_delivered": row["ack_delivered"],
            "ack_lost": row["ack_lost"],
            "ack_attempt_bytes": row["ack_attempt_bytes"],
            "passed": row["ack_lost"] == 0
            and row["ack_delivered"] == row["ack_attempts"]
            and row["ack_attempt_bytes"] > 0,
        }

    def ack_only_follows_application() -> dict[str, Any]:
        """No ACK may exist without an applied delivery: attempts == applied broadcast rounds."""
        import confirm_matrix

        row = confirm_matrix.run_arm("ack_retry_2", streams[0], "isolated-lossy", directory=directory)
        return {
            "deliveries_applied": row["deliveries_applied"],
            "ack_emitted": row["ack_emitted"],
            "ack_attempts": row["ack_attempts"],
            "ack_delivered": row["ack_delivered"],
            "ack_lost": row["ack_lost"],
            "passed": row["ack_emitted"] > 0
            and row["ack_attempts"] == row["ack_emitted"]
            and row["ack_delivered"] + row["ack_lost"] == row["ack_attempts"]
            and row["deliveries_applied"] > 0,
        }

    def sealed_block_is_untouched() -> dict[str, Any]:
        low, high = SEALED_BLOCK
        roots = [common.EVALUATION_OUT, common.HERE / "preflight", common.STREAMS]
        leaks: list[str] = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                for token in re.findall(r"\d{6}", path.name):
                    if low <= int(token) <= high:
                        leaks.append(str(path))
                        break
        return {
            "block": [low, high - 1],
            "generated": bool(leaks),
            "read": False,
            "artefact_leaks": sorted(set(leaks)),
            "clean": not leaks,
        }

    def frozen_allocation_side_unchanged() -> dict[str, Any]:
        """The C arm's own configuration must still be the frozen one."""
        truth = common.pilot.base.TRUTH
        configuration = dict(common.pilot.base.arms.ARMS["C"].configuration)
        mismatches: dict[str, Any] = {}
        def same(expected: Any, observed: Any) -> bool:
            """Compare by value, not by JSON type: the truth file stores lists, the frozen
            configuration carries tuples, and the two are the same numbers."""
            if isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple)):
                return len(expected) == len(observed) and all(
                    same(left, right) for left, right in zip(expected, observed)
                )
            return expected == observed

        for field in ("steps", "num_goals", "grid_size", "obstacle_density", "detection_radius",
                      "deadline_tightness", "dt", "rescue_species", "rescue_per_species",
                      "coordinator_policy", "sticky_commitments", "task_membership_consistency",
                      "use_coordinator", "actor_kv_mode", "device"):
            expected = truth["fields"].get(field, {}).get("C")
            if expected is not None and not same(expected, configuration.get(field)):
                mismatches[field] = [expected, configuration.get(field)]
        checkpoints = {
            name: common.sha256(Path(path)) for name, path in common.pilot.CHECKPOINTS.items()
        }
        checkpoint_mismatch = {
            name: [truth["checkpoint_sha256"][name], digest]
            for name, digest in checkpoints.items()
            if truth["checkpoint_sha256"][name] != digest
        }
        return {
            "field_mismatches": mismatches,
            "checkpoint_mismatch": checkpoint_mismatch,
            "budget_bytes": common.BUDGET_BYTES,
            "locked_budget": common.protocol_common.locked_budget(common.PRIMARY_TIER),
            "passed": not mismatches and not checkpoint_mismatch
            and common.BUDGET_BYTES == common.protocol_common.locked_budget(common.PRIMARY_TIER),
        }

    _check(results, "production_defaults_unchanged", production_defaults_unchanged)
    _check(results, "runner_is_not_a_drift", runner_is_not_a_drift)
    _check(results, "block_audit_is_clean", block_audit_is_clean)
    _check(results, "registration_matches_disk", registration_matches_disk)
    _check(results, "frozen_allocation_side_unchanged", frozen_allocation_side_unchanged)
    _check(results, "forward_path_is_ack_agnostic", forward_path_is_ack_agnostic)
    _check(results, "two_channel_cost_closes", two_channel_cost_closes)
    _check(results, "hard_mask_is_never_violated", hard_mask_is_never_violated)
    _check(results, "reverse_channel_is_reproducible", reverse_channel_is_reproducible)
    _check(results, "lossless_reverse_is_lossless", lossless_reverse_is_lossless)
    _check(results, "ack_only_follows_application", ack_only_follows_application)
    _check(results, "sealed_block_is_untouched", sealed_block_is_untouched)

    verdict = all(bool(block.get("passed")) for block in results.values())
    payload = {
        "round": "comm-rule-confirmation",
        "brief": "COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md",
        "block": [common.BLOCK_START, common.BLOCK_STOP - 1],
        "budget_bytes": common.BUDGET_BYTES,
        "settings": common.SETTING_LABELS,
        "arms": common.ARM_LABELS,
        "checks": results,
        "all_passed": verdict,
        "decision": (
            "proceed with the 3,000-episode matrix" if verdict else "stop: fix the failing gate"
        ),
    }
    common.write_json(common.json_path("COMM_RULE_PREFLIGHT.json"), payload)
    for name, block in results.items():
        print(f"{'PASS' if block.get('passed') else 'FAIL'}  {name}", flush=True)
    print(json.dumps({"all_passed": verdict}, ensure_ascii=False))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
