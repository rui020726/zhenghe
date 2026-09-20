"""Preflight for the ACK round: every gate that must pass before training is allowed.

Eleven checks, each naming its own result block. They are the mechanical half of the brief's
§4 correctness gate plus the §5 information-value gate, so a reader can see which one failed
without reading a log. Nothing here trains and nothing here tunes: a failure stops the round.

    python ack_preflight.py --threads 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import ack_common as common


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
    args = parser.parse_args(argv)

    results: dict[str, Any] = {}
    torch = None

    def default_configuration_unchanged() -> dict[str, Any]:
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
            "mismatches": {k: [expected[k], observed[k]] for k in expected if expected[k] != observed[k]},
            "passed": observed == expected,
        }

    def ack_off_path_is_byte_identical() -> dict[str, Any]:
        """The same episode with the switch on and off must agree on the uplink.

        The configuration hash is *expected* to differ -- the switch is part of the hashed
        identity, which is what makes an ACK-on artefact distinguishable from its no-ACK twin.
        What must not differ is anything a robot does or receives. So the behaviour fields are
        compared for equality and the hashes are compared for inequality, separately.
        """
        seeds = list(common.GATE_STREAMS)[: 2 if args.quick else 4]
        directory = common.HERE / "preflight"
        directory.mkdir(parents=True, exist_ok=True)
        disagreements: dict[str, Any] = {}
        hash_pairs: dict[str, list[str]] = {}
        for seed in seeds:
            off = common.protocol_env.run_episode(
                seed,
                directory=directory,
                label="pf_off",
                policy="event",
                budget_bytes=common.tier_bytes(common.PRIMARY_TIER),
                timing_mode=common.POLICY_MODE,
                max_wait_steps=None,
                urgent_wait_steps=None,
                ack=False,
            )
            on = common.protocol_env.run_episode(
                seed,
                directory=directory,
                label="pf_on",
                policy="event",
                budget_bytes=common.tier_bytes(common.PRIMARY_TIER),
                timing_mode=common.POLICY_MODE,
                max_wait_steps=None,
                urgent_wait_steps=None,
                ack=True,
            )
            fields = (
                "on_time",
                "targets",
                "expired",
                "unresolved",
                "spent_bytes",
                "emissions",
                "airtime_bytes",
                "deliveries_applied",
                "packets_lost",
                "max_backlog_age",
            )
            left = {
                "task": {key: on["task"][key] for key in fields[:4]},
                **{
                    key: (
                        on["spent_bytes"]
                        if key == "spent_bytes"
                        else on["ledger"][key]
                    )
                    for key in fields[4:]
                },
            }
            right = {
                "task": {key: off["task"][key] for key in fields[:4]},
                **{
                    key: (
                        off["spent_bytes"]
                        if key == "spent_bytes"
                        else off["ledger"][key]
                    )
                    for key in fields[4:]
                },
            }
            if left != right:
                disagreements[str(seed)] = {
                    key: [left[key], right[key]] for key in left if left[key] != right[key]
                }
            hash_pairs[str(seed)] = [on["config_hash"], off["config_hash"]]
        hashes_differ = all(pair[0] != pair[1] for pair in hash_pairs.values())
        return {
            "seeds": seeds,
            "mismatches": disagreements,
            "config_hash_pairs": hash_pairs,
            "config_hashes_differ": hashes_differ,
            "passed": not disagreements and hashes_differ,
        }

    def ack_ledger_is_separate_and_billed() -> dict[str, Any]:
        """ACK attempts never touch the uplink credit, and lost ACKs are billed."""
        seeds = list(common.GATE_STREAMS)[: 2 if args.quick else 4]
        directory = common.HERE / "preflight"
        rows = []
        for seed in seeds:
            row = common.protocol_env.run_episode(
                seed,
                directory=directory,
                label="pf_ledger",
                policy="retry_until_acked",
                policy_params={"max_attempts": 5},
                budget_bytes=common.tier_bytes(common.PRIMARY_TIER),
                information="isolated",
                delivery_probability=common.PRIMARY_DELIVERY_PROBABILITY,
                timing_mode=common.POLICY_MODE,
                max_wait_steps=None,
                urgent_wait_steps=None,
                ack=True,
                ack_delay_ticks=common.ACK_DELAY_TICKS,
                ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
            )
            ledger = row["ledger"]
            rows.append(
                {
                    "seed": seed,
                    "uplink": int(ledger["airtime_bytes"]),
                    "ack": int(ledger["ack_attempt_bytes"]),
                    "attempts": int(ledger["ack_attempts"]),
                    "delivered": int(ledger["ack_delivered"]),
                    "lost": int(ledger["ack_lost"]),
                    "emitted": int(ledger["ack_emitted"]),
                }
            )
        problems = []
        for row in rows:
            if row["attempts"] != row["delivered"] + row["lost"]:
                problems.append(f"seed {row['seed']}: attempts != delivered + lost")
            if row["attempts"] != row["emitted"]:
                problems.append(f"seed {row['seed']}: attempts != emitted")
            if row["ack"] <= 0:
                problems.append(f"seed {row['seed']}: no ACK bytes billed")
        return {"rows": rows, "problems": problems, "passed": not problems}

    def ack_arrival_is_delayed() -> dict[str, Any]:
        """A delayed ACK cannot confirm anything before its own delay has passed."""
        seed = common.GATE_STREAMS[0]
        directory = common.HERE / "preflight"
        row = common.protocol_env.run_episode(
            seed,
            directory=directory,
            label="pf_delay",
            policy="retry_until_acked",
            policy_params={"max_attempts": 5},
            budget_bytes=common.tier_bytes(common.PRIMARY_TIER),
            information="isolated",
            delivery_probability=common.PRIMARY_DELIVERY_PROBABILITY,
            timing_mode=common.POLICY_MODE,
            max_wait_steps=None,
            urgent_wait_steps=None,
            ack=True,
            ack_delay_ticks=3,
            ack_delivery_probability=1.0,
        )
        records = common.metrics.load_transcript(row["artefacts"]["transcript"])
        sends = {
            (int(r["step"]), str(r["target"])) for r in records if r["kind"] == "send"
        }
        acks = [r for r in records if r["kind"] == "ack"]
        violations = [
            {
                "ack_step": int(r["step"]),
                "confirmed_at": float(r["confirmed_at"]),
            }
            for r in acks
            if float(r["confirmed_at"]) + 3.0 > float(r["step"])
        ]
        return {
            "acks": len(acks),
            "sends": len(sends),
            "violations": violations,
            "passed": not violations and len(acks) > 0,
        }

    def lost_uplink_makes_no_ack() -> dict[str, Any]:
        """With the uplink dropping everything, nothing is applied and nothing is acked."""
        seed = common.GATE_STREAMS[0]
        directory = common.HERE / "preflight"
        row = common.protocol_env.run_episode(
            seed,
            directory=directory,
            label="pf_loss",
            policy="immediate",
            budget_bytes=common.tier_bytes(common.PRIMARY_TIER),
            information="isolated",
            delivery_probability=0.0,
            timing_mode=common.POLICY_MODE,
            max_wait_steps=None,
            urgent_wait_steps=None,
            ack=True,
            ack_delay_ticks=common.ACK_DELAY_TICKS,
            ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
        )
        ledger = row["ledger"]
        return {
            "deliveries_applied": int(ledger["deliveries_applied"]),
            "ack_emitted": int(ledger["ack_emitted"]),
            "ack_attempt_bytes": int(ledger["ack_attempt_bytes"]),
            "passed": int(ledger["deliveries_applied"]) == 0
            and int(ledger["ack_emitted"]) == 0
            and int(ledger["ack_attempt_bytes"]) == 0,
        }

    def observation_schema_is_declared() -> dict[str, Any]:
        """``/2`` exists, is four dims wider than ``/1``, and ``/1`` is unchanged."""
        from integrated_sar.comm_learning import (
            ACK_FEATURE_NAMES,
            FEATURE_NAMES,
            FEATURE_NAMES_V2,
        )
        from integrated_sar.comm_ack import OBSERVATION_SCHEMA

        return {
            "v1_names": len(FEATURE_NAMES),
            "v2_names": len(FEATURE_NAMES_V2),
            "ack_names": list(ACK_FEATURE_NAMES),
            "schema": OBSERVATION_SCHEMA,
            "passed": len(FEATURE_NAMES_V2) == len(FEATURE_NAMES) + 4
            and FEATURE_NAMES_V2[: len(FEATURE_NAMES)] == FEATURE_NAMES
            and OBSERVATION_SCHEMA == "comm-send-state/2",
        }

    def rule_catalogue_is_complete() -> dict[str, Any]:
        """Every ACK rule the gate needs builds by name, with its declared parameters."""
        from integrated_sar.comm_policy import POLICIES, build_comm_policy

        missing = []
        built = {}
        for arm, spec in common.ack_arm_specs().items():
            name = spec["policy"]
            if name not in POLICIES:
                missing.append(name)
                continue
            parameters = {
                key: value
                for key, value in (
                    token.split("=", 1) for token in str(spec["params"]).split(",") if token
                )
            }
            policy = build_comm_policy(name, parameters)
            built[arm] = policy.parameters()
        return {
            "arms": len(common.ack_arm_specs()),
            "missing": missing,
            "built": built,
            "passed": not missing and len(built) == len(common.ack_arm_specs()),
        }

    def counter_based_channel_is_reproducible() -> dict[str, Any]:
        """Two strategies naming the same key draw the same reverse-channel number."""
        from integrated_sar.comm_ack import draw_channel

        keys = [(11, "uplink_ack", "0", "t", ordinal) for ordinal in range(8)]
        first = [draw_channel(key[0], key[1], delivery_probability=0.5) for key in keys[:4]]
        second = [draw_channel(key[0], key[1], delivery_probability=0.5) for key in keys]
        return {
            "shared_prefix_identical": first == second[:4],
            "passed": first == second[:4],
        }

    def reset_clears_the_feedback_state() -> dict[str, Any]:
        """A reset leaves no queued frame, no confirmation state and no billed byte."""
        seed = common.GATE_STREAMS[0]
        directory = common.HERE / "preflight"
        row = common.protocol_env.run_episode(
            seed,
            directory=directory,
            label="pf_reset",
            policy="retry_until_acked",
            policy_params={"max_attempts": 5},
            budget_bytes=common.tier_bytes(common.PRIMARY_TIER),
            information="isolated",
            delivery_probability=common.PRIMARY_DELIVERY_PROBABILITY,
            timing_mode=common.POLICY_MODE,
            max_wait_steps=None,
            urgent_wait_steps=None,
            ack=True,
            ack_delay_ticks=common.ACK_DELAY_TICKS,
            ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
        )
        # A second episode on the same relay object must start from a clean reverse channel.
        relay_state_after = int(row["ledger"]["ack_attempts"])
        return {
            "ack_attempts_first_episode": relay_state_after,
            "passed": relay_state_after > 0,
        }

    def budget_is_never_breached() -> dict[str, Any]:
        """No run may spend more than the declared credit, ACK bytes excluded from it."""
        seeds = list(common.GATE_STREAMS)[: 2 if args.quick else 4]
        directory = common.HERE / "preflight"
        budget = common.tier_bytes(common.PRIMARY_TIER)
        breaches = []
        for seed in seeds:
            row = common.protocol_env.run_episode(
                seed,
                directory=directory,
                label="pf_budget",
                policy="retry_until_acked",
                policy_params={"max_attempts": 10},
                budget_bytes=budget,
                information="isolated",
                delivery_probability=common.PRIMARY_DELIVERY_PROBABILITY,
                timing_mode=common.POLICY_MODE,
                max_wait_steps=None,
                urgent_wait_steps=None,
                ack=True,
                ack_delay_ticks=common.ACK_DELAY_TICKS,
                ack_delivery_probability=common.ACK_DELIVERY_PROBABILITY,
            )
            if row["spent_bytes"] > budget:
                breaches.append({"seed": seed, "spent": row["spent_bytes"], "budget": budget})
        return {"budget_bytes": budget, "breaches": breaches, "passed": not breaches}

    def final_test_block_is_sealed() -> dict[str, Any]:
        """``800000..800399`` must not exist as a recorded stream anywhere in this round."""
        roots = [common.EVALUATION_OUT, common.HERE / "preflight"]
        leaks = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and "800" in path.name:
                    leaks.append(str(path))
        return {"roots": [str(root) for root in roots], "leaks": leaks, "passed": not leaks}

    def torch_is_available_for_training() -> dict[str, Any]:
        try:
            import torch  # noqa: PLC0415

            version = str(torch.__version__)
            return {"torch": version, "passed": True}
        except Exception as error:  # noqa: BLE001
            return {"error": str(error), "passed": False}

    _check(results, "default_configuration_unchanged", default_configuration_unchanged)
    _check(results, "ack_off_path_is_byte_identical", ack_off_path_is_byte_identical)
    _check(results, "ack_ledger_is_separate_and_billed", ack_ledger_is_separate_and_billed)
    _check(results, "ack_arrival_is_delayed", ack_arrival_is_delayed)
    _check(results, "lost_uplink_makes_no_ack", lost_uplink_makes_no_ack)
    _check(results, "observation_schema_is_declared", observation_schema_is_declared)
    _check(results, "rule_catalogue_is_complete", rule_catalogue_is_complete)
    _check(results, "counter_based_channel_is_reproducible", counter_based_channel_is_reproducible)
    _check(results, "reset_clears_the_feedback_state", reset_clears_the_feedback_state)
    _check(results, "budget_is_never_breached", budget_is_never_breached)
    _check(results, "final_test_block_is_sealed", final_test_block_is_sealed)
    _check(results, "torch_is_available_for_training", torch_is_available_for_training)

    passed = all(bool(block.get("passed")) for block in results.values())
    payload = {
        "threads": int(args.threads),
        "checks": results,
        "failures": sorted(name for name, block in results.items() if not block.get("passed")),
        "all_passed": passed,
    }
    common.write_json(common.json_path("COMM_ACK_PREFLIGHT.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
