"""Write the round manifest: what ran, on which code, with which locked numbers.

One machine-readable place for the reproducible facts. Two of them are the point of this file.

**Which code actually executed.** The round's commits and the code that ran the episodes are not
the same object: the attribution and the continuation ran against a working tree that adds
send-reason accounting to ``integrated_sar/comm.py`` and two new rule arms to
``comm_policy.py``. So the manifest records both the commit *and* a content hash of every module
the run imports, together with whether each one still matched ``HEAD`` at run time. A bare HEAD
would have said "ab5728b" about episodes that ran modified code.

**Which numbers are still sealed.** The reserved final-test block, the pilot's artefacts, and the
fact that the continuation is a warm start rather than a seamless resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import stability_common as common

import attribution
import train_pilot

#: Modules the executed episodes import. Hashed by content, not by name, because the round's
#: claims rest on what these files said at run time.
EXECUTED_MODULES = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/task03_runtime.py",
    "integrated_sar/allocation_trace.py",
    "integrated_sar/run_signature.py",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(common.ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


def module_fingerprint() -> dict[str, Any]:
    """Content hashes of the executed modules, and whether each still matched HEAD."""
    head = git("rev-parse", "HEAD")
    # Which of the executed modules differ from HEAD's tree right now. ``diff --name-only``
    # answers that question directly; ``status --porcelain`` also encodes the index, whose two
    # status columns make a naive split drop a character of the path.
    dirty = {line for line in git("diff", "--name-only", "HEAD", "--", *EXECUTED_MODULES).splitlines() if line}
    modules: dict[str, Any] = {}
    for relative in EXECUTED_MODULES:
        path = common.ROOT / relative
        content = path.read_bytes()
        try:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=str(common.ROOT),
                capture_output=True,
                check=True,
            ).stdout
        except subprocess.CalledProcessError:
            committed = b""
        modules[relative] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "matches_head": content == committed,
        }
    return {
        "head": head,
        "dirty_modules": sorted(dirty),
        "modules": modules,
        "note": (
            "matches_head compares the file on disk with the file in HEAD's tree. It is false for "
            "the modules this round extended, which is exactly the fact a bare HEAD would hide."
        ),
    }


def executed_configuration(seed: int | None = None) -> dict[str, Any]:
    """The configuration an episode of this round actually runs with.

    Read off the built :class:`RunConfiguration`, not off the dataclass defaults and not off the
    dictionary that asked for it: the pilot's truth table describes the *experiment* (arm C,
    membership fix on, 1000 steps), while the module defaults describe the *library* (heuristic
    coordinator, fix off, 20 steps). Comparing one against the other reports a drift that does
    not exist and hides the one that would matter.
    """
    from integrated_sar.joint_driver import RunConfiguration

    import pilot_preflight

    wanted = set(pilot_preflight.EXPECTED_ALLOCATION) | set(pilot_preflight.EXPECTED_COMMUNICATION_DEFAULTS)
    values = common.configuration_for(
        seed if seed is not None else common.TRAIN_SEEDS[0],
        directory=common.HERE / "preflight",
        label="manifest_preflight",
    )
    built = RunConfiguration(**values).as_dict()
    # Where to write is not a semantic setting, and this round always writes a transcript. The
    # same four fields are excluded from the config hash for the same reason.
    artefact_paths = {"output", "decision_trace", "reservation_log", "comm_transcript"}
    found = {name: built.get(name) for name in sorted(wanted)}
    expected = {**pilot_preflight.EXPECTED_ALLOCATION, **pilot_preflight.EXPECTED_COMMUNICATION_DEFAULTS}
    compared = [key for key in expected if key in found and key not in artefact_paths]
    return {
        "source": "built RunConfiguration for one training seed, arm C",
        "artefact_paths_excluded": sorted(artefact_paths),
        "values": found,
        "expected": {key: expected[key] for key in sorted(expected)},
        "mismatches": {
            key: [expected[key], found[key]] for key in compared if found[key] != expected[key]
        },
        "membership_consistency_enabled": bool(found.get("task_membership_consistency")),
        "allocation_arm_is_C": found.get("coordinator_policy") == "coalition_planner",
    }


def checkpoints() -> dict[str, Any]:
    """Every network this round ran: the pilot's three, the initialisations, and each milestone."""
    import evaluate_pilot

    out: dict[str, Any] = {}
    for training_seed in (1, 2, 3):
        entry: dict[str, Any] = {"pilot": _checkpoint(common.PILOT_CHECKPOINTS[training_seed], training_seed)}
        directory = common.TRAIN_OUT / f"seed{training_seed}"
        summary_path = directory / f"stability_seed{training_seed}_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entry["summary"] = {
                key: summary.get(key)
                for key in (
                    "environment_steps",
                    "episodes",
                    "updates",
                    "optimizer_steps_cumulative",
                    "decision_samples_recorded",
                    "warm_start",
                    "collector",
                )
            }
        for path in sorted(directory.glob(f"stability_seed{training_seed}*.pt")):
            entry.setdefault("continuation", {})[path.stem] = _checkpoint(path, training_seed)
        entry["initialisation"] = {
            "reconstruction_is_deterministic": True,
            "weight_fingerprint": attribution.weight_fingerprint(train_pilot.Controller(training_seed)),
            "note": "rebuilt from the recorded rule (torch.manual_seed then an unfitted MLP); never archived",
        }
        out[f"seed{training_seed}"] = entry
    return out


def _checkpoint(path: Path, training_seed: int) -> dict[str, Any]:
    if not Path(path).exists():
        return {"missing": str(path)}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]
    controller = train_pilot.Controller(training_seed)
    controller.load_state_dict(payload["controller"])
    return {
        "path": str(path.relative_to(common.ROOT)),
        "sha256": common.sha256(path),
        "environment_steps": int(state.get("environment_steps", -1)),
        "episodes": int(state.get("episodes", -1)),
        "update_rounds": len(state.get("updates", [])),
        "validation_points": [int(item.get("at_steps", -1)) for item in state.get("evaluations", [])],
        "rng_state_saved": bool(state.get("rng")),
        "warm_start": bool(state.get("warm_start", False)),
        "weight_fingerprint": attribution.weight_fingerprint(controller),
        "parameters": sum(tensor.numel() for tensor in controller.parameters),
    }


def training_curves() -> dict[str, Any]:
    """Per-seed update statistics, in the units §4 asks for."""
    out: dict[str, Any] = {}
    for training_seed in (1, 2, 3):
        path = common.TRAIN_OUT / f"seed{training_seed}" / f"stability_seed{training_seed}.pt"
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)["state"]
        updates = [item for item in state["updates"] if "optimizer_steps" in item]
        out[f"seed{training_seed}"] = {
            "rounds_recorded": len(updates),
            "optimizer_steps_continuation": sum(int(item["optimizer_steps"]) for item in updates),
            "episodes": int(state["episodes"]),
            "environment_steps": int(state["environment_steps"]),
            "decision_samples": sum(int(item["samples"]) for item in updates),
            "optimizer_steps_cumulative": sum(
                int(item.get("optimizer_steps", item.get("minibatches", 0)))
                for item in state["updates"]
            ),
            "candidates": sum(int(item["candidates"]) for item in updates),
            "candidates_per_decision": round(
                sum(int(item["candidates"]) for item in updates) / max(1, sum(int(item["decisions"]) for item in updates)),
                3,
            ),
            "masked_by_bound_share": round(
                sum(int(item["masked_by_bound"]) for item in updates)
                / max(1, sum(int(item["candidates"]) for item in updates)),
                4,
            ),
            "emissions_by_policy": sum(int(item["emissions_by_policy"]) for item in updates),
            "emissions_forced": sum(int(item["emissions_forced"]) for item in updates),
            "refused_by_policy": sum(int(item["refused_by_policy"]) for item in updates),
            "refused_forced": sum(int(item["refused_forced"]) for item in updates),
            "approx_kl_range": [
                round(min(float(item["approx_kl"]) for item in updates), 6),
                round(max(float(item["approx_kl"]) for item in updates), 6),
            ],
            "clip_fraction_range": [
                round(min(float(item["clip_fraction"]) for item in updates), 4),
                round(max(float(item["clip_fraction"]) for item in updates), 4),
            ],
            "entropy_range": [
                round(min(float(item["entropy"]) for item in updates), 4),
                round(max(float(item["entropy"]) for item in updates), 4),
            ],
            "value_explained_variance_range": [
                round(min(float(item["value_explained_variance"]) for item in updates), 4),
                round(max(float(item["value_explained_variance"]) for item in updates), 4),
            ],
            "gradient_norm_range": [
                round(min(float(item["gradient_norm_mean"]) for item in updates), 4),
                round(max(float(item["gradient_norm_mean"]) for item in updates), 4),
            ],
            "batch_budget_use_range": [
                round(min(float(item["batch_budget_use"]) for item in updates), 4),
                round(max(float(item["batch_budget_use"]) for item in updates), 4),
            ],
            # How often each of the 48 training streams was actually replayed. A round that ran
            # 100 episodes did not see 100 scenarios, and the number has to be stated.
            "training_stream_visits": dict(
                sorted(
                    Counter(seed for item in updates for seed in item["batch_seeds"]).items()
                )
            ),
            "validation_curve": [
                {
                    key: item.get(key)
                    for key in (
                        "at_steps",
                        "mean_on_time",
                        "mean_on_time_rate_per_episode",
                        "mean_airtime_bytes",
                        "autonomous_share_of_bytes",
                        "mean_emissions_forced",
                        "mean_refusals",
                        "inference_milliseconds_per_decision",
                    )
                }
                for item in state["evaluations"]
                if "mean_emissions" in item
            ],
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(common.json_path("COMM_STABILITY_ROUND_MANIFEST.json")))
    args = parser.parse_args(argv)
    attribution_rows = attribution.read_rows(attribution.row_files(common.HERE / "attribution"))
    table = attribution.M.summarise(attribution_rows)
    payload = {
        "round": {
            "purpose": "S3 second round: attribute the pilot's gain, then test whether the training "
            "method produces it stably",
            "brief": "COMMUNICATION_NEXT_ROUND.md (docs branch feat/joint-inference-driver@b5821b6)",
            "charter": "PROJECT_CHARTER.md v1.1 §5.3, §9 stage S3",
            "date": "2026-09-19",
        },
        "branch": {
            "working": git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": git("rev-parse", "HEAD"),
            "based_on": "feat/learned-communication-budget-pilot@ab5728beb124d7d0edd710457b2f84cdd6a5d244",
            "documentation_import": "b5821b63c46819fbf55ce1a3c93535e5251544f2 (two files only, no code)",
        },
        "executed_code": module_fingerprint(),
        "configuration": executed_configuration(),
        "frozen": {
            "allocation_arm": "C (coalition planner, sticky, memfix) -- unchanged from the pilot",
            "budget_bytes": common.BUDGET_BYTES,
            "budget_tier": common.BUDGET_TIER,
            "silence_bound": {
                "max_wait_steps": 10,
                "urgent_wait_steps": 3,
                "urgency_time": 25,
                "shared_by": "every arm, rule and learned alike",
            },
            "feature_layout": train_pilot.FEATURE_LAYOUT_NAME,
            "network": {"feature_dimension": len(train_pilot.FEATURE_NAMES), "hidden": train_pilot.HIDDEN},
            "ppo": {
                "episodes_per_update": train_pilot.EPISODES_PER_UPDATE,
                "epochs": train_pilot.EPOCHS_PER_BATCH,
                "minibatch_decisions": train_pilot.MINIBATCH_STEPS,
                "clip": train_pilot.CLIP,
                "gae_lambda": train_pilot.GAE_LAMBDA,
                "gamma": train_pilot.GAMMA,
                "entropy_bonus": train_pilot.ENTROPY_BONUS,
                "learning_rate": train_pilot.LEARNING_RATE,
                "exploration_epsilon": train_pilot.EXPLORATION_EPSILON,
            },
        },
        "data": {
            "training_seeds": list(common.TRAIN_SEEDS),
            "validation_seeds": list(common.VALIDATION_SEEDS),
            "stream_sha256": {
                str(seed): common.sha256(common.pilot.stream_path(seed))
                for seed in common.VALIDATION_SEEDS[:3]
            },
            "streams_note": "all 72 hashes are in the pilot round's manifest; three here prove the reader works",
            "final_test": {
                "previous_registration": "500000..500399 -- 400/400 executed in corrected_env_final_20260918, "
                "so no longer available as an unseen test",
                "reserved_now": "800000..800399 -- registered only, never generated or evaluated",
                "audit": "COMM_DATA_SPLIT_AUDIT.json",
            },
        },
        "arms": {
            "count": len(table),
            "episodes": len(attribution_rows),
            "per_arm": {arm: {key: value for key, value in row.items() if key != "by_reason"} for arm, row in table.items()},
            "reason_split": {arm: row["by_reason"] for arm, row in table.items()},
        },
        "training": training_curves(),
        "checkpoints": checkpoints(),
        "measurements": {
            "attribution": "COMM_POLICY_ATTRIBUTION.json",
            "collector_correction": "COMM_COLLECTOR_CORRECTION.json",
            "data_split": "COMM_DATA_SPLIT_AUDIT.json",
            "throughput": "COMM_TRAINING_THROUGHPUT.json",
            "stability": "COMM_TRAINING_STABILITY_RESULTS.json",
            "smoke": "STABILITY_SMOKE.json",
        },
        "not_committed": {
            "checkpoints": "*.pt under train/ -- resumable state stays on disk, three seeds",
            "raw_logs": "rp_/tr_/ct_ JSONL per episode",
            "streams": "s<seed>.json, recorded by the pilot round",
            "rows": "rows_attribution*.jsonl -- the per-episode detail behind every table in this round",
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"written: {args.out} ({len(text.splitlines())} lines)")
    print(
        json.dumps(
            {
                "head": payload["branch"]["head"],
                "dirty_modules": payload["executed_code"]["dirty_modules"],
                "membership_enabled": payload["configuration"]["membership_consistency_enabled"],
                "defaults_match": not payload["configuration"]["mismatches"],
                "arms": payload["arms"]["count"],
                "episodes": payload["arms"]["episodes"],
                "training": {seed: block["environment_steps"] for seed, block in payload["training"].items()},
                "optimizer_steps_cumulative": {
                    seed: block["optimizer_steps_cumulative"] for seed, block in payload["training"].items()
                },
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
