"""Write the round manifest: what was run, against what code, with which parameters.

The point is reproducibility without reading prose: a later reader can check the branch, the
frozen commit the allocation side came from, the checkpoint hashes, the condition table as it
stood before any run, the seed provenance, and the gate results, from one JSON file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import comm_experiment
import common
import conditions

from integrated_sar.comm_policy import policy_descriptors  # noqa: E402

HERE = common.HERE


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(common.ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(directory: Path, pattern: str) -> dict[str, Any]:
    paths = sorted(Path(directory).rglob(pattern))
    return {"count": len(paths), "bytes": sum(path.stat().st_size for path in paths)}


def main(argv: list[str]) -> int:
    rows = comm_experiment.read_rows(HERE / "rows")
    clean = [row for row in rows if not row.get("technical_failure")]
    stats_path = HERE / "COMMUNICATION_BASELINE_STATS.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    identity_path = HERE / "IDENTITY_CHECK.json"
    identity = (
        json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
    )
    probe_path = HERE / "TRAFFIC_PROBE.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
    production = git("status", "--porcelain", "integrated_sar", "task_allocation")
    manifest = {
        "round": "communication_control_baselines_20260919",
        "date": "2026-09-19",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "inherited_from": {
            "branch": "fix/task-membership-consistency",
            "head": "fd06d45abab83e595c907e3a5d6b0d3d28554117",
            "frozen_production_commit": common.FROZEN["code_commit"],
            "production_paths_dirty_here": bool(production),
            "note": (
                "the allocation side is the frozen block's C arm; this branch adds a send-policy "
                "seam to the relay, so the previous block's freeze check reports drift by design -- "
                "the equivalence of the default path is the identity check below, not the freeze file"
            ),
        },
        "checkpoints": {
            name: {
                "path": str(Path(path).relative_to(common.ROOT)).replace(chr(92), "/"),
                "sha256": digest,
            }
            for (name, path), digest in zip(common.CHECKPOINTS.items(), common.CHECKPOINT_SHA256.values())
        },
        "scenario": {
            key: value
            for key, value in common.FROZEN.items()
            if key
            in {
                "steps",
                "dt",
                "grid_size",
                "obstacle_density",
                "detection_radius",
                "search_agents",
                "num_goals",
                "deadline_tightness",
                "residual_scale",
                "allocation_interval",
                "coordination_interval",
                "command_ttl",
                "trace_schema",
            }
        },
        "default_configuration_unchanged": {
            "comm_policy": "immediate",
            "comm_policy_params": "",
            "comm_sends_per_tick": None,
            "comm_transcript": None,
            "comm_range": 0.3,
            "gossip_interval": 2,
            "comm_delivery_probability": 1.0,
            "comm_delay_ticks": 0,
            "comm_subscribers": "all",
        },
        "policy_menu": list(policy_descriptors()),
        "conditions": [condition.as_dict() for condition in conditions.CONDITIONS],
        "seeds": {
            "development_used_for_baselines": list(common.DEV_SEEDS[:12]),
            "development_stream_dir": str(common.DEV_STREAM_DIR.relative_to(common.ROOT)).replace(chr(92), "/"),
            "untouched_registered_independent_test": "500000..500399 (previous round)",
            "live_system_seeds_not_used": "600000..600059",
        },
        "episodes": {
            "cells": len(rows),
            "conditions": len({row["condition"] for row in rows}),
            "seeds": len({int(row["seed"]) for row in rows}),
            "technical_failures": len(rows) - len(clean),
            "targets_discovered_total": int(
                sum(row["performance"]["targets"] for row in clean if row["condition"] == "isolated_immediate")
            ),
        },
        "gates": {
            "identity_vs_frozen_commit": {
                "seeds": identity.get("seeds"),
                "all_identical": identity.get("all_identical"),
                "reference_commit": identity.get("frozen_reference_commit"),
                "records_per_seed": [
                    row["old_records"] for row in identity.get("rows", [])
                ],
            },
            "bootstrap": stats.get("bootstrap"),
            "zero_detection_seeds": stats.get("zero_detection_seeds"),
            "per_condition_headline": {
                name: {
                    "on_time_total": values["metrics"]
                    .get("performance.on_time", {})
                    .get("total"),
                    "packets_total": values["metrics"]
                    .get("transport.packets_sent", {})
                    .get("total"),
                    "payload_bytes_total": values["metrics"]
                    .get("transport.payload_bytes_total", {})
                    .get("total"),
                }
                for name, values in sorted((stats.get("per_condition") or {}).items())
            },
            "headroom": stats.get("headroom"),
            "spread": stats.get("spread"),
        },
        "artefacts": {
            "rows": inventory(HERE / "rows", "rows_shard*.jsonl"),
            "transcripts": inventory(HERE / "rows", "ct_C_*.jsonl"),
            "probe_lifecycle_logs_kept_for_drilldown": inventory(HERE / "probe", "rp_C_*.jsonl"),
            "documents": sorted(
                str(path.name) for path in HERE.glob("*.md")
            ),
            "statistics": stats_path.name if stats_path.exists() else None,
            "raw_jsonl_and_streams_not_committed": True,
        },
        "hashes": {
            name: sha256(HERE / name)
            for name in (
                "COMMUNICATION_BASELINE_STATS.json",
                "TRAFFIC_PROBE.json",
                "IDENTITY_CHECK.json",
            )
            if (HERE / name).exists()
        },
    }
    path = common.write_json(HERE / "COMM_ROUND_MANIFEST.json", manifest)
    print(f"wrote {path}")
    print(
        f"episodes={manifest['episodes']['cells']} technical_failures={manifest['episodes']['technical_failures']} "
        f"identity_all_identical={manifest['gates']['identity_vs_frozen_commit']['all_identical']}"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
