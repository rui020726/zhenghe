"""The release run manifest: what a reproduction needs to know (brief section 6.5).

Nothing here re-runs an experiment. It records, in one machine-readable place, the code
commit, the model checkpoints by digest, the arm switches, every seed manifest by digest,
where the raw artefacts live, and a fold hash for each raw directory so a reader can check
that the data they are looking at is the data the numbers came from -- the raw traces,
streams and logs themselves stay out of Git by design.

    D:/project_env/python.exe runs/release_validation_20260918/run_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.run_signature import TRACE_SCHEMA  # noqa: E402

# Every block this candidate's evidence rests on, with where its raw artefacts live.
BLOCKS = {
    "development_failure_seeds": {
        "seeds": "runs/dispatch_trigger_audit_20260918/CLOSED_LOOP_GATE.json",
        "raw": ["runs/dispatch_trigger_audit_20260918/closedloop"],
        "role": "development",
        "code": "6915d63",
    },
    "development_ablation": {
        "seeds": "runs/coalition_validation_20260918/ablation_seeds.json",
        "raw": ["runs/coalition_validation_20260918/ablation"],
        "role": "development",
        "code": "1f4a598",
    },
    "paired_expansion": {
        "seeds": "runs/dispatch_trigger_audit_20260918/paired/expansion.json",
        "raw": ["runs/dispatch_trigger_audit_20260918/paired"],
        "role": "development",
        "code": "6915d63",
    },
    "independent_replay_400": {
        "seeds": "runs/coalition_validation_20260918/HELDOUT_SEEDS.json",
        "raw": [
            "runs/coalition_validation_20260918/heldout",
            "runs/coalition_validation_20260918/streams",
        ],
        "role": "independent",
        "code": "1f4a598",
    },
    "live_system_loop_60": {
        "seeds": "runs/coalition_validation_20260918/SYSTEM_SEEDS.json",
        "raw": ["runs/coalition_validation_20260918/systemloop"],
        "role": "live-system",
        "code": "1f4a598",
    },
    "coverage_verdict_development_rerun": {
        "seeds": "runs/coalition_validation_20260918/ablation_seeds.json",
        "raw": ["runs/release_validation_20260918/dev"],
        "role": "development",
        "code": "fb1c85b",
    },
    "provenance_scan": {
        "seeds": "runs/coalition_validation_20260918/ablation_seeds.json",
        "raw": ["runs/release_validation_20260918/provenance"],
        "role": "analysis",
        "code": "7e5ba3a",
    },
    "stress_180": {
        "seeds": "runs/release_validation_20260918/STRESS_SEEDS.json",
        "raw": [
            "runs/release_validation_20260918/stress",
            "runs/release_validation_20260918/stress_streams",
        ],
        "role": "stress",
        "code": "fd0dd76",
    },
    "stress_release_safety_scan": {
        "seeds": "runs/release_validation_20260918/STRESS_SEEDS.json",
        "raw": ["runs/release_validation_20260918/stress_safety"],
        "role": "analysis",
        "code": "3fcef63",
    },
}
PILOC = "runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth"
CTDE = "runs/joint_eval_20260916/ctde_final_7_1.pth"
ARMS = {
    "baseline": {
        "coordinator_policy": "coalition_planner",
        "sticky_commitments": True,
        "release_redundant_members": False,
        "suppress_recommand_own_task": False,
    },
    "candidate": {
        "coordinator_policy": "coalition_planner",
        "sticky_commitments": True,
        "release_redundant_members": True,
        "suppress_recommand_own_task": True,
    },
}
SHARED_CONFIGURATION = {
    "steps": 1000,
    "dt": 1.0,
    "rescue_species": 3,
    "rescue_per_species": 2,
    "search_agents": 2,
    "grid_size": 60,
    "obstacle_density": 0.15,
    "detection_radius": 5,
    "residual_scale": 0.25,
    "coordination_interval": 5,
    "allocation_interval": 5,
    "command_ttl": 10.0,
    "comm_delivery_probability": 1.0,
    "comm_delay_ticks": 0,
    "comm_subscribers": "all",
    "actor_kv_mode": "tokens",
    "device": "cpu",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_fold(path: Path) -> dict:
    """A digest over the sorted per-file digests: same files, same hash, any order."""
    if not path.exists():
        return {"path": str(path.relative_to(REPO)).replace("\\", "/"), "present": False}
    files = sorted(item for item in path.rglob("*") if item.is_file() and "__pycache__" not in item.parts)
    digest = hashlib.sha256()
    total = 0
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(item)))
        total += item.stat().st_size
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "present": True,
        "files": len(files),
        "bytes": total,
        "fold_sha256": digest.hexdigest(),
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout.strip()


def main() -> int:
    payload = {
        "candidate_branch": "release/coalition-controller-validation",
        "delivery_head": git("rev-parse", "HEAD"),
        "production_code_identical_to_test_execution_point": "1f4a598",
        "key_commits": {
            "stable_baseline": "868d68a",
            "event_driven_dispatch": "df2255a",
            "development_experiment_source": "6915d63",
            "v1_delivery": "7e5ba3a",
            "coverage_preserving_verdict": "fb1c85b",
            "stress_freeze": "fd0dd76",
        },
        "models": {
            "piloc_checkpoint": {"path": PILOC, "sha256": sha256(REPO / PILOC)},
            "ctde_checkpoint": {"path": CTDE, "sha256": sha256(REPO / CTDE)},
        },
        "arms": ARMS,
        "shared_configuration": SHARED_CONFIGURATION,
        "trace_schema": TRACE_SCHEMA,
        "release_verdict_rule": "coverage-preserving-stage-requirement",
        "seed_manifests": {
            relative: {"sha256": sha256(REPO / relative)}
            for relative in [
                "runs/coalition_validation_20260918/ablation_seeds.json",
                "runs/coalition_validation_20260918/HELDOUT_SEEDS.json",
                "runs/coalition_validation_20260918/SYSTEM_SEEDS.json",
                "runs/release_validation_20260918/STRESS_SEEDS.json",
            ]
        },
        "blocks": {
            name: {
                "role": block["role"],
                "code": block["code"],
                "seed_manifest": {
                    block["seeds"]: sha256(REPO / block["seeds"])
                },
                "raw_artefacts": [directory_fold(REPO / raw) for raw in block["raw"]],
            }
            for name, block in BLOCKS.items()
        },
        "raw_artefacts_in_git": False,
        "retained_untracked_by_policy": ["runs/coordinator_baselines_20260917/dev_summary.json"],
    }
    (HERE / "RELEASE_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    for name, block in payload["blocks"].items():
        for raw in block["raw_artefacts"]:
            print(
                f"{name:<36} {raw['path']:<58} files={raw.get('files', 0):>5} "
                f"bytes={raw.get('bytes', 0):>12} fold={str(raw.get('fold_sha256'))[:12]}"
            )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
