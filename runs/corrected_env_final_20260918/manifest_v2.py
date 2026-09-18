"""The run manifest for the corrected-environment block.

Everything a reader needs to decide whether the numbers mean what they claim, and to find the
bytes behind them: the commit the production files are frozen at, the branch, the environment
variant, both checkpoints by digest, the two arms' configuration, the seed manifest, the
statistics rule, the thread settings, the trace schema, and a digest over every raw artefact
the block produced (the raw files stay out of git; their location and content are what is
recorded here).

    D:/project_env/python.exe runs/corrected_env_final_20260918/manifest_v2.py
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
sys.path.insert(0, str(HERE))

import arms  # noqa: E402

RAW_DIRECTORIES = ("streams", "block", "block_smoke", "system_block", "ghost_recheck", "records")
SUMMARY_FILES = (
    "HELDOUT_SEEDS_V2.json",
    "ARMS_CONFIG_TRUTH.json",
    "FREEZE.json",
    "GHOST_RECHECK_dev.json",
    "GHOST_RECHECK_stress.json",
    "METRIC_ACCOUNTING.json",
    "CORRECTED_ENV_HELDOUT_STATS.json",
    "BLOCK_SUMMARY.json",
    "SYSTEM_SUMMARY.json",
    "SMOKE_STATS.json",
    "MEMBERSHIP_FINAL_CHECK.md",
    "RELEASE_EXECUTION_CHECK.md",
    "CORRECTED_ENV_HELDOUT_PROTOCOL.md",
    "CORRECTED_ENV_HELDOUT_VERDICT.md",
    "CORRECTED_ENV_SYSTEM_VERDICT.md",
)
SCRIPTS = (
    "arms.py",
    "corrected_block.py",
    "freeze.py",
    "ghost_recheck.py",
    "metric_accounting.py",
    "release_event_20083.py",
    "seeds_v2.py",
    "stats_v2.py",
    "system_v2.py",
)
PRODUCTION = ("integrated_sar", "task_allocation")
EXECUTION_ONLY = ("integrated_sar", "task_allocation", ":(exclude)integrated_sar/tests")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()


def fold(directories: tuple[str, ...]) -> dict:
    """One digest per directory: file count, bytes, and a digest over sorted name+content pairs."""
    out = {}
    for name in directories:
        root = HERE / name
        if not root.exists():
            out[name] = {"present": False}
            continue
        entries = []
        total = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix in {".part", ".pyc"}:
                continue
            size = path.stat().st_size
            total += size
            entries.append(f"{path.relative_to(root).as_posix()}:{size}:{sha256(path)}")
        digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
        out[name] = {
            "present": True,
            "files": len(entries),
            "bytes": total,
            "fold_sha256": digest,
            "location": str(root.relative_to(REPO)),
        }
    return out


def main() -> int:
    production_digest = hashlib.sha256(
        "\n".join(
            f"{path.relative_to(REPO / area).as_posix()}:{sha256(path)}"
            for area in PRODUCTION
            for path in sorted((REPO / area).rglob("*.py"))
            if "__pycache__" not in path.parts
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "block": "corrected-environment independent C vs D, 2026-09-18",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "manifest_written_at_head": git("rev-parse", "HEAD"),
        "frozen_code_commit": arms.freeze_commit(),
        "production_tree_matches_freeze": not git(
            "diff", arms.freeze_commit(), "--", *EXECUTION_ONLY
        ),
        "production_python_digest": production_digest,
        "environment_variant": {
            "module": "task_allocation/old_task_env.py",
            "membership_consistency": True,
            "arm_suffix": "+memfix",
        },
        "checkpoints": {
            "piloc": {"path": arms.FROZEN["piloc_checkpoint"], "sha256": arms.sha256(arms.PIL_CHECKPOINT)},
            "ctde": {"path": arms.FROZEN["ctde_checkpoint"], "sha256": arms.sha256(arms.CTDE_CHECKPOINT)},
        },
        "threads": {
            **arms.FROZEN["thread_environment"],
            "torch_set_num_threads": 1,
            "applied_before_torch_import_in": list(SCRIPTS),
        },
        "trace_schema": arms.FROZEN["trace_schema"],
        "arms": arms.truth_table()["construction_expectations"],
        "arm_configurations": {
            label: {
                key: value
                for key, value in arms.ARMS[label].configuration.items()
                if key not in {"output", "decision_trace", "replay_stream", "seed"}
            }
            for label in arms.ARMS
        },
        "seeds": json.loads((HERE / "HELDOUT_SEEDS_V2.json").read_text(encoding="utf-8")),
        "statistics": json.loads(
            (HERE / "CORRECTED_ENV_HELDOUT_STATS.json").read_text(encoding="utf-8")
        ) if (HERE / "CORRECTED_ENV_HELDOUT_STATS.json").exists() else None,
        "raw_artefacts": fold(RAW_DIRECTORIES),
        "tracked_summaries": {
            name: ({"sha256": sha256(HERE / name), "bytes": (HERE / name).stat().st_size}
                   if (HERE / name).exists() else {"present": False})
            for name in SUMMARY_FILES
        },
        "scripts": {name: sha256(HERE / name) for name in SCRIPTS if (HERE / name).exists()},
        "not_in_git_but_locatable": {
            "streams": "runs/corrected_env_final_20260918/streams/s<seed>.json",
            "lifecycle_logs": "runs/corrected_env_final_20260918/block/<arm>/rp_..._s<seed>.jsonl",
            "decision_traces": "runs/corrected_env_final_20260918/block/<arm>/tr_..._s<seed>.jsonl",
            "reproduce_with": "python runs/corrected_env_final_20260918/corrected_block.py --replay <arm> --worker i --workers n",
        },
        "protected_untracked": [
            "runs/coordinator_baselines_20260917/dev_summary.json",
        ],
    }
    destination = HERE / "RELEASE_MANIFEST_V2.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    import os

    os.replace(part, destination)
    print(
        f"manifest written: freeze {payload['frozen_code_commit'][:12]}, "
        f"tree matches={payload['production_tree_matches_freeze']}, "
        f"raw dirs={ {k: v.get('files') for k, v in payload['raw_artefacts'].items()} }"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
