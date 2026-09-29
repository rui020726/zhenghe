"""ROUND27 代码身份登记：启动时身份可复现、事后编辑分开记账（0 启动）。

The five cells all recorded one launch-time identity (``5b5ca69b…``). It is reproducible: hash
the eight launch-time modules as LF bytes -- which is exactly what the working tree holds, the
Write tool never wrote CRLF here -- taken from the harness commit ``12a2a87``. The only file that
changed afterwards is ``r27_compare.py`` (a zero-launch refinement of the prefix report that now
names both sides of the first transcript divergence); it is analysis code, it never enters an
episode, and its provenance is this round's later commits. The four post-processing modules are
registered the same way.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import r27_common as common

LAUNCH_TIME_MODULES = (
    "r27_common.py", "r27_intervention.py", "r27_run.py", "r27_freeze.py",
    "r27_preflight.py", "r27_gate.py", "r27_compare.py", "r27_target5_readback.py",
)
HARNESS_COMMIT = "12a2a87"
POST_PROCESSING_MODULES = (
    "r27_preflight_report.py", "r27_timelines.py", "r27_manifest.py", "r27_verdict.py",
    "r27_code_identity.py",
)


def _blob(commit: str, name: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(common.ROOT), "show",
         f"{commit}:runs/{common.OUTPUT_DIR.name}/{name}"],
        capture_output=True, check=True)
    return result.stdout


def _reconstruct(commit: str) -> tuple[str, dict[str, str]]:
    harness = {name: hashlib.sha256(_blob(commit, name)).hexdigest() for name in LAUNCH_TIME_MODULES}
    identity = common.code_identity()
    payload = {key: value for key, value in identity.items()
               if key not in ("harness", "freeze_sha256")}
    payload["harness"] = harness
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                       sort_keys=True).encode("utf-8")).hexdigest()
    return digest, harness


def build() -> dict:
    identity = common.code_identity()
    recorded = sorted({common.read_json(common.row_path(name, 1))["code_identity_freeze_sha256"]
                       for name in common.PATH_ORDER})
    digest, harness = _reconstruct(HARNESS_COMMIT)
    return {
        "round": "27",
        "run_id": common.OUTPUT_DIR.name,
        "launch_time": {
            "modules": list(LAUNCH_TIME_MODULES),
            "recorded_in_rows": recorded,
            "single_identity_across_all_five_cells": len(recorded) == 1,
            "reproduced_from_commit": HARNESS_COMMIT,
            "reproduction_rule": "sha256 of each module's LF blob at that commit (the working tree "
                                 "holds LF bytes here), hashed into the same identity payload",
            "recomputed_identity": digest,
            "reproduction_verified": len(recorded) == 1 and recorded[0] == digest,
            "module_sha256_at_launch": harness,
        },
        "post_launch_edits": [
            {"module": "r27_compare.py",
             "change": "the prefix-consistency record now names both sides of the first "
                       "transcript divergence and the identical-prefix length",
             "launches_after": 0,
             "note": "analysis-only; the five rows are untouched and keep the launch-time identity",
             "current_sha256": identity["harness"]["r27_compare.py"]}
        ],
        "post_processing_modules": {
            "modules": list(POST_PROCESSING_MODULES),
            "module_sha256": {name: identity["harness"].get(name) or common.sha256_file(
                common.OUTPUT_DIR / name)
                for name in POST_PROCESSING_MODULES if (common.OUTPUT_DIR / name).exists()},
            "note": "written after the last launch; they read artefacts and construct no driver",
        },
        "shared_sections": {
            key: identity[key] for key in ("production", "imported_round_modules", "checkpoints",
                                           "learned_checkpoint")
        },
        "current_full_identity": {
            "freeze_sha256": identity["freeze_sha256"],
            "note": "the identity a fresh run would record now; NOT the launch-time identity, "
                    "because r27_compare.py changed after the last launch",
        },
    }


if __name__ == "__main__":
    payload = build()
    common.write_json("CODE_IDENTITY.json", payload)
    print("launch-time identity verified:", payload["launch_time"]["reproduction_verified"],
          payload["launch_time"]["recorded_in_rows"][0][:16])
    print("current identity:", payload["current_full_identity"]["freeze_sha256"][:16])
