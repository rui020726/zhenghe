"""ROUND28 代码身份登记（0 启动）：113 格共享单一启动身份及其复现路径。

The 113 cells recorded one launch-time identity (``d5b3cee7…``). It is reproducible from the
harness commit ``0db4ef2`` with one caveat: ``r28_common.py`` sat in the working tree as CRLF
bytes at launch time (a Python text-mode edit had written them that way), while the commit blob
stores LF. Hashing the blob converted to CRLF -- every other module as it stands -- reproduces the
recorded value exactly. This is the same eol caveat ROUND25 registered; it changes no code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import r28_common as common

HARNESS_COMMIT = "0db4ef2"
#: The six modules the 113 cells ran with (the launch-time set; the analysis modules came later).
LAUNCH_MODULES = ("r28_common.py", "r28_freeze.py", "r28_preflight.py", "r28_run.py",
                  "r28_select.py", "r28_verify.py")
#: The two modules whose launch-time working tree held CRLF bytes (both had been touched by a
#: Python text-mode edit before the first launch); the commit blobs store LF.
CRLF_AT_LAUNCH = ("r28_common.py", "r28_preflight.py")
POST_PROCESSING_MODULES = (
    "r28_preflight_report.py", "r28_analyse.py", "r28_manifest.py", "r28_verdict.py",
    "r28_code_identity.py",
)


def _blob(commit: str, name: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(common.ROOT), "show",
         f"{commit}:runs/{common.OUTPUT_DIR.name}/{name}"],
        capture_output=True, check=True).stdout


def _crlf(blob: bytes) -> bytes:
    return blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def build() -> dict:
    identity = common.code_identity()
    recorded = _recorded_identities()
    harness = {name: hashlib.sha256(
        _crlf(_blob(HARNESS_COMMIT, name)) if name in CRLF_AT_LAUNCH
        else _blob(HARNESS_COMMIT, name)).hexdigest()
        for name in LAUNCH_MODULES}
    payload = {key: value for key, value in identity.items()
               if key not in ("harness", "freeze_sha256")}
    payload["harness"] = harness
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                       sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "round": "28",
        "run_id": common.OUTPUT_DIR.name,
        "launch_time": {
            "cells": len(_all_cells()),
            "modules": list(LAUNCH_MODULES),
            "recorded_in_rows": recorded,
            "single_identity_across_all_cells": len(recorded) == 1,
            "reproduced_from_commit": HARNESS_COMMIT,
            "reproduction_rule": "hash the commit blob of each launch-time module, converting "
                                 "r28_common.py and r28_preflight.py from LF to CRLF (the "
                                 "working-tree bytes the harness actually ran with; both had "
                                 "been touched by a Python text-mode edit before launch); every "
                                 "other module uses its blob as stored",
            "recomputed_identity": digest,
            "reproduction_verified": len(recorded) == 1 and recorded[0] == digest,
            "post_launch_edits": [
                {"module": "r28_verify.py",
                 "change": "verify_sample gained its optional frozen-point argument so the "
                           "analysis could check opportunity reproduction",
                 "launches_after": 0,
                 "note": "analysis-only; the 113 rows are untouched and keep the launch-time "
                         "identity"},
            ],
        },
        "post_processing_modules": {
            "modules": list(POST_PROCESSING_MODULES),
            "module_sha256": {name: common.sha256_file(common.OUTPUT_DIR / name)
                              for name in POST_PROCESSING_MODULES
                              if (common.OUTPUT_DIR / name).exists()},
            "note": "written after the last launch; they read artefacts and construct no driver",
        },
        "shared_sections": {key: identity[key] for key in
                            ("production", "imported_round_modules", "checkpoints",
                             "learned_checkpoint")},
        "current_working_tree_identity": {
            "freeze_sha256": identity["freeze_sha256"],
            "note": "what a fresh run would record now; NOT the launch-time identity, because "
                    "r28_verify.py changed after the last launch",
        },
    }


def _all_cells() -> list[tuple[str, int, str | None]]:
    cells = [("baseline", scene, None) for scene in common.SCENARIOS]
    cells.append(("disabled", common.GATE_SCENARIO, None))
    points = common.read_json(common.OUTPUT_DIR / "INTERVENTION_POINTS.json")
    for entry in points["scenes"]:
        for point in entry["intervention_points"]:
            cells.append(("intervention", int(entry["scene"]), point["target"]))
    return cells


def _recorded_identity(kind: str, scene: int, target: str | None) -> str:
    return common.read_json(common.row_path(kind, scene, target))["code_identity_freeze_sha256"]


def _recorded_identities() -> list[str]:
    return sorted({_recorded_identity(kind, scene, target)
                   for kind, scene, target in _all_cells()})


if __name__ == "__main__":
    payload = build()
    common.write_json("CODE_IDENTITY.json", payload)
    print("single identity:", payload["launch_time"]["single_identity_across_all_cells"],
          payload["launch_time"]["recorded_in_rows"])
    print("reproduction verified:", payload["launch_time"]["reproduction_verified"])
