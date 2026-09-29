"""ROUND30 代码身份登记：启动身份、冻结后变更、复现路径（只读，0 环境启动）。

与 ROUND29 先例一致：工作树 CRLF 与提交 blob LF 是两个合法字节形态；身份复现需按文件逐个
对齐。本脚本以冻结 manifest 的 harness 段为启动身份，逐模块比对当前工作树，并显式登记
冻结后的两处变更（入口层分片参数、门禁 G3 修正——均非行为代码，且都发生在声明的时间点）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import r30_common as common

R29_PROBE = Path(__file__).resolve().parent


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def blob_sha(commit: str, path: str) -> str | None:
    output = git("show", f"{commit}:{path}")
    if output.startswith("fatal"):
        return None
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def working_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json").read_text(
        encoding="utf-8"))
    launch_harness = manifest["code_identity"]["harness"]
    freeze_commit = manifest["head_at_freeze"]
    current_commit = git("rev-parse", "HEAD")

    per_module: dict[str, Any] = {}
    for name in sorted(launch_harness):
        path = R29_PROBE / name
        if not path.exists():
            per_module[name] = {"status": "missing"}
            continue
        working = working_sha(path)
        blob = blob_sha(freeze_commit, f"runs/comm_deadline_encoding_ablation_20260930/{name}")
        entry: dict[str, Any] = {
            "launch_identity_sha256": launch_harness[name],
            "working_tree_sha256": working,
            "freeze_commit_blob_sha256": blob,
            "matches_launch_identity": working == launch_harness[name],
            "working_tree_is_crlf": b"\r\n" in path.read_bytes(),
        }
        if blob is not None:
            entry["blob_matches_launch_identity"] = blob == launch_harness[name]
        per_module[name] = entry

    # production / imported / prototype / checkpoints: byte comparison against the manifest
    identity = common.code_identity()
    production_same = identity["production"] == manifest["code_identity"]["production"]
    imported_same = (identity["imported_round_modules"]
                     == manifest["code_identity"]["imported_round_modules"])
    prototype_same = (identity["round29_encoder_prototype"]
                      == manifest["code_identity"]["round29_encoder_prototype"])
    checkpoints_same = (identity["checkpoints"] == manifest["code_identity"]["checkpoints"])

    payload: dict[str, Any] = {
        "round": "30",
        "section": "code identity: launch freeze, post-freeze registrations, reproduction",
        "freeze_commit": freeze_commit,
        "current_commit": current_commit,
        "freeze_sha256": manifest["code_identity"]["freeze_sha256"],
        "production_modules_identical_to_freeze": production_same,
        "imported_round_modules_identical_to_freeze": imported_same,
        "round29_encoder_prototype_identical_to_freeze": prototype_same,
        "frozen_checkpoints_identical_to_freeze": checkpoints_same,
        "harness_modules": per_module,
        "post_freeze_registrations": [
            {
                "module": "r30_blocks.py",
                "commit": "82cea7f",
                "launches_before": 11,
                "change": "the eval block gained optional sharding arguments (--arm/--seed/"
                          "--mode); the loop body and every other path are unchanged",
                "behavioural": False,
                "why_allowed": "entry-point orchestration only; no launch had used the changed "
                               "function; the reference launches in flight ran on the freeze "
                               "commit's code",
            },
            {
                "module": "r30_check.py",
                "commit": None,
                "launches_before": 72,
                "change": "first-4 gate G3 recomputes log_prob with the batch-0 pre-update "
                          "weights (the registered fresh initialisation, reconstructible from "
                          "the seed) instead of the post-update rolling checkpoint, and "
                          "load_cell_row/load_cell_tape locate cells by tag glob (training "
                          "cells are named by scenario seed)",
                "behavioural": False,
                "why_allowed": "verification-tool correction only: the earlier G3 run failed "
                               "because it compared pre-update tapes against post-update "
                               "weights; the corrected gates pass with gaps 2.4e-07 / 1.8e-07 "
                               "<= 1e-6. No training, sampling, encoder or environment code "
                               "was touched; the corrected gate overwrote the two failing "
                               "verification outputs (gaps 0.213 / 0.179 under the wrong "
                               "weights were recorded in the session transcript)",
            },
        ],
        "eol_note": "core.autocrlf=true: the working tree holds CRLF where the frozen harness "
                    "was written by python text mode; the per-module blob_matches field lets a "
                    "reviewer see both legitimate byte forms of the same code",
        "reproduction": "from the freeze commit, check out the harness modules (converting the "
                        "CRLF working-tree pairs where needed), run r30_check.py freeze to "
                        "reproduce the manifest, then r30_preflight.py; both must report the "
                        "same freeze_sha256 and 0 failures",
    }
    out = common.OUTPUT_DIR / "CODE_IDENTITY.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    mismatches = [name for name, entry in per_module.items()
                  if entry.get("status") == "missing"
                  or not entry.get("matches_launch_identity")]
    print(json.dumps({"freeze_sha256": manifest["code_identity"]["freeze_sha256"][:16],
                      "production_same": production_same, "imported_same": imported_same,
                      "prototype_same": prototype_same, "checkpoints_same": checkpoints_same,
                      "harness_mismatched_vs_launch": mismatches}, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
