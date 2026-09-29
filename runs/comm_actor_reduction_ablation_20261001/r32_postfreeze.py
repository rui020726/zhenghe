"""ROUND32 登记件：冻结之后改过的文件，逐处归类并证明训练路径未被触碰（0 环境启动）。

这不是"顺手改了就说没事"：它对每个与冻结哈希不同的 harness 文件做 AST 函数级定位 +
difflib 行级差异，要求
  (a) 该文件不在训练/采样/更新路径里（r32_common / r32_trainer / r32_run / r32_train /
      r32_blocks / r32_freeze 六件的哈希必须与冻结件完全相同），并且
  (b) 改动只落在只读分析/验证函数内。
任何一条不成立即报告为问题，不生成"通过"。

本脚本自身不参与冻结哈希（它是冻结之后的登记工具）。
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import r32_common as common

#: 决定采样、优势、更新、恢复行为的文件：任何一个被改动都必须停止并说明。
TRAINING_PATH = ("r32_common.py", "r32_trainer.py", "r32_run.py", "r32_train.py",
                 "r32_blocks.py", "r32_freeze.py", "r32_math.py", "r32_preflight.py")
#: 允许改动的分析件，以及各自允许被改的函数名（只读分析/验证函数）
ALLOWED = {
    "r32_check.py": {"first4_gate", "cross_arm_gate"},
    "r32_curves.py": {"run", "main", "mean", "summary", "recount", "train_rows", "<module>"},
    "r32_mechanism.py": {"analyse", "main", "summary", "mean", "load_episode", "<module>"},
    "r32_verify.py": {"bootstrap_recheck", "main", "lambda_recheck", "rebuilt_table",
                      "init_pairing_provenance_recheck", "<module>"},
    "r32_stats.py": {"absolute_block"},
    "r32_verdict.py": {"main", "sending", "message_read", "advantage_scale_read",
                       "outcome_direction", "<module>"},
}
#: the only module-level shapes a post-freeze edit may take (local analysis helpers, imports)
MODULE_LEVEL_OK = re.compile(
    r'^(def (mean|summary)\(|import [a-z]|from [a-z_]+ import|\)$|' + '"""' + r'|\s*$)')


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True,
                          encoding="utf-8").stdout


def frozen_harness() -> dict[str, str]:
    manifest = common.read_json(common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json")
    return manifest["code_identity"]["harness"]


def module_level_lines(old_src: str, new_src: str) -> list[str]:
    """Changed new-file lines that sit outside any function: these must be helpers or imports."""
    tree = ast.parse(new_src)
    spans = [(node.lineno, node.end_lineno)
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    new_lines = new_src.splitlines()
    out: list[str] = []
    matcher = difflib.SequenceMatcher(a=old_src.splitlines(), b=new_lines)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for line in range(j1 + 1, j2 + 1):
            if not any(lo <= line <= hi for lo, hi in spans):
                out.append(new_lines[line - 1].strip())
    return out


def functions_touched(old_src: str, new_src: str) -> tuple[list[str], int]:
    """Map every changed line of the new file to the function that owns it."""
    tree = ast.parse(new_src)
    spans = [(node.name, node.lineno, node.end_lineno)
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    owners: set[str] = set()
    changed = 0
    matcher = difflib.SequenceMatcher(a=old_src.splitlines(), b=new_src.splitlines())
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += j2 - j1
        for line in range(j1 + 1, j2 + 1):
            hits = [name for name, lo, hi in spans if lo <= line <= hi]
            owners.add(hits[0] if hits else "<module>")
    return sorted(owners), changed


def preserve_prelaunch_pairing(freeze_commit: str) -> dict[str, Any]:
    """Copy the committed pre-launch pairing evidence out of Git before it is overwritten again.

    ``r32_train.record_init_pairing`` rewrites ``INIT_PAIRING_<arm>_s<seed>.json`` at the first
    real fresh initialisation of each run, and its payload carries no provenance marker. Every
    load-bearing field is identical to the pre-launch version, so the pairing claim is unaffected,
    but the in-file statement that the evidence predates the first launch survives only in Git.
    This writes both versions' comparison into one file so the delivery does not depend on anyone
    remembering to run ``git show``.
    """
    prelaunch: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    for reduction in common.ARM_ORDER:
        for seed in common.TRAIN_SEEDS:
            key = f"{reduction}_s{seed}"
            relative = f"runs/{common.OUTPUT_DIR.name}/INIT_PAIRING_{reduction}_s{seed}.json"
            blob = git("show", f"{freeze_commit}:{relative}")
            if not blob.strip():
                comparison[key] = {"found_in_freeze_commit": False}
                continue
            before = json.loads(blob)
            after = common.read_json(common.OUTPUT_DIR / f"INIT_PAIRING_{reduction}_s{seed}.json")
            prelaunch[key] = before
            comparison[key] = {
                "found_in_freeze_commit": True,
                "parameter_sha256_equal": (before["controller_parameters"]["parameter_sha256"]
                                           == after["controller_parameters"]["parameter_sha256"]),
                "module_digests_equal": before["module_digests"] == after["module_digests"],
                "optimizer_state_equal": (before["optimizer_state_sha256"]
                                          == after["optimizer_state_sha256"]),
                "lambda_equal": before["lambda"] == after["lambda"],
                "keys_only_in_the_prelaunch_version": sorted(set(before) - set(after)),
                "keys_only_in_the_run_time_version": sorted(set(after) - set(before)),
                "other_fields_equal": all(before[f] == after[f] for f in set(before) & set(after)),
            }
    payload = {
        "round": "32",
        "section": "pre-launch initialisation pairing, as committed before the first launch",
        "taken_from_commit": freeze_commit,
        "prelaunch_pairing_evidence": prelaunch,
        "comparison_with_the_run_time_files_on_disk": comparison,
        "why_this_file_exists": "r32_train rewrites its own copy of these six files when a run "
                                "first initialises, and that payload drops the "
                                "written_before_the_first_environment_launch marker; the values "
                                "are the same, so this file preserves the version that carries "
                                "the provenance statement instead of leaving it only in Git",
        "defect_registered": True,
    }
    out = common.OUTPUT_DIR / "INIT_PAIRING_PRELAUNCH.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    return {"written": out.name,
            "cells": len(comparison),
            "all_load_bearing_fields_equal": all(
                value.get("other_fields_equal") and value.get("parameter_sha256_equal")
                for value in comparison.values()),
            "marker_key_lost_in_the_run_time_copy": {
                key: value.get("keys_only_in_the_prelaunch_version") or None
                for key, value in comparison.items()}}


def main() -> int:
    frozen = frozen_harness()
    current = common.code_identity()["harness"]
    manifest_commit = git("rev-parse", "HEAD").strip()
    problems: list[str] = []
    entries: list[dict[str, Any]] = []

    for name in TRAINING_PATH:
        if frozen.get(name) != current.get(name):
            problems.append(f"{name} is on the sampling/update/restore path but its hash changed "
                            "after the freeze")

    for name in sorted(set(frozen) | set(current)):
        if frozen.get(name) == current.get(name):
            continue
        owners, changed, module_lines = [], 0, []
        spec = f":/runs/{common.OUTPUT_DIR.name}/{name}"  # repo-root relative
        history = git("log", "--format=%H", "--", spec).split()
        baseline = None
        for commit in history:
            blob = git("show", f"{commit}:runs/{common.OUTPUT_DIR.name}/{name}")
            if blob.strip():
                baseline = blob
                break
        path = common.OUTPUT_DIR / name
        new_src = path.read_text(encoding="utf-8") if path.exists() else ""
        if baseline is None:
            problems.append(f"{name}: changed after the freeze but no frozen baseline found")
        else:
            owners, changed = functions_touched(baseline, new_src)
            allowed = ALLOWED.get(name, set())
            outside = sorted(set(owners) - allowed)
            if outside:
                problems.append(f"{name}: functions changed outside the allowed analysis set: "
                                f"{outside}")
            loose = [line for line in module_level_lines(baseline, new_src)
                      if not MODULE_LEVEL_OK.match(line)]
            if loose:
                problems.append(f"{name}: module-level changed lines are not helper/import "
                                f"shapes: {loose[:6]}")
            module_lines = module_level_lines(baseline, new_src)
        entries.append({"file": name, "frozen_sha256": frozen.get(name),
                        "current_sha256": current.get(name),
                        "functions_touched": owners, "lines_changed": changed,
                        "module_level_changed_lines": module_lines,
                        "allowed_function_names": sorted(ALLOWED.get(name, set())),
                        "on_training_path": name in TRAINING_PATH})

    pairing_preserved = preserve_prelaunch_pairing(manifest_commit)

    payload = {
        "round": "32",
        "section": "post-freeze changes registered, not silently applied (0 launches)",
        "frozen_at_head": manifest_commit,
        "prelaunch_pairing_preserved": pairing_preserved,
        "freeze_sha256": common.read_json(
            common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json")["code_identity"]["freeze_sha256"],
        "training_path_files": list(TRAINING_PATH),
        "training_path_unchanged": not any(
            frozen.get(name) != current.get(name) for name in TRAINING_PATH),
        "changed_files": entries,
        "why_no_launch_is_repeated": "every change below sits in a read-only verification or "
                                     "post-run analysis function over artefacts that were already "
                                     "metered; no episode, action, reward, advantage, update or "
                                     "checkpoint is produced by these functions, so no cell is "
                                     "re-run and the ledger is untouched",
        "reasons": {
            "r32_check.py": "G8 compared per-minibatch N/M sums (which are per epoch, so the "
                            "batch sum is epochs x rows: recorded 2492 = 4*623 rows and "
                            "120 = 4*30 active rows) against raw tape row counts. The counts were "
                            "right; the assertion was wrong.",
            "r32_curves.py": "the same epochs x rows relation was asserted per epoch instead of "
                             "against the raw row count, and local descriptive helpers were added "
                             "so no training constant is re-declared here.",
            "r32_mechanism.py": "the amplification check used an absolute 1e-5 band on a ratio "
                                "that reaches 51, which only tests float32 digit count; it now "
                                "uses a relative 1e-5 band and records both residuals. The "
                                "episode loader's row glob omitted the `_s{scenario}` segment of "
                                "the archived file names, so it read nothing.",
            "r32_verify.py": "the bootstrap recheck compared unrounded slow-loop intervals "
                             "against the delivered artefact's own rounding (delta_y 6 decimals, "
                             "delta_c 3 decimals), and the training-row glob omitted the "
                             "scenario segment of the archived file names. Row pattern fix also "
                             "applied to the mechanism loader. A new independent check, "
                             "init_pairing_provenance_recheck, compares every cell's pairing file "
                             "against the committed pre-launch version -- that is what caught the "
                             "lost `written_before_the_first_environment_launch` marker "
                             "(report section 1.1).",
            "r32_stats.py": "the absolute per-arm block now also carries the message count "
                            "(emissions_total_16_scenes, mean_emissions_per_episode) so the "
                            "behavioural reading 'who sent more' is in the same artefact as the "
                            "criterion (bytes and on-time tasks) instead of being retyped in "
                            "prose. Purely additive: no existing field, estimator, resampling "
                            "index or rounding rule was touched, and r32_verify's rebuilt table "
                            "re-derives the delta fields from the raw rows independently.",
            "r32_verdict.py": "the verdict file is generated from the delivered artefacts; its "
                              "seven answers, the verified block and the message read were added "
                              "so no number in the conclusion is hand-typed. It reads JSON only.",
        },
        "problems": problems,
        "clean": not problems,
    }
    out = common.OUTPUT_DIR / "POST_FREEZE_CHANGES.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"clean": payload["clean"], "problems": problems,
                      "training_path_unchanged": payload["training_path_unchanged"],
                      "changed": [{k: e[k] for k in
                                   ("file", "functions_touched", "lines_changed")}
                                  for e in entries]}, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
