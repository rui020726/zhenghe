"""ROUND25 统计缺陷登记与更正：bootstrap 重采样算术错误及其影响面。

零环境启动。事实链：
1. 缺陷：`r25_stats.bootstrap_ci` 用 `matrix @ values / 64`（按索引加权乘积）代替重采样均值
   `values[matrix].mean(axis=1)`。均值与各计数字段不经过该矩阵，未受影响；所有 bootstrap 95% CI
   全部错误。第一次算出的 PAIRED_COMPARISON/VERDICT（原样保存为 SUPERSEDED_FIRST_*.json/md）
   据此把主成本判成 lower_cost_supported——该判定作废。
2. 发现：更正前预览时 CI 界超出逐场景差值的可能值域；手工可算案例（单场景 -2、63 平手）确认。
3. 影响面：该函数只在聚合路径（paired_table）调用；episode 路径从不调用。192 行 final record
   全部携带同一代码身份 ⇒ episode 矩阵同质，统计更正不触及任何一次启动。
4. 修复：改为 values[matrix].mean(axis=1)，并加入不变量（CI 必在差值值域内）防再犯。
5. 身份核对：冻结时身份 vs 启动时身份（git 394aa0f，须等于行内记录身份）vs 当前身份
   （统计修复后，仅 r25_stats.py 变化），逐文件列明。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import r25_common as common
import r25_stats as stats

FIRST_COMMIT = "394aa0f"


def _git_show(commit: str, path: Path) -> bytes:
    rel = path.resolve().relative_to(common.ROOT).as_posix()
    out = subprocess.run(["git", "-C", str(common.ROOT), "show", f"{commit}:{rel}"],
                         capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"git show {commit}:{rel} failed: {out.stderr.decode('utf-8', 'replace')}")
    return out.stdout


def identity_from_commit(commit: str) -> dict[str, Any]:
    """The code identity the launches ran with, recomputed from the committed tree."""
    import hashlib

    payload: dict[str, Any] = {
        "production": {name: hashlib.sha256(_git_show(commit, common.ROOT / name)).hexdigest()
                       for name in common.PRODUCTION_DEPENDENCIES},
        "imported_round_modules": {
            common._rel(path): hashlib.sha256(_git_show(commit, path)).hexdigest()
            for path in common.IMPORTED_ROUND_MODULES if path.exists()},
        "harness": {name: hashlib.sha256(_git_show(commit, common.HERE / name)).hexdigest()
                    for name in common.HARNESS_MODULES if (common.HERE / name).exists()},
        "checkpoints": {common._rel(path): common.sha256_file(path)
                        for path in common.CHECKPOINT_FILES},
        "learned_checkpoint": {common._rel(common.LEARNED_CHECKPOINT):
                               common.sha256_file(common.LEARNED_CHECKPOINT)},
    }
    payload["freeze_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def eol_equivalence(commit: str) -> dict[str, Any]:
    """File-by-file proof that the executed bytes and the committed blobs differ only by eol.

    ``core.autocrlf=true`` stores LF blobs while the working tree carries CRLF for the
    pre-existing tracked files; git's clean filter makes the tree look unmodified. The rows hash
    the bytes that actually executed, so both identities are registered with this proof.
    """
    rows: list[dict[str, Any]] = []
    only_eol = True
    tracked = list(common.PRODUCTION_DEPENDENCIES) + [common._rel(path)
                                                     for path in common.IMPORTED_ROUND_MODULES
                                                     if path.exists()]
    for rel in tracked:
        path = common.ROOT / rel
        work = path.read_bytes()
        blob = _git_show(commit, path)
        if work == blob:
            rows.append({"file": rel, "relation": "identical"})
        elif work.replace(b"\r\n", b"\n") == blob:
            rows.append({"file": rel, "relation": "crlf_only"})
        else:
            only_eol = False
            rows.append({"file": rel, "relation": "content_differs"})
    return {"only_eol_differences": only_eol, "files": rows,
            "core_autocrlf": "true (git config)"}


def file_diffs(commit: str) -> list[dict[str, str]]:
    diffs = []
    for name in common.HARNESS_MODULES:
        path = common.HERE / name
        if not path.exists():
            continue
        committed = _git_show(commit, path)
        if committed != path.read_bytes():
            diffs.append({"file": name, "state": "changed after the recorded commit"})
    return diffs


def run_time_identity_reproduced(commit: str) -> dict[str, Any]:
    """Rebuild the exact payload the launches hashed and compare with the rows' identity.

    The launches hashed the working-tree bytes with the pre-fix r25_stats.py; every other file is
    unchanged on disk since, so substituting the committed pre-fix blob must reproduce the rows'
    recorded identity exactly.
    """
    import hashlib

    payload: dict[str, Any] = {
        "production": {name: hashlib.sha256((common.ROOT / name).read_bytes()).hexdigest()
                       for name in common.PRODUCTION_DEPENDENCIES},
        "imported_round_modules": {common._rel(path): common.sha256_file(path)
                                   for path in common.IMPORTED_ROUND_MODULES if path.exists()},
        "harness": {name: (hashlib.sha256(_git_show(commit, common.HERE / name)).hexdigest()
                           if name == "r25_stats.py" else common.sha256_file(common.HERE / name))
                    for name in common.HARNESS_MODULES if (common.HERE / name).exists()},
        "checkpoints": {common._rel(path): common.sha256_file(path)
                        for path in common.CHECKPOINT_FILES},
        "learned_checkpoint": {common._rel(common.LEARNED_CHECKPOINT):
                               common.sha256_file(common.LEARNED_CHECKPOINT)},
    }
    payload["freeze_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def hand_computed_check() -> dict[str, Any]:
    matrix = stats.resample_matrix()
    values = [-2.0] + [0.0] * 63          # one scene loses 2 on-time tasks, 63 tie
    counts = (matrix == 0).sum(axis=1)    # how often scene 0 is drawn per resample row
    expected_low, expected_high = np.quantile(-2.0 * counts / 64.0, [0.025, 0.975],
                                              method="linear")
    got = stats.bootstrap_ci(values, matrix)
    ok = abs(got["ci_low"] - float(expected_low)) < 1e-12 and abs(
        got["ci_high"] - float(expected_high)) < 1e-12
    return {"case": "one scene dY=-2, 63 tied",
            "expected_ci": [float(expected_low), float(expected_high)],
            "recomputed_ci": [got["ci_low"], got["ci_high"]], "passed": ok}


def main() -> int:
    matrix = stats.resample_matrix()
    current = common.code_identity()
    launch_identity = identity_from_commit(FIRST_COMMIT)
    reproduced = run_time_identity_reproduced(FIRST_COMMIT)
    eol_proof = eol_equivalence(FIRST_COMMIT)
    arm_freeze = json.loads((common.OUTPUT_DIR / "ARM_FREEZE.json").read_text(encoding="utf-8"))
    rows_identity = sorted({json.loads(path.read_text(encoding="utf-8"))["code_identity_freeze_sha256"]
                            for path in common.ROWS.glob("r25_*_a1.json")})
    superseded = json.loads((common.OUTPUT_DIR / "SUPERSEDED_FIRST_STATISTICS.json").read_text(
        encoding="utf-8"))
    corrected = json.loads((common.OUTPUT_DIR / "PAIRED_COMPARISON.json").read_text(
        encoding="utf-8"))

    def field(comparator: str, metric: str, key: str) -> Any:
        return corrected["comparisons"][comparator][metric][key]

    def old_field(comparator: str, metric: str, key: str) -> Any:
        return superseded["comparisons"][comparator][metric][key]

    payload = {
        "round": "25",
        "section": "statistics defect registration and correction",
        "defect": {
            "function": "r25_stats.bootstrap_ci",
            "defective_code": "means = matrix @ values / float(matrix.shape[1])",
            "correct_code": "means = values[matrix].mean(axis=1)",
            "nature": "the matrix product is an index-weighted sum, not the mean of the deltas at "
                      "the resampled scene indices; every percentile CI was therefore wrong",
            "unaffected_fields": ["per-scene dY/dC", "means", "all count fields",
                                  "the zero-event bound", "absolute metrics",
                                  "goal identity tradeoffs"],
            "affected_fields": ["every bootstrap 95% CI (task and cost, both comparators)"],
            "blast_radius": "the first PAIRED_COMPARISON.json and the first ROUND25_VERDICT.json/.md; "
                            "both are preserved verbatim as SUPERSEDED_FIRST_STATISTICS.json and "
                            "SUPERSEDED_FIRST_VERDICT.json/.md",
            "first_verdict_was_wrong": "the superseded verdict claimed primary_cost_evidence = "
                                       "lower_cost_supported from a defective CI; that claim is "
                                       "void",
        },
        "episodes_unaffected": {
            "call_path": "bootstrap_ci is called only from paired_table (aggregation); the runner "
                         "(r25_run/r25_blocks) never calls it",
            "row_code_identities": rows_identity,
            "homogeneous_matrix": len(rows_identity) == 1,
            "fix_timing": "r25_stats.py was edited after the last launch of the 192; no launch was "
                          "affected and no cell was re-run",
        },
        "identity_reconciliation": {
            "freeze_time_recorded_in_ARM_FREEZE": arm_freeze["code_identity_freeze_sha256"],
            "freeze_time_note": "recorded before zero-episode preflight edits; the preflight is "
                                "not part of any episode path",
            "run_time_identity_in_rows": rows_identity[0] if len(rows_identity) == 1 else rows_identity,
            "run_time_identity_reproduced": reproduced["freeze_sha256"],
            "run_time_identity_reproduced_exactly": reproduced["freeze_sha256"] == rows_identity[0],
            "launch_commit": FIRST_COMMIT,
            "launch_commit_identity_lf_blobs": launch_identity["freeze_sha256"],
            "launch_commit_vs_run_time": eol_proof,
            "current_identity_after_fix": current["freeze_sha256"],
            "files_changed_after_launch_commit": file_diffs(FIRST_COMMIT),
            "interpretation": "the rows embed the identity of the bytes that actually executed; "
                              "the committed blobs are the same code with git's autocrlf "
                              "normalisation applied to the pre-existing tracked files, so both "
                              "identities are registered and the relation is proven file by file",
        },
        "verification": {
            "hand_computed_case": hand_computed_check(),
            "range_invariant": "every CI must lie inside [min(deltas), max(deltas)]; the defective "
                               "arithmetic would now raise instead of producing a number",
            "launch_ledger_final": {"used": common.launch_count(),
                                    "per_process_total": common.ledger_total(),
                                    "ledgers_agree": common.ledger_total() == common.launch_count()},
            "shared_matrix_sha256": corrected["shared_resampling"]["matrix_sha256"],
            "shared_matrix_unchanged": corrected["shared_resampling"]["matrix_sha256"]
            == superseded["shared_resampling"]["matrix_sha256"],
        },
        "before_after": {
            "primary_cost_evidence": {"superseded": "lower_cost_supported (defective CI)",
                                      "corrected": "lower_cost_not_established"},
            "late_dC_ci": {"superseded": [old_field("late_discovery_gate_147", "cost", "ci_low"),
                                          old_field("late_discovery_gate_147", "cost", "ci_high")],
                           "corrected": [field("late_discovery_gate_147", "cost", "ci_low"),
                                         field("late_discovery_gate_147", "cost", "ci_high")]},
            "late_dY_ci": {"superseded": [old_field("late_discovery_gate_147", "task", "ci_low"),
                                          old_field("late_discovery_gate_147", "task", "ci_high")],
                           "corrected": [field("late_discovery_gate_147", "task", "ci_low"),
                                         field("late_discovery_gate_147", "task", "ci_high")]},
            "immediate_dY_ci": {"superseded": [old_field("immediate", "task", "ci_low"),
                                               old_field("immediate", "task", "ci_high")],
                                "corrected": [field("immediate", "task", "ci_low"),
                                              field("immediate", "task", "ci_high")]},
            "immediate_dC_ci": {"superseded": [old_field("immediate", "cost", "ci_low"),
                                               old_field("immediate", "cost", "ci_high")],
                                "corrected": [field("immediate", "cost", "ci_low"),
                                              field("immediate", "cost", "ci_high")]},
            "observed_task_count_status": {"superseded": "task_count_tradeoff_observed",
                                           "corrected": "task_count_tradeoff_observed"},
            "note": "means and counts are identical before and after; only the intervals moved",
        },
        "final_positions": {
            "primary_cost_evidence": "lower_cost_not_established",
            "observed_task_count_status": "task_count_tradeoff_observed",
        },
        "matrix_digest_unchanged": corrected["shared_resampling"]["matrix_sha256"]
        == stats.matrix_digest(matrix),
    }
    (common.OUTPUT_DIR / "STATS_CORRECTION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    # the corrected verdict carries the correction reference in both formats
    verdict_path = common.OUTPUT_DIR / "ROUND25_VERDICT.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    verdict["correction"] = {
        "artefact": "STATS_CORRECTION.json",
        "summary": "the first statistics run had defective bootstrap intervals; every CI was "
                   "recomputed with the corrected resampling arithmetic; means and counts are "
                   "unchanged; the superseded first verdict is preserved as "
                   "SUPERSEDED_FIRST_VERDICT.json/.md",
        "superseded_first_verdict": "SUPERSEDED_FIRST_VERDICT.json/.md (claim of lower_cost_supported "
                                    "is void)",
        "episodes_affected": 0,
        "launches_rerun": 0,
    }
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
    md_path = common.OUTPUT_DIR / "ROUND25_VERDICT.md"
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## 更正记录\n\n")
        handle.write("- 首次统计的 bootstrap 区间存在重采样算术缺陷（详见 STATS_CORRECTION.json）；"
                     "均值与计数不变，全部 CI 已用更正后的算法重算。\n")
        handle.write("- 作废的首次判定原样保存为 SUPERSEDED_FIRST_VERDICT.json/.md；"
                     "其中 lower_cost_supported 的声明无效。\n")
        handle.write("- 0 次 episode 受影响、0 次启动重跑；192 行 final record 代码身份一致。\n")

    print(json.dumps({"hand_computed": payload["verification"]["hand_computed_case"]["passed"],
                      "run_time_identity_reproduced_exactly":
                          payload["identity_reconciliation"]["run_time_identity_reproduced_exactly"],
                      "commit_vs_runtime_only_eol":
                          payload["identity_reconciliation"]["launch_commit_vs_run_time"]["only_eol_differences"],
                      "matrix_unchanged": payload["matrix_digest_unchanged"],
                      "primary_cost_evidence": payload["final_positions"]["primary_cost_evidence"]},
                     ensure_ascii=False))
    ok = (payload["verification"]["hand_computed_case"]["passed"]
          and payload["identity_reconciliation"]["run_time_identity_reproduced_exactly"]
          and payload["identity_reconciliation"]["launch_commit_vs_run_time"]["only_eol_differences"]
          and payload["matrix_digest_unchanged"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
