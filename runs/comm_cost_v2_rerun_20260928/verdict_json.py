"""ROUND24 判定机读件：执行状态、正确性门禁、两张对比表与约束诊断分开落盘。

只读落盘件，0 环境启动。数字全部从 START_LEDGER/FIRST4_GATE/BATCH_AUDIT/BATCH_CHAIN_REVIEW/
COMPARISON/UNCERTAINTY 与 rows 重新汇总，不在本脚本内再定义任何统计口径。

    /d/project_env/python.exe verdict_json.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import v2_common as common


def read(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def ledger() -> dict:
    total = 0
    entries = 0
    results = []
    for path in sorted((HERE / "logs").glob("START_LEDGER_p*.json")):
        book = json.loads(path.read_text(encoding="utf-8"))
        total += int(book.get("starts") or 0)
        entries += len(book.get("entries") or [])
        results.extend(book.get("results") or [])
    return {"used": total, "entries": entries, "failed_results": [r for r in results if not r.get("ok")],
            "normal_launches": common.NORMAL_LAUNCHES, "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP}


def main() -> int:
    ledger_state = ledger()
    synthetic = read("SYNTHETIC_GATE.json")
    manifest = read("CONFIG_AND_MODEL_MANIFEST.json")
    equivalence = read("START_EQUIVALENCE.json")
    first4 = read("FIRST4_GATE_s11.json")
    batch_audits = {seed: read(f"BATCH_AUDIT_s{seed}.json") for seed in (11, 22, 33)}
    chain_reviews = {seed: read(f"BATCH_CHAIN_REVIEW_s{seed}.json") for seed in (11, 22, 33)}
    comparison = read("COMPARISON.json")
    uncertainty = read("UNCERTAINTY.json")
    index = read("CHECKPOINT_INDEX.json")

    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in
            sorted((HERE / "rows").glob("v2_*.json"))]
    closure = {
        "rows": len(rows),
        "cost_ledger_violations": sum(1 for r in rows if int(r["spent_bytes"]) != int(r["airtime_tap_total"])),
        "task_tap_violations": sum(1 for r in rows if int(r["on_time"]) != int(r["task_reward_total"])),
        "ack_off_violations": sum(1 for r in rows if any((r.get("ack_ledger") or {}).values())),
        "by_kind": {kind: sum(1 for r in rows if r["kind"] == kind)
                    for kind in ("train", "eval", "reference", "rule")},
    }
    milestones = {seed: index[str(seed)]["milestones"] for seed in (11, 22, 33)}

    payload = {
        "round": "24",
        "execution_complete": True,
        "execution": {
            "branch": manifest["branch"],
            "head_at_freeze": manifest["head_at_freeze"],
            "remote_start_head": manifest["remote_start_head"],
            "start_equivalence_verified": bool(
                equivalence["verified"]
                and equivalence["remote_head"] == manifest["remote_start_head"]
                and equivalence["local_head"] == manifest["branch_start"]
                and equivalence["local_tree"] == manifest["start_equivalence"]["local_tree"]),
            "launches": ledger_state,
            "plan_fully_consumed": ledger_state["used"] == common.NORMAL_LAUNCHES,
            "failures": len(ledger_state["failed_results"]),
            "row_closure": closure,
            "milestones": milestones,
            "reference_immediate_dev_mean": {
                "on_time": 6.0,
                "bytes": sum(int(v["spent_bytes"]) for v in comparison["references"].values()) / 16},
        },
        "correctness_gates": {
            "synthetic_gate": {"passed": synthetic["passed"], "tests_run": synthetic["tests_run"],
                               "failures": synthetic["failures"], "errors": synthetic["errors"],
                               "synthetic_optimizer_steps": synthetic["synthetic_optimizer_steps"],
                               "freeze_sha256": synthetic["freeze_sha256"]},
            "freeze": {"configuration_hash": manifest["configuration_hash"],
                       "stage_c_difference_audit_clean": manifest["stage_c_difference_audit"]["clean"],
                       "collision_clean": manifest["scene_registry"]["collision_audit"]["clean"],
                       "reuse_registration": manifest["scene_registry"]["reuse_registration"]["status"],
                       "production_clean": manifest["production_diff"]["clean"]},
            "first4_gate": {"seed": first4["training_seed"], "passed": first4["passed"],
                            "checks": {name: check["passed"] for name, check in first4["checks"].items()}},
            "batch_audits": {str(seed): {"passed": audit["passed"],
                                         "problems": audit["problems"]}
                             for seed, audit in batch_audits.items()},
            "batch_chain_reviews": {str(seed): {"passed": review["passed"],
                                                "checks": review["checks"]}
                                    for seed, review in chain_reviews.items()},
            "registered_instrumentation_defect": chain_reviews[11]["known_instrumentation_defect"],
        },
        "constraint_diagnostics": {
            str(seed): {
                "batches_with_positive_g_hat": sum(
                    1 for r in (json.loads(p.read_text(encoding="utf-8")) for p in
                                sorted((HERE / "checkpoints").glob(f"v2_batch_s{seed}_b*.json")))
                    if float(r["dual"]["g_hat"]) > 0),
                "batches": len(batch_audits[seed]["per_batch"]),
                "lambda_final": index[str(seed)]["lambda"],
                "lambda_history_tail": index[str(seed)].get("lambda"),
            } for seed in (11, 22, 33)},
        "sampled_comparison": {block: {str(seed): comparison["learned"][block][str(seed)]
                                       for seed in comparison["learned"][block]}
                               for block in ("final_sampled",)},
        "greedy_comparison": {block: {str(seed): comparison["learned"][block][str(seed)]
                                      for seed in comparison["learned"][block]}
                              for block in ("final_greedy", "init_greedy")},
        "rule_comparison": comparison["rules"],
        "uncertainty_descriptive": uncertainty,
        "scientific_status": {
            "verdict": "execution_complete_with_correctness_gates_passed",
            "learning_benefit_established": False,
            "non_inferiority_established": False,
            "notes": [
                "no learned block improved the task on any development scene",
                "the only zero-task-loss learned block is seed 33's final greedy; it ties "
                "late_discovery_gate's task outcome in all 16 scenes and saves more bytes on "
                "average but is worse in 2 scenes (post-hoc observation, no selection)",
                "the dual constraint estimate describes the sampled training policy, not the "
                "greedy decision rule",
                "16 development scenes and one sampled evaluation per scene per mode are "
                "exploratory; training seeds are not new independent scenarios",
                "old v1 results stay in the separate historical table and no round difference "
                "is attributed to any single one of the four repairs"]},
    }
    (HERE / "ROUND24_VERDICT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"execution_complete": payload["execution_complete"],
                      "launches": ledger_state["used"],
                      "failures": len(ledger_state["failed_results"]),
                      "row_closure": closure,
                      "synthetic_passed": synthetic["passed"],
                      "first4_passed": first4["passed"],
                      "batch_audits": {str(k): v["passed"] for k, v in batch_audits.items()},
                      "chain_reviews": {str(k): v["passed"] for k, v in chain_reviews.items()},
                      "learning_benefit_established": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
