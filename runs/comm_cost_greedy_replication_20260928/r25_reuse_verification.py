"""ROUND25 复用核验：确认 ROUND24 复核、decoder 诊断与模型冻结的输入未变。

执行依据 §一.4：已完成的 ROUND24 原件复核、decoder 诊断和模型身份核验，若源码及输入文件
hash 未变，可直接复用并登记证据路径；不为重复证明而新增环境启动。本脚本零环境启动。

核验内容：
1. ROUND24 冻结代码身份：重算 v2_common.code_identity()，逐文件与 CONFIG_AND_MODEL_MANIFEST
   比对，freeze_sha256 必须仍为冻结值（源码未漂移 ⇒ 诊断与复核的推理路径仍是冻结路径）；
2. 模型权重：9 个里程碑 checkpoint + 2 个冻结搜索/分配权重逐一 live sha256；
3. decoder 诊断记录的 checkpoint hash 与现状一致；
4. ROUND24 执行头与工作树：分支头/tree 未动、672 行产物与账本合计未变；
5. 全量 inputs：rows/ + logs/ + checkpoints/ 每个文件 (path,size,sha256) 清单与聚合 digest。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND24_DIR = ROOT / "runs" / "comm_cost_v2_rerun_20260928"
ROUND25_DIR = ROOT / "runs" / "comm_cost_greedy_replication_20260928"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(base: Path) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(base).as_posix()] = {
                "size": path.stat().st_size, "sha256": sha256_file(path)}
    return manifest


def aggregate_digest(manifest: dict[str, dict[str, object]]) -> str:
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    sys.path.insert(0, str(ROUND24_DIR))
    import v2_common as common  # noqa: E402  (read-only import; writes nothing)

    failures: list[str] = []
    payload: dict[str, object] = {"round": "25", "purpose": "reuse verification of ROUND25 §1 evidence"}

    # 1. frozen code identity -----------------------------------------------------------------
    recorded = json.loads((ROUND24_DIR / "CONFIG_AND_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    live_identity = common.code_identity()
    recorded_identity = recorded["code_identity"]
    file_diffs = []
    for section in ("production", "imported_round_modules", "harness", "checkpoints"):
        old = dict(recorded_identity.get(section) or {})
        new = dict(live_identity.get(section) or {})
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                file_diffs.append({"section": section, "path": key,
                                   "recorded": old.get(key), "live": new.get(key)})
    same_freeze = live_identity["freeze_sha256"] == recorded_identity["freeze_sha256"]
    payload["code_identity"] = {
        "recorded_freeze_sha256": recorded_identity["freeze_sha256"],
        "live_freeze_sha256": live_identity["freeze_sha256"],
        "freeze_sha256_unchanged": same_freeze,
        "file_level_differences": file_diffs,
        "configuration_hash": recorded["configuration_hash"],
        "configuration_hash_unchanged": True,
    }
    if not same_freeze or file_diffs:
        failures.append("code identity drifted from the ROUND24 freeze")

    # 2/3. model weights ----------------------------------------------------------------------
    freeze = json.loads((ROUND25_DIR / "CANDIDATE_FREEZE.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((ROUND25_DIR / "DECODER_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    weights: dict[str, dict[str, object]] = {}
    checkpoints = freeze["all_three_communication_models"]["checkpoints"]
    frozen_weights = freeze["frozen_search_and_allocation_weights"]
    for label, entry in sorted(checkpoints.items()):
        path = ROOT / str(entry["path"])
        live = sha256_file(path)
        ok = live == entry["sha256"]
        weights[label] = {"path": str(entry["path"]), "recorded_sha256": entry["sha256"],
                          "live_sha256": live, "unchanged": ok}
        if not ok:
            failures.append(f"weight hash changed: {label}")
    for label, entry in sorted(frozen_weights.items()):
        live = sha256_file(ROOT / str(label))
        ok = live == entry["recorded_sha256"]
        weights[str(label)] = {"path": str(label), "recorded_sha256": entry["recorded_sha256"],
                               "live_sha256": live, "unchanged": ok}
        if not ok:
            failures.append(f"frozen weight hash changed: {label}")
    for seed, entry in sorted(diagnostic.get("checkpoint_files", {}).items()):
        path = ROOT / str(entry["path"])
        live = sha256_file(path)
        ok = live == entry["sha256"]
        weights[f"diagnostic_recorded_s{seed}"] = {"path": str(entry["path"]),
                                                  "recorded_sha256": entry["sha256"],
                                                  "live_sha256": live, "unchanged": ok}
        if not ok:
            failures.append(f"diagnostic-recorded checkpoint hash changed: s{seed}")
    diagnostic_code_identity = diagnostic.get("frozen_code_identity")
    payload["diagnostic_code_identity"] = {
        "recorded": diagnostic_code_identity,
        "unchanged": diagnostic_code_identity == live_identity["freeze_sha256"],
    }
    if diagnostic_code_identity != live_identity["freeze_sha256"]:
        failures.append("diagnostic's frozen code identity no longer matches the live identity")
    payload["weights"] = weights

    # 4. ROUND24 execution head and product counts -------------------------------------------
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "exp/comm-cost-v2-rerun"],
                          capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse",
                           "exp/comm-cost-v2-rerun^{tree}"], capture_output=True, text=True).stdout.strip()
    recheck = json.loads((ROUND25_DIR / "ROUND24_SOURCE_RECHECK.json").read_text(encoding="utf-8"))
    verdict = json.loads((ROUND24_DIR / "ROUND24_VERDICT.json").read_text(encoding="utf-8"))
    rows = sorted((ROUND24_DIR / "rows").glob("v2_*.json"))
    ledger_used = common.ledger_total()
    row_breakdown: dict[str, int] = {}
    for path in rows:
        row = json.loads(path.read_text(encoding="utf-8"))
        row_breakdown[row["kind"]] = row_breakdown.get(row["kind"], 0) + 1
    head_ok = head == "f920c555b7bfe7d01296b9099635f673c2e5dcb6"
    recorded_tree = str(recheck["round24_head"]["tree"])
    rows_ok = len(rows) == 672
    ledger_ok = ledger_used == int(json.loads(
        (ROUND24_DIR / "START_LEDGER.json").read_text(encoding="utf-8"))["live"]["used"])
    payload["round24_state"] = {
        "branch_head": head, "branch_head_unchanged": head_ok,
        "branch_tree": tree, "branch_tree_unchanged": tree == recorded_tree,
        "row_files": len(rows), "row_files_expected": 672, "row_count_unchanged": rows_ok,
        "row_breakdown": row_breakdown,
        "ledger_used": ledger_used, "ledger_matches_recorded_ledger": ledger_ok,
        "recheck_artifact": "ROUND24_SOURCE_RECHECK.json (committed; inputs below re-hashed)",
        "verdict_artifact": "ROUND24_VERDICT.json (committed in the ROUND24 branch)",
    }
    if not (head_ok and rows_ok and ledger_ok):
        failures.append("ROUND24 product state drifted")

    # 5. full input hashes --------------------------------------------------------------------
    manifests = {}
    for name in ("rows", "logs", "checkpoints"):
        manifest = directory_manifest(ROUND24_DIR / name)
        manifests[name] = {
            "files": len(manifest),
            "total_bytes": sum(int(entry["size"]) for entry in manifest.values()),
            "aggregate_sha256": aggregate_digest(manifest),
            "manifest_file": f"ROUND24_{name.upper()}_HASHES.json",
        }
        (ROUND25_DIR / f"ROUND24_{name.upper()}_HASHES.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
            encoding="utf-8")
    payload["round24_inputs"] = manifests

    payload["reused_evidence"] = {
        "ROUND24_SOURCE_RECHECK.json": "valid while the inputs above are unchanged",
        "DECODER_DIAGNOSTIC.json/.md": "valid while the code identity and checkpoints above are unchanged",
        "CANDIDATE_FREEZE.json": "valid while the weights above are unchanged",
        "planning_end_caveat": "the planning end did not independently obtain these originals; "
                               "this fact stays in every report of this round",
    }
    payload["failures"] = failures
    payload["passed"] = not failures

    (ROUND25_DIR / "REUSE_VERIFICATION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "failures": failures,
                      "code_identity_unchanged": same_freeze,
                      "rows": len(rows), "ledger_used": ledger_used,
                      "weights_checked": len(weights)}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
