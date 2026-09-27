"""第二十轮 Stage A 的 MANIFEST：边界声明 + 交付件哈希。

只读本目录与 git；不启动任何环境。
"""

from __future__ import annotations

import json
import sys

import c20_common as c20

START_SHA = "efca2a50c8d06f6fa7062c204d94c98b137206a4"
SPEC_COMMIT = "72c4c3f71e07339bf6533fbd771aef3b2682103c"
BUNDLE = c20.REPO_ROOT.parent / "specs" / "comm-cost-original-specs-72c4c3f.bundle"
DELIVERED = ("STAGE_A_IMPLEMENTATION.md", "STAGE_BC_RESOURCE_AND_STATISTICS.md",
             "STAGE_A_TEST_RESULTS.json", "SPEC_ITEM_VERIFICATION.json",
             "cost_math.py", "test_cost_math.py", "c20_common.py", "c20_audit.py",
             "c20_spec_items.py")
UPDATED = ("DESIGN_SPEC_COMPLIANCE.md",)


def main() -> int:
    results = c20.read_json(c20.ROUND_DIR / "STAGE_A_TEST_RESULTS.json")
    specs = c20.read_json(c20.ROUND_DIR / "SPEC_ITEM_VERIFICATION.json")
    compliance_dir = c20.REPO_ROOT / "runs" / "comm_cost_objective_design_20260926"
    payload = {
        "round": "20", "phase": "Stage A",
        "branch": c20.git("rev-parse", "--abbrev-ref", "HEAD"),
        "start_sha": START_SHA,
        "head_at_manifest": c20.git("rev-parse", "HEAD"),
        "imported_document_commit": {
            "sha": SPEC_COMMIT, "parent": "80af16ba4b90ce0c2d5456dd8b8e57068b4513e4",
            "files": 2, "local_cherry_pick": c20.git("rev-parse", "HEAD^"),
            "bundle_path": str(BUNDLE),
            "bundle_sha256": c20.sha256_file(BUNDLE)},
        "boundary": {
            "environment_launches": 0, "new_streams": 0, "optimizer_steps": 0,
            "real_checkpoint_loaded": False, "sealed_artefacts_read": 0,
            "sealed_ranges": ["800000..800399", "900000..900199"],
            "production_trees_diff": results["production_diff"],
            "import_audit": results["import_audit"],
            "tests_ran": results["tests"]["test_count"],
            "tests_passed": results["tests"]["passed"],
            "evidence": "STAGE_A_TEST_RESULTS.json（子进程跑测试、导入审计、git diff 三件套）"},
        "new_deliverables": {name: {"path": f"runs/comm_cost_controller_stage_a_20260927/{name}",
                                    "bytes": (c20.ROUND_DIR / name).stat().st_size,
                                    "sha256": c20.sha256_file(c20.ROUND_DIR / name)}
                             for name in DELIVERED if (c20.ROUND_DIR / name).exists()},
        "updated_deliverables": {
            name: {"path": f"runs/comm_cost_objective_design_20260926/{name}",
                   "bytes": (compliance_dir / name).stat().st_size,
                   "sha256": c20.sha256_file(compliance_dir / name)}
            for name in UPDATED if (compliance_dir / name).exists()},
        "quarantined": {path.name: c20.sha256_file(path)
                        for path in sorted((c20.ROUND_DIR / "rejected").iterdir())}
        if (c20.ROUND_DIR / "rejected").is_dir() else {},
        "spec_item_verification": {"all_checks_pass": specs["all_checks_pass"],
                                   "step_total": specs["step_total_provenance"]["value_quoted_in_round18"],
                                   "laggards_resolved": len(
                                       specs["targets_that_were_detected_but_not_on_time"]),
                                   "per_seed_labels_verified":
                                       specs["per_seed_difference_check"]["matches_round18_labels"]},
        "remote_verification": {
            "successful_remote_contacts": 0,
            "note": "github.com:443 unreachable from this host; offline bundle transfer used "
                    "instead; see REMOTE_RETRY_LOG.md"},
        "stage_a_passed": bool(results["passed"] and specs["all_checks_pass"])}
    c20.write_json("MANIFEST.json", payload)
    print(json.dumps({"stage_a_passed": payload["stage_a_passed"],
                      "tests": results["tests"]["test_count"],
                      "new_deliverables": len(payload["new_deliverables"]),
                      "quarantined": len(payload["quarantined"]),
                      "head": payload["head_at_manifest"][:7]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
