#!/usr/bin/env python
"""Two read-only identity checks for the ROUND75 S1G pre-charge hash ABI."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import r75_s1g_outer_exec


HERE = Path(r75_s1g_outer_exec.__file__).resolve().parent
REPO = HERE
while not (REPO / ".git").exists():
    REPO = REPO.parent
PLAN_PATH = HERE / "ROUND75_S1G_AR1_RUN_PLAN.json"
RUNTIME_DIR = HERE / "raw" / "runtime"


class Round75S1GPrechargeHashTests(unittest.TestCase):
    def test_real_precharge_baseline_identity_matches(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        record = r75_s1g_outer_exec.verify_pre_charge(REPO, HERE, RUNTIME_DIR, plan)
        self.assertEqual(record["problems"], [])
        for name, expected in r75_s1g_outer_exec.EXPECT_CODE.items():
            self.assertEqual(record["code_sha256"][name], expected)
        for role, (relative, expected) in r75_s1g_outer_exec.EXPECT_DATA.items():
            self.assertEqual(record["data_sha256"][role]["path"], relative)
            self.assertEqual(record["data_sha256"][role]["sha256"], expected)
        baseline = plan["s1g_execution_contract"]["ledger_baseline"]
        snapshot = record["ledger_zero_start"]
        self.assertEqual(snapshot["total_ledger_events"],
                         baseline["total_ledger_events"])
        self.assertEqual(snapshot["files"]["r75_total_ledger.jsonl"]["sha256"],
                         baseline["total_ledger_sha256"])
        self.assertTrue(snapshot["guard_ledger_reconciliation"]["ok"])
        self.assertTrue(snapshot["frozen_snapshot_match"] is None)

    def test_wrong_declared_hash_is_reported_without_key_error(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(plan)
        tampered["s1g_execution_contract"]["ledger_baseline"]["total_ledger_sha256"] = "0" * 64
        record = r75_s1g_outer_exec.verify_pre_charge(
            REPO, HERE, RUNTIME_DIR, tampered)
        self.assertIn("canonical ledger SHA256 does not match the S1G baseline",
                      record["problems"])
        self.assertTrue(record["ledger_zero_start"]["guard_ledger_reconciliation"]["ok"])


if __name__ == "__main__":
    unittest.main()
