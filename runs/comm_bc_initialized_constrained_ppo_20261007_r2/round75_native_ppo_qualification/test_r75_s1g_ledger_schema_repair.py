"""Pure stdlib checks for the A-R5 ledger-snapshot schema repair."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTER = HERE / "r75_s1g_outer_exec.py"
FIXTURE_ROOT = (HERE / "raw" / "ar5_ledger_schema_fixture" /
                "R75_S1G_AR5_LEDGER_SCHEMA_READONLY_REPAIR_20261015")


def _load_module():
    spec = importlib.util.spec_from_file_location("r75_s1g_outer_exec_ar5", OUTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _guard_kind(unit):
    return "r75-" + unit.replace("_", "-") + "-guard/1"


def _guard_payload(unit, *, used_normal=0, used_reserve=0, refusals=0, events=()):
    return {"kind": _guard_kind(unit), "unit": unit,
            "used_normal": used_normal, "used_reserve": used_reserve,
            "refusals": refusals, "events": list(events)}


def _ledger_line(unit, event, category, value, what, after):
    return {"guard": _guard_kind(unit), "event": event, "unit": unit,
            "category": category, "n": value, "what": what, "used_after": after}


def _make_fixture(case_name, unit_payloads, ledger_lines):
    root = FIXTURE_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix=case_name + "_", dir=str(root))).resolve()
    expected = root.resolve()
    if case != expected and expected not in case.parents:
        raise AssertionError("AR5 fixture escaped the approved root: " + str(case))
    for unit, payload in unit_payloads.items():
        path = case / ("r75_" + unit + "_guard.json")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    ledger = case / "r75_total_ledger.jsonl"
    if ledger_lines is not None:
        ledger.write_text("".join((json.dumps(line) if isinstance(line, dict) else str(line)) + "\n"
                                  for line in ledger_lines),
                          encoding="utf-8", newline="\n")
    return case


class LedgerSchemaRepairTests(unittest.TestCase):
    def test_real_production_b2_artifacts_reconcile(self):
        module = _load_module()
        root = HERE / "raw" / "runtime"
        expected_files = {
            "r75_environment_construction.json": (288,
                "abd847cfce1c782c7d6a3832557d865083fcf0695e24aab002e54754a39182f6"),
            "r75_environment_construction_guard.json": (223,
                "be58a93de378a1fe9e05bc3f2d2cb7059f705b161ef59191c6aab09701c3981b"),
            "r75_joint_record.json": (273,
                "08544238cf7e06c57ee22c0d87589d27bff9c259a09eac185375e29e3e0cb1c3"),
            "r75_joint_record_guard.json": (739,
                "98b1117e75c87345d95d92a0acace928f6cc3b9c811d2f7ad109bfc1bf9a0463"),
            "r75_offline_trunk_target_row.json": (297,
                "5531991a84b76129693a9993eb511354ccfcde0d019b6f35a04b1ffb36244527"),
            "r75_offline_trunk_target_row_guard.json": (33643,
                "86ca5b82e2e6e9a0bf791df515f11a2a13750c9ced0b1948b0b654003ab674b7"),
            "r75_physical_optimizer_step.json": (288,
                "23a4022bb542fc9e5cef6d4f23cdef66f2795128c689beab904e5dbf054328aa"),
            "r75_physical_optimizer_step_guard.json": (2788,
                "4b8af23ecb3995a142d7e534574dfb15dd4a075ae9bda844b28ab47710ded9e7"),
            "r75_total_ledger.jsonl": (55011,
                "f6095bd1b8c9d95c6005ad368d0a69e6e9f7af25a222abf760bcba7455062124"),
        }
        snapshot = module.ledger_snapshot(root, None)
        self.assertEqual(snapshot["total_ledger_events"], 240)
        self.assertTrue(snapshot["guard_ledger_reconciliation"]["ok"])
        self.assertEqual(snapshot["per_unit_last_known"]
                         ["physical_optimizer_step"]["used_normal"], 10)
        self.assertEqual(snapshot["per_unit_last_known"]
                         ["physical_optimizer_step"]["used_reserve"], 4)
        self.assertEqual(snapshot["per_unit_last_known"]
                         ["offline_trunk_target_row"]["used_normal"], 2520)
        self.assertEqual(snapshot["per_unit_last_known"]
                         ["joint_record"]["used_normal"], 3)
        self.assertEqual(snapshot["per_unit_last_known"]
                         ["environment_construction"]["used_normal"], 0)
        self.assertEqual(snapshot["files"].get("r75_total_ledger.jsonl"),
                         {"bytes": 55011,
                          "sha256": "f6095bd1b8c9d95c6005ad368d0a69e6e9f7af25a222abf760bcba7455062124"})
        for name, (expected_bytes, expected_sha256) in expected_files.items():
            actual = snapshot["files"].get(name)
            self.assertIsNotNone(actual, name)
            self.assertEqual(actual["bytes"], expected_bytes, name)
            self.assertEqual(actual["sha256"], expected_sha256, name)
        print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))

    def test_interleaved_categories_use_unit_total_cumulative(self):
        module = _load_module()
        unit = "physical_optimizer_step"
        guard_events = [
            {"event": "charged", "what": "normal_2", "n": 2, "category": "normal",
             "used_after": 2},
            {"event": "charged", "what": "reserve_1", "n": 1, "category": "reserve",
             "used_after": 3},
            {"event": "charged", "what": "normal_1", "n": 1, "category": "normal",
             "used_after": 4},
            {"event": "charged", "what": "reserve_2", "n": 2, "category": "reserve",
             "used_after": 6},
        ]
        root = _make_fixture(
            "case2_interleaved_categories",
            {unit: _guard_payload(unit, used_normal=3, used_reserve=3,
                                  refusals=0, events=guard_events),
             "offline_trunk_target_row": _guard_payload("offline_trunk_target_row"),
             "joint_record": _guard_payload("joint_record"),
             "environment_construction": _guard_payload("environment_construction")},
            [_ledger_line(unit, "charged", "normal", 2, "normal_2", 2),
             _ledger_line(unit, "charged", "reserve", 1, "reserve_1", 3),
             _ledger_line(unit, "charged", "normal", 1, "normal_1", 4),
             _ledger_line(unit, "charged", "reserve", 2, "reserve_2", 6)])
        snapshot = module.ledger_snapshot(root, None)
        summary = snapshot["per_unit_last_known"][unit]
        self.assertTrue(snapshot["guard_ledger_reconciliation"]["ok"])
        self.assertEqual(summary["used_normal"], 3)
        self.assertEqual(summary["used_reserve"], 3)
        self.assertEqual(summary["used_after_normal"], 4)
        self.assertEqual(summary["used_after_reserve"], 6)

    def test_refusal_normalization_accepts_total_cumulative(self):
        module = _load_module()
        unit = "physical_optimizer_step"
        guard_events = [
            {"event": "charged", "what": "normal_step", "n": 3, "category": "normal",
             "used_after": 3},
            {"event": "refused", "what": "refused_step", "n": 1, "category": "normal",
             "used_after": 3},
        ]
        root = _make_fixture(
            "case3_refusal_normalization",
            {unit: _guard_payload(unit, used_normal=3, used_reserve=0,
                                  refusals=1, events=guard_events),
             "offline_trunk_target_row": _guard_payload("offline_trunk_target_row"),
             "joint_record": _guard_payload("joint_record"),
             "environment_construction": _guard_payload("environment_construction")},
            [_ledger_line(unit, "charged", "normal", 3, "normal_step", 3),
             _ledger_line(unit, "charge_refused", "normal", 1, "refused_step", 3)])
        snapshot = module.ledger_snapshot(root, None)
        summary = snapshot["per_unit_last_known"][unit]
        self.assertTrue(snapshot["guard_ledger_reconciliation"]["ok"])
        self.assertEqual(summary["used_normal"], 3)
        self.assertEqual(summary["refusals"], 1)
        self.assertEqual(summary["used_after_normal"], 3)

    def test_category_cumulative_used_after_is_refused(self):
        module = _load_module()
        unit = "physical_optimizer_step"
        guard_events = [
            {"event": "charged", "what": "normal_2", "n": 2, "category": "normal",
             "used_after": 2},
            {"event": "charged", "what": "reserve_1", "n": 1, "category": "reserve",
             "used_after": 3},
            {"event": "charged", "what": "normal_1", "n": 1, "category": "normal",
             "used_after": 4},
            {"event": "charged", "what": "reserve_2", "n": 2, "category": "reserve",
             "used_after": 3},
        ]
        root = _make_fixture(
            "case4_category_cumulative_used_after",
            {unit: _guard_payload(unit, used_normal=3, used_reserve=3,
                                  refusals=0, events=guard_events),
             "offline_trunk_target_row": _guard_payload("offline_trunk_target_row"),
             "joint_record": _guard_payload("joint_record"),
             "environment_construction": _guard_payload("environment_construction")},
            [_ledger_line(unit, "charged", "normal", 2, "normal_2", 2),
             _ledger_line(unit, "charged", "reserve", 1, "reserve_1", 3),
             _ledger_line(unit, "charged", "normal", 1, "normal_1", 4),
             _ledger_line(unit, "charged", "reserve", 2, "reserve_2", 6)])
        snapshot = module.ledger_snapshot(root, None)
        self.assertFalse(snapshot["guard_ledger_reconciliation"]["ok"])
        self.assertEqual(snapshot["guard_ledger_reconciliation"]["mismatches"][0]["unit"], unit)
        self.assertTrue(any("guard event 3 used_after=3 != total cumulative 6" in problem
                            for problem in snapshot["per_unit_last_known"][unit]["problems"]))

    def test_changed_what_breaks_event_sequence(self):
        module = _load_module()
        unit = "physical_optimizer_step"
        guard_events = [
            {"event": "charged", "what": "original_step", "n": 3, "category": "normal",
             "used_after": 3},
        ]
        root = _make_fixture(
            "case5_changed_what",
            {unit: _guard_payload(unit, used_normal=3, used_reserve=0,
                                  refusals=0, events=guard_events),
             "offline_trunk_target_row": _guard_payload("offline_trunk_target_row"),
             "joint_record": _guard_payload("joint_record"),
             "environment_construction": _guard_payload("environment_construction")},
            [_ledger_line(unit, "charged", "normal", 3, "changed_step", 3)])
        snapshot = module.ledger_snapshot(root, None)
        self.assertFalse(snapshot["guard_ledger_reconciliation"]["ok"])
        self.assertTrue(any("event sequences" in problem
                            for problem in snapshot["guard_ledger_reconciliation"]["mismatches"][0]
                            ["problems"]))

    def test_corrupt_ledger_line_fails_closed(self):
        module = _load_module()
        unit = "physical_optimizer_step"
        guard_events = [
            {"event": "charged", "what": "normal_step", "n": 3, "category": "normal",
             "used_after": 3},
        ]
        root = _make_fixture(
            "case6_corrupt_ledger_line",
            {unit: _guard_payload(unit, used_normal=3, used_reserve=0,
                                  refusals=0, events=guard_events),
             "offline_trunk_target_row": _guard_payload("offline_trunk_target_row"),
             "joint_record": _guard_payload("joint_record"),
             "environment_construction": _guard_payload("environment_construction")},
            [_ledger_line(unit, "charged", "normal", 3, "normal_step", 3),
             "{not-json"])
        snapshot = module.ledger_snapshot(root, None)
        summary = snapshot["per_unit_last_known"][unit]
        self.assertFalse(snapshot["guard_ledger_reconciliation"]["ok"])
        self.assertTrue(snapshot["unparsed_total_ledger_lines"])
        self.assertTrue(any(problem["unit"] == "ledger"
                            and any("unparsed" in text for text in problem["problems"])
                            for problem in snapshot["guard_ledger_reconciliation"]["mismatches"]))
        self.assertEqual(summary["used_normal"], "unknown")
        self.assertEqual(summary["used_reserve"], "unknown")
        self.assertEqual(summary["refusals"], "unknown")
        self.assertEqual(summary["charged_events"], "unknown")
        self.assertFalse(summary["schema_ok"])
        self.assertFalse(summary["guard_ledger_match"])
        self.assertEqual(summary["guard_events"], 1)
        self.assertEqual(summary["total_ledger_events"], 1)
        self.assertEqual(summary["used_after_normal"], 3)
        self.assertTrue(any("unparsed" in text for text in summary["problems"]))


if __name__ == "__main__":
    unittest.main()
