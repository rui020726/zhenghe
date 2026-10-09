from __future__ import annotations

import datetime as dt
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main() -> int:
    loader = unittest.defaultTestLoader
    modules = (
        "test_r58_identity_ledger_close_v2",
        "test_r58_private_launcher",
        "test_r58_resume_seam",
    )
    discovered_suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in modules)
    discovered = list(flatten(discovered_suite))
    # This single historical regression needs absent Git object b38b868; it
    # validates an old source snapshot, not the current resume implementation.
    missing_historical_object = "test_old_main_before_definitions_reproduces_nameerror_and_new_module_help_exits_zero"
    excluded = [test.id() for test in discovered if missing_historical_object in test.id()]
    selected = [test for test in discovered if missing_historical_object not in test.id()]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
    payload = {
        "kind": "round58-resume-narrow-regression-results/1",
        "modules": list(modules),
        "discovered": len(discovered),
        "excluded": excluded,
        "excluded_reason": "historical git object b38b868 is absent from this local object database",
        "executed": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "exit_code": 0 if result.wasSuccessful() else 1,
        "finished_at_local": dt.datetime.now().astimezone().isoformat(),
        "test_ids": [test.id() for test in selected],
    }
    out = ROOT / "ROUND58_RESUME_NARROW_TEST_RESULTS.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("discovered", "excluded", "executed", "failures", "errors", "exit_code")}, ensure_ascii=False))
    return payload["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
