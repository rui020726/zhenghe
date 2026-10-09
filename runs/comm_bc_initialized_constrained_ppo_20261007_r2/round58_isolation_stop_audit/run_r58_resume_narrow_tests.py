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
        "test_r58_plan_identity_migration",
        "test_r58_train_pilot_restore_import",
    )
    discovered_suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in modules)
    discovered = list(flatten(discovered_suite))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(discovered))
    payload = {
        "kind": "round58-resume-narrow-regression-results/1",
        "modules": list(modules),
        "discovered": len(discovered),
        "excluded": [],
        "historical_git_objects": {
            "b38b8688dee7ffc2626aa2064c8af74af75e216b":
                "present; historical private-launcher and checkpoint-contract blobs are read by their exact repository paths",
        },
        "executed": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "exit_code": 0 if result.wasSuccessful() else 1,
        "finished_at_local": dt.datetime.now().astimezone().isoformat(),
        "test_ids": [test.id() for test in discovered],
    }
    out = ROOT / "ROUND58_RESUME_NARROW_TEST_RESULTS.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("discovered", "excluded", "executed", "failures", "errors", "exit_code")}, ensure_ascii=False))
    return payload["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
