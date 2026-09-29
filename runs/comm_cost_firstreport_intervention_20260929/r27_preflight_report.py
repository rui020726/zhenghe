"""ROUND27 预检结果件：把零环境合成测试跑成一份可引用的机读记录（0 启动）。"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

import r27_common as common


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten(item))
        else:
            tests.append(item)
    return tests


def run_suite() -> dict:
    import r27_preflight

    suite = unittest.defaultTestLoader.loadTestsFromModule(r27_preflight)
    names = sorted(test.id().split(".", 1)[1] for test in _flatten(suite))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {
        "round": "27",
        "section": "zero-environment, zero-optimizer preflight (synthetic only)",
        "launches": 0,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "test_names": names,
        "stream_tail": stream.getvalue().splitlines()[-6:],
        "required_six": {
            "only_named_target": "TestOnlyNamedTarget.test_only_named_target_is_hit",
            "disabled_delegates": "TestDisableDelegation.test_disabled_delegates_verbatim",
            "already_sent_no_duplicate": "TestAlreadySent.test_already_sent_not_forced_again",
            "sequential_budget": ["TestSequentialBudget."
                                 "test_override_behind_later_candidate_keeps_shadow_exact",
                                 "TestSequentialBudget."
                                 "test_override_behind_earlier_send_keeps_shadow_exact"],
            "no_legal_opportunity": "TestNoLegalOpportunity."
                                    "test_masked_named_target_never_overridden",
            "no_rng_or_action_change": ["TestNoRngTouch.test_override_draws_no_randomness",
                                        "TestNoRngTouch."
                                        "test_hook_source_references_no_rng_objects"],
        },
    }


def main() -> None:
    payload = run_suite()
    common.write_json("RUN_PREFLIGHT.json", payload)
    print("tests:", payload["tests_run"], "failures:", payload["failures"],
          "errors:", payload["errors"], "passed:", payload["passed"])


if __name__ == "__main__":
    main()
