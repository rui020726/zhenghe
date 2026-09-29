"""ROUND28 预检记录件：零环境合成测试跑成可引用机读记录（0 启动）。

This round deliberately does NOT re-run the production unittest suite: ROUND27's suite run did
construct short simulators outside its cell ledger, and re-running it would add more unbooked
simulator constructions. That scope is registered in ROUND27_SCOPE_REGISTRATION.json instead.
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from typing import Any

import r28_common as common


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten(item))
        else:
            tests.append(item)
    return tests


def run_suite() -> dict[str, Any]:
    import r28_preflight

    suite = unittest.defaultTestLoader.loadTestsFromModule(r28_preflight)
    names = sorted(test.id().split(".", 1)[1] for test in _flatten(suite))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {
        "round": "28",
        "section": "zero-environment, zero-optimizer preflight (synthetic only, 0 launches)",
        "launches": 0,
        "production_suite_rerun": False,
        "production_suite_note": "not re-run on purpose: ROUND27's suite run constructed short "
                                 "simulators outside its cell ledger (ROUND27_SCOPE_REGISTRATION."
                                 "json); re-running would add more unbooked constructions",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "test_names": names,
        "required_six_for_the_reused_interceptor": {
            "only_named_target": "TestReusedInterceptor.test_only_named_target_is_hit",
            "disabled_delegates": "TestReusedInterceptor.test_disabled_delegates_verbatim",
            "already_sent_no_duplicate": "TestReusedInterceptor.test_already_sent_not_forced_again",
            "sequential_budget": "TestReusedInterceptor.test_sequential_budget_exact",
            "no_legal_opportunity": "TestReusedInterceptor.test_no_legal_opportunity_no_bypass",
            "no_rng_or_action_change": "TestReusedInterceptor.test_no_rng_consumption",
        },
        "selection_algorithm_tests": [
            "TestSelectionAlgorithm.test_hold_at_first_payable_opportunity_is_a_point",
            "TestSelectionAlgorithm.test_send_at_first_payable_opportunity_is_excluded",
            "TestSelectionAlgorithm.test_never_payable_is_excluded_and_masked_steps_are_skipped",
            "TestSelectionAlgorithm.test_never_discovered_is_excluded_without_being_a_candidate",
            "TestSelectionAlgorithm.test_detected_but_never_offered_is_excluded_as_terminal",
            "TestSelectionAlgorithm.test_first_payable_beats_earlier_masked_steps",
            "TestSelectionAlgorithm.test_cap_is_enforced",
            "TestSelectionAlgorithm.test_goal_id_space_guard",
        ],
    }


def main() -> None:
    payload = run_suite()
    common.write_json("RUN_PREFLIGHT.json", payload)
    print("tests:", payload["tests_run"], "failures:", payload["failures"],
          "errors:", payload["errors"], "passed:", payload["passed"])


if __name__ == "__main__":
    main()
