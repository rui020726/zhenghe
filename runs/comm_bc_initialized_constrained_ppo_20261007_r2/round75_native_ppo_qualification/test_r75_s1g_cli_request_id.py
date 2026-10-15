#!/usr/bin/env python
"""Three fixed CLI-routing checks for the ROUND75 S1G outer entry point."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import r75_s1g_outer_exec


HERE = Path(r75_s1g_outer_exec.__file__).resolve().parent
REQUEST_ID = "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"
FIXTURE_PARENT = HERE / "_r75_s1g_cli_request_id_test_fixture"
RUNTIME_DIR = HERE / "raw" / "runtime"


class _Sentinel(Exception):
    pass


class Round75S1GCliRequestIdTests(unittest.TestCase):
    def test_launch_without_explicit_id_is_refused(self):
        run_calls = []
        prepare_calls = []
        original_run = r75_s1g_outer_exec.run
        original_prepare = r75_s1g_outer_exec.prepare

        def run_spy(*args, **kwargs):
            run_calls.append((args, kwargs))
            raise AssertionError("run must not be called for a CLI refusal")

        def prepare_spy(*args, **kwargs):
            prepare_calls.append((args, kwargs))
            raise AssertionError("prepare must not be called for a CLI refusal")

        r75_s1g_outer_exec.run = run_spy
        r75_s1g_outer_exec.prepare = prepare_spy
        try:
            self.assertEqual(r75_s1g_outer_exec.main(["--launch"]), 2)
        finally:
            r75_s1g_outer_exec.run = original_run
            r75_s1g_outer_exec.prepare = original_prepare
        self.assertEqual(run_calls, [])
        self.assertEqual(prepare_calls, [])

    def test_explicit_launch_id_reaches_run(self):
        observed = []
        original_run = r75_s1g_outer_exec.run
        original_prepare = r75_s1g_outer_exec.prepare

        def run_spy(repo, here, attempt_dir, request_id, plan, runtime_dir):
            observed.append({
                "attempt_dir": Path(attempt_dir),
                "request_id": request_id,
                "plan": plan,
                "runtime_dir": Path(runtime_dir),
            })
            raise _Sentinel

        def prepare_spy(*args, **kwargs):
            raise AssertionError("prepare must not be called for --launch")

        r75_s1g_outer_exec.run = run_spy
        r75_s1g_outer_exec.prepare = prepare_spy
        try:
            FIXTURE_PARENT.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(
                    prefix="ar3_", dir=FIXTURE_PARENT) as temp:
                attempt_dir = Path(temp) / "attempt"
                argv = ["--launch", "--request-id", REQUEST_ID,
                        "--attempt-dir", str(attempt_dir)]
                with self.assertRaises(_Sentinel):
                    r75_s1g_outer_exec.main(argv)
                plan = json.loads(
                    (HERE / "ROUND75_S1G_AR1_RUN_PLAN.json").read_text(
                        encoding="utf-8"))
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0]["attempt_dir"], attempt_dir)
                self.assertEqual(observed[0]["request_id"], REQUEST_ID)
                self.assertEqual(observed[0]["plan"], plan)
                self.assertEqual(observed[0]["runtime_dir"], RUNTIME_DIR)
        finally:
            r75_s1g_outer_exec.run = original_run
            r75_s1g_outer_exec.prepare = original_prepare

    def test_prepare_without_id_uses_default(self):
        observed = []
        original_run = r75_s1g_outer_exec.run
        original_prepare = r75_s1g_outer_exec.prepare

        def run_spy(*args, **kwargs):
            raise AssertionError("run must not be called for --prepare")

        def prepare_spy(repo, here, attempt_dir, request_id, plan, runtime_dir):
            observed.append({
                "attempt_dir": Path(attempt_dir),
                "request_id": request_id,
                "plan": plan,
                "runtime_dir": Path(runtime_dir),
            })
            raise _Sentinel

        r75_s1g_outer_exec.run = run_spy
        r75_s1g_outer_exec.prepare = prepare_spy
        try:
            FIXTURE_PARENT.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(
                    prefix="ar3_prepare_", dir=FIXTURE_PARENT) as temp:
                attempt_dir = Path(temp) / "attempt"
                argv = ["--prepare", "--attempt-dir", str(attempt_dir)]
                with self.assertRaises(_Sentinel):
                    r75_s1g_outer_exec.main(argv)
                plan = json.loads(
                    (HERE / "ROUND75_S1G_AR1_RUN_PLAN.json").read_text(
                        encoding="utf-8"))
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0]["attempt_dir"], attempt_dir)
                self.assertEqual(observed[0]["request_id"],
                                 r75_s1g_outer_exec.FOLLOW_UP_REQUEST_ID)
                self.assertEqual(observed[0]["plan"], plan)
                self.assertEqual(observed[0]["runtime_dir"], RUNTIME_DIR)
        finally:
            r75_s1g_outer_exec.run = original_run
            r75_s1g_outer_exec.prepare = original_prepare


if __name__ == "__main__":
    unittest.main()
