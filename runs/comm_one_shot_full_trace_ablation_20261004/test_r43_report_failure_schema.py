"""Pure regressions for ROUND43's report-only failure-record rendering seam."""

from __future__ import annotations

import copy
import unittest

from r43_report import render_preflight_failure_correction


CURRENT_FAILURE_MESSAGE = "synthetic actor gate invoked an Adam optimizer update"


def current_record() -> dict:
    # Minimal redacted copy of ROUND43_PREFLIGHT_COUNTER_ASSERTION_C8.json's saved schema.
    return {
        "kind": "round43-preflight-gate-failure/1",
        "status": "preserved_superseded_assertion_failure",
        "failure": {
            "type": "RuntimeError",
            "message": CURRENT_FAILURE_MESSAGE,
            "source": "r43_preflight.py:_test_live_actor_path cumulative counter assertion",
            "cause": "the no-update check expected the process-lifetime Adam call counter to be "
                     "zero even though the committed batch0 already had 12 historical Adam calls",
        },
        "observed_at_failure": {
            "normal_claim_delta": 0,
            "environment_constructor_delta": 0,
            "environment_run_call_delta": 0,
            "ppo_update_delta": 0,
            "real_optimizer_updates": 0,
            "synthetic_optimizer_step_attempts": 1,
            "synthetic_optimizer_updates": 0,
            "checkpoint_serialization_writes": 0,
        },
        "post_failure_read_only_audit": {
            "preserved_artifacts_unchanged": True,
        },
    }


def legacy_record() -> dict:
    # The prior renderer contract used these two flat fields.
    record = current_record()
    record.pop("failure")
    record.pop("post_failure_read_only_audit")
    record["exception"] = "RuntimeError: legacy report contract"
    record["preserved_artifacts_unchanged"] = True
    return record


class ReportFailureSchemaTests(unittest.TestCase):
    def render(self, record: dict) -> dict:
        return render_preflight_failure_correction(
            record,
            source_path="failures/ROUND43_PREFLIGHT_COUNTER_ASSERTION_C8.json",
            source_sha256="a" * 64,
        )

    def test_current_saved_schema_reproduces_old_keyerror_and_renders(self) -> None:
        record = current_record()
        original = copy.deepcopy(record)

        # The exact previous report accesses that caused the observed KeyError.
        with self.assertRaisesRegex(KeyError, "exception"):
            _ = record["exception"]
        with self.assertRaisesRegex(KeyError, "preserved_artifacts_unchanged"):
            _ = record["preserved_artifacts_unchanged"]

        rendered = self.render(record)
        self.assertEqual(rendered["schema_layout"], "nested_failure_and_post_failure_audit")
        self.assertEqual(rendered["exception"], {
            "type": "RuntimeError",
            "message": "synthetic actor gate invoked an Adam optimizer update",
            "source": "r43_preflight.py:_test_live_actor_path cumulative counter assertion",
            "cause": "the no-update check expected the process-lifetime Adam call counter to be "
                     "zero even though the committed batch0 already had 12 historical Adam calls",
        })
        self.assertEqual(rendered["execution_claims_delta"], 0)
        self.assertEqual(rendered["environment_constructors_delta"], 0)
        self.assertEqual(rendered["real_optimizer_updates"], 0)
        self.assertEqual(rendered["observed_at_failure"]["synthetic_optimizer_step_attempts"], 1)
        self.assertTrue(rendered["preserved_artifacts_unchanged"])
        self.assertEqual(record, original)

    def test_legacy_renderer_schema_remains_supported_without_guessing(self) -> None:
        rendered = self.render(legacy_record())
        self.assertEqual(rendered["schema_layout"], "legacy_exception_and_flat_preservation_flag")
        self.assertEqual(rendered["exception"], "RuntimeError: legacy report contract")
        self.assertTrue(rendered["preserved_artifacts_unchanged"])

    def test_failure_summary_is_operational_and_not_a_scientific_result(self) -> None:
        rendered = self.render(current_record())
        self.assertFalse({"scientific_result", "on_time_total", "spent_bytes_total"}
                         & set(rendered))
        self.assertEqual(rendered["classification"],
                         "preflight assertion assumed zero historical Adam calls; "
                         "fixed to assert zero delta")

    def test_unknown_or_missing_failure_schema_fails_closed(self) -> None:
        missing = current_record()
        missing.pop("failure")
        missing.pop("post_failure_read_only_audit")
        with self.assertRaisesRegex(ValueError, "neither supported failure field"):
            self.render(missing)

        unknown = current_record()
        unknown["kind"] = "round43-preflight-gate-failure/999"
        with self.assertRaisesRegex(ValueError, "unsupported.*kind"):
            self.render(unknown)

        unknown_status = current_record()
        unknown_status["status"] = "new-unregistered-status"
        with self.assertRaisesRegex(ValueError, "unsupported.*status"):
            self.render(unknown_status)

        ambiguous = current_record()
        ambiguous["exception"] = "also present"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.render(ambiguous)

    def test_missing_fields_and_wrong_types_fail_closed(self) -> None:
        no_message = current_record()
        del no_message["failure"]["message"]
        with self.assertRaisesRegex(ValueError, "failure.message"):
            self.render(no_message)

        unknown_failure_field = current_record()
        unknown_failure_field["failure"]["details"] = "unregistered field"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            self.render(unknown_failure_field)

        wrong_counter = current_record()
        wrong_counter["observed_at_failure"]["normal_claim_delta"] = True
        with self.assertRaisesRegex(ValueError, "normal_claim_delta"):
            self.render(wrong_counter)

        wrong_audit = current_record()
        wrong_audit["post_failure_read_only_audit"]["preserved_artifacts_unchanged"] = "yes"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.render(wrong_audit)

        with self.assertRaisesRegex(ValueError, "SHA256 is malformed"):
            render_preflight_failure_correction(
                current_record(), source_path="failure.json", source_sha256="not-a-hash")


if __name__ == "__main__":
    unittest.main()
