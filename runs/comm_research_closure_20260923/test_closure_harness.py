"""Unit tests for this round's harness. 0 episodes: nothing here opens an environment.

The tests exist because two earlier rounds found that a reader-side bug can invalidate
evidence while leaving every published number looking plausible: a bucket that should be
"not measurable" turning into 0, a tripwire wrapping nothing, two implementations of one
count disagreeing. Each of those has a test here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import closure_common as cc
import closure_bottleneck as bottleneck
import closure_chain as chain
import closure_stages as stages


class Tripwire(unittest.TestCase):
    def test_wraps_every_entry_point(self) -> None:
        report = cc.install_env_tripwire()
        self.assertEqual(len(report), len(cc.FORBIDDEN_CALLS))
        for entry in report:
            self.assertIn(".", entry)

    def test_calling_an_entry_point_raises_and_logs(self) -> None:
        import pressure_env  # the audited round's runner, resolved from SOURCE_ROUNDS

        cc.install_env_tripwire()
        before = cc.tripwire_report()["blocked_call_count"]
        with self.assertRaises(RuntimeError):
            pressure_env.run_episode()
        self.assertEqual(cc.tripwire_report()["blocked_call_count"], before + 1)

    def test_this_rounds_modules_win_name_resolution(self) -> None:
        cc.enter_source_round()
        self.assertEqual(sys.path[0], str(cc.ROUND_DIR))
        self.assertEqual(Path(chain.__file__).parent, cc.ROUND_DIR)
        self.assertEqual(Path(stages.__file__).parent, cc.ROUND_DIR)


class Sandbox(unittest.TestCase):
    def test_refuses_paths_outside_the_round(self) -> None:
        for name in ("../evil.json", "nested/evil.json", "/tmp/evil.json"):
            with self.assertRaises(ValueError):
                cc.write_json(name, {})

    def test_refuses_to_rewrite_delivered_content(self) -> None:
        cc.write_json("_test_round_artefact.json", {"a": 1})
        with self.assertRaises(RuntimeError):
            cc.write_json("_test_round_artefact.json", {"a": 2})
        cc.write_json("_test_round_artefact.json", {"a": 1})
        (cc.ROUND_DIR / "_test_round_artefact.json").unlink()


class Statistics(unittest.TestCase):
    def test_quantiles_of_empty_list_are_none_not_zero(self) -> None:
        empty = cc.quantiles([])
        self.assertEqual(empty["n"], 0)
        for key in ("min", "median", "max", "mean"):
            self.assertIsNone(empty[key])

    def test_ratio_guarded_against_zero_denominator(self) -> None:
        self.assertIsNone(cc.ratio(5, 0))
        self.assertEqual(cc.ratio(1, 4), 0.25)


def _record(event_type: str, target: str, stamp: float, status: str | None = None,
            detail: dict | None = None, **fields) -> dict:
    body = {"target_id": target, "event_type": event_type, "timestamp": stamp,
            "assigned_agents": fields.get("agents", []), "status": status,
            "progress": fields.get("progress"), "detail": detail or {}}
    return body


class Classification(unittest.TestCase):
    GOAL = {"detected_at": 10.0, "deadline": 200.0, "first_en_route": 12.0,
            "first_assigned": 20.0, "first_in_progress": 22.0, "completed_at": 60.0,
            "expired_seen": False, "max_progress": 1.0, "goal_level_assignments": 3,
            "rejections": 0, "final_status": "completed"}

    def test_on_time_when_completed_before_deadline(self) -> None:
        self.assertEqual(chain.classify(dict(self.GOAL), None, False), "c4_complete_on_time")

    def test_late_completion_is_not_counted_as_on_time(self) -> None:
        goal = dict(self.GOAL, completed_at=210.0)
        self.assertEqual(chain.classify(goal, None, False), "c4_complete_late")

    def test_never_dispatched_and_worked_are_different_classes(self) -> None:
        idle = dict(self.GOAL, first_en_route=None, first_assigned=None,
                    first_in_progress=None, completed_at=None, expired_seen=True)
        partial = dict(self.GOAL, completed_at=None, expired_seen=True)
        self.assertEqual(chain.classify(idle, None, False), "c3_known_never_dispatched")
        self.assertEqual(chain.classify(partial, None, False), "c3_worked_but_incomplete")

    def test_information_class_only_where_the_uplink_is_the_only_channel(self) -> None:
        goal = dict(self.GOAL, first_en_route=None, first_assigned=None,
                    first_in_progress=None, completed_at=None, expired_seen=True)
        comm = {"comm_first_known_step": None}
        self.assertEqual(chain.classify(goal, comm, True), "c2_info_never_arrived")
        # On a deployed path the same missing record must not become an information loss.
        self.assertEqual(chain.classify(goal, comm, False), "c3_known_never_dispatched")

    def test_information_arriving_after_the_deadline_is_its_own_class(self) -> None:
        goal = dict(self.GOAL, first_en_route=None, first_assigned=None,
                    first_in_progress=None, completed_at=None, expired_seen=True)
        self.assertEqual(chain.classify(goal, {"comm_first_known_step": 250.0}, True),
                         "c2_info_arrived_after_deadline")


class Decomposition(unittest.TestCase):
    ARM = {"goals_placed": 100, "not_discovered": 4, "on_time_total": 70,
           "per_goal_classes": {"c2_info_never_arrived": 6, "c3_worked_but_incomplete": 20,
                                "c4_complete_on_time": 70},
           "per_goal_class_comm_facts": {}}

    def test_unmeasurable_information_bucket_stays_none(self) -> None:
        parts = bottleneck.decompose(dict(self.ARM), information_attributable=False)
        self.assertIsNone(parts["2_information_loss"])
        self.assertIsNotNone(parts["3_allocation_or_execution_loss"])
        self.assertIsNone(parts["loss_share_of_placed"]["information"])

    def test_attributable_bucket_is_the_sum_of_its_classes(self) -> None:
        parts = bottleneck.decompose(dict(self.ARM), information_attributable=True)
        self.assertEqual(parts["2_information_loss"], 6)
        self.assertEqual(parts["3_allocation_or_execution_loss"], 20)


class ParsedLogs(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name: str, records: list[dict]) -> Path:
        path = self.dir / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                        encoding="utf-8")
        return path

    def test_lifecycle_stage_times_and_deadline(self) -> None:
        path = self.write("rp_x.jsonl", [
            _record("run_config", "__run__", 0.0,
                    detail={"configuration": {"num_goals": 1}}),
            _record("detection", "t-0", 5.0, detail={"deadline_offset": 100.0,
                                                     "total_stages": 2}),
            _record("progress", "t-0", 7.0, status="en_route"),
            _record("progress", "t-0", 20.0, status="in_progress", progress=0.4),
            _record("progress", "t-0", 40.0, status="completed", progress=1.0),
            _record("final", "t-0", 1000.0, status="completed",
                    detail={"deadline": 105.0}),
            _record("final", "__run__", 1000.0, status="run_finished"),
        ])
        data = chain.chain_lifecycle(path)
        goal = data["goals"]["t-0"]
        self.assertEqual(data["placed"], 1)
        self.assertEqual((goal["first_en_route"], goal["first_in_progress"]), (7.0, 20.0))
        self.assertEqual(goal["deadline"], 105.0)
        self.assertEqual(data["metrics"]["on_time"], 1)

    def test_transcript_identifies_the_first_receiver_and_closes_the_loss_ledger(self) -> None:
        path = self.write("ct_x.jsonl", [
            {"kind": "event", "step": 1, "target": "t-0", "revision": 1},
            {"kind": "send", "step": 3, "target": "t-0", "first": True, "cost_bytes": 2400,
             "packets": 4, "packets_lost": 2, "attempts": 6},
            {"kind": "deliver", "step": 3, "target": "t-0", "agent": "2", "accepted": True},
            {"kind": "deliver", "step": 4, "target": "t-0", "agent": "5", "accepted": True},
            {"kind": "deliver", "step": 9, "target": "t-0", "agent": "1", "accepted": False,
             "reason": "stale"},
            {"kind": "refused", "step": 30, "target": "t-0", "reason": "budget_exhausted",
             "cost_bytes": 2400},
        ])
        goal = chain.chain_transcript(path)["t-0"]
        self.assertEqual(goal["comm_first_known_step"], 3.0)
        self.assertEqual(goal["comm_robots_informed"], 2)
        self.assertEqual(goal["comm_refusals"], 1)
        self.assertEqual(goal["comm_packets_arrived"] + goal["comm_packets_lost"], 6)

    def test_a_truncated_file_fails_loudly(self) -> None:
        path = self.dir / "broken.jsonl"
        path.write_text('{"kind": "event"}\n{"kind": "sen', encoding="utf-8")
        with self.assertRaises(RuntimeError):
            list(cc.iter_jsonl(path))


class AnchorsAndArtefacts(unittest.TestCase):
    def test_every_stage_anchor_resolves_in_the_current_tree(self) -> None:
        resolved = stages.resolve_anchors()
        self.assertEqual(len(resolved), 8)
        for entries in resolved.values():
            self.assertTrue(entries)

    def test_absence_claims_are_actually_checked(self) -> None:
        checks = stages.check_absent_records()
        for claim, result in checks.items():
            self.assertTrue(result["claim_holds"], claim)
            self.assertGreater(result["scanned_files"], 20)

    def test_delivered_audit_reports_no_disagreement(self) -> None:
        path = cc.ROUND_DIR / "JOINT_SYSTEM_BOTTLENECK_AUDIT.json"
        if not path.exists():
            self.skipTest("aggregator has not run yet")
        document = json.loads(path.read_text(encoding="utf-8"))
        summary = document["reconciliation_summary"]
        self.assertEqual(summary["disagreeing"], 0)
        self.assertGreaterEqual(summary["checks"], 30)

    def test_delivered_verification_passed(self) -> None:
        path = cc.ROUND_DIR / "CLOSURE_VERIFICATION.json"
        if not path.exists():
            self.skipTest("verification has not run yet")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(document["passed"], json.dumps(document, ensure_ascii=False)[:600])
        self.assertEqual(document["env_tripwire"]["blocked_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
