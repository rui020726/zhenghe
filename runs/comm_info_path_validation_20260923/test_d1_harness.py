"""Tests for the D1 round harness. 0 episodes: nothing here builds a run.

The bugs these guard against are the ones this project has already paid for: two cells
sharing one filename, a cap enforced by reading somebody else's ledger, a "paired" label
on a comparison that is not paired, and an unmeasurable quantity quietly reported as zero.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import d1_common as dc
import d1_gate as gate
import d1_run as runner

from integrated_sar.info_probe import behaviour_view


class ArmTable(unittest.TestCase):
    def test_each_contrast_moves_exactly_the_declared_fields(self) -> None:
        for name, left, right in dc.CONTRASTS:
            a = dict(dc.base_configuration_fields())
            a.update(dc.ARM_OVERRIDES[left])
            b = dict(dc.base_configuration_fields())
            b.update(dc.ARM_OVERRIDES[right])
            moved = {key for key in a if a[key] != b[key]}
            self.assertEqual(moved, dc.CONTRAST_FIELDS[name],
                             f"{name} is supposed to be a single-variable contrast")

    def test_silence_and_loss_are_declared_in_the_base_configuration(self) -> None:
        base = dc.base_configuration_fields()
        self.assertEqual(base["comm_policy"], "immediate")
        self.assertEqual(base["comm_delivery_probability"], 1.0)
        self.assertEqual(base["gossip_interval"], 2, "deployed path keeps line-of-sight and gossip")
        self.assertEqual(dc.ARM_OVERRIDES["A4"]["gossip_interval"], 100000)
        self.assertIsNone(base.get("comm_budget_bytes"),
                          "D1 does not test the budget; the credit stays unset")
        self.assertIsNone(base.get("comm_max_wait_steps"),
                          "with no bound declared, the timing mode cannot speak for an arm")


class Plans(unittest.TestCase):
    def test_launch_plan_stays_inside_the_frozen_cap(self) -> None:
        b0, smoke, data = runner.b0_plan(), runner.smoke_plan(), runner.data_plan()
        self.assertEqual(sum(1 for cell in b0 if cell["kind"] == "b0"), 16)
        self.assertEqual(sum(1 for cell in b0 if cell["kind"] == "mode"), 4)
        self.assertEqual(len(smoke), 4)
        self.assertEqual(len(data), 240)
        self.assertLessEqual(len(b0) + len(smoke) + len(data), dc.LAUNCH_CAP)

    def test_no_two_cells_share_an_artefact_path(self) -> None:
        seen: dict[str, str] = {}
        for cell in runner.b0_plan() + runner.smoke_plan() + runner.data_plan():
            for kind, path in dc.cell_paths(cell["tag"], cell["arm"], cell["seed"]).items():
                key = str(path)
                self.assertNotIn(key, seen,
                                 f"{cell} collides with {seen.get(key)} — data would be lost")
                seen[key] = str(cell)

    def test_every_data_cell_has_the_probe_on(self) -> None:
        self.assertTrue(all(cell["probe"] for cell in runner.data_plan()),
                        "the mechanism record needs F2/F3/F4 in every arm")


class Ledger(unittest.TestCase):
    def test_reservation_grows_and_respects_the_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original_logs, original_cap = dc.LOGS, dc.LAUNCH_CAP
            dc.LOGS, dc.LAUNCH_CAP = Path(directory), 2
            try:
                self.assertEqual(dc.ledger_reserve("test", "one"), 1)
                self.assertEqual(dc.ledger_reserve("test", "two"), 2)
                with self.assertRaises(RuntimeError):
                    dc.ledger_reserve("test", "three")
                self.assertEqual(dc.ledger_total(), 2, "a refused launch must not be booked")
            finally:
                dc.LOGS, dc.LAUNCH_CAP = original_logs, original_cap

    def test_each_process_writes_its_own_book(self) -> None:
        path = dc._ledger_path("data")
        self.assertIn(f"_p{__import__('os').getpid()}.json", path.name)


class Extractors(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name: str, records: list[dict]) -> Path:
        path = self.dir / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                         encoding="utf-8")
        return path

    def test_detection_sequence_is_order_normalised_and_exact(self) -> None:
        records = [
            {"event_type": "detection", "target_id": "g-1", "timestamp": 7.0,
             "discovered_by": "agent_0"},
            {"event_type": "detection", "target_id": "g-0", "timestamp": 3.0,
             "discovered_by": "agent_1"},
            {"event_type": "progress", "target_id": "g-0", "timestamp": 9.0},
        ]
        self.assertEqual(dc.detection_sequence(self.write("a.jsonl", records)),
                         [("g-0", 3.0, "agent_1"), ("g-1", 7.0, "agent_0")])

    def test_emission_sequence_numbers_reports_per_target_in_emission_order(self) -> None:
        records = [
            {"kind": "send", "step": 1, "target": "g-0", "subscribers": 6, "cost_bytes": 2514},
            {"kind": "send", "step": 2, "target": "g-0", "subscribers": 6, "cost_bytes": 2520},
            {"kind": "send", "step": 2, "target": "g-1", "subscribers": 6, "cost_bytes": 2400},
            {"kind": "refused", "step": 3, "target": "g-1", "reason": "budget_exhausted"},
        ]
        self.assertEqual(dc.emission_sequence(self.write("b.jsonl", records)),
                         [(1.0, "g-0", 6, 2514, 1), (2.0, "g-0", 6, 2520, 2),
                          (2.0, "g-1", 6, 2400, 1)])

    def test_packet_outcomes_keep_the_recipient_named(self) -> None:
        records = [
            {"kind": "delivery", "target_id": "g-0", "emission": 1, "robot_id": "3",
             "arrived": True},
            {"kind": "delivery", "target_id": "g-0", "emission": 1, "robot_id": "4",
             "arrived": False},
            {"kind": "knowledge", "target_id": "g-0", "robot_id": "3", "step": 1},
        ]
        self.assertEqual(dc.packet_outcomes(self.write("c.jsonl", records)),
                         [("g-0", 1, "3", True), ("g-0", 1, "4", False)])

    def test_counting_difference_is_detected_not_absorbed(self) -> None:
        left = {"airtime_bytes": 100, "emissions": 4, "packets_lost": 0, "on_time": 6,
                "first_emissions": 4, "refresh_emissions": 0, "transmission_attempts": 24,
                "packets_delivered": 24, "delivery_refused": 0, "budget_refused_sends": 0,
                "capacity_refused_sends": 0, "events_accepted": 4, "status_counts": {}}
        right = dict(left, packets_lost=3, packets_delivered=21)
        self.assertEqual(sorted(gate.counters_equal(left, right)),
                         sorted(["packets_lost", "packets_delivered"]))
        self.assertEqual(gate.counters_equal(left, dict(left)), [])

    def test_configuration_without_probe_path_still_names_the_probe_field(self) -> None:
        fields = dc.behaviour_fields({"output": "x", "info_probe": None, "steps": 1000})
        self.assertEqual(fields, {"steps": 1000})


class DeliveredArtefacts(unittest.TestCase):
    def test_the_b0_gate_that_unlocked_the_data_stage_actually_passed(self) -> None:
        path = dc.ROUND_DIR / "B0_PAIRED_REPLAY.json"
        if not path.exists():
            self.skipTest("the gate has not been judged yet")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(document["passed"], json.dumps(document)[:400])
        self.assertEqual(document["launches_booked_so_far"]["used"], 20)
        for check in document["b0_replay"]["pairs"]:
            self.assertEqual(check["counter_differences"], [])
            self.assertEqual(check["probe_anomalies"], 0)

    def test_behaviour_view_drops_only_declarations(self) -> None:
        record = {"event_type": "final", "target_id": "g-0", "timestamp": 4.0,
                  "status": "completed", "detail": {"deadline": 90.0}}
        self.assertEqual(behaviour_view(record), record)


if __name__ == "__main__":
    unittest.main()
