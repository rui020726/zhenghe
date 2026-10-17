"""R1 protocol-history source review: stdlib-only cases; never run by A0 preparation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import r76_b_contracts as contracts
import r76_episode_protocol as protocol


HERE = Path(__file__).resolve().parent
FEATURE_WIDTH = 19


def _record(targets: list[str], one_shot_state: dict[str, str]) -> dict:
    return {
        "schema": protocol.TAPE_SCHEMA,
        "comm_action_semantics": "one-shot-send-skip/1",
        "comm_action_eligibility": "first-report-only/1",
        "base_action_mask_schema": "budget-sequential-mask/1",
        "step": 0,
        "targets": list(targets),
        "features": [[0.0] * FEATURE_WIDTH for _ in targets],
        "actions": {target: protocol.SEND for target in targets},
        "effective_actions": {target: protocol.SEND for target in targets},
        "requested": {target: True for target in targets},
        "decision_class": {
            target: "free_one_shot_choice" for target in targets
        },
        "free_one_shot_choices": list(targets),
        "free_first_reports": list(targets),
        "free": list(targets),
        "actionable": list(targets),
        "masked": [],
        "masked_reasons": {},
        "fixed_hold_one_shot_closed": [],
        "fixed_hold_refresh": [],
        "fixed_hold_reasons": {},
        "forced": [],
        "attempt_charge_bytes": {target: 100 for target in targets},
        "send_executable_initial": {target: True for target in targets},
        "send_block_reason_initial": {target: None for target in targets},
        "candidate_audit": {
            target: {"attempt_charge_bytes": 100} for target in targets
        },
        "relay_observation_remaining_bytes": 0.0,
        "shadow_budget_before": 100.0,
        "shadow_budget_after": 0.0,
        "shadow_capacity_before": 1,
        "shadow_capacity_after": 0,
        "log_prob": 0.0,
        "value": 0.0,
        "cost_value": 0.0,
        "send_probabilities": {target: 0.75 for target in targets},
        "one_shot_state_after_step": dict(one_shot_state),
    }


class R76ProtocolHistoryR1(unittest.TestCase):
    def test_1_historical_target_passes_and_unknown_future_target_rejects(self):
        record = _record(
            ["target-2"],
            {"target-1": "send_now_chosen", "target-2": "undecided"})
        self.assertEqual(
            protocol.validate_tape_record(
                record, historical_target_ids={"target-1"}),
            record)
        future = _record(["target-2"], {"target-3": "undecided"})
        with self.assertRaises(protocol.EpisodeProtocolError):
            protocol.validate_tape_record(
                future, record_index=1, historical_target_ids={"target-1"})

    def test_2_state_enum_and_existing_checks_are_preserved(self):
        allowed_states = (
            "undecided", "send_now_chosen", "skip_episode_chosen",
            "closed_by_prior_real_emission")
        for state in allowed_states:
            record = _record(["target-1"], {"target-1": state})
            self.assertEqual(protocol.validate_tape_record(record), record)
        with self.assertRaises(protocol.EpisodeProtocolError):
            protocol.validate_tape_record(
                _record(["target-1"], {"target-1": "invalid_state"}))
        with self.assertRaises(protocol.EpisodeProtocolError):
            protocol.validate_tape_record(
                _record(["target-2"], {"target-3": "undecided"}),
                historical_target_ids={"target-1"})
        wrong_actions = _record(["target-1"], {"target-1": "undecided"})
        wrong_actions["effective_actions"] = {"target-1": protocol.HOLD}
        with self.assertRaises(protocol.EpisodeProtocolError):
            protocol.validate_tape_record(wrong_actions)
        wrong_features = _record(["target-1"], {"target-1": "undecided"})
        wrong_features["features"] = [[0.0] * (FEATURE_WIDTH - 1)]
        with self.assertRaises(protocol.EpisodeProtocolError):
            protocol.validate_tape_record(wrong_features)
        future_pollution = [
            _record(["target-2"], {"target-2": "send_now_chosen"}),
            _record(["target-1"], {"target-2": "send_now_chosen"}),
        ]
        future_pollution[0]["step"] = 20
        future_pollution[1]["step"] = 10
        with self.assertRaises(protocol.EpisodeProtocolError):
            protocol.write_tape(
                HERE / "raw" / "r1_fixture_temp" / "future-pollution.jsonl",
                future_pollution)

    def test_3_json_round_trip_preserves_episode_level_state_history(self):
        records = [
            _record(["target-1"], {"target-1": "send_now_chosen"}),
            _record(
                ["target-2"],
                {"target-1": "send_now_chosen", "target-2": "skip_episode_chosen"}),
        ]
        fixture_root = (HERE / "raw" / "r1_fixture_temp").resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r76_r1_", dir=fixture_root) as temporary:
            self.assertEqual(Path(temporary).resolve().parent, fixture_root)
            path = Path(temporary) / "history.jsonl"
            records[1]["step"] = 1
            self.assertEqual(protocol.write_tape(path, records), str(path))
            self.assertEqual(protocol.read_tape(path), records)

    def test_4_pure_pre_model_validation_of_frozen_r71_inputs(self):
        total_records = 0
        samples_n = 0
        effective_feature_rows = 0
        max_targets_per_record = 0
        episode_count = 0
        episode_reports = []
        for frozen in contracts.EPISODES:
            row_path = contracts.RUN_ROOT / frozen["row_path"]
            tape_path = contracts.RUN_ROOT / frozen["tape_path"]
            self.assertEqual(row_path.stat().st_size, int(frozen["row_bytes"]))
            self.assertEqual(
                hashlib.sha256(row_path.read_bytes()).hexdigest(),
                frozen["row_sha256"])
            self.assertEqual(tape_path.stat().st_size, int(frozen["tape_bytes"]))
            self.assertEqual(
                hashlib.sha256(tape_path.read_bytes()).hexdigest(),
                frozen["tape_sha256"])
            row = protocol.read_row(row_path)
            tape = protocol.read_tape(tape_path)
            self.assertEqual(int(row["training_seed"]), 11)
            self.assertEqual(int(row["episode_index"]), int(frozen["episode"]))
            self.assertEqual(int(row["scenario_seed"]), int(frozen["scenario"]))
            self.assertEqual(int(row["attempt"]), int(frozen["attempt"]))
            self.assertEqual(float(row["lambda"]), 1.0)
            for record in tape:
                targets = record["targets"]
                features = record["features"]
                self.assertEqual(len(features), len(targets))
                self.assertLessEqual(len(targets), 6)
                self.assertTrue(
                    all(len(feature) == FEATURE_WIDTH for feature in features))
                effective_feature_rows += len(features)
                samples_n += 1
                max_targets_per_record = max(max_targets_per_record, len(targets))
            total_records += len(tape)
            protocol.PersistedCollector(tape)
            episode_count += 1
            episode_reports.append({
                "episode": int(frozen["episode"]),
                "scenario_seed": int(frozen["scenario"]),
                "attempt_expected": int(frozen["attempt"]),
                "attempt_observed": int(row["attempt"]),
                "row_path": frozen["row_path"],
                "row_bytes": int(row_path.stat().st_size),
                "row_sha256": frozen["row_sha256"],
                "row_sha256_match": True,
                "identity_match": True,
                "tape_path": frozen["tape_path"],
                "tape_bytes": int(tape_path.stat().st_size),
                "tape_sha256": frozen["tape_sha256"],
                "tape_sha256_match": True,
                "M": len(tape),
                "N": sum(1 for _ in tape),
                "T": sum(len(record["targets"]) for record in tape),
                "max_candidates": max(len(record["targets"]) for record in tape),
                "feature_width": FEATURE_WIDTH,
                "data_read_units": {
                    "row_deserializations": 1,
                    "tape_deserializations": 1,
                },
            })
        print(json.dumps({
            "kind": "round76-r1-protocol-history-report/1",
            "episodes": episode_reports,
            "summary": {
                "episode_count": episode_count,
                "M": total_records,
                "N": samples_n,
                "T": effective_feature_rows,
                "max_candidates": max_targets_per_record,
                "feature_width": FEATURE_WIDTH,
                "sample_semantics": "batch_samples emits one N sample per recorded tape row",
            },
        }, sort_keys=True, separators=(",", ":")))
        self.assertEqual(episode_count, 4)
        self.assertEqual(total_records, contracts.SAMPLES_PER_BATCH)
        self.assertEqual(samples_n, contracts.SAMPLES_PER_BATCH)
        self.assertLessEqual(total_records, contracts.MAX_UNIQUE_RECORDS)
        self.assertLessEqual(effective_feature_rows, contracts.MAX_SINGLE_TRUNK_ROWS)
        self.assertGreaterEqual(max_targets_per_record, 1)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
