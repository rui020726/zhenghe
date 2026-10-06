from __future__ import annotations

import sys
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import r45_data


class TeacherTimeBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy, cls.provenance = r45_data.load_teacher(
            r45_data.read_json(r45_data.FROZEN_R44_PATH)
        )

    def test_actual_rule_uses_discovery_tick_not_current_tick(self) -> None:
        # Each object intentionally exposes only the fields the production rule may read.
        items = (
            SimpleNamespace(target_id="early_at_late_current_tick", first_emission=True,
                            ready_step=147),
            SimpleNamespace(target_id="strict_boundary", first_emission=True,
                            ready_step=147),
            SimpleNamespace(target_id="late_discovery", first_emission=True,
                            ready_step=148),
            SimpleNamespace(target_id="already_emitted", first_emission=False,
                            ready_step=200),
        )
        observation = SimpleNamespace(items=items)

        # Hand-set expected set: strict greater-than excludes 147 and first_emission excludes
        # an already-emitted target. A naive current-tick > 147 label would wrongly include the
        # first item, whose current tick is 200 although it was ready at tick 147.
        self.assertEqual(
            self.policy.decide(observation),
            ("late_discovery",),
        )

    def test_tape_age_and_relay_tick_recover_teacher_ready_step(self) -> None:
        features = np.zeros(19, dtype=np.float32)
        features[0] = np.float32(53.0)
        features[5] = np.float32(1.0)
        features[18] = np.float32(0.2)
        record = {
            "step": 200,
            "targets": ["target"],
            "features": [features.tolist()],
        }
        (item,) = r45_data._teacher_items(record)
        self.assertEqual(item.ready_step, 147)
        self.assertEqual(self.policy.decide(SimpleNamespace(items=(item,))), ())

    def test_float32_encodes_all_episode_ticks_uniquely(self) -> None:
        encoded = np.asarray(
            [np.float32(step / 1000.0) for step in range(1, 1001)],
            dtype=np.float32,
        )
        self.assertEqual(len(set(encoded.tobytes()[i:i + 4]
                                 for i in range(0, encoded.nbytes, 4))), 1000)
        recovered = np.rint(encoded.astype(np.float64) * 1000.0).astype(np.int64)
        np.testing.assert_array_equal(recovered, np.arange(1, 1001))

    def test_policy_provenance_is_canonical_and_registered(self) -> None:
        self.assertEqual(self.provenance["rule_source_sha256"], r45_data.TEACHER_SHA256)
        self.assertEqual(
            self.provenance["rule_source_path"],
            str(r45_data.TEACHER_PATH.resolve()),
        )
        self.assertEqual(
            self.provenance["policy_parameters"],
            {"cut_steps": 147.0},
        )
        self.assertEqual(
            self.provenance["decision_fields_read_by_policy"],
            [
                "observation.items",
                "item.target_id",
                "item.first_emission",
                "item.ready_step",
            ],
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TeacherTimeBasisTests)
    test_names = [test.id() for test in suite]
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise SystemExit(1)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = {
        "kind": "round45-teacher-time-basis-regression/1",
        "status": "passed",
        "test_source_sha256": source_hash,
        "tests_run": test_result.testsRun,
        "test_names": test_names,
        "failures": len(test_result.failures),
        "errors": len(test_result.errors),
        "environment_constructed": 0,
        "optimizer_step_calls": 0,
    }
    target = RUN_DIR / "TEACHER_REGRESSION.json"
    body = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != body:
            raise RuntimeError("existing teacher regression receipt differs")
    else:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
            handle.flush()
