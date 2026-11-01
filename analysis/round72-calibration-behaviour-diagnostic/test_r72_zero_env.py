# -*- coding: utf-8 -*-
"""Zero-environment tests for the ROUND72 diagnostic (synthetic data only).

No environment construction, no model load, no forward pass, no RNG use: every
test builds small synthetic tapes / row lists in memory and checks the pure
functions the audit relies on.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_boundary_audit import (ready_step, free_decisions, margin_block, summarise,
                                compare_cell, READY_CUT, HORIZON)


def row(step, age=0.0):
    f = [0.0] * 19
    f[0] = age
    f[5] = 1.0
    f[18] = step / 1000.0
    return f


def rec(step, targets, probs, actions, charges=None):
    feats = [row(step) for _ in targets]
    return {
        'step': step,
        'targets': list(targets),
        'features': feats,
        'actions': {t: a for t, a in zip(targets, actions)},
        'free': list(targets),
        'send_probabilities': {t: p for t, p in zip(targets, probs)},
        'candidate_audit': {t: {'attempt_charge_bytes': (charges or {}).get(t, 1000)}
                            for t in targets},
    }


class TestRouting(unittest.TestCase):
    def test_boundary_is_strictly_above_147(self):
        self.assertEqual(ready_step(row(147)), 147)
        self.assertEqual(ready_step(row(148)), 148)
        self.assertLessEqual(ready_step(row(147)), READY_CUT)
        self.assertGreater(ready_step(row(148)), READY_CUT)

    def test_age_subtracts(self):
        self.assertEqual(ready_step(row(200, age=53.0)), 147)
        self.assertEqual(ready_step(row(200, age=52.0)), 148)

    def test_float32_clock_round_trip(self):
        import struct
        for step in (0, 1, 134, 147, 148, 999, 1000):
            f = [0.0] * 19
            f[18] = struct.unpack('<f', struct.pack('<f', step / 1000.0))[0]
            self.assertEqual(ready_step(f), step)


class TestExtraction(unittest.TestCase):
    def test_free_decisions_keep_candidate_order_and_charge(self):
        tape = [rec(4, ['a'], [0.2], [0]), rec(189, ['b', 'c'], [0.9, 0.6], [1, 0])]
        dec = free_decisions(tape)
        self.assertEqual([d['target'] for d in dec], ['a', 'b', 'c'])
        self.assertEqual([d['region'] for d in dec], ['early', 'late', 'late'])
        self.assertEqual([d['action'] for d in dec], [0, 1, 0])
        self.assertEqual([d['charge'] for d in dec], [1000, 1000, 1000])

    def test_steps_without_free_candidates_are_ignored(self):
        r = rec(4, ['a'], [0.2], [0])
        r['free'] = []
        self.assertEqual(free_decisions([r]), [])

    def test_split_matches_published_order(self):
        from r72_validation import split_of
        tape = [rec(4, ['a'], [0.2], [0]), rec(189, ['b'], [0.9], [1])]
        self.assertEqual(split_of(tape),
                         {'sends_early': 0, 'skips_early': 1, 'sends_late': 1,
                          'skips_late': 0, 'total': 2})


class TestMarginGeometry(unittest.TestCase):
    def test_identical_early_and_small_late_shift_has_no_flip(self):
        rows = [{'region': 'early', 'p_a': 1e-6, 'p_b': 1e-6, 'dp': 0.0, 'act_a': 0, 'act_b': 0},
                {'region': 'late', 'p_a': 0.90, 'p_b': 0.80, 'dp': -0.10, 'act_a': 1, 'act_b': 1}]
        m = margin_block(rows)
        self.assertEqual(m['early_max_abs_dp'], 0.0)
        self.assertEqual(m['argmax_flips'], 0)
        self.assertEqual(m['late_moved_toward_boundary'], 1)
        self.assertAlmostEqual(m['late_max_encroachment_ratio'], 0.10 / 0.40)

    def test_a_real_crossing_is_detected(self):
        rows = [{'region': 'late', 'p_a': 0.60, 'p_b': 0.40, 'dp': -0.20, 'act_a': 1, 'act_b': 0}]
        m = margin_block(rows)
        self.assertEqual(m['argmax_flips'], 1)

    def test_movement_away_from_boundary_is_not_encroachment(self):
        rows = [{'region': 'late', 'p_a': 0.60, 'p_b': 0.95, 'dp': 0.35, 'act_a': 1, 'act_b': 1}]
        m = margin_block(rows)
        self.assertEqual(m['late_moved_away_from_boundary'], 1)
        self.assertEqual(m['late_moved_toward_boundary'], 0)
        self.assertIsNone(m['late_max_encroachment_ratio'])
        self.assertEqual(m['argmax_flips'], 0)


class TestCellComparison(unittest.TestCase):
    def test_identical_tapes_are_fully_identical(self):
        tape = [rec(4, ['a'], [0.2], [0]), rec(189, ['b'], [0.9], [1])]
        ident, rows, oa, ob = compare_cell(tape, tape)
        self.assertTrue(all(ident.values()))
        self.assertEqual(len(rows), 2)
        self.assertEqual((oa, ob), (0, 0))

    def test_changed_late_probability_keeps_identity_of_actions(self):
        a = [rec(4, ['a'], [0.2], [0]), rec(189, ['b'], [0.9], [1])]
        b = [rec(4, ['a'], [0.2], [0]), rec(189, ['b'], [0.8], [1])]
        ident, rows, oa, ob = compare_cell(a, b)
        self.assertTrue(ident['steps_equal'])
        self.assertTrue(ident['features_equal'])
        self.assertTrue(ident['actions_equal'])
        self.assertAlmostEqual(rows[1]['dp'], -0.1, places=12)
        self.assertEqual(margin_block(rows)['argmax_flips'], 0)

    def test_summarise_counts(self):
        rows = [{'region': 'early', 'p_a': 0.0, 'p_b': 0.0, 'dp': 0.0, 'act_a': 0, 'act_b': 0},
                {'region': 'late', 'p_a': 0.9, 'p_b': 0.8, 'dp': -0.1, 'act_a': 1, 'act_b': 1}]
        s = summarise(rows)
        self.assertEqual(s['n_free'], 2)
        self.assertEqual(s['early']['skips'], 1)
        self.assertEqual(s['late']['sends'], 1)
        self.assertEqual(s['action_flips'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)