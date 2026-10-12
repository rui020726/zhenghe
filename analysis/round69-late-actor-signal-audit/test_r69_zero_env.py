#!/usr/bin/env python
"""ROUND69 zero-environment unit tests: pure arithmetic only.

No ROUND68 run file is opened, no environment is constructed, no model is loaded. These tests
lock the hand-computed GAE miniature, the reward bundling boundary, the combined-advantage
identity used by the audit, and the early/late routing constant.
"""
import unittest

import r69_late_actor_signal_audit as audit


class TestHandExamples(unittest.TestCase):
    def test_gae_miniature_mc_boundary(self):
        adv, ret = audit.gae([1.0, 2.0, 3.0], [0.5, 0.7, 0.9], 1.0, 1.0,
                             [False, False, True], [0.7, 0.9, 0.0])
        self.assertAlmostEqual(adv[2], 2.1, places=12)
        self.assertAlmostEqual(adv[1], 4.3, places=12)
        self.assertAlmostEqual(adv[0], 5.5, places=12)
        self.assertAlmostEqual(ret[0], 6.0, places=12)  # MC sum of the bundled stream

    def test_bundle_prefix_and_latest_decision(self):
        stream = {0: 5.0, 3: 1.0, 7: 2.0, 10: 4.0}
        bundled, prefix = audit.bundle(stream, [3, 7], 10)
        self.assertEqual(prefix, 5.0)               # tick 0 before the first decision
        self.assertEqual(bundled[3], 1.0)           # tick 3 credited to decision 3
        self.assertEqual(bundled[7], 6.0)           # ticks 7 and 10 credited to decision 7

    def test_combined_advantage_identity(self):
        # combined = lambda*A_task - A_cost == (lambda*R_task - R_cost) - (lambda*V_task - V_cost)
        lam, r_task, r_cost, v_task, v_cost = 1.5, 2.0, 0.4, 0.5, 0.1
        a_task, a_cost = r_task - v_task, r_cost - v_cost
        self.assertAlmostEqual(lam * a_task - a_cost,
                               (lam * r_task - r_cost) - (lam * v_task - v_cost), places=12)

    def test_ready_step_routing_constant(self):
        early = [0.0] * 19
        early[18], early[0] = 0.147, 0.0            # ready = 147 -> early head
        late = [0.0] * 19
        late[18], late[0] = 0.148, 0.0              # ready = 148 -> late head
        self.assertEqual(audit.ready_step(early), 147)
        self.assertEqual(audit.ready_step(late), 148)
        self.assertLessEqual(audit.ready_step(early), audit.READY_CUT)
        self.assertGreater(audit.ready_step(late), audit.READY_CUT)


if __name__ == "__main__":
    unittest.main()