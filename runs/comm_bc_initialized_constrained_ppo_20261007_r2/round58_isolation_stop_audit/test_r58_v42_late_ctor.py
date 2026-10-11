"""Zero-environment regression for archived cut/cut_steps constructor seam."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r58_private_launcher_repaired as launcher
sys.path.insert(0, str(launcher.CANON.parents[1]))
import v2_rules


class FrozenLateConstructorTests(unittest.TestCase):
    def test_old_keyword_fails_new_bound_keyword_preserves_exact_decision(self):
        original = v2_rules.LateDiscoveryGatePolicy
        with self.assertRaises(TypeError):
            original(cut=147.0)
        binding = launcher._bind_frozen_late_rule_constructor(v2_rules)
        try:
            bridged = v2_rules.LateDiscoveryGatePolicy(cut=147.0)
            reference = original(cut_steps=147.0)
            self.assertEqual(bridged.parameters(), reference.parameters())
            self.assertIs(type(bridged).decide, original.decide)
            observations = SimpleNamespace(items=[
                SimpleNamespace(target_id="early", first_emission=True, ready_step=146.999),
                SimpleNamespace(target_id="boundary", first_emission=True, ready_step=147.0),
                SimpleNamespace(target_id="late", first_emission=True, ready_step=147.001),
                SimpleNamespace(target_id="not_first", first_emission=False, ready_step=300.0),
            ])
            self.assertEqual(bridged.decide(observations), reference.decide(observations))
            self.assertEqual(bridged.decide(observations), ("late",))
            for value in (146.0, 148.0, None):
                if value is None:
                    with self.assertRaises(TypeError):
                        v2_rules.LateDiscoveryGatePolicy(cut=147.0, cut_steps=147.0)
                else:
                    with self.assertRaises(ValueError):
                        v2_rules.LateDiscoveryGatePolicy(cut=value)
        finally:
            binding["undo"]()
        self.assertIs(v2_rules.LateDiscoveryGatePolicy, original)
        with self.assertRaises(TypeError):
            original(cut=147.0)

    def test_wrong_source_rejected(self):
        fake = SimpleNamespace(__file__=str(HERE / "r58_private_launcher_repaired.py"),
                               LateDiscoveryGatePolicy=v2_rules.LateDiscoveryGatePolicy)
        with self.assertRaises(PermissionError):
            launcher._bind_frozen_late_rule_constructor(fake)


if __name__ == "__main__":
    unittest.main()
