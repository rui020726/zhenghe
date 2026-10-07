from __future__ import annotations

import unittest
from collections import OrderedDict
from types import SimpleNamespace

from round53_trial_adapter import (
    TRAINER_SHA256, critic_input_detached, install_on_verified_trainer,
    require_explicit_authorization, require_private_output_root, validate_live_binding,
)


class Handle:
    def __init__(self, module, key): self.module, self.key = module, key
    def remove(self): self.module._forward_pre_hooks.pop(self.key, None)


class Head:
    def __init__(self): self._forward_pre_hooks = OrderedDict(); self._next = 0
    def register_forward_pre_hook(self, fn):
        self._next += 1; self._forward_pre_hooks[self._next] = fn
        return Handle(self, self._next)
    def apply(self, x):
        args = (x,)
        for hook in list(self._forward_pre_hooks.values()): args = hook(self, args)
        return args[0]


class DummyTensor:
    def __init__(self, label="live"): self.label = label
    def detach(self): return DummyTensor("detached")


class TestRound53Pure(unittest.TestCase):
    def setUp(self):
        self.c = SimpleNamespace(inner=SimpleNamespace(value_head=Head()), cost_head=Head())

    def test_hooks_detach_only_head_inputs_and_restore(self):
        with critic_input_detached(self.c):
            self.assertEqual(self.c.inner.value_head.apply(DummyTensor()).label, "detached")
            self.assertEqual(self.c.cost_head.apply(DummyTensor()).label, "detached")
        self.assertEqual(len(self.c.inner.value_head._forward_pre_hooks), 0)
        self.assertEqual(len(self.c.cost_head._forward_pre_hooks), 0)

    def test_exception_cleanup(self):
        with self.assertRaisesRegex(ValueError, "fixture"):
            with critic_input_detached(self.c): raise ValueError("fixture")
        self.assertEqual(len(self.c.inner.value_head._forward_pre_hooks), 0)
        self.assertEqual(len(self.c.cost_head._forward_pre_hooks), 0)

    def test_wrong_source_refused_and_method_restored(self):
        trainer = SimpleNamespace(ppo_update=lambda controller, *a, **k: "called")
        original = trainer.ppo_update
        with self.assertRaisesRegex(RuntimeError, "identity"):
            install_on_verified_trainer(trainer, source_sha256="bad")
        undo = install_on_verified_trainer(trainer, source_sha256=TRAINER_SHA256)
        undo()
        self.assertIs(trainer.ppo_update, original)

    def test_role_and_runtime_identity_refusal(self):
        trainer = SimpleNamespace(ppo_update=lambda *a, **k: None)
        r40 = SimpleNamespace(trainer=trainer)
        validate_live_binding({"trainer": trainer, "r40_train": r40}, trainer)
        with self.assertRaisesRegex(RuntimeError, "role/object"):
            validate_live_binding({"trainer": object(), "r40_train": r40}, trainer)
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            validate_live_binding({"trainer": trainer, "r40_train": SimpleNamespace(trainer=object())}, trainer)

    def test_counter_fixture_is_zero_without_invoking_update(self):
        counts = {name: 0 for name in (
            "environment_construction", "simulation", "new_scenes_or_flows",
            "new_evaluation", "real_training_updates", "synthetic_optimizer_steps",
            "checkpoint_writes")}
        self.assertTrue(all(value == 0 for value in counts.values()))
        self.assertEqual(sum(counts.values()), 0)

    def test_authorization_gate_and_fake_ledger_closure(self):
        with self.assertRaises(PermissionError):
            require_explicit_authorization({"status": "proposed_not_authorized", "hard_cap": 552})
        # In-memory slot/claim fixture only: reserve distinct slots and close at cap.
        slots = {"normal": 528, "reserve": 24}
        self.assertEqual(sum(slots.values()), 552)
        claims = [(kind, n) for kind, count in slots.items() for n in range(count)]
        self.assertEqual(len(claims), 552)
        self.assertEqual(len(set(claims)), 552)
        self.assertEqual(sum(1 for kind, _ in claims if kind == "normal"), 528)
        self.assertEqual(sum(1 for kind, _ in claims if kind == "reserve"), 24)

    def test_private_output_path_containment(self):
        from pathlib import Path
        root = Path("C:/sandbox/round53")
        require_private_output_root(root, [root / "rows" / "a.json", root / "runtime" / "ledger.json"])
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            require_private_output_root(root, [Path("C:/sandbox/round46/v6/rows/a.json")])


if __name__ == "__main__": unittest.main()
