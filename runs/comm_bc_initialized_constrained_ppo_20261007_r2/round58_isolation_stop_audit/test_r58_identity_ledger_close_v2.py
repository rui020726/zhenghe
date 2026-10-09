from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r58_identity_ledger_close_v2",
                                              HERE / "r58_identity_ledger_close_v2.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)


class IdentityCloseTests(unittest.TestCase):
    def setUp(self):
        self.version = "approved/gen3/version/1"
        self.semantics = "one-shot-send-skip/1"
        self.identity = {"critic_route_version": "critic_representation_stop_gradient/1",
                         "repair_id": "new-private-identity"}
        self.base_freeze = {"freeze_sha256": "approved-freeze", "runner": "gen3"}
        self.payload = {
            "training_version": self.version,
            "comm_action_semantics": self.semantics,
            "round58_repair_attestation": self.identity,
            "freeze": module.attested_freeze(self.base_freeze, self.identity, self.version),
            "training_state": {"training_version": self.version,
                               "comm_action_semantics": self.semantics,
                               "round58_repair_attestation": self.identity},
        }
        self.expected = {"training_version": self.version,
                         "action_semantics": self.semantics,
                         "base_freeze": self.base_freeze,
                         "repair_identity": self.identity}

    def test_reproduces_legacy_compound_freeze_failure(self):
        # Old check compared the serializer's attested freeze to the unwrapped input.
        self.assertNotEqual(self.payload["freeze"], self.base_freeze)
        self.assertEqual(self.payload["training_version"], self.version)
        self.assertEqual(self.payload["comm_action_semantics"], self.semantics)
        self.assertEqual([d["field"] for d in module.identity_differences(
            self.payload, **self.expected)], [])

    def test_gate_adapter_passes_exact_attested_freeze_to_frozen_gate(self):
        runtime = {"round42_freeze": self.base_freeze, "sentinel": object()}
        frozen = {"round42": {"training_version": self.version}}
        observed = {}
        def actual_gate_shape(seed, *, runtime, frozen):
            observed.update(seed=seed, runtime=runtime, frozen=frozen)
            self.assertEqual(self.payload["freeze"], runtime["round42_freeze"])
            return {"status": "identity_checked"}
        result = module.call_first_batch_gate_with_attested_freeze(
            actual_gate_shape, 11, runtime, frozen,
            repair_identity=self.identity, training_version=self.version)
        self.assertEqual(result["status"], "identity_checked")
        self.assertEqual(observed["runtime"]["sentinel"], runtime["sentinel"])
        self.assertEqual(runtime["round42_freeze"], self.base_freeze)
        with self.assertRaisesRegex(RuntimeError, "training_version"):
            module.call_first_batch_gate_with_attested_freeze(
                actual_gate_shape, 11, runtime, {"round42": {"training_version": "old"}},
                repair_identity=self.identity, training_version=self.version)

    def test_private_v2_accepts_only_exact_approved_gen3_identity(self):
        module.require_identity(self.payload, **self.expected)

    def test_declared_metadata_normalizes_but_arbitrary_freeze_fields_do_not(self):
        altered = {**self.payload, "freeze": dict(self.payload["freeze"]),
                   "training_state": dict(self.payload["training_state"])}
        altered["freeze"]["unregistered_extension"] = True
        fields = {item["field"] for item in module.identity_differences(
            altered, **self.expected)}
        self.assertIn("freeze.base_science_identity", fields)
        altered["freeze"].pop("unregistered_extension")
        altered["freeze"]["training_version"] = "wrong/version"
        fields = {item["field"] for item in module.identity_differences(
            altered, **self.expected)}
        self.assertIn("freeze.training_version", fields)

    def test_old_generation_without_attestation_is_rejected(self):
        legacy = {**self.payload, "training_version": "legacy/gen1",
                  "round58_repair_attestation": None,
                  "freeze": {"freeze_sha256": "old"}}
        fields = {item["field"] for item in module.identity_differences(
            legacy, **self.expected)}
        self.assertTrue({"training_version", "round58_repair_attestation",
                         "freeze.training_version", "freeze.base_science_identity"}
                        .issubset(fields))

    def test_each_identity_component_is_fail_closed_and_specific(self):
        cases = [
            ("training_version", lambda p: p.update(training_version="legacy")),
            ("comm_action_semantics", lambda p: p.update(comm_action_semantics="other")),
            ("freeze.base_science_identity",
             lambda p: p["freeze"].update(freeze_sha256="tampered")),
            ("round58_repair_attestation", lambda p: p.update(round58_repair_attestation={"legacy": True})),
            ("training_state.training_version", lambda p: p["training_state"].update(training_version="legacy")),
        ]
        for wanted, mutate in cases:
            with self.subTest(wanted=wanted):
                altered = {**self.payload,
                           "freeze": dict(self.payload["freeze"]),
                           "training_state": dict(self.payload["training_state"])}
                mutate(altered)
                fields = {item["field"] for item in module.identity_differences(
                    altered, **self.expected)}
                self.assertIn(wanted, fields)

    def test_real_serializer_attestation_shape_is_accepted(self):
        # Exercise the existing private checkpoint contract's metadata constructor,
        # not a hand-written expected answer from the system-under-test.
        path = HERE / "r58_checkpoint_contract_repaired.py"
        contract_spec = importlib.util.spec_from_file_location("r58_contract_readonly", path)
        contract = importlib.util.module_from_spec(contract_spec)
        assert contract_spec.loader
        contract_spec.loader.exec_module(contract)
        built = dict(self.payload)
        built["freeze"] = contract._attested(self.base_freeze, self.identity,
                                             version=self.version)
        built["training_state"] = contract._attested(self.payload["training_state"],
                                                     self.identity, version=self.version)
        built["round58_repair_attestation"] = dict(self.identity)
        module.require_identity(built, **self.expected)
        with self.assertRaisesRegex(RuntimeError, "before restore"):
            contract.require_payload({**built, "training_version": "old"},
                                     self.identity, version=self.version)


class LedgerCloseTests(unittest.TestCase):
    def test_multi_seed_roots_count_once_and_close(self):
        root_a = [{"logical_job": f"train_s11_e{i}", "attempt": 1,
                   "category": "normal"} for i in range(4)]
        root_b = [{"logical_job": f"train_s{s}_e{i}", "attempt": 1,
                   "category": "normal"} for s in (22, 33) for i in range(4)]
        claims = root_a + root_b
        slots = [dict(c, slot=i + 1) for i, c in enumerate(claims)]
        result = module.reconcile_claims_slots(claims, slots, expected_normal=1024,
                                               expected_reserve=24)
        self.assertEqual((result["normal_consumed"], result["total_consumed"]), (12, 12))
        final = module.reconcile_claims_slots(claims, slots, expected_normal=12,
                                              expected_reserve=0, final=True)
        self.assertEqual(final["status"], "complete")

    def test_derived_snapshot_keeps_hook_gap_separate_and_never_fakes_success(self):
        def pack(root_id, seed, count, offset):
            claims = [{"logical_job": f"s{seed}e{i}", "attempt": 1,
                       "category": "normal"} for i in range(count)]
            return {"root_id": root_id, "claims": claims,
                    "slots": [dict(c, slot=offset + i + 1) for i, c in enumerate(claims)],
                    "committed_batches": ([{"seed": seed, "batch": 0}] if count == 4 else [])}
        partial = module.derive_snapshot(
            [pack("root-a", 11, 4, 0), pack("root-b", 22, 4, 4)],
            expected_normal=1024, expected_reserve=24,
            ppo_update_calls_observed=0, optimizer_step_calls_observed=16)
        self.assertEqual(partial["status"], "partial_with_counter_gap")
        self.assertEqual(partial["normal_consumed"], 8)
        self.assertFalse(partial["success"])
        final = module.derive_snapshot(
            [pack("root-a", 11, 4, 0)], expected_normal=4, expected_reserve=0,
            ppo_update_calls_observed=0, optimizer_step_calls_observed=8, final=True)
        self.assertEqual(final["status"], "failed")
        self.assertFalse(final["success"])

    def test_partial_progress_allowed_but_final_incomplete_fails(self):
        claims = [{"logical_job": "s11e0", "attempt": 1, "category": "normal"}]
        slots = [dict(claims[0], slot=1)]
        self.assertEqual(module.reconcile_claims_slots(claims, slots,
            expected_normal=1024, expected_reserve=24)["status"], "partial_valid")
        self.assertEqual(module.reconcile_claims_slots(claims, slots,
            expected_normal=12, expected_reserve=0, final=True)["status"], "failed")

    def test_duplicate_or_missing_records_fail(self):
        claim = {"logical_job": "s11e0", "attempt": 1, "category": "normal"}
        result = module.reconcile_claims_slots([claim, claim], [dict(claim, slot=1)],
                                                expected_normal=10)
        self.assertEqual(result["status"], "failed")

    def test_ppo_counter_bridge_counts_once_and_preserves_return(self):
        trainer = SimpleNamespace(ppo_update=lambda value: value + 1)
        run = SimpleNamespace(_PPO_UPDATE_CALLS=0)
        module.bridge_ppo_update_counter(trainer, run)
        self.assertEqual(trainer.ppo_update(4), 5)
        self.assertEqual(run._PPO_UPDATE_CALLS, 1)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            module.bridge_ppo_update_counter(trainer, run)

    def test_bridge_is_outermost_live_call_and_undo_restores_same_adapter(self):
        calls = []
        adapter = lambda x: calls.append(x) or "adapter-result"
        trainer = SimpleNamespace(ppo_update=adapter)
        run = SimpleNamespace(_PPO_UPDATE_CALLS=7)
        bridge = module.bridge_ppo_update_counter(trainer, run)
        self.assertIs(trainer.ppo_update.__wrapped__, adapter)
        self.assertEqual(trainer.ppo_update("spy"), "adapter-result")
        self.assertEqual(calls, ["spy"])
        self.assertEqual(run._PPO_UPDATE_CALLS, 8)
        bridge["undo"]()
        self.assertIs(trainer.ppo_update, adapter)


if __name__ == "__main__":
    unittest.main(verbosity=2)
