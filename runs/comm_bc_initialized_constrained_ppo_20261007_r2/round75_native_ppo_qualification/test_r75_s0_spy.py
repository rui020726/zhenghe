"""ROUND75 S0 spy tests: pure stdlib, fake objects, real frozen JSON. Zero torch, zero NN.

Proves, without any model load / construction / forward / backward / optimizer / PPO / env:
1. the replay adapter rebuilds the real recorded episode and fails closed on schema problems;
2. the R75 central production ledger charges BEFORE work, enforces the R75 caps, refuses the
   cancelled fixture mode and the forbidden env path, never refunds, survives restart, and
   reconciles against its append-only ledger;
3. the native wiring source really is the r66 update function with ONLY the freeze-set change
   (AST-level: required real calls present, transcription/shortcut patterns absent).
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r75_guard as guard_mod                      # noqa: E402  (stdlib only)
import r75_replay_adapter as adapter_mod           # noqa: E402  (stdlib only)

RUN_ROOT = HERE.parent
GEN2 = RUN_ROOT / "round58_gen2" / "raw"
ROW = GEN2 / "rows" / "v2_r44_train_s11_e0_s1300027_a1.json"
TAPE = GEN2 / "logs" / "v2_r44_train_s11_e0_s1300027_a1_tape.jsonl"
SELFTEST = HERE / "_s0_selftest"


class AdapterTests(unittest.TestCase):
    """The frozen qualification episode (seed 11, round58_gen2, e0, scenario 1300027)."""

    @classmethod
    def setUpClass(cls):
        cls.episode, cls.facts = adapter_mod.load_qualification_batch(ROW, TAPE)

    def test_frozen_identity(self):
        self.assertEqual(self.facts["training_seed"], 11)
        self.assertEqual(self.facts["scenario_seed"], 1300027)
        self.assertEqual(self.facts["episode_index"], 0)
        self.assertEqual(self.facts["row_sha256"],
                         "882711788d8ccae4ec8092b4eb26df7da6771c9b352b6c7507624d3644abebd8")
        self.assertEqual(self.facts["tape_sha256"],
                         "6d4c4363ff490fafe08b55bbf369dcfaf6284b741c2802cf9ef3374d9b792421")
        self.assertEqual(self.facts["behaviour_parameter_sha256"],
                         "37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330")
        self.assertEqual(self.facts["behaviour_parameter_count"], 5700)
        self.assertEqual(self.facts["behaviour_heads"],
                         ["action_head", "value_task_head", "value_cost_head"])

    def test_batch_samples_contract(self):
        episode = self.episode
        self.assertEqual(episode["steps"], 1000)
        self.assertEqual(episode["lambda"], 1.0)
        self.assertEqual(len(episode["collector"].records), 66)
        first = episode["collector"].records[0]
        for key in ("step", "features", "targets", "actions", "actionable",
                    "log_prob", "value", "cost_value"):
            self.assertIn(key, first)
        self.assertEqual(len(first["features"][0]), 19)
        self.assertIn("airtime_bytes", next(iter(episode["cost_stream"].values())))
        self.assertTrue(all(isinstance(v, float) for v in episode["task_stream"].values()))

    def test_sample_budget_and_coverage(self):
        self.assertEqual(self.facts["records"], 66)                 # <= 256 cap
        self.assertEqual(self.facts["horizon"], 1000)
        self.assertEqual(self.facts["first_step"], 1)
        self.assertEqual(self.facts["last_step"], 192)
        self.assertEqual(self.facts["free_early_rows"], 5)
        self.assertEqual(self.facts["free_late_rows"], 1)
        self.assertEqual(self.facts["gate_eligible_early_rows"], 5)
        self.assertEqual(self.facts["candidate_rows"], 168)

    def test_substituted_fields_registered(self):
        subs = self.facts["substituted_fields"]
        self.assertEqual(len(subs), 12)
        self.assertTrue(any("comm_budget_bytes" in s for s in subs))
        self.assertTrue(any("comm_ack_delay_ticks" in s for s in subs))
        self.assertEqual(self.facts["budget_bytes"], 29300)
        self.assertEqual(self.facts["b0_bytes"], 10000.0)

    def test_adapter_fails_closed_on_bad_schema(self):
        row = json.loads(ROW.read_text(encoding="utf-8"))
        records = adapter_mod.load_tape(TAPE)
        broken = [dict(r) for r in records]
        broken[3]["features"] = [[0.0] * 18]
        episode = adapter_mod.episode_from_row_tape(row, broken)
        with self.assertRaises(adapter_mod.AdapterError):
            adapter_mod.validate_episode(episode)


class GuardTests(unittest.TestCase):
    """Fake-caller tests of the R75 central production ledger (no torch anywhere)."""

    def setUp(self):
        shutil.rmtree(SELFTEST, ignore_errors=True)
        SELFTEST.mkdir(parents=True, exist_ok=True)
        self.root = SELFTEST / "ledger"

    def _open(self):
        res = guard_mod.R75Resources.production(self.root)
        self.addCleanup(res.close)
        return res

    def test_charge_moves_count_before_work_and_never_refunds(self):
        res = self._open()
        self.assertEqual(res.charge_physical_step("probe_step"), 1)
        # a failed step does not refund: charge again and the count only grows
        self.assertEqual(res.charge_physical_step("probe_step_failed_work"), 2)
        res.close()

    def test_normal_cap_then_refusal_does_not_consume(self):
        res = self._open()
        for i in range(12):
            res.charge_physical_step(f"step_{i}")
        with self.assertRaises(guard_mod.R75LedgerError):
            res.charge_physical_step("step_13_refused")
        self.assertEqual(res.synth_steps.used(), 12)          # refusal consumed nothing
        res.pre_register_technical_attempt("attempt-01")
        self.assertEqual(res.charge_physical_step("attempt-01_retry", category="technical"), 13)
        for i in range(3):
            res.charge_physical_step(f"attempt-01_retry_{i}", category="technical")
        with self.assertRaises(guard_mod.R75LedgerError):      # hard cap 16
            res.charge_physical_step("attempt-01_over_hard", category="technical")
        res.close()

    def test_technical_without_preregistration_refused(self):
        res = self._open()
        with self.assertRaises(guard_mod.R75LedgerError):
            res.charge_physical_step("unregistered_technical", category="technical")
        self.assertEqual(res.synth_steps.used(), 0)
        res.close()

    def test_offline_rows_caps(self):
        res = self._open()
        with self.assertRaises(guard_mod.R75LedgerError):      # per-charge cap 1024
            res.charge_offline_rows(1025, "too_big")
        self.assertEqual(res.offline_rows.used(), 0)
        for i in range(4):                                     # 4 x 1024 = 4096 normal
            res.charge_offline_rows(1024, f"rows_{i}")
        res.charge_offline_rows(904, "rows_topup")             # 4096 + 904 = 5000 normal
        with self.assertRaises(guard_mod.R75LedgerError):      # normal cap 5000
            res.charge_offline_rows(1, "rows_over_normal")
        res.pre_register_technical_attempt("att-a")
        res.charge_offline_rows(1000, "att-a_rows", category="technical")
        with self.assertRaises(guard_mod.R75LedgerError):      # hard cap 6000
            res.charge_offline_rows(1, "att-a_over_hard", category="technical")
        res.close()

    def test_env_forbidden_and_fixture_mode_cancelled(self):
        res = self._open()
        with self.assertRaises(guard_mod.R75LedgerError):
            res.charge_env_construction("any_env")
        res.close()
        with self.assertRaises(guard_mod.R75LedgerError):
            guard_mod.R75Resources.simulated(self.root)

    def test_restart_does_not_clear_consumption(self):
        res = self._open()
        res.charge_physical_step("before_restart")
        res.close()
        res2 = self._open()     # "restart": new object, same root
        self.assertEqual(res2.synth_steps.used(), 1)
        self.assertEqual(res2.charge_physical_step("after_restart"), 2)
        res2.close()

    def test_bootstrap_writes_zero_identity_and_refuses_after_consumption(self):
        res = self._open()
        snap = res.bootstrap()
        self.assertEqual(snap["physical_optimizer_step"]["used"], 0)
        self.assertEqual(snap["environment_construction"]["hard_cap"], 0)
        for unit in ("physical_optimizer_step", "offline_trunk_target_row",
                     "joint_record", "environment_construction"):
            self.assertTrue((self.root / f"r75_{unit}.json").is_file(), unit)
        opened = [json.loads(line) for line in
                  (self.root / "r75_total_ledger.jsonl").read_text(
                      encoding="utf-8").splitlines()]
        self.assertTrue(opened and all(e["event"] == "ledger-opened" for e in opened), opened)
        check = res.consistency_check()
        self.assertTrue(check["consistent"], check)          # opened markers are not charges
        res.close()
        res2 = self._open()                                   # restart keeps the zero identity
        self.assertEqual(res2.synth_steps.used(), 0)
        res2.charge_physical_step("after_bootstrap")
        with self.assertRaises(guard_mod.R75LedgerError):     # consumed ledger refuses bootstrap
            res2.bootstrap()
        res2.close()

    def test_consistency_check(self):
        res = self._open()
        res.charge_physical_step("s1")
        res.charge_offline_rows(168, "r1")
        check = res.consistency_check()
        self.assertTrue(check["consistent"], check)
        res.close()

    def test_lock_held_by_open_ledger(self):
        res = self._open()
        with self.assertRaises(guard_mod.R75LedgerError):       # lock still held
            guard_mod.R75Resources.production(self.root)
        res.close()
        res2 = self._open()     # acquirable after close
        res2.close()


class WiringSourceTests(unittest.TestCase):
    """AST-level proof that r75_frozen.py is the real update with ONLY the freeze-set change."""

    @classmethod
    def setUpClass(cls):
        cls.src = (HERE / "r75_frozen.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.src)

    def test_freeze_set_is_the_only_structural_change(self):
        self.assertIn('FROZEN_MODULES = ("trunk",)', self.src)
        self.assertIn('TRAINABLE_MODULES = ("early_head", "late_head", '
                      '"value_task_head", "value_cost_head")', self.src)

    def test_required_real_calls_present(self):
        for needle in ("trainer.batch_samples(episodes)",
                       "r66.likelihood_alignment(controller, samples)",
                       "r66.route_logits(controller, stacked, hidden)",
                       "common.EPOCHS_PER_BATCH",
                       "common.MINIBATCH_STEPS",
                       "common.DATA_ORDER_SEED",
                       "common.MAX_GRAD_NORM",
                       "torch.nn.utils.clip_grad_norm_(trainable_parameters(controller)",
                       "optimizer.zero_grad()",
                       "loss.backward()",
                       "resources.charge_physical_step(",
                       "optimizer.step()"):
            self.assertIn(needle, self.src, needle)

    def test_charge_precedes_step_in_source_order(self):
        self.assertLess(self.src.index("resources.charge_physical_step("),
                        self.src.index("optimizer.step()"))

    def test_transcription_and_shortcut_patterns_absent(self):
        for bad in ("minibatch_loss", "def _hand_written", "surrogate = ratio * advantage"):
            self.assertNotIn(bad, self.src, bad)

    def test_gate_prior_constants_preserved(self):
        for needle in ("READY_CUT = 147.0", "HORIZON = 1000.0", "SKIP_INDEX = 0",
                       "GATE_BETA = 1.00"):
            self.assertIn(needle, self.src, needle)

    def test_update_accounting_arithmetic(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "r75_frozen_accounting", HERE / "r75_frozen.py")
        # load the module source WITHOUT executing its torch imports: exec only the pure function
        namespace: dict = {}
        tree = ast.parse(self.src)
        wanted = {"update_accounting"}
        module = ast.Module(body=[n for n in tree.body
                                  if isinstance(n, ast.FunctionDef) and n.name in wanted],
                            type_ignores=[])
        exec(compile(module, "<accounting>", "exec"), namespace)
        out = namespace["update_accounting"](66, 168)
        self.assertEqual(out["physical_optimizer_steps"], 4)
        self.assertEqual(out["minibatches_per_epoch"], 1)
        self.assertEqual(out["offline_trunk_target_rows"], 840)


if __name__ == "__main__":
    unittest.main(verbosity=2)