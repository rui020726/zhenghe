from __future__ import annotations

import sys
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
R2 = HERE.parent
R61_RAW = R2 / "round61_early_report_value_intervention" / "raw"
MASK_SOURCE = HERE.parents[1] / "comm_budget_action_mask_fix_20260922"
ELIGIBILITY_SOURCE = HERE.parents[1] / "comm_first_report_training_ablation_20261003"
STAGE_B_SOURCE = HERE.parents[1] / "comm_cost_stage_b_20260927"
ONPOLICY_SOURCE = HERE.parents[1] / "comm_tight_budget_20260919"
for path in (HERE, R61_RAW, MASK_SOURCE, ELIGIBILITY_SOURCE, STAGE_B_SOURCE,
             ONPOLICY_SOURCE):
    sys.path.insert(0, str(path))

from budget_masked import HOLD, SEND
from integrated_sar.comm_policy import SendItem, SendObservation
from r42_one_shot import OneShotSendSkipCollector
from r64_execute import (RAW, R64_EXECUTION_VERSION, build_cell_collector,
                         run_cell)
from r64_guard import R64Guard, resume_plan_status
from r64_identity import BINDING, validate_live_binding
from r64_intervention import make_intervene_collector

FOCUS = BINDING.focus_target
SCRATCH = HERE / ".test_scratch"
SCRATCH.mkdir(exist_ok=True)


def _model_identity(version=BINDING.execution_version):
    return {
        "model_path": str(R2 / BINDING.model_path_relative_to_r2),
        "file_sha256": BINDING.model_file_sha256,
        "parameter_sha256": BINDING.model_parameter_sha256,
        "execution_version": version,
        "load_mode": "read_only",
        "load_scope": "model_only",
        "optimizer_restored": False,
        "rng_restored": False,
        "training_cursor_restored": False,
    }


def _limits():
    return {"normal_cap": BINDING.normal_cap, "reserve_cap": BINDING.reserve_cap,
            "hard_cap": BINDING.hard_cap}


class VersionBindingTests(unittest.TestCase):
    def test_one_binding_drives_executor_guard_and_candidate(self):
        self.assertEqual(R64_EXECUTION_VERSION, BINDING.execution_version)
        self.assertEqual(R64Guard.EXECUTION_VERSION, BINDING.execution_version)
        self.assertEqual((R64Guard.NORMAL_CAP, R64Guard.RESERVE_CAP, R64Guard.HARD_CAP),
                         (BINDING.normal_cap, BINDING.reserve_cap, BINDING.hard_cap))
        receipt = validate_live_binding(
            executor_version=R64_EXECUTION_VERSION,
            guard_version=BINDING.execution_version,
            model_identity=_model_identity(), guard_limits=_limits(), raw_root=RAW,
        )
        self.assertTrue(receipt["validated"])
        self.assertEqual(receipt["limits"], {"normal_cap": 10, "reserve_cap": 3,
                                              "hard_cap": 13})

    def test_old_6_executor_4_guard_is_rejected_before_slot_or_builder(self):
        calls = {"claim": 0, "builder": 0}

        class MixedGuard:
            root = RAW.resolve()
            allow_build = True
            execution_version = "round64-controlled-deadline-intervention/4"
            normal_cap = 10
            reserve_cap = 2
            hard_cap = 12

            def terminal(self, logical):
                raise AssertionError("terminal lookup must follow identity validation")

            def attempt(self, *args, **kwargs):
                calls["claim"] += 1
                raise AssertionError("slot claim must not be reached")

        class BuilderSpy:
            def __call__(self, *_args, **_kwargs):
                calls["builder"] += 1
                raise AssertionError("builder must not be reached")

        cfg = {"r64_guard": MixedGuard(), "model_identity": _model_identity(),
               "builder": BuilderSpy()}
        with patch("r64_execute.R64_EXECUTION_VERSION",
                   "round64-controlled-deadline-intervention/6"):
            with self.assertRaisesRegex(RuntimeError, "executor/guard version mismatch"):
                run_cell(cfg, {"logical": "synthetic", "delta": 30, "arm": "FORCE_SEND"})
        self.assertEqual(calls, {"claim": 0, "builder": 0})

    def test_approved_receipt_makes_four_cells_fit_only_under_cap_thirteen(self):
        logicals = ["d+30_FORCE_SEND", "d+30_FORCE_SKIP",
                    "d+60_FORCE_SEND", "d+60_FORCE_SKIP"]
        slots = [{"category": "normal", "logical": f"historical-{i}"}
                 for i in range(1, 7)]
        slots += [{"category": "normal", "logical": "d+30_FORCE_SEND"}]
        slots += [{"category": "reserve", "logical": f"historical-reserve-{i}"}
                  for i in range(1, 3)]
        rows = []
        for logical in ("d-30_FORCE_SEND", "d-30_FORCE_SKIP",
                        "d+0_FORCE_SEND", "d+0_FORCE_SKIP"):
            rows.append({"logical": logical, "event": "result_qualified"})
        rows.append({"logical": "d+30_FORCE_SEND", "event": "slot_consumed"})
        rows.append({"logical": "d+30_FORCE_SEND", "event": "run_called"})
        receipt = {"d+30_FORCE_SEND": {"receipt_sha256": "synthetic"}}
        old_cap = resume_plan_status(slots, rows, logicals,
                                     normal_cap=10, reserve_cap=2, hard_cap=12,
                                     recovery_receipts=receipt)
        self.assertFalse(old_cap["can_resume"])
        self.assertEqual(old_cap["unresolved_logicals"], logicals)
        self.assertEqual(old_cap["additional_attempts_required"], 4)
        self.assertEqual(old_cap["normal_available"], 3)
        self.assertEqual(old_cap["reserve_available"], 0)
        self.assertEqual(old_cap["hard_cap_headroom"], 3)
        approved = resume_plan_status(slots, rows, logicals,
                                      normal_cap=10, reserve_cap=3, hard_cap=13,
                                      recovery_receipts=receipt)
        self.assertTrue(approved["can_resume"])
        self.assertEqual(approved["nonterminal_logicals"], [])
        self.assertEqual(approved["normal_attempts_required"], 3)
        self.assertEqual(approved["reserve_retries_required"], 1)
        self.assertEqual(approved["hard_cap_headroom"], 4)

    def test_original_partial_is_retryable_only_with_verified_receipt(self):
        guard = R64Guard(RAW, allow_build=False)
        records = guard.verified_recovery_receipts()
        self.assertEqual(set(records), {"d+30_FORCE_SEND"})
        self.assertTrue(records["d+30_FORCE_SEND"]["receipt_sha256"])
        receipt = records["d+30_FORCE_SEND"]["receipt"]
        self.assertEqual(receipt["interrupted_attempt"]["slot"], "normal-07")
        self.assertFalse(receipt["interrupted_attempt"]["ledger_prefix"]["terminal_event_present"])
        self.assertEqual(receipt["recovery_authorized"]["only_slot"], "reserve-03")

    def test_fourteenth_attempt_is_refused_after_thirteen_synthetic_slots(self):
        with tempfile.TemporaryDirectory(prefix="r64-cap13-", dir=SCRATCH) as directory:
            root = Path(directory)
            slots_root = root / "guard_slots"
            slots_root.mkdir()
            ledger = []
            for ordinal in range(1, 14):
                category = "normal" if ordinal <= 10 else "reserve"
                slot = f"{category}-{ordinal if category == 'normal' else ordinal - 10:02d}"
                logical = f"fixture-{ordinal:02d}"
                payload = {"task": "ROUND64", "execution_version": BINDING.execution_version,
                           "slot": slot, "logical": logical, "category": category,
                           "retry": False, "created_unix": 0}
                (slots_root / f"{slot}.slot").write_text(json.dumps(payload), encoding="utf-8")
                ledger.append({"task": "ROUND64", "execution_version": BINDING.execution_version,
                               "slot": slot, "logical": logical, "event": "slot_consumed"})
            (root / "ROUND64_LEDGER.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8")
            guard = R64Guard(root)
            with self.assertRaisesRegex(RuntimeError, "normal cap reached"):
                guard._claim("fixture-14", "normal")
            self.assertEqual(len(guard._slot_files()), 13)
            cfg = {"r64_guard": guard, "model_identity": _model_identity(),
                   "raw_root": root, "cells_dir": root / "cells"}
            with patch("r64_execute.RAW", root), \
                 patch("r64_execute.validate_live_binding"):
                with self.assertRaisesRegex(RuntimeError, "normal cap reached"):
                    run_cell(cfg, {"logical": "fixture-14", "delta": 60,
                                   "arm": "FORCE_SKIP"})
            self.assertFalse((root / "launch_claims").exists())


class _StubController:
    forward_calls = 0

    def forward(self, matrix):
        type(self).forward_calls += 1
        logits = torch.zeros((matrix.shape[0], 2), dtype=torch.float32)
        for i, row in enumerate(matrix):
            # The ordinary greedy policy holds age-1 and sends age-2.
            if float(row[0]) == 1.0:
                logits[i] = torch.tensor([1.0, -1.0])
            else:
                logits[i] = torch.tensor([-1.0, 1.0])
        return logits, torch.zeros((matrix.shape[0], 1), dtype=torch.float32)


def _item(target, age, charge=100):
    return SendItem(
        target_id=target, ready_step=134, age_steps=age, pending_event=True,
        changed_view=True, first_emission=True, revision=1, revision_delta=1,
        stage_index=0, stage_changed=False, progress=0.0, progress_delta=0.0,
        requirements_changed=True, members_changed=False, deadline=234.0,
        remaining_time=100.0, sent_count=0, last_sent_step=None,
        since_last_emission_time=None, has_future_requirements=True,
        attempt_charge_bytes=charge, send_executable=True, send_block_reason="allowed",
    )


def _observation(items, *, remaining=1000, budget=1000):
    return SendObservation(
        step=134, current_time=134.0, items=tuple(items),
        remaining_budget_bytes=remaining, budget_bytes_per_episode=budget,
        capacity_remaining=5, backlog_targets=len(items), subscribers=6,
    )


class ActualCollectorPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _StubController.forward_calls = 0
        cls.intervene_cls = make_intervene_collector(OneShotSendSkipCollector)

    @classmethod
    def tearDownClass(cls):
        if _StubController.forward_calls != 7:
            raise AssertionError(f"expected 7 pure stub forwards, got {_StubController.forward_calls}")

    def collector(self, arm):
        cfg = {"plain_cls": OneShotSendSkipCollector, "r64_cls": self.intervene_cls,
               "controller": _StubController()}
        return build_cell_collector(cfg, arm, 0)

    def call_without_side_effects(self, collector, observation):
        import integrated_sar.comm as comm
        import integrated_sar.joint_driver as joint_driver
        import r64_execute

        with patch.object(comm.CommRelay, "_emit", side_effect=AssertionError("emit blocked")), \
             patch.object(joint_driver, "build_joint_run",
                          side_effect=AssertionError("environment builder blocked")), \
             patch.object(r64_execute, "write_exclusive",
                          side_effect=AssertionError("checkpoint/write path blocked")):
            return collector._choose_masked(observation)

    def test_send_and_skip_use_the_actual_free_candidate_branch(self):
        for arm, expected, expected_asked, state in (
            ("FORCE_SEND", SEND, [FOCUS], "send_now_chosen"),
            ("FORCE_SKIP", HOLD, [], "skip_episode_chosen"),
        ):
            collector = self.collector(arm)
            asked, actions, free, _logp, _value, tape = self.call_without_side_effects(
                collector, _observation([_item(FOCUS, 1)])
            )
            self.assertEqual(asked, expected_asked)
            self.assertEqual(actions[FOCUS], expected)
            self.assertEqual(free, [FOCUS])
            self.assertEqual(tape["decision_class"][FOCUS], "free_one_shot_choice")
            self.assertEqual(tape["one_shot_state_after_step"][FOCUS], state)
            self.assertEqual(collector.r64_record["original_greedy_action"], HOLD)
            self.assertEqual(collector.r64_record["actual_action"], expected)
            self.assertTrue(tape["candidate_audit"][FOCUS]["sequentially_affordable_before_action"])

    def test_sequential_mask_prevents_forced_send(self):
        collector = self.collector("FORCE_SEND")
        asked, actions, free, _logp, _value, tape = self.call_without_side_effects(
            collector, _observation([_item("spender", 2), _item(FOCUS, 1)],
                                    remaining=150, budget=1000)
        )
        self.assertEqual(actions["spender"], SEND)
        self.assertEqual(actions[FOCUS], HOLD)
        self.assertEqual(asked, ["spender"])
        self.assertEqual(free, ["spender"])
        self.assertEqual(tape["decision_class"][FOCUS], "existing_budget_or_capacity_mask")
        self.assertFalse(tape["candidate_audit"][FOCUS]["sequentially_affordable_before_action"])
        self.assertIsNone(collector.r64_record)
        self.assertFalse(collector._r64_done)

    def test_one_override_closes_only_target_then_other_target_is_greedy(self):
        collector = self.collector("FORCE_SEND")
        self.call_without_side_effects(collector, _observation([_item(FOCUS, 1)]))
        asked, actions, free, _logp, _value, tape = self.call_without_side_effects(
            collector, _observation([_item(FOCUS, 1), _item("other", 2)])
        )
        self.assertEqual(actions[FOCUS], HOLD)
        self.assertEqual(tape["decision_class"][FOCUS], "fixed_hold_one_shot_closed")
        self.assertEqual(actions["other"], SEND)
        self.assertEqual(asked, ["other"])
        self.assertEqual(free, ["other"])
        self.assertEqual(collector.r64_record["actual_action"], SEND)

    def test_disabled_interposer_matches_parent_actions_and_tape(self):
        observation = _observation([_item(FOCUS, 1), _item("other", 2)])
        controller = _StubController()
        parent = OneShotSendSkipCollector(controller, greedy=True)
        cfg = {"plain_cls": OneShotSendSkipCollector, "r64_cls": self.intervene_cls,
               "controller": controller}
        disabled = build_cell_collector(cfg, "L0", 0)
        self.assertEqual(parent._choose_masked(observation),
                         disabled._choose_masked(observation))
        self.assertIsNone(disabled.r64_record)


if __name__ == "__main__":
    unittest.main()
