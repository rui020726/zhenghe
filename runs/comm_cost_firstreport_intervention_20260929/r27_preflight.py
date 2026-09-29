"""ROUND27 零环境、零 optimizer 预检：干预器六项合成测试 + 账本/血缘/禁用扫描。

Zero environment launches: every test runs on synthetic SendObservations and a deterministic
stub controller. The suite also proves the harness never constructs an optimizer (the round's
0-optimizer.step rule, including synthetic updates) and never references a driver RNG.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

import r27_common as common
from budget_masked import SEND, BudgetMaskedCollector
from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_policy import SendItem, SendObservation
from r27_intervention import FirstReportAdvancer

HERE = Path(__file__).resolve().parent


class StubController:
    """Deterministic logits driven by the first feature (age_steps): SEND logit = age - 3.

    age < 3 -> greedy HOLD, age > 3 -> greedy SEND, age == 3 -> the documented tie to SEND.
    """

    def forward(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ages = matrix[:, 0]
        logits = torch.stack([torch.zeros_like(ages), ages - 3.0], dim=1)
        return logits, torch.zeros((matrix.shape[0], 1))

    def parameter_summary(self) -> dict:
        return {"stub": True, "parameters": 0}


def make_item(target: str, *, age: int, charge: int = 3000, sent_count: int = 0,
              first_emission: bool = True, ready_step: int = 1) -> SendItem:
    return SendItem(
        target_id=target, ready_step=ready_step, age_steps=age, pending_event=True,
        changed_view=True, first_emission=first_emission, revision=1, revision_delta=0,
        stage_index=0, stage_changed=False, progress=0.0, progress_delta=0.0,
        requirements_changed=False, members_changed=False, deadline=100.0, remaining_time=50.0,
        sent_count=sent_count, last_sent_step=None, since_last_emission_time=None,
        has_future_requirements=False, attempt_charge_bytes=charge, send_executable=True,
        send_block_reason="allowed",
    )


def make_observation(items: list[SendItem], *, step: int = 1, remaining: float = 10000.0,
                     credit: int | None = 10000, capacity: int | None = None,
                     subscribers: int = 6, backlog: int = 0) -> SendObservation:
    return SendObservation(
        step=step, current_time=float(step), items=tuple(items),
        remaining_budget_bytes=float(remaining), budget_bytes_per_episode=credit,
        capacity_remaining=capacity, backlog_targets=backlog, subscribers=subscribers,
    )


def canonical(tape: dict) -> str:
    return json.dumps(tape, ensure_ascii=False, sort_keys=True)


class TestDisableDelegation(unittest.TestCase):
    """Disabled overrides (and an absent named target) must reproduce the parent byte for byte."""

    def test_disabled_delegates_verbatim(self) -> None:
        stream = [
            make_observation([make_item("t-a", age=1), make_item("t-b", age=5)], step=1),
            make_observation([make_item("t-a", age=2), make_item("t-b", age=1),
                              make_item("t-c", age=7)], step=2, remaining=7000.0),
            make_observation([make_item("t-a", age=4)], step=3, remaining=3500.0),
        ]
        parent = BudgetMaskedCollector(StubController(), forced=lambda item: False, mask=False,
                                       greedy=True, budget_mask=True)
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=False)
        copied = FirstReportAdvancer(StubController(), targets=("absent-target",), enabled=True)
        for observation in stream:
            for collector in (parent, shadow, copied):
                collector.decide(observation)
        self.assertEqual([canonical(t) for t in parent.records],
                         [canonical(t) for t in shadow.records],
                         "the disabled interceptor must delegate to the parent verbatim")
        self.assertEqual([canonical(t) for t in parent.records],
                         [canonical(t) for t in copied.records],
                         "the copied loop with no named target present must equal the parent")

    def test_parent_tape_has_no_override_key(self) -> None:
        collector = BudgetMaskedCollector(StubController(), forced=lambda item: False, mask=False,
                                          greedy=True, budget_mask=True)
        collector.decide(make_observation([make_item("t-a", age=1)]))
        self.assertNotIn("first_report_overrides", collector.records[0])


class TestOnlyNamedTarget(unittest.TestCase):
    def test_only_named_target_is_hit(self) -> None:
        items = [make_item("t-a", age=1), make_item("t-b", age=1), make_item("t-c", age=5)]
        observation = make_observation(items)
        parent = BudgetMaskedCollector(StubController(), forced=lambda item: False, mask=False,
                                       greedy=True, budget_mask=True)
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        asked_parent, actions_parent, _, _, _, tape_parent = parent.choose(observation)
        asked_shadow, actions_shadow, _, _, _, tape_shadow = shadow.choose(observation)
        self.assertEqual(actions_parent["t-a"], 0)
        self.assertEqual(actions_parent["t-c"], 1)
        self.assertEqual(actions_shadow["t-a"], actions_parent["t-a"])
        self.assertEqual(actions_shadow["t-c"], actions_parent["t-c"])
        self.assertEqual(actions_shadow["t-b"], 1)
        self.assertEqual(asked_parent, ["t-c"])
        self.assertEqual(asked_shadow, ["t-b", "t-c"])
        block = tape_shadow["first_report_overrides"]
        self.assertEqual(block["overrides_applied_this_step"], ["t-b"])
        note = block["opportunities"][0]
        self.assertEqual(note["outcome"], "overridden_hold_to_send")
        self.assertEqual(note["greedy_action"], "HOLD")
        self.assertLess(note["greedy_send_probability"], 0.5)
        self.assertTrue(note["override_applied"])
        # the parent's own tape fields are unchanged by the hook for every row
        for key in ("schema", "action_mask_schema", "step", "targets", "features", "actionable",
                    "masked", "forced", "send_probabilities", "send_executable_initial",
                    "send_block_reason_initial", "attempt_charge_bytes", "value",
                    "shadow_budget_before", "shadow_capacity_before"):
            self.assertEqual(tape_parent[key], tape_shadow[key], key)

    def test_one_override_per_target_then_original_logic(self) -> None:
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        shadow.decide(make_observation([make_item("t-b", age=1)], step=1))
        self.assertEqual(shadow.consumed, {"t-b"})
        # later steps keep the original greedy decisions; no second override, no note spam
        for step in range(2, 6):
            shadow.decide(make_observation(
                [make_item("t-b", age=1, sent_count=1, first_emission=False)], step=step))
        notes = [n for tape in shadow.records for n in
                 (tape.get("first_report_overrides") or {}).get("opportunities", [])]
        self.assertEqual(len(notes), 1, "only the first opportunity may be recorded")
        self.assertEqual(shadow.records[1]["actions"]["t-b"], 0)


class TestAlreadySent(unittest.TestCase):
    def test_already_sent_not_forced_again(self) -> None:
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        observation = make_observation([make_item("t-b", age=1, sent_count=1,
                                                  first_emission=False)])
        asked, actions, _, _, _, tape = shadow.choose(observation)
        self.assertEqual(asked, [])
        self.assertEqual(actions["t-b"], 0)
        note = tape["first_report_overrides"]["opportunities"][0]
        self.assertEqual(note["outcome"], "already_sent_no_duplicate")
        self.assertFalse(note["override_applied"])
        self.assertIn("t-b", shadow.consumed)


class TestSequentialBudget(unittest.TestCase):
    """The override must spend real credit in real candidate order, like any other send."""

    def _real_gate_walk(self, asked: list[str], items: dict[str, SendItem], spent: int,
                        credit: int) -> float:
        gate = CommBudget(bytes_per_episode=credit)
        gate.spent_bytes = spent
        for target in asked:
            charge = items[target].attempt_charge_bytes
            self.assertTrue(gate.can_send(charge),
                            f"the asked send {target} is not payable in the real gate")
            gate.charge(charge)
        return gate.remaining_bytes

    def test_override_behind_later_candidate_keeps_shadow_exact(self) -> None:
        # named target first, greedy HOLD, then a candidate greedy would send
        items = {"t-a": make_item("t-a", age=1, charge=3000),
                 "t-b": make_item("t-b", age=5, charge=3000)}
        observation = make_observation([items["t-a"], items["t-b"]], remaining=4000.0)
        shadow = FirstReportAdvancer(StubController(), targets=("t-a",), enabled=True)
        asked, actions, _, _, _, tape = shadow.choose(observation)
        # the override spends first, so t-b is masked by the real remaining credit
        self.assertEqual(asked, ["t-a"])
        self.assertEqual(tape["masked"], ["t-b"])
        self.assertEqual(tape["masked_reasons"]["t-b"], "budget_exhausted")
        self.assertEqual(tape["shadow_budget_after"], 1000.0)
        self.assertEqual(self._real_gate_walk(asked, items, spent=6000, credit=10000), 1000.0)

    def test_override_behind_earlier_send_keeps_shadow_exact(self) -> None:
        # an earlier candidate greedy-sends, then the named target's override follows it
        items = {"t-x": make_item("t-x", age=5, charge=3000),
                 "t-b": make_item("t-b", age=1, charge=3000)}
        observation = make_observation([items["t-x"], items["t-b"]], remaining=7000.0)
        parent = BudgetMaskedCollector(StubController(), forced=lambda item: False, mask=False,
                                       greedy=True, budget_mask=True)
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        asked_parent, _, _, _, _, tape_parent = parent.choose(observation)
        asked_shadow, _, _, _, _, tape_shadow = shadow.choose(observation)
        self.assertEqual(asked_parent, ["t-x"])
        self.assertEqual(asked_shadow, ["t-x", "t-b"])
        self.assertEqual(tape_shadow["shadow_budget_after"], 1000.0)
        self.assertEqual(self._real_gate_walk(asked_shadow, items, spent=3000, credit=10000),
                         1000.0)
        note = [n for n in tape_shadow["first_report_overrides"]["opportunities"]
                if n["target"] == "t-b"][0]
        self.assertTrue(note["override_applied"])


class TestNoLegalOpportunity(unittest.TestCase):
    def test_masked_named_target_never_overridden(self) -> None:
        items = {"t-a": make_item("t-a", age=1, charge=3000),
                 "t-b": make_item("t-b", age=5, charge=500)}
        observation = make_observation([items["t-a"], items["t-b"]], remaining=1000.0)
        shadow = FirstReportAdvancer(StubController(), targets=("t-a",), enabled=True)
        asked, actions, _, _, _, tape = shadow.choose(observation)
        self.assertEqual(asked, ["t-b"])
        self.assertEqual(actions["t-a"], 0)
        self.assertEqual(tape["masked"], ["t-a"])
        notes = {n["target"]: n for n in tape["first_report_overrides"]["opportunities"]}
        self.assertEqual(notes["t-a"]["outcome"], "no_legal_opportunity_no_override")
        self.assertEqual(notes["t-a"]["class"], "masked")

    def test_unpriceable_candidate_fails_closed_like_the_parent(self) -> None:
        broken = make_item("t-a", age=1, charge=3000)
        unpriced = SendItem(**{**broken.__dict__, "attempt_charge_bytes": None})
        observation = make_observation([unpriced])
        for collector in (BudgetMaskedCollector(StubController(), forced=lambda item: False,
                                                mask=False, greedy=True, budget_mask=True),
                          FirstReportAdvancer(StubController(), targets=("t-a",), enabled=True)):
            with self.assertRaises(RuntimeError):
                collector.choose(observation)


class TestNoRngTouch(unittest.TestCase):
    def test_override_draws_no_randomness(self) -> None:
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        before = torch.get_rng_state()
        shadow.decide(make_observation([make_item("t-b", age=1)]))
        after = torch.get_rng_state()
        self.assertTrue(torch.equal(before, after),
                        "the interceptor must not consume the torch RNG stream")

    def test_hook_source_references_no_rng_objects(self) -> None:
        import inspect

        source = inspect.getsource(FirstReportAdvancer._first_report_hook)
        for forbidden in ("random", "multinomial", "torch.rand", "np.random"):
            self.assertNotIn(forbidden, source)
        instance = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        for name, value in vars(instance).items():
            self.assertNotIsInstance(value, torch.Generator)
            self.assertFalse(type(value).__module__.startswith("numpy.random"), name)


class TestLedgerAndCap(unittest.TestCase):
    def test_cap_and_double_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            claims = Path(tmp) / "claims"
            logs = Path(tmp) / "logs"
            claims.mkdir()
            logs.mkdir()
            original_claims, original_logs = common.CLAIMS, common.LOGS
            common.CLAIMS, common.LOGS = claims, logs
            try:
                ledger = common.LaunchLedger()
                ledger.reserve("normal", "L", 1, "L")
                with self.assertRaises(RuntimeError):
                    ledger.reserve("normal", "L", 1, "L")  # double assignment of one cell
                for index in range(2, common.LAUNCH_CAP + 1):
                    ledger.reserve("normal", f"p{index}", 1, f"p{index}")
                with self.assertRaises(RuntimeError):
                    ledger.reserve("reserve", "extra", 1, "extra")
                report = common.ledger_report()
                self.assertTrue(report["ledgers_agree"])
                self.assertEqual(report["used"], common.LAUNCH_CAP)
            finally:
                common.CLAIMS, common.LOGS = original_claims, original_logs


class TestProvenanceAndScans(unittest.TestCase):
    def test_override_set_matches_round25_frozen_set(self) -> None:
        import r25_common

        self.assertEqual(common.STAGE_B_INHERITED_OVERRIDES, r25_common.STAGE_B_INHERITED_OVERRIDES)

    def test_no_optimizer_anywhere_in_the_harness(self) -> None:
        """0 optimizer.step, including synthetic updates: an AST scan, prose ignored."""
        import ast

        forbidden_calls = {"optimizer", "Adam", "SGD", "RMSprop", "Adagrad", "backward",
                           "torch.optim"}
        for name in common.HARNESS_MODULES:
            tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = [alias.name for alias in node.names]
                    modules.append(getattr(node, "module", None) or "")
                    for module in modules:
                        self.assertFalse(module.startswith("torch.optim"), f"{name}: {module}")
                elif isinstance(node, ast.Call):
                    text = ast.dump(node.func)
                    for token in forbidden_calls:
                        self.assertNotIn(f"'{token}'", text, f"{name} calls {token}")
                elif isinstance(node, ast.Attribute) and node.attr in {
                        "backward", "zero_grad"}:
                    self.fail(f"{name} touches gradient plumbing: .{node.attr}")

    def test_cell_naming_and_paths(self) -> None:
        paths = common.cell_paths("E05", 1)
        self.assertEqual(paths["row"].name, "r27_E05_s1400031_a1.json")
        self.assertTrue(paths["tape"].name.endswith("_tape.jsonl"))
        self.assertEqual(common.PATH_ORDER, ("L", "L0", "E0", "E5", "E05"))

    def test_intervention_evidence_join(self) -> None:
        import r27_run

        class FakeCollector:
            records = [{
                "first_report_overrides": {"opportunities": [
                    {"step": 19, "target": "piloc-e1-target-0", "override_applied": True,
                     "outcome": "overridden_hold_to_send"}]}}]

        transcript = [
            {"kind": "send", "step": 19, "target": "piloc-e1-target-0", "revision": 1,
             "cost_bytes": 2682, "packets": 4, "packets_lost": 2, "subscribers": 6, "first": True,
             "send_reason": "policy", "ready_step": 19},
            {"kind": "deliver", "step": 19, "target": "piloc-e1-target-0", "agent": "2",
             "accepted": True, "reason": "accepted", "sent_step": 19},
            {"kind": "deliver", "step": 19, "target": "piloc-e1-target-0", "agent": "3",
             "accepted": False, "reason": "delivery_superseded", "sent_step": 19},
            {"kind": "refused", "step": 40, "target": "piloc-e1-target-0",
             "reason": "budget_exhausted", "cost_bytes": 2682, "send_reason": "policy"},
        ]
        evidence = r27_run._intervention_evidence("E0", FakeCollector(), transcript,
                                                  ("piloc-e1-target-0",))
        entry = evidence["per_target"]["piloc-e1-target-0"]
        self.assertTrue(entry["override_applied"])
        self.assertEqual(entry["override_steps"], [19])
        self.assertEqual(len(entry["real_sends"]), 1)
        self.assertEqual(entry["real_sends"][0]["packets_lost"], 2)
        self.assertEqual(len(entry["deliveries"]), 2)
        self.assertEqual(entry["real_refusals"][0]["reason"], "budget_exhausted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
