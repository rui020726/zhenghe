"""ROUND28 零环境、零 optimizer 预检：干预器复用核验 + 冻结算法在合成数据上的行为 + 账本/扫描。

Zero environment launches. This round reuses ROUND27's validated interceptor, so the synthetic
suite here proves three things beyond ROUND27's own suite: the reused class still satisfies the
six required properties, the frozen selection algorithm behaves as documented on synthetic tapes,
and this round's harness never touches an optimizer or a driver RNG. The full production unittest
suite is deliberately not re-run (scope registered in ROUND27_SCOPE_REGISTRATION.json).
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import torch

import r28_common as common
from budget_masked import SEND, BudgetMaskedCollector
from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_policy import SendItem, SendObservation
from r27_intervention import FirstReportAdvancer

HERE = Path(__file__).resolve().parent


class StubController:
    """Deterministic logits driven by the first feature (age_steps): SEND logit = age - 3."""

    def forward(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ages = matrix[:, 0]
        logits = torch.stack([torch.zeros_like(ages), ages - 3.0], dim=1)
        return logits, torch.zeros((matrix.shape[0], 1))

    def parameter_summary(self) -> dict:
        return {"stub": True, "parameters": 0}


def make_item(target: str, *, age: int, charge: int = 3000, sent_count: int = 0,
              first_emission: bool = True) -> SendItem:
    return SendItem(
        target_id=target, ready_step=1, age_steps=age, pending_event=True, changed_view=True,
        first_emission=first_emission, revision=1, revision_delta=0, stage_index=0,
        stage_changed=False, progress=0.0, progress_delta=0.0, requirements_changed=False,
        members_changed=False, deadline=100.0, remaining_time=50.0, sent_count=sent_count,
        last_sent_step=None, since_last_emission_time=None, has_future_requirements=False,
        attempt_charge_bytes=charge, send_executable=True, send_block_reason="allowed",
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


class TestReusedInterceptor(unittest.TestCase):
    """ROUND27's interceptor, re-checked in this round's harness."""

    def test_only_named_target_is_hit(self) -> None:
        observation = make_observation([make_item("t-a", age=1), make_item("t-b", age=1),
                                        make_item("t-c", age=5)])
        parent = BudgetMaskedCollector(StubController(), forced=lambda item: False, mask=False,
                                       greedy=True, budget_mask=True)
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        asked_parent, actions_parent, _, _, _, _ = parent.choose(observation)
        asked_shadow, actions_shadow, _, _, _, tape_shadow = shadow.choose(observation)
        self.assertEqual(actions_parent["t-a"], 0)
        self.assertEqual(actions_parent["t-c"], 1)
        self.assertEqual(actions_shadow["t-a"], 0)
        self.assertEqual(actions_shadow["t-c"], 1)
        self.assertEqual(actions_shadow["t-b"], 1)
        self.assertEqual(asked_parent, ["t-c"])
        self.assertEqual(asked_shadow, ["t-b", "t-c"])
        self.assertEqual(tape_shadow["first_report_overrides"]["overrides_applied_this_step"],
                         ["t-b"])

    def test_disabled_delegates_verbatim(self) -> None:
        stream = [make_observation([make_item("t-a", age=1), make_item("t-b", age=5)], step=1),
                  make_observation([make_item("t-a", age=2), make_item("t-c", age=7)], step=2,
                                   remaining=7000.0)]
        parent = BudgetMaskedCollector(StubController(), forced=lambda item: False, mask=False,
                                       greedy=True, budget_mask=True)
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=False)
        absent = FirstReportAdvancer(StubController(), targets=("absent",), enabled=True)
        for observation in stream:
            for collector in (parent, shadow, absent):
                collector.decide(observation)
        self.assertEqual([canonical(t) for t in parent.records],
                         [canonical(t) for t in shadow.records])
        self.assertEqual([canonical(t) for t in parent.records],
                         [canonical(t) for t in absent.records])

    def test_already_sent_not_forced_again(self) -> None:
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        observation = make_observation([make_item("t-b", age=1, sent_count=1,
                                                  first_emission=False)])
        asked, actions, _, _, _, tape = shadow.choose(observation)
        self.assertEqual(asked, [])
        self.assertEqual(actions["t-b"], 0)
        note = tape["first_report_overrides"]["opportunities"][0]
        self.assertEqual(note["outcome"], "already_sent_no_duplicate")

    def test_sequential_budget_exact(self) -> None:
        items = {"t-a": make_item("t-a", age=1, charge=3000),
                 "t-b": make_item("t-b", age=5, charge=3000)}
        observation = make_observation([items["t-a"], items["t-b"]], remaining=4000.0)
        shadow = FirstReportAdvancer(StubController(), targets=("t-a",), enabled=True)
        asked, _, _, _, _, tape = shadow.choose(observation)
        self.assertEqual(asked, ["t-a"])
        self.assertEqual(tape["masked"], ["t-b"])
        self.assertEqual(tape["shadow_budget_after"], 1000.0)
        gate = CommBudget(bytes_per_episode=10000)
        gate.spent_bytes = 6000
        for target in asked:
            self.assertTrue(gate.can_send(items[target].attempt_charge_bytes))
            gate.charge(items[target].attempt_charge_bytes)
        self.assertEqual(gate.remaining_bytes, 1000.0)

    def test_no_legal_opportunity_no_bypass(self) -> None:
        observation = make_observation([make_item("t-a", age=1, charge=3000),
                                        make_item("t-b", age=5, charge=500)], remaining=1000.0)
        shadow = FirstReportAdvancer(StubController(), targets=("t-a",), enabled=True)
        asked, _, _, _, _, tape = shadow.choose(observation)
        self.assertEqual(asked, ["t-b"])
        self.assertEqual(tape["masked"], ["t-a"])
        self.assertEqual(tape["first_report_overrides"]["opportunities"][0]["outcome"],
                         "no_legal_opportunity_no_override")

    def test_no_rng_consumption(self) -> None:
        shadow = FirstReportAdvancer(StubController(), targets=("t-b",), enabled=True)
        before = torch.get_rng_state()
        shadow.decide(make_observation([make_item("t-b", age=1)]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class TestSelectionAlgorithm(unittest.TestCase):
    """The frozen selector's documented behaviour, on synthetic baseline tapes."""

    def _write_baseline(self, scene: int, tape_rows: list[dict], discovered: tuple[str, ...]):
        """Synthetic baselines in a temp directory: never the real rows/logs of this round."""
        import r28_select

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self._original = (common.ROWS, common.LOGS)
        common.ROWS, common.LOGS = base / "rows", base / "logs"
        self.addCleanup(lambda: setattr(common, "ROWS", self._original[0]))
        self.addCleanup(lambda: setattr(common, "LOGS", self._original[1]))
        paths = common.cell_paths("baseline", scene)
        paths["row"].parent.mkdir(parents=True, exist_ok=True)
        paths["tape"].parent.mkdir(parents=True, exist_ok=True)
        with paths["tape"].open("w", encoding="utf-8", newline="\n") as handle:
            for row in tape_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with paths["lifecycle"].open("w", encoding="utf-8", newline="\n") as handle:
            for target in discovered:
                handle.write(json.dumps({"target_id": target, "event_type": "detection",
                                         "timestamp": 1.0, "detail": {"deadline_offset": 50.0}},
                                        ensure_ascii=False) + "\n")
        row = {"kind": "baseline", "scenario_seed": scene, "target": None,
               "code_identity_freeze_sha256": "synthetic-test-identity",
               "model_identity": {"stub": True},
               "artefacts": {"tape": str(paths["tape"]), "lifecycle": str(paths["lifecycle"])}}
        paths["row"].write_text(json.dumps(row) + "\n", encoding="utf-8")
        return r28_select.select_scene(scene)

    def _row(self, step: int, targets: list[str], actions: dict[str, int],
             actionable: list[str], masked: list[str] | None = None) -> dict:
        features = [[float(index)] for index, _ in enumerate(targets)]
        shadow = {target: {"budget_before": 10000.0, "honours_this_candidate": True}
                  for target in targets}
        return {"step": step, "targets": list(targets), "features": features,
                "actions": dict(actions), "free": list(actionable), "actionable": list(actionable),
                "forced": [], "masked": list(masked or []), "masked_reasons": {},
                "attempt_charge_bytes": {target: 3000 for target in targets},
                "candidate_shadow": shadow,
                "send_probabilities": {target: 0.4 for target in targets}}

    def test_hold_at_first_payable_opportunity_is_a_point(self) -> None:
        entry = self._write_baseline(
            1300000,
            [self._row(5, ["piloc-e1-target-0", "piloc-e1-target-1"], {"piloc-e1-target-0": 0, "piloc-e1-target-1": 0}, ["piloc-e1-target-0", "piloc-e1-target-1"],
                       masked=[])],
            ("piloc-e1-target-0", "piloc-e1-target-1"))
        self.assertEqual(entry["selected"], 2)
        self.assertEqual(sorted(p["target"] for p in entry["intervention_points"]), ["piloc-e1-target-0", "piloc-e1-target-1"])
        point = entry["intervention_points"][0]
        self.assertEqual(point["step"], 5)
        self.assertEqual(point["original_action"], "HOLD")
        self.assertEqual(point["attempt_charge_bytes"], 3000)

    def test_send_at_first_payable_opportunity_is_excluded(self) -> None:
        entry = self._write_baseline(
            1300000,
            [self._row(3, ["piloc-e1-target-0"], {"piloc-e1-target-0": 1}, ["piloc-e1-target-0"])],
            ("piloc-e1-target-0",))
        self.assertEqual(entry["selected"], 0)
        self.assertEqual(entry["exclusions"][0]["reason"], "first_legal_opportunity_greedy_send")

    def test_never_payable_is_excluded_and_masked_steps_are_skipped(self) -> None:
        entry = self._write_baseline(
            1300000,
            [self._row(1, ["piloc-e1-target-0"], {"piloc-e1-target-0": 0}, [], masked=["piloc-e1-target-0"]),
             self._row(2, ["piloc-e1-target-0"], {"piloc-e1-target-0": 0}, [], masked=["piloc-e1-target-0"])],
            ("piloc-e1-target-0",))
        self.assertEqual(entry["selected"], 0)
        self.assertEqual(entry["exclusions"][0]["reason"], "no_legal_payable_opportunity")

    def test_never_discovered_is_excluded_without_being_a_candidate(self) -> None:
        entry = self._write_baseline(1300000, [self._row(1, ["piloc-e1-target-0"], {"piloc-e1-target-0": 0}, ["piloc-e1-target-0"])],
                                     ("piloc-e1-target-0",))
        reasons = {e["target"]: e["reason"] for e in entry["exclusions"]}
        self.assertEqual(reasons["piloc-e1-target-1"], "never_discovered")
        self.assertEqual(reasons["piloc-e1-target-5"], "never_discovered")
        self.assertEqual([e["target"] for e in entry["exclusions"]],
                         ["piloc-e1-target-1", "piloc-e1-target-2", "piloc-e1-target-3",
                          "piloc-e1-target-4", "piloc-e1-target-5"])

    def test_detected_but_never_offered_is_excluded_as_terminal(self) -> None:
        entry = self._write_baseline(1300000, [self._row(1, ["piloc-e1-target-0"], {"piloc-e1-target-0": 0}, ["piloc-e1-target-0"])],
                                     ("piloc-e1-target-0", "piloc-e1-target-1"))
        reasons = {e["target"]: e["reason"] for e in entry["exclusions"]}
        self.assertEqual(reasons["piloc-e1-target-1"], "terminal_before_first_candidate")

    def test_first_payable_beats_earlier_masked_steps(self) -> None:
        entry = self._write_baseline(
            1300000,
            [self._row(1, ["piloc-e1-target-0"], {"piloc-e1-target-0": 0}, [], masked=["piloc-e1-target-0"]),
             self._row(2, ["piloc-e1-target-0"], {"piloc-e1-target-0": 0}, [], masked=["piloc-e1-target-0"]),
             self._row(7, ["piloc-e1-target-0"], {"piloc-e1-target-0": 0}, ["piloc-e1-target-0"])],
            ("piloc-e1-target-0",))
        self.assertEqual(entry["selected"], 1)
        self.assertEqual(entry["intervention_points"][0]["step"], 7)

    def test_cap_is_enforced(self) -> None:
        rows = [self._row(step, [f"piloc-e1-target-{index}" for index in range(3)],
                          {f"piloc-e1-target-{index}": 0 for index in range(3)},
                          [f"piloc-e1-target-{index}" for index in range(3)])
                for step in range(1, 2)]
        original = common.MAX_TARGETS_PER_SCENE
        common.MAX_TARGETS_PER_SCENE = 2
        try:
            entry = self._write_baseline(1300000, rows,
                                         tuple(f"piloc-e1-target-{i}" for i in range(3)))
            self.assertFalse(entry["cap_check"]["within_cap"])
        finally:
            common.MAX_TARGETS_PER_SCENE = original

    def test_goal_id_space_guard(self) -> None:
        rows = [self._row(1, ["piloc-e1-target-7"], {"piloc-e1-target-7": 0},
                          ["piloc-e1-target-7"])]
        with self.assertRaises(RuntimeError):
            self._write_baseline(1300000, rows, ("piloc-e1-target-7",))


class TestLedgerAndScans(unittest.TestCase):
    def test_cap_and_double_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            claims, logs = Path(tmp) / "claims", Path(tmp) / "logs"
            claims.mkdir()
            logs.mkdir()
            original_claims, original_logs = common.CLAIMS, common.LOGS
            common.CLAIMS, common.LOGS = claims, logs
            try:
                ledger = common.LaunchLedger()
                ledger.reserve("baseline", 1300000, None, 1, "b")
                with self.assertRaises(RuntimeError):
                    ledger.reserve("baseline", 1300000, None, 1, "b")
                for index in range(2, common.LAUNCH_CAP + 1):
                    ledger.reserve("intervention", 1300000, f"piloc-e1-target-{index}", 1, f"p{index}")
                with self.assertRaises(RuntimeError):
                    ledger.reserve("intervention", 1300000, "extra", 1, "extra")
                report = common.ledger_report()
                self.assertTrue(report["ledgers_agree"])
                self.assertEqual(report["used"], common.LAUNCH_CAP)
            finally:
                common.CLAIMS, common.LOGS = original_claims, original_logs

    def test_no_optimizer_in_this_rounds_harness(self) -> None:
        forbidden = ("optimizer", "Adam", "SGD", "RMSprop", "Adagrad", "backward", "zero_grad")
        for name in common.HARNESS_MODULES:
            source = (HERE / name).read_text(encoding="utf-8") if (HERE / name).exists() else ""
            tree = ast.parse(source) if source else ast.parse("")
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = [alias.name for alias in node.names]
                    modules.append(getattr(node, "module", None) or "")
                    for module in modules:
                        self.assertFalse(module.startswith("torch.optim"), f"{name}: {module}")
                elif isinstance(node, ast.Call):
                    text = ast.dump(node.func)
                    for token in forbidden:
                        self.assertNotIn(f"'{token}'", text, f"{name} calls {token}")
                elif isinstance(node, ast.Attribute) and node.attr in {"backward", "zero_grad"}:
                    self.fail(f"{name} touches gradient plumbing: .{node.attr}")

    def test_override_set_matches_round27_frozen_set(self) -> None:
        import r27_common

        self.assertEqual(common.STAGE_B_INHERITED_OVERRIDES, r27_common.STAGE_B_INHERITED_OVERRIDES)

    def test_cell_naming(self) -> None:
        self.assertEqual(common.cell_name("baseline", 1300000), "r28_baseline_s1300000")
        self.assertEqual(common.cell_name("disabled", 1300000), "r28_disabled_s1300000")
        self.assertEqual(common.cell_name("intervention", 1300003, "piloc-e1-target-2"),
                         "r28_i_s1300003_piloc-e1-target-2")
        self.assertEqual(common.NORMAL_LAUNCHES, 113)
        self.assertEqual(common.LAUNCH_CAP, 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
