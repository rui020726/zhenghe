"""ROUND25 零环境预检：实际推理路径的纯计算检查（0 环境启动、0 optimizer.step）。

执行依据 §五：模型加载、greedy 解码、特征、顺序预算 mask、费用汇总、唯一命名、缺格聚合
失败关闭——全部在合成输入上用生产代码路径完成；同时核验启动账本原子性与硬上限。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r25_common as common
import r25_check as checker
import r25_run as runner
import r25_stats as stats
from integrated_sar.comm_policy import SendItem, SendObservation


def _item(target_id: str, charge: int, *, executable: bool = True, reason: str = "allowed",
          first: bool = True, revision: int = 1) -> SendItem:
    return SendItem(
        target_id=target_id, ready_step=10, age_steps=5, pending_event=True, changed_view=True,
        first_emission=first, revision=revision, revision_delta=1, stage_index=0,
        stage_changed=False, progress=0.5, progress_delta=0.1, requirements_changed=False,
        members_changed=False, deadline=50.0, remaining_time=40.0, sent_count=0,
        last_sent_step=None, since_last_emission_time=None, has_future_requirements=True,
        attempt_charge_bytes=charge, send_executable=executable, send_block_reason=reason)


class StubController:
    """A controller with the real forward signature and a fixed send logit.

    The action layout is the frozen one: logit index 1 is SEND, index 0 is HOLD.
    """

    def __init__(self, send_logit: float) -> None:
        self.send_logit = float(send_logit)

    def forward(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(matrix.shape[0])
        logits = torch.tensor([[-self.send_logit, self.send_logit]] * rows, dtype=torch.float32)
        values = torch.zeros(rows, dtype=torch.float32)
        return logits, values


def _observation(items: tuple[SendItem, ...], *, remaining: float, subscribers: int = 4,
                 step: int = 42) -> SendObservation:
    return SendObservation(step=step, current_time=float(step), items=items,
                           remaining_budget_bytes=remaining,
                           budget_bytes_per_episode=common.COMM_BUDGET_BYTES,
                           capacity_remaining=None, backlog_targets=len(items),
                           subscribers=subscribers)


def main() -> int:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(name)

    # ---------------------------------------------------------------- A. model load and identity
    try:
        controller, identity = runner.load_candidate_controller()
        record("model_load_ep128_seed33",
               identity["training_seed"] == 33 and identity["episodes_done"] == 128,
               {key: identity[key] for key in ("training_seed", "episodes_done",
                                               "lambda_final", "checkpoint_sha256")})
        recorded = json.loads((common.OUTPUT_DIR / "CANDIDATE_FREEZE.json").read_text(
            encoding="utf-8"))
        expected = recorded["candidate"]["checkpoint"]["sha256"]
        record("checkpoint_sha256_matches_freeze", identity["checkpoint_sha256"] == expected,
               {"live": identity["checkpoint_sha256"], "frozen": expected})
        import torch as _torch

        payload = _torch.load(common.LEARNED_CHECKPOINT, map_location="cpu", weights_only=False)
        record("checkpoint_freeze_payload_matches_round24",
               payload["freeze"] == common.round24_freeze_payload(), payload["freeze"])
        record("checkpoint_training_history_present",
               int(payload["training_state"]["optimizer_steps"]) > 0,
               payload["training_state"]["optimizer_steps"])
        controller2, identity2 = runner.load_candidate_controller()
        record("two_loads_identical",
               identity["controller_parameter_summary"] == identity2["controller_parameter_summary"],
               "parameter summaries compared field by field")
        piloc = common.sha256_file(common.CHECKPOINT_FILES[0])
        ctde = common.sha256_file(common.CHECKPOINT_FILES[1])
        recorded_weights = recorded["frozen_search_and_allocation_weights"]
        record("piloc_weight_frozen",
               piloc == recorded_weights["runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth"]["recorded_sha256"], piloc)
        record("ctde_weight_frozen",
               ctde == recorded_weights["runs/joint_eval_20260916/ctde_final_7_1.pth"]["recorded_sha256"], ctde)
    except Exception as error:  # noqa: BLE001
        record("model_load_ep128_seed33", False, f"{type(error).__name__}: {error}")

    # ------------------------------------------------------- B. inference path (synthetic input)
    try:
        from budget_masked import BudgetMaskedCollector

        # B1: the real frozen controller on a synthetic observation (real features, real gate).
        items = (_item("t-a", 900), _item("t-b", 900), _item("t-c", 900))
        observation = _observation(items, remaining=20000.0, subscribers=4)
        collector = BudgetMaskedCollector(controller, forced=lambda item: False, mask=False,
                                          greedy=True, budget_mask=True)
        _asked, actions, free, _log_prob, _value, tape = collector.choose(observation)
        rows = len(tape["features"])
        record("feature_rows_match_candidates", rows == len(items), rows)
        record("feature_layout_19d", all(len(row) == 19 for row in tape["features"]),
               [len(row) for row in tape["features"]])
        record("tape_schemas", tape["schema"] == "onpolicy-tape/2"
               and tape["action_mask_schema"] == "budget-sequential-mask/1",
               {"tape": tape["schema"], "mask": tape["action_mask_schema"]})
        matrix = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
        with torch.no_grad():
            probabilities = torch.softmax(controller.forward(matrix)[0], dim=-1)
        gaps = {target: abs(float(probabilities[index, 1].item())
                            - float(tape["send_probabilities"].get(target, float("nan"))))
                for index, target in enumerate(tape["targets"])
                if target in tape["actionable"]}
        record("recomputed_probabilities_match_tape",
               bool(gaps) and max(gaps.values()) <= 1e-6, gaps)
        greedy_ok = all(
            int(tape["actions"][target]) == (1 if float(probabilities[index, 1].item())
                                             >= float(probabilities[index, 0].item()) else 0)
            for index, target in enumerate(tape["targets"]) if target in tape["actionable"])
        record("greedy_matches_argmax_tie_to_send", greedy_ok,
               {t: int(tape["actions"][t]) for t in tape["targets"]})
        record("masked_candidates_are_hold",
               all(int(tape["actions"][target]) == 0 for target in tape["masked"]),
               tape["masked"])
        record("free_equals_actionable", list(tape["free"]) == list(tape["actionable"]),
               {"free": tape["free"], "actionable": tape["actionable"]})
        charge_sum = sum(int(tape["attempt_charge_bytes"][target])
                         for target in tape["targets"] if int(tape["actions"][target]) == 1)
        record("shadow_charges_only_sends",
               abs(float(tape["shadow_budget_after"])
                   - (float(tape["shadow_budget_before"]) - charge_sum)) <= 1e-9,
               {"charged": charge_sum, "before": tape["shadow_budget_before"],
                "after": tape["shadow_budget_after"]})
        record("charges_are_the_priced_candidates",
               all(int(tape["attempt_charge_bytes"][t]) == 900 for t in tape["targets"]),
               {t: int(tape["attempt_charge_bytes"][t]) for t in tape["targets"]})

        # B2: deterministic stub -- the SEND path, sequential masking and refusal reasons.
        stub = StubController(send_logit=2.0)   # p(SEND) ~= 0.88: greedy sends
        collector = BudgetMaskedCollector(stub, forced=lambda item: False, mask=False,
                                          greedy=True, budget_mask=True)
        items = (_item("t-a", 5000), _item("t-b", 5000), _item("t-c", 5000))
        observation = _observation(items, remaining=12000.0, subscribers=4)
        _asked, actions, free, _lp, _v, tape = collector.choose(observation)
        record("stub_sends_until_credit_runs_out",
               actions == {"t-a": 1, "t-b": 1, "t-c": 0}
               and set(free) == {"t-a", "t-b"} and tape["masked"] == ["t-c"],
               {"actions": actions, "free": list(free), "masked": tape["masked"],
                "masked_reasons": tape["masked_reasons"]})
        record("sequential_mask_reason_budget_exhausted",
               tape["masked_reasons"].get("t-c") == "budget_exhausted",
               tape["masked_reasons"])
        record("shadow_after_matches_remaining_minus_sends",
               abs(float(tape["shadow_budget_after"]) - (12000.0 - 2 * 5000.0)) <= 1e-9,
               tape["shadow_budget_after"])

        # B3: a candidate priced beyond the remaining credit is masked from the start.
        items = (_item("t-a", 900), _item("t-huge", 999999))
        observation = _observation(items, remaining=12000.0, subscribers=4)
        collector = BudgetMaskedCollector(StubController(2.0), forced=lambda item: False,
                                          mask=False, greedy=True, budget_mask=True)
        _asked, actions, free, _lp, _v, tape = collector.choose(observation)
        record("unaffordable_candidate_masked",
               actions == {"t-a": 1, "t-huge": 0} and "t-huge" in tape["masked"]
               and tape["masked_reasons"]["t-huge"] == "budget_exhausted",
               {"actions": actions, "masked": tape["masked"],
                "reasons": tape["masked_reasons"]})

        # B4: the HOLD side of the greedy decode on the stub network.
        collector = BudgetMaskedCollector(StubController(-2.0), forced=lambda item: False,
                                          mask=False, greedy=True, budget_mask=True)
        observation = _observation((_item("t-a", 900), _item("t-b", 900)), remaining=20000.0)
        _asked, actions, free, _lp, _v, tape = collector.choose(observation)
        record("stub_holds_on_low_send_probability",
               actions == {"t-a": 0, "t-b": 0} and free == ["t-a", "t-b"]
               and float(tape["shadow_budget_after"]) == float(tape["shadow_budget_before"]),
               {"actions": actions, "free": list(free)})
    except Exception as error:  # noqa: BLE001
        record("inference_path_synthetic", False, f"{type(error).__name__}: {error}")

    # ---------------------------------------------------------------- C. naming and ledger
    paths = {arm: {scene: common.row_path(arm, scene, 1)
                   for scene in common.NEW_SCENARIOS} for arm in common.ARM_ORDER}
    flat = [str(path) for arm in paths for path in paths[arm].values()]
    record("row_paths_unique", len(flat) == len(set(flat)) == 192, len(set(flat)))
    record("attempt_two_has_its_own_path",
           common.row_path("immediate", 1400000, 1) != common.row_path("immediate", 1400000, 2)
           and common.cell_paths("immediate", 1400000, 2)["row"].name
           == "r25_immediate_s1400000_a2.json",
           common.cell_paths("immediate", 1400000, 2)["row"].name)
    record("real_ledger_empty_before_launch",
           common.launch_count() == 0 and common.ledger_total() == 0,
           {"claims": common.launch_count(), "per_process": common.ledger_total()})
    record("no_round_artefacts_yet",
           not any(common.ROWS.glob("r25_*")) and not any(common.LOGS.glob("r25_*")),
           {"rows": len(list(common.ROWS.glob('r25_*'))), "logs": len(list(common.LOGS.glob('r25_*')))})

    with tempfile.TemporaryDirectory() as tmp:
        original = common.CLAIMS
        common.CLAIMS = Path(tmp) / "claims"
        try:
            common.CLAIMS.mkdir(parents=True, exist_ok=True)
            record("claim_is_atomic_first_wins", common.claim_cell("immediate", 1400000, 1, "x")
                   and not common.claim_cell("immediate", 1400000, 1, "x"), "second claim refused")
            ledger = common.LaunchLedger()
            ledger.path = common.LOGS / f"START_LEDGER_p{__import__('os').getpid()}.json"
            try:
                ledger.reserve("normal", "immediate", 1400000, 1, "immediate_s1400000")
                record("reserve_refuses_double_booking", False, "reserve did not refuse")
            except RuntimeError:
                record("reserve_refuses_double_booking", True, "double booking refused")
            for index in range(common.LAUNCH_CAP):
                (common.CLAIMS / f"filler_{index}.json").write_text("{}\n", encoding="utf-8")
            try:
                ledger.reserve("normal", "immediate", 1400001, 1, "immediate_s1400001")
                record("hard_cap_enforced", False, "reserve did not refuse at the cap")
            except RuntimeError as error:
                record("hard_cap_enforced", "cap" in str(error), str(error))
        finally:
            common.CLAIMS = original
            ledger_file = common.LOGS / f"START_LEDGER_p{__import__('os').getpid()}.json"
            if ledger_file.exists():
                ledger_file.unlink()
    record("real_ledger_still_empty",
           common.launch_count() == 0 and common.ledger_total() == 0,
           {"claims": common.launch_count(), "per_process": common.ledger_total()})

    # ---------------------------------------------------------------- D. aggregation and statistics
    def synthetic_row(arm: str, scene: int, on_time: int, spent: int) -> dict[str, Any]:
        return {"arm": arm, "scenario_seed": scene, "on_time": on_time, "spent_bytes": spent,
                "completed": on_time, "expired": 0, "discovered": 6, "emissions": 3,
                "budget_refused_sends": 1, "capacity_refused_sends": 0,
                "budget_used_fraction": spent / common.COMM_BUDGET_BYTES,
                "goal_identity": {"on_time_goal_ids": [f"g{index}" for index in range(on_time)]}}

    full = {arm: {} for arm in common.ARM_ORDER}
    for index in range(64):
        scene = 1400000 + index
        full[common.CANDIDATE_ARM][scene] = synthetic_row(common.CANDIDATE_ARM, scene, 5, 12000)
        full["immediate"][scene] = synthetic_row("immediate", scene, 5, 18000)
        full[common.PRIMARY_COST_COMPARATOR][scene] = synthetic_row(
            common.PRIMARY_COST_COMPARATOR, scene, 5, 15000)
    metrics = checker.absolute_metrics(full)
    candidate_metrics = metrics["arms"][common.CANDIDATE_ARM]
    record("absolute_metrics_sums",
           candidate_metrics["on_time_total"] == 320
           and candidate_metrics["spent_bytes_total"] == 768000
           and candidate_metrics["on_time_mean_per_scene"] == 5.0
           and candidate_metrics["spent_bytes_mean_per_scene"] == 12000.0,
           {key: candidate_metrics[key] for key in ("on_time_total", "spent_bytes_total",
                                                    "on_time_mean_per_scene",
                                                    "spent_bytes_mean_per_scene")})

    paired = stats.paired_table(full)
    late = paired["comparisons"][common.PRIMARY_COST_COMPARATOR]
    record("synthetic_primary_dC",
           late["cost"]["mean"] == -3000.0 and late["cost"]["ci_high"] < 0, late["cost"])
    record("synthetic_counts_partition",
           all(sum(1 for item in late["per_scene"] if item["counts"][key]) == 64
               for key in ("cost_lower", "task_tied", "joint_dominant")),
           {key: sum(1 for item in late["per_scene"] if item["counts"][key])
            for key in ("cost_lower", "task_tied", "joint_dominant")})
    record("zero_event_bound_value",
           abs(late["zero_event_upper_bound_95"] - (1 - 0.05 ** (1 / 64))) < 1e-12,
           late["zero_event_upper_bound_95"])
    record("shared_matrix_matches_plan",
           paired["shared_resampling"]["matrix_sha256"]
           == json.loads((common.OUTPUT_DIR / "STATISTICS_PLAN.json").read_text(
               encoding="utf-8"))["resampling"]["matrix_sha256"],
           paired["shared_resampling"]["matrix_sha256"])

    broken = {arm: dict(scenes) for arm, scenes in full.items()}
    broken[common.CANDIDATE_ARM].pop(1400063)
    try:
        stats.paired_table(broken)
        record("aggregation_hard_fails_on_missing_cell", False, "no failure raised")
    except SystemExit:
        record("aggregation_hard_fails_on_missing_cell", True, "SystemExit on a missing cell")

    if (common.OUTPUT_DIR / "PAIRED_COMPARISON.json").exists():
        record("statistics_artefact_absent_before_launch", False,
               "PAIRED_COMPARISON.json already exists before the batch")
    else:
        record("statistics_artefact_absent_before_launch", True, "not written yet")
    with tempfile.TemporaryDirectory() as tmp:
        original = common.OUTPUT_DIR
        common.OUTPUT_DIR = Path(tmp)
        try:
            (Path(tmp) / "PAIRED_COMPARISON.json").write_text(json.dumps(
                {"shared_resampling": {"matrix_sha256": "deadbeef"}}), encoding="utf-8")
            try:
                stats.read_statistics()
                record("statistics_reader_rejects_foreign_matrix", False, "no failure raised")
            except RuntimeError:
                record("statistics_reader_rejects_foreign_matrix", True,
                       "reader refused a statistics file with a foreign resampling matrix")
        finally:
            common.OUTPUT_DIR = original

    # ---------------------------------------------------------------- E. zero optimizer, units
    # The scanner's own pattern strings are not code; the self-exclusion is registered here.
    self_excluded = {"r25_preflight.py"}
    offenders = []
    for name in common.HARNESS_MODULES:
        if name in self_excluded:
            continue
        path = common.HERE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "optimizer.step(" in text or "torch.optim" in text:
            offenders.append(name)
    record("no_optimizer_in_round_harness", not offenders,
           {"offenders": offenders, "self_excluded": sorted(self_excluded),
            "note": "the round harness never constructs or steps an optimizer; "
                    "v2_trainer.load_checkpoint only restores Adam state"})
    record("cost_unit_registered",
           common.B0_BYTES == 10000.0 and common.COMM_BUDGET_BYTES == 29300,
           {"b0": common.B0_BYTES, "budget": common.COMM_BUDGET_BYTES})

    payload = {
        "round": "25",
        "section": "zero-environment, zero-optimizer preflight of the actual inference path",
        "launches_used": common.launch_count(),
        "environment_launches": 0,
        "optimizer_steps": 0,
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }
    (common.OUTPUT_DIR / "RUN_PREFLIGHT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "failures": failures,
                      "checks": len(checks)}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
