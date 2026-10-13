#!/usr/bin/env python
"""Narrow ROUND71 trial tests: STATIC / SPY / PURE ARITHMETIC only.

Zero consumption by construction: no real or synthetic neural forward on a real model, no
real-data fit, no optimizer step, no autograd on a real controller, no .pt open, no RNG
restore, no checkpoint write, no environment and no episode.  The only numbers are
synthetic in-memory arrays, private scratch files and read-only registered JSON evidence.
A spy or no-op here is NOT a real update and proves nothing about the licensed run; the real
gates live in ``r71_trial_gates.py`` and write their own receipts.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
                               / "round68_late_initial_confidence_ablation"), str(HERE)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from test_r71_readiness import SCRATCH, _ephemeral, _scratch_dir            # noqa: E402

import r71_trial_binding as trial                                          # noqa: E402
import r71_trial_gates as gates                                            # noqa: E402
import r71_trial_execute as execute                                        # noqa: E402
import r71_runtime_binding as readiness                                    # noqa: E402
import r68_guards                                                          # noqa: E402
from r71_critic_calibration_seam import (                                  # noqa: E402
    EMBED_DIM, ForwardLedger, RealLicense, BudgetExceeded,
    standardized_to_raw_affine, predict_standardized, predict_raw)


def _synthetic_order() -> list:
    """A 128-episode order over exactly 48 distinct training scenes (shape only)."""
    base = [1300000 + i for i in range(48)]
    return [base[i % 48] for i in range(128)]


# ------------------------------------------------------------------ 1. the fixed plan ----
def test_fixed_plan_is_582_unique_jobs():
    plan = execute.build_plan(_synthetic_order())
    jobs = plan["jobs"]
    assert plan["normal_count"] == 582 and plan["reserve_cap"] == 18 and plan["hard_cap"] == 600
    assert len(jobs) == 582 and len({j["logical_job"] for j in jobs}) == 582
    assert plan["training_episode_count"] == 384
    assert plan["model_eval_count"] == 192 and plan["focal_count"] == 6
    train = [j for j in jobs if j["scope"] == "ppo_training"]
    ev = [j for j in jobs if j["scope"] == "model_eval"]
    focal = [j for j in jobs if j["scope"] == "focal"]
    assert len(train) == 384 and len(ev) == 192 and len(focal) == 6
    assert [j["normal_slot"] for j in jobs] == list(range(1, 583))
    assert plan["eval_scenes"] == list(range(1800000, 1800032))
    assert plan["focal_scene"] == 1700027
    assert plan["historical_probe"] == 0


def test_eval_plan_is_198_cells_of_the_582():
    import r71_trial_eval as trial_eval
    plan = trial_eval.build_eval_plan()
    jobs = plan["jobs"]
    assert len(jobs) == 198 and len({j["logical_job"] for j in jobs}) == 198
    assert [j["planned_normal_slot"] for j in jobs] == list(range(385, 583))
    assert plan["matrix"] == {"model_eval": 192, "focal": 6, "total": 198}
    assert plan["model_eval_arm"] == "NEW_EP128"
    assert plan["primary_comparison"] == "NEW_EP128_minus_R68_TRAINED_ALPHA"
    names = {j["logical_job"] for j in jobs}
    model_names = {n for n in names if "_model_" in n}
    focal_names = {n for n in names if "_focal_" in n}
    # the 192-cell block carries the trained arm ONLY (licence: 192 evaluation launches)
    assert len(model_names) == 192 and all("new_ep128" in n for n in model_names)
    assert any(n.startswith("r71eval_model_new_ep128_s11_greedy_scene1800000") for n in names)
    assert any(n.startswith("r71eval_model_new_ep128_s33_sampled_scene1800031") for n in names)
    # both arms run on the 6 focal cells (scene 1700027, greedy, 3 seeds)
    assert len(focal_names) == 6
    assert any(n.startswith("r71eval_focal_new_init_s11_scene1700027") for n in names)
    assert any(n.startswith("r71eval_focal_new_ep128_s33_scene1700027") for n in names)
    assert not any("_model_new_init_" in n for n in names)


# ------------------------------------------------------------------ 2. teacher batching --
def test_batch_packing_never_exceeds_the_row_cap():
    from r71_critic_calibration_seam import build_batches, BATCH_INPUT_ROW_CAP
    rng = np.random.default_rng(7)
    counts = [int(k) for k in rng.integers(1, 7, size=4000)]
    batches = build_batches(counts, cap=BATCH_INPUT_ROW_CAP)
    assert BATCH_INPUT_ROW_CAP == 1024
    for batch in batches:
        assert sum(batch) <= BATCH_INPUT_ROW_CAP
    assert sum(sum(b) for b in batches) == sum(counts)
    # _pack aligns the matrix rows with the per-joint-row target counts
    samples = [{"features": np.zeros((k, 19), dtype=np.float32), "targets": [f"t{i}"] * k,
                "return_task": 1.0, "return_cost": 0.5, "episode_index": 0}
               for i, k in enumerate(counts[:5])]
    import r71_offline_fit as fit
    matrix, packed_counts, returns, meta = fit._pack(samples, [])
    assert matrix.shape == (sum(counts[:5]), 19)
    assert packed_counts == counts[:5]
    assert returns.shape == (5, 2)
    assert meta["target_rows"] == sum(counts[:5])


# ------------------------------------------------------------------ 3. ridge round-trip --
def test_ridge_roundtrip_on_synthetic_embeddings_with_a_spy_trunk():
    """The installed-head prediction equals the standardised ridge prediction (spy trunk)."""
    rng = np.random.default_rng(11)
    rows = 40
    raw = rng.normal(size=(rows, 19)).astype(np.float32)      # the REAL (N, 19) trunk input
    X = rng.normal(size=(rows, EMBED_DIM))                    # the embeddings the spy returns
    counts = [4] * (rows // 4)                                # 10 joint rows of 4 target rows
    Y = rng.normal(size=(len(counts), 2))                     # aligned with the POOLED rows
    calls = []

    def spy_trunk(matrix):
        calls.append(np.asarray(matrix).shape)
        return X[:np.asarray(matrix).shape[0]]

    ledger = ForwardLedger(cap=10000)                     # in-memory, no persistence
    provider = readiness.CalibrationProvider(_SpyController(), ledger, trunk_callable=spy_trunk)
    hidden = provider.trunk_forward(raw, n_joint_rows=len(counts))
    assert calls, "the spy trunk must be the delegate"
    pooled = provider.mean_over_targets(hidden, counts)
    fit = provider.fit_dual_output_ridge(pooled, Y)
    heads = provider.calibrated_heads(fit)
    # rebuild the full dual-output (B, b) from BOTH installed heads and compare like with like
    B = np.concatenate([heads.heads["value_task_head"]["weight"].T,
                        heads.heads["value_cost_head"]["weight"].T], axis=1)
    b = np.array([heads.heads["value_task_head"]["bias"][0],
                  heads.heads["value_cost_head"]["bias"][0]])
    gap = float(np.max(np.abs(predict_standardized(pooled, fit["mu"], fit["sigma"],
                                                   fit["A"], fit["c"])
                              - predict_raw(pooled, B, b))))
    assert gap < 1e-9, gap
    assert ledger.rows == rows, "the ledger must charge the real input rows"
    assert fit["ridge"] == 1e-3 and fit["intercept_penalized"] is False


class _SpyController:
    """Structural stand-in: only the two critic heads exist, both writable by the installer."""

    class _Head:
        def __init__(self):
            self.weight = np.zeros((1, EMBED_DIM))
            self.bias = np.zeros(1)

        def named_parameters(self):
            return iter([("weight", self.weight), ("bias", self.bias)])

    def __init__(self):
        self.inner = type("inner", (), {})()
        self.inner.value_head = self._Head()
        self.cost_head = self._Head()


# ------------------------------------------------------------------ 4. licence -----------
def test_archived_licence_opens_the_gate_and_lookalikes_do_not():
    receipt = trial.licence_receipt()
    assert receipt["status"] == "passed", receipt
    assert receipt["archived_licence_opens_gate"] is True
    assert receipt["gate_requires_environment_build"] is True
    gate = trial.trial_gate()
    gate.require("environment_build")                     # must not raise
    base = trial.load_trial_licence()
    for field, value in (("test_only", True), ("approval_id", "SOMETHING-ELSE"),
                         ("task_id", "ROUND71-REAL-RUNTIME-BINDING-ZERO-EXEC"),
                         ("quota_identity", "round71-critic-calibration/plan-1/1"),
                         ("source_freeze_sha256", "0" * 64),
                         ("input_identity_sha256", "0" * 64)):
        tampered = dict(base)
        tampered[field] = value
        assert trial.verify_r71_trial_licence(
            RealLicense.from_mapping(tampered)) is False, field
    missing = dict(base)
    missing.pop("input_identity_sha256")
    try:
        # from_mapping itself fails closed on a missing field; either way it is refused
        refused = trial.verify_r71_trial_licence(RealLicense.from_mapping(missing)) is False
    except Exception:
        refused = True
    assert refused
    assert trial.verify_r71_trial_licence("a-raw-token") is False
    assert trial.make_trial_license_gate().present() is False


# ------------------------------------------------------------------ 5. restore refusals --
def test_descriptive_restore_source_refusals():
    run_identity = trial.trial_run_identity()
    cases = [
        ({"identity": {"seed": 11, "structure": readiness.SEAM_R68_STRUCTURE,
                       "training_version": readiness.R68_TRAINING_VERSION_EXPECTED,
                       "module_set": list(readiness.MODULES), "source_kind": "r68_ep0"},
          "file_sha256": "0" * 64, "parameter_sha256": "0" * 64, "episodes_done": 0,
          "updates": 0, "load_mode": "read_only", "optimizer_restored": False,
          "lambda_restored": False, "rng_restored": False, "from_trained_ep128": True,
          "used_as_init_source": True}, "r68_ep128_readonly_control"),
        ({"identity": {"seed": 11, "structure": readiness.R71_STRUCTURE,
                       "training_version": readiness.R68_TRAINING_VERSION_EXPECTED,
                       "module_set": list(readiness.MODULES), "source_kind": "this_run"},
          "run_identity": run_identity.as_dict(),
          "carries_trained_optimizer_state": True, "carries_trained_lambda": True,
          "carries_advanced_training_rng": True, "from_trained_ep128": False},
         "r71_new_full_restore"),
        ({"identity": {"seed": 11, "structure": readiness.SEAM_R68_STRUCTURE,
                       "training_version": readiness.R68_TRAINING_VERSION_EXPECTED,
                       "module_set": list(readiness.MODULES), "source_kind": "r68_ep0"},
          "file_sha256": "0" * 64, "parameter_sha256": "0" * 64, "episodes_done": 0,
          "updates": 0, "load_mode": "model_only", "optimizer_restored": False,
          "lambda_restored": False, "rng_restored": False, "from_trained_ep128": True},
         "r68_ep0_model_only_init"),
    ]
    for payload, role in cases:
        try:
            readiness.validate_calibration_source(payload, role=role,
                                                  run_identity=run_identity)
        except Exception:
            continue
        raise AssertionError(f"role {role} was not refused")


def test_trial_run_identity_is_stable_and_bound():
    a = trial.trial_run_identity()
    b = trial.trial_run_identity()
    assert a.as_dict() == b.as_dict()
    assert a.task_id == trial.R71_TRIAL_TASK_ID and a.branch == trial.R71_TRIAL_BRANCH
    assert a.quota_identity == readiness.R71_QUOTA_IDENTITY
    assert a.training_version == readiness.R71_TRAINING_VERSION
    assert a.matches_block(a.as_dict()) is True
    assert a.matches_block(dict(a.as_dict(), run_id="someone-else")) is False


# ------------------------------------------------------------------ 6. guards ------------
def test_forward_ledger_charges_before_the_forward_and_refuses_over_cap():
    calls = []
    ledger = ForwardLedger(cap=10)

    def spy(matrix):
        calls.append(ledger.rows)                      # rows already charged
        return np.zeros((np.asarray(matrix).shape[0], EMBED_DIM))

    provider = readiness.CalibrationProvider(_SpyController(), ledger, trunk_callable=spy)
    provider.trunk_forward(np.zeros((4, 19)), n_joint_rows=1)
    assert calls == [4], "the charge must happen before the forward"
    provider.trunk_forward(np.zeros((5, 19)), n_joint_rows=1)
    assert calls[-1] == 9
    try:
        provider.trunk_forward(np.zeros((2, 19)), n_joint_rows=1)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("an over-cap batch must be refused before any forward")
    assert len(calls) == 2, "the refused batch must never reach the trunk"
    assert ledger.refusals == 1


def test_synthetic_step_guard_counts_physical_calls():
    d = _scratch_dir("r71_trial_synth")
    for stale in d.glob("*.json"):
        stale.unlink()
    with _ephemeral(d):
        path = d / "steps.json"
        guard = r68_guards.SyntheticStepGuard(path, cap=3)
        for _ in range(3):
            guard.charge(1, "synthetic step")
        assert guard.used() == 3 and guard.remaining() == 0
        try:
            guard.charge(1, "one step too many")
        except Exception:
            pass
        else:
            raise AssertionError("the synthetic step cap must refuse the next step")
        again = r68_guards.SyntheticStepGuard(path, cap=3)      # a restart must not zero it
        assert again.used() == 3


def test_critic_input_detached_hands_the_heads_a_detached_tensor():
    """Spy-only proof of the isolation mechanism: no autograd, no real model."""
    import torch
    from types import SimpleNamespace
    from round53_trial_adapter import critic_input_detached
    seen = []

    class Head(torch.nn.Module):
        def forward(self, x):
            seen.append(bool(x.requires_grad))
            return x.sum()

    trunk_out = torch.randn(4, 3, requires_grad=True)
    controller = SimpleNamespace(inner=SimpleNamespace(value_head=Head()), cost_head=Head())
    with critic_input_detached(controller):
        controller.inner.value_head(trunk_out)
        controller.cost_head(trunk_out)
    assert seen == [False, False], "inside the context the critic input must be detached"
    controller.inner.value_head(trunk_out)
    assert seen[-1] is True, "outside the context the critic input keeps its graph"
    assert len(controller.inner.value_head._forward_pre_hooks) == 0


# ------------------------------------------------------------------ 7. statistics math ----
def test_paired_delta_math_on_synthetic_rows():
    import r68_eval as r68eval
    scenes = list(range(1800000, 1800032))
    idx, idx_sha = r68eval._bootstrap_index()
    assert idx.shape == (10000, 32)
    a = {s: {"on_time": 6, "spent_bytes": 1000, "on_time_ids": ["t0", "t1"]} for s in scenes}
    b = {s: {"on_time": 5, "spent_bytes": 1200, "on_time_ids": ["t0"]} for s in scenes}
    delta = r68eval._paired_delta(lambda s: a[s], lambda s: b[s], scenes, idx, "A", "B")
    assert delta["task_delta_A_minus_B_mean"] == 1.0
    assert delta["cost_delta_A_minus_B_B_per_scene"] == -200.0
    assert delta["task_ci95_A_minus_B"] == [1.0, 1.0]
    assert delta["scenes_A_better_task_than_B"] == 32
    assert delta["scenes_A_cheaper_than_B"] == 32
    assert delta["targets_gained_A_vs_B"] == [
        {"scene": s, "target_id": "t1"} for s in scenes]
    assert delta["targets_lost_A_vs_B"] == []


def test_first_report_split_on_a_synthetic_tape(tmp=None):
    import r71_trial_eval as trial_eval
    d = _scratch_dir("r71_trial_first_report")
    for stale in d.glob("*.json*"):
        stale.unlink()
    with _ephemeral(d):
        tape = d / "tape.jsonl"
        lines = [
            {"step": 1, "targets": ["a", "b"], "actionable": ["a", "b"],
             "actions": {"a": 1, "b": 0},
             "features": [[0.0] * 18 + [0.147], [0.0] * 18 + [0.148]]},
            {"step": 5, "targets": ["a", "b"], "actionable": ["b"], "actions": {"b": 1},
             "features": [[0.0] * 19, [0.0] * 18 + [0.148]]},
        ]
        tape.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        row_path = d / "row.json"
        row_path.write_text(json.dumps({"artefacts": {"tape": str(tape)}}), encoding="utf-8")
        events = trial_eval.first_report_sets(row_path)
        assert [(e["target"], e["action"], e["region"]) for e in events] == [
            ("a", 1, "early"), ("b", 0, "late")]
        assert events[0]["ready_step"] == 147 and events[1]["ready_step"] == 148
        summary = trial_eval._first_report_summary({1800000: {"row_path": str(row_path)}})
        assert summary["sends_early"] == 1 and summary["skips_early"] == 0
        assert summary["skips_late"] == 1 and summary["sends_late"] == 0