#!/usr/bin/env python
"""Narrow PURE-ARITHMETIC regression test for ROUND70 (no network, no data files, no forward).

Validates only the in-memory statistics used by the diagnostic: the dual-output ridge fit
(intercept unpenalised, constant-column scale 1, centred/scaled design), the intercept-only
baseline, the bias/MAE/RMSE + composite-residual block, the persistent forward-budget counter,
and the scene-grouped PCG64 fold builder.  Everything here is closed-form arithmetic checked
against hand/independent computations; it consumes no neural forward and touches no R68 artefact.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r70_frozen_critic_calibration_check import (  # noqa: E402
    fit_ridge, predict_ridge, fit_intercept, predict_intercept,
    _metrics, _block, ForwardBudget, build_folds, RIDGE)


def test_ridge_normal_equations_and_independent_solution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    Y = rng.normal(size=(40, 2))
    model = fit_ridge(X, Y, RIDGE)
    n, d = X.shape
    Xn = (X - model["mu"]) / model["sigma"]
    Xa = np.concatenate([Xn, np.ones((n, 1))], axis=1)
    P = np.eye(d + 1) * RIDGE
    P[d, d] = 0.0
    # normal equations must hold: (Xa^T Xa + P) W = Xa^T Y
    lhs = Xa.T @ Xa + P
    resid = lhs @ model["W"] - Xa.T @ Y
    assert np.allclose(resid, 0.0, atol=1e-8), resid
    # independent solve (append explicit penalty rows; intercept row unpenalised)
    A = np.concatenate([Xa, np.sqrt(RIDGE) * np.eye(d + 1)[:d]], axis=0)
    B = np.concatenate([Y, np.zeros((d, 2))], axis=0)
    W2, *_ = np.linalg.lstsq(A, B, rcond=None)
    assert np.allclose(model["W"], W2, atol=1e-6)


def test_huge_ridge_collapses_to_intercept_mean():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 5))
    Y = rng.normal(size=(30, 2))
    model = fit_ridge(X, Y, 1e12)
    pred = predict_ridge(model, X)
    assert np.allclose(pred, Y.mean(axis=0), atol=1e-4)   # feature weights vanish -> mean


def test_intercept_not_penalised():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 4))
    Y = rng.normal(size=(25, 2))
    m1 = fit_ridge(X, Y, RIDGE)
    m2 = fit_ridge(X, Y + 3.0, RIDGE)
    assert np.allclose(m1["W"][:-1], m2["W"][:-1], atol=1e-9)         # feature weights unchanged
    assert np.allclose(m2["W"][-1] - m1["W"][-1], 3.0, atol=1e-6)     # intercept shifts by const


def test_constant_column_scale_is_one():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0]])
    Y = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    model = fit_ridge(X, Y, RIDGE)
    assert model["sigma"][0] == 1.0            # constant column scale forced to 1
    assert np.all(np.isfinite(model["W"]))
    pred = predict_ridge(model, X)
    assert np.all(np.isfinite(pred))


def test_intercept_baseline_is_train_mean():
    Y = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
    model = fit_intercept(Y)
    assert np.allclose(model, [4.0, 3.0])
    assert np.allclose(predict_intercept(model, 5), np.tile([4.0, 3.0], (5, 1)))


def test_metrics_hand_computed():
    m = _metrics([1.0, -1.0, 2.0, -2.0])
    assert m["n"] == 4 and abs(m["bias"]) < 1e-12
    assert abs(m["mae"] - 1.5) < 1e-12
    assert abs(m["rmse"] - np.sqrt(2.5)) < 1e-12
    assert _metrics([])["bias"] is None


def test_composite_block():
    blk = _block([3.0], [1.0], [1.0], [0.5], 2.0)   # lam*(Rtask-pred_task) - (Rcost-pred_cost)
    assert abs(blk["task"]["bias"] - 2.0) < 1e-12
    assert abs(blk["cost"]["bias"] - 0.5) < 1e-12
    assert abs(blk["composite"]["bias"] - (2.0 * 2.0 - 0.5)) < 1e-12


def test_forward_budget_rejects_before_exceeding():
    b = ForwardBudget(10)
    b.charge(6, 11)
    b.charge(4, 7)
    assert b.rows == 10 and b.target_evals == 18
    raised = False
    try:
        b.charge(1, 1)
    except RuntimeError:
        raised = True
    assert raised and b.rows == 10


def test_fold_grouping_is_scene_shared_and_deterministic():
    rows = [{"scene": 1300000 + (i % 7)} for i in range(70)]   # 7 scenes, many repeats
    s2f1, folds1, scenes = build_folds(rows)
    s2f2, folds2, _ = build_folds(rows)                          # deterministic: same PCG64 seed
    assert s2f1 == s2f2
    assert len(folds1) == 5
    # every scene assigned to exactly one fold; union covers all scenes
    assigned = set(s2f1)
    assert assigned == set(scenes)
    for s in scenes:
        assert 0 <= s2f1[s] < 5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL_PASS", len(fns))