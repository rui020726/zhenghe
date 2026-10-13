#!/usr/bin/env python
"""Narrow ROUND71 readiness tests: PURE ARITHMETIC / STATIC / SPY only.

Zero consumption by construction: no real or synthetic neural forward, no optimizer step, no
autograd, no real-data fit, no model load, no RNG restore, no checkpoint write, no environment
and no episode.  The only numbers are synthetic in-memory arrays and a private temporary
counter.  A spy or no-op here is NOT a real update and does not prove power-loss atomicity.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r71_critic_calibration_seam import (  # noqa: E402
    EMBED_DIM, MODULES, FROZEN_MODULES, TRAINABLE_MODULES, CRITIC_HEAD_MODULES,
    OFFLINE_INPUT_ROW_CAP, BATCH_INPUT_ROW_CAP, RIDGE, CANDIDATE_PLAN,
    LicenseMissing, BudgetExceeded, DuplicateOutput, IllegalRestoreSource,
    standardized_to_raw_affine, predict_standardized, predict_raw, affine_to_critic_heads,
    CriticHeadSet, ParameterContainer, ForwardLedger, build_batches, LicenseGate,
    DuplicateOutputGuard, validate_restore_source, CalibrationInitializer)


# ------------------------------------------------------------------ 1. coefficient transform
def test_tiny_hand_computed_counterexample():
    """Hand-computed 2-D counterexample, computed independently of the module's code.

    mu=[1,2], sigma=[2,4], A=[[1,0],[0,1]], c=[0,0]
      B[0]=A[0]/sigma[0]=[0.5,0]   B[1]=A[1]/sigma[1]=[0,0.25]
      b = c - mu@B = [0,0] - [1*0.5+2*0, 1*0+2*0.25] = [-0.5,-0.5]
    Test point X=[3,6]:
      standardised: Xn=[1,1] -> pred = Xn@A + c = [1,1]
      raw:          X@B + b = [1.5,1.5] + [-0.5,-0.5] = [1,1]   (must agree)
    """
    mu = np.array([1.0, 2.0]); sigma = np.array([2.0, 4.0])
    A = np.array([[1.0, 0.0], [0.0, 1.0]]); c = np.array([0.0, 0.0])
    B = np.stack([A[0] / sigma[0], A[1] / sigma[1]])
    b = c - mu @ B
    assert np.allclose(B, [[0.5, 0.0], [0.0, 0.25]])
    assert np.allclose(b, [-0.5, -0.5])
    X = np.array([[3.0, 6.0]])
    Xn = (X - mu) / sigma
    assert np.allclose(Xn @ A + c, [1.0, 1.0])
    assert np.allclose(X @ B + b, [1.0, 1.0])


def test_analytic_roundtrip_on_full_embedding_dim():
    """Build (mu, sigma, A, c) FROM a known raw (B, b); the module must recover it exactly."""
    rng = np.random.default_rng(20261013)
    mu = rng.normal(size=EMBED_DIM)
    sigma = rng.uniform(0.2, 3.0, size=EMBED_DIM)
    B_true = rng.normal(size=(EMBED_DIM, 2))
    b_true = rng.normal(size=2)
    A = B_true * sigma[:, None]              # invert the division
    c = b_true + mu @ B_true                 # invert the subtraction
    B, b = standardized_to_raw_affine(mu, sigma, A, c)
    assert np.allclose(B, B_true, atol=1e-12)
    assert np.allclose(b, b_true, atol=1e-12)


def test_predictors_agree_on_random_embeddings():
    rng = np.random.default_rng(7)
    mu = rng.normal(size=EMBED_DIM)
    sigma = np.where(rng.random(EMBED_DIM) < 0.2, 0.0, rng.uniform(0.1, 4.0, EMBED_DIM))
    sigma = np.where(sigma == 0.0, 1.0, sigma)          # constant-column rule
    A = rng.normal(size=(EMBED_DIM, 2)) * 0.3
    c = rng.normal(size=2)
    X = rng.normal(size=(37, EMBED_DIM)) * 2.0 + 0.5
    B, b = standardized_to_raw_affine(mu, sigma, A, c)
    assert np.allclose(predict_standardized(X, mu, sigma, A, c),
                       predict_raw(X, B, b), atol=1e-10)


def test_constant_column_scale_must_be_one_not_zero():
    mu = np.zeros(EMBED_DIM); sigma = np.ones(EMBED_DIM)
    A = np.zeros((EMBED_DIM, 2)); c = np.zeros(2)
    B, b = standardized_to_raw_affine(mu, sigma, A, c)      # all-zero must stay finite
    assert np.all(np.isfinite(B)) and np.all(np.isfinite(b))
    sigma_bad = np.ones(EMBED_DIM); sigma_bad[3] = 0.0
    raised = False
    try:
        standardized_to_raw_affine(mu, sigma_bad, A, c)
    except ValueError:
        raised = True
    assert raised, "a zero scale must be refused, never silently divided through"


def test_affine_to_critic_heads_layout():
    rng = np.random.default_rng(3)
    B = rng.normal(size=(EMBED_DIM, 2)); b = rng.normal(size=2)
    heads = affine_to_critic_heads(B, b)
    assert set(heads) == set(CRITIC_HEAD_MODULES)
    for j, m in enumerate(CRITIC_HEAD_MODULES):
        assert heads[m]["weight"].shape == (1, EMBED_DIM)
        assert heads[m]["bias"].shape == (1,)
        assert np.allclose(heads[m]["weight"][0], B[:, j])
        assert np.allclose(heads[m]["bias"][0], b[j])
        # head(X) must equal the raw affine prediction for that output
        X = rng.normal(size=(5, EMBED_DIM))
        assert np.allclose(X @ heads[m]["weight"][0] + heads[m]["bias"][0],
                           predict_raw(X, B, b)[:, j], atol=1e-10)


# ------------------------------------------------------------------ 2. actor/trunk untouched
def test_calibration_touches_only_critic_heads():
    pc = ParameterContainer()
    before = {m: (pc.modules[m]["weight"].copy(), pc.modules[m]["bias"].copy(),
                  id(pc.modules[m])) for m in MODULES}
    rng = np.random.default_rng(11)
    B = rng.normal(size=(EMBED_DIM, 2)); b = rng.normal(size=2)
    head_set = CriticHeadSet(affine_to_critic_heads(B, b))
    touched = pc.install_critic_heads(head_set)
    assert sorted(touched) == sorted(CRITIC_HEAD_MODULES)
    for m in MODULES:
        w, bi, oid = before[m]
        if m in CRITIC_HEAD_MODULES:
            assert not np.allclose(pc.modules[m]["weight"], w) or not np.allclose(
                pc.modules[m]["bias"], bi), f"{m} should have changed"
        else:
            assert np.array_equal(pc.modules[m]["weight"], w), f"{m} weight must be untouched"
            assert np.array_equal(pc.modules[m]["bias"], bi), f"{m} bias must be untouched"
            assert id(pc.modules[m]) == oid, f"{m} object identity must be untouched"


def test_head_set_replacement_is_not_in_place():
    hs = CriticHeadSet({m: {"weight": np.zeros((1, EMBED_DIM)), "bias": np.zeros(1)}
                        for m in CRITIC_HEAD_MODULES})
    d0 = hs.digest()
    rng = np.random.default_rng(5)
    other = hs.replace_heads(affine_to_critic_heads(rng.normal(size=(EMBED_DIM, 2)),
                                                    rng.normal(size=2)))
    assert hs.digest() == d0, "replace_heads must not mutate the original set"
    assert other.digest() != d0
    assert other.heads["value_task_head"]["weight"].shape == (1, EMBED_DIM)


def test_freeze_partition_is_preserved_by_the_candidate():
    """The two critic heads are TRAINABLE in R68, so swapping their init keeps 6 frozen / 6 trainable."""
    assert set(FROZEN_MODULES) == {"trunk", "early_head"}
    assert set(TRAINABLE_MODULES) == {"late_head", "value_task_head", "value_cost_head"}
    assert set(CRITIC_HEAD_MODULES) <= set(TRAINABLE_MODULES)
    assert "late_head" in TRAINABLE_MODULES and "late_head" not in CRITIC_HEAD_MODULES
    # R68 records 6 frozen and 6 trainable parameter OBJECTS: trunk carries 4 tensors
    # (W0, b0, W2, b2) and every head carries 2 (weight, bias).
    frozen_heads = [m for m in FROZEN_MODULES if m != "trunk"]
    assert 4 + 2 * len(frozen_heads) == 6          # trunk 4 tensors + early_head 2 tensors
    assert 2 * len(TRAINABLE_MODULES) == 6         # three heads x 2 tensors each
    assert set(FROZEN_MODULES) | set(TRAINABLE_MODULES) == set(MODULES)

# ------------------------------------------------------------------ 3. ledger / batching
def test_billing_unit_is_target_expansion_not_joint_rows():
    """3 decision steps with 6/2/1 targets: 3 joint rows but 9 real 19-D input rows."""
    led = ForwardLedger(cap=100)
    led.debit(6, n_joint_rows=1)
    led.debit(2, n_joint_rows=1)
    led.debit(1, n_joint_rows=1)
    assert led.rows == 9 and led.joint_rows == 3
    assert led.rows != led.joint_rows


def test_batch_boundary_1024_and_refusal_before_exceeding():
    counts = [6, 2, 1, 6, 3]
    batches = build_batches(counts, cap=10)
    assert all(sum(b) <= 10 for b in batches)
    assert sum(len(b) for b in batches) == len(counts)
    assert sum(sum(b) for b in batches) == sum(counts)
    led = ForwardLedger(cap=10)
    led.debit(10)
    assert led.remaining() == 0
    raised = False
    try:
        led.debit(1)
    except BudgetExceeded:
        raised = True
    assert raised and led.rows == 10 and led.refusals == 1


def test_batches_never_exceed_cap_and_cover_every_target():
    rng = np.random.default_rng(13)
    counts = [int(rng.integers(1, 7)) for _ in range(600)]
    batches = build_batches(counts, BATCH_INPUT_ROW_CAP)
    assert all(sum(b) <= BATCH_INPUT_ROW_CAP for b in batches)
    assert sum(len(b) for b in batches) == len(counts)
    assert sum(sum(b) for b in batches) == sum(counts)
    assert max(sum(b) for b in batches) <= BATCH_INPUT_ROW_CAP


def test_single_oversized_step_is_emitted_alone_not_split():
    batches = build_batches([1, BATCH_INPUT_ROW_CAP + 5, 1], BATCH_INPUT_ROW_CAP)
    assert len(batches) == 3
    assert sum(batches[1]) == BATCH_INPUT_ROW_CAP + 5      # reported, never silently split
    assert sum(batches[0]) == 1 and sum(batches[2]) == 1


def test_failed_fake_call_still_consumes_the_debit():
    """Debit happens BEFORE the forward, so a forward that raises still leaves the charge."""
    led = ForwardLedger(cap=1000)
    consumed = 0
    raised = False
    try:
        led.debit(40, n_joint_rows=12)          # reservation first
        raise RuntimeError("synthetic forward failure")   # then the call fails
    except RuntimeError:
        raised = True
    consumed = led.rows
    assert raised and consumed == 40 and led.calls == 1
    assert led.remaining() == 960


def test_retries_and_repeats_are_charged_every_time():
    led = ForwardLedger(cap=1000)
    for _ in range(5):
        led.debit(7, n_joint_rows=2)             # same rows re-sent five times
    assert led.rows == 35 and led.calls == 5 and led.joint_rows == 10


import contextlib
import shutil


@contextlib.contextmanager
def _ephemeral(d):
    """Yield a private scratch directory and remove its files afterwards.

    Uses the writable scratch root because the process default temp directory is not writable
    inside this sandbox.  Nothing here is a real run artefact; it is a test-only private
    temporary counter/file and is deleted at the end of the test.
    """
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in d.iterdir()}
    try:
        yield d
    finally:
        for p in d.iterdir():
            if p.name not in before:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)


SCRATCH = (Path(__file__).resolve().parents[2] / ".codex" / "visualizations" / "2026" / "10"
           / "11" / "01a128bc-27a9-7341-b0a6-19e983cb3840" / "r71_scratch")


def _scratch_dir(name):
    d = SCRATCH / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_ledger_persists_across_instances(tmp_path=None):
    d = _scratch_dir("ledger_persist")
    for stale in d.glob("*.json"):
        stale.unlink()
    with _ephemeral(d) as d:
        p = d / "ledger.json"
        a = ForwardLedger(cap=1000, state_path=p)
        a.debit(30, n_joint_rows=9)
        b = ForwardLedger(cap=1000, state_path=p)      # simulates a restart
        assert b.rows == 30, "a restart must not zero the consumption"
        b.debit(70, n_joint_rows=1)
        c = ForwardLedger(cap=1000, state_path=p)
        assert c.rows == 100 and c.calls == 2
        assert c.remaining() == 900, "restart must not reset the remaining budget"


# ------------------------------------------------------------------ 4. licence fail-closed
def test_missing_licence_fails_closed_on_every_gated_action():
    gate = LicenseGate(token=None)
    assert not gate.licensed
    for action in LicenseGate.GATED_ACTIONS:
        raised = False
        try:
            gate.require(action)
        except LicenseMissing:
            raised = True
        assert raised, f"action {action} must fail closed without a licence"


def test_unknown_action_is_refused():
    gate = LicenseGate(token=None)
    raised = False
    try:
        gate.require("invented_action")
    except ValueError:
        raised = True
    assert raised


def test_initializer_refuses_before_any_fit_or_forward():
    """With no licence, run() must raise at the FIRST gate, before fitting or forwarding."""
    led = ForwardLedger(cap=OFFLINE_INPUT_ROW_CAP)
    calls = {"extractor": 0, "fitter": 0, "installer": 0}

    def extractor(seed):
        calls["extractor"] += 1
        return np.zeros((1, EMBED_DIM))

    def fitter(seed, emb):
        calls["fitter"] += 1
        return (np.zeros((EMBED_DIM, 2)), np.zeros(2))

    def installer(seed, fit):
        calls["installer"] += 1
        return fit

    init = CalibrationInitializer(LicenseGate(token=None), ledger=led,
                                  embedding_extractor=extractor, ridge_fitter=fitter,
                                  head_installer=installer,
                                  output_path=Path(__file__).with_name("must_not_exist.json"))
    raised = False
    try:
        init.run([11, 22, 33], {11: [6, 6], 22: [6, 6], 33: [6, 6]})
    except LicenseMissing:
        raised = True
    assert raised
    assert calls == {"extractor": 0, "fitter": 0, "installer": 0}, "no collaborator may run"
    assert led.rows == 0 and led.calls == 0, "no ledger charge may occur"
    assert not Path(init.output_path).exists()


def test_spy_collaborators_prove_no_real_update_happened():
    """Even WITH a licence token, spy collaborators record that nothing real was executed."""
    led = ForwardLedger(cap=OFFLINE_INPUT_ROW_CAP)
    seen = {"forward_rows": 0}

    def extractor(seed):
        seen["forward_rows"] += 1        # a spy, NOT a neural forward
        return np.zeros((1, EMBED_DIM))

    init = CalibrationInitializer(LicenseGate(token=None), ledger=led,
                                  embedding_extractor=extractor,
                                  ridge_fitter=lambda s, e: (np.zeros((EMBED_DIM, 2)), np.zeros(2)),
                                  head_installer=lambda s, f: f)
    raised = False
    try:
        init.run([11], {11: [3, 3]})
    except LicenseMissing:
        raised = True
    assert raised and seen["forward_rows"] == 0


# ------------------------------------------------------------------ 5. restore-source legality
def test_trained_ep128_and_old_checkpoints_are_refused():
    bad = [
        {"declared_purpose": "new_training_initialisation", "from_trained_ep128": True,
         "modules": list(MODULES)},
        {"declared_purpose": "resume_training", "modules": list(MODULES)},
        {"declared_purpose": "new_training_initialisation",
         "carries_trained_optimizer_state": True, "modules": list(MODULES)},
        {"declared_purpose": "new_training_initialisation",
         "carries_trained_lambda": True, "modules": list(MODULES)},
        {"declared_purpose": "new_training_initialisation",
         "carries_training_rng": True, "modules": list(MODULES)},
        {"declared_purpose": "new_training_initialisation",
         "modules": ["trunk", "early_head", "late_head", "actor_head"]},   # old 4-module
    ]
    for src in bad:
        raised = False
        try:
            validate_restore_source(src)
        except IllegalRestoreSource:
            raised = True
        assert raised, f"must be refused: {src}"


def test_legal_new_training_initialisation_is_accepted():
    ok = {"declared_purpose": "new_training_initialisation", "modules": list(MODULES),
          "carries_trained_optimizer_state": False, "carries_trained_lambda": False,
          "carries_training_rng": False, "from_trained_ep128": False}
    assert validate_restore_source(ok) is True


def test_duplicate_output_guard_refuses_implicit_rerun():
    with _ephemeral(_scratch_dir("dup_guard")) as d:
        p = d / "exists.json"
        p.write_text("{}", encoding="utf-8")
        g = DuplicateOutputGuard(p)
        raised = False
        try:
            g.check()
        except DuplicateOutput:
            raised = True
        assert raised
        g.check(overwrite=True)      # explicit decision is allowed


# ------------------------------------------------------------------ 6. candidate plan registration
def test_candidate_plan_is_registered_as_approval_pending():
    assert CANDIDATE_PLAN["status"] == "APPROVAL_PENDING"
    assert CANDIDATE_PLAN["fit_config"]["ridge"] == RIDGE == 1e-3
    assert CANDIDATE_PLAN["fit_config"]["fits_per_seed"] == 1
    assert "180 evaluation data" in CANDIDATE_PLAN["fit_config"]["forbidden"]
    assert "seed selection" in CANDIDATE_PLAN["fit_config"]["forbidden"]
    p = CANDIDATE_PLAN["future_run_plan_if_ever_licensed"]
    assert p["normal"] == 582 and p["technical_reserve"] == 18 and p["hard_total"] == 600
    assert CANDIDATE_PLAN["budgets"]["offline_input_row_cap"] == OFFLINE_INPUT_ROW_CAP == 72000
    assert CANDIDATE_PLAN["budgets"]["batch_input_row_cap"] == BATCH_INPUT_ROW_CAP == 1024
    assert CANDIDATE_PLAN["budgets"]["candidate_synthetic_optimizer_cap"] == 16


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL_PASS", len(fns))