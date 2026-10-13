#!/usr/bin/env python
"""ROUND71 private critic-calibration initialisation seam (ZERO-CONSUMPTION preparation).

This module is PRIVATE PREPARATION ONLY.  It contains no experiment authorisation.  Nothing
here opens a real model, fits real data, runs a neural forward, restores RNG, writes a
checkpoint or constructs an environment.  Every entry point that would touch real state is
gated behind an explicit licence token that is ABSENT this round, so each one fails closed.

The candidate intervention (APPROVAL_PENDING, NOT licensed this round) is: initialise each
R68 seed model's dual critic affine heads from a ridge calibration fitted on that seed's own
existing training teacher data over the frozen trunk's true mean-over-targets representation.
Everything else is inherited unchanged: the trunk, the early/late actor heads (late head
initial value scaled x0.10), the initial policy, and a legitimate NEW training state starting
from the qualified R68 ep0 (optimizer / lambda / RNG), never from a trained ep128.

Because the real critic head is an affine map on the 64-D frozen-trunk embedding
(Linear(64,1) per head, verified read-only in ROUND70), a dual-output ridge fitted on the
STANDARDISED embedding is the same function class.  Converting its coefficients back to the
raw 64-D embedding affine head is exact closed-form algebra, implemented below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------
# frozen structural facts, taken from on-disk R68/R70 JSON evidence (no .pt opened here)
# ----------------------------------------------------------------------------------------
MODULES = ("trunk", "early_head", "late_head", "value_task_head", "value_cost_head")
FROZEN_MODULES = ("trunk", "early_head")            # requires_grad=False in R68
TRAINABLE_MODULES = ("late_head", "value_task_head", "value_cost_head")
CRITIC_HEAD_MODULES = ("value_task_head", "value_cost_head")
EMBED_DIM = 64
TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2"
    "+r53-critic-representation-stop-gradient/1+r56-soft-early-report-prior-beta0p10/1"
    "+r58-prior-strength-beta1p00/1+r60-partitioned-early-late-actor-head/1"
    "+r66-frozen-early-path-late-adaptation/1+r68-late-initial-confidence-alpha0.10/1")
R68_STRUCTURE = "partitioned-early-late-actor-head/1+frozen-early-path-late-adaptation/1"

# candidate diagnostic configuration, LOCKED (registered, not licensed)
RIDGE = 1e-3                 # fixed; no sweep
CENTER_SCALE_SCOPE = "all training teacher rows of that seed"   # no held-out fold in the candidate
CONSTANT_COLUMN_SCALE = 1.0  # constant column scale := 1
INTERCEPT_PENALIZED = False  # intercept unpenalised
FITS_PER_SEED = 1            # one dual-output fit per seed, no CV, no seed selection

# candidate resource budgets (offline phase only; any licensed run is counted separately)
OFFLINE_INPUT_ROW_CAP = 72000   # actual 19-D trunk input-vector rows
BATCH_INPUT_ROW_CAP = 1024      # max real input rows per forward batch
CANDIDATE_SYNTHETIC_OPTIMIZER_CAP = 16   # registered only; 0 consumed this round


class LicenseMissing(RuntimeError):
    """Raised when a gated action is attempted without a real experiment licence."""


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a forward that would push the ledger past its cap."""


class DuplicateOutput(RuntimeError):
    """Raised when an existing artefact would be implicitly overwritten by a re-run."""


class IllegalRestoreSource(RuntimeError):
    """Raised when an old checkpoint is offered as a NEW training restore source."""


# ========================================================================================
# 1. exact coefficient transform: standardised ridge  ->  raw 64-D embedding affine head
# ========================================================================================
def standardized_to_raw_affine(mu, sigma, A, c):
    """Convert a dual-output affine fitted on the STANDARDISED embedding to the RAW embedding.

    The candidate ridge is fitted on the centred/scaled design exactly as ROUND70 did:

        Xn = (X - mu) / sigma          (column-wise; sigma forced to 1 on constant columns)
        Xa = [Xn, 1]                   (intercept column last, unpenalised)
        pred = Xn @ A + c              A is (EMBED_DIM, 2), c is (2,)

    We need the equivalent map on the RAW embedding, pred = X @ B + b, because the real critic
    head is ``Linear(EMBED_DIM, 1)`` applied to the raw mean-over-targets embedding.

    Substituting Xn and collecting terms column by column:

        pred = sum_j ((X[:,j] - mu[j]) / sigma[j]) * A[j,:] + c
             = sum_j X[:,j] * (A[j,:] / sigma[j])  +  ( c - sum_j (mu[j] / sigma[j]) * A[j,:] )

    so, exactly:

        B[j, :] = A[j, :] / sigma[j]
        b       = c - mu @ B

    Round-trip check used by the tests: predict_standardised(X) == predict_raw(X) for every X.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    A = np.asarray(A, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    if mu.shape != (EMBED_DIM,) or sigma.shape != (EMBED_DIM,):
        raise ValueError(f"mu/sigma must be ({EMBED_DIM},), got {mu.shape} / {sigma.shape}")
    if A.shape != (EMBED_DIM, 2):
        raise ValueError(f"A must be ({EMBED_DIM}, 2), got {A.shape}")
    if c.shape != (2,):
        raise ValueError(f"c must be (2,), got {c.shape}")
    if np.any(sigma == 0.0):
        raise ValueError("sigma must already have constant columns set to 1 (never divide by 0)")
    B = A / sigma[:, None]
    b = c - mu @ B
    return B, b


def predict_standardized(X, mu, sigma, A, c):
    """Reference predictor in standardised space (what the ridge fit directly produces)."""
    X = np.asarray(X, dtype=np.float64)
    Xn = (X - np.asarray(mu, dtype=np.float64)) / np.asarray(sigma, dtype=np.float64)
    return Xn @ np.asarray(A, dtype=np.float64) + np.asarray(c, dtype=np.float64)


def predict_raw(X, B, b):
    """Reference predictor in raw 64-D embedding space (what the critic head computes)."""
    return np.asarray(X, dtype=np.float64) @ np.asarray(B, dtype=np.float64) + np.asarray(b, dtype=np.float64)


def affine_to_critic_heads(B, b):
    """Express (B, b) as the two real critic heads' Linear(EMBED_DIM, 1) weight/bias."""
    B = np.asarray(B, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if B.shape != (EMBED_DIM, 2) or b.shape != (2,):
        raise ValueError(f"expected ({EMBED_DIM}, 2) and (2,), got {B.shape} / {b.shape}")
    return {
        "value_task_head": {"weight": B[:, 0].reshape(1, EMBED_DIM).copy(),
                            "bias": np.array([b[0]], dtype=np.float64)},
        "value_cost_head": {"weight": B[:, 1].reshape(1, EMBED_DIM).copy(),
                            "bias": np.array([b[1]], dtype=np.float64)},
    }


# ========================================================================================
# 2. head-set container: the ONLY thing the candidate is allowed to replace
# ========================================================================================
class CriticHeadSet:
    """Holds ONLY the two critic heads.

    ``replace_heads`` returns a NEW CriticHeadSet and never mutates, rebinds or closes over any
    actor/trunk parameter object, so the candidate cannot silently change the policy modules.
    """

    def __init__(self, heads):
        missing = [m for m in CRITIC_HEAD_MODULES if m not in heads]
        if missing:
            raise ValueError(f"critic head set missing modules: {missing}")
        self.heads = {m: {"weight": np.array(heads[m]["weight"], dtype=np.float64, copy=True),
                          "bias": np.array(heads[m]["bias"], dtype=np.float64, copy=True)}
                      for m in CRITIC_HEAD_MODULES}

    def digest(self):
        import hashlib
        h = hashlib.sha256()
        for m in CRITIC_HEAD_MODULES:
            h.update(m.encode())
            h.update(np.ascontiguousarray(self.heads[m]["weight"]).tobytes())
            h.update(np.ascontiguousarray(self.heads[m]["bias"]).tobytes())
        return h.hexdigest()

    def replace_heads(self, new_heads):
        return CriticHeadSet(new_heads)


class ParameterContainer:
    """Minimal stand-in for the real 5-module parameter container, for tests only."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.modules = {}
        for m in MODULES:
            if m in CRITIC_HEAD_MODULES:
                self.modules[m] = {"weight": rng.normal(size=(1, EMBED_DIM)),
                                   "bias": rng.normal(size=(1,))}
            else:
                self.modules[m] = {"weight": rng.normal(size=(EMBED_DIM, EMBED_DIM)),
                                   "bias": rng.normal(size=(EMBED_DIM,))}

    def install_critic_heads(self, head_set):
        """Replace ONLY the two critic heads; returns the untouched module names."""
        touched = []
        for m in CRITIC_HEAD_MODULES:
            self.modules[m] = {"weight": np.array(head_set.heads[m]["weight"], copy=True),
                               "bias": np.array(head_set.heads[m]["bias"], copy=True)}
            touched.append(m)
        return touched

# ========================================================================================
# 3. resource / entry guard
# ========================================================================================
class ForwardLedger:
    """Persistent, atomic-before-forward counter of REAL 19-D trunk input-vector rows.

    Billing unit (LOCKED, and deliberately NOT the joint decision row and NOT the call count):
    the actual rows of the 19-D matrix handed to the trunk.  A decision step whose embedding is
    the mean over k targets therefore costs k input rows, and every first attempt, failure,
    retry and repeat is charged.

    ``debit`` is called BEFORE the forward and refuses when the charge would cross the cap, so
    an over-cap batch is never started.  This is an in-process guard over a persisted counter;
    it does NOT claim physical power-loss atomicity -- a spy or no-op is not a real update, and
    this class does not pretend otherwise.
    """

    def __init__(self, cap=OFFLINE_INPUT_ROW_CAP, state_path=None):
        self.cap = int(cap)
        self.rows = 0
        self.calls = 0
        self.refusals = 0
        self.joint_rows = 0          # informational only; NEVER used for the cap comparison
        self.state_path = Path(state_path) if state_path else None
        if self.state_path is not None and self.state_path.is_file():
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.rows = int(saved.get("rows", 0))
            self.calls = int(saved.get("calls", 0))

    def _persist(self):
        if self.state_path is None:
            return
        self.state_path.write_text(json.dumps({"rows": self.rows, "calls": self.calls}),
                                   encoding="utf-8")

    def debit(self, n_input_rows, n_joint_rows=0):
        """Atomically reserve n_input_rows BEFORE any forward.  Refuses before exceeding."""
        n_input_rows = int(n_input_rows)
        if n_input_rows < 0:
            raise ValueError("negative input-row charge")
        if self.rows + n_input_rows > self.cap:
            self.refusals += 1
            raise BudgetExceeded(
                f"forward ledger: refusing {n_input_rows} input rows; used {self.rows} of "
                f"cap {self.cap} (would reach {self.rows + n_input_rows})")
        self.rows += n_input_rows
        self.joint_rows += int(n_joint_rows)
        self.calls += 1
        self._persist()
        return self.rows

    def remaining(self):
        return self.cap - self.rows


def build_batches(target_counts, cap=BATCH_INPUT_ROW_CAP):
    """Group decision steps into forward batches of at most ``cap`` REAL input rows.

    Whole decision steps are kept together (a step's embedding is the mean over its targets),
    and a step is only added when it still fits.  A single step whose own target count exceeds
    the cap is emitted alone and reported as an over-cap batch by the caller, never silently
    split, because splitting would change the mean-over-targets aggregation.
    """
    batches, batch, batch_rows = [], [], 0
    for k in target_counts:
        k = int(k)
        if batch and (batch_rows + k > cap):
            batches.append(batch)
            batch, batch_rows = [], 0
        batch.append(k)
        batch_rows += k
    if batch:
        batches.append(batch)
    return batches


class LicenseGate:
    """Fail-closed gate.  A real experiment licence token is REQUIRED for every gated action.

    No licence is present this round, so ``require`` always raises.  The gate is checked
    BEFORE any data fit, model forward, claim or checkpoint write, and before any episode
    call.  It deliberately adds no extra conditions of its own (no "receipt must live outside
    the repo", no "exactly N claim files", and similar invented requirements).
    """

    GATED_ACTIONS = ("fit_real_data", "model_forward", "claim_write", "checkpoint_write",
                     "episode_call", "optimizer_step", "ppo_replay", "environment_build")

    def __init__(self, token=None):
        self.token = token

    @property
    def licensed(self):
        return bool(self.token)

    def require(self, action):
        if action not in self.GATED_ACTIONS:
            raise ValueError(f"unknown gated action: {action}")
        if not self.licensed:
            raise LicenseMissing(
                f"action '{action}' refused: no real experiment licence token is present "
                f"(ROUND71 is zero-consumption preparation only; the candidate plan is "
                f"APPROVAL_PENDING and must not be treated as an authorisation)")


class DuplicateOutputGuard:
    """Refuse to implicitly overwrite an existing artefact with a re-run."""

    def __init__(self, path):
        self.path = Path(path)

    def check(self, overwrite=False):
        if self.path.exists() and not overwrite:
            raise DuplicateOutput(
                f"refusing implicit re-run: {self.path} already exists; pass an explicit "
                f"overwrite decision instead of silently replacing prior output")


def validate_restore_source(source):
    """Refuse any old checkpoint as a NEW training restore source.

    The candidate must start from a legitimate NEW training state built on the qualified R68
    ep0 parameters.  A trained ep128, an R68 four-module checkpoint, or anything carrying a
    trained optimizer / lambda / RNG state is NOT a legal initialisation or restore source.
    """
    if not isinstance(source, dict):
        raise IllegalRestoreSource("restore source must be a declared dict")
    declared = source.get("declared_purpose")
    if declared != "new_training_initialisation":
        raise IllegalRestoreSource(
            f"restore source declared_purpose={declared!r}; only "
            f"'new_training_initialisation' is legal for the candidate")
    if source.get("carries_trained_optimizer_state"):
        raise IllegalRestoreSource(
            "restore source carries a trained optimizer state; the candidate must start a "
            "fresh optimizer from the R68 ep0 parameters")
    if source.get("carries_trained_lambda"):
        raise IllegalRestoreSource(
            "restore source carries a trained dual lambda; the candidate must start from the "
            "R68 ep0 lambda")
    if source.get("carries_training_rng"):
        raise IllegalRestoreSource(
            "restore source carries a training RNG state; the candidate must start from the "
            "frozen initial random state")
    if source.get("from_trained_ep128"):
        raise IllegalRestoreSource(
            "restore source is a trained ep128 checkpoint; the candidate may not be "
            "initialised or resumed from a trained ep128")
    modules = set(source.get("modules", ()))
    if modules and modules != set(MODULES):
        raise IllegalRestoreSource(
            f"restore source module set {sorted(modules)} != the 5-module contract "
            f"{sorted(MODULES)}; an old single-head / four-module checkpoint is not a legal "
            f"NEW restore source")
    return True


# ========================================================================================
# 4. the injectable initialiser seam
# ========================================================================================
class CalibrationInitializer:
    """Orchestrates the candidate critic-calibration initialisation.

    Every step that would touch real state is gated.  The collaborators (embedding extractor,
    ridge fitter, head installer) are injected so tests can substitute pure-arithmetic or spy
    implementations and never reach a real model.  With no licence token, ``run`` fails closed
    at the FIRST gate, before any fit, forward, claim or checkpoint write.
    """

    def __init__(self, gate, ledger, embedding_extractor=None, ridge_fitter=None,
                 head_installer=None, output_path=None):
        self.gate = gate
        self.ledger = ledger
        self.embedding_extractor = embedding_extractor
        self.ridge_fitter = ridge_fitter
        self.head_installer = head_installer
        self.output_path = Path(output_path) if output_path else None
        self.trace = []

    def run(self, seeds, target_counts_by_seed):
        # gate 1: real-data fit
        self.gate.require("fit_real_data")
        self.trace.append("gate:fit_real_data")
        # gate 2: model forward (the embedding extraction itself)
        self.gate.require("model_forward")
        self.trace.append("gate:model_forward")
        # reserve the ledger BEFORE any forward, per batch of real input rows
        for seed in seeds:
            for batch in build_batches(target_counts_by_seed[seed]):
                self.ledger.debit(sum(batch), n_joint_rows=len(batch))
            self.trace.append(f"ledger:seed{seed}")
        embeddings = {s: self.embedding_extractor(s) for s in seeds}
        fits = {s: self.ridge_fitter(s, embeddings[s]) for s in seeds}
        # gate 3: checkpoint write
        self.gate.require("checkpoint_write")
        self.trace.append("gate:checkpoint_write")
        DuplicateOutputGuard(self.output_path).check()
        heads = {s: self.head_installer(s, fits[s]) for s in seeds}
        return heads


# ========================================================================================
# 5. candidate plan registration (APPROVAL_PENDING; explicitly NOT an authorisation)
# ========================================================================================
CANDIDATE_PLAN = {
    "status": "APPROVAL_PENDING",
    "intervention": ("initialise the dual critic affine heads of each R68 seed model from a "
                     "ridge calibration on that seed's own existing training teacher data over "
                     "the frozen trunk's true mean-over-targets representation"),
    "unchanged": ("trunk, early/late actor heads (late initial value scaled x0.10), the initial "
                  "policy, and a legitimate NEW training state from the qualified R68 ep0 "
                  "(optimizer / lambda / RNG)"),
    "fit_config": {"ridge": RIDGE, "center_scale_scope": CENTER_SCALE_SCOPE,
                   "constant_column_scale": CONSTANT_COLUMN_SCALE,
                   "intercept_penalized": INTERCEPT_PENALIZED,
                   "fits_per_seed": FITS_PER_SEED,
                   "forbidden": ["180 evaluation data", "threshold fitting", "seed selection",
                                 "treating late labels as safe ground truth"]},
    "budgets": {"offline_input_row_cap": OFFLINE_INPUT_ROW_CAP,
                "batch_input_row_cap": BATCH_INPUT_ROW_CAP,
                "candidate_synthetic_optimizer_cap": CANDIDATE_SYNTHETIC_OPTIMIZER_CAP,
                "note": ("the offline cap constrains the offline phase only; any future "
                         "licensed training/evaluation consumption is counted separately and "
                         "in full")},
    "future_run_plan_if_ever_licensed": {
        "normal": 582, "technical_reserve": 18, "hard_total": 600,
        "breakdown": ("384 training from the calibrated initialisation + 192 fixed-ep128 "
                      "evaluations (3 seeds x 2 modes x 32 old dev scenes 1800000..1800031) "
                      "+ 6 frozen-focus greedy cells at scene 1700027 (INIT/TRAINED x 3 seeds)"),
        "controls": ("R68 same-seed / same-mode / same-scenario controls and the rule arms are "
                     "read-only reuse subject to source, config and behaviour-policy identity "
                     "eligibility; no old-round quota is borrowed"),
    },
    "not_claimed": ("extra offline learning cost and a policy-distribution change risk are "
                    "acknowledged; this is not a same-training-budget comparison and no "
                    "improvement is pre-claimed"),
}