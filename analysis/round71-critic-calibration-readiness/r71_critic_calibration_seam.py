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

import contextlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

FEATURE_DIM = 19          # the real 19-D per-target trunk input row

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

# restore-source role contract (correction pass): an INIT source is the qualified R68 ep0
# model-only parameters; a FULL restore may only carry THIS NEW run's own trained state.
RESTORE_ROLES = ("r68_ep0_model_only", "new_full_restore", "rule_control_readonly")
RESTORE_IDENTITY_FIELDS = ("seed", "role", "structure", "module_set",
                           "training_version", "source_kind")
INIT_SOURCE_KINDS = ("r68_ep0",)            # the qualified R68 ep0 model-only parameters
OWN_RUN_SOURCE_KINDS = ("this_run",)        # this NEW run's own checkpoint
FOREIGN_SOURCE_KINDS = ("foreign_run", "other_round", "r58", "r60", "r66", "r68_trained")

# candidate diagnostic configuration, LOCKED (registered, not licensed)
RIDGE = 1e-3                 # fixed; no sweep
CENTER_SCALE_SCOPE = "all training teacher rows of that seed"   # no held-out fold in the candidate
CONSTANT_COLUMN_SCALE = 1.0  # constant column scale := 1
INTERCEPT_PENALIZED = False  # intercept unpenalised
FITS_PER_SEED = 1            # one dual-output fit per seed, no CV, no seed selection

# candidate resource budgets (offline phase only; any licensed run is counted separately)
OFFLINE_INPUT_ROW_CAP = 72000   # actual 19-D trunk input-vector rows
BATCH_INPUT_ROW_CAP = 1024      # max real input rows per forward batch

# persisted-ledger identity: a state file of any other identity is NEVER adopted as ours
LEDGER_IDENTITY = "round71-critic-calibration-offline-trunk-input-ledger/1"
CANDIDATE_SYNTHETIC_OPTIMIZER_CAP = 16   # registered only; 0 consumed this round


class LicenseMissing(RuntimeError):
    """Raised when a gated action is attempted without a real experiment licence."""


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a forward that would push the ledger past its cap."""


class DuplicateOutput(RuntimeError):
    """Raised when an existing artefact would be implicitly overwritten by a re-run."""


class IllegalRestoreSource(RuntimeError):
    """Raised when an old checkpoint is offered as a NEW training restore source."""


class LedgerStateError(RuntimeError):
    """Persisted ledger state is missing, corrupt, inconsistent, or of a foreign identity."""


class LicenseInvalid(RuntimeError):
    """A presented licence object failed verification and is NOT a licence."""


class BatchTooLarge(RuntimeError):
    """A REAL trunk input batch exceeds the per-batch input-row cap."""


class TrunkInputError(RuntimeError):
    """A real trunk input matrix failed shape validation at the trunk entry."""


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
    an over-cap batch is never started.  This is an in-process guard over a persisted counter.

    Final correction (R71): the budget determination, the refusal, the rows/calls/joint_rows
    computation and the atomic save all happen INSIDE the same lock, with the base value taken
    from the validated latest DURABLE state (not from a possibly stale in-memory copy), and the
    in-memory state is published only after the new state is durable.  An exclusive lock alone
    does not prevent a lost update: two instances constructed from the same state file would
    both pass a check made against their own stale ``self.rows``.  A refusal or a persist
    failure therefore also computes from the fresh durable base and never overwrites another
    instance's already-consumed state.

    Lock recovery is deliberately NOT implemented: ANY residual lock fails closed and no basis
    text authorises deleting a lock.  A formal recovery path would need a prior legal approval
    plus independent read-only owner-finished evidence, and this zero-consumption preparation
    has no real run lock to recover.  This class does NOT claim physical power-loss atomicity,
    and a spy or no-op is not a real update; it does not pretend otherwise.
    """

    def __init__(self, cap=OFFLINE_INPUT_ROW_CAP, state_path=None):
        self.cap = int(cap)
        self.rows = 0
        self.calls = 0
        self.refusals = 0
        self.joint_rows = 0          # informational only; NEVER used for the cap comparison
        self.state_path = Path(state_path) if state_path else None
        self._load()

    # ------------------------------------------------------------------ state validation
    def _read_persisted(self):
        """Validated latest DURABLE state dict, or None when no durable file exists yet."""
        if self.state_path is None or not self.state_path.is_file():
            return None
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LedgerStateError(
                f"forward ledger state {self.state_path} is unreadable or corrupt: {exc}") from exc
        if not isinstance(saved, dict):
            raise LedgerStateError("forward ledger state must be a JSON object")
        for key in ("identity", "cap", "rows", "calls", "refusals", "joint_rows"):
            if key not in saved:
                raise LedgerStateError(
                    f"forward ledger state is missing the required key {key!r}; a partial or "
                    f"foreign state file is never silently treated as zero")
        for key in ("cap", "rows", "calls", "refusals", "joint_rows"):
            v = saved[key]
            if isinstance(v, bool) or not isinstance(v, int):
                raise LedgerStateError(f"forward ledger state key {key!r} must be an int")
            if v < 0:
                raise LedgerStateError(
                    f"forward ledger state key {key!r} must be non-negative")
        if saved["identity"] != LEDGER_IDENTITY:
            raise LedgerStateError(
                f"forward ledger state identity {saved['identity']!r} != {LEDGER_IDENTITY!r}; "
                f"a foreign ledger is never adopted as this round's consumption")
        if saved["cap"] != self.cap:
            raise LedgerStateError(
                f"forward ledger state cap {saved['cap']} != this ledger's cap {self.cap}")
        if saved["rows"] > saved["cap"]:
            raise LedgerStateError(
                f"forward ledger state rows {saved['rows']} exceed its cap {saved['cap']}")
        return saved

    def _load(self):
        """Adopt persisted state ONLY after full validation; never silently zero."""
        fresh = self._read_persisted()
        if fresh is None:
            return
        self.rows = fresh["rows"]
        self.calls = fresh["calls"]
        self.refusals = fresh["refusals"]
        self.joint_rows = fresh["joint_rows"]

    def _state_dict(self):
        return {"identity": LEDGER_IDENTITY, "cap": self.cap, "rows": self.rows,
                "calls": self.calls, "refusals": self.refusals,
                "joint_rows": self.joint_rows}

    def _persist(self):
        """Atomically replace the state file: same-dir temp, fsync, os.replace."""
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._state_dict(), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    def _publish(self, candidate):
        """Persist the candidate state FIRST; commit to memory only once it is durable."""
        saved = (self.rows, self.calls, self.refusals, self.joint_rows)
        (self.rows, self.calls, self.refusals, self.joint_rows) = candidate
        try:
            self._persist()
        except OSError:
            (self.rows, self.calls, self.refusals, self.joint_rows) = saved
            raise

    # ------------------------------------------------------------------ locking
    @contextlib.contextmanager
    def _locked(self):
        """Serialise mutation.  ANY residual lock fails closed, with no exceptions.

        Correction note (R71, final): the previous version deleted a residual lock when the
        caller supplied a ``recovery_basis`` string, which could delete a LIVE lock, and the
        recorded basis was memory-only, not durable recovery evidence.  Lock recovery is not
        implemented this round: no text authorises deleting a lock, there is no pid probing
        and no process is ever signalled.
        """
        if self.state_path is None:
            yield
            return
        lock = self.state_path.with_name(self.state_path.name + ".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LedgerStateError(
                f"residual ledger lock {lock} is present; refusing to mutate (lock recovery "
                f"is not implemented and no basis text authorises deleting a lock)")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(str(os.getpid()))
            yield
        finally:
            try:
                os.unlink(lock)
            except OSError:
                pass

    def debit(self, n_input_rows, n_joint_rows=0):
        """Atomically reserve n_input_rows BEFORE any forward.  Refuses before exceeding.

        The whole decision (base value, budget check, refusal, new counters, atomic save) runs
        inside the lock and reads the validated latest durable state, so two instances sharing
        a state file cannot both pass a check made against a stale copy.  In-memory state is
        published only after the new state is durable, and a refusal or persist failure never
        overwrites another instance's consumption.
        """
        n_input_rows = int(n_input_rows)
        n_joint_rows = int(n_joint_rows)
        if n_input_rows < 0:
            raise ValueError("negative input-row charge")
        with self._locked():
            fresh = self._read_persisted()
            if fresh is None:
                base = (self.rows, self.calls, self.refusals, self.joint_rows)
            else:
                base = (fresh["rows"], fresh["calls"], fresh["refusals"],
                        fresh["joint_rows"])
            if base[0] + n_input_rows > self.cap:
                # refusal against the FRESH base: another instance's consumption is preserved
                self._publish((base[0], base[1], base[2] + 1, base[3]))
                raise BudgetExceeded(
                    f"forward ledger: refusing {n_input_rows} input rows; used {base[0]} of "
                    f"cap {self.cap} (would reach {base[0] + n_input_rows})")
            self._publish((base[0] + n_input_rows, base[1] + 1, base[2],
                           base[3] + n_joint_rows))
        return self.rows

    def remaining(self):
        return self.cap - self.rows


def build_batches(target_counts, cap=BATCH_INPUT_ROW_CAP):
    """ESTIMATE-ONLY planner: group decision steps into prospective forward batches.

    Correction note (R71): the per-step target counts are ESTIMATES known before any real
    trunk call, so this planner is NOT billing evidence and must never pre-debit the ledger.
    The real charge is the actual row count of the matrix handed to the trunk, validated and
    atomically debited at the trunk entry (see ``CalibrationInitializer.guarded_trunk_forward``),
    which REFUSES an over-cap real batch (``BatchTooLarge``) instead of forwarding it.  The
    earlier "never silently split, because splitting would change the mean-over-targets
    aggregation" rationale is not a licence to forward an over-cap batch; it only explains why
    the planner does not itself split.  Planner behaviour is unchanged: whole decision steps
    are kept together, a step is only added when it still fits, and a single step whose own
    target count exceeds the cap is emitted alone for the caller to report.
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


@dataclass(frozen=True)
class RealLicense:
    """A verified experiment licence.  A raw token string or dict is NEVER a licence.

    Correction note (R71): the previous gate accepted ANY truthy token.  A licence must now
    carry a complete declared identity, and only a verifier-accepted ``RealLicense`` counts.
    """

    approval_id: str
    task_id: str
    source_freeze_sha256: str
    input_identity_sha256: str
    quota_identity: str
    issued_by: str
    test_only: bool = False

    @classmethod
    def from_mapping(cls, mapping):
        if not isinstance(mapping, dict):
            raise LicenseInvalid("a licence must be a declared mapping")
        required = ("approval_id", "task_id", "source_freeze_sha256",
                    "input_identity_sha256", "quota_identity", "issued_by")
        missing = [k for k in required if not mapping.get(k)]
        if missing:
            raise LicenseInvalid(f"licence mapping is missing fields: {missing}")
        return cls(**{k: mapping[k] for k in required},
                   test_only=bool(mapping.get("test_only", False)))


def spy_license_verifier(license):
    """Spy verifier for tests ONLY; it can never unlock the real provider."""
    return isinstance(license, RealLicense)


spy_license_verifier.test_only = True


class LicenseGate:
    """Fail-closed gate.  A VERIFIED RealLicense is REQUIRED for every gated action.

    Correction note (R71): ``licensed`` is no longer ``bool(self.token)``.  A licence must be
    a ``RealLicense`` accepted by a verifier; with no verifier the gate refuses everything; a
    test-only spy verifier can never unlock the real provider; and ``require`` refuses both an
    absent licence and a test-only one.  No real licence is present this round, so every gated
    action still fails closed, and the gate deliberately adds no invented conditions of its
    own (no "receipt must live outside the repo", no "exactly N claim files").
    """

    GATED_ACTIONS = ("fit_real_data", "model_forward", "claim_write", "checkpoint_write",
                     "episode_call", "optimizer_step", "ppo_replay", "environment_build")

    def __init__(self, token=None, verifier=None):
        self.verifier = verifier
        self._license = None
        if token is None:
            return
        if not isinstance(token, RealLicense):
            raise LicenseInvalid(
                f"a raw {type(token).__name__} is not a licence; a verifier-accepted "
                f"RealLicense object is required")
        if verifier is None or not verifier(token):
            raise LicenseInvalid("the presented licence failed verification")
        self._license = token

    @property
    def licensed(self):
        return self.present()

    def present(self):
        """True only on the REAL path: a verified, non-test-only licence AND verifier."""
        if self._license is None or self._license.test_only:
            return False
        if self.verifier is None or getattr(self.verifier, "test_only", False):
            return False
        return True

    def present_for_test(self):
        """Spy path: a test-only licence was accepted by the test-only spy verifier."""
        return (self._license is not None and self._license.test_only
                and getattr(self.verifier, "test_only", False))

    def require(self, action):
        if action not in self.GATED_ACTIONS:
            raise ValueError(f"unknown gated action: {action}")
        if self.present():
            return
        if self._license is not None and (self._license.test_only
                                          or getattr(self.verifier, "test_only", False)):
            raise LicenseInvalid(
                f"action '{action}' refused: only a TEST-ONLY licence/verifier is present; "
                f"it cannot unlock the real provider")
        raise LicenseMissing(
            f"action '{action}' refused: no verified real experiment licence is present "
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
    """Refuse any illegal NEW training initialise/restore source (descriptive layer only).

    Correction note (R71): the previous version trusted caller boolean flags and let an EMPTY
    module list pass.  A source must now declare a complete identity block (seed, role,
    structure, module_set, training_version, source_kind), and the flags are interpreted
    through that identity:

      * ``r68_ep0_model_only`` (source_kind ``r68_ep0``) is the legal INIT source: the
        qualified R68 ep0 model parameters.  It MAY carry the frozen INITIAL random state
        (``carries_initial_rng_state``) -- that is NOT a trained RNG and must not be
        misclassified as one -- but it may NOT carry a trained optimizer, a trained dual
        lambda, or advanced training RNG state;
      * ``new_full_restore`` (source_kind ``this_run``) may carry THIS NEW run's own trained
        optimizer / lambda / advanced training RNG; a FOREIGN run's trained state is refused,
        so a future legitimate resume of this run is not blocked while R58/R60/R66/R68
        artefacts are still refused;
      * ``rule_control_readonly`` is a read-only rule control and may carry no trained state.

    A trained ep128, an old single-head / four-module checkpoint, an incomplete identity, a
    foreign structure/version, or both initial and advanced RNG flags together are all
    refused.  This check is descriptive only: it opens no .pt file and restores nothing.
    """
    if not isinstance(source, dict):
        raise IllegalRestoreSource("restore source must be a declared dict")
    identity = source.get("identity")
    if not isinstance(identity, dict):
        raise IllegalRestoreSource(
            "restore source must declare an 'identity' block; caller flags alone are not a "
            "declaration")
    missing = [f for f in RESTORE_IDENTITY_FIELDS if f not in identity]
    if missing:
        raise IllegalRestoreSource(f"restore source identity is incomplete: missing {missing}")
    role = identity["role"]
    if role not in RESTORE_ROLES:
        raise IllegalRestoreSource(
            f"restore source role {role!r} not in {RESTORE_ROLES}")
    seed = identity["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise IllegalRestoreSource("restore source identity 'seed' must be an int")
    if identity["structure"] != R68_STRUCTURE:
        raise IllegalRestoreSource(
            f"restore source structure {identity['structure']!r} != {R68_STRUCTURE!r}")
    tv = identity["training_version"]
    if not isinstance(tv, str) or not tv.startswith(TRAINING_VERSION):
        raise IllegalRestoreSource(
            "restore source training_version must extend the frozen R68 version string")
    kind = identity["source_kind"]
    module_set = set(identity["module_set"] or ())
    if role != "rule_control_readonly" and module_set != set(MODULES):
        raise IllegalRestoreSource(
            f"restore source module set {sorted(module_set)} != the 5-module contract "
            f"{sorted(MODULES)}; an old single-head / four-module checkpoint is not a legal "
            f"NEW restore source")
    carries_opt = bool(source.get("carries_trained_optimizer_state"))
    carries_lam = bool(source.get("carries_trained_lambda"))
    carries_adv_rng = bool(source.get("carries_advanced_training_rng"))
    carries_init_rng = bool(source.get("carries_initial_rng_state"))
    if carries_init_rng and carries_adv_rng:
        raise IllegalRestoreSource(
            "a source cannot carry both the initial and an advanced training RNG state")
    if source.get("from_trained_ep128"):
        raise IllegalRestoreSource(
            "restore source is a trained ep128 checkpoint; the candidate may not be "
            "initialised or resumed from a trained ep128")
    if role == "r68_ep0_model_only":
        if kind not in INIT_SOURCE_KINDS:
            raise IllegalRestoreSource(
                f"r68_ep0_model_only source_kind {kind!r} not in {INIT_SOURCE_KINDS}")
        if carries_opt or carries_lam or carries_adv_rng:
            raise IllegalRestoreSource(
                "the R68 ep0 INIT source carries trained optimizer/lambda/RNG state; the "
                "candidate must start a fresh optimizer, the ep0 lambda and the frozen "
                "initial random state")
    elif role == "new_full_restore":
        if kind not in OWN_RUN_SOURCE_KINDS:
            raise IllegalRestoreSource(
                f"new_full_restore source_kind {kind!r} not in {OWN_RUN_SOURCE_KINDS}; a "
                f"foreign run's trained state (e.g. {list(FOREIGN_SOURCE_KINDS)}) is never a "
                f"legal restore source for this run")
    else:  # rule_control_readonly
        if carries_opt or carries_lam or carries_adv_rng:
            raise IllegalRestoreSource("a read-only rule control may carry no trained state")
    return True


# ========================================================================================
# 4. the injectable initialiser seam
# ========================================================================================
class CalibrationInitializer:
    """Orchestrates the candidate critic-calibration initialisation.

    Every step that would touch real state is gated.  The collaborators (embedding extractor,
    ridge fitter, head installer, trunk delegate) are injected so tests can substitute
    pure-arithmetic or spy implementations and never reach a real model.  With no licence
    token, ``run`` fails closed at the FIRST gate, before any fit, forward, claim or
    checkpoint write.

    Correction note (R71): ``run`` no longer pre-debits the ledger from ESTIMATED per-step
    target counts.  The only charge is the REAL row count of the matrix handed to the trunk,
    validated and atomically debited inside ``guarded_trunk_forward`` at the trunk entry,
    which then delegates the same matrix exactly once.  Estimated counts are recorded as
    trace metadata only and are never billing evidence; joint decision rows and real
    target-level input rows are counted separately by the ledger.
    """

    def __init__(self, gate, ledger, embedding_extractor=None, ridge_fitter=None,
                 head_installer=None, output_path=None, trunk_delegate=None):
        self.gate = gate
        self.ledger = ledger
        self.embedding_extractor = embedding_extractor
        self.ridge_fitter = ridge_fitter
        self.head_installer = head_installer
        self.output_path = Path(output_path) if output_path else None
        # the real trunk forward is NOT wired this round; the default delegate is an identity
        # echo so the guard boundary itself can be tested without any neural computation
        self.trunk_delegate = trunk_delegate or (lambda matrix: matrix)
        self.trace = []

    def guarded_trunk_forward(self, matrix, n_joint_rows=1):
        """The single REAL trunk entry: validate, debit the actual rows, delegate once.

        Refuses BEFORE any charge: a non-2-D matrix, a wrong feature width, a non-positive
        row count, or more than ``BATCH_INPUT_ROW_CAP`` real input rows.  Every accepted call
        is charged, including repeats and calls whose delegate then fails.
        """
        m = np.asarray(matrix)
        if m.ndim != 2 or m.shape[1] != FEATURE_DIM:
            raise TrunkInputError(
                f"trunk input must be a 2-D (N, {FEATURE_DIM}) matrix, got shape {m.shape}")
        n = int(m.shape[0])
        if n <= 0:
            raise TrunkInputError("trunk input must carry at least one real row")
        if n > BATCH_INPUT_ROW_CAP:
            raise BatchTooLarge(
                f"trunk input batch of {n} real rows exceeds the per-batch cap "
                f"{BATCH_INPUT_ROW_CAP}; refused BEFORE any charge")
        self.ledger.debit(n, n_joint_rows=n_joint_rows)
        return self.trunk_delegate(m)

    def run(self, seeds, target_counts_by_seed=None):
        # gate 1: real-data fit
        self.gate.require("fit_real_data")
        self.trace.append("gate:fit_real_data")
        # gate 2: model forward (the embedding extraction itself)
        self.gate.require("model_forward")
        self.trace.append("gate:model_forward")
        # estimated per-step counts are TRACE METADATA ONLY -- never a ledger pre-debit
        if target_counts_by_seed:
            for seed in seeds:
                est = sum(int(k) for k in target_counts_by_seed.get(seed, ()))
                self.trace.append(f"estimate_only:seed{seed}:{est}")
        # the extractor receives the guarded trunk entry and is charged for REAL rows only
        embeddings = {s: self.embedding_extractor(s, self.guarded_trunk_forward)
                      for s in seeds}
        fits = {s: self.ridge_fitter(s, embeddings[s]) for s in seeds}
        # gate 3: checkpoint write
        self.gate.require("checkpoint_write")
        self.trace.append("gate:checkpoint_write")
        if self.output_path is None:
            raise ValueError("a licensed run must declare an explicit output path")
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