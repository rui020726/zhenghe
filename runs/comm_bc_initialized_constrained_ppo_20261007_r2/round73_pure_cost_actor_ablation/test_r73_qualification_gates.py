# -*- coding: utf-8 -*-
"""ROUND73 qualification gates: the REAL paths, on the live CLI-configured runtime.

These gates close the five protocol gaps the pre-launch suite deliberately left open.  They
perform ZERO real environment constructions and ZERO trunk forwards over real episodes.  Every
optimizer step they take is a SYNTHETIC one, charged to a synthetic-step guard under its own cap
(8) and written to an independent synthetic fixture directory that is never the real raw root.

Gap 1  the real three-seed controller really comes from the registered ROUND71 calibrated INIT,
        model-only, five modules, the late x0.10 scale is NOT repeated, and optimizer / lambda /
        training RNG all start fresh.
Gap 2  on the LIVE trainer and the REAL routed PPO surrogate / likelihood path, with A_cost and
        the other actor inputs fixed, a non-zero A_task and a different lambda cannot change the
        actor loss or the pre-clip actor gradient.  (The total update, which also contains the
        critic losses and the global clip, is NOT claimed invariant.)
Gap 3  the p=0.58 / SEND-cost c / SKIP-cost 0 / V_cost=p*c hand example goes through the real
        legal-action, forced, mask and ALL_ROWS actor path; old == new; the pre-clip ascent
        gradient is -0.2436c and the minimised loss has the opposite sign; the entropy and gate
        prior contributions are fixed and separated.
Gap 4  a real ROUND73 save / load / full-resume round trip with NON-EMPTY Adam momentum, an
        advanced lambda, an advanced RNG and an advanced cursor reproduces the next update
        deterministically, and the real load entry refuses a ROUND71 / ROUND68 payload (and an
        empty-Adam payload) as a NEW resume source.
Gap 5  synthetic optimizer steps and offline trunk rows are charged BEFORE they happen and
        blocked by the hard limit; the guards read their real persisted state and never
        hard-fill a number.

The live entry point exercised here is ``trainer.ppo_update`` exactly as the real training loop
calls it (``r68_execute.routed_ppo``: the R53 critic->trunk isolation context around the routed
``frozen_early_ppo_update``).  That entry point appends to the REAL training journals, which are
rehydrated into the live call counters at configure time and must close exactly against committed
batch records.  So every call here (a) charges the synthetic-step guard and the trunk-row ledger
BEFORE anything runs, (b) swaps the ROUND43 Adam.step journal hook for the pristine step, and
(c) quarantines - byte for byte, with sha256, into an independent synthetic archive - every line
the routed update appended to ``raw/runtime/ppo_update_calls.jsonl``, restoring that journal to
its exact pre-call bytes.  The final reconciliation therefore separates synthetic gate calls from
real training updates using the committed batch records and this receipt, never a journal that
silently mixes the two.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# ONLY this round's directory is put on the path here.  The frozen-source copies under
# runs/comm_cost_* and round51_frozen_source_gradient_close are deliberately NOT added: they
# would shadow the LIVE runtime modules and break the runtime's own import-provenance assertion
# (v2_common must resolve to zhenghe\runs\..., not to a copy inside zhenghe-round42).  The live
# runtime paths are established by r73_trial_execute / r68_execute themselves, so the imports
# below are ordered AFTER the r73 imports that set them up.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r73_trial_binding as trial                                          # noqa: E402
import r73_trial_execute as execute                                        # noqa: E402
import r73_checkpoint as ckpt                                              # noqa: E402
import r73_pure_cost_objective as r73                                      # noqa: E402

import r66_frozen as r66                                                   # noqa: E402
import r68_frozen as r68                                                   # noqa: E402
import r68_guards                                                          # noqa: E402

# ``v2_common`` and ``r73_trial_gates`` are deliberately NOT imported at module load.  The live
# canonical tree reaches sys.path only through the real CLI path itself (r68_execute.
# configure_and_inject -> ... -> r42_run.configure, which first verifies the module-default GAE
# lambda and only then installs the ROUND58 full-trace override).  setUpClass therefore stands the
# runtime up through that same entry point and binds the LIVE modules from the configured
# runtime, so every assertion below reads byte-for-byte the objects the real training loop uses.
_LIVE_ROOTS = ("zhenghe\\runs\\",)

SEEDS = (11, 22, 33)
RAW = HERE / "raw"
SYNTH_ROOT = RAW / "synthetic_test_fixtures"
PPO_JOURNAL = RAW / "runtime" / "ppo_update_calls.jsonl"
SYNTH_JOURNAL_RECEIPT = SYNTH_ROOT / "R73_SYNTHETIC_PPO_JOURNAL_RECEIPT.json"
# The REAL training journals.  They must close exactly against committed batch records, so a
# synthetic gate call may never leave a line behind: the ROUND43 Adam.step hook is swapped for
# the pristine step around every gate update, and any line the routed update appends to the PPO
# journal is quarantined byte-for-byte into the synthetic archive afterwards.
STEP_JOURNAL = RAW / "runtime" / "optimizer_step_calls.jsonl"
DIVERT_DIR = SYNTH_ROOT / "diverted_real_journal_lines"
# Pristine Adam.step captured at import, BEFORE configure installs the ROUND43 journal hook.
_PRISTINE_ADAM_STEP = torch.optim.Adam.step

# Every explicit gate trunk forward is debited from the ONE persisted round ledger BEFORE it
# runs (never refunded).  The list keeps the per-charge receipt so the closing gate can prove
# the ledger equals exactly what these gates consumed.
_GATE_FORWARD_CHARGES: list = []


def _charge_gate_forward(n_rows: int, what: str) -> None:
    """Debit the persisted trunk-input-row ledger BEFORE a gate forward; never refunded."""
    gates.forward_ledger(RAW).debit(int(n_rows))
    _GATE_FORWARD_CHARGES.append({"what": str(what), "rows": int(n_rows)})


# The hand example (identical arithmetic to the pre-launch gate, now run through the real path).
P_SEND = 0.58
COST_C = 3.0
EXPECTED_ASCENT = -COST_C * P_SEND * (1.0 - P_SEND)          # -0.2436 * c
def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_journal(path) -> bytes:
    return path.read_bytes() if path.is_file() else b""


def _divert_appended(path, before: bytes, *, what: str) -> list:
    """Quarantine the lines a SYNTHETIC gate call appended to a REAL training journal.

    ``ppo_update_calls.jsonl`` / ``optimizer_step_calls.jsonl`` are rehydrated into the live
    call counters at configure time and must close exactly against committed batch records, so
    a gate call may never leave a line in them.  Every appended line is moved - byte for byte,
    with its sha256 - into an independent synthetic archive and the real journal is restored to
    its exact pre-call bytes.  No real line is ever read, rewritten or removed.
    """
    after = _read_journal(path)
    if not after.startswith(before):
        raise AssertionError(f"{path.name}: the journal was rewritten, not appended to")
    appended = after[len(before):]
    if not appended:
        return []
    lines = [ln for ln in appended.decode("utf-8").splitlines() if ln.strip()]
    archive = DIVERT_DIR / f"{path.name}.synthetic.jsonl"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("a", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(json.dumps({"diverted_from": path.name, "what": what,
                                 "sha256": hashlib.sha256(ln.encode("utf-8")).hexdigest(),
                                 "line": json.loads(ln)}, ensure_ascii=False) + "\n")
    with path.open("r+b") as fh:
        fh.truncate(len(before))
        fh.flush()
        import os as _os
        _os.fsync(fh.fileno())
    if _read_journal(path) != before:
        raise AssertionError(f"{path.name}: byte-exact restore failed after diverting gate lines")
    return lines


class _FakeCollector:
    def __init__(self, records):
        self.records = records


def _late_features() -> np.ndarray:
    """A legal 19-D input that routes to the LATE head (ready_step = 200 > 147)."""
    f = np.zeros((1, 19), dtype=np.float32)
    f[0, 0] = 0.0            # round(f[0]) = 0
    f[0, 18] = 0.2           # round(0.2*1000) = 200  ->  ready_step = 200 > 147
    f[0, 5] = 0.0            # not a first-emission gate row
    return f


def _early_features() -> np.ndarray:
    """A legal 19-D input that routes to the EARLY head and is a gate row (ready_step = 100)."""
    f = np.zeros((1, 19), dtype=np.float32)
    f[0, 0] = 0.0
    f[0, 18] = 0.1           # ready_step = 100 <= 147
    f[0, 5] = 1.0            # first emission -> the beta=1.00 gate prior applies
    return f


def _record(controller, features, action: int, *, value: float, cost_value: float):
    """One decision step, one free (legal) target, log_prob taken from the LIVE controller."""
    _charge_gate_forward(1, "gate record: actor+value forward of one 19-D row")
    with torch.no_grad():
        logits, _values = controller.forward(torch.as_tensor(features, dtype=torch.float32))
    log_prob = float(torch.log_softmax(logits, dim=-1)[0, int(action)])
    return {"step": 0, "features": features, "targets": [0],
            "actions": {0: int(action)}, "actionable": {0},
            "log_prob": log_prob, "value": float(value), "cost_value": float(cost_value)}


def _episode(records, *, lam: float, task_val: float, cost_air: float, idx: int = 0):
    return {"collector": _FakeCollector(records), "steps": 6,
            "cost_stream": {1: {"airtime_bytes": float(cost_air)}},
            "task_stream": {1: float(task_val), 5: float(task_val) * 0.5},
            "lambda": float(lam), "episode_index": int(idx),
            # the two fields the dual-lambda step reads; the scenario is the one whose cached
            # immediate reference the gap-4 dual update passes in
            "scenario_seed": 1800000, "on_time": 6}


def _snapshot(controller) -> dict:
    """Deep clone of the five-module controller state (module -> parameter -> tensor)."""
    return {module: {name: tensor.detach().clone() for name, tensor in params.items()}
            for module, params in controller.state_dict().items()}


def _restore(controller, snap: dict) -> None:
    controller.load_state_dict({module: {name: tensor.clone() for name, tensor in params.items()}
                                for module, params in snap.items()})


class TestR73QualificationGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SYNTH_ROOT.mkdir(parents=True, exist_ok=True)
        # The REAL CLI configuration path: licence gate -> runtime stand-up -> objective install.
        cls.cfg = execute.configure_r73_trial(RAW, trial.trial_gate(), trial.trial_run_identity(),
                                              model_read_approval=trial.MODEL_READ_APPROVAL)
        # Bind the LIVE modules from the configured runtime: these gates must exercise the same
        # objects the real CLI stands up, not a second import of the same names.
        global common, gates
        common = cls.cfg["common"]
        import importlib
        gates = importlib.import_module("r73_trial_gates")
        for _label, _mod in (("v2_common", common),
                             ("v2_trainer", importlib.import_module("v2_trainer"))):
            _path = str(Path(_mod.__file__).resolve())
            assert any(root in _path for root in _LIVE_ROOTS), (
                f"{_label} resolved to {_path}, not the live zhenghe runtime copy")
        cls.trainer = cls.cfg["trainer"]
        cls.run_identity = cls.cfg["r73_run_identity"]
        cls.controllers = {}
        cls.identities = {}
        for seed in SEEDS:
            paths = execute.init_paths(cls.cfg, seed)
            registered = json.loads(Path(paths["identity"]).read_text(encoding="utf-8"))
            controller, _state, _ident = ckpt.r73_load_init_read_only(paths["init"],
                                                                     registered=registered)
            cls.controllers[seed] = controller
            cls.identities[seed] = registered
        cls._saved_entropy = common.ENTROPY_BONUS
        cls._saved_clip = common.MAX_GRAD_NORM
        cls._saved_value_coeff = common.VALUE_COEFFICIENT
        cls._updates = []
        # the ONE persisted round ledgers these gates charge BEFORE anything runs
        cls._forward = gates.forward_ledger(RAW)
        cls._synth = gates.synthetic_guard(RAW)
        cls._synth_start = int(cls._synth.used())
        cls._forward_start = int(cls._forward.rows)
        # pre-flight hygiene: no real training has run yet, so the real journals must be empty.
        # A line left behind by an earlier gate run would break the configure-time invariant
        # that the call journals close against committed batch records.
        for _journal in (PPO_JOURNAL, STEP_JOURNAL):
            if _read_journal(_journal).strip():
                raise AssertionError(
                    f"{_journal.name} is non-empty before any real training: an earlier gate "
                    f"run left lines in a journal that must close against committed batches")

    @classmethod
    def tearDownClass(cls):
        common.ENTROPY_BONUS = cls._saved_entropy
        common.MAX_GRAD_NORM = cls._saved_clip
        common.VALUE_COEFFICIENT = cls._saved_value_coeff
        SYNTH_ROOT.mkdir(parents=True, exist_ok=True)
        divert_archive = DIVERT_DIR / "ppo_update_calls.jsonl.synthetic.jsonl"
        # A FRESH ledger object over the durable state file.  The instance opened in
        # setUpClass never advances: every charge builds its own ledger object, so reading
        # the held one reports a stale row count.
        _durable_forward = gates.forward_ledger(RAW)
        receipt = {
            "kind": "round73-qualification-gates-receipt/1",
            "note": (
                "ZERO real environment constructions, ZERO real training episodes, ZERO committed "
                "batch records.  Every live trainer.ppo_update call listed below is a SYNTHETIC "
                "gate call over hand-built episodes; the real training journals were restored "
                "byte-for-byte after each call, so the final ledger must count real updates from "
                "the committed batch records and never from these gate calls."),
            "live_update_count": len(cls._updates),
            "live_updates": cls._updates,
            "physical_optimizer_steps": sum(int(u["physical_optimizer_steps"])
                                            for u in cls._updates),
            "synthetic_optimizer_steps": cls._synth.snapshot(),
            "synthetic_step_guard_note": (
                "one charge = one synthetic PPO update call, charged BEFORE the call and never "
                "refunded; each such call internally performs EPOCHS_PER_BATCH x minibatches "
                "physical Adam steps, reported per call under live_updates"),
            "forward_rows": {"rows": int(_durable_forward.rows),
                             "calls": int(_durable_forward.calls),
                             "joint_rows": int(_durable_forward.joint_rows),
                             "cap": int(_durable_forward.cap),
                             "rows_at_suite_start": int(cls._forward_start),
                             "rows_charged_by_this_suite": sum(
                                 int(c["rows"]) for c in _GATE_FORWARD_CHARGES),
                             "note": ("gate trunk input rows only; ROUND73 runs no offline "
                                      "fit. Read from a FRESH ledger object over the durable "
                                      "state file, never from the instance the suite opened "
                                      "in setUpClass, because every charge constructs its "
                                      "own ledger object and the held instance would report "
                                      "a stale row count.")},
            "gate_forward_charges": _GATE_FORWARD_CHARGES,
            "diverted_real_journal_lines": {
                "archive": str(divert_archive),
                "archive_sha256": (_sha256_file(divert_archive)
                                   if divert_archive.is_file() else None),
                "ppo_lines": sum(int(u["diverted_ppo_journal_lines"]) for u in cls._updates),
                "optimizer_lines": sum(int(u["diverted_optimizer_journal_lines"])
                                       for u in cls._updates),
                "note": ("lines the routed update appended to the REAL journals during a gate "
                         "call, quarantined byte-for-byte; the real journals were restored")},
            "real_journals_after_suite": {
                "ppo_update_calls_lines": len([ln for ln in
                                               _read_journal(PPO_JOURNAL).decode("utf-8")
                                               .splitlines() if ln.strip()]),
                "optimizer_step_calls_lines": len([ln for ln in
                                                   _read_journal(STEP_JOURNAL).decode("utf-8")
                                                   .splitlines() if ln.strip()]),
            },
        }
        SYNTH_JOURNAL_RECEIPT.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")

    def setUp(self):
        common.ENTROPY_BONUS = self._saved_entropy
        common.MAX_GRAD_NORM = self._saved_clip
        common.VALUE_COEFFICIENT = self._saved_value_coeff

    def tearDown(self):
        common.ENTROPY_BONUS = self._saved_entropy
        common.MAX_GRAD_NORM = self._saved_clip
        common.VALUE_COEFFICIENT = self._saved_value_coeff

    # ------------------------------------------------------------------ helpers ----
    def _live_update(self, controller, optimizer, episodes, *, update_index, what):
        """Charge the guards, quarantine gate journal lines, then call the LIVE routed update.

        The call itself is ``trainer.ppo_update`` exactly as the real training loop calls it
        (the ROUND68 routed update: the R53 critic->trunk isolation around the unchanged
        frozen-early-path math).  Around it: the trunk-row ledger and the synthetic-step guard
        are charged BEFORE anything runs, the ROUND43 Adam.step journal hook is swapped for the
        pristine step so no gate line lands in the real optimizer journal, and every line the
        routed update appends to the real PPO journal is quarantined afterwards.
        """
        n_rows = sum(len(episode["collector"].records) for episode in episodes)
        # trunk input rows: each epoch forwards every row once (actor + value in one call) and
        # the cost head is evaluated per row as well, so 2 x rows x epochs is the charge.
        _charge_gate_forward(2 * n_rows * int(common.EPOCHS_PER_BATCH),
                             f"{what}: trunk rows forwarded by the update")
        self._synth.charge(1, f"{what}: one synthetic PPO update, charged before the call")
        ppo_before = _read_journal(PPO_JOURNAL)
        step_before = _read_journal(STEP_JOURNAL)
        live_adam_step = torch.optim.Adam.step
        torch.optim.Adam.step = _PRISTINE_ADAM_STEP
        # the charge is booked BEFORE the call and is never refunded, so the receipt entry
        # exists even if the call raises
        entry = {"what": what, "update_index": int(update_index)}
        self._updates.append(entry)
        try:
            report = self.trainer.ppo_update(controller, optimizer, episodes,
                                             update_index=update_index)
        finally:
            torch.optim.Adam.step = live_adam_step
            diverted_ppo = _divert_appended(PPO_JOURNAL, ppo_before, what=what)
            diverted_steps = _divert_appended(STEP_JOURNAL, step_before, what=what)
        entry.update({
            "samples": int(report.get("samples", 0)),
            "minibatches": int(report.get("minibatches", 0)),
            "physical_optimizer_steps": int(report.get("optimizer_steps", 0)),
            "diverted_ppo_journal_lines": len(diverted_ppo),
            "diverted_optimizer_journal_lines": len(diverted_steps)})
        return report, []

    def _fresh_optimizer(self, controller):
        return r68.build_optimizer(controller, common.LEARNING_RATE)
    # ----------------------------------------------------------- gap 1: real init ----
    def test_gap1_real_three_seed_controller_from_registered_init(self):
        for seed in SEEDS:
            controller = self.controllers[seed]
            registered = self.identities[seed]
            cal = registered["calibration"]

            summary = controller.parameter_summary()
            self.assertEqual(summary["parameter_object_count"], 12, f"seed {seed}")
            self.assertEqual(summary["parameter_count"], 5830, f"seed {seed}")
            self.assertEqual(sorted(summary["shapes"].keys()),
                             sorted(["trunk.0.weight", "trunk.0.bias", "trunk.2.weight",
                                     "trunk.2.bias", "early_head.weight", "early_head.bias",
                                     "late_head.weight", "late_head.bias",
                                     "value_task_head.weight", "value_task_head.bias",
                                     "value_cost_head.weight", "value_cost_head.bias"]),
                             f"seed {seed}")
            self.assertEqual(summary["parameter_sha256"], registered["init_parameter_sha256"],
                             f"seed {seed}")

            # the frozen mask is really established on the live object
            self.assertFalse(any(p.requires_grad for p in r66.frozen_parameters(controller)),
                             f"seed {seed}")
            self.assertTrue(all(p.requires_grad for p in r66.trainable_parameters(controller)),
                            f"seed {seed}")
            self.assertEqual(len(r66.trainable_parameters(controller)), 6, f"seed {seed}")

            # late_head is the REGISTERED POST-SCALE value: the one-time x0.10 is not repeated.
            # A second scale would move the digest away from the registered one.
            digests = ckpt._registered_module_sha(controller)
            self.assertEqual(digests["late_head"], cal["late_head_sha256"], f"seed {seed}")
            self.assertEqual(digests["trunk"], cal["trunk_sha256"], f"seed {seed}")
            self.assertEqual(digests["early_head"], cal["early_head_sha256"], f"seed {seed}")
            self.assertEqual(digests["value_task_head"], cal["value_task_head_sha256"],
                             f"seed {seed}")
            self.assertEqual(digests["value_cost_head"], cal["value_cost_head_sha256"],
                             f"seed {seed}")

            # fresh optimizer / lambda / RNG: an empty Adam and an untouched cursor
            optimizer = self._fresh_optimizer(controller)
            self.assertEqual(optimizer.state_dict()["state"], {}, f"seed {seed}")
            self.assertEqual([len(g["params"]) for g in optimizer.state_dict()["param_groups"]],
                             [6], f"seed {seed}")
            state = r68.fresh_training_state(int(seed), gae_lambda=common.GAE_LAMBDA,
                                             eta_lambda=common.ETA_LAMBDA, gamma=common.GAMMA,
                                             delta=0.0, lambda_init=1.0,
                                             data_order_seed=common.DATA_ORDER_SEED,
                                             b0_bytes=common.B0_BYTES,
                                             alpha=r68.LATE_INIT_ALPHA)
            self.assertEqual(state["lambda"], 1.0, f"seed {seed}")
            self.assertEqual(state["updates"], 0, f"seed {seed}")
            self.assertEqual(state["episodes_done"], 0, f"seed {seed}")
            self.assertEqual(state["optimizer_steps"], 0, f"seed {seed}")
            self.assertEqual(state["lambda_history"], [], f"seed {seed}")
            self.assertEqual(state["late_init_alpha"], r68.LATE_INIT_ALPHA, f"seed {seed}")
            # the live GAE lambda is 1.0, not the v2_common module default 0.95
            self.assertEqual(float(common.GAE_LAMBDA), 1.0, f"seed {seed}")
            self.assertEqual(float(common.ETA_LAMBDA), 0.5, f"seed {seed}")

    def test_gap1_initial_outputs_on_legal_inputs_match_a_direct_restore(self):
        """The controller's initial outputs on legal saved inputs equal a direct R71 restore's."""
        for seed in SEEDS:
            paths = execute.init_paths(self.cfg, seed)
            payload = torch.load(Path(paths["init"]), map_location="cpu", weights_only=False)
            direct = ckpt._restore_controller(payload, payload["training_state"])
            live = self.controllers[seed]
            for features in (_late_features(), _early_features()):
                _charge_gate_forward(2, f"gap1 seed {seed}: identity forward + cost head")
                batch = torch.as_tensor(features, dtype=torch.float32)
                with torch.no_grad():
                    l_logits, l_values = live.forward(batch)
                    d_logits, d_values = direct.forward(batch)
                    l_cost = live.cost_values(batch)
                    d_cost = direct.cost_values(batch)
                self.assertTrue(torch.equal(l_logits, d_logits), f"seed {seed} logits")
                self.assertTrue(torch.equal(l_values, d_values), f"seed {seed} values")
                self.assertTrue(torch.equal(l_cost, d_cost), f"seed {seed} cost values")
    # ------------------------------------------- gap 2: invariance through live path ----
    def test_gap2_actor_loss_and_preclip_actor_gradient_invariant_to_task_and_lambda(self):
        seed = 11
        controller = self.controllers[seed]
        features = _late_features()
        cost_value = 0.20
        cost_air = 40000.0                       # A_cost = 4.0 - 0.20 = 3.80, fixed below
        task_val = 1.0                           # return_task = 1.5, fixed below

        # The ONLY differences: the behaviour task-value estimate (hence A_task) and lambda.
        # A_cost, return_cost, return_task, features, actions, free set and log_prob are fixed.
        value_a, value_b = 0.5, 2.0
        lam_a, lam_b = 1.0, 3.75
        rec_a = _record(controller, features, r66.SEND_INDEX, value=value_a, cost_value=cost_value)
        rec_b = _record(controller, features, r66.SEND_INDEX, value=value_b, cost_value=cost_value)
        self.assertEqual(rec_a["log_prob"], rec_b["log_prob"])

        ep_a = _episode([rec_a], lam=lam_a, task_val=task_val, cost_air=cost_air)
        ep_b = _episode([rec_b], lam=lam_b, task_val=task_val, cost_air=cost_air)

        # A_task really is non-zero and really differs; A_cost really is identical.
        samples_a = self.trainer.batch_samples([ep_a])
        samples_b = self.trainer.batch_samples([ep_b])
        self.assertEqual(len(samples_a), 1)
        a_cost_a = samples_a[0]["return_cost"] - samples_a[0]["behaviour_value_cost"]
        a_cost_b = samples_b[0]["return_cost"] - samples_b[0]["behaviour_value_cost"]
        a_task_a = samples_a[0]["return_task"] - samples_a[0]["behaviour_value_task"]
        a_task_b = samples_b[0]["return_task"] - samples_b[0]["behaviour_value_task"]
        self.assertEqual(a_cost_a, a_cost_b)
        self.assertNotEqual(a_task_a, a_task_b)
        self.assertTrue(abs(a_task_a) > 1e-9 and abs(a_task_b) > 1e-9)
        self.assertNotEqual(lam_a, lam_b)
        self.assertEqual(samples_a[0]["advantage"], samples_b[0]["advantage"])
        # the CONTROL objective would have differed, so the override is what removed the term
        self.assertNotEqual(lam_a * a_task_a - a_cost_a, samples_a[0]["advantage"])

        snap = _snapshot(controller)
        try:
            report_a, _ = self._live_update(controller, self._fresh_optimizer(controller),
                                            [ep_a], update_index=0,
                                            what="gap2 task/lambda invariance, arm A")
            _restore(controller, snap)
            report_b, _ = self._live_update(controller, self._fresh_optimizer(controller),
                                            [ep_b], update_index=0,
                                            what="gap2 task/lambda invariance, arm B")
        finally:
            _restore(controller, snap)

        # --- the actor loss and the pre-clip actor gradient are EXACTLY invariant ---
        self.assertEqual(report_a["policy_surrogate"], report_b["policy_surrogate"])
        self.assertEqual(report_a["module_gradient_l1_mean"]["late_head"],
                         report_b["module_gradient_l1_mean"]["late_head"])
        self.assertEqual(report_a["module_gradient_l1_min"]["late_head"],
                         report_b["module_gradient_l1_min"]["late_head"])
        # the cost critic target/return did not move either
        self.assertEqual(report_a["module_gradient_l1_mean"]["cost_head"],
                         report_b["module_gradient_l1_mean"]["cost_head"])
        self.assertEqual(report_a["value_loss_cost"], report_b["value_loss_cost"])
        # entropy and the beta=1.00 gate prior depend on the logits only
        self.assertEqual(report_a["entropy"], report_b["entropy"])
        self.assertEqual(report_a["gate_loss"], report_b["gate_loss"])
        self.assertEqual(report_a["gate_rows"], report_b["gate_rows"])
        self.assertEqual(report_a["early_free"], report_b["early_free"])
        self.assertEqual(report_a["late_free"], report_b["late_free"])
        self.assertEqual(report_a["approx_kl"], report_b["approx_kl"])
        self.assertEqual(report_a["clip_fraction"], report_b["clip_fraction"])
        self.assertEqual(report_a["samples"], report_b["samples"])
        self.assertEqual(report_a["minibatches"], report_b["minibatches"])
        self.assertEqual(report_a["optimizer_steps"], report_b["optimizer_steps"])
        # the actor head really received a gradient (the invariance is not the trivial zero)
        self.assertGreater(report_a["module_gradient_l1_mean"]["late_head"], 0.0)
        # the task critic is NOT claimed invariant: return_task = (r - V) + V, so its last
        # bits move with the behaviour value V even though the exact value is r.
        self.assertLess(abs(report_a["value_loss_task"] - report_b["value_loss_task"]), 1e-9)
        self.assertLess(abs(report_a["gradient_norm_mean"] - report_b["gradient_norm_mean"]),
                        1e-6)
    # ------------------------------------------------ gap 3: hand example, real path ----
    def test_gap3_hand_cost_example_through_the_real_actor_path(self):
        seed = 11
        controller = self.controllers[seed]
        features = _late_features()

        # Make the LIVE late head emit exactly p(SEND) = 0.58: weight 0, bias carries the logits.
        z_send = float(np.log(P_SEND / (1.0 - P_SEND)))
        with torch.no_grad():
            controller.inner.late_head.weight.zero_()
            controller.inner.late_head.bias.copy_(torch.tensor([0.0, z_send],
                                                               dtype=torch.float32))
        with torch.no_grad():
            check = controller.forward(torch.as_tensor(features, dtype=torch.float32))[0]
        p_live = float(torch.softmax(check, dim=-1)[0, r66.SEND_INDEX])
        self.assertAlmostEqual(p_live, P_SEND, places=6)

        # The action-independent cost estimate V_cost = p*c, so
        #   A_cost(SEND) = c - p*c = c(1-p),   A_cost(SKIP) = 0 - p*c = -p*c
        v_cost = P_SEND * COST_C
        n_send, n_skip = 58, 42
        records = []
        for _ in range(n_send):
            # A_cost = c(1-p)  ->  bundled_cost/B0 - V = c(1-p)  ->  bundled_cost/B0 = c
            records.append(_record(controller, features, r66.SEND_INDEX,
                                   value=0.5, cost_value=v_cost))
        for _ in range(n_skip):
            # A_cost = -p*c  ->  bundled_cost/B0 = 0
            records.append(_record(controller, features, r66.SKIP_INDEX,
                                   value=0.5, cost_value=v_cost))
        # one episode per record keeps each sample's own cost stream; the batch is ALL_ROWS
        episodes = []
        for i, rec in enumerate(records):
            air = 10000.0 * COST_C if rec["actions"][0] == r66.SEND_INDEX else 0.0
            episodes.append(_episode([rec], lam=1.0, task_val=1.0, cost_air=air, idx=i))
        samples = self.trainer.batch_samples(episodes)
        self.assertEqual(len(samples), n_send + n_skip)
        for s in samples:
            taken_send = s["actions"][0] == r66.SEND_INDEX
            want = (COST_C * (1.0 - P_SEND)) if taken_send else (-P_SEND * COST_C)
            self.assertAlmostEqual(s["return_cost"] - s["behaviour_value_cost"], want, places=6)
            self.assertAlmostEqual(s["advantage"], -want, places=6)

        # Isolate the actor term: entropy bonus and gate prior are fixed at 0 for this reading
        # and are asserted separately below with their live values.
        common.ENTROPY_BONUS = 0.0
        snap = _snapshot(controller)
        captured = []
        handle = controller.inner.late_head.bias.register_hook(
            lambda g: captured.append(g.detach().clone()))
        try:
            report, _ = self._live_update(controller, self._fresh_optimizer(controller),
                                          episodes, update_index=0,
                                          what="gap3 hand cost example through the real path")
        finally:
            handle.remove()
            _restore(controller, snap)

        # old == new: the logged probability describes the draw through the SAME routed forward
        # (likelihood_alignment re-computes it and fails closed otherwise), so the FIRST
        # minibatch runs at ratio exactly 1 and nothing is clipped.
        alignment = report["likelihood_alignment"]
        self.assertTrue(alignment["passed"], alignment)
        self.assertLess(float(alignment["max_abs_gap"]), 1e-6)
        self.assertEqual(report["clip_fraction"], 0.0)
        # later epochs legitimately re-evaluate the ratio after the first optimizer step, so the
        # reported mean approx_kl is small but NOT exactly zero; it is bounded, not asserted 0.
        self.assertLess(report["approx_kl"], 1e-3)
        # the entropy BONUS is zero here, but the reported entropy is the policy's own action
        # entropy, averaged over the four epochs.  The FIRST epoch is exactly the binary entropy
        # of the hand-set p(SEND)=0.58, and the later epochs sit above it and below the maximum
        # ln(2), because the pure-cost ascent moves the SEND logit down toward the uniform policy.
        expected_entropy = float(-(P_SEND * np.log(P_SEND)
                                   + (1.0 - P_SEND) * np.log(1.0 - P_SEND)))
        self.assertGreater(report["entropy"], expected_entropy - 1e-4)
        self.assertLess(report["entropy"], float(np.log(2.0)))
        self.assertEqual(report["gate_rows"], 0)
        # The FIRST minibatch runs at ratio exactly 1 (proven by likelihood_alignment above), so
        # its surrogate IS the mean advantage, and the hand example is built so that the mean
        # advantage is zero by construction: 58 x c(1-p) against 42 x p*c.  The REPORTED value is
        # the mean over the four epochs, and the later epochs deviate slightly as the ratio moves
        # off 1 after the first optimizer step, so it is bounded rather than asserted to be zero.
        self.assertAlmostEqual(float(np.mean([s["advantage"] for s in samples])), 0.0, places=6)
        self.assertLess(abs(report["policy_surrogate"]), 0.05)

        # The pre-clip MINIMISED-loss gradient of the FIRST minibatch (ratio exactly 1) w.r.t.
        # the SEND logit is +0.2436c, i.e. the ASCENT gradient is -0.2436c, exactly the
        # hand-computed answer.  The first minibatch is the one whose ratio is provably 1.
        # update_index 0 is a gate-diagnostic update, and that diagnostic re-reads the pre-clip
        # gradient of the first minibatch twice through autograd.grad (base loss, then the late
        # head alone) before the real backward, so the hook fires EPOCHS_PER_BATCH + 2 times;
        # every reading of the first minibatch is the same gradient.
        self.assertEqual(len(captured), common.EPOCHS_PER_BATCH + 2)
        pre_clip = captured[0]
        self.assertTrue(torch.equal(captured[0], captured[1]))
        self.assertTrue(torch.equal(captured[0], captured[2]))
        self.assertAlmostEqual(float(pre_clip[r66.SEND_INDEX]), -EXPECTED_ASCENT, places=5)
        self.assertAlmostEqual(float(pre_clip[r66.SKIP_INDEX]), EXPECTED_ASCENT, places=5)
        self.assertAlmostEqual(float(pre_clip.sum()), 0.0, places=6)
        self.assertLess(EXPECTED_ASCENT, 0.0)            # ascent lowers the SEND logit
        self.assertGreater(-EXPECTED_ASCENT, 0.0)        # the minimised loss has the opposite sign
        # the reported pre-clip module L1 is the same measurement, read by the update itself
        self.assertGreater(report["module_gradient_l1_mean"]["late_head"], 0.0)

    def test_gap3_prior_and_entropy_are_fixed_and_separated(self):
        """With the LIVE entropy bonus and beta=1.00 gate prior, both are reported, non-trivial,
        and still invariant to A_task / lambda (they depend on the logits only)."""
        controller = self.controllers[11]
        snap = _snapshot(controller)
        try:
            late = _late_features()
            early = _early_features()
            recs = [_record(controller, late, r66.SEND_INDEX, value=0.5, cost_value=0.2),
                    _record(controller, early, r66.SKIP_INDEX, value=0.5, cost_value=0.2)]
            self.assertEqual(common.ENTROPY_BONUS, self._saved_entropy)
            self.assertGreater(common.ENTROPY_BONUS, 0.0)
            ep_a = _episode(recs, lam=1.0, task_val=1.0, cost_air=40000.0)
            recs_b = [_record(controller, late, r66.SEND_INDEX, value=2.0, cost_value=0.2),
                      _record(controller, early, r66.SKIP_INDEX, value=2.0, cost_value=0.2)]
            ep_b = _episode(recs_b, lam=4.25, task_val=1.0, cost_air=40000.0)
            report_a, _ = self._live_update(controller, self._fresh_optimizer(controller),
                                            [ep_a], update_index=0,
                                            what="gap3 prior/entropy separation, arm A")
            _restore(controller, snap)
            report_b, _ = self._live_update(controller, self._fresh_optimizer(controller),
                                            [ep_b], update_index=0,
                                            what="gap3 prior/entropy separation, arm B")
        finally:
            _restore(controller, snap)
        # the gate prior really fired on the early first-emission free row
        self.assertGreater(report_a["gate_rows"], 0)
        self.assertEqual(report_a["gate_rows"], report_b["gate_rows"])
        self.assertEqual(report_a["gate_loss"], report_b["gate_loss"])
        self.assertGreater(report_a["entropy"], 0.0)
        self.assertEqual(report_a["entropy"], report_b["entropy"])
        self.assertEqual(report_a["policy_surrogate"], report_b["policy_surrogate"])
        self.assertEqual(report_a["module_gradient_l1_mean"]["late_head"],
                         report_b["module_gradient_l1_mean"]["late_head"])
        diag = report_a["gate_diag"]
        self.assertIsNotNone(diag)
        self.assertEqual(diag["trunk"], "frozen_requires_grad_false")
        self.assertEqual(diag["early_head"], "frozen_requires_grad_false")
        self.assertEqual(diag["trainable_parameter_object_count"], 6)
        self.assertGreater(diag["late_actor_grad_l1"], 0.0)
        self.assertGreater(diag["critic_grad_l1"], 0.0)
        self.assertIsNotNone(diag["prior_late_actor_grad_l1"])
    # ---------------------------------------------- gap 4: real save / load / resume ----
    def _synthetic_batch(self, controller, *, lam=1.0):
        recs = [_record(controller, _late_features(), r66.SEND_INDEX, value=0.5, cost_value=0.2),
                _record(controller, _early_features(), r66.SKIP_INDEX, value=0.5, cost_value=0.2)]
        return [_episode(recs, lam=lam, task_val=1.0, cost_air=40000.0)]

    def _synthetic_state(self, seed):
        state = r68.fresh_training_state(int(seed), gae_lambda=common.GAE_LAMBDA,
                                         eta_lambda=common.ETA_LAMBDA, gamma=common.GAMMA,
                                         delta=0.0, lambda_init=1.0,
                                         data_order_seed=common.DATA_ORDER_SEED,
                                         b0_bytes=common.B0_BYTES,
                                         alpha=r68.LATE_INIT_ALPHA)
        state["training_version"] = ckpt.R73_TRAINING_VERSION
        state["loss_version"] = ckpt.R73_LOSS_VERSION
        state["state_schema"] = ckpt.R73_STATE_SCHEMA
        state["optimizer_schema"] = ckpt.R73_OPTIMIZER_SCHEMA
        state["structure"] = ckpt.R73_STRUCTURE
        state["actor_objective"] = ckpt.R73_ACTOR_OBJECTIVE
        state["objective_version"] = ckpt.R73_OBJECTIVE_VERSION
        state["run_identity"] = self.run_identity.as_dict()
        state["init_source"] = self.identities[seed]
        return state

    def test_gap4_full_resume_roundtrip_with_nonempty_adam(self):
        seed = 11
        controller = self.controllers[seed]
        snap = _snapshot(controller)
        state = self._synthetic_state(seed)

        optimizer = self._fresh_optimizer(controller)
        batch = self._synthetic_batch(controller)
        # the real load entry restores the trainer RNG, so the live RNG is snapshotted and put
        # back afterwards: a gate must never leave the live runtime in a different RNG state.
        rng_before = self.trainer.rng_state()
        try:
            report1, _ = self._live_update(controller, optimizer, batch, update_index=0,
                                           what="gap4 roundtrip first synthetic update")

            # a real dual-lambda step, so the resume must carry an ADVANCED lambda
            g_record = self.trainer.batch_g_hat(batch, {1800000: 1.0})
            dual = self.trainer.dual_update(state["lambda"], g_record["g_hat"])
            state["lambda"] = dual["lambda_next"]
            state["lambda_history"].append({"batch": 0, "g_hat": dual["g_hat"],
                                            "lambda_before": dual["lambda_before"],
                                            "lambda_next": dual["lambda_next"]})
            state["updates"] = 1
            state["episodes_done"] = 4
            state["optimizer_steps"] = int(report1.get("optimizer_steps", 0))

            # the Adam state is genuinely NON-EMPTY after a real step
            adam = optimizer.state_dict()["state"]
            self.assertEqual(len(adam), 6)
            for slot in adam.values():
                self.assertIn("exp_avg", slot)
                self.assertIn("exp_avg_sq", slot)
                self.assertGreater(float(slot["exp_avg"].abs().sum()), 0.0)
            self.assertNotEqual(state["lambda"], 1.0)

            out = SYNTH_ROOT / "roundtrip"
            out.mkdir(parents=True, exist_ok=True)
            path = out / "v2_train_s11_synthetic.pt"
            self.trainer.save_checkpoint(controller, optimizer, state, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(payload["carries_trained_optimizer_state"])
            self.assertTrue(payload["carries_trained_lambda"])
            self.assertTrue(payload["carries_advanced_training_rng"])
            self.assertEqual(len(payload["optimizer"]["state"]), 6)
            self.assertEqual([len(g["params"]) for g in payload["optimizer"]["param_groups"]], [6])

            # The continuation batch is re-collected from the UPDATED controller, exactly as the
            # real loop collects a fresh batch per update: the logged probabilities must describe
            # the current draw or the update refuses to run.
            batch2 = self._synthetic_batch(controller)

            # ---- the reference continuation, in memory ----
            ref_controller = copy.deepcopy(controller)
            ref_optimizer = self._fresh_optimizer(ref_controller)
            ref_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            report_ref, _ = self._live_update(ref_controller, ref_optimizer, batch2,
                                            update_index=1,
                                            what="gap4 in-memory reference continuation")
            ref_summary = ref_controller.parameter_summary()

            # ---- the resumed continuation, through the real load entry ----
            r_controller, r_optimizer, r_state = ckpt.r73_load_checkpoint(
                path, run_identity=self.run_identity)
            self.assertEqual(r_state["updates"], 1)
            self.assertEqual(r_state["episodes_done"], 4)
            self.assertEqual(r_state["lambda"], state["lambda"])
            self.assertEqual(r_state["optimizer_steps"], state["optimizer_steps"])
            self.assertEqual(len(r_optimizer.state_dict()["state"]), 6)
            self.assertEqual(r_controller.parameter_summary()["parameter_sha256"],
                             controller.parameter_summary()["parameter_sha256"])
            # the restored Adam momentum equals the saved one, tensor for tensor
            for slot, ref_slot in zip(r_optimizer.state_dict()["state"].values(),
                                      optimizer.state_dict()["state"].values()):
                self.assertTrue(torch.equal(slot["exp_avg"], ref_slot["exp_avg"]))
                self.assertTrue(torch.equal(slot["exp_avg_sq"], ref_slot["exp_avg_sq"]))
                self.assertEqual(slot["step"], ref_slot["step"])

            report_res, _ = self._live_update(r_controller, r_optimizer, batch2,
                                            update_index=1,
                                            what="gap4 resumed continuation through the load entry")
            res_summary = r_controller.parameter_summary()
            self.assertEqual(report_ref["policy_surrogate"], report_res["policy_surrogate"])
            self.assertEqual(report_ref["module_gradient_l1_mean"]["late_head"],
                             report_res["module_gradient_l1_mean"]["late_head"])
            self.assertEqual(report_ref["gradient_norm_mean"], report_res["gradient_norm_mean"])
            self.assertEqual(ref_summary["parameter_sha256"], res_summary["parameter_sha256"])
        finally:
            _restore(controller, snap)
            self.trainer.restore_rng(rng_before)

    def test_gap4_real_load_entry_refuses_foreign_and_empty_adam_resumes(self):
        seed = 11
        paths = execute.init_paths(self.cfg, seed)
        out = SYNTH_ROOT / "refusals"
        out.mkdir(parents=True, exist_ok=True)

        # (a) the ROUND71 calibrated INIT is never a NEW training-resume source
        with self.assertRaises(RuntimeError):
            ckpt.r73_load_checkpoint(paths["init"], run_identity=self.run_identity)

        # (b) a ROUND68 payload is never a NEW training-resume source
        r68_src = None
        r68_dir = (REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
                   / "round68_late_initial_confidence_ablation" / "raw" / "checkpoints")
        for candidate in sorted(r68_dir.glob("*.pt")):
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            if payload.get("trainer_tag") == r68.R68_TRAINER_TAG:
                r68_src = candidate
                break
        self.assertIsNotNone(r68_src, "no ROUND68 checkpoint was found to refuse")
        with self.assertRaises(RuntimeError):
            ckpt.r73_load_checkpoint(r68_src, run_identity=self.run_identity)

        # (c) a ROUND73 payload whose Adam state is EMPTY is refused (no empty-Adam substitute)
        controller = self.controllers[seed]
        state = self._synthetic_state(seed)
        empty_path = out / "v2_train_s11_empty_adam.pt"
        self.trainer.save_checkpoint(controller, self._fresh_optimizer(controller), state,
                                     empty_path)
        payload = torch.load(empty_path, map_location="cpu", weights_only=False)
        self.assertFalse(payload["carries_trained_optimizer_state"])
        with self.assertRaises(RuntimeError):
            ckpt.r73_load_checkpoint(empty_path, run_identity=self.run_identity)

        # (d) a payload whose run-identity block differs is refused field for field
        tampered = dict(payload)
        tampered_state = dict(payload["training_state"])
        tampered_state["run_identity"] = dict(self.run_identity.as_dict(), run_id="somebody-else")
        tampered["training_state"] = tampered_state
        tampered_path = out / "v2_train_s11_foreign_identity.pt"
        torch.save(tampered, tampered_path)
        with self.assertRaises(RuntimeError):
            ckpt.r73_load_checkpoint(tampered_path, run_identity=self.run_identity)
    # ------------------------------------------- gap 5: accounting before execution ----
    def test_gap5_synthetic_steps_charged_before_execution_and_blocked_at_cap(self):
        path = SYNTH_ROOT / "cap_probe" / "steps.json"
        if path.is_file():
            path.unlink()
        guard = r68_guards.SyntheticStepGuard(path, cap=8)
        self.assertEqual(guard.used(), 0)
        self.assertEqual(guard.remaining(), 8)
        for i in range(1, 9):
            self.assertEqual(guard.charge(1, f"synthetic step {i}"), i)
        self.assertEqual(guard.remaining(), 0)
        with self.assertRaises(RuntimeError):
            guard.charge(1, "the 9th step must be refused")
        # a refused charge does not consume and does not refund
        self.assertEqual(guard.used(), 8)
        with self.assertRaises(RuntimeError):
            guard.charge(2, "a batch over the cap must be refused")
        self.assertEqual(guard.used(), 8)
        # a NEW instance reads the DURABLE state, not a fresh counter
        self.assertEqual(r68_guards.SyntheticStepGuard(path, cap=8).used(), 8)

    def test_gap5_concurrent_contention_admits_exactly_one(self):
        path = SYNTH_ROOT / "contention" / "steps.json"
        if path.is_file():
            path.unlink()
        guard = r68_guards.SyntheticStepGuard(path, cap=1)
        results = []
        lock = threading.Lock()

        def _try():
            g = r68_guards.SyntheticStepGuard(path, cap=1)
            try:
                g.charge(1, "last slot")
                with lock:
                    results.append("ok")
            except RuntimeError:
                with lock:
                    results.append("refused")

        threads = [threading.Thread(target=_try) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count("ok"), 1, results)
        self.assertEqual(results.count("refused"), 5, results)
        self.assertEqual(guard.used(), 1)

    def test_gap5_offline_trunk_rows_debited_before_the_forward(self):
        from r71_critic_calibration_seam import ForwardLedger
        path = SYNTH_ROOT / "cap_probe" / "forward_rows.json"
        if path.is_file():
            path.unlink()
        ledger = ForwardLedger(cap=4, state_path=path)
        self.assertEqual(ledger.rows, 0)
        ledger.debit(3)
        self.assertEqual(ledger.rows, 3)
        self.assertEqual(ledger.remaining(), 1)
        with self.assertRaises(RuntimeError):
            ledger.debit(2)
        self.assertEqual(ledger.rows, 3)
        ledger.debit(1)
        self.assertEqual(ledger.rows, 4)
        with self.assertRaises(RuntimeError):
            ledger.debit(1)
        self.assertEqual(ledger.rows, 4)
        # a NEW instance reads the DURABLE state, not a fresh counter
        self.assertEqual(ForwardLedger(cap=4, state_path=path).rows, 4)

    def test_gap5_real_ledgers_are_read_from_persisted_state(self):
        """The real raw root's ledgers report their ACTUAL persisted numbers, never a hard 0."""
        forward = gates.forward_ledger(RAW)
        synth = gates.synthetic_guard(RAW)
        # the synthetic guard carries EXACTLY the charges this suite made - no more, no less -
        # and a brand-new instance reads the same durable number instead of a fresh counter.
        self.assertEqual(synth.used(), self._synth_start + len(self._updates))
        self.assertLessEqual(synth.used(), synth.cap)
        self.assertEqual(gates.synthetic_guard(RAW).used(), synth.used())
        # the trunk-row ledger carries exactly the gate rows debited here, read from the file.
        # ROUND73 runs no offline fit, so every row in it is a gate row.
        expected_rows = self._forward_start + sum(int(c["rows"]) for c in _GATE_FORWARD_CHARGES)
        # every booked charge has a receipt entry (a failed call still consumed its charge)
        self.assertTrue(all("what" in u for u in self._updates))
        self.assertEqual(forward.rows, expected_rows)
        self.assertLessEqual(forward.rows, forward.cap)
        self.assertEqual(gates.forward_ledger(RAW).rows, forward.rows)
        self.assertGreater(len(_GATE_FORWARD_CHARGES), 0)
        # the REAL training journals are empty again: every gate line was quarantined.
        self.assertEqual(_read_journal(PPO_JOURNAL).strip(), b"")
        self.assertEqual(_read_journal(STEP_JOURNAL).strip(), b"")
        self.assertTrue(gates.gate_paths(RAW)["forward_ledger"].parent.is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)