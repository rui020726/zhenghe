"""ROUND76 R71-INIT native-mainline wiring (trunk frozen; early actually trainable).

PREPARATION-ONLY module for the human-approved R76 A wiring.  A0 executes none of its real paths:
no torch.load, no model construction, no forward/backward, no optimizer, no PPO, and no env.

The update function ``r75_native_ppo_update`` is a faithful full-function adaptation of the REAL
``r66_frozen.frozen_early_ppo_update`` (round66_bc_early_path_preserving_late_adaptation/r66_frozen.py,
sha256 A22425729C9327FFCB1D57F4820F3B38F2BB7E35A6D6D81890FE64A75A246B10). Preserved verbatim:
``trainer.batch_samples`` + GAE, ``likelihood_alignment``, the ``DATA_ORDER_SEED``-seeded shuffle,
``EPOCHS_PER_BATCH`` x ``MINIBATCH_STEPS`` loop, the beta=1.00 / ready_cut=147 gate prior, the
clipped surrogate + dual-critic + entropy + gate loss, the global clip, the optimizer protocol and
the return contract.

THE ONLY structural change vs R66: the freeze set. R66 froze ("trunk", "early_head"); R75 freezes
("trunk",) only, so the trainable / optimizer / clip parameter set grows from 6 to 8 objects
(early_head + late_head + value_task_head + value_cost_head = 390 scalars; trunk stays 4 objects /
5440 scalars; 12 objects / 5830 scalars total). No loss term, weight, coefficient, reduction,
advantage combination, mask or protocol was added, removed or re-derived.

R76 hard-wires its own resource seam: every real NN/Adam path REQUIRES the unique ROUND76 central
production contract (``resources``), verified by ACTUAL class identity, production mode, canonical
root and the pinned byte identity of r76_guard.py (a forged boolean is not enough; a fixture is a
separate non-production interface and is refused). A0 leaves all new caps at zero. Each physical
optimizer step is
charged BEFORE ``optimizer.step``; every real trunk forward is metered at the REAL
``controller.inner.trunk.forward`` entry (actual input shape, pre-charge, attempted / charged /
completed reported separately). Failures do not refund.

Provenance caveats carried with this module:
- R58 rollout data is NOT an R71 trajectory. It is used here only as a real recorded batch for
  qualification, replayed through a gate-only controller whose behaviour actor is the R58
  single-head model copied into the two heads. Whether the recorded log-probs are reproduced is
  decided by the paid native ``likelihood_alignment``; if it fails, the run STOPS (the data is
  never rewritten and the alignment is never disabled).
- The gate-only helper inherited from R75 remains qualification-only and is never the R76 research
  INIT. The R76 INIT must come from the R71 calibrated five-module checkpoint by the model-only
  adapter in r76_runtime.py.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent                                   # .../comm_bc_initialized_constrained_ppo_20261007_r2
REPO = RUN_ROOT.parents[1]                               # .../zhenghe-round42
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
for _d in (str(R66_DIR),):
    if str(_d) not in sys.path:
        sys.path.insert(0, _d)

import r66_frozen as r66                                  # noqa: E402  real frozen-early-path module

# The frozen canonical sources must reach sys.path through the EXISTING R45/R58 wiring BEFORE any
# canonical module is imported or used. r66_frozen.configure_source_imports() is that wiring: it
# puts the canonical directories on sys.path, imports the canonical modules, verifies their byte
# identity against the pinned values and applies the R58 living GAE/ETA overrides. It therefore has
# to run before the canonical imports below, and the modules it returns are the bindings this
# module must use; importing by name afterwards would only re-read sys.modules, so the identity is
# asserted by _verify_canonical_bindings() rather than assumed.
CANONICAL_SOURCES = r66.configure_source_imports()

import v2_trainer as trainer                              # noqa: E402  frozen canonical trainer
import v2_common as common                                # noqa: E402  frozen canonical constants
from onpolicy import action_entropy, joint_log_prob       # noqa: E402  frozen canonical math
import onpolicy                                           # noqa: E402  frozen canonical math module
# The bindings this module actually uses, bound explicitly so the check below compares real
# objects. The controller module is reached only through the helper return, so it is bound once
# here and then compared with what the import system cached under its canonical name.
CONTROLLER_MODULE = CANONICAL_SOURCES.get("controller_module")
# cost_math was imported by configure_source_imports(); reading it back from sys.modules keeps this
# module from importing a second copy out of a different tree.
COST_MATH_MODULE = sys.modules.get("cost_math")
if str(HERE) not in sys.path:                    # the hardened central production ledger lives here
    sys.path.insert(0, str(HERE))
import r76_guard                                  # noqa: E402  R76 resources (stdlib-only reuse)

# ------------------------------------------------------------------ frozen identity ------
# The exact frozen canonical module byte identity accepted by ROUND58 and inherited since.
EXPECTED_TRAINER_SHA256 = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
EXPECTED_COMMON_SHA256 = "66d019a344d9cfef3c67bd8b722f8cc27144b9f71ddc8e5496451929d12f746a"
EXPECTED_R66_SOURCE_SHA256 = "a22425729c9327ffcb1d57f4820f3b38f2bb7e35a6d6d81890fe64a75a246b10"
EXPECTED_CONTROLLER_SHA256 = "9b2362ee5dc436c04e5fd2d841646d637b5b3f371607bd13a759a60c4cb4cab1"
# onpolicy.py does not live in the canonical run directory. It is reached through the same wiring,
# from the tight-budget round that v2_common.enter_round() puts on sys.path. This digest was READ
# from that live file (9830 bytes); it is not a guessed hash and no replacement copy is introduced.
EXPECTED_ONPOLICY_SHA256 = "49a92f6694ba21dd2d0e42950c8b7f91c118b81971d4108a18579eb5b38f5ab3"
# cost_math.py is imported by configure_source_imports() from the stage-A directory that
# v2_common.enter_round() puts on sys.path. This digest was READ from that live file (17384
# bytes); it is not a guessed hash and no replacement copy is introduced. The checkout copy under
# zhenghe-round42 carries the SAME digest, so the resolved path is what tells the two apart.
EXPECTED_COST_MATH_SHA256 = "bca99af08479398a7aec2a2f085163d31edd532f109b8cfc2043c1bfcfb49026"


def _sha256_of(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


CANONICAL_IMPORT_NAMES = {"trainer": "v2_trainer", "common": "v2_common",
                          "controller": "v2_controller"}


def _verify_canonical_bindings() -> dict[str, Any]:
    """Fail closed unless the canonical modules really are the pinned ones from the canonical tree.

    Wiring the directories onto sys.path does not by itself prove WHICH file was imported, and
    comparing a dictionary entry with itself proves nothing at all. Each binding is therefore
    checked against the object this module actually uses, three ways: the module-level name bound
    by the canonical imports must BE the object configure_source_imports() returned, so a
    same-path, same-digest but different object cannot ride in; sys.modules under the canonical
    import name must hold that same object, so a stale or shadowing cache entry is caught; and the
    file that was actually read must sit under the canonical root and hash to the pinned value.
    The R58 living GAE/ETA/GAMMA values are read back from the configured module. onpolicy and
    cost_math are checked the same way, cost_math being read back from sys.modules (the helper
    imported it) and never imported again here.
    """
    problems: list[str] = []
    canon = Path(r66.CANON).resolve()
    expected = {"trainer": ("trainer", "v2_trainer.py", EXPECTED_TRAINER_SHA256),
                "common": ("common", "v2_common.py", EXPECTED_COMMON_SHA256),
                "controller": ("controller_module", "v2_controller.py", EXPECTED_CONTROLLER_SHA256)}
    used = {"trainer": trainer, "common": common, "controller": CONTROLLER_MODULE}
    checked: dict[str, Any] = {}
    for label, (key, filename, digest) in expected.items():
        returned = CANONICAL_SOURCES.get(key)
        import_name = CANONICAL_IMPORT_NAMES[label]
        if returned is None:
            problems.append(f"configure_source_imports() returned no {label}")
            continue
        bound = used[label]
        if bound is None:
            problems.append(f"this module holds no {label} binding to check")
            continue
        if bound is not returned:
            problems.append(f"the {label} binding used here is not the object "
                            f"configure_source_imports() returned; a different file may be bound")
        cached = sys.modules.get(import_name)
        if cached is not returned:
            problems.append(f"sys.modules[{import_name!r}] is not the object "
                            f"configure_source_imports() returned (stale or shadowing entry)")
        path = Path(getattr(bound, "__file__", "") or "").resolve()
        if path != canon / filename:
            problems.append(f"{label}.__file__ resolved to {path}, expected {canon / filename}")
            checked[label] = {"used_is_helper_return": bound is returned}
            continue
        actual = _sha256_of(path)
        if actual != digest:
            problems.append(f"{label} byte identity is {actual[:16]}, expected {digest[:16]}")
        checked[label] = {"path": str(path), "sha256": actual,
                          "used_is_helper_return": bound is returned,
                          "sys_modules_is_helper_return": cached is returned}
    tight = Path(getattr(common, "TIGHT_ROUND", "") or "").resolve()
    onpolicy_path = Path(getattr(onpolicy, "__file__", "") or "").resolve()
    if onpolicy_path != tight / "onpolicy.py":
        problems.append(f"onpolicy.__file__ resolved to {onpolicy_path}, "
                        f"expected {tight / 'onpolicy.py'}")
    else:
        actual = _sha256_of(onpolicy_path)
        if actual != EXPECTED_ONPOLICY_SHA256:
            problems.append(f"onpolicy byte identity is {actual[:16]}, "
                            f"expected {EXPECTED_ONPOLICY_SHA256[:16]}")
    if sys.modules.get("onpolicy") is not onpolicy:
        problems.append("sys.modules['onpolicy'] is not the onpolicy module bound here")
    for helper_name, helper_object in (("action_entropy", action_entropy),
                                       ("joint_log_prob", joint_log_prob)):
        if getattr(onpolicy, helper_name, None) is not helper_object:
            problems.append(f"onpolicy.{helper_name} is not the object bound here")
        elif not callable(helper_object):
            problems.append(f"onpolicy.{helper_name} is not callable")
    if COST_MATH_MODULE is None:
        problems.append("cost_math is absent from sys.modules; configure_source_imports() must "
                        "have imported it and this module never imports it")
    else:
        stage_a = Path(getattr(common, "STAGE_A_DIR", "") or "")
        cost_path = Path(getattr(COST_MATH_MODULE, "__file__", "") or "").resolve()
        wanted = (stage_a / "cost_math.py").resolve() if str(stage_a) else None
        if wanted is None or cost_path != wanted:
            problems.append(f"cost_math.__file__ resolved to {cost_path}, expected {wanted}")
        else:
            actual = _sha256_of(cost_path)
            if actual != EXPECTED_COST_MATH_SHA256:
                problems.append(f"cost_math byte identity is {actual[:16]}, "
                                f"expected {EXPECTED_COST_MATH_SHA256[:16]}")
            checked["cost_math"] = {"path": str(cost_path), "sha256": actual}
    for attribute, wanted_value in (("GAE_LAMBDA", 1.0), ("ETA_LAMBDA", 0.5), ("GAMMA", 1.0)):
        value = getattr(common, attribute, None)
        if value is None or float(value) != float(wanted_value):
            problems.append(f"common.{attribute} is {value!r}, expected {wanted_value!r} (the "
                            f"living override must come from the configured module, not the "
                            f"default)")
    if problems:
        raise RuntimeError("R75 canonical binding refused: " + "; ".join(problems))
    return {"canonical_root": str(canon), "onpolicy_root": str(tight),
            "trainer": str(Path(trainer.__file__).resolve()),
            "common": str(Path(common.__file__).resolve()),
            "controller": str(Path(CONTROLLER_MODULE.__file__).resolve()),
            "onpolicy": str(onpolicy_path),
            "trainer_sha256": _sha256_of(Path(trainer.__file__).resolve()),
            "common_sha256": _sha256_of(Path(common.__file__).resolve()),
            "controller_sha256": _sha256_of(Path(CONTROLLER_MODULE.__file__).resolve()),
            "onpolicy_sha256": _sha256_of(onpolicy_path),
            "cost_math_sha256": _sha256_of(Path(COST_MATH_MODULE.__file__).resolve()),
            "gae_lambda": float(common.GAE_LAMBDA), "eta_lambda": float(common.ETA_LAMBDA),
            "gamma": float(common.GAMMA), "bindings": checked,
            "note": ("checked at import time by object identity against the helper return and "
                     "against sys.modules, by resolved __file__ under the canonical root and by "
                     "the pinned byte identity of the file actually read")}
CANONICAL_BINDINGS = _verify_canonical_bindings()
R76_TRAINER_TAG = "constrained-cost-v2-r76-native-early-mainline"
R76_TRAINING_VERSION = "round76-constrained-early-native-mainline/1"
R76_STRUCTURE = ("partitioned-early-late-actor-head/1+r76-native-early-mainline-trunk-frozen/1")
R76_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1+r76-native-early-mainline/1"
R76_STATE_SCHEMA = ("constrained-cost-state/2+partitioned-actor-head/1+r76-native-early-mainline/1")
R76_OPTIMIZER_SCHEMA = "stage-c-native-early-mainline-eight-object-parameter-set/1"
R75_TRAINER_TAG = R76_TRAINER_TAG
R75_TRAINING_VERSION = R76_TRAINING_VERSION
R75_STRUCTURE = R76_STRUCTURE
R75_LOSS_VERSION = R76_LOSS_VERSION
R75_STATE_SCHEMA = R76_STATE_SCHEMA
R75_OPTIMIZER_SCHEMA = R76_OPTIMIZER_SCHEMA

# The byte identity of the hardened central production ledger the native path demands. Any edit
# to r75_guard.py fails the native entry closed until this pin is re-registered through an
# explicit acceptance revision (a weakened guard cannot ride in on a renamed object).
EXPECTED_R76_GUARD_SOURCE_SHA256 = "61b840977f6299b5435ad8406013bf67ca767bb3393e5dec2246e0e46887c39a"

# THE single structural change vs R66/R71: early_head moves out of the frozen set.
FROZEN_MODULES = ("trunk",)
TRAINABLE_MODULES = ("early_head", "late_head", "value_task_head", "value_cost_head")
MODULES = ("trunk", "early_head", "late_head", "value_task_head", "value_cost_head")

READY_CUT = 147.0
HORIZON = 1000.0
SKIP_INDEX = 0
GATE_BETA = 1.00


def _module_of(controller, name: str):
    return {"trunk": controller.inner.trunk,
            "early_head": controller.inner.early_head,
            "late_head": controller.inner.late_head,
            "value_task_head": controller.inner.value_head,
            "value_cost_head": controller.cost_head}[name]


def module_sha(controller, name: str) -> str:
    """Hash a module's parameter values under stable parameter names, not object order."""
    digest = hashlib.sha256()
    for parameter_name, parameter in sorted(
            _module_of(controller, name).named_parameters(), key=lambda item: item[0]):
        digest.update(parameter_name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def partition(controller) -> tuple[list, list]:
    """Independently enumerate (frozen, trainable) parameter objects; object identity only."""
    frozen: list[Any] = []
    for name in FROZEN_MODULES:
        frozen.extend(_module_of(controller, name).parameters())
    trainable: list[Any] = []
    for name in TRAINABLE_MODULES:
        trainable.extend(_module_of(controller, name).parameters())
    frozen_ids = [id(p) for p in frozen]
    trainable_ids = [id(p) for p in trainable]
    if len(frozen_ids) != len(set(frozen_ids)):
        raise RuntimeError("R75 frozen enumeration contains a duplicate parameter object")
    if len(trainable_ids) != len(set(trainable_ids)):
        raise RuntimeError("R75 trainable enumeration contains a duplicate parameter object")
    if set(frozen_ids) & set(trainable_ids):
        raise RuntimeError("R75 frozen and trainable parameter sets overlap")
    all_ids = [id(p) for p in controller.parameters]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("R75 controller shares a parameter object across modules")
    if len(frozen) + len(trainable) != len(all_ids):
        raise RuntimeError("R75 frozen + trainable does not partition the parameter objects")
    return frozen, trainable


def assert_mask(controller) -> dict[str, Any]:
    """Assert the trunk-only freeze; return the mask facts (8 trainable / 4 frozen objects)."""
    frozen, trainable = partition(controller)
    if any(p.requires_grad for p in frozen):
        raise RuntimeError("R75 trunk is NOT frozen: trunk still requires grad")
    if any(not p.requires_grad for p in trainable):
        raise RuntimeError("R75 early_head/late_head/a critic is frozen but must stay trainable")
    return {"frozen_parameter_object_count": len(frozen),
            "trainable_parameter_object_count": len(trainable),
            "frozen_scalar_count": int(sum(p.numel() for p in frozen)),
            "trainable_scalar_count": int(sum(p.numel() for p in trainable)),
            "frozen_modules": list(FROZEN_MODULES),
            "trainable_modules": list(TRAINABLE_MODULES)}


def freeze_trunk_only(controller) -> dict[str, Any]:
    """Explicitly trainable-ize the four modules, then freeze the trunk only.

    r66.fresh_partitioned_controller performs NO requires_grad assignment (statically verified:
    neither the canonical StageCController chain nor _PartitionedInner sets any flag), so every
    module relies on the nn.Module default requires_grad=True. R75 no longer RELIES on that
    default: the four trainable modules are set True EXPLICITLY before the trunk is set False,
    and assert_mask verifies both directions afterwards.
    """
    frozen, trainable = partition(controller)
    for parameter in trainable:
        parameter.requires_grad_(True)
    for parameter in frozen:
        parameter.requires_grad_(False)
    mask = assert_mask(controller)
    digest = hashlib.sha256()
    for index, tensor in enumerate(frozen):
        digest.update(f"{index}".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {**mask, "frozen_sha256": digest.hexdigest(),
            "frozen_scalar_count": int(sum(t.numel() for t in frozen))}


def trainable_parameters(controller) -> list:
    """The EIGHT trainable parameter objects: early+late heads and both critics. Mask asserted."""
    assert_mask(controller)
    _frozen, trainable = partition(controller)
    return trainable


def frozen_parameters(controller) -> list:
    """The FOUR frozen parameter objects: the shared trunk. Mask asserted."""
    assert_mask(controller)
    frozen, _trainable = partition(controller)
    return frozen


def build_optimizer(controller, lr: float):
    """The R75 optimizer: Adam over the eight trainable objects only."""
    return torch.optim.Adam(trainable_parameters(controller), lr=float(lr))


def _require_production_ledger(resources) -> None:
    """Fail closed: real NN/Adam may only attach to the ROUND76 central PRODUCTION ledger.

    Delegates to the strict native-entry check in r76_guard: the ACTUAL class identity (a
    duck-typed object carrying a forgeable ``r75_central_production`` boolean is refused), the
    production mode (a fixture is the separate non-production spy interface and is refused),
    the canonical durable root, and the pinned byte identity of r76_guard.py itself.
    """
    r76_guard.assert_native_production_resources(
        resources, expected_guard_source_sha256=EXPECTED_R76_GUARD_SOURCE_SHA256)


# ------------------------------------------------------------------ gate-only init ------
def build_gate_only_controller(seed: int) -> tuple:
    """QUALIFICATION-ONLY controller. NOT CALLED IN S0 (torch.load is a paid, licensed step).

    Uses the evidenced R58 single-head -> two-head conversion already implemented and verified in
    ``r66_frozen``: the four OLD modules (trunk, action_head, value_task_head, value_cost_head)
    are read from the byte-verified R58 ep0 BC_POST checkpoint, and BOTH independent heads are set
    to the OLD action head. Because the two heads are then value-identical, ``route_logits``
    (early<=147 / late>147) selects the same logits the single head produced, so the recorded
    behaviour log-probs are reproducible by construction (the gate prior is a training-loss term
    only and never enters the forward, exactly as the R71 alignment gap 2.3e-13 attests).

    These parameters are for qualification ONLY: they never enter the R71 research INIT and never
    become a research training checkpoint.
    """
    state = r66.load_r58_ep0_model(int(seed))            # byte-verified, model-only
    controller = r66.fresh_partitioned_controller(int(seed), ep0_state=state)
    frozen_identity = freeze_trunk_only(controller)
    summary = controller.parameter_summary()
    return controller, state, frozen_identity, summary


# ------------------------------------------------------------------ native PPO update ---
_METER_KEYS = ("calls_attempted", "calls_charged", "rows_charged", "calls_completed",
               "rows_completed", "calls_refused", "calls_failed")


def _meter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Per-phase trunk-entry meter delta (attempted / charged / completed stay separate)."""
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in _METER_KEYS}


def r75_native_ppo_update(controller, optimizer, episodes: list[dict[str, Any]], *,
                          update_index: int, resources,
                          diag_updates: tuple[int, ...] = (0, 15, 31),
                          gate_beta: float = GATE_BETA,
                          physical_step_category: str = "normal",
                          physical_step_registration_prefix: str | None = None) -> dict[str, Any]:
    """One native PPO update over a real recorded batch; trunk frozen; 8 trainable objects.

    Faithful adaptation of ``r66_frozen.frozen_early_ppo_update``. Everything except the freeze
    set, the parameter ownership (6 -> 8 objects) and the R75 resource seam is the real function.

    Resource instrumentation (S0 acceptance revision): the strict native-entry identity check;
    ONE joint record per native batch read (the SAME unique recorded records are read again - a
    repeated exposure, never a new environment episode); the unified trunk-entry meter at the
    REAL controller.inner.trunk.forward (actual input shape, pre-charge at the entry, attempted
    / charged / completed reported separately). No caller-side aggregate pre-charge exists.
    """
    _require_production_ledger(resources)
    if physical_step_category not in ("normal", "technical"):
        raise ValueError("physical_step_category must be 'normal' or 'technical'")
    if (physical_step_category == "technical") != (physical_step_registration_prefix is not None):
        raise ValueError("technical physical steps require a registration prefix; normal steps "
                         "must not carry one")
    frozen_parameters(controller)
    trainable_parameters(controller)

    # One native batch READ over the recorded episode, pre-charged as ONE joint record: the
    # SAME unique input records are read again (a repeated exposure, never a new environment
    # episode; the environment guard stays 0/0/0).
    resources.charge_joint_record(1, what=f"r75_native_update_{update_index}_batch_read")
    unique_input_records = 0
    for episode in episodes:
        collector = episode.get("collector")
        records = getattr(collector, "records", None) if collector is not None else None
        if records is None:
            raise RuntimeError(
                "R75 native update: an episode without collector.records cannot be accounted; "
                "the unique input record count is required for the joint-record accounting")
        unique_input_records += len(records)
    batch_read = {"update_index": int(update_index), "native_batch_reads": 1,
                  "unique_input_records": int(unique_input_records),
                  "record_exposures": int(unique_input_records),
                  "joint_records_charged": 1,
                  "note": ("a repeated read of the SAME recorded records; not a new "
                           "environment episode; the env guard stays 0/0/0")}

    # Real recorded batch -> samples (real batch_samples + GAE + reward-event attribution).
    # batch_samples performs no trunk forward (recorded-value tensor math only): no charge here.
    samples = trainer.batch_samples(episodes)
    prefix_task, prefix_cost_bytes = trainer._unassigned_prefix_totals(episodes)

    # Unified real-entry metering: every controller.inner.trunk.forward call (the alignment
    # pass, every epoch, any other legitimate check) counts its ACTUAL input shape and is
    # pre-charged at the entry. There is no caller-side aggregate pre-charge any more.
    meter = resources.ensure_trunk_meter(
        controller, what_prefix=f"r75_native_update_{update_index}")
    pre_meter = meter.snapshot()

    # Real likelihood recomputation through the SAME routed forward as sampling (metered).
    alignment = r66.likelihood_alignment(controller, samples)
    if not alignment["passed"]:
        raise RuntimeError(
            f"update {update_index}: logged probability does not describe the draw: {alignment}")
    # Cumulative snapshot at the END of the alignment pass. The per-phase report must
    # difference CUMULATIVE snapshots: _meter_delta(before, after) = after - before, so the
    # epoch phase has to start from this cumulative value and not from the alignment DELTA
    # (subtracting a delta from a cumulative over-counts the epoch phase by whatever the
    # controller already consumed before this update). alignment + epochs == total for every
    # component, and the cross-update cumulative is preserved because the meter stays on.
    alignment_end = meter.snapshot()
    alignment_meter = _meter_delta(pre_meter, alignment_end)
    if not samples:
        return {"samples": 0, "minibatches": 0, "optimizer_steps": 0,
                "unassigned_prefix_task_reward": prefix_task,
                "unassigned_prefix_cost_bytes": prefix_cost_bytes,
                "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
                "likelihood_alignment": alignment, "note": "no decision step in the batch",
                "batch_read": batch_read,
                "trunk_entry_meter": {"alignment": alignment_meter,
                                      "epochs": {key: 0 for key in _METER_KEYS},
                                      "total": alignment_meter}}

    shuffle = random.Random(int(hashlib.sha256(
        f"{common.DATA_ORDER_SEED}:{int(update_index)}".encode("utf-8")).hexdigest()[:16], 16))
    order = list(range(len(samples)))
    totals = {"surrogate": 0.0, "value_task": 0.0, "value_cost": 0.0, "entropy": 0.0,
              "kl": 0.0, "clip": 0.0, "grad": 0.0, "gate": 0.0, "gate_rows": 0,
              "early_free": 0, "late_free": 0}
    modules = (("trunk", controller.inner.trunk),
               ("early_head", controller.inner.early_head),
               ("late_head", controller.inner.late_head),
               ("task_value_head", controller.inner.value_head),
               ("cost_head", controller.cost_head))
    module_gradient_reads: list[dict[str, float]] = []
    gate_records: list[dict[str, Any]] = []
    gate_diag: dict[str, Any] = {"pending": update_index in diag_updates, "done": False, "record": None}
    minibatches = 0
    for _epoch in range(common.EPOCHS_PER_BATCH):
        shuffle.shuffle(order)
        for start in range(0, len(order), common.MINIBATCH_STEPS):
            chunk = [samples[index] for index in order[start:start + common.MINIBATCH_STEPS]]
            stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
            hidden = controller.inner.trunk(stacked)   # metered at the real trunk entry
            logits = r66.route_logits(controller, stacked, hidden)
            candidate_values_task = controller.inner.value_head(hidden).squeeze(-1)
            candidate_values_cost = controller.cost_head(hidden).squeeze(-1)
            row = 0
            new_log_prob: list[torch.Tensor] = []
            entropy: list[torch.Tensor] = []
            pooled_task: list[torch.Tensor] = []
            pooled_cost: list[torch.Tensor] = []
            gate_skip_log_probs: list[torch.Tensor] = []
            early_free = 0
            late_free = 0
            for item in chunk:
                count = len(item["targets"])
                rows = torch.arange(row, row + count)
                row += count
                picked = torch.tensor([item["actions"][t] for t in item["targets"]], dtype=torch.long)
                usable = torch.tensor([1.0 if t in item["free"] else 0.0
                                       for t in item["targets"]], dtype=torch.float32)
                new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
                entropy.append(action_entropy(logits[rows], usable))
                pooled_task.append(candidate_values_task[rows].mean())
                pooled_cost.append(candidate_values_cost[rows].mean())
                # Gate prior (beta=1.00, ready_cut=147): early first-emission free candidates.
                for gate_index, gate_target in enumerate(item["targets"]):
                    if gate_target not in item["free"]:
                        continue
                    gate_feature = item["features"][gate_index]
                    ready = int(round(float(gate_feature[18]) * HORIZON) - round(float(gate_feature[0])))
                    if ready <= READY_CUT:
                        early_free += 1
                        if float(gate_feature[5]) == 1.0:
                            gate_skip_log_probs.append(
                                torch.log_softmax(logits[rows][gate_index], dim=-1)[SKIP_INDEX])
                    else:
                        late_free += 1
            new_log_prob_t = torch.stack(new_log_prob)
            entropy_t = torch.stack(entropy)
            old_log_prob = torch.tensor([item["log_prob"] for item in chunk], dtype=torch.float32)
            advantage = torch.tensor([item["advantage"] for item in chunk], dtype=torch.float32)
            target_task = torch.tensor([item["return_task"] for item in chunk], dtype=torch.float32)
            target_cost = torch.tensor([item["return_cost"] for item in chunk], dtype=torch.float32)
            ratio = torch.exp(new_log_prob_t - old_log_prob)
            surrogate = torch.min(ratio * advantage,
                                  torch.clamp(ratio, 1.0 - common.CLIP, 1.0 + common.CLIP) * advantage)
            value_task_loss = ((torch.stack(pooled_task) - target_task) ** 2).mean()
            value_cost_loss = ((torch.stack(pooled_cost) - target_cost) ** 2).mean()
            gate_loss = (-(torch.stack(gate_skip_log_probs)).sum() / len(chunk)
                         if gate_skip_log_probs else torch.zeros((), dtype=torch.float32))
            loss = (-surrogate.mean()
                    + common.VALUE_COEFFICIENT * (value_task_loss + value_cost_loss)
                    - common.ENTROPY_BONUS * entropy_t.mean()
                    + gate_beta * gate_loss)
            if gate_diag["pending"] and not gate_diag["done"]:
                # Passive diagnostic: raw early/late/critic gradients read AFTER backward and
                # BEFORE clip/step, via autograd.grad on the TRAINABLE objects only. The frozen
                # trunk has requires_grad=False and is never handed to autograd.grad.
                trainable = trainable_parameters(controller)
                early_params = list(controller.inner.early_head.parameters())
                late_params = list(controller.inner.late_head.parameters())
                critic_params = (list(controller.inner.value_head.parameters())
                                 + list(controller.cost_head.parameters()))
                base_only = loss - gate_beta * gate_loss
                prior_only = gate_beta * gate_loss
                record = {"early_free_rows": early_free, "late_free_rows": late_free,
                          "base_output_requires_grad": bool(base_only.requires_grad),
                          "prior_output_requires_grad": bool(prior_only.requires_grad),
                          "trunk": "frozen_requires_grad_false",
                          "trainable_parameter_object_count": len(trainable)}
                gb = gg = None
                if base_only.requires_grad:
                    gb = torch.autograd.grad(base_only, trainable, retain_graph=True, allow_unused=True)
                    record["base_trainable_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gb if g is not None))
                    ge = torch.autograd.grad(base_only, early_params, retain_graph=True, allow_unused=True)
                    record["early_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in ge if g is not None))
                    gr = torch.autograd.grad(base_only, late_params, retain_graph=True, allow_unused=True)
                    record["late_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gr if g is not None))
                    gc = torch.autograd.grad(base_only, critic_params, retain_graph=True, allow_unused=True)
                    record["critic_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gc if g is not None))
                else:
                    record["base_trainable_grad_l1"] = None
                    record["early_actor_grad_l1"] = None
                    record["late_actor_grad_l1"] = None
                    record["critic_grad_l1"] = None
                    record["base_gradient"] = "absent_output_has_no_graph"
                if prior_only.requires_grad:
                    gg = torch.autograd.grad(prior_only, trainable, retain_graph=True, allow_unused=True)
                    record["prior_trainable_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gg if g is not None))
                    ge2 = torch.autograd.grad(prior_only, early_params, retain_graph=True, allow_unused=True)
                    record["prior_early_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in ge2 if g is not None))
                    if gb is not None:
                        dot = float(sum(float((a * b).sum()) for a, b in zip(gb, gg)
                                        if a is not None and b is not None))
                        nb = math_sqrt(sum(float((g ** 2).sum()) for g in gb if g is not None))
                        ng = math_sqrt(sum(float((g ** 2).sum()) for g in gg if g is not None))
                        record["dot"] = dot
                        record["cosine"] = (dot / (nb * ng)) if nb > 0 and ng > 0 else None
                else:
                    record["prior_trainable_grad_l1"] = None
                    record["prior_early_actor_grad_l1"] = None
                    record["prior_gradient"] = "absent_no_eligible_gate_rows"
                gate_diag["record"] = record
                gate_diag["done"] = True
            optimizer.zero_grad()
            loss.backward()
            module_gradient_reads.append({
                module_name: (None if module_name in FROZEN_MODULES else
                              sum(float(parameter.grad.detach().abs().sum())
                                  for parameter in module.parameters() if parameter.grad is not None))
                for module_name, module in modules})
            # The R75 global clip norm is over the EIGHT trainable objects only.
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(controller),
                                                  common.MAX_GRAD_NORM)
            # Pre-charge BEFORE the physical step; a step failure does not refund.
            physical_step_what = (
                f"{physical_step_registration_prefix}:"
                f"r75_native_update_{update_index}_minibatch_{minibatches}"
                if physical_step_registration_prefix is not None else
                f"r75_native_update_{update_index}_minibatch_{minibatches}")
            resources.charge_physical_step(
                what=physical_step_what, category=physical_step_category)
            optimizer.step()
            if not (torch.isfinite(loss) and torch.isfinite(norm)):
                raise RuntimeError(f"update {update_index}: non-finite loss or gradient norm")
            totals["surrogate"] += float(surrogate.mean().item())
            totals["value_task"] += float(value_task_loss.item())
            totals["value_cost"] += float(value_cost_loss.item())
            totals["entropy"] += float(entropy_t.mean().item())
            totals["kl"] += float((old_log_prob - new_log_prob_t).mean().item())
            totals["clip"] += float(((ratio - 1.0).abs() > common.CLIP).float().mean().item())
            totals["grad"] += float(norm)
            totals["gate"] += float(gate_loss.item())
            totals["gate_rows"] += len(gate_skip_log_probs)
            totals["early_free"] += early_free
            totals["late_free"] += late_free
            minibatches += 1
            gate_records.append({"update_index": int(update_index), "minibatch": minibatches - 1,
                                 "n_rows": len(chunk), "gate_rows": len(gate_skip_log_probs),
                                 "early_free": early_free, "late_free": late_free,
                                 "gate_loss_raw": float(gate_loss.item()),
                                 "gate_loss_weighted": float(gate_beta * gate_loss.item())})
    epochs_meter = _meter_delta(alignment_end, meter.snapshot())
    diag_record = None
    if gate_diag["done"]:
        diag_record = {"update_index": update_index, **gate_diag["record"]}
    return {
        "samples": len(samples), "minibatches": minibatches, "optimizer_steps": minibatches,
        "epochs": common.EPOCHS_PER_BATCH,
        "unassigned_prefix_task_reward": prefix_task,
        "unassigned_prefix_cost_bytes": prefix_cost_bytes,
        "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
        "policy_surrogate": round(totals["surrogate"] / minibatches, 6),
        "value_loss_task": round(totals["value_task"] / minibatches, 6),
        "value_loss_cost": round(totals["value_cost"] / minibatches, 6),
        "entropy": round(totals["entropy"] / minibatches, 6),
        "approx_kl": round(totals["kl"] / minibatches, 6) if minibatches else 0.0,
        "clip_fraction": round(totals["clip"] / minibatches, 4),
        "gradient_norm_mean": round(totals["grad"] / minibatches, 4),
        "gate_loss": round(totals["gate"] / minibatches, 6) if minibatches else 0.0,
        "gate_rows": totals["gate_rows"], "early_free": totals["early_free"],
        "late_free": totals["late_free"],
        "module_gradient_l1_mean": {
            module_name: (None if module_name in FROZEN_MODULES else
                          round(sum(read[module_name] for read in module_gradient_reads)
                                / len(module_gradient_reads), 9))
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_l1_min": {
            module_name: (None if module_name in FROZEN_MODULES else
                          round(min(read[module_name] for read in module_gradient_reads), 9))
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_scope": ("per-minibatch pre-clip L1 of every TRAINABLE module's "
                                  "parameters, read after backward and before clipping/step; "
                                  "the frozen trunk is None because requires_grad=False, which "
                                  "is a structural fact and not a measured zero gradient"),
        "gate_diag": diag_record, "gate_records": gate_records,
        "batch_read": batch_read,
        "trunk_entry_meter": {"alignment": alignment_meter, "epochs": epochs_meter,
                              "total": _meter_delta(pre_meter, meter.snapshot()),
                              "scope": ("actual controller.inner.trunk.forward calls; "
                                        "attempted / charged(licensed) / completed are separate; "
                                        "a mid-way failure never passes a pre-charge off as a "
                                        "completed forward")},
        "likelihood_alignment": alignment,
        "structure": R75_STRUCTURE,
        "frozen_modules": list(FROZEN_MODULES),
        "trainable_modules": list(TRAINABLE_MODULES),
        "trainable_parameter_object_count": len(trainable_parameters(controller)),
        "frozen_parameter_object_count": len(frozen_parameters(controller)),
    }


def math_sqrt(value: float) -> float:
    import math
    return math.sqrt(value)


def update_accounting(n_samples: int, trunk_rows_per_epoch: int) -> dict[str, int]:
    """Pure arithmetic (no torch): the conservative per-update resource accounting.

    steps = EPOCHS_PER_BATCH x ceil(n_samples / MINIBATCH_STEPS); each minibatch is one physical
    optimizer step. Offline trunk target rows = one alignment pass + one pass per epoch.
    """
    import math as _math
    minibatches_per_epoch = max(1, _math.ceil(n_samples / 256))
    steps = 4 * minibatches_per_epoch
    return {"samples": int(n_samples),
            "minibatches_per_epoch": int(minibatches_per_epoch),
            "physical_optimizer_steps": int(steps),
            "offline_trunk_target_rows": int(trunk_rows_per_epoch * (4 + 1)),
            "epochs": 4, "minibatch_steps": 256}
