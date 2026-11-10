"""The real-world seam: frozen A4 Decoupled/K0 adapter plus the Stage C transport fields.

A4 owns the Decoupled belief isolation: ``adapter.install(a3, RuntimeSpec(sensing_mode=
"Decoupled", initial_pool_k=0, ...))`` patches ``refresh_local_sensing`` and the arrival guard on
the run it builds. This module therefore builds the run through ``a3.build_joint_run`` under that
installed spec instead of calling ``a2_a3_runner.run_world``, whose scene whitelist is the A3
eight-scene manifest and whose ledger and labels belong to an older round.

No old file is modified. The scene authority, this stage's 4320-slot plan (six resumed
training runs of 720 worlds each, nothing inherited) and the unique labels are this
directory protocol; everything physical still comes from the frozen archive configuration.
"""

from __future__ import annotations

import contextlib
import importlib
import time
import re
from pathlib import Path

from . import constants as C
from . import scale_features as scale_features_mod
from . import scale_program
from . import unlimited_budget
from .paths import REPO_ROOT

#: A4 transport overrides. Values are quoted from a4_hooks.py:34-42, not re-invented here.
A4_TRANSPORT_OVERRIDES = {
    "comm_policy": "learned",
    "comm_budget_bytes": C.BUDGET_BYTES,
    "comm_sends_per_tick": None,
    "comm_ack": False,
    "comm_subscribers": "all",
    "comm_delivery_probability": 0.7,
    "comm_delay_ticks": 0,
    "comm_timing_mode": "policy_owned_timing",
    "comm_max_wait_steps": None,
    "comm_urgent_wait_steps": None,
    "gossip_interval": 12,
    "steps": C.STEPS,
}

#: Local copy of a1_metrics.CONFIG_EXPECTATIONS with exactly one registered change: the byte
#: budget of this experiment. The frozen external file keeps its own 29300 and is never edited;
#: configuration_fields() additionally proves that every other key still equals the frozen
#: value, so a real physical drift cannot hide behind this copy.
FROZEN_CONFIG_EXPECTATIONS = {
    "num_goals": 24,
    "steps": 1000,
    "comm_budget_bytes": C.BUDGET_BYTES,
    "comm_delivery_probability": 0.7,
    "comm_delay_ticks": 0,
    "comm_ack": False,
    "search_agents": 2,
    "rescue_robots": 6,
}

#: The only key this experiment is allowed to differ on.
REGISTERED_EXPECTATION_OVERRIDE = "comm_budget_bytes"

#: The cell this process is currently running, set by the driver before every world.
#: ``None`` means "not under the scale programme"; a world built then uses the frozen values,
#: which keeps the module importable and testable without a programme.
CURRENT_CELL = None

#: Extra registered expectation overrides a cell needs, as {a1 key: expected value}. A cell
#: with G != 24 moves ``num_goals``; one with R != 6 moves ``rescue_robots``. Both are derived
#: from the cell rather than hand-written, so the divergence set can be checked against the
#: cell itself instead of against a second copy of the same numbers.
def cell_expectation_overrides(cell, *, allow_extended_goals=True):
    """``{a1 key: value}`` for the deviations a cell causes, with the frozen value removed.

    ``allow_extended_goals`` defaults to True because this is the *parser*: a G-scaling world
    at G=30 must resolve its own goal count instead of being refused. It cannot widen the
    training grid, because ``scale_program.cell_for`` still samples only the 15 registered
    cells -- a G that no training episode ever drew stays unreachable from the trainer.
    """
    from . import scale_program
    if cell is None:
        return {}
    goals, robots, subscribers = scale_program.validate_cell(
        cell, allow_extended_goals=allow_extended_goals)
    overrides = {}
    if goals != 24:
        overrides["num_goals"] = goals
    if robots != 6:
        overrides["rescue_robots"] = robots
    return overrides


#: Why each deviation is admissible, written out for the report.
SCALE_EXPECTATION_OVERRIDE_REASONS = {
    "num_goals": "G is an experimental axis: the cell's goal count is the world's goal count, "
                 "and a1's frozen 24 is simply the value every earlier round used.",
    "rescue_robots": "R is an experimental axis. joint_driver derives rescue_robots as "
                     "rescue_species * rescue_per_species, so registering it registers the "
                     "roster the cell actually runs, not a hidden edit.",
    "num_goals_extended": "The G-scaling sweep evaluates the sealed weights at goal counts "
                          "the trainer never sampled (18/30/36/42/54/60/72). The world is a "
                          "registered configuration, not a relaxed check: scale_program."
                          "GOALS_ACCEPTED names every admitted count, and CELLS still holds "
                          "only the 15 training cells.",
}

#: The three in-memory budget references the frozen runtime actually reads. A 58600 B world
#: needs all three to agree: the executor's credit, the value v2_run._assert_transport compares
#: it with, and the two A4 relay constants that write warm-up metadata. These are module
#: attributes only, no file is touched, and every one of them is restored in a finally block.
BUDGET_AUTHORITY_KEYS = (
    "v2_common.STAGE_B_INHERITED_OVERRIDES['comm_budget_bytes']",
    "a4_hooks.RELAY_BUDGET_BYTES",
    "a4_world_adapter.RELAY_BUDGET_BYTES",
)
#: What all three hold before this revision binds them; another value is a frozen drift.
FROZEN_AUTHORITY_BUDGET_BYTES = C.FROZEN_AUTHORITY_BUDGET_BYTES

_LAST_BUDGET_AUTHORITY: dict = {}


def budget_authority_snapshot(common, hooks, adapter):
    """Pure: the three live budget references exactly as this process holds them now."""
    return {
        BUDGET_AUTHORITY_KEYS[0]: int(common.STAGE_B_INHERITED_OVERRIDES["comm_budget_bytes"]),
        BUDGET_AUTHORITY_KEYS[1]: int(hooks.RELAY_BUDGET_BYTES),
        BUDGET_AUTHORITY_KEYS[2]: int(adapter.RELAY_BUDGET_BYTES),
    }


def _remember_budget_authority(record):
    """Keep the latest before/inside/after evidence, including on a failed world."""
    _LAST_BUDGET_AUTHORITY.clear()
    _LAST_BUDGET_AUTHORITY.update(
        {key: record.get(key)
         for key in ("profile_budget_bytes", "before", "inside", "after_exit", "restore_ok",
                     "budget_credit")})
    return dict(_LAST_BUDGET_AUTHORITY)


def reset_budget_authority():
    """Forget the previous world, so a failure can never report another world's evidence."""
    _LAST_BUDGET_AUTHORITY.clear()
    return {}


def last_budget_authority():
    """The most recent context record; an empty mapping before any world in this process."""
    return dict(_LAST_BUDGET_AUTHORITY)


def budget_authority_evidence():
    """Persistable evidence of the last context; never raises and never invents a value.

    OBSERVED_IN_WORKER means this worker really opened the context for the last world.
    UNKNOWN_NOT_OBSERVED means it did not, so every field stays null instead of being filled
    with the previous world's record or with the expected constant. usage_proof is unknown when
    the world failed before the executor's own credit was read.
    """
    try:
        record = last_budget_authority()
        if not record:
            return {"status": "UNKNOWN_NOT_OBSERVED", "before": None, "inside": None,
                    "after_exit": None, "restore_ok": None,
                    "profile_budget_bytes": None, "usage_proof": None}
        usage = record.get("budget_credit")
        return {
            "status": "OBSERVED_IN_WORKER",
            "before": dict(record.get("before") or {}) or None,
            "inside": dict(record.get("inside") or {}) or None,
            "after_exit": dict(record.get("after_exit") or {}) or None,
            "restore_ok": record.get("restore_ok"),
            "profile_budget_bytes": record.get("profile_budget_bytes"),
            "usage_proof": (dict(usage) if usage
                            else {"status": "UNKNOWN_NOT_OBSERVED_BEFORE_USAGE"}),
        }
    except Exception:  # evidence must never replace the original world error
        return {"status": "UNKNOWN_EVIDENCE_ERROR"}


@contextlib.contextmanager
def budget_authority(common, hooks, adapter, budget_bytes):
    """Point the three budget authorities at this experiment's 58600 B for one world.

    configuration_fields has already run outside this context: it binds the frozen 29,300 and
    checks this revision's one registered deviation against the local expectation copy. Inside,
    the executor's credit, the constant the frozen v2_run._assert_transport compares it with,
    and the two A4 relay constants all read 58600. The frozen assertion body runs unchanged; it
    is neither copied nor relaxed. All three references are restored in a finally block and
    read back, so a leaked override fails closed instead of quietly relaxing the next world.
    """
    budget = int(budget_bytes)
    if budget < 1:
        raise ValueError("budget must be >= 1, got %r" % (budget_bytes,))
    before = budget_authority_snapshot(common, hooks, adapter)
    drifted = {key: value for key, value in before.items()
               if value != FROZEN_AUTHORITY_BUDGET_BYTES}
    if drifted:
        raise RuntimeError("frozen budget authorities moved before the context opened: %r"
                           % (drifted,))
    overrides = common.STAGE_B_INHERITED_OVERRIDES
    record = {"profile_budget_bytes": budget, "authority_keys": list(BUDGET_AUTHORITY_KEYS),
              "before": before, "inside": None, "after_exit": None, "restore_ok": None}
    error = None
    try:
        overrides["comm_budget_bytes"] = budget
        hooks.RELAY_BUDGET_BYTES = budget
        adapter.RELAY_BUDGET_BYTES = budget
        inside = budget_authority_snapshot(common, hooks, adapter)
        if set(inside.values()) != {budget}:
            raise RuntimeError("budget authority override did not take effect: %r" % (inside,))
        record["inside"] = inside
        yield record
    except BaseException as caught:  # the world error must survive the restore
        error = caught
        raise
    finally:
        overrides["comm_budget_bytes"] = before[BUDGET_AUTHORITY_KEYS[0]]
        hooks.RELAY_BUDGET_BYTES = before[BUDGET_AUTHORITY_KEYS[1]]
        adapter.RELAY_BUDGET_BYTES = before[BUDGET_AUTHORITY_KEYS[2]]
        record["after_exit"] = budget_authority_snapshot(common, hooks, adapter)
        record["restore_ok"] = record["after_exit"] == before
        if not record["restore_ok"] and error is None:
            raise RuntimeError("budget authority restore failed: %r != %r"
                               % (record["after_exit"], before))


def check_budget_credit(usage, fields, record, budget_bytes=None):
    """Pure: the four numbers that must agree before a world may run.

    executor is what the relay's own CommBudget holds, run_configuration is what was handed to
    RunConfiguration, context is the profile this revision opened and protocol is the constant
    every report divides by. Anything but the registered budget - the frozen 29300, an
    off-by-one 58601, or an unbounded None - is refused.
    """
    budget = int(C.BUDGET_BYTES if budget_bytes is None else budget_bytes)
    executor = usage.get("bytes_per_episode")
    four = {
        "executor_budget_bytes": None if executor is None else int(executor),
        "run_configuration_budget_bytes": int(fields["comm_budget_bytes"]),
        "context_budget_bytes": int(record["profile_budget_bytes"]),
        "protocol_budget_bytes": budget,
    }
    wrong = {name: value for name, value in four.items() if value != budget}
    if wrong:
        raise RuntimeError("budget credit disagreement (expected %d B): %r" % (budget, wrong))
    return four

#: Transcript vocabulary verified in integrated_sar/comm.py:920-931 (refused) and :995-1010
#: (send). A refusal names the budget layer's own reason, never a policy label.
TRANSCRIPT_SEND = "send"
TRANSCRIPT_REFUSED = "refused"
REFUSAL_BUDGET = "budget_exhausted"
REFUSAL_CAPACITY = "capacity_step"


def _a4_modules(layout):
    """Import the frozen A3 runner and A4 adapter by path, never by guessing module aliases."""
    from .episode import ensure_import_paths
    ensure_import_paths(layout)
    repo = REPO_ROOT
    a4_root = repo / "runs" / "a4_staged_pool_probe_20261020"
    import sys
    for path in (str(a4_root), str(a4_root / "implementation"), str(repo / "runs" / "attribution_and_gossip_probe_20261020")):
        if path not in sys.path:
            sys.path.insert(0, path)
    a3 = importlib.import_module("a2_a3_runner")
    adapter = importlib.import_module("a4_world_adapter")
    hooks = importlib.import_module("a4_hooks")
    return a3, adapter, hooks


def verify_resolved_configuration(fields, expected):
    """Check every a1 expectation key plus the N24 goal count; pure and testable."""
    checks = {
        "steps": int(fields.get("steps", -1)),
        "comm_budget_bytes": int(fields.get("comm_budget_bytes", -1)),
        "comm_delivery_probability": float(fields.get("comm_delivery_probability", -1.0)),
        "comm_delay_ticks": int(fields.get("comm_delay_ticks", -1)),
        "comm_ack": bool(fields.get("comm_ack", True)),
        "search_agents": int(fields.get("search_agents", -1)),
        "rescue_robots": (int(fields.get("rescue_species", -1))
                          * int(fields.get("rescue_per_species", -1))),
    }
    missing = sorted(set(expected) - set(checks) - {"num_goals"})
    if missing:
        raise RuntimeError("a1 expectations without a resolver: %r" % (missing,))
    wrong = {name: (checks[name], expected[name]) for name in sorted(expected)
             if name != "num_goals"
             if checks[name] != expected[name]}
    if wrong:
        raise RuntimeError("frozen physical configuration moved: %r" % (wrong,))
    # num_goals is the cell's goal count. Under the scale programme that is G, not a constant:
    # 12, 24 and 48 are all registered cells, so the check is "the resolved value equals what
    # this cell asked for", which is exactly what the frozen literal could not express. Seeing
    # a value the cell did not ask for means the authority binding never ran.
    if "num_goals" not in fields:
        raise RuntimeError("resolved configuration has no num_goals")
    expected_goals = int(expected["num_goals"])
    if int(fields["num_goals"]) != expected_goals:
        raise RuntimeError("resolved num_goals %s != the cell's %d (authority binding missing?)"
                           % (fields["num_goals"], expected_goals))
    from . import scale_program
    if int(expected_goals) not in scale_program.GOALS_ACCEPTED:
        raise RuntimeError("num_goals %d is neither a training cell nor a registered "
                           "G-scaling goal count %r"
                           % (expected_goals, scale_program.GOALS_ACCEPTED))
    if int(checks["steps"]) != C.STEPS:
        raise RuntimeError("resolved steps %d != protocol %d" % (checks["steps"], C.STEPS))
    return checks


def frozen_expectation_divergence(frozen_module_expectations, ours=None):
    """Pure: every key where this directory differs from the frozen a1 expectations.

    ``ours`` defaults to the module constant, so every pre-existing caller sees exactly the
    comparison it always saw. The scale probe passes its own expectation set so a registered
    N-related deviation shows up as a divergence to be accounted for rather than being
    silently absent from both sides.
    """
    ours = dict(FROZEN_CONFIG_EXPECTATIONS if ours is None else ours)
    theirs = dict(frozen_module_expectations)
    return {key: (ours.get(key), theirs.get(key))
            for key in sorted(set(ours) | set(theirs))
            if ours.get(key) != theirs.get(key)}


def configuration_fields(layout, scene, cell=None):
    """Frozen archive configuration for the seed, with this cell's G/R/N applied.

    The order matters. ``stage_b_inherited_config`` first resolves the frozen 29,300 B world
    and its 24 goals / 6 robots; the cell then overwrites exactly three keys, and the
    divergence gate asserts the resulting deviation set equals exactly
    ``{comm_budget_bytes} | cell overrides``. Nothing else may move, so a physical drift
    cannot hide behind the axes.
    """
    from . import scale_program
    if cell is None:
        cell = CURRENT_CELL
    a3, adapter, hooks = _a4_modules(layout)
    adapter.verify_a3_identity(a3, hooks.A3_RUNNER_SHA256, hooks.A3_METRICS_SHA256,
                               hooks.A3_PREFLIGHT_SHA256)
    a3.bind_configuration_authority()
    import v2_common as common
    fields, substituted = common.stage_b_inherited_config(int(scene))
    fields.update(A4_TRANSPORT_OVERRIDES)
    overrides = cell_expectation_overrides(cell)
    if cell is not None:
        goals, robots, subscribers = scale_program.validate_cell(
            cell, allow_extended_goals=True)
        fields["num_goals"] = int(goals)
        fields["rescue_per_species"] = int(scale_program.rescue_per_species(robots))
        fields["comm_subscribers"] = scale_program.subscriber_spec_for(cell)
    else:
        fields["comm_subscribers"] = "all"
    fields["seed"] = int(scene)
    import a1_metrics
    expected = dict(FROZEN_CONFIG_EXPECTATIONS)
    expected.update(overrides)
    divergence = frozen_expectation_divergence(a1_metrics.CONFIG_EXPECTATIONS, expected)
    allowed = {REGISTERED_EXPECTATION_OVERRIDE} | set(overrides)
    if set(divergence) != allowed:
        raise RuntimeError("frozen a1 expectations moved beyond the registered budget and "
                           "the cell axes: %r" % (divergence,))
    verify_resolved_configuration(fields, expected)
    return fields, substituted


def masked_budget_totals(collector):
    """Read the learner credit/capacity masks and cross-check the collector's own counters.

    ``BudgetMaskedCollector`` masks a candidate before the relay sees it.  The tape's
    ``masked_reasons`` is the per-step evidence; ``distribution_reads`` is the sampler's own
    count of the same decision.  They must agree, otherwise the compact budget fields and the
    relay refusal fields would describe different populations.  A missing step on a masked
    tape is a hard error rather than a guessed tick.
    """
    if collector is None or not hasattr(collector, "records"):
        return None, None, None
    credit_total = 0
    capacity_total = 0
    first_credit = None
    for tape in collector.records:
        step = int(tape.get("step", -1))
        reasons = tape.get("masked_reasons") or {}
        credit = [target for target, reason in reasons.items()
                  if str(reason) == REFUSAL_BUDGET]
        capacity = [target for target, reason in reasons.items()
                    if str(reason) == REFUSAL_CAPACITY]
        if credit and step < 0:
            raise RuntimeError("masked credit candidate has no recorded step")
        credit_total += len(credit)
        capacity_total += len(capacity)
        if credit:
            first_credit = step if first_credit is None else min(first_credit, step)
    reads = getattr(collector, "distribution_reads", None)
    if isinstance(reads, dict):
        expected = {"masked_by_credit": credit_total,
                    "masked_by_capacity": capacity_total}
        for key, value in expected.items():
            if key not in reads:
                raise RuntimeError("collector has no %s artifact" % key)
            if int(reads[key]) != value:
                raise RuntimeError("collector %s=%s but tapes hold %s"
                                   % (key, reads[key], value))
    if credit_total > 0 and first_credit is None:
        raise RuntimeError("credit masks were recorded without a first tick")
    return credit_total, capacity_total, first_credit


def _close_run_logger(run):
    """Close only this run's own lifecycle stream, with the frozen idempotent close().

    revision_03 lost three completed worlds here: this layer builds the run directly (not
    through joint_driver.run_from_configuration), so nothing closed the lifecycle stream and
    the later gzip/unlink of the plaintext log failed on Windows because the file was still
    open. close() flushes and closes the owned stream, writes no event and touches no counter.
    No other handler is scanned or closed.
    """
    logger = getattr(run, "logger", None)
    if logger is not None and hasattr(logger, "close"):
        logger.close()
    return logger


def _release_run(run, hook, taps):
    """The single finally path: unhook, restore the taps, then close this run's logger.

    It runs on success and on failure alike, and it always finishes before any log is read or
    archived, so the plaintext streams are no longer held open when the archive step unlinks
    them.
    """
    if hook is not None:
        hook.remove()
    taps["relay"].tick = taps["original_tick"]
    if taps.get("causal_restore_emit") is not None:
        taps["relay"]._emit = taps["causal_restore_emit"]
    if taps.get("scale_restore_forward") is not None:
        scale_features_mod.ScaleFeatures.restore(taps)
    from integrated_sar.contracts import ProgressEvent
    taps["event_bus"].unsubscribe(ProgressEvent, taps["reward_handler"])
    _close_run_logger(run)


def artifact_directory(layout, label, tag="", artifact_namespace=None):
    """Log paths only: never change RuntimeSpec.label or any physical/RNG argument."""
    if Path(label).name != label or label in ("", ".", ".."):
        raise ValueError("invalid artifact leaf label")
    if artifact_namespace is None:
        namespace = tag or "world"
    else:
        if not re.fullmatch(r"(train|pacing|eval|replay):[A-Za-z0-9_]+:(none|[0-9]+):[0-9]{3,}",
                            artifact_namespace):
            raise ValueError("invalid logical-slot artifact namespace")
        namespace = artifact_namespace.replace(":", "__")
    if Path(namespace).name != namespace or namespace in ("", ".", ".."):
        raise ValueError("artifact namespace cannot escape world_logs")
    base = layout.assert_inside(layout.world_logs)
    target = layout.assert_inside(base / namespace / label)
    if base not in target.parents:
        raise RuntimeError("artifact path escapes world_logs")
    return target


def reserve_artifact_directory(layout, label, tag="", artifact_namespace=None):
    target = artifact_directory(layout, label, tag, artifact_namespace)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic leaf creation before any native runtime/World construction. Never reopen a leaf.
    target.mkdir(exist_ok=False)
    return target


def run_episode(layout, *, label, kind, scene, policy, attach, rule="learned", controller=None,
                collector=None, action_rng_seed=None, tag="", artifact_namespace=None,
                cell=None):
    """One real world: budget authorities bound, then the frozen A4 Decoupled/K0 build.

    ``cell`` is the ``(G, R, N)`` this world runs under. Omitting it means "no programme", and
    the frozen 24-goal / 6-robot / all-subscribers configuration is built unchanged.
    """
    reset_budget_authority()
    a3, adapter, hooks = _a4_modules(layout)
    # configuration_fields runs outside the budget context on purpose: it binds the frozen
    # 29,300 first and checks that the deviation set is exactly the budget plus this cell's
    # axes, before anything is moved.
    fields, substituted = configuration_fields(layout, scene, cell)
    import v2_common as common
    record: dict = {}
    try:
        with budget_authority(common, hooks, adapter, C.BUDGET_BYTES) as record:
            row = _run_episode_body(
                layout, label=label, kind=kind, scene=scene, policy=policy, attach=attach,
                rule=rule, controller=controller, collector=collector,
                action_rng_seed=action_rng_seed, tag=tag,
                artifact_namespace=artifact_namespace, fields=fields,
                substituted=substituted, a3=a3, adapter=adapter, common=common,
                budget_record=record, cell_arg=cell)
    except BaseException:
        _remember_budget_authority(record)
        raise
    _remember_budget_authority(record)
    row["budget_authority_after_exit"] = dict(record.get("after_exit") or {})
    row["budget_authority_restored"] = record.get("restore_ok")
    return row


def _run_episode_body(layout, *, label, kind, scene, policy, attach, rule, controller,
                      collector, action_rng_seed, tag, artifact_namespace,
                      fields, substituted, a3, adapter, common, budget_record, cell_arg):
    """Construct, run and verify one A4 Decoupled/K0 world inside the budget context."""
    out_dir = reserve_artifact_directory(layout, label, tag, artifact_namespace)
    fields["output"] = str(out_dir / "lifecycle.jsonl")
    fields["comm_transcript"] = str(out_dir / "relay_transcript.jsonl")
    fields["info_probe"] = None
    fields["search_probe"] = None
    from integrated_sar.joint_driver import RunConfiguration
    configuration = RunConfiguration(**fields)
    configuration.validate()
    spec = adapter.RuntimeSpec(label=label, scene=int(scene), initial_pool_k=0,
                               sensing_mode="Decoupled", rule=str(rule))
    if int(spec.execution_steps) != C.STEPS:
        raise RuntimeError("RuntimeSpec.execution_steps is %s, protocol requires %d"
                           % (spec.execution_steps, C.STEPS))
    adapter.reset_world_registry()
    adapter.install(a3, spec)
    run = a3.build_joint_run(configuration)
    # The real generated-task denominator: len(run.scenario.goal_positions) must equal the
    # scenario's own placed count. discovered is a runtime observation and never substitutes
    # for this number.
    scenario = getattr(run, "scenario", None)
    placed = int(getattr(scenario, "num_goals_placed", -1))
    goal_positions = tuple(getattr(scenario, "goal_positions", ()) or ())
    generated_goals = len(goal_positions)
    if scenario is None or placed < 0:
        raise RuntimeError("%s: run exposes no scenario goal count" % (label,))
    if placed != generated_goals:
        raise RuntimeError("%s: scenario placed %d goals but holds %d positions"
                           % (label, placed, generated_goals))
    if generated_goals < 1:
        raise RuntimeError("%s: scenario generated no goals" % (label,))
    if getattr(adapter, "_ACTIVE", None) is None or adapter._ACTIVE.sensing_mode != "Decoupled":
        raise RuntimeError("A4 adapter did not install the Decoupled spec")
    if int(adapter._ACTIVE.execution_steps) != C.STEPS:
        raise RuntimeError("installed spec ran with %s steps, protocol requires %d"
                           % (adapter._ACTIVE.execution_steps, C.STEPS))
    if attach is None:
        raise RuntimeError("%s: no communication policy attached" % (label,))
    run.task_adapter.comm_relay.attach_policy(attach)
    import v2_run
    import torch
    expected_policy = str(getattr(attach, "name", "learned"))
    v2_run._assert_transport(run, label, expected_policy)
    # The frozen assertion can only compare the executor against one inherited constant. These
    # four numbers additionally pin the run configuration and the protocol constant itself.
    budget_credit = check_budget_credit(run.task_adapter.comm_relay.budget.usage(),
                                       fields, budget_record)
    budget_record["budget_credit"] = dict(budget_credit)
    taps = v2_run._install_taps(run)
    #: Causal credit assignment (credit_mode = "causal_task_id"): re-key each task credit
    #: onto the decision step that put the target on the air.  The frozen collector still
    #: produces the tick-keyed stream, and the reconciliation below runs on whichever
    #: stream this yields, so the totals are preserved by construction.
    causal = None
    if C.CREDIT_MODE == "causal_task_id":
        from . import causal_credit
        causal = causal_credit.CausalCreditAssignment(rule=C.CREDIT_RULE)
        taps["causal_restore_emit"] = causal.install(run.task_adapter.comm_relay)
    #: Scale feature transform (SCALE_FEATURES). Installed on every learned world, including
    #: "off", so an off world proves the wrapper itself is inert rather than proving that
    #: nothing was installed. At "off" the transform writes nothing back.
    #: It wraps ``controller.inner.trunk``, which the network observer below also needs as a
    #: real module, so the install happens after the observer is registered.
    scale_features = None
    cell = cell_arg
    network = {"status": "NOT_APPLICABLE_RULE"}
    hook = None
    if controller is not None:
        trunk = controller.inner.trunk
        network = {"status": "OBSERVING", "calls": 0, "first_shape": None,
                   "dtype": None, "last_dim": None,
                   "structure": [{"name": name, "type": type(module).__name__,
                                  "parameters": {key: list(value.shape)
                                                 for key, value in
                                                 module.named_parameters(recurse=False)}}
                                 for name, module in trunk.named_modules()]}

        def _observe(_module, inputs):
            tensor = inputs[0]
            if int(network["calls"]) == 0:
                network["first_shape"] = list(tensor.shape)
                network["dtype"] = str(tensor.dtype)
                network["last_dim"] = int(tensor.shape[-1])
            network["calls"] = int(network["calls"]) + 1

        hook = trunk.register_forward_pre_hook(_observe)
        if C.SCALE_FEATURES in scale_features_mod.MODES:
            # The divisors come from the cell, not from what the relay happens to hold: a
            # narrow cell must normalise by its own G and N, or the same world would look
            # different depending on which switch resolved first.
            goals = int(cell[0]) if cell is not None else int(C.NUM_GOALS)
            robots = int(cell[1]) if cell is not None else int(C.SCALE_ROBOTS_REF)
            subscribers = (int(cell[2]) if cell is not None
                           else len(getattr(run.task_adapter.comm_relay, "_subscribers", ())))
            scale_features = scale_features_mod.ScaleFeatures(
                mode=C.SCALE_FEATURES, goals=goals, subscribers=subscribers, robots=robots)
            taps["scale_restore_forward"] = scale_features.install(controller)
            taps["scale_features"] = scale_features
    started = time.time()
    if action_rng_seed is not None:
        torch.manual_seed(int(action_rng_seed))
    try:
        summary = run.run()
    finally:
        _release_run(run, hook, taps)
    if int(network.get("calls", 0)) > 0:
        network["status"] = "OBSERVED"
        if int(network["last_dim"]) != C.FEATURE_DIM:
            raise RuntimeError("%s: communication network input last dim %s != %d"
                               % (label, network["last_dim"], C.FEATURE_DIM))
    from integrated_sar.lifecycle_metrics import load_records, window_metrics
    records = load_records(Path(str(fields["output"])))
    metrics = window_metrics(records, configuration.steps)
    relay = run.task_adapter.comm_relay
    tick_task_stream = dict(taps["task_stream"])
    if causal is not None:
        causal_stream = causal.apply(tick_task_stream, taps["task_reward_events"])
        causal.assert_credit_preserved(tick_task_stream, causal_stream)
        taps["causal_audit"] = causal.audit()
        taps["task_stream_tick_keyed"] = tick_task_stream
        taps["task_stream"] = causal_stream
    credited = int(sum(taps["task_stream"].values()))
    if int(metrics.on_time) != credited:
        raise RuntimeError("%s: lifecycle on_time %s != tap credit %s"
                           % (label, metrics.on_time, credited))
    spent = int(relay.budget.spent_bytes)
    tap_cost = int(sum(row["airtime_bytes"] for row in taps["cost_stream"].values()))
    if spent != tap_cost:
        raise RuntimeError("%s: tap cost %d != ledger %d" % (label, tap_cost, spent))
    transcript = list(common.iter_jsonl(Path(str(fields["comm_transcript"]))))
    sends = [record for record in transcript if record.get("kind") == TRANSCRIPT_SEND]
    refusals = [record for record in transcript if record.get("kind") == TRANSCRIPT_REFUSED]
    counted = {
        "emissions": len(sends),
        "budget_refused_sends": sum(1 for r in refusals if r.get("reason") == REFUSAL_BUDGET),
        "capacity_refused_sends": sum(1 for r in refusals if r.get("reason") == REFUSAL_CAPACITY),
    }
    for record in refusals:
        if record.get("reason") not in (REFUSAL_BUDGET, REFUSAL_CAPACITY):
            raise RuntimeError("%s: unknown refusal reason %r"
                               % (label, record.get("reason")))
    for key, value in counted.items():
        if int(relay.counters.get(key, -1)) != value:
            raise RuntimeError("%s: relay %s=%s but transcript holds %s"
                               % (label, key, relay.counters.get(key), value))
    if int(summary.get("steps", -1)) != int(configuration.steps):
        raise RuntimeError("%s: ran %s steps" % (label, summary.get("steps")))
    # Learned arms mask an unaffordable candidate before the relay sees it. Read that mask
    # from the collector tapes and keep it strictly separate from the relay's own refusals.
    masked_credit_total, masked_capacity_total, first_masked_credit = \
        masked_budget_totals(collector)
    # All endpoint/charge/transcript checks have passed; only now is the two-file event log
    # compressed with a verified round trip.
    from . import compact as _compact
    archive = _compact.archive_world_logs(layout, out_dir)
    tape_summary = None
    if collector is not None:
        tape_summary = (collector.tape_summary() if hasattr(collector, "tape_summary")
                        else {"note": "greedy collector"})
    return {
        "kind": kind, "label": label, "scenario_seed": int(scene),
        "policy": policy, "spent_bytes": spent,
        # ---- scale programme: the cell this world ran, and what it implies for the report
        "cell": list(cell_arg) if cell_arg is not None else None,
        "cell_label": (scale_program.cell_label(cell_arg)
                       if cell_arg is not None else None),
        "goals": int(configuration.num_goals),
        "rescue_robots": int(configuration.rescue_species) * int(configuration.rescue_per_species),
        "subscribers_configured": (int(cell_arg[2]) if cell_arg is not None else None),
        #: The cross-cell cost divisor actually used by the trainer, recorded with the row so a
        #: report never has to re-derive it.
        "cost_scale": (float(cell_arg[2]) * float(generated_goals)
                       if cell_arg is not None else None),
        #: The multiplier the trainer applies to charged bytes. One definition, in
        #: ``unlimited_budget.scaled_cost``: charged bytes per subscriber, rescaled so the frozen
        #: (6, 24) cell keeps the ``bytes / B0`` magnitude. ``G`` is passed for validation and
        #: recorded, but it no longer prices anything -- see that function's docstring.
        "cost_reward_factor": (
            unlimited_budget.scaled_cost(1.0, int(cell_arg[2]), generated_goals)
            if cell_arg is not None and generated_goals > 0 else None),
        "bytes_per_on_time": (float(spent) / float(metrics.on_time)
                              if int(metrics.on_time) > 0 else None),
        "on_time_rate": (float(metrics.on_time) / float(generated_goals)
                         if generated_goals > 0 else None),
        "generated_goals": int(generated_goals),
        "num_goals_placed": int(placed),
        "num_goals_configured": int(configuration.num_goals),
        "budget_bytes": int(fields["comm_budget_bytes"]),
        "budget_credit": budget_credit,
        "scale_invariant": (taps["scale_features"].audit()
                            if taps.get("scale_features") is not None else None),
        "subscribers_live": len(getattr(relay, "_subscribers", ())),
        "budget_authority_before": dict(budget_record["before"]),
        "airtime_tap_total": tap_cost, "emissions": len(sends),
        "on_time": int(metrics.on_time), "completed": int(metrics.completed),
        "expired": int(metrics.expired), "discovered": int(metrics.discovered),
        "active": int(metrics.active), "unassigned": int(metrics.unassigned),
        "task_reward_total": credited,
        "b0_bytes": C.B0_BYTES,
        "emission_sequence": [[int(r["step"]), str(r["target"]), bool(r.get("first"))]
                              for r in sends],
        "refusal_sequence": [[int(r["step"]), str(r["target"]), str(r["reason"])]
                             for r in refusals],
        "cost_stream": {str(k): v for k, v in sorted(taps["cost_stream"].items())},
        "task_stream": {str(k): v for k, v in sorted(taps["task_stream"].items())},
        "task_stream_tick_keyed": ({str(k): v for k, v in
                                    sorted(taps["task_stream_tick_keyed"].items())}
                                   if "task_stream_tick_keyed" in taps else None),
        "causal_credit_audit": taps.get("causal_audit"),
        "relay_counters": {k: int(v) for k, v in relay.counters.items()},
        "masked_by_credit_total": masked_credit_total,
        "masked_by_capacity_total": masked_capacity_total,
        "first_masked_credit_tick": first_masked_credit,
        "behaviour_sha256": v2_run._behaviour_sha256(records),
        "config_hash": getattr(run, "run_signature", {}).get("config_hash"),
        "steps": int(summary.get("steps", 0)),
        "seconds": round(time.time() - started, 2),
        "collector_summary": tape_summary,
        "substituted_fields": substituted,
        "sensing_mode": "Decoupled", "initial_pool_k": 0,
        "artefacts": {item["name"]: item["gzip"]["path"] for item in archive["files"]},
        "log_provenance": archive,
        "communication_network_observation": network,
    }


def runtime_spec_fields(scene, label):
    """Pure description of the spec, used by tests without importing the A4 runtime."""
    return {"label": label, "scene": int(scene), "initial_pool_k": 0,
            "sensing_mode": "Decoupled", "rule": "learned"}

