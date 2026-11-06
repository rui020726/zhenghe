"""Thin policy wiring on top of stage5b_comm/world.py; no environment import at module load."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from . import constants as C
from .pacing import build_pacing_policy
from .paths import REPO_ROOT, REVISION_ROOT, ensure_repo_path

REPO_DIRS = (
    "runs/comm_cost_v2_rerun_20260928",
    "runs/comm_cost_stage_b_20260927",
    "runs/comm_budget_action_mask_fix_20260922",
    "runs/comm_tight_budget_20260919",
    "runs/comm_budget_pilot_20260919",
    "runs/comm_cost_controller_stage_a_20260927",
    "runs/a4_staged_pool_probe_20261020",
    "runs/a4_staged_pool_probe_20261020/implementation",
    "runs/attribution_and_gossip_probe_20261020",
    "runs/stage5a_ceiling_and_feature_audit_20261020",
)


def ensure_import_paths(layout=None):
    """Put the frozen modules on sys.path once; run-local names win over repo root."""
    ensure_repo_path()
    repo = REPO_ROOT
    ordered = [repo / item for item in REPO_DIRS] + [repo]
    for path in reversed(ordered):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo


RUNTIME_IMPORTS = (
    "v2_run", "v2_common", "v2_controller", "v2_rules", "budget_masked", "sb_collector", "onpolicy",
    "cost_math", "integrated_sar.comm_policy", "integrated_sar.comm_budget",
    "integrated_sar.expired_attribution", "integrated_sar.joint_driver", "a2_a3_runner",
    "a4_world_adapter", "a4_hooks", "a1_metrics",
    #: The pilot branch belongs to the real fresh_controller closure. a2_a3_runner (above)
    #: has already bound the authority common when these are imported, so the pinned
    #: origin/hash of common and of the three modules stays exactly as frozen.
    "train_pilot", "pilot_common", "pilot_env",
)


def validate_runtime_imports(layout=None):
    """Bootstrap and import every frozen runtime seam before any slot is claimed.

    This performs imports only: it never builds a world, loads a checkpoint, or calls a model.
    It makes an import-path defect fail before the phase can consume a world slot.
    """
    sys.dont_write_bytecode = True
    ensure_import_paths(layout)
    modules = tuple(importlib.import_module(name) for name in RUNTIME_IMPORTS)
    native_import_preflight(layout)
    return modules


#: Calls that would mean the bootstrap stopped being import-only.
FORBIDDEN_DURING_BOOTSTRAP = ("torch.load", "a2_a3_runner.build_joint_run")

#: root key -> recorded state, so one process pays the import cost only once.
_NATIVE_PREFLIGHT = {}


def _native_sentinel(name):
    def _sentinel(*args, **kwargs):
        raise RuntimeError(
            "native import preflight is import-only, but %s was called during bootstrap" % (name,))
    return _sentinel


def _module_origin_paths(module):
    """Every filesystem location a package-like object claims to live in."""
    paths = []
    for entry in getattr(module, "__path__", None) or ():
        if str(entry):
            paths.append(Path(str(entry)).resolve())
    if not paths:
        filename = getattr(module, "__file__", None)
        if filename:
            paths.append(Path(str(filename)).resolve().parent)
    return paths


def assert_native_src_origin(module, piloc_dir):
    """``sys.modules['src']`` must be PILOC's package, never an experiment namespace."""
    expected = Path(piloc_dir).resolve() / "src"
    paths = _module_origin_paths(module)
    if not paths:
        raise RuntimeError("the imported 'src' package has no filesystem origin")
    for path in paths:
        try:
            path.relative_to(expected)
        except ValueError as error:
            raise RuntimeError(
                "the imported 'src' package resolves to %s, not the frozen PILOC package %s; a "
                "local src package would shadow the native one" % (path, expected)) from error
    return expected


def native_import_preflight(layout=None, *, force=False):
    """Import the frozen native PILOC modules before any slot is claimed.

    The frozen seam is ``integrated_sar.piloc_runtime``: ``ensure_piloc_on_path`` puts
    ``piloc`` and ``piloc/src/DDQN`` on ``sys.path``, ``import_timemap_module`` imports
    ``src.Map.timemap`` and ``import_agent_module`` imports ``src.DDQN.Agent``. The import is
    definitions only: no World is constructed, no checkpoint is loaded and no model is called,
    so ``torch.load`` and ``a2_a3_runner.build_joint_run`` are replaced by sentinels for the
    duration of the bootstrap and restored in ``finally``.

    Two native import-time side effects are recorded but never changed: ``timemap`` calls
    ``plt.ion()`` and ``Agent`` sets ``torch.backends.cudnn.benchmark = True``.
    """
    root = Path(layout.root if layout is not None else REVISION_ROOT).resolve()
    key = str(root)
    if not force and key in _NATIVE_PREFLIGHT:
        return dict(_NATIVE_PREFLIGHT[key])
    sys.dont_write_bytecode = True
    shadow = root / "src"
    if shadow.is_dir():
        raise RuntimeError(
            "%s is a top-level 'src' package; it would shadow the frozen PILOC 'src' namespace"
            % (shadow,))
    ensure_import_paths(layout)
    import torch
    import a2_a3_runner

    saved = {"torch.load": torch.load}
    torch.load = _native_sentinel("torch.load")
    build = getattr(a2_a3_runner, "build_joint_run", None)
    if build is not None:
        saved["a2_a3_runner.build_joint_run"] = build
        a2_a3_runner.build_joint_run = _native_sentinel("a2_a3_runner.build_joint_run")
    try:
        from integrated_sar import piloc_runtime
        piloc_dir = Path(piloc_runtime.ensure_piloc_on_path()).resolve()
        timemap = piloc_runtime.import_timemap_module()
        agent = piloc_runtime.import_agent_module()
    finally:
        torch.load = saved["torch.load"]
        if "a2_a3_runner.build_joint_run" in saved:
            a2_a3_runner.build_joint_run = saved["a2_a3_runner.build_joint_run"]
    src_module = sys.modules.get("src")
    if src_module is None:
        raise RuntimeError("the frozen PILOC bootstrap did not register a 'src' package")
    if src_module is sys.modules.get(__package__):
        raise RuntimeError("'src' is aliased to this experiment package")
    expected = assert_native_src_origin(src_module, piloc_dir)
    pyplot = getattr(timemap, "plt", None)
    interactive = None
    if pyplot is not None and hasattr(pyplot, "isinteractive"):
        interactive = bool(pyplot.isinteractive())
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    benchmark = None if cudnn is None else getattr(cudnn, "benchmark", None)
    origin = getattr(src_module, "__file__", None)
    state = {
        "schema": "stage5b-native-import-preflight/1",
        "root": key,
        "piloc_dir": str(piloc_dir),
        "src_package_dir": str(expected),
        "src_package_file": (None if origin is None else str(Path(str(origin)).resolve())),
        "timemap_module": str(getattr(timemap, "__name__", "UNKNOWN")),
        "agent_module": str(getattr(agent, "__name__", "UNKNOWN")),
        "imports_only": True,
        "world_built": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "sentinels": list(FORBIDDEN_DURING_BOOTSTRAP),
        "sentinels_triggered": 0,
        "side_effects": {
            "matplotlib_interactive": interactive,
            "cudnn_benchmark": (None if benchmark is None else bool(benchmark)),
            "changed_by_preflight": False,
            "note": ("frozen timemap calls plt.ion() and frozen Agent sets "
                     "torch.backends.cudnn.benchmark = True at import time; both are recorded "
                     "as found and not modified"),
        },
    }
    _NATIVE_PREFLIGHT[key] = dict(state)
    return state


def build_rule_policy(layout, arm, scene):
    """Baseline rules by their frozen names, plus this revision's dedup arm."""
    # This must run before the pacing branch returns: pacing.py itself only needs the repo, but
    # the pre-world entry also needs the frozen run directories (notably v2_run for the seed).
    ensure_import_paths(layout)
    if arm in C.PACING_ARMS:
        return build_pacing_policy(arm)
    if arm == "dedup_immediate":
        # Local to this revision: first sighting or a material view change, never a blind
        # repeat. It reads only sender-visible SendItem fields, so it obeys the same belief
        # boundary as the learner.
        from . import dedup_policy
        return dedup_policy.build_dedup_policy()
    from integrated_sar.comm_policy import build_comm_policy
    if arm == "normal_immediate":
        return build_comm_policy("immediate")
    if arm == "always_hold":
        return build_comm_policy("always_hold")
    if arm == "late147":
        # late_discovery_gate lives in the frozen Stage C rule catalogue, not in the
        # production policy module; build it through that catalogue so its own parameter
        # assertion runs.
        import v2_rules
        return v2_rules.build_rule("late_discovery_gate", int(scene))
    raise ValueError("unknown rule arm %r" % (arm,))


def adapter_rule(arm):
    """A4 adapter rule name; learned covers the trained network and the dedup baseline."""
    return {"normal_immediate": "normal_immediate", "always_hold": "always_hold",
            "late147": "late147", "dedup_immediate": "dedup_immediate"}.get(arm, "learned")


def eval_arm_kind(arm):
    """Every fixed rule arm must bypass the learned controller; only networks use it.

    The registered converged-run names (C.EVAL_LEARNING_ARMS) are part of the learned set: the
    parent round crashed precisely because a milestone arm was missing from this whitelist, so
    the name check is an exact set membership test that includes the declared arms.
    """
    if arm in C.BASELINE_ARMS or arm in C.PACING_ARMS:
        return "rule"
    # An exact set, not a suffix test: "half_trained" must fail instead of being dispatched.
    if arm in ("untrained", "T1_trained", "T2_trained") or arm in C.EVAL_LEARNING_ARMS:
        return "learned"
    raise ValueError("unknown evaluation arm %r" % (arm,))


def dispatch_eval(arm, rule, learned):
    """Call exactly one branch; the learned branch is never entered for a fixed rule arm."""
    if eval_arm_kind(arm) == "rule":
        return rule()
    return learned()


def eval_action_seed(scene):
    """One shared eval stream for every arm: fixed EVAL_WORLD_SEED + scene + greedy."""
    # This entry can be called directly by a fresh spawn, before any other test or phase
    # bootstrapped the frozen run directories.  It must add all of them, not only the repo.
    ensure_import_paths()
    import v2_run
    return int(v2_run.eval_action_seed(int(C.EVAL_WORLD_SEED), int(scene), "greedy"))


def run_rule_world(layout, arm, scene, phase="pacing", cell=None):
    from . import scale_program, world
    # The real pre-world entrance must be independently bootstrap-safe in a fresh spawn.
    ensure_import_paths(layout)
    policy = build_rule_policy(layout, arm, int(scene))
    tag = "%s_s%d" % (arm, int(scene))
    if cell is not None:
        # A rule arm sees the same seed a learned arm would in this cell, so the two are
        # genuinely paired rather than merely similar.
        action_seed = scale_program.eval_seed(int(scene), cell) & ((1 << 63) - 1)
    else:
        action_seed = (eval_action_seed(int(scene))
                       if phase in ("eval", "pacing") else None)
    # The artifact namespace carries the logical slot (phase/arm/weight seed/scene). A rule
    # arm has no weight seed, so it stays "none"; the eval action seed is a physics argument
    # and never part of the directory identity. Under the scale programme the cell IS part of
    # the identity: two cells on one scene are different worlds and must not share a leaf.
    if cell is not None:
        # The namespace grammar is ``phase:arm:none:scene`` with no colons in the arm slot, so
        # the cell goes into that slot: two cells on one scene are different worlds and must
        # never share a leaf.
        cell_tag = scale_program.cell_label(cell).replace("_", "")
        namespace = "%s:%s_%s:none:%d" % (phase, arm, cell_tag, int(scene))
        label = "s5b_%s_s%d_%s" % (arm, int(scene), scale_program.cell_label(cell))
        tag = "%s_s%d_%s" % (arm, int(scene), scale_program.cell_label(cell))
    else:
        namespace = "%s:%s:none:%d" % (phase, arm, int(scene))
        label = "s5b_%s_s%d" % (arm, int(scene))
    row = world.run_episode(layout, label=label,
                            kind=("eval" if phase == "eval" else "rule"),
                            scene=int(scene), policy="learned", attach=policy,
                            rule=adapter_rule(arm), tag=tag,
                            action_rng_seed=action_seed,
                            artifact_namespace=namespace, cell=cell)
    row["arm"] = arm
    row["attach_kind"] = "rule"
    if action_seed is not None:
        row["eval_world_seed"] = int(C.EVAL_WORLD_SEED)
    row["rule_parameters"] = dict(policy.parameters())
    return row


def run_training_world(layout, arm, seed, index, scene, controller, action_rng_seed=None,
                       cell=None):
    """One training world inside a ``(G, R, N)`` cell.

    ``cell`` decides the goal count, the rescue roster and the subscriber fan-out, and the
    action seed is derived from ``(scene, cell, index)`` so a replay lands on the same world.
    """
    explicit_override = action_rng_seed is not None
    if explicit_override and (type(action_rng_seed) is not int
                           or not 0 <= action_rng_seed < 2 ** 64):
        raise ValueError("action_rng_seed override must be a non-bool uint64 integer")
    ensure_import_paths(layout)
    import v2_run
    from sb_collector import StageBCollector
    from . import scale_program, world
    collector = StageBCollector(controller)
    resolved_action_seed = (action_rng_seed if explicit_override
                            else (scale_program.episode_seed(int(scene), cell, int(index))
                                  if cell is not None
                                  else v2_run.action_seed(int(seed), int(index))))
    tag = "train_%s_s%d" % (arm, int(seed))
    row = world.run_episode(
        layout, label="s5b_train_%s_s%d_e%03d" % (arm, int(seed), int(index)),
        kind="train", scene=int(scene), policy="learned", attach=collector.policy(),
        controller=controller, collector=collector,
        action_rng_seed=resolved_action_seed,
        tag=tag + "_e%03d" % int(index),
        artifact_namespace="train:%s:%d:%03d" % (arm, int(seed), int(index)),
        cell=cell)
    row["scene"] = int(scene)
    row["action_rng_seed"] = resolved_action_seed
    if explicit_override:
        row["action_rng_seed_source"] = "EXPLICIT_OVERRIDE"
    elif cell is not None:
        row["action_rng_seed_source"] = "SCALE_PROGRAM"
    row["arm"] = arm
    row["training_seed"] = int(seed)
    row["episode_index"] = int(index)
    row["lambda"] = None   # set by the driver from its own dual state, not by a constant
    row["collector"] = collector
    row["controller"] = controller
    return row


def run_replay_world(layout, arm, seed, index, scene, controller):
    """One reproduction world of the frozen parent segment; never a science row.

    The action seed is the parent's own action_seed(seed, index) and the artifact namespace
    is a separate ``replay:`` leaf, so a replay world can never be confused with, or overwrite,
    a training or evaluation artefact.
    """
    ensure_import_paths(layout)
    import v2_run
    from sb_collector import StageBCollector
    from . import world
    collector = StageBCollector(controller)
    tag = "replay_%s_s%d" % (arm, int(seed))
    row = world.run_episode(
        layout, label="s5b_replay_%s_s%d_e%03d" % (arm, int(seed), int(index)),
        kind="replay", scene=int(scene), policy="learned", attach=collector.policy(),
        controller=controller, collector=collector,
        action_rng_seed=v2_run.action_seed(int(seed), int(index)),
        tag=tag + "_e%03d" % int(index),
        artifact_namespace="replay:%s:%d:%03d" % (arm, int(seed), int(index)))
    row["arm"] = arm
    row["training_seed"] = int(seed)
    row["episode_index"] = int(index)
    row["lambda"] = float(C.LAMBDA_FIXED)
    row["collector"] = collector
    row["controller"] = controller
    return row


def run_eval_world(layout, arm, seed, scene, controller, cell=None, episode_index=None):
    ensure_import_paths(layout)
    import v2_run
    from budget_masked import BudgetMaskedCollector
    from . import scale_program, world
    collector = BudgetMaskedCollector(controller, forced=lambda item: False, mask=False,
                                      greedy=True, budget_mask=True)
    if cell is not None:
        action_seed = scale_program.eval_seed(int(scene), cell) & ((1 << 63) - 1)
    else:
        action_seed = eval_action_seed(int(scene))
    label = "s5b_eval_%s_s%d" % (arm, int(scene))
    if cell is not None:
        label += "_" + scale_program.cell_label(cell)
    tag = "eval_%s_s%d" % (arm, int(scene))
    # The checkpoint index is part of the identity. Without it every checkpoint in one run
    # writes to the same leaf, so the second evaluation point fails with FileExistsError on
    # all 15 cells and reports 0/15 -- measured at episode 200 of train_v3.
    stamp = "" if episode_index is None else "@%d" % int(episode_index)
    namespace = "eval:%s%s:%d:%d" % (arm, stamp, int(seed), int(scene))
    if cell is not None:
        namespace = "eval:%s_%s%s:%d:%d" % (
            arm, scale_program.cell_label(cell).replace("_", ""), stamp, int(seed), int(scene))
    row = world.run_episode(layout, label=label,
                            kind="eval", scene=int(scene), policy="learned",
                            attach=collector.policy(), controller=controller,
                            collector=collector,
                            action_rng_seed=action_seed,
                            tag=tag,
                            artifact_namespace=namespace,
                            cell=cell)
    row["arm"] = arm
    row["training_seed"] = int(seed) if seed is not None else None
    row["eval_world_seed"] = int(C.EVAL_WORLD_SEED)
    return row


def exhaustion(row):
    """First budget refusal tick, or 1000 with right_censored when there was none."""
    refusals = [int(item[0]) for item in row.get("refusal_sequence", [])
                if str(item[2]) == "budget_exhausted"]
    relay_tick = min(refusals) if refusals else None
    mask_total = row.get("masked_by_credit_total")
    mask_tick = row.get("first_masked_credit_tick")
    if relay_tick is None:
        relay_fields = {"exhausted": False, "budget_exhaustion_tick": C.STEPS,
                        "right_censored": True}
    else:
        relay_fields = {"exhausted": True, "budget_exhaustion_tick": relay_tick,
                        "right_censored": False,
                        "true_1000_refusal": relay_tick == C.STEPS}
    candidates = [tick for tick in (relay_tick, mask_tick) if tick is not None]
    block = min(candidates) if candidates else None
    if relay_tick is not None and mask_tick is not None:
        source = "both_relay_and_mask"
    elif relay_tick is not None:
        source = "relay_refusal"
    elif mask_tick is not None:
        source = "credit_mask"
    elif mask_total is None:
        source = "no_credit_mask_observable"
    else:
        source = "none_observed"
    relay_fields.update({
        "first_relay_budget_refusal_tick": relay_tick,
        "masked_by_credit_total": mask_total,
        "first_masked_credit_tick": mask_tick,
        "budget_block_tick": (block if block is not None else C.STEPS),
        "budget_block_right_censored": block is None,
        "budget_block_source": source,
    })
    return relay_fields


def compact_row(row, *, phase, arm, seed, episode_index, scene, slot, state_hashes=None):
    """The per-episode record the analysis reads; no per-tick candidate body."""
    spent = int(row.get("spent_bytes", 0))
    generated = int(row.get("generated_goals", -1))
    placed = int(row.get("num_goals_placed", -1))
    configured = int(row.get("num_goals_configured", -1))
    discovered = int(row.get("discovered", 0))
    if generated < 1:
        raise RuntimeError("%s: no generated-goal denominator was recorded" % (slot,))
    if placed != generated:
        raise RuntimeError("%s: placed %d != generated %d" % (slot, placed, generated))
    if configured < generated:
        raise RuntimeError("%s: configured %d < generated %d"
                           % (slot, configured, generated))
    if discovered > generated:
        raise RuntimeError("%s: discovered %d > generated %d"
                           % (slot, discovered, generated))
    on_time_value = int(row.get("on_time", 0))
    counters = dict(row.get("relay_counters", {}))
    first = int(counters.get("first_emissions", 0))
    repeat = int(counters.get("refresh_emissions", 0))
    paid = int(counters.get("emissions", 0))
    if first + repeat != paid:
        raise RuntimeError("%s: first %d + repeat %d != paid emissions %d"
                           % (slot, first, repeat, paid))
    if int(row.get("emissions", paid)) != paid:
        raise RuntimeError("%s: transcript emissions %s != relay emissions %d"
                           % (slot, row.get("emissions"), paid))
    if int(counters.get("airtime_bytes", spent)) != spent:
        raise RuntimeError("%s: relay airtime %s != charged %d"
                           % (slot, counters.get("airtime_bytes"), spent))
    eval_seed = row.get("eval_world_seed")
    if phase in ("eval", "pacing") and eval_seed != C.EVAL_WORLD_SEED:
        raise RuntimeError("%s: %s row has eval_world_seed %r, expected %d"
                           % (slot, phase, eval_seed, C.EVAL_WORLD_SEED))
    payload = {
        "schema": C.ROW_SCHEMA,
        "phase": phase,
        "slot": slot,
        "arm": arm,
        "seed": int(seed) if seed is not None else None,
        "episode_index": int(episode_index) if episode_index is not None else None,
        "scene": int(scene),
        "sensing_mode": row.get("sensing_mode"),
        "initial_pool_k": row.get("initial_pool_k"),
        "on_time": on_time_value,
        "completed": int(row.get("completed", 0)),
        "expired": int(row.get("expired", 0)),
        "discovered": discovered,
        "generated_goals": generated,
        "num_goals_placed": placed,
        "num_goals_configured": configured,
        "undiscovered": generated - discovered,
        "success_rate": round(on_time_value / float(generated), 9),
        "active": int(row.get("active", 0)),
        "unassigned": int(row.get("unassigned", 0)),
        "charged_bytes": spent,
        "paid_bytes": spent,
        "airtime_bytes": spent,
        "budget_used_fraction": round(spent / float(C.BUDGET_BYTES), 6),
        "emissions": paid,
        "paid_emissions": paid,
        "first_report_count": first,
        "repeat_report_count": repeat,
        "first_report_share": (None if paid <= 0 else round(first / float(paid), 9)),
        "transmission_attempts": int(counters.get("transmission_attempts", 0)),
        "packets_lost": int(counters.get("packets_lost", 0)),
        "packets_delivered": int(counters.get("packets_delivered", 0)),
        "deliveries_applied": int(counters.get("deliveries_applied", 0)),
        "budget_refused_sends": int(counters.get("budget_refused_sends", 0)),
        "capacity_refused_sends": int(counters.get("capacity_refused_sends", 0)),
        "budget_bytes": int(C.BUDGET_BYTES),
        "budget_credit": dict(row.get("budget_credit") or {}),
        "budget_authority_before": dict(row.get("budget_authority_before") or {}),
        "budget_authority_after_exit": dict(row.get("budget_authority_after_exit") or {}),
        "budget_authority_restored": row.get("budget_authority_restored"),
        "eval_world_seed": (int(eval_seed) if eval_seed is not None else None),
        "masked_by_credit_total": row.get("masked_by_credit_total"),
        "masked_by_capacity_total": row.get("masked_by_capacity_total"),
        "first_masked_credit_tick": row.get("first_masked_credit_tick"),
        "steps": int(row.get("steps", 0)),
        "seconds": float(row.get("seconds", 0.0)),
        "config_hash": row.get("config_hash"),
        "behaviour_sha256": row.get("behaviour_sha256"),
        "energy_on_time_per_10kb": (None if spent <= 0
                                    else round(on_time_value / (spent / 10000.0), 9)),
        "artefacts": dict(row.get("artefacts", {})),
        "communication_network_observation": dict(
            row.get("communication_network_observation") or {}),
    }
    payload.update(exhaustion(row))
    if row.get("action_rng_seed_source") == "EXPLICIT_OVERRIDE":
        explicit_seed = row.get("action_rng_seed")
        if type(explicit_seed) is not int or not 0 <= explicit_seed < 2 ** 64:
            raise RuntimeError("%s: explicit action_rng_seed is not uint64" % (slot,))
        payload["action_rng_seed"] = explicit_seed
    if state_hashes:
        payload["state_hashes"] = dict(state_hashes)
    return payload

