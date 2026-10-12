"""ROUND68 corrected evaluation adapter: late-initial-confidence ablation.

Two learned model arms on the already-used 1,800,000..1,800,031 development block (NOT a
confirmation set):
  * INIT_ALPHA    - the fresh ROUND68 controller (R60 ep0 with the one-time late_head 0.10 scale),
                   UNTRAINED. Built via the injected cfg["frozen_fresh"] so it scales exactly once.
  * TRAINED_ALPHA - the ROUND68 ep128 checkpoint (late-only training from the scaled init).

PRIMARY statistic: TRAINED_ALPHA - INIT_ALPHA (per seed, per mode, scene-paired, R58-frozen
bootstrap). This asks whether lowering the late head's initial confidence lets the trainable late
head produce ANY additional learning effect. greedy is primary; sampled is reported separately;
three seeds are NOT pooled.

CONTEXT (read-only reuse from ROUND66, NOT new ROUND68 constructions): ROUND66 NEW = R66 ep128
(frozen early path, late-only training) is the historical TRAINING comparator; ROUND66 BC = R60 ep0
is the initial-value comparator. Rows are identified by ROUND66 claim eligibility + row SHA, never
by the literal "OLD" label (ROUND66 OLD is the compliant R60 ep128, a different thing).

Budget: 384 model-arm eval + 6 focal = 390 of the 774 normal plan; reserve 26; hard cap 800.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_execute as base       # injects the ROUND68 runtime/controller/checkpoint seams
import r68_frozen as r68         # ROUND68 read-only loaders + identity

SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES_NEW = tuple(range(1_800_000, 1_800_032))
FOCAL_SCENE = 1_700_027
MODEL_ARMS = ("INIT_ALPHA", "TRAINED_ALPHA")
ROW_TAG = "r68eval"
R66_ROW_TAG = "r66eval2"
EVAL_VERSION = r68.R68_TRAINING_VERSION + "+r68-eval/1"
NORMAL_CAP, RESERVE_CAP, HARD_CAP = base.NORMAL_CAP, base.RESERVE_CAP, base.HARD_CAP
# No deviation is booked at plan time, so the eval reserve ceiling equals the inherited reserve.
EVAL_RESERVE_CAP = RESERVE_CAP
UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS = 0


def eval_job_order():
    """Every ROUND68 evaluation cell in the exact order the executor requests it. Order matches
    r68_execute.build_plan's eval+focal enumeration so planned slots align (they are enumerations,
    not bindings; the real slot comes from r40_common._reserve_slot)."""
    for arm in MODEL_ARMS:
        for seed in SEEDS:
            for mode in MODES:
                for scene in SCENES_NEW:
                    yield {"scope": "model_eval", "arm": arm, "seed": int(seed), "mode": mode,
                           "scene": int(scene)}
    for seed in SEEDS:
        for arm in MODEL_ARMS:
            yield {"scope": "focal", "arm": arm, "seed": int(seed), "mode": "greedy",
                   "scene": FOCAL_SCENE}


def logical_job(cell: dict[str, Any]) -> str:
    if cell["scope"] == "model_eval":
        return f"{ROW_TAG}_model_{cell['arm'].lower()}_s{cell['seed']}_{cell['mode']}_scene{cell['scene']}"
    return f"{ROW_TAG}_focal_{cell['arm'].lower()}_s{cell['seed']}_scene{cell['scene']}"


def build_eval_plan() -> dict[str, Any]:
    cells = list(eval_job_order())
    jobs = []
    for index, cell in enumerate(cells):
        jobs.append({"scope": cell["scope"], "arm": cell["arm"], "seed": cell["seed"],
                     "mode": cell["mode"], "scene": cell["scene"],
                     "logical_job": logical_job(cell), "planned_normal_slot": 385 + index})
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["scope"]] = counts.get(j["scope"], 0) + 1
    if counts != {"model_eval": 384, "focal": 6} or len(jobs) != 390:
        raise AssertionError(f"ROUND68 eval plan is not 384+6=390 unique jobs: {counts}")
    if len({j["logical_job"] for j in jobs}) != 390:
        raise AssertionError("ROUND68 eval plan contains duplicate logical jobs")
    planned = [j["planned_normal_slot"] for j in jobs]
    if planned != list(range(385, 775)):
        raise AssertionError("planned normal slots are not the contiguous range 385..774")
    if 384 + 390 != NORMAL_CAP:
        raise AssertionError("384 training + 390 eval cells != the 774 normal cap")
    return {"kind": "round68-eval-plan/1", "task_id": base.TASK_ID, "branch": base.BRANCH,
            "eval_version": EVAL_VERSION, "training_version": r68.R68_TRAINING_VERSION,
            "matrix": {"model_eval": 384, "focal": 6, "total": 390},
            "already_numbered_cells": 384,
            "already_numbered_note": "384 training episodes (no block-A build; 48 immediate "
                                     "references reused read-only from ROUND66)",
            "model_arms": list(MODEL_ARMS),
            "primary_comparison": "TRAINED_ALPHA_minus_INIT_ALPHA",
            "context_reuse": {"r66_new": "ROUND66 ep128 (frozen early, late-only training)",
                              "r66_bc": "ROUND60 ep0 initial value",
                              "note": "read-only from ROUND66; NOT new ROUND68 constructions"},
            "eval_scenes": list(SCENES_NEW), "focal_scene": FOCAL_SCENE,
            "execution_order": ["model_eval", "focal"],
            "slot_allocation": {"mechanism": "r40_common._reserve_slot takes the lowest free number",
                                "planned_field": "planned_normal_slot is a plan enumeration, NOT a binding",
                                "planned_range": [385, 774]},
            "normal_cap": NORMAL_CAP, "reserve_cap": EVAL_RESERVE_CAP, "hard_cap": HARD_CAP,
            "unregistered_extra_environment_constructions": UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS,
            "row_namespace": {"tag_prefix": ROW_TAG,
                              "row_globs": [f"v2_{ROW_TAG}_model_*.json", f"v2_{ROW_TAG}_focal_*.json"],
                              "disjoint_from": [f"v2_{R66_ROW_TAG}_*.json", "v2_r66_*.json"]},
            "jobs": jobs}


def configure_r68_eval(raw_root: Path) -> dict[str, Any]:
    """Bind the live ROUND68 runtime (via r68_execute.configure_and_inject) and prove the caps on
    the LIVE modules before any builder can run."""
    cfg = base.configure_and_inject(raw_root)
    r40 = cfg["runtime"]["r40_common"]
    common = cfg["common"]
    r40.RESERVE_CAP = EVAL_RESERVE_CAP
    common.RESERVE_LAUNCHES = EVAL_RESERVE_CAP
    r40.plan = lambda: {"normal_total": NORMAL_CAP, "technical_reserve_max": EVAL_RESERVE_CAP,
                        "hard_cap": HARD_CAP, "task_id": base.TASK_ID,
                        "unregistered_extra_environment_constructions":
                            UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS}
    if int(r40.RESERVE_CAP) != EVAL_RESERVE_CAP or int(common.RESERVE_LAUNCHES) != EVAL_RESERVE_CAP:
        raise RuntimeError("ROUND68-EVAL reserve cap did not bind on the live modules")
    if int(r40.NORMAL_CAP) != NORMAL_CAP or int(r40.HARD_CAP) != HARD_CAP:
        raise RuntimeError("ROUND68-EVAL normal/hard cap drifted from the frozen ROUND68 quota")
    guard = quota_guard(cfg)
    cfg["r68_eval_quota_guard"] = guard
    return cfg


def quota_guard(cfg: dict[str, Any]) -> dict:
    """The real hard-cap judgement: booked slots + any unregistered extra construction <= 800."""
    r40 = cfg["runtime"]["r40_common"]
    snapshot = r40.ledger_snapshot()  # itself raises if a live cap is already exceeded
    normal_used = int(snapshot["normal_used"])
    reserve_used = int(snapshot["reserve_used"])
    booked = normal_used + reserve_used
    actual = booked + UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS
    hard = int(r40.HARD_CAP)
    if normal_used > NORMAL_CAP:
        raise RuntimeError(f"ROUND68-EVAL: booked normal {normal_used} exceeds the normal cap {NORMAL_CAP}")
    if reserve_used > EVAL_RESERVE_CAP:
        raise RuntimeError(f"ROUND68-EVAL: booked reserve {reserve_used} exceeds ceiling {EVAL_RESERVE_CAP}")
    if actual > hard:
        raise RuntimeError(f"ROUND68-EVAL: actual environment constructions {actual} exceed the hard cap {hard}")
    return {"kind": "round68-eval-quota-guard/1", "eval_version": EVAL_VERSION,
            "booked_normal": normal_used, "booked_reserve": reserve_used, "booked_total": booked,
            "unregistered_extra_environment_constructions": UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS,
            "actual_environment_constructions": actual, "normal_cap": NORMAL_CAP,
            "reserve_cap": EVAL_RESERVE_CAP, "hard_cap": hard, "hard_headroom": hard - actual,
            "historical_non_plan_probe": int(base.HISTORICAL_PROBE),
            "live_bindings": {"r40_RESERVE_CAP": int(r40.RESERVE_CAP),
                              "common_RESERVE_LAUNCHES": int(cfg["common"].RESERVE_LAUNCHES),
                              "r40_NORMAL_CAP": int(r40.NORMAL_CAP), "r40_HARD_CAP": int(r40.HARD_CAP)}}


class AlreadyComplete(RuntimeError):
    pass


def _receipt_is_sound(claim: dict, cell: dict | None) -> bool:
    """A complete claim only counts when its durable result re-proves from disk."""
    receipt = claim.get("result_receipt")
    if not isinstance(receipt, dict):
        return False
    row_path = receipt.get("row_path")
    if not row_path:
        return False
    try:
        path = Path(row_path)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if base._sha256_file(path) != receipt.get("row_sha256"):
            return False
        row = base._read_json(path)
        if base._sha256_file(path) != receipt.get("row_sha256"):
            return False
    except Exception:
        return False
    for key, value in (receipt.get("companions") or {}).items():
        try:
            if not isinstance(value, dict) or not value.get("path"):
                return False
            companion = Path(value["path"])
            if not companion.is_file() or companion.stat().st_size == 0:
                return False
            if int(companion.stat().st_size) != int(value.get("bytes", -1)):
                return False
            if base._sha256_file(companion) != value.get("sha256"):
                return False
        except Exception:
            return False
    for entry in receipt.get("declared_but_not_produced") or []:
        try:
            if entry.get("path") and Path(entry["path"]).exists():
                return False
        except Exception:
            return False
    if cell is None:
        return True
    try:
        if str(row.get("model_arm") or "").upper() != str(cell["arm"]).upper():
            return False
        if str(row.get("eval_mode")) != str(cell["mode"]):
            return False
        if int(row.get("scenario_seed", -1)) != int(cell["scene"]):
            return False
        if cell.get("seed") is not None and int(row.get("training_seed", -1)) != int(cell["seed"]):
            return False
        if int(row.get("steps", -1)) <= 0:
            return False
    except Exception:
        return False
    return True


def _cell_complete(cfg: dict[str, Any], logical: str, cell: dict | None = None) -> bool:
    """True only when SOME attempt is complete AND its durable result re-proves from disk. Any
    unreadable claim, missing receipt, missing/empty row, hash or identity mismatch fails closed."""
    r40 = cfg["runtime"]["r40_common"]
    directory = r40.ROUND_DIR / "launch_claims" / r40._slug(logical)
    if not directory.is_dir():
        return False
    for path in sorted(directory.glob("attempt_*.json")):
        try:
            claim = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if claim.get("status") != "complete":
            continue
        if _receipt_is_sound(claim, cell):
            return True
    return False


ARTEFACT_KEYS = ("row", "lifecycle", "transcript", "tape")


def persist_result(cfg, row, run_tag: str) -> dict:
    """Write the result row and prove the durable artifacts exist, before anything is committed.
    Raises on any problem so the caller marks the attempt failed_technical instead of committing a
    complete claim with no result behind it."""
    base._write_row(cfg, row)
    artefacts = row.get("artefacts") or {}
    if not artefacts.get("row"):
        raise RuntimeError(f"{run_tag}: result row declares no row artefact")
    row_path = Path(artefacts["row"])
    if not row_path.is_file() or row_path.stat().st_size == 0:
        raise RuntimeError(f"{run_tag}: result row is missing or empty at {row_path}")
    first = base._sha256_file(row_path)
    reread = base._read_json(row_path)
    if int(reread.get("steps", -1)) <= 0:
        raise RuntimeError(f"{run_tag}: persisted row records no steps")
    if int(reread.get("on_time", -1)) != int(row.get("on_time", -2)):
        raise RuntimeError(f"{run_tag}: persisted row does not read back consistently")
    if base._sha256_file(row_path) != first:
        raise RuntimeError(f"{run_tag}: persisted row hash is not stable across re-reads")
    companions, not_produced, present = {}, [], []
    for key in ARTEFACT_KEYS:
        if key == "row":
            continue
        declared = artefacts.get(key)
        if not declared:
            not_produced.append({"key": key, "reason": "not declared by this row"})
            continue
        path = Path(declared)
        if not path.is_file():
            not_produced.append({"key": key, "path": str(path),
                                 "reason": "declared by the runner but never written: this run "
                                           "kind attached no collector"})
            continue
        if path.stat().st_size == 0:
            raise RuntimeError(f"{run_tag}: artefact {key} is empty at {path}")
        companions[key] = {"path": str(path), "sha256": base._sha256_file(path),
                           "bytes": int(path.stat().st_size), "status": "present"}
        present.append(key)
    if not present:
        raise RuntimeError(f"{run_tag}: no companion artefact was produced at all")
    return {"row_path": str(row_path), "row_sha256": first,
            "row_bytes": int(row_path.stat().st_size),
            "companions": companions, "declared_but_not_produced": not_produced,
            "run_tag": run_tag, "persisted_at_unix": time.time()}


def _safe_atomic_run(cfg, logical, metadata, run_fn, *, detail, tag, persist=None, cell=None):
    """ROUND68 evaluation booking.

    Delegates to the SHARED base._r68_atomic_run, so training and evaluation go through ONE
    claimed-but-started-safe helper and ONE construction-permit regime instead of two booking paths
    that can drift apart (ROUND68_FREEZE_ADDENDUM_2.json). Behaviour preserved:
      * a complete attempt with a sound receipt is skipped (AlreadyComplete);
      * a complete attempt with an UNSOUND receipt is retried on a NEW reserve attempt, never
        reused (base allow_complete_retry=True; base itself refuses a free reuse of any attempt
        that already committed to an environment construction);
      * persist still runs INSIDE the success path, before finish_launch, so a complete claim
        always has a re-provable result;
      * the per-attempt run tag stays <tag>_a<attempt> and is stamped on the claim.
    """
    r40 = cfg["runtime"]["r40_common"]
    r42_run = cfg["r42_run"]
    if r42_run._ACTIVE_CLAIM is not None:
        raise RuntimeError("nested ROUND68 evaluation claims are not allowed")
    directory = r40.ROUND_DIR / "launch_claims" / r40._slug(logical)
    prior = []
    if directory.is_dir():
        for path in sorted(directory.glob("attempt_*.json")):
            try:
                prior.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                prior.append({"attempt": None, "status": "unreadable"})
    if any(c.get("status") == "complete" and _receipt_is_sound(c, cell) for c in prior):
        raise AlreadyComplete(f"{logical}: a complete attempt with a sound receipt already exists")

    holder = {"run_tag": f"{tag}_a1", "claim_path": None, "attempt": None}

    def _on_claim(claim_path, attempt):
        holder["run_tag"] = f"{tag}_a{int(attempt)}"
        holder["claim_path"] = claim_path
        holder["attempt"] = int(attempt)
        r40._update_claim(claim_path, row_tag=holder["run_tag"], eval_version=EVAL_VERSION)

    def _wrapped():
        row = run_fn(holder["run_tag"])
        if persist is not None:
            receipt = persist(row, holder["run_tag"])
            r40._update_claim(holder["claim_path"], result_receipt=receipt)
        if isinstance(row, dict):
            row["claim_attempt"] = int(holder["attempt"])
            row["claim_path"] = str(holder["claim_path"])
        return row

    return base._r68_atomic_run(cfg, logical, metadata, _wrapped, detail=detail,
                                on_claim=_on_claim, allow_complete_retry=True)


def _load_arm(cfg, seed: int, arm: str):
    """Read-only arm controllers.
    INIT_ALPHA: the fresh ROUND68 scaled-ep0 controller, UNTRAINED, built via the injected
                cfg["frozen_fresh"] so the late-head 0.10 scale happens EXACTLY ONCE (calling
                r68.fresh_r68_controller here would re-enter the injected vc.fresh_controller and
                double-scale). TRAINED_ALPHA: the ROUND68 ep128, read-only (no optimizer/RNG)."""
    import r42_one_shot  # noqa: F401  (ensures the frozen collector module is importable)
    common = cfg["common"]
    seed = int(seed)
    if arm == "INIT_ALPHA":
        controller = cfg["frozen_fresh"](seed)
        if any(p.requires_grad for p in r68.frozen_parameters(controller)):
            raise RuntimeError(f"seed {seed}: INIT_ALPHA lost its frozen mask")
        ident_path = Path(common.CHECKPOINTS) / f"r68_init_identity_s{seed}.json"
        ident = base._read_json(ident_path) if ident_path.is_file() else {}
        if ident and not bool(ident.get("pre_scale_equals_r60_ep0")):
            raise RuntimeError(f"seed {seed}: INIT_ALPHA pre-scale is not the compliant R60 ep0")
        identity = {"load_mode": "fresh_init_alpha", "structure": r68.R68_STRUCTURE,
                    "late_init_alpha": r68.LATE_INIT_ALPHA,
                    "parameter_identity": r68.frozen_parameter_summary(controller),
                    "init_identity": ident}
        return controller, identity
    if arm == "TRAINED_ALPHA":
        ep128 = Path(common.CHECKPOINTS) / f"v2_train_s{seed}_ep128.pt"
        if not ep128.is_file():
            raise FileNotFoundError(f"seed {seed}: ROUND68 ep128 checkpoint missing")
        controller, state, identity = r68.load_checkpoint_read_only(ep128, freeze=base.R68_FREEZE)
        if int(state.get("episodes_done", -1)) != base.EPISODES_PER_SEED:
            raise RuntimeError(f"seed {seed}: ROUND68 ep128 is not {base.EPISODES_PER_SEED} episodes")
        if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
            raise RuntimeError(f"seed {seed}: TRAINED_ALPHA read-only eval restored training state")
        if any(p.requires_grad for p in r68.frozen_parameters(controller)):
            raise RuntimeError(f"seed {seed}: TRAINED_ALPHA ep128 lost its frozen mask")
        return controller, identity
    raise ValueError(f"unknown ROUND68 arm {arm}")


VISITED: list = []


def eval_model_arms(cfg, plan) -> dict:
    common = cfg["common"]; runner = cfg["runner"]
    import r42_one_shot
    for cell in eval_job_order():
        if cell["scope"] != "model_eval":
            continue
        seed, arm, mode, scene = int(cell["seed"]), cell["arm"], cell["mode"], int(cell["scene"])
        logical = logical_job(cell)
        VISITED.append(logical)
        if _cell_complete(cfg, logical, cell):
            continue
        controller, identity = _load_arm(cfg, seed, arm)
        collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
        tag = f"{ROW_TAG}_model_{arm.lower()}_s{seed}_{mode}"
        metadata = {"arm": arm, "training_seed": seed, "mode": mode, "scenario_seed": scene,
                    "eval_version": EVAL_VERSION}

        def _do(run_tag, controller=controller, collector=collector, tag=tag, scene=scene,
                mode=mode, arm=arm, identity=identity, seed=seed):
            row = runner._run_episode(run_tag, "eval", scene, policy="learned",
                                      attach=collector.policy(), controller=controller,
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(seed, scene, mode),
                                      tag=run_tag, checkpoint_load_identity=identity)
            row["model_arm"] = arm; row["eval_mode"] = mode; row["training_seed"] = seed
            return row

        _safe_atomic_run(cfg, logical, metadata, _do,
                         detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}",
                         tag=tag, persist=lambda r, t: persist_result(cfg, r, t), cell=cell)
    return {"status": "model_arms_eval_complete", "arms": list(MODEL_ARMS), "cells": 384}


def eval_focal(cfg, plan) -> dict:
    runner = cfg["runner"]
    import r42_one_shot
    for cell in eval_job_order():
        if cell["scope"] != "focal":
            continue
        seed, arm = int(cell["seed"]), cell["arm"]
        logical = logical_job(cell)
        VISITED.append(logical)
        if _cell_complete(cfg, logical, cell):
            continue
        controller, identity = _load_arm(cfg, seed, arm)
        collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=True)
        tag = f"{ROW_TAG}_focal_{arm.lower()}_s{seed}"
        metadata = {"arm": arm, "training_seed": seed, "mode": "greedy",
                    "scenario_seed": FOCAL_SCENE, "block": "focal", "eval_version": EVAL_VERSION}

        def _do(run_tag, controller=controller, collector=collector, tag=tag, arm=arm,
                identity=identity, seed=seed):
            row = runner._run_episode(run_tag, "eval", FOCAL_SCENE, policy="learned",
                                      attach=collector.policy(), controller=controller,
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(seed, FOCAL_SCENE, "greedy"),
                                      tag=run_tag, checkpoint_load_identity=identity)
            row["model_arm"] = arm; row["eval_mode"] = "greedy"; row["training_seed"] = seed
            row["focal"] = True
            return row

        _safe_atomic_run(cfg, logical, metadata, _do,
                         detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}",
                         tag=tag, persist=lambda r, t: persist_result(cfg, r, t), cell=cell)
    return {"status": "focal_complete", "cells": 6}


def _assert_visited_order(plan) -> dict:
    expected = [j["logical_job"] for j in plan["jobs"]]
    if VISITED != expected:
        for i, (a, b) in enumerate(zip(VISITED, expected)):
            if a != b:
                raise AssertionError(f"execution order diverges from the plan at cell {i}: ran {a}, plan {b}")
        raise AssertionError(f"visited {len(VISITED)} cells, plan has {len(expected)}")
    return {"kind": "round68-eval-order-check/1", "status": "passed", "cells": len(VISITED),
            "note": "the executor visited every planned cell in exactly the planned order"}


def _bootstrap_index():
    import hashlib
    gen = np.random.Generator(np.random.PCG64(20261012))
    idx = gen.integers(0, 32, size=(10000, 32))
    return idx, hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()


def on_time_target_ids(row: dict, scene: int) -> list:
    """The evidenced (scene, target_id) on-time set, rebuilt from the row's own task events. A real
    row has no on_time_ids field; an "or []" fallback would fabricate an empty set. Closed against
    row['on_time'] two ways; anything inconsistent raises rather than writing []."""
    on_time = row.get("on_time")
    if not isinstance(on_time, int) or isinstance(on_time, bool):
        raise RuntimeError(f"scene {scene}: row has no integer on_time count")
    events = row.get("task_reward_events")
    if events is None:
        events = []
    if not isinstance(events, list):
        raise RuntimeError(f"scene {scene}: task_reward_events is {type(events).__name__}, not a list")
    if not events:
        if int(on_time) == 0:
            return []
        raise RuntimeError(f"scene {scene}: row reports on_time={on_time} but carries no task_reward_events")
    ids, on_time_events = set(), 0
    for event in events:
        if not isinstance(event, dict):
            raise RuntimeError(f"scene {scene}: malformed task reward event {event!r}")
        if event.get("on_time") is not True:
            continue
        on_time_events += 1
        task_id = event.get("task_id")
        if not task_id:
            raise RuntimeError(f"scene {scene}: on-time event without a task_id")
        ids.add(str(task_id))
    if len(ids) != int(on_time) or on_time_events != int(on_time):
        raise RuntimeError(f"scene {scene}: {len(ids)} ids and {on_time_events} events but on_time={on_time}")
    return sorted(ids)


def _paired_target_diff(new_ids_of, other_ids_of, scenes):
    """Per-(scene, target_id) differences, so a same-named target in two scenes cannot cancel."""
    lost, gained = [], []
    for scene in scenes:
        new_ids = set(new_ids_of(scene))
        other_ids = set(other_ids_of(scene))
        lost += [{"scene": int(scene), "target_id": t} for t in sorted(other_ids - new_ids)]
        gained += [{"scene": int(scene), "target_id": t} for t in sorted(new_ids - other_ids)]
    return lost, gained


def _qualified_row_paths(raw_root: Path, row_tag: str) -> dict:
    """The only admissible row source is a row NAMED BY a complete claim whose receipt re-proves."""
    claims_root = Path(raw_root) / "launch_claims"
    out: dict = {}
    if not claims_root.is_dir():
        return out
    for directory in sorted(p for p in claims_root.iterdir() if p.is_dir()):
        for path in sorted(directory.glob("attempt_*.json")):
            try:
                claim = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if claim.get("status") != "complete":
                continue
            logical = claim.get("logical_job")
            receipt = claim.get("result_receipt")
            if not logical or not isinstance(receipt, dict) or not receipt.get("row_path"):
                continue
            if not _receipt_is_sound(claim, None):
                continue
            resolved = str(Path(receipt["row_path"]).resolve())
            bucket = out.setdefault(str(logical), [])
            if resolved not in bucket:
                bucket.append(resolved)
    return out


def _load_model_rows(raw_root: Path, row_tag: str, arms=None):
    """Model-eval rows from one raw root, keyed (ARM, seed, mode, scene), each from a single sound
    receipt. Works for ROUND68's own rows and for read-only ROUND66 context rows."""
    raw_root = Path(raw_root)
    qualified = _qualified_row_paths(raw_root, row_tag)
    on_disk = {str(q.resolve()) for q in (raw_root / "rows").glob(f"v2_{row_tag}_model_*.json")}
    named = {r for paths in qualified.values() for r in paths
             if Path(r).name.startswith(f"v2_{row_tag}_model_")}
    rows, audit = {}, {"raw_root": str(raw_root), "row_tag": row_tag,
                       "qualified_logical_jobs": 0, "row_files_on_disk": len(on_disk),
                       "orphan_row_files_excluded_from_statistics": sorted(on_disk - named)}
    for logical in sorted(qualified):
        if not logical.startswith(f"{row_tag}_model_"):
            continue
        paths = qualified[logical]
        if len(paths) != 1:
            raise RuntimeError(f"{logical}: {len(paths)} sound complete receipts name different rows "
                               f"({paths}); refusing to choose one by order")
        row = base._read_json(paths[0])
        arm = str(row["model_arm"]).upper()
        if arms is not None and arm not in arms:
            continue
        key = (arm, int(row["training_seed"]), str(row["eval_mode"]), int(row["scenario_seed"]))
        if key in rows:
            raise RuntimeError(f"{logical}: row key {key} is already owned by {rows[key]['logical_job']}")
        rows[key] = {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                     "on_time_ids": on_time_target_ids(row, int(row["scenario_seed"])),
                     "row_path": paths[0], "logical_job": logical}
        audit["qualified_logical_jobs"] += 1
    return rows, audit


def _load_r68_model_rows(raw_root: Path):
    return _load_model_rows(raw_root, ROW_TAG)


def _load_r66_context(arm: str):
    """Read-only ROUND66 context rows for one arm (NEW = R66 ep128 late-only; BC = R60 ep0),
    identified by ROUND66 claim eligibility + row SHA, never by the literal label."""
    r66_raw = base.R66_DIR / "raw"
    rows, audit = _load_model_rows(r66_raw, R66_ROW_TAG, arms={arm.upper()})
    return rows, audit


def _paired_delta(a_of, b_of, scenes, idx, label_a, label_b):
    d_task = np.array([a_of(s)["on_time"] - b_of(s)["on_time"] for s in scenes], dtype=np.float64)
    d_cost = np.array([a_of(s)["spent_bytes"] - b_of(s)["spent_bytes"] for s in scenes], dtype=np.float64)
    lost, gained = _paired_target_diff(lambda s: a_of(s)["on_time_ids"],
                                       lambda s: b_of(s)["on_time_ids"], scenes)
    return {
        f"task_delta_{label_a}_minus_{label_b}_mean": float(d_task.mean()),
        f"task_ci95_{label_a}_minus_{label_b}": [
            float(np.quantile(d_task[idx].mean(axis=1), 0.025)),
            float(np.quantile(d_task[idx].mean(axis=1), 0.975))],
        f"cost_delta_{label_a}_minus_{label_b}_B_per_scene": float(d_cost.mean()),
        f"cost_ci95_{label_a}_minus_{label_b}_B_per_scene": [
            float(np.quantile(d_cost[idx].mean(axis=1), 0.025)),
            float(np.quantile(d_cost[idx].mean(axis=1), 0.975))],
        f"scenes_{label_a}_better_task_than_{label_b}": int((d_task > 0).sum()),
        f"scenes_{label_a}_worse_task_than_{label_b}": int((d_task < 0).sum()),
        f"scenes_{label_a}_cheaper_than_{label_b}": int((d_cost < 0).sum()),
        f"targets_lost_{label_a}_vs_{label_b}": lost,
        f"targets_gained_{label_a}_vs_{label_b}": gained,
        f"targets_lost_{label_a}_vs_{label_b}_count": len(lost),
        f"targets_gained_{label_a}_vs_{label_b}_count": len(gained),
    }


def compute_stats(raw_root: Path) -> dict:
    idx, idx_sha = _bootstrap_index()
    r68_rows, r68_audit = _load_r68_model_rows(raw_root)
    r66_new, r66_new_audit = _load_r66_context("NEW")
    r66_bc, r66_bc_audit = _load_r66_context("BC")
    scenes = list(SCENES_NEW)
    per: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            def cell(arm, s, seed=seed, mode=mode):
                return r68_rows.get((arm, seed, mode, s))
            missing = [(a, s) for a in MODEL_ARMS for s in scenes if cell(a, s) is None]
            if missing:
                raise RuntimeError(f"missing ROUND68 paired cells seed={seed} mode={mode}: "
                                   f"{missing[:4]} ({len(missing)})")
            entry: dict[str, Any] = {}
            for arm in MODEL_ARMS:
                entry[f"{arm}_on_time_total"] = int(sum(cell(arm, s)["on_time"] for s in scenes))
                entry[f"{arm}_bytes_per_scene"] = float(np.mean([cell(arm, s)["spent_bytes"] for s in scenes]))
            # PRIMARY: does late-only training from the lowered init change anything vs the init?
            entry["primary"] = _paired_delta(lambda s: cell("TRAINED_ALPHA", s),
                                             lambda s: cell("INIT_ALPHA", s), scenes, idx,
                                             "TRAINED_ALPHA", "INIT_ALPHA")
            # CONTEXT (read-only ROUND66): TRAINED_ALPHA vs R66 ep128 (late-only); INIT vs R60 ep0.
            def r66_cell(store, arm, s, seed=seed, mode=mode):
                return store.get((arm, seed, mode, s))
            if all(r66_cell(r66_new, "NEW", s) is not None for s in scenes):
                entry["context_trained_vs_r66_new"] = _paired_delta(
                    lambda s: cell("TRAINED_ALPHA", s), lambda s: r66_cell(r66_new, "NEW", s),
                    scenes, idx, "TRAINED_ALPHA", "R66_NEW_ep128")
            else:
                entry["context_trained_vs_r66_new"] = {"unavailable": "R66 NEW rows incomplete for this cell"}
            if all(r66_cell(r66_bc, "BC", s) is not None for s in scenes):
                entry["context_init_vs_r66_bc"] = _paired_delta(
                    lambda s: cell("INIT_ALPHA", s), lambda s: r66_cell(r66_bc, "BC", s),
                    scenes, idx, "INIT_ALPHA", "R66_BC_r60ep0")
            else:
                entry["context_init_vs_r66_bc"] = {"unavailable": "R66 BC rows incomplete for this cell"}
            per[f"{seed}/{mode}"] = entry
    return {"kind": "round68-statistics/1", "eval_version": EVAL_VERSION,
            "training_version": r68.R68_TRAINING_VERSION, "late_init_alpha": r68.LATE_INIT_ALPHA,
            "primary_comparison": "TRAINED_ALPHA_minus_INIT_ALPHA",
            "primary_meaning": ("R68 ep128 (late-only training from the 0.10-scaled init) minus the "
                                "R68 fresh scaled-ep0 init, UNTRAINED. Isolates whether lowering the "
                                "late head's initial confidence lets the trainable late head produce "
                                "any additional learning effect. NOT an added-learning gain claim vs R60."),
            "context_comparisons": {
                "TRAINED_ALPHA_minus_R66_NEW_ep128": "R68 ep128 vs ROUND66 ep128 (both late-only "
                    "training; only the init scale differs). Identified by eligibility/SHA, not label.",
                "INIT_ALPHA_minus_R66_BC_r60ep0": "R68 scaled-ep0 init vs ROUND60 ep0 (untrained)."},
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "eval_block": "1800000..1800031 (already-used development block, NOT a confirmation set)",
            "eval_rng_note": ("eval action RNG is runner.eval_action_seed(seed, scene, mode), shared "
                              "across arms and with ROUND66 for the same (seed, scene, mode); arms "
                              "differ only by model, so comparisons are scene-paired"),
            "on_time_target_id_source": {
                "field_used": "task_reward_events[].task_id where on_time is true",
                "closure": "distinct on-time ids == on-time event count == row['on_time'], per cell",
                "pairing": "differences reported per (scene, target_id)"},
            "bootstrap": {"rng": "PCG64", "seed": 20261012, "draws": 10000, "index_sha256": idx_sha,
                          "quantile": "numpy linear 0.025/0.975"},
            "row_source_audit": {"r68_model": r68_audit, "r66_new_context": r66_new_audit,
                                 "r66_bc_context": r66_bc_audit,
                                 "admission_rule": "every admitted row is named by a complete claim "
                                                   "whose result_receipt re-proves from disk"},
            "per_seed_mode": per, "r68_cells": len(r68_rows),
            "note": ("CI containing 0 is not equivalence. Single frozen policy per arm, single "
                     "development block, three seeds not pooled, one random realisation per cell. "
                     "Results support only the whole late-initial-confidence structure on this block; "
                     "they do not prove saturation is the only root cause or support deployment.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 evaluation adapter")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("plan", "eval", "stats"), required=True)
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    if args.phase == "plan":
        plan = build_eval_plan()
        base._write_json(raw_root / "ROUND68_EVAL_PLAN.json", plan)
        print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}, indent=1))
        return 0
    cfg = configure_r68_eval(raw_root)
    base._write_json(raw_root / "runtime" / "ROUND68_EVAL_QUOTA_GUARD.json", cfg["r68_eval_quota_guard"])
    if args.phase == "eval":
        plan = base._read_json(raw_root / "ROUND68_EVAL_PLAN.json")
        VISITED.clear()
        print(json.dumps(eval_model_arms(cfg, plan), indent=1), flush=True)
        print(json.dumps(eval_focal(cfg, plan), indent=1), flush=True)
        order = _assert_visited_order(plan)
        base._write_json(raw_root / "runtime" / "ROUND68_EVAL_ORDER_CHECK.json", order)
        print(json.dumps(order, indent=1), flush=True)
        base._write_json(raw_root / "runtime" / "ROUND68_EVAL_QUOTA_GUARD.json", quota_guard(cfg))
        return 0
    stats = compute_stats(raw_root)
    base._write_json(raw_root / "ROUND68_STATISTICS.json", stats)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
