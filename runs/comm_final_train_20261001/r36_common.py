"""ROUND36 公共件：输出路径重绑定、授权额度与账本、只读参照缓存、归档基准与代码身份。

纪律与 ROUND35 相同且更严：本轮**不复制任何训练数学**。rollout、PPO 更新、对偶更新、
checkpoint 保存与恢复全部调用 ROUND24+ROUND34 的真实模块（`v2_run` / `v2_trainer` /
`v2_train` / `v2_reward_timing`）。本文件只做四件事：

1. 把 `v2_common` 的五个输出路径全局与启动上限**重绑定到"每 seed 一个子目录"**，这样真实
   代码写下的每一份产物都落在本轮目录内，ROUND24/32/33/34/35 的既有产物一个字节都不动；
2. 登记本轮授权额度（3 次训练启动 / 384 次 episode 构造 / 1 次备用重跑 / 每 seed 的
   wall-clock 上限取自 ROUND32 归档同 seed 实际耗时的 1.5 倍）并在超限时拒绝；
3. 以只读方式取 ROUND24 的 64 份 immediate reference 作配对参照（本轮不启动任何 reference）；
4. 登记代码身份与归档基准，使"训练量与 ROUND32 冻结 Stage C 完全一致"成为可核对的断言。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE_C = ROOT / "runs" / "comm_cost_v2_rerun_20260928"      # 真实训练代码所在（只读）
ROUND32 = ROOT / "runs" / "comm_actor_reduction_ablation_20261001"
ROUND34 = ROOT / "runs" / "comm_task_reward_event_timing_20260930"
ROUND35 = ROOT / "runs" / "comm_reward_smoke_20260930"
STAGE_B = ROOT / "runs" / "comm_cost_stage_b_20260927"
STAGE_A = ROOT / "runs" / "comm_cost_controller_stage_a_20260927"

if str(STAGE_C) not in sys.path:
    sys.path.insert(0, str(STAGE_C))

import v2_common  # noqa: E402  (the real Stage C common; this module only rebinds its globals)
import v2_run  # noqa: E402
import v2_trainer  # noqa: E402
from v2_reward_timing import REWARD_TIMING_VERSION, TRAINING_VERSION  # noqa: E402

# ------------------------------------------------------------------ 授权额度 -----------------
MAX_TRAINING_LAUNCHES = 3                 # 每次 1 个 seed，三个 seed 均从初值训练
MAX_RESERVE_RERUNS = 1                    # 全部 seed 合计最多 1 次重跑，须先登记失败原因
SEEDS: tuple[int, ...] = tuple(v2_common.TRAIN_SEEDS)          # (11, 22, 33) 来自冻结 Stage C
EPISODES_PER_SEED = v2_common.EPISODES_PER_SEED                 # 128
BATCHES_PER_SEED = EPISODES_PER_SEED // v2_common.EPISODES_PER_UPDATE   # 32
MAX_EPISODE_CONSTRUCTIONS = len(SEEDS) * EPISODES_PER_SEED      # 384
RESERVE_CONSTRUCTIONS = EPISODES_PER_SEED                       # 只在动用储备时额外放行
WALL_CLOCK_MULTIPLIER = 1.5               # 单次启动 ≤ ROUND32 同 seed 归档实际耗时 × 1.5
ROUND32_FREEZE_HEAD = "df176bdf8c3c265d8109494446073b23aee384a6"
ROUND33_BASELINE = "c7d47d25e638cbe3728cf36e3a22b9cfc563a353"
EXPECTED_HEAD_AT_START = "577344aab8a5c9bfcf75e7ffbf6f77e503dbc200"
EXPECTED_BRANCH = "fix/comm-task-reward-event-timing"

STAGE_C_MODULES = ("v2_common.py", "v2_run.py", "v2_trainer.py", "v2_train.py",
                   "v2_reward_timing.py", "v2_recover.py", "v2_check.py", "v2_controller.py",
                   "v2_blocks.py", "v2_rules.py")
#: 训练路径实际 import 的归档模块（本轮不修改，只 import）；身份必须逐字节登记。
IMPORTED_TRAINING_PATH_MODULES = (
    STAGE_B / "sb_common.py", STAGE_B / "sb_controller.py", STAGE_B / "sb_collector.py",
    STAGE_B / "sb_run.py", STAGE_A / "cost_math.py",
    ROOT / "runs" / "comm_budget_pilot_20260919" / "train_pilot.py",
    ROOT / "runs" / "comm_budget_pilot_20260919" / "pilot_common.py",
    ROOT / "runs" / "comm_budget_pilot_20260919" / "pilot_env.py",
    ROOT / "runs" / "comm_budget_action_mask_fix_20260922" / "budget_masked.py",
    ROOT / "runs" / "comm_budget_action_mask_fix_20260922" / "train_budget_masked.py",
    ROOT / "runs" / "comm_tight_budget_20260919" / "onpolicy.py",
    ROOT / "runs" / "comm_tight_budget_20260919" / "train_onpolicy.py",
    ROOT / "runs" / "search_trace_and_policy_20260925" / "sp_common.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name: str, payload: Any, *, refresh: bool = False) -> Path:
    """This round's artefact writer: refuses to rewrite a delivered file unless asked."""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = HERE / name
    if target.exists() and not refresh:
        raise RuntimeError(f"{name} already exists; pass refresh=True only for a live tally")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8").stdout


# ------------------------------------------------------------------ 路径绑定 -----------------
def seed_directory(training_seed: int, *, reserve: bool = False) -> Path:
    """One artefact directory per training launch; a reserve re-run gets its own directory."""
    name = f"seed{training_seed}" + ("_reserve" if reserve else "")
    return HERE / name


def configure(training_seed: int, *, reserve: bool = False) -> dict[str, Path]:
    """Point the real Stage C writer at this seed's own subdirectory.

    Only the five path globals and the launch cap are touched; hyper-parameters, world config,
    scenario order, reward value and cost definition stay exactly as the frozen round defines
    them. The per-directory cap is the planned episode count for that seed, so the real ledger
    inside the directory is what refuses a 129th construction.
    """
    base = seed_directory(training_seed, reserve=reserve)
    paths = {"OUTPUT_DIR": base, "LOGS": base / "logs", "ROWS": base / "rows",
             "REJECTED": base / "rejected", "CHECKPOINTS": base / "checkpoints"}
    v2_common.OUTPUT_DIR = paths["OUTPUT_DIR"]
    v2_common.LOGS = paths["LOGS"]
    v2_common.ROWS = paths["ROWS"]
    v2_common.REJECTED = paths["REJECTED"]
    v2_common.CHECKPOINTS = paths["CHECKPOINTS"]
    v2_common.LAUNCH_CAP = EPISODES_PER_SEED
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def starts_in(training_seed: int, *, reserve: bool = False) -> dict[str, Any]:
    """Episode starts this seed's own ledger recorded, failed starts included (no refunds)."""
    base = seed_directory(training_seed, reserve=reserve)
    starts = 0
    results: list[dict[str, Any]] = []
    ledger_dir = base / "logs"
    if ledger_dir.exists():
        for book in sorted(ledger_dir.glob("START_LEDGER_p*.json")):
            payload = read_json(book)
            starts += int(payload.get("starts") or 0)
            results.extend(payload.get("results") or [])
    return {"directory": base.name, "starts": starts,
            "failed": [item for item in results if not item.get("ok")]}


def round_constructions() -> dict[str, Any]:
    """Every episode construction this round's own ledgers booked, across all seed directories."""
    per_directory: dict[str, int] = {}
    failed: list[dict[str, Any]] = []
    for directory in sorted(HERE.glob("seed*")):
        if not directory.is_dir():
            continue
        reserve = directory.name.endswith("_reserve")
        seed = int(directory.name.removeprefix("seed").removesuffix("_reserve"))
        book = starts_in(seed, reserve=reserve)
        per_directory[directory.name] = book["starts"]
        failed.extend({"directory": directory.name, **item} for item in book["failed"])
    total = sum(per_directory.values())
    allowed = MAX_EPISODE_CONSTRUCTIONS + (RESERVE_CONSTRUCTIONS if reserve_used() else 0)
    return {"per_directory": per_directory, "total": total, "allowed_now": allowed,
            "failed_starts": failed, "remaining": allowed - total}


# ------------------------------------------------------------------ 启动账本 -----------------
BOOK = HERE / "TRAIN_LAUNCH_BOOK.json"


def empty_book() -> dict[str, Any]:
    return {"training_launches": 0, "reserve_reruns": 0, "episode_constructions": 0,
            "optimizer_steps": 0, "reserve_used": False, "events": [], "failures": []}


def book() -> dict[str, Any]:
    return read_json(BOOK) if BOOK.exists() else empty_book()


def reserve_used() -> bool:
    return bool(book().get("reserve_used"))


def book_launch(*, tag: str, training_seed: int, constructions: int, optimizer_steps: int,
                wall_clock_seconds: float, ceiling_seconds: float, reserve: bool = False,
                detail: dict[str, Any]) -> dict[str, Any]:
    """Register one finished training launch against the authorised quota.

    The quota is spent by the attempt to run the environment, not by a successful row, so a
    failed launch is booked exactly like a productive one and stays booked.
    """
    state = book()
    if reserve:
        if state["reserve_used"]:
            raise RuntimeError("the single authorised reserve re-run is already used; refusing "
                               "a further re-run")
        if not state["failures"]:
            raise RuntimeError("a reserve re-run requires a registered failure reason first")
        state["reserve_reruns"] += 1
    state["training_launches"] += 1
    state["episode_constructions"] += int(constructions)
    state["optimizer_steps"] += int(optimizer_steps)
    state["reserve_used"] = bool(state["reserve_used"] or reserve)
    state["events"].append({"tag": tag, "training_seed": int(training_seed), "pid": os.getpid(),
                            "episode_constructions": int(constructions),
                            "optimizer_steps": int(optimizer_steps),
                            "wall_clock_seconds": round(float(wall_clock_seconds), 2),
                            "wall_clock_ceiling_seconds": round(float(ceiling_seconds), 2),
                            "wall_clock_within_ceiling": wall_clock_seconds <= ceiling_seconds,
                            "reserve": bool(reserve), "detail": detail})
    limit = MAX_EPISODE_CONSTRUCTIONS + (RESERVE_CONSTRUCTIONS if state["reserve_used"] else 0)
    if state["training_launches"] > MAX_TRAINING_LAUNCHES + MAX_RESERVE_RERUNS:
        raise RuntimeError(f"only {MAX_TRAINING_LAUNCHES} training launches plus "
                           f"{MAX_RESERVE_RERUNS} reserve re-run are authorised")
    if state["episode_constructions"] > limit:
        raise RuntimeError(f"episode constructions {state['episode_constructions']} exceed the "
                           f"authorised {limit}")
    write_json("TRAIN_LAUNCH_BOOK.json", state, refresh=True)
    return state


def record_failure(*, training_seed: int, reason_class: str, summary: str,
                   evidence: dict[str, Any]) -> dict[str, Any]:
    """A failed seed must be classified (被测系统缺陷 / 执行方操作 / 环境) before any re-run."""
    allowed = ("subject_system_defect", "executor_operation", "environment")
    if reason_class not in allowed:
        raise RuntimeError(f"failure reason class must be one of {allowed}")
    state = book()
    state["failures"].append({"training_seed": int(training_seed), "reason_class": reason_class,
                              "summary": summary, "recorded_at": time.strftime("%F %T"),
                              "evidence": evidence})
    write_json("TRAIN_LAUNCH_BOOK.json", state, refresh=True)
    return state


# ------------------------------------------------------------------ 身份与参照 -----------------
def freeze_identity() -> dict[str, Any]:
    """The four-field token the real `v2_check.freeze_record()` produces from ROUND24's manifest.

    The Stage C manifest is read with its own output directory temporarily bound, so this is the
    archived identity string itself, not a re-derivation. It is inherited (ROUND34 edited the
    Stage C harness without re-issuing that manifest); the current files are hashed separately
    by `code_identity()`.
    """
    import v2_check

    saved = v2_common.OUTPUT_DIR
    try:
        v2_common.OUTPUT_DIR = STAGE_C
        record = v2_check.freeze_record()
    finally:
        v2_common.OUTPUT_DIR = saved
    record["token_provenance"] = (
        "read from runs/comm_cost_v2_rerun_20260928/CONFIG_AND_MODEL_MANIFEST.json through "
        "v2_check.freeze_record(); it is an inherited identity string, not a hash of the current "
        "files -- the current files are hashed in code_identity() and the two version fields are "
        "recorded in FROZEN_CONFIG_MANIFEST.json")
    record["training_version"] = TRAINING_VERSION
    record["reward_timing_version"] = REWARD_TIMING_VERSION
    return record


def reference_cache() -> dict[str, Any]:
    """The 64 cached immediate reference rows, read only from ROUND24; no reference is launched."""
    cache: dict[int, dict[str, Any]] = {}
    used: dict[str, str] = {}
    for path in sorted((STAGE_C / "rows").glob("v2_reference_s*_a1.json")):
        row = read_json(path)
        scenario = int(row["scenario_seed"])
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has two different cached references")
        cache[scenario] = row
        used[str(path.relative_to(ROOT))] = sha256_file(path)
    order = v2_common.scenario_order()
    needed = sorted(set(order))
    return {"by_scenario": cache, "files_read_only": used, "count": len(cache),
            "unique_scenarios_in_frozen_order": len(needed),
            "missing": sorted(set(needed) - set(cache)),
            "order_sha256": hashlib.sha256(json.dumps(order).encode("utf-8")).hexdigest()}


def code_identity() -> dict[str, Any]:
    stage_c = {name: sha256_file(STAGE_C / name) for name in STAGE_C_MODULES
               if (STAGE_C / name).exists()}
    production = {name: sha256_file(ROOT / name) for name in v2_common.PRODUCTION_DEPENDENCIES}
    imported = {_rel(path): sha256_file(path) for path in IMPORTED_TRAINING_PATH_MODULES
                if path.exists()}
    round36 = {path.name: sha256_file(path) for path in sorted(HERE.glob("r36_*.py"))}
    return {"stage_c_harness": stage_c, "production": production, "imported_training_modules":
            imported, "round36_scripts": round36,
            "trainer_versions": {"trainer_tag": v2_trainer.TRAINER_TAG,
                                 "training_version": TRAINING_VERSION,
                                 "reward_timing_version": REWARD_TIMING_VERSION,
                                 "loss_version": v2_trainer.LOSS_VERSION,
                                 "state_schema": v2_trainer.STATE_SCHEMA,
                                 "gae_schema": v2_trainer.GAE_SCHEMA,
                                 "optimizer_schema": v2_trainer.OPTIMIZER_SCHEMA,
                                 "reward_timing_schema": v2_trainer.REWARD_TIMING_SCHEMA},
            "frozen_checkpoints": {str(path.relative_to(ROOT)).replace("\\", "/"):
                                   sha256_file(path) for path in v2_common.CHECKPOINT_FILES},
            "runner_reward_collector": v2_run.PublishedProgressRewardCollector.__name__,
            "freeze_sha256_of_this_identity_block": hashlib.sha256(json.dumps(
                {"stage_c_harness": stage_c, "production": production, "imported": imported},
                sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()}


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


# ------------------------------------------------------------------ 归档基准 -----------------
def archived_baseline(arm: str = "ALL_ROWS") -> dict[str, Any]:
    """What ROUND32 actually spent per seed: constructions, rollout seconds, batches, steps.

    ROUND32 recorded no per-launch wall-clock, only a per-episode `seconds` field, so the
    archived rollout sum is the closest measure of "同 seed 归档实际耗时"; the per-round
    update overhead is registered separately from this round's own measurement.
    """
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        rows = sorted((ROUND32 / "rows").glob(f"r32_train_{arm}_s{seed}_e*_s*_a*.json"))
        seconds = [float(read_json(path)["seconds"]) for path in rows]
        batches = sorted((ROUND32 / "checkpoints").glob(f"r32_batch_{arm}_s{seed}_b*.json"),
                         key=lambda path: int(path.stem.split("_b")[-1]))
        records = [read_json(path) for path in batches]
        init = records[0]["controller_parameters_before"]["parameter_sha256"] if records else None
        final = records[-1]["controller_parameters_after"]["parameter_sha256"] if records else None
        per_seed[f"s{seed}"] = {
            "rows": len(rows),
            "rollout_seconds_sum": round(sum(seconds), 2),
            "rollout_seconds_mean": round(sum(seconds) / len(rows), 4) if rows else None,
            "batches": len(records),
            "optimizer_steps": sum(int(item["update"]["optimizer_steps"]) for item in records),
            "samples": sum(int(item["update"]["samples"]) for item in records),
            "lambda_first": records[0]["dual"]["lambda_next"] if records else None,
            "lambda_last": records[-1]["dual"]["lambda_next"] if records else None,
            "lambda_history": [item["dual"]["lambda_next"] for item in records],
            "g_hat_history": [item["dual"]["g_hat"] for item in records],
            "parameters_before_batch0": init,
            "parameters_after_last_batch": final,
            "on_time_total": sum(int(read_json(path)["on_time"]) for path in rows),
            "spent_bytes_total": sum(int(read_json(path)["spent_bytes"]) for path in rows),
        }
    return {"arm": arm, "source": str(ROUND32.name), "per_seed": per_seed,
            "wall_clock_multiplier": WALL_CLOCK_MULTIPLIER,
            "note": "read-only from ROUND32's own rows and batch records; nothing here is "
                    "recomputed or re-run"}


def wall_clock_ceiling(training_seed: int, baseline: dict[str, Any] | None = None) -> float:
    baseline = baseline or archived_baseline()
    archived = float(baseline["per_seed"][f"s{training_seed}"]["rollout_seconds_sum"])
    return round(archived * WALL_CLOCK_MULTIPLIER, 2)


def fresh_controller_hash(training_seed: int) -> str:
    """The registered initialisation of this seed, computed without constructing an environment."""
    from v2_controller import fresh_controller, parameter_identity

    return parameter_identity(fresh_controller(int(training_seed)))["parameter_sha256"]


def milestone_path(training_seed: int, episodes: int) -> Path:
    return seed_directory(training_seed) / "checkpoints" / f"v2_train_s{training_seed}_ep{episodes}.pt"


def write_seed_json(training_seed: int, name: str, payload: Any, *, reserve: bool = False,
                    refresh: bool = False) -> Path:
    """One per-seed record file, kept inside that seed's own artefact directory."""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the seed directory: {name!r}")
    target = seed_directory(training_seed, reserve=reserve) / name
    if target.exists() and not refresh:
        raise RuntimeError(f"{name} already exists in seed{training_seed}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def checkpoint_version_fields(path: Path) -> dict[str, Any]:
    """Read the two version fields out of a real .pt without restoring anything."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("training_state") or {}
    return {"file": path.name, "bytes_sha256": sha256_file(path),
            "training_version": payload.get("training_version"),
            "reward_timing_version": payload.get("reward_timing_version"),
            "in_training_state_training_version": state.get("training_version"),
            "in_training_state_reward_timing_version": state.get("reward_timing_version"),
            "episodes_done": state.get("episodes_done"), "updates": state.get("updates"),
            "optimizer_steps": state.get("optimizer_steps"), "lambda": state.get("lambda"),
            "both_fields_complete": bool(payload.get("training_version")
                                         and payload.get("reward_timing_version")
                                         and state.get("training_version")
                                         and state.get("reward_timing_version"))}
