"""ROUND35 smoke 公共件（0 环境启动即可导入）：路径、上限、身份与只读参照缓存。

设计纪律：本轮**不复制任何训练数学**。真实的 rollout、PPO 更新、对偶更新、checkpoint 保存与
恢复全部调用 ROUND24/ROUND34 的实际模块（`v2_run` / `v2_trainer` / `v2_train` /
`v2_reward_timing`）。本文件只做四件事：

1. 把 `v2_common` 的输出路径与启动上限**重绑定到本轮目录**，这样真实代码写下的每一份产物都落在
   smoke 目录里，ROUND24/32/33/34 的既有产物一个字节都不动（重绑定是模块全局赋值，不改文件）；
2. 记录本轮授权上限：2 次 runner 调用、16 次 episode 构造、100 次 optimizer.step；
3. 以只读方式取 ROUND24 的 64 份 immediate reference 行作为配对参照缓存（本轮不启动 reference）；
4. 登记代码身份：Stage C harness 当前哈希 vs ROUND34 交付时的哈希，并说明 checkpoint 里的 freeze
   token 仍是 ROUND24 manifest 的值（ROUND34 改了 harness，但沿用同一 token 作身份字符串）。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE_C = ROOT / "runs" / "comm_cost_v2_rerun_20260928"
ROUND32 = ROOT / "runs" / "comm_actor_reduction_ablation_20261001"
ROUND34 = ROOT / "runs" / "comm_task_reward_event_timing_20260930"

if str(STAGE_C) not in sys.path:
    sys.path.insert(0, str(STAGE_C))

import v2_common  # noqa: E402  (the real Stage C common; this module only rebinds its globals)
import v2_run  # noqa: E402
import v2_trainer  # noqa: E402
from v2_reward_timing import REWARD_TIMING_VERSION, TRAINING_VERSION  # noqa: E402

# ------------------------------------------------------------------ 授权上限 -----------------
MAX_LAUNCH_INVOCATIONS = 2
MAX_EPISODE_CONSTRUCTIONS = 16          # 启动①8 集连续 + 启动②4 集保存后再 4 集恢复
MAX_OPTIMIZER_STEPS = 100
SMOKE_SEED = 11
SMOKE_EPISODES = 8                       # 冻结数据顺序的前 8 个场景
#: 契约六类情形里，本场景块真实可达的四类（另两类见 PRE_SMOKE_CASE_COVERAGE.json 的登记）
EXPECTED_REACHABLE_CASES = ("same_tick", "cross_tick_new_action_next_tick",
                            "final_execution_tick", "deadline_expired_without_credit")
UNREACHABLE_IN_THIS_BLOCK = ("no_action_row_at_the_event_tick",
                             "completion_before_the_first_action")

STAGE_C_MODULES = ("v2_common.py", "v2_run.py", "v2_trainer.py", "v2_train.py",
                   "v2_reward_timing.py", "v2_recover.py", "v2_check.py", "v2_controller.py",
                   "v2_blocks.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(launch_tag: str, cap: int | None = None) -> dict[str, Path]:
    """Point the real Stage C writer at this round, inside one subdirectory per launch.

    Only the five path globals and the launch cap are touched; hyper-parameters, world config,
    scenario order, reward value and cost definition stay exactly as the frozen round defines them.
    `cap` lets a launch be bound to the constructions still available rather than to the whole
    round's total, so the real ledger is the thing that refuses an over-budget start.
    """
    base = HERE / launch_tag
    paths = {"OUTPUT_DIR": base, "LOGS": base / "logs", "ROWS": base / "rows",
             "REJECTED": base / "rejected", "CHECKPOINTS": base / "checkpoints"}
    v2_common.OUTPUT_DIR = paths["OUTPUT_DIR"]
    v2_common.LOGS = paths["LOGS"]
    v2_common.ROWS = paths["ROWS"]
    v2_common.REJECTED = paths["REJECTED"]
    v2_common.CHECKPOINTS = paths["CHECKPOINTS"]
    v2_common.LAUNCH_CAP = MAX_EPISODE_CONSTRUCTIONS if cap is None else int(cap)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ledger_starts_across_launches() -> dict[str, Any]:
    """Sum every episode start this round's own ledgers recorded, failures included.

    A failed start stays in the book: the quota is spent by the attempt to run the environment,
    not by a successful row.
    """
    per_launch: dict[str, Any] = {}
    total = 0
    failures: list[dict[str, Any]] = []
    for directory in sorted(HERE.glob("launch*/logs")):
        starts = 0
        for book in sorted(directory.glob("START_LEDGER_p*.json")):
            payload = json.loads(book.read_text(encoding="utf-8"))
            starts += int(payload.get("starts") or 0)
            for result in payload.get("results") or []:
                if not result.get("ok"):
                    failures.append({"launch": directory.parent.name, **result})
        per_launch[directory.parent.name] = starts
        total += starts
    return {"per_launch": per_launch, "total_starts": total, "failed_starts": failures,
            "remaining": MAX_EPISODE_CONSTRUCTIONS - total}


def freeze_identity() -> dict[str, Any]:
    """The same four-field token the real `v2_check.freeze_record()` produces from ROUND24.

    The Stage C manifest is read with its own output directory temporarily bound, so this is the
    archived identity string itself, not a re-derivation.
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
        "v2_check.freeze_record(); ROUND34 edited v2_run/v2_trainer and added v2_reward_timing "
        "without re-issuing that manifest, so this token is an inherited identity string, not a "
        "hash of the current files -- the current files are hashed separately in code_identity()")
    return record


def reference_cache() -> dict[str, Any]:
    """The 64 cached immediate reference rows, read only from ROUND24; no reference is launched."""
    cache: dict[int, dict[str, Any]] = {}
    used: dict[str, str] = {}
    for path in sorted((STAGE_C / "rows").glob("v2_reference_s*_a1.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = int(row["scenario_seed"])
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has two different cached references")
        cache[scenario] = row
        used[str(path.relative_to(ROOT))] = sha256_file(path)
    return {"by_scenario": cache, "files_read_only": used, "count": len(cache)}


def code_identity() -> dict[str, Any]:
    stage_c = {name: sha256_file(STAGE_C / name) for name in STAGE_C_MODULES
               if (STAGE_C / name).exists()}
    production = {name: sha256_file(ROOT / name) for name in v2_common.PRODUCTION_DEPENDENCIES}
    smoke = {path.name: sha256_file(path) for path in sorted(HERE.glob("r35_*.py"))}
    round34 = {path.name: sha256_file(path) for path in sorted(ROUND34.iterdir())
               if path.is_file() and path.suffix in (".py", ".md", ".json")}
    return {"stage_c_harness": stage_c, "production": production, "round35_scripts": smoke,
            "round34_deliverables": round34,
            "freeze_sha256_of_this_identity_block": hashlib.sha256(json.dumps(
                {"stage_c_harness": stage_c, "production": production}, sort_keys=True,
                ensure_ascii=False).encode("utf-8")).hexdigest(),
            "trainer_versions": {"trainer_tag": v2_trainer.TRAINER_TAG,
                                 "training_version": TRAINING_VERSION,
                                 "reward_timing_version": REWARD_TIMING_VERSION,
                                 "loss_version": v2_trainer.LOSS_VERSION,
                                 "state_schema": v2_trainer.STATE_SCHEMA,
                                 "gae_schema": v2_trainer.GAE_SCHEMA,
                                 "optimizer_schema": v2_trainer.OPTIMIZER_SCHEMA,
                                 "reward_timing_schema": v2_trainer.REWARD_TIMING_SCHEMA},
            "runner_provenance": {"run_episode_body_module": v2_run.__name__,
                                  "reward_collector_class": v2_run.
                                  PublishedProgressRewardCollector.__name__}}


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8").stdout


def archived_expectation(arm: str = "ALL_ROWS", seed: int = SMOKE_SEED,
                         episodes: tuple[int, ...] = (0, 1, 2, 3)) -> dict[str, Any]:
    """What the archived legacy run says the fixed run must reproduce, per episode.

    Batch 0 rolls episodes 0..3 out before any weight moves, so those four cells are replayable
    byte for byte: same Controller(seed), same registered action stream, same scenario, same world
    configuration. The only permitted difference is the task_stream key (one tick earlier) and the
    new event ledger.
    """
    order = v2_common.scenario_order()
    per_episode: dict[str, Any] = {}
    for episode in episodes:
        matches = sorted((ROUND32 / "rows").glob(
            f"r32_train_{arm}_s{seed}_e{episode}_s*_a{episode}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one archived row for {arm} s{seed} e{episode}")
        row = json.loads(matches[0].read_text(encoding="utf-8"))
        legacy_keys = sorted(int(key) for key in row["task_stream"])
        steps = [int(json.loads(line)["step"])
                 for line in (ROUND32 / "logs" /
                              f"{matches[0].stem}_tape.jsonl").read_text(
                              encoding="utf-8").splitlines() if line.strip()]

        def owner(tick: int) -> Any:
            earlier = [step for step in steps if step <= tick]
            return max(earlier) if earlier else "prefix"

        per_episode[f"e{episode}"] = {
            "archived_row_file": matches[0].name,
            "archived_row_file_sha256": sha256_file(matches[0]),
            "scenario_seed": int(row["scenario_seed"]),
            "frozen_order_scenario": order[episode],
            "scenario_matches_order": int(row["scenario_seed"]) == order[episode],
            "legacy_task_stream_keys": legacy_keys,
            "predicted_fixed_task_stream_keys": sorted(key - 1 for key in legacy_keys),
            "archived_behaviour_sha256": row["behaviour_sha256"],
            "archived_config_hash": row["config_hash"],
            "archived_emission_sequence": row["emission_sequence"],
            "archived_cost_stream": {str(k): v["airtime_bytes"]
                                     for k, v in sorted(row["cost_stream"].items(),
                                                        key=lambda item: int(item[0]))},
            "archived_relay_counters": row["relay_counters"],
            "archived_on_time": int(row["on_time"]),
            "archived_completed": int(row["completed"]),
            "archived_expired": int(row["expired"]),
            "archived_spent_bytes": int(row["spent_bytes"]),
            "archived_decision_rows": len(steps),
            "archived_owner_per_legacy_key": {str(key): owner(key) for key in legacy_keys},
            "archived_owner_per_predicted_key": {str(key - 1): owner(key - 1)
                                                 for key in legacy_keys},
            "legacy_keys_whose_owner_row_the_fix_moves": [str(key) for key in legacy_keys
                                                          if owner(key) != owner(key - 1)],
        }
    return {"arm": arm, "seed": seed, "episodes": per_episode}


def book_verification_optimizer_steps(steps: int, purpose: str) -> dict[str, Any]:
    """Book real optimizer.step calls made by a verification script, not by a launch.

    The smoke's authorisation counts every optimizer step against one ceiling of 100, including
    the steps a restore check performs while recomputing an update from a checkpoint.
    """
    book = smoke_book()
    book["verification_optimizer_steps"] = int(book.get("verification_optimizer_steps") or 0) + \
        int(steps)
    book["optimizer_steps"] = int(book.get("optimizer_steps") or 0) + int(steps)
    book["events"].append({"kind": "verification", "optimizer_steps": int(steps),
                           "pid": os.getpid(), "purpose": purpose})
    if book["optimizer_steps"] > MAX_OPTIMIZER_STEPS:
        raise RuntimeError(f"booking {steps} more steps would exceed the authorised ceiling of "
                           f"{MAX_OPTIMIZER_STEPS} (running total "
                           f"{book['optimizer_steps']})")
    write_json("SMOKE_LAUNCH_BOOK.json", book, refresh=True)
    return book


def digest_of(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json(name: str, payload: dict[str, Any], *, refresh: bool = False) -> Path:
    """This round's artefact writer: refuses to rewrite a delivered file unless asked."""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = HERE / name
    if target.exists() and not refresh:
        raise RuntimeError(f"{name} already exists; pass refresh=True only for a live tally")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def smoke_book() -> dict[str, Any]:
    path = HERE / "SMOKE_LAUNCH_BOOK.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "invocations": 0, "episode_constructions": 0, "optimizer_steps": 0, "events": []}


def book_invocation(launch_tag: str, *, episode_constructions: int, optimizer_steps: int,
                    detail: dict[str, Any]) -> dict[str, Any]:
    book = smoke_book()
    book["invocations"] += 1
    book["episode_constructions"] += episode_constructions
    book["optimizer_steps"] += optimizer_steps
    book["events"].append({"launch_tag": launch_tag, "pid": os.getpid(),
                           "episode_constructions": episode_constructions,
                           "optimizer_steps": optimizer_steps, "detail": detail})
    if book["invocations"] > MAX_LAUNCH_INVOCATIONS:
        raise RuntimeError("the smoke's authorised invocation count has been exceeded")
    if book["episode_constructions"] > MAX_EPISODE_CONSTRUCTIONS:
        raise RuntimeError("the smoke's authorised episode-construction count has been exceeded")
    if book["optimizer_steps"] > MAX_OPTIMIZER_STEPS:
        raise RuntimeError("the smoke's authorised optimizer-step ceiling has been exceeded")
    write_json("SMOKE_LAUNCH_BOOK.json", book, refresh=True)
    return book
