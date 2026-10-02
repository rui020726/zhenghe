"""ROUND37 公共件：冻结的评估对象与额度、路径重绑定、原子 claim、身份与统计常量。

纪律：本轮**不训练、不评估以外的任何事**。 rollout 与决策全部走真实模块
（`v2_run.run_eval` / `v2_run.run_rule` / `v2_trainer.load_checkpoint_read_only` /
`integrated_sar` 的适配器与总线），本文件只做六件事：

1. 把 `v2_common` 的输出路径重绑定到本轮目录、启动上限绑定到 672 硬上限，使旧轮产物不被写入；
2. 固定九臂与 64 个新场景（占用审计之后，不看结局）；
3. 在首次环境启动之前把三个 ep128 模型的身份（路径、字节 SHA256、参数摘要、两个版本字段、
   四模块 digest、状态字段）读出来登记；
4. 提供跨进程原子 claim（O_CREAT|O_EXCL），保证一个 cell 只被启动一次；
5. 登记统计方案常量（场景升序、PCG64(20261002)、10000×64 共享索引、linear 分位）；
6. 记账工具（本轮自有账本 + 真实 START_LEDGER 对账）。
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
STAGE_C = ROOT / "runs" / "comm_cost_v2_rerun_20260928"
ROUND34 = ROOT / "runs" / "comm_task_reward_event_timing_20260930"
ROUND35 = ROOT / "runs" / "comm_reward_smoke_20260930"
ROUND36 = ROOT / "runs" / "comm_final_train_20261001"

if str(STAGE_C) not in sys.path:
    sys.path.insert(0, str(STAGE_C))

import v2_common  # noqa: E402  (the real Stage C common; this module only rebinds its globals)
import v2_controller  # noqa: E402
import v2_run  # noqa: E402
import v2_train  # noqa: E402
import v2_trainer  # noqa: E402
from v2_reward_timing import REWARD_TIMING_VERSION, TRAINING_VERSION  # noqa: E402

# ------------------------------------------------------------------ 冻结的评估对象 ----------
EXPECTED_BRANCH = "eval/comm-final-fixed-evaluation"
EXPECTED_HEAD = "e2a272c0303f580d957e778f7e222a56d265372f"      # ROUND36 交付头
STAGE_C_MODULES = ("v2_common.py", "v2_run.py", "v2_trainer.py", "v2_train.py",
                   "v2_reward_timing.py", "v2_check.py", "v2_controller.py", "v2_blocks.py",
                   "v2_rules.py")
TRAINING_SEEDS: tuple[int, ...] = (11, 22, 33)
MILESTONE_EPISODES = v2_common.EPISODES_PER_SEED            # 128：只评 ep128 里程碑
EXPECTED_UPDATES = MILESTONE_EPISODES // v2_common.EPISODES_PER_UPDATE   # 32 个更新批

#: 编号口径（授权边界 §1 要求先确认，避免把 0..127 写成 128 个更新批）
NUMBERING_CONVENTION = {
    "episodes": "episode_index 0..127 —— 128 次 rollout，每次 1000 步",
    "update_batches": "每 4 个 episode 一次 PPO + 对偶更新 ⇒ 32 批，批索引 0..31",
    "milestones": "ep0（登记初值）/ ep64 / ep128；只有 ep128 是本轮评估对象",
    "warning": "128 是 episode 数，不是更新批数；更新批数是 32"}

#: 新的 64 场景块：由 SCENE_USAGE_AUDIT.json 证明 1600000..1600063 在路径、跟踪文本、
#: 全 ref git 历史和 runs/ 之外都无占用；选择只看占用，不看任何结局。
SCENARIOS: tuple[int, ...] = tuple(range(1_600_000, 1_600_064))

#: 九臂。三个模型 × greedy、三个模型 × sampled、三条规则臂。
MODEL_ARMS: tuple[tuple[str, int, str], ...] = tuple(
    (f"model_s{seed}_{mode}", seed, mode)
    for seed in TRAINING_SEEDS for mode in ("greedy", "sampled"))
RULE_ARMS: tuple[str, ...] = ("immediate", "late_discovery_gate", "always_hold")
ALL_ARMS: tuple[str, ...] = tuple(item[0] for item in MODEL_ARMS) + RULE_ARMS

# ------------------------------------------------------------------ 额度 ---------------------
SCENARIO_COUNT = len(SCENARIOS)                      # 64
NORMAL_EVALUATION_LAUNCHES = len(ALL_ARMS) * SCENARIO_COUNT   # 9 x 64 = 576
RECORDING_RESERVE = 64               # 只有确需独立流录制才可用；本轮判定为不需要
FAILURE_RESERVE = 32                 # 技术失败储备（性能差不算技术失败）
HARD_CAP = NORMAL_EVALUATION_LAUNCHES + RECORDING_RESERVE + FAILURE_RESERVE   # 672
MAX_PARALLEL_WORKERS = 4
TRAINING_LAUNCHES_ALLOWED = 0
OPTIMIZER_STEPS_ALLOWED = 0

#: 冻结的描述性 bootstrap（授权边界 §6 的逐字段转写）
STATISTICS = {"scenarios_sorted": "ascending", "generator": "numpy.random.Generator(PCG64)",
              "pcg64_seed": 20261002, "draws": 10000, "shared_indices": SCENARIO_COUNT,
              "quantile_method": "linear", "statistic": "values[indices].mean(axis=1)",
              "interval": [2.5, 97.5]}

DECODER_SEMANTICS = {
    "greedy": "argmax with ties to SEND；实现 `BudgetMaskedCollector(greedy=True, mask=False, "
              "budget_mask=True, forced=lambda item: False)`（v2_run.run_eval）",
    "sampled": "与训练同一个 masked 多项式采样器：`BudgetMaskedCollector(greedy=False, "
               "mask=False, budget_mask=True, forced=lambda item: False)`",
    "action_rng": 'sha256("round24-eval:<seed>:<scenario>:<mode>")  '
                  "(v2_run.eval_action_seed, common.EVAL_RNG_LABEL 未改)",
    "never_mixed": "greedy 与 sampled 是同一权重的两种决策规则，分开报表，不合并成一个均值"}


def configure() -> dict[str, Path]:
    """Bind the real Stage C writer to this round and the hard cap to the real ledger."""
    paths = {"OUTPUT_DIR": HERE, "LOGS": HERE / "logs", "ROWS": HERE / "rows",
             "REJECTED": HERE / "rejected", "CHECKPOINTS": HERE / "checkpoints"}
    v2_common.OUTPUT_DIR = paths["OUTPUT_DIR"]
    v2_common.LOGS = paths["LOGS"]
    v2_common.ROWS = paths["ROWS"]
    v2_common.REJECTED = paths["REJECTED"]
    v2_common.CHECKPOINTS = paths["CHECKPOINTS"]
    v2_common.LAUNCH_CAP = HARD_CAP
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (HERE / "claims").mkdir(parents=True, exist_ok=True)
    return paths


# ------------------------------------------------------------------ 文件工具 -----------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name: str, payload: Any, *, refresh: bool = False) -> Path:
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = HERE / name
    if target.exists() and not refresh:
        raise RuntimeError(f"{name} already exists; refresh=True is only for live tallies")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8").stdout


# ------------------------------------------------------------------ 模型身份 -----------------
def freeze_identity() -> dict[str, Any]:
    """The four-field token the ROUND36 checkpoints were saved under (ROUND24's manifest)."""
    import v2_check

    saved = v2_common.OUTPUT_DIR
    try:
        v2_common.OUTPUT_DIR = STAGE_C
        record = v2_check.freeze_record()
    finally:
        v2_common.OUTPUT_DIR = saved
    record["token_provenance"] = (
        "ROUND36 saved its milestones with exactly this inherited token (see "
        "runs/comm_final_train_20261001/seed*/FREEZE_MANIFEST.json); the read-only loader "
        "requires it to match")
    return record


def stored_freeze_token(path: Path) -> dict[str, Any]:
    """Read the freeze block a checkpoint was actually saved under (no restore, no assumption)."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("freeze") or {})


def model_registry(*, verify_on_disk: bool = True) -> dict[str, Any]:
    """Register the three frozen final checkpoints before the first environment start."""
    import v2_train as train

    canonical = freeze_identity()
    canonical_core = {key: canonical[key] for key in
                      ("freeze_sha256", "config_hash", "branch_start", "head_at_freeze")}
    out: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        path = ROUND36 / f"seed{seed}" / "checkpoints" / f"v2_train_s{seed}_ep{MILESTONE_EPISODES}.pt"
        if not path.exists():
            raise SystemExit(f"model for seed {seed} is missing: {path}")
        entry: dict[str, Any] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                 "bytes_sha256": sha256_file(path)}
        if verify_on_disk:
            stored = stored_freeze_token(path)
            # the loader is the gate: hand it exactly the token the file carries, and separately
            # prove that token's four canonical fields are the inherited ROUND24 manifest values
            controller, state, identity = v2_trainer.load_checkpoint_read_only(path,
                                                                               freeze=stored)
            entry.update({
                "freeze_token_fields": sorted(stored),
                "freeze_token_canonical_fields_match_round24_manifest": all(
                    stored.get(key) == value for key, value in canonical_core.items()),
                "load_mode": identity["load_mode"],
                "training_version": identity["training_version"],
                "reward_timing_version": identity["reward_timing_version"],
                "timing_interpretation": identity["timing_interpretation"],
                "optimizer_restored": identity["optimizer_restored"],
                "rng_restored": identity["rng_restored"],
                "training_seed_in_state": int(state["training_seed"]),
                "episodes_done": int(state["episodes_done"]),
                "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda_final": float(state["lambda"]),
                "parameter_sha256": v2_controller.parameter_identity(controller)["parameter_sha256"],
                "module_digests": v2_train_module_digests(controller)})
            if int(state["episodes_done"]) != MILESTONE_EPISODES:
                raise SystemExit(f"seed {seed}: milestone holds {state['episodes_done']} episodes")
            if int(state["updates"]) != EXPECTED_UPDATES:
                raise SystemExit(f"seed {seed}: milestone holds {state['updates']} update "
                                 f"batches, expected {EXPECTED_UPDATES}")
            if identity["timing_interpretation"] != "versioned_capture_semantics":
                raise SystemExit(f"seed {seed}: checkpoint is not under the corrected reward "
                                 f"capture version: {identity}")
            out[f"s{seed}"] = entry
        else:
            out[f"s{seed}"] = entry
    cross = read_json(ROUND36 / "delivery_draft" / "ARTEFACT_HASH_REGISTRY.json")[
        "model_inventory"]
    for seed in TRAINING_SEEDS:
        recorded = cross[f"s{seed}"]["milestone_ep128"]
        out[f"s{seed}"]["matches_round36_registry"] = (
            out[f"s{seed}"]["bytes_sha256"] == recorded["bytes_sha256"]
            and out[f"s{seed}"]["parameter_sha256"] == recorded["parameter_sha256"]
            and out[f"s{seed}"]["module_digests"] == recorded["module_digests"])
    out["all_match_round36_registry"] = all(item["matches_round36_registry"] for item in
                                            out.values())
    all_match = bool(out.pop("all_match_round36_registry"))
    return {"models": out, "all_match_round36_registry": all_match,
            "expected_episodes": MILESTONE_EPISODES,
            "expected_update_batches": EXPECTED_UPDATES,
            "numbering_convention": NUMBERING_CONVENTION,
            "read_only_only": True,
            "rolling_checkpoints_used": False, "intermediate_milestones_used": False}


def v2_train_module_digests(controller: Any) -> dict[str, str]:
    return v2_train.module_digests(controller)


# ------------------------------------------------------------------ 作业与 claim ------------
def jobs() -> list[dict[str, Any]]:
    """The frozen 576-cell matrix, enumerated deterministically."""
    out: list[dict[str, Any]] = []
    index = 0
    for scenario in SCENARIOS:
        for arm in ALL_ARMS:
            out.append({"index": index, "scenario": scenario, "arm": arm})
            index += 1
    return out


def claim_path(index: int) -> Path:
    return HERE / "claims" / f"job_{index:04d}.claim"


def claim(index: int) -> bool:
    """Atomically take one cell; a second process or a re-run can never take it again."""
    path = claim_path(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        sink.write(json.dumps({"index": index, "pid": os.getpid(),
                               "claimed_at": time.strftime("%F %T")}) + "\n")
    return True


def claims_held() -> int:
    return len(list((HERE / "claims").glob("job_*.claim")))


def parse_arm(arm: str) -> tuple[str, int | None, str | None]:
    """`model_s11_greedy` -> ("model", 11, "greedy"); a rule name -> ("rule", None, None)."""
    if not arm.startswith("model_"):
        return "rule", None, None
    _, seed_token, mode = arm.split("_")
    return "model", int(seed_token.removeprefix("s")), mode


def row_exists(job: dict[str, Any]) -> bool:
    return row_path(job).exists()


def row_path(job: dict[str, Any]) -> Path:
    arm, scenario = job["arm"], job["scenario"]
    kind, seed, mode = parse_arm(arm)
    if kind == "model":
        return v2_common.ROWS / f"v2_eval_s{seed}_final_{mode}_s{scenario}_a0.json"
    return v2_common.ROWS / f"v2_{arm}_s{scenario}_a1.json"


# ------------------------------------------------------------------ 账本 ---------------------
BOOK = HERE / "EVALUATION_LAUNCH_BOOK.json"


def empty_book() -> dict[str, Any]:
    return {"starts": 0, "completed": 0, "technical_failures": 0, "reused_rows": 0,
            "recording_launches": 0, "training_launches": 0, "optimizer_steps": 0,
            "events": [], "written_at": time.strftime("%F %T")}


def book() -> dict[str, Any]:
    return read_json(BOOK) if BOOK.exists() else empty_book()


class _Lock:
    """Cross-process mutex for the shared book: O_CREAT|O_EXCL with a bounded wait."""

    def __init__(self, timeout: float = 120.0) -> None:
        self.path = HERE / "logs" / "_book.lock"
        self.timeout = timeout

    def __enter__(self) -> "_Lock":
        import time as _time

        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = _time.time()
        while True:
            try:
                self.handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                if _time.time() - started > self.timeout:
                    raise RuntimeError(f"book lock {self.path} stayed held past {self.timeout}s")
                _time.sleep(0.05)

    def __exit__(self, *_exc: Any) -> None:
        os.close(self.handle)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def book_event(kind: str, *, arm: str, scenario: int, ok: bool, detail: str,
               seconds: float) -> dict[str, Any]:
    """Book one cell start in this round's own book, under the cross-process lock.

    The authoritative record of a start is the Stage C per-PID ledger (written before the driver
    is constructed); this book is a convenience tally, so its correctness matters for reporting
    only. Without the lock, four parallel shards lose increments to read-modify-write races --
    exactly what happened in the first run of this round (registered in
    `ROUND37_HARNESS_DEFECTS.json`).
    """
    with _Lock():
        state = book()
        state["starts"] += 1
        state["events"].append({"kind": kind, "arm": arm, "scenario": int(scenario),
                                "ok": bool(ok), "pid": os.getpid(),
                                "seconds": round(float(seconds), 2),
                                "detail": detail[:300]})
        if ok:
            state["completed"] += 1
        else:
            state["technical_failures"] += 1
        if state["starts"] > HARD_CAP:
            raise RuntimeError(f"{state['starts']} starts exceed the hard cap {HARD_CAP}")
        write_json("EVALUATION_LAUNCH_BOOK.json", state, refresh=True)
        return state


def real_ledger_total() -> int:
    """Episode/cell starts as booked by the real Stage C ledger (failures included)."""
    return v2_common.ledger_total()


def bump_reused_rows() -> dict[str, Any]:
    """Count a skipped cell (row already on disk) without spending a launch, under the lock."""
    with _Lock():
        state = book()
        state["reused_rows"] += 1
        write_json("EVALUATION_LAUNCH_BOOK.json", state, refresh=True)
        return state
