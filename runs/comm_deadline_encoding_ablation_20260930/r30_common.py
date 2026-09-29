"""ROUND30 公共件：路径、启动台账（并行安全）、冻结常量引用、编码臂身份与统计方案。

对应执行依据 §4-§6。与 ROUND24 的登记关系：训练/评估常量、PPO 超参、目标函数常量、
场景块、训练种子、数据顺序算法全部**从 v2_common 引用**（唯一事实来源），本文件只声明
ROUND30 自己的启动额度、臂定义、输出路径与统计方案，不重新定义任何训练数学。

台账：每个 episode 启动先记账再构造 driver；本轮的 6 条训练按 (arm, seed) 并行，故
reserve/record 用跨进程互斥锁文件保护临界区（读总数 + 写我的 pid 账本），硬上限在
reserve 时判定；纯合成更新不构造环境、不计启动、另行分账。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# The shared archived rounds' directories must be importable before v2_common itself; v2_common's
# own enter_round() then installs the rest of the path set.
for _shared in ("runs/comm_cost_v2_rerun_20260928", "runs/comm_cost_stage_b_20260927",
                "runs/comm_budget_pilot_20260919", "runs/comm_budget_action_mask_fix_20260922",
                "runs/comm_tight_budget_20260919", "runs/search_trace_and_policy_20260925",
                "runs/comm_deadline_encoding_audit_20260929"):
    _path = ROOT / _shared
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v2_common as v2c  # noqa: E402  (importing v2_common also installs the shared sys.path)

OUTPUT_DIR = ROOT / "runs" / "comm_deadline_encoding_ablation_20260930"

LOGS = OUTPUT_DIR / "logs"
ROWS = OUTPUT_DIR / "rows"
REJECTED = OUTPUT_DIR / "rejected"
CHECKPOINTS = OUTPUT_DIR / "checkpoints"

# ------------------------------------------------------------------ 从 ROUND24 冻结件引用的常量
TRAINING_SCENARIOS = v2c.TRAINING_SCENARIOS          # 1300000..1300047（48）
DEVELOPMENT_SCENARIOS = v2c.DEVELOPMENT_SCENARIOS    # 1301000..1301015（16）
TRAIN_SEEDS = v2c.TRAIN_SEEDS                        # (11, 22, 33)
EPISODES_PER_SEED = v2c.EPISODES_PER_SEED            # 128
STEPS_PER_EPISODE = v2c.STEPS_PER_EPISODE            # 1000
EPISODES_PER_UPDATE = v2c.EPISODES_PER_UPDATE        # 4
DATA_ORDER_SEED = v2c.DATA_ORDER_SEED                # 20260927（数据顺序 RNG）
B0_BYTES = v2c.B0_BYTES                              # 10000.0
DELTA = v2c.DELTA                                    # 0.0
LAMBDA_INIT = v2c.LAMBDA_INIT                        # 1.0
ETA_LAMBDA = v2c.ETA_LAMBDA                          # 0.05
LOG_PROB_TOLERANCE = v2c.LOG_PROB_TOLERANCE          # 1e-6
MILESTONE_EPISODES = v2c.MILESTONE_EPISODES          # (0, 64, 128)

#: ROUND30 自己的两个编码臂（执行依据 §1/§4.4）。唯一主动改变的是期限列定义。
ARMS: dict[str, dict[str, Any]] = {
    "old": {
        "feature_layout": "comm-send-state/1",
        "deadline_column": "clip(remaining_time / 50.0, -1.0, 1.0)",
        "encoder": "integrated_sar.comm_learning.features（生产编码器原样）",
        "registered_change": "none: this arm reproduces the archived encoder byte for byte",
    },
    "new": {
        "feature_layout": "comm-send-state/1r29-deadline-over-horizon",
        "deadline_column": "remaining_time / configured_episode_horizon(1000.0)，不截断",
        "encoder": "runs/comm_deadline_encoding_audit_20260929/r29_encoding_prototype."
                   "candidate_features（ROUND29 登记原型）",
        "registered_change": "only feature column 12 (0-based index 11) changes definition; the "
                             "other 18 columns, their order, urgency_now, and the 19-dim shape "
                             "are unchanged",
    },
}
REFERENCE_ARM = "immediate"
RULE_ARMS = v2c.RULE_ARMS                            # 规则臂定义沿用已核验件
LATE_RULE_ARM = "late_discovery_gate"

# ------------------------------------------------------------------ 启动额度（执行依据 §5）
REFERENCE_LAUNCHES = 48 + 16                         # 64 immediate reference
TRAINING_LAUNCHES = len(ARMS) * len(TRAIN_SEEDS) * EPISODES_PER_SEED   # 2 * 3 * 128 = 768
EVAL_LAUNCHES = len(ARMS) * len(TRAIN_SEEDS) * 2 * len(DEVELOPMENT_SCENARIOS)  # 192
LATE_RULE_LAUNCHES = 16                              # late_discovery_gate_147 开发场景
NORMAL_LAUNCHES = REFERENCE_LAUNCHES + TRAINING_LAUNCHES + EVAL_LAUNCHES + LATE_RULE_LAUNCHES  # 1040
RESERVE_LAUNCHES = 40
LAUNCH_CAP = NORMAL_LAUNCHES + RESERVE_LAUNCHES      # 1080
SYNTHETIC_UPDATE_CAP = 64                            # 纯合成 optimizer 更新上限（与真实更新分账）

#: 统计方案（执行依据 §8.3，启动前冻结）
STATISTICS_PLAN = {
    "scene_order": "ascending scenario seed",
    "cluster_unit": "the 16 development scenes (paired per training seed and mode)",
    "resampler": "numpy PCG64",
    "resample_seed": 20260930,
    "resamples": 10000,
    "quantile": "linear",
    "index_matrix": "one shared 10000 x 16 index matrix reused by every comparison",
    "estimator": "values[indices].mean(axis=1)",
    "hand_check": "an independent hand-computed counterexample is kept in r30_stats.py",
    "coverage_note": "a per-seed descriptive 95% interval covers scene uncertainty under the "
                     "fixed trained weights only; it does not cover training-seed uncertainty, "
                     "and a single sampled rollout per scene does not estimate its full "
                     "sampling variance",
}


def scenario_order() -> list[int]:
    """ROUND24 冻结的 128 集场景访问顺序（引用而非重写）。"""
    return v2c.scenario_order()


def cell_paths(kind: str, seed: int, attempt: int = 1, tag: str = "") -> dict[str, Path]:
    """本轮 cell 的全部产物路径；名字带 tag（或 kind）、seed 与 attempt。"""
    stem = f"r30_{tag}_s{seed}_a{attempt}" if tag else f"r30_{kind}_s{seed}_a{attempt}"
    return {
        "lifecycle": LOGS / f"{stem}_rp.jsonl",
        "transcript": LOGS / f"{stem}_ct.jsonl",
        "tape": LOGS / f"{stem}_tape.jsonl",
        "row": ROWS / f"{stem}.json",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name: str, payload: Any, *, refresh: bool = False) -> Path:
    """写本轮产物；只有 START_LEDGER.json 可刷新（计划块冻结、只增不减）。"""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = OUTPUT_DIR / name
    if refresh and name != "START_LEDGER.json":
        raise RuntimeError("only START_LEDGER.json may be refreshed")
    if target.exists() and refresh:
        previous = read_json(target)
        if previous.get("plan") != payload.get("plan"):
            raise RuntimeError(f"the frozen plan block of {name} changed")
        before = int((previous.get("live") or {}).get("used") or 0)
        after = int((payload.get("live") or {}).get("used") or 0)
        if after < before:
            raise RuntimeError(f"{name}: tally shrank from {before} to {after}")
        merged = {"plan": previous["plan"], "live": payload["live"]}
        target.write_text(json.dumps(merged, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
        return target
    if target.exists():
        raise RuntimeError(f"refusing to rewrite a delivered artefact: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def iter_jsonl(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{number}: unparsable record") from error


# ------------------------------------------------------------------ 并行安全台账
def _ledger_path() -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"START_LEDGER_p{os.getpid()}.json"


def _lock_path() -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / "_ledger.lock"


class _FileLock:
    """跨进程互斥：O_CREAT|O_EXCL 抢占锁文件，忙等后重试；退出即释放。"""

    def __init__(self, timeout: float = 120.0) -> None:
        self.path = _lock_path()
        self.timeout = timeout

    def __enter__(self) -> "_FileLock":
        started = time.time()
        while True:
            try:
                self.handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                if time.time() - started > self.timeout:
                    raise RuntimeError(f"ledger lock {self.path} stayed held past "
                                       f"{self.timeout}s; refusing to book without it")
                time.sleep(0.02)

    def __exit__(self, *_exc: Any) -> None:
        os.close(self.handle)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def ledger_total() -> int:
    total = 0
    for path in sorted(LOGS.glob("START_LEDGER_p*.json")):
        try:
            total += int(read_json(path).get("starts") or 0)
        except Exception:
            continue
    return total


class LaunchLedger:
    """Book one episode launch before the driver is touched; the cap is judged on the sum.

    reserve/record 的临界区（读 pid 账本 + 写回）由锁文件保护，因此 6 条并行训练的计数
    与串行执行一致，硬上限不会被并发绕过。
    """

    def __init__(self) -> None:
        self.path = _ledger_path()

    def _book(self) -> dict[str, Any]:
        return read_json(self.path) if self.path.exists() else {"starts": 0, "entries": []}

    @staticmethod
    def _write(path: Path, book: dict[str, Any]) -> None:
        path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")

    def reserve(self, kind: str, label: str, seed: int, attempt: int = 1) -> int:
        with _FileLock():
            used = ledger_total()
            if used + 1 > LAUNCH_CAP:
                raise RuntimeError(
                    f"launch cap {LAUNCH_CAP} reached at {used}; refusing a {kind} start "
                    f"({label} s{seed} a{attempt})")
            book = self._book()
            book["starts"] += 1
            book["entries"].append({
                "n": book["starts"], "round_total_before_this": used, "kind": kind,
                "label": label, "seed": int(seed), "attempt": int(attempt),
            })
            self._write(self.path, book)
            return used + 1

    def record(self, *, label: str, seed: int, attempt: int, ok: bool, detail: str) -> None:
        with _FileLock():
            book = self._book()
            book.setdefault("results", []).append({
                "label": label, "seed": int(seed), "attempt": int(attempt), "ok": bool(ok),
                "detail": detail[:400],
            })
            self._write(self.path, book)


def ledger_report() -> dict[str, Any]:
    per_kind: dict[str, int] = {}
    per_label: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for path in sorted(LOGS.glob("START_LEDGER_p*.json")):
        try:
            book = read_json(path)
        except Exception:
            continue
        for entry in book.get("entries") or []:
            per_kind[str(entry.get("kind", "?"))] = per_kind.get(str(entry.get("kind", "?")), 0) + 1
            per_label[str(entry.get("label", "?"))] = per_label.get(str(entry.get("label", "?")), 0) + 1
            entries.append(entry)
        results.extend(book.get("results") or [])
    return {
        "normal_launches": NORMAL_LAUNCHES, "reserve_launches": RESERVE_LAUNCHES,
        "hard_cap": LAUNCH_CAP, "used": ledger_total(),
        "by_kind": per_kind, "by_label": per_label, "entries": entries, "results": results,
        "synthetic_update_cap": SYNTHETIC_UPDATE_CAP,
    }


# ------------------------------------------------------------------ 代码身份
PRODUCTION_DEPENDENCIES = v2c.PRODUCTION_DEPENDENCIES
IMPORTED_ROUND_MODULES = v2c.IMPORTED_ROUND_MODULES
CHECKPOINT_FILES = v2c.CHECKPOINT_FILES
ROUND29_PROTOTYPE = ROOT / "runs/comm_deadline_encoding_audit_20260929/r29_encoding_prototype.py"

HARNESS_MODULES = (
    "r30_common.py",
    "r30_encoding.py",
    "r30_run.py",
    "r30_train.py",
    "r30_blocks.py",
    "r30_check.py",
    "r30_preflight.py",
    "r30_stats.py",
    "r30_manifest.py",
    "r30_verdict.py",
    "r30_conflict_recheck.py",
    "r30_clock_recheck.py",
)


def code_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "production": {name: sha256_file(ROOT / name) for name in PRODUCTION_DEPENDENCIES},
        "imported_round_modules": {
            str(path.resolve().relative_to(ROOT)): sha256_file(path)
            for path in IMPORTED_ROUND_MODULES if path.exists()},
        "round29_encoder_prototype": {str(ROUND29_PROTOTYPE.resolve().relative_to(ROOT)):
                                     sha256_file(ROUND29_PROTOTYPE)},
        "harness": {name: sha256_file(HERE / name) for name in HARNESS_MODULES
                    if (HERE / name).exists()},
        "checkpoints": {str(path.resolve().relative_to(ROOT)): sha256_file(path)
                        for path in CHECKPOINT_FILES},
    }
    payload["freeze_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return payload
