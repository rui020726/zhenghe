"""ROUND32 公共件：路径、并行安全启动台账、冻结常量引用、两臂定义与代码身份。

与 ROUND30 的登记关系：训练/评估常量、PPO 超参、目标函数常量、场景块、训练 seed、数据顺序、
episode 配置、bootstrap 方案**全部从 `v2_common` / `r30_common` 引用**（唯一事实来源）。
本文件只声明 ROUND32 自己的额度、臂定义、输出路径与代码身份清单，不重新定义训练数学。

两臂唯一主动改变的是 actor surrogate 的归约集合与分母（见 `r32_trainer`）；编码器两臂都是
ROUND30 的 `old` 臂（生产 `comm-send-state/1`，`clip(remaining_time / 50, -1, 1)`），
因此不混入 ROUND29/30 的新期限编码。

台账：每个 episode 启动先记账再构造 driver；6 条训练按 (reduction, seed) 分进程并行，
reserve/record 的临界区由锁文件保护，硬上限在 reserve 时判定；纯合成更新不构造环境、
不计启动、另行分账（上限 64）。
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

R30_DIR = ROOT / "runs" / "comm_deadline_encoding_ablation_20260930"
R31_DIR = ROOT / "runs" / "comm_constraint_optimization_diagnosis_20260930"

for _shared in ("runs/comm_cost_v2_rerun_20260928", "runs/comm_cost_stage_b_20260927",
                "runs/comm_budget_pilot_20260919", "runs/comm_budget_action_mask_fix_20260922",
                "runs/comm_tight_budget_20260919", "runs/search_trace_and_policy_20260925",
                "runs/comm_deadline_encoding_audit_20260929",
                "runs/comm_deadline_encoding_ablation_20260930",
                "runs/comm_constraint_optimization_diagnosis_20260930"):
    _path = ROOT / _shared
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import r30_common as c30  # noqa: E402  (the archived ROUND30 common installs the shared path set)
import v2_common as v2c  # noqa: E402

OUTPUT_DIR = ROOT / "runs" / "comm_actor_reduction_ablation_20261001"
LOGS = OUTPUT_DIR / "logs"
ROWS = OUTPUT_DIR / "rows"
REJECTED = OUTPUT_DIR / "rejected"
CHECKPOINTS = OUTPUT_DIR / "checkpoints"

# ------------------------------------------------------------------ 从冻结件引用的常量
TRAINING_SCENARIOS = v2c.TRAINING_SCENARIOS          # 1300000..1300047（48）
DEVELOPMENT_SCENARIOS = v2c.DEVELOPMENT_SCENARIOS    # 1301000..1301015（16）
TRAIN_SEEDS = v2c.TRAIN_SEEDS                        # (11, 22, 33)
EPISODES_PER_SEED = v2c.EPISODES_PER_SEED            # 128
STEPS_PER_EPISODE = v2c.STEPS_PER_EPISODE            # 1000
EPISODES_PER_UPDATE = v2c.EPISODES_PER_UPDATE        # 4
DATA_ORDER_SEED = v2c.DATA_ORDER_SEED                # 20260927
B0_BYTES = v2c.B0_BYTES                              # 10000.0
DELTA = v2c.DELTA                                    # 0.0
LAMBDA_INIT = v2c.LAMBDA_INIT                        # 1.0
ETA_LAMBDA = v2c.ETA_LAMBDA                          # 0.05
LOG_PROB_TOLERANCE = v2c.LOG_PROB_TOLERANCE          # 1e-6
MILESTONE_EPISODES = v2c.MILESTONE_EPISODES          # (0, 64, 128)
RULE_ARMS = v2c.RULE_ARMS
REFERENCE_ARM = "immediate"
LATE_RULE_ARM = "late_discovery_gate"
#: 诊断证据要求额外保存"更新前完整 checkpoint"的批索引（执行依据 §五/§六）。
PRE_UPDATE_BATCHES = (0, 15, 31)

#: 两臂。唯一主动差异是 actor surrogate 的归约集合与分母；编码器两臂同为 ROUND30 的 old 臂。
REDUCTIONS: dict[str, dict[str, Any]] = {
    "ALL_ROWS": {
        "actor_reduction": "sum(s_i for every logged row) / N",
        "implementation": "the frozen v2_trainer expression -surrogate.mean() verbatim",
        "encoder_arm": "old",
        "feature_layout": c30.ARMS["old"]["feature_layout"],
    },
    "ACTIVE_ROWS": {
        "actor_reduction": "sum(s_i for rows with >=1 free candidate) / M, M=0 -> exact zero",
        "implementation": "(surrogate * row_active).sum() / max(M, 1); row_active is the saved "
                          "tape's non-empty `actionable`/`free` set",
        "encoder_arm": "old",
        "feature_layout": c30.ARMS["old"]["feature_layout"],
    },
}
ARM_ORDER = ("ALL_ROWS", "ACTIVE_ROWS")
MODES = ("greedy", "sampled")

# ------------------------------------------------------------------ 启动额度（执行依据 §4.3）
REFERENCE_LAUNCHES = len(TRAINING_SCENARIOS) + len(DEVELOPMENT_SCENARIOS)   # 64
TRAINING_LAUNCHES = len(ARM_ORDER) * len(TRAIN_SEEDS) * EPISODES_PER_SEED   # 768
EVAL_LAUNCHES = (len(ARM_ORDER) * len(TRAIN_SEEDS) * 2 * len(DEVELOPMENT_SCENARIOS))  # 192
LATE_RULE_LAUNCHES = len(DEVELOPMENT_SCENARIOS)                             # 16
NORMAL_LAUNCHES = REFERENCE_LAUNCHES + TRAINING_LAUNCHES + EVAL_LAUNCHES + LATE_RULE_LAUNCHES
RESERVE_LAUNCHES = 40
LAUNCH_CAP = NORMAL_LAUNCHES + RESERVE_LAUNCHES                             # 1080
SYNTHETIC_UPDATE_CAP = 64
MAX_PARALLEL_WORKERS = 4

#: 统计方案：逐字段沿用 ROUND30 的冻结登记（执行依据 §七要求"沿用"）。
STATISTICS_PLAN = dict(c30.STATISTICS_PLAN)
STATISTICS_PLAN["inherited_from"] = ("ROUND30 STATISTICS_PLAN, byte-copied; the only change is "
                                     "that the two arms are reductions instead of encoders")

#: 评估 RNG 方案（沿用已核验派生，不新造）。
EVAL_RNG_SCHEME = 'sha256("round24-eval:<seed>:<scenario>:<mode>")  (v2_run.eval_action_seed)'
ACTION_RNG_SCHEME = 'sha256("sc-action:<seed>:<episode>")           (v2_run.action_seed)'


def scenario_order() -> list[int]:
    """ROUND24/30 冻结的 128 集场景访问顺序（引用而非重写）。"""
    return v2c.scenario_order()


def cell_paths(kind: str, seed: int, attempt: int = 1, tag: str = "") -> dict[str, Path]:
    """本轮 cell 的全部产物路径；名字带 tag（或 kind）、seed 与 attempt。"""
    stem = f"r32_{tag}_s{seed}_a{attempt}" if tag else f"r32_{kind}_s{seed}_a{attempt}"
    return {
        "lifecycle": LOGS / f"{stem}_rp.jsonl",
        "transcript": LOGS / f"{stem}_ct.jsonl",
        "tape": LOGS / f"{stem}_tape.jsonl",
        "row": ROWS / f"{stem}.json",
    }


# ------------------------------------------------------------------ 文件工具
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

    def __init__(self, timeout: float = 180.0) -> None:
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
                time.sleep(0.05)

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
    """每个 episode 启动在构造 driver 之前记账；硬上限按总数判定，失败不返还额度。"""

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
            key = str(entry.get("kind", "?"))
            per_kind[key] = per_kind.get(key, 0) + 1
            label = str(entry.get("label", "?"))
            per_label[label] = per_label.get(label, 0) + 1
            entries.append(entry)
        results.extend(book.get("results") or [])
    return {"normal_launches": NORMAL_LAUNCHES, "reserve_launches": RESERVE_LAUNCHES,
            "hard_cap": LAUNCH_CAP, "used": ledger_total(), "by_kind": per_kind,
            "by_label": per_label, "entries": entries, "results": results,
            "synthetic_update_cap": SYNTHETIC_UPDATE_CAP}


# ------------------------------------------------------------------ 合成更新分账
SYNTHETIC_BOOK = OUTPUT_DIR / "SYNTHETIC_UPDATES.json"


def synthetic_updates_used() -> int:
    if not SYNTHETIC_BOOK.exists():
        return 0
    return int(read_json(SYNTHETIC_BOOK).get("used") or 0)


def book_synthetic_updates(count: int, note: str, key: str = "") -> int:
    """Count pure synthetic optimizer steps against their own cap; never an environment start.

    The cap bounds the distinct synthetic work that was authorised, so a `key` is booked once.
    Repeated executions during development are not hidden: they are counted separately in
    `performed_repeats`, so the artefact reports both "distinct booked steps" and "steps actually
    performed across all runs of the test suite".
    """
    with _FileLock():
        SYNTHETIC_BOOK.parent.mkdir(parents=True, exist_ok=True)
        book = read_json(SYNTHETIC_BOOK) if SYNTHETIC_BOOK.exists() else {}
        events = book.get("events") or []
        used = sum(int(event.get("count") or 0) for event in events)
        repeats = int(book.get("performed_repeats") or 0)
        match = next((event for event in events if key and event.get("key") == key), None)
        if match is not None:
            match["performed_times"] = int(match.get("performed_times", 1)) + 1
            repeats += int(match.get("count") or 0)
        else:
            if used + count > SYNTHETIC_UPDATE_CAP:
                raise RuntimeError(
                    f"synthetic optimizer update cap {SYNTHETIC_UPDATE_CAP} reached at {used}; "
                    f"refusing {count} more ({note})")
            used += count
            events = events + [{"count": count, "note": note, "key": key, "performed_times": 1}]
        SYNTHETIC_BOOK.write_text(json.dumps({
            "used": used, "cap": SYNTHETIC_UPDATE_CAP, "performed_repeats": repeats,
            "performed_total": used + repeats,
            "note": "pure synthetic optimizer.step inside zero-launch tests only; no environment "
                    "is constructed and these are booked separately from the 1040 launch quota. "
                    "`used` counts distinct keyed bookings (the authorised bound); "
                    "`performed_total` also counts repeated executions of the same keyed step "
                    "during development.",
            "events": events},
            ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return used


# ------------------------------------------------------------------ 代码身份
PRODUCTION_DEPENDENCIES = v2c.PRODUCTION_DEPENDENCIES
IMPORTED_ROUND_MODULES = v2c.IMPORTED_ROUND_MODULES
CHECKPOINT_FILES = v2c.CHECKPOINT_FILES

#: 本轮直接复用的归档件（身份必须逐字节登记：本轮不修改它们，只 import）。
INHERITED_ROUND30_MODULES = (
    R30_DIR / "r30_common.py",
    R30_DIR / "r30_encoding.py",
    R30_DIR / "r30_run.py",
    R30_DIR / "r30_train.py",
    R30_DIR / "r30_stats.py",
    R30_DIR / "r30_check.py",
    R30_DIR / "r30_manifest.py",
    R30_DIR / "r30_code_identity.py",
    R30_DIR / "CONFIG_AND_MODEL_MANIFEST.json",
    R30_DIR / "ARTEFACT_MANIFEST.json",
)
INHERITED_ROUND31_MODULES = (
    R31_DIR / "r31_common.py",
    R31_DIR / "ABSOLUTE_PERFORMANCE.json",
    R31_DIR / "DUAL_TRAJECTORIES.json",
    R31_DIR / "UPDATE_SIGNAL_DIAGNOSIS.json",
    R31_DIR / "ROUND31_VERDICT.json",
)

HARNESS_MODULES = (
    "r32_common.py",
    "r32_trainer.py",
    "r32_run.py",
    "r32_train.py",
    "r32_blocks.py",
    "r32_check.py",
    "r32_math.py",
    "r32_preflight.py",
    "r32_freeze.py",
    "r32_stats.py",
    "r32_curves.py",
    "r32_mechanism.py",
    "r32_manifest.py",
    "r32_verify.py",
    "r32_verdict.py",
)


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def code_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "production": {name: sha256_file(ROOT / name) for name in PRODUCTION_DEPENDENCIES},
        "imported_round_modules": {
            _rel(path): sha256_file(path)
            for path in IMPORTED_ROUND_MODULES if Path(path).exists()},
        "inherited_round30_modules": {
            _rel(path): sha256_file(path)
            for path in INHERITED_ROUND30_MODULES if Path(path).exists()},
        "inherited_round31_modules": {
            _rel(path): sha256_file(path)
            for path in INHERITED_ROUND31_MODULES if Path(path).exists()},
        "harness": {name: sha256_file(HERE / name) for name in HARNESS_MODULES
                    if (HERE / name).exists()},
        "checkpoints": {_rel(path): sha256_file(path) for path in CHECKPOINT_FILES},
    }
    payload["freeze_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def round30_manifest() -> dict[str, Any]:
    """ROUND30's full frozen manifest (the reference for the identity audit)."""
    return read_json(R30_DIR / "CONFIG_AND_MODEL_MANIFEST.json")


def round30_frozen_code_identity() -> dict[str, Any]:
    """ROUND30's own frozen code-identity block, for the `identity unchanged` comparison."""
    manifest = read_json(R30_DIR / "CONFIG_AND_MODEL_MANIFEST.json")
    return manifest["code_identity"]
