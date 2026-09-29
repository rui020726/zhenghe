"""ROUND27 公共件：路径、五条运行路径、原子启动账本、冻结常量与代码身份。

对应执行依据 §二-§四：仅场景 1400031、模型固定 ROUND25 的 seed33 ep128 与原 greedy 解码，
5 条路径各一次正常启动（5 正常 + 3 技术储备 = 8 硬上限），每格先原子 claim 再记账再构造 driver。
配置逐字段继承 ROUND24/ROUND25 已核验的 Stage B 冻结集（与 r25_common 逐字节相同，
由 r27_preflight 的合成测试机械核对）。

与 ROUND25 harness 的关系：本模块不导入 r25_common（其 OUTPUT_DIR 与账本指向 ROUND25 目录），
配置继承改从 sp_common 直接取，避免跨轮共享可变状态。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT_DIR = ROOT / "runs" / "comm_cost_firstreport_intervention_20260929"
ROUND25_DIR = ROOT / "runs" / "comm_cost_greedy_replication_20260928"   # 只读：对照行与冻结身份来源
ROUND24_DIR = ROOT / "runs" / "comm_cost_v2_rerun_20260928"             # 只读：checkpoint 与冻结身份来源
STAGE_B_DIR = ROOT / "runs" / "comm_cost_stage_b_20260927"
STAGE_A_DIR = ROOT / "runs" / "comm_cost_controller_stage_a_20260927"
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"
MASK_FIX_ROUND = ROOT / "runs" / "comm_budget_action_mask_fix_20260922"
TIGHT_ROUND = ROOT / "runs" / "comm_tight_budget_20260919"
ROUND18_DIR = ROOT / "runs" / "search_trace_and_policy_20260925"

LOGS = OUTPUT_DIR / "logs"
ROWS = OUTPUT_DIR / "rows"
REJECTED = OUTPUT_DIR / "rejected"
CLAIMS = OUTPUT_DIR / "ledger_claims"

# ------------------------------------------------------------------ 冻结的运行常数 -------------
#: 本轮唯一场景（ROUND26 事后选定的失败案例；特权诊断场景，不是随机抽样）。
SCENE = 1400031
#: 两个失败目标的 target ID，来自 ROUND26 的三臂诊断（SCENE_1400031_DIAGNOSIS.json）。
INTERVENTION_TARGETS: tuple[str, ...] = ("piloc-e1-target-0", "piloc-e1-target-5")

#: 五条运行路径（执行依据 §二）。
#: L  = 原 learned 路径，不安装干预器；
#: L0 = 安装干预器但禁用全部覆盖；
#: E0/E5/E05 = 分别只提前 target-0 / 只提前 target-5 / 同时提前两者的首报。
PATHS: dict[str, dict[str, Any]] = {
    "L": {"kind": "learned", "interceptor": False, "override_targets": (),
          "role": "original learned path, no interceptor installed"},
    "L0": {"kind": "learned", "interceptor": True, "override_enabled": False,
           "override_targets": INTERVENTION_TARGETS,
           "role": "interceptor installed, every override disabled"},
    "E0": {"kind": "learned", "interceptor": True, "override_enabled": True,
           "override_targets": ("piloc-e1-target-0",),
           "role": "advance target-0 first report only"},
    "E5": {"kind": "learned", "interceptor": True, "override_enabled": True,
           "override_targets": ("piloc-e1-target-5",),
           "role": "advance target-5 first report only"},
    "E05": {"kind": "learned", "interceptor": True, "override_enabled": True,
            "override_targets": INTERVENTION_TARGETS,
            "role": "advance both targets' first reports"},
}
PATH_ORDER: tuple[str, ...] = ("L", "L0", "E0", "E5", "E05")
GATE_PATHS: tuple[str, ...] = ("L", "L0")
INTERVENTION_PATHS: tuple[str, ...] = ("E0", "E5", "E05")
CANDIDATE_PATH = "L"

#: 启动预算（执行依据 §二）：5 正常 + 3 技术储备 = 8 独立硬上限。
NORMAL_LAUNCHES = 5
RESERVE_LAUNCHES = 3
LAUNCH_CAP = NORMAL_LAUNCHES + RESERVE_LAUNCHES

STEPS_PER_EPISODE = 1000
B0_BYTES = 10000.0
COMM_BUDGET_BYTES = 29300

#: 学习臂动作 RNG 派生标签：逐字继承 ROUND24/ROUND25 final_greedy 块。
EVAL_RNG_LABEL = "round24-eval"
CANDIDATE_TRAINING_SEED = 33
CANDIDATE_EPISODES = 128

#: Stage B 冻结传输配置的继承键（与 r25_common.STAGE_B_INHERITED_OVERRIDES 逐字节相同；
#: 相等性由 r27_preflight 的合成测试核对，因为两处都源自 ROUND24 冻结集）。
STAGE_B_INHERITED_OVERRIDES: dict[str, Any] = {
    "comm_timing_mode": "policy_owned_timing",
    "comm_max_wait_steps": None,
    "comm_urgent_wait_steps": None,
    "comm_budget_bytes": 29300,
    "comm_sends_per_tick": None,
    "comm_ack": False,
    "comm_subscribers": "all",
    "comm_delivery_probability": 0.7,
    "comm_delay_ticks": 0,
    "comm_range": 0.3,
    "gossip_interval": 2,
    "steps": 1000,
    "num_goals": 6,
    "mock": False,
    "quiet": True,
    "device": "cpu",
    "info_probe": None,
    "search_probe": None,
    "decision_trace": None,
    "record_stream": None,
    "replay_stream": None,
}

PRODUCTION_DEPENDENCIES = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/run_signature.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/lifecycle_metrics.py",
    "integrated_sar/task_registry.py",
)
IMPORTED_ROUND_MODULES = (
    ROUND24_DIR / "v2_controller.py",
    ROUND24_DIR / "v2_rules.py",
    ROUND24_DIR / "v2_trainer.py",
    ROUND24_DIR / "v2_common.py",
    STAGE_B_DIR / "sb_common.py",
    STAGE_B_DIR / "sb_controller.py",
    STAGE_B_DIR / "sb_collector.py",
    STAGE_B_DIR / "sb_run.py",
    PILOT_ROUND / "train_pilot.py",
    PILOT_ROUND / "pilot_common.py",
    PILOT_ROUND / "pilot_env.py",
    MASK_FIX_ROUND / "budget_masked.py",
    MASK_FIX_ROUND / "train_budget_masked.py",
    TIGHT_ROUND / "onpolicy.py",
    TIGHT_ROUND / "train_onpolicy.py",
    ROUND18_DIR / "sp_common.py",
)
HARNESS_MODULES = (
    "r27_common.py",
    "r27_intervention.py",
    "r27_run.py",
    "r27_freeze.py",
    "r27_preflight.py",
    "r27_gate.py",
    "r27_compare.py",
    "r27_target5_readback.py",
)
CHECKPOINT_FILES = (
    ROOT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth",
    ROOT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth",
)
LEARNED_CHECKPOINT = ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "checkpoints" / "v2_train_s33_ep128.pt"


def enter_round() -> None:
    for directory in (str(ROOT), str(ROUND24_DIR), str(ROUND25_DIR), str(ROUND18_DIR),
                      str(PILOT_ROUND), str(MASK_FIX_ROUND), str(TIGHT_ROUND), str(STAGE_B_DIR),
                      str(STAGE_A_DIR)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    while str(HERE) in sys.path:
        sys.path.remove(str(HERE))
    sys.path.insert(0, str(HERE))


enter_round()


# ------------------------------------------------------------------------------ 文件 -------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name: str, payload: Any, *, refresh: bool = False) -> Path:
    """Write one artefact of this round; only START_LEDGER.json may be refreshed (plan frozen)."""
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


def cell_paths(path_name: str, attempt: int = 1) -> dict[str, Path]:
    """Every artefact of one path run; the name carries the path and attempt."""
    stem = f"r27_{path_name}_s{SCENE}_a{attempt}"
    return {
        "lifecycle": LOGS / f"{stem}_rp.jsonl",
        "transcript": LOGS / f"{stem}_ct.jsonl",
        "tape": LOGS / f"{stem}_tape.jsonl",
        "row": ROWS / f"{stem}.json",
    }


def row_path(path_name: str, attempt: int) -> Path:
    return cell_paths(path_name, attempt)["row"]


def done_paths() -> set[str]:
    """Paths whose final row exists on disk (resume set)."""
    done: set[str] = set()
    for path in sorted(ROWS.glob("r27_*_s*_a*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        done.add(str(row["path"]))
    return done


# ------------------------------------------------------------------ 配置继承 -------------
def stage_b_inherited_config(seed: int) -> tuple[dict[str, Any], list[str]]:
    """The scenario's world fields, inherited exactly the way ROUND25 inherited them.

    Same provenance rule: the archived D1 A3 record of the configuration source (1100000),
    the Stage B frozen override set, and only the scenario seed changed. The override dict is
    byte-identical to r25_common's; the equality is checked mechanically in r27_preflight.
    """
    import sp_common as sc

    source = sc.d1_archive_path("A3", 1100000)
    fields, substituted = sc.configuration_from_archive(sc.archived_configuration(source))
    fields.update(STAGE_B_INHERITED_OVERRIDES)
    fields["seed"] = int(seed)
    substituted.append(f"Stage B frozen overrides applied: {sorted(STAGE_B_INHERITED_OVERRIDES)}")
    return fields, substituted


# ------------------------------------------------------------------ 启动账本（原子） -------------
def _claim_path(path_name: str, attempt: int) -> Path:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    return CLAIMS / f"{path_name}_a{attempt}.json"


def claim_cell(path_name: str, attempt: int, label: str) -> bool:
    """Atomically book one launch. Returns False when another process already books it."""
    target = _claim_path(path_name, attempt)
    payload = {"path": path_name, "attempt": int(attempt),
               "label": label, "pid": os.getpid(), "created": _now()}
    try:
        handle = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return True


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def launch_count() -> int:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    return len(list(CLAIMS.glob("*.json")))


def _ledger_path() -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"START_LEDGER_p{os.getpid()}.json"


def ledger_total() -> int:
    total = 0
    for path in sorted(LOGS.glob("START_LEDGER_p*.json")):
        try:
            total += int(read_json(path).get("starts") or 0)
        except Exception:
            continue
    return total


class LaunchLedger:
    """Book one episode launch before the driver is touched.

    Two independent books must agree: the per-process ledger and the atomic claim directory.
    ``reserve`` refuses above the hard cap and refuses a cell another process already booked.
    """

    def __init__(self) -> None:
        self.path = _ledger_path()

    def _book(self) -> dict[str, Any]:
        return read_json(self.path) if self.path.exists() else {"starts": 0, "entries": []}

    def reserve(self, kind: str, path_name: str, attempt: int, label: str) -> int:
        used = launch_count()
        if used + 1 > LAUNCH_CAP:
            raise RuntimeError(
                f"launch cap {LAUNCH_CAP} reached at {used}; refusing a {kind} start "
                f"({label} a{attempt})")
        if not claim_cell(path_name, attempt, label):
            raise RuntimeError(
                f"cell {label} a{attempt} is already booked by another process; "
                "refusing a double assignment")
        book = self._book()
        book["starts"] += 1
        book["entries"].append({
            "n": book["starts"], "round_total_before_this": used, "kind": kind,
            "label": label, "attempt": int(attempt),
        })
        self.path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
        return used + 1

    def record(self, *, label: str, attempt: int, ok: bool, detail: str) -> None:
        book = self._book()
        book.setdefault("results", []).append({
            "label": label, "attempt": int(attempt), "ok": bool(ok),
            "detail": detail[:400],
        })
        self.path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")


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
    claims = launch_count()
    return {
        "normal_launches": NORMAL_LAUNCHES, "reserve_launches": RESERVE_LAUNCHES,
        "hard_cap": LAUNCH_CAP, "used": claims,
        "per_process_ledger_total": ledger_total(),
        "atomic_claim_total": claims,
        "ledgers_agree": ledger_total() == claims,
        "by_kind": per_kind, "by_label": per_label, "entries": entries, "results": results,
    }


# ------------------------------------------------------------------ 代码身份 -------------
def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def code_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "production": {name: sha256_file(ROOT / name) for name in PRODUCTION_DEPENDENCIES},
        "imported_round_modules": {
            _rel(path): sha256_file(path) for path in IMPORTED_ROUND_MODULES if path.exists()},
        "harness": {name: sha256_file(HERE / name) for name in HARNESS_MODULES
                    if (HERE / name).exists()},
        "checkpoints": {_rel(path): sha256_file(path) for path in CHECKPOINT_FILES},
        "learned_checkpoint": {_rel(LEARNED_CHECKPOINT): sha256_file(LEARNED_CHECKPOINT)},
    }
    payload["freeze_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def round24_freeze_payload() -> dict[str, Any]:
    """The freeze identity the ROUND24 milestones were written under (read-only source)."""
    manifest = read_json(ROUND24_DIR / "CONFIG_AND_MODEL_MANIFEST.json")
    return {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
            "config_hash": manifest["configuration_hash"],
            "branch_start": manifest["branch_start"],
            "head_at_freeze": manifest["head_at_freeze"]}
