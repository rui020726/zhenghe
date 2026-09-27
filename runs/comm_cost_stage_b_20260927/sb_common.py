"""Stage B 公共件：路径、启动台账、配置继承与代码身份。

边界（STAGE_B_PROTOCOL.md §1）：每个 episode 启动都先经 LaunchLedger 记账，全局求和判 24 硬上限；
纯数学单测不构造 `JointRun`、不启动环境，不计启动。

复用而非重写：配置继承用第十八轮的 `sp_common`（identity 字段天然被剔除，路径类字段
不会写回 D1 归档目录），网络/采样器用第四/五/七轮的 `train_pilot.Controller` 与
`BudgetMaskedCollector`，计数器口径用 relay 自己的账本。
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
STAGE_A_DIR = ROOT / "runs" / "comm_cost_controller_stage_a_20260927"
D1_LOGS = ROOT / "runs" / "comm_info_path_validation_20260923" / "logs"
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"
MASK_FIX_ROUND = ROOT / "runs" / "comm_budget_action_mask_fix_20260922"
TIGHT_ROUND = ROOT / "runs" / "comm_tight_budget_20260919"
ROUND18_DIR = ROOT / "runs" / "search_trace_and_policy_20260925"

LOGS = HERE / "logs"
ROWS = HERE / "rows"
REJECTED = HERE / "rejected"

# ------------------------------------------------------------------ 冻结的运行常数 -------------
#: 三个既有开发 seed（第十六轮 D1 已claim块 1100000..1100059 内；开发接线复跑，非新独立测试集）
SEEDS: tuple[int, ...] = (1100000, 1100001, 1100002)

#: 六条路径，按 spec §4 表。顺序即执行顺序（等价对相邻，P_init 紧接 P_repeat 之前）。
PATHS: tuple[str, ...] = ("R_send", "I_send", "R_hold", "I_hold", "P_init", "P_repeat")

#: 六路径 × 三 seed 的正常启动数；储备仅用于技术失败复跑；两者相加为硬上限。
NORMAL_LAUNCHES = len(PATHS) * len(SEEDS)      # 18
RESERVE_LAUNCHES = 6
LAUNCH_CAP = NORMAL_LAUNCHES + RESERVE_LAUNCHES  # 24

#: 固定集成测试配置（STAGE_B_PROTOCOL.md §2/§4）：与 D1 A3 归档的差异只有这些键。
STAGE_B_OVERRIDES: dict[str, Any] = {
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
    # 路径类 identity 字段一律显式置空/指向本轮，绝不继承归档路径。
    "info_probe": None,
    "search_probe": None,
    "decision_trace": None,
    "record_stream": None,
    "replay_stream": None,
}

#: 接线算例常数——**仅可重算验证用，不批准为训练超参数或任务容忍度**（spec §3/§4）。
B0_TEST = 10_000.0
LAMBDA_TEST = 1.0
DELTA_TEST = 0.0
ETA_TEST = 1.0
GAMMA = 1.0
GAE_LAMBDA = 0.95
ACTOR_INIT_SEED = 20260927
SAMPLING_SEED = 20260928
LOG_PROB_TOLERANCE = 1e-6   # 既有门槛：runs/comm_budget_masked_retrain_20260922/train_retrain.py:57

#: 生产依赖源文件：冻结后不并发改代码；哈希进 CONFIG_AND_MODEL_MANIFEST。
PRODUCTION_DEPENDENCIES = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/comm_ack.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/lifecycle_metrics.py",
    "integrated_sar/expired_attribution.py",
    "integrated_sar/task_registry.py",
)
IMPORTED_ROUND_MODULES = (
    ROUND18_DIR / "sp_common.py",
    PILOT_ROUND / "train_pilot.py",
    PILOT_ROUND / "pilot_common.py",
    PILOT_ROUND / "pilot_env.py",
    MASK_FIX_ROUND / "budget_masked.py",
    TIGHT_ROUND / "onpolicy.py",
)
HARNESS_MODULES = (
    "sb_common.py",
    "sb_controller.py",
    "sb_collector.py",
    "sb_run.py",
    "sb_check.py",
    "test_stage_b_math.py",
)
CHECKPOINTS = (
    ROOT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth",
    ROOT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth",
)

# B1 预先列明的允许差异：只有这两类元数据可以不同（spec §5：不得在读差异后扩大白名单）。
ALLOWED_METADATA_DIFFERENCES = ("artefact_paths", "run_id_and_arm_labels")


def enter_round() -> None:
    """Put every dependency round on sys.path, keeping this round's modules in front.

    Several rounds ship modules with the same name (``rule_matrix`` exists twice, for
    instance); without this ordering an import could silently resolve to another round's
    copy and mix two harnesses into one aggregate.
    """
    for directory in (str(ROOT), str(ROUND18_DIR), str(PILOT_ROUND), str(MASK_FIX_ROUND),
                      str(TIGHT_ROUND), str(STAGE_A_DIR)):
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
    """Write one artefact of this round, refusing to touch anything else.

    ``refresh`` is granted to exactly one file, the launch ledger's live tally: the plan
    block is frozen and re-checked on every refresh, and the tally may only move forward --
    a refresh reporting fewer launches than the delivered file is a lost book.
    """
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = HERE / name
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


def cell_paths(path_name: str, seed: int, attempt: int = 1) -> dict[str, Path]:
    """Every artefact of one cell; the name carries path, seed and attempt."""
    tag = f"{path_name}_s{seed}_a{attempt}"
    return {
        "lifecycle": LOGS / f"sb_{tag}_rp.jsonl",
        "transcript": LOGS / f"sb_{tag}_ct.jsonl",
        "tape": LOGS / f"sb_{tag}_tape.jsonl",
        "row": ROWS / f"sb_{tag}.json",
    }


def d1_archive_path(seed: int) -> Path:
    """The seed's own archived D1 A3 (normal-lossy) run_config record."""
    return D1_LOGS / f"data_A3_s{seed}_rp.jsonl"


# ----------------------------------------------------------------------------- 台账 -------------
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
    """Book one episode launch before the driver is touched; the cap is judged on the sum.

    A shared read-modify-write ledger would break under parallel writers, so each process
    keeps its own book and the cap is the sum over all of them -- the same discipline the
    frozen rounds used.
    """

    def __init__(self) -> None:
        self.path = _ledger_path()

    def _book(self) -> dict[str, Any]:
        return read_json(self.path) if self.path.exists() else {"starts": 0, "entries": []}

    def reserve(self, kind: str, path_name: str, seed: int, attempt: int = 1) -> int:
        used = ledger_total()
        if used + 1 > LAUNCH_CAP:
            raise RuntimeError(
                f"launch cap {LAUNCH_CAP} reached at {used}; refusing a {kind} start "
                f"({path_name} s{seed} a{attempt})"
            )
        book = self._book()
        book["starts"] += 1
        book["entries"].append({
            "n": book["starts"], "round_total_before_this": used, "kind": kind,
            "path": path_name, "seed": int(seed), "attempt": int(attempt),
        })
        self.path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
        return used + 1

    def record(self, *, path_name: str, seed: int, attempt: int, ok: bool, detail: str) -> None:
        book = self._book()
        book.setdefault("results", []).append({
            "path": path_name, "seed": int(seed), "attempt": int(attempt), "ok": bool(ok),
            "detail": detail[:400],
        })
        self.path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")


def ledger_report() -> dict[str, Any]:
    per_kind: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for path in sorted(LOGS.glob("START_LEDGER_p*.json")):
        try:
            book = read_json(path)
        except Exception:
            continue
        for entry in book.get("entries") or []:
            kind = str(entry.get("kind", "?"))
            per_kind[kind] = per_kind.get(kind, 0) + 1
            entries.append(entry)
        results.extend(book.get("results") or [])
    return {
        "normal_launches": NORMAL_LAUNCHES, "reserve_launches": RESERVE_LAUNCHES,
        "hard_cap": LAUNCH_CAP, "used": ledger_total(),
        "by_kind": per_kind, "entries": entries, "results": results,
    }


# --------------------------------------------------------------------- 配置继承与身份 -------------
def inherit_configuration(seed: int) -> tuple[dict[str, Any], list[str]]:
    """The seed's own archived D1 A3 behaviour fields, plus this round's overrides.

    Path-like identity fields are dropped by ``configuration_from_archive`` itself, so an
    archived ``info_probe`` path can never make this round write into the D1 directory.
    """
    import sp_common as sc

    archive = d1_archive_path(seed)
    if not archive.exists():
        raise FileNotFoundError(f"seed {seed} has no archived D1 A3 record at {archive}")
    fields, substituted = sc.configuration_from_archive(sc.archived_configuration(archive))
    fields["seed"] = int(seed)
    substituted.append(f"Stage B overrides applied: {sorted(STAGE_B_OVERRIDES)}")
    return fields, substituted


def stage_b_fields(path_name: str, seed: int, attempt: int = 1) -> tuple[dict[str, Any], list[str]]:
    """The complete field dict for one cell: inherited world, overridden transport, per-path policy."""
    fields, substituted = inherit_configuration(seed)
    fields.update(STAGE_B_OVERRIDES)
    policy = {"R_send": "immediate", "I_send": "learned", "R_hold": "always_hold",
              "I_hold": "learned", "P_init": "learned", "P_repeat": "learned"}[path_name]
    fields["comm_policy"] = policy
    paths = cell_paths(path_name, seed, attempt)
    fields["output"] = str(paths["lifecycle"])
    fields["comm_transcript"] = str(paths["transcript"])
    substituted.append(f"comm_policy overridden for {path_name}: {policy}")
    return fields, substituted


def code_identity() -> dict[str, Any]:
    """Content hashes of everything the episode path depends on, frozen per run."""
    payload: dict[str, Any] = {}
    payload["production"] = {name: sha256_file(ROOT / name) for name in PRODUCTION_DEPENDENCIES}
    payload["imported_round_modules"] = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in IMPORTED_ROUND_MODULES
    }
    payload["harness"] = {name: sha256_file(HERE / name) for name in HARNESS_MODULES}
    payload["checkpoints"] = {str(path.relative_to(ROOT)): sha256_file(path)
                              for path in CHECKPOINTS}
    payload["round_dir_allow_list_note"] = "logs/ rows/ rejected/ and START_LEDGER_p*.json stay out of Git"
    return payload
