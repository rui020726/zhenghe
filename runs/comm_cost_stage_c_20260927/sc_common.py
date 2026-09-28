"""Stage C 公共件：路径、启动台账、冻结常量、配置继承与 reference 缓存键。

对应 ROUND22_COMM_COST_LEARNING_PILOT.md §2-§5。边界：每个 episode 启动先记账再构造 driver；
672 正常 + 28 储备 = 700 硬上限；纯合成测试不构造 `JointRun`、不计启动。

复用而非重写：episode 接缝复用 Stage B（sb_run），网络/采样器复用 pilot 轮 Controller 与
BudgetMaskedCollector，PPO 结构复用 train_onpolicy 的联合似然/熵 helper（但**不**复用其
advantage 标准化与单 critic——见 sc_trainer 的登记差异）。
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
OUTPUT_DIR = ROOT / "runs" / "comm_cost_stage_c_correctness_20260928"
STAGE_B_DIR = ROOT / "runs" / "comm_cost_stage_b_20260927"
STAGE_A_DIR = ROOT / "runs" / "comm_cost_controller_stage_a_20260927"
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"
MASK_FIX_ROUND = ROOT / "runs" / "comm_budget_action_mask_fix_20260922"
TIGHT_ROUND = ROOT / "runs" / "comm_tight_budget_20260919"
ROUND18_DIR = ROOT / "runs" / "search_trace_and_policy_20260925"

LOGS = OUTPUT_DIR / "logs"
ROWS = OUTPUT_DIR / "rows"
REJECTED = OUTPUT_DIR / "rejected"
CHECKPOINTS = OUTPUT_DIR / "checkpoints"

# ------------------------------------------------------------------ 冻结的运行常数 -------------
#: 场景块（ROUND22 spec §4）：training 48 个 + development 16 个，均为新注册开发用途块。
TRAINING_SCENARIOS: tuple[int, ...] = tuple(range(1300000, 1300048))
DEVELOPMENT_SCENARIOS: tuple[int, ...] = tuple(range(1301000, 1301016))

#: 三个通信训练随机种子（spec §4）。与场景种子是不同命名空间。
TRAIN_SEEDS: tuple[int, ...] = (11, 22, 33)
EPISODES_PER_SEED = 128
STEPS_PER_EPISODE = 1000
EPISODES_PER_UPDATE = 4          # pilot 轮 EPISODES_PER_UPDATE；对偶 λ 也按此批更新
DATA_ORDER_SEED = 20260927       # spec §4：固定数据顺序 RNG

#: 启动预算（spec §5）：672 正常 + 28 储备 = 700 硬上限。
NORMAL_LAUNCHES = (3 * EPISODES_PER_SEED          # 384 训练
                   + (48 + 16)                     # 64 immediate reference
                   + 3 * 3 * 16                    # 144 学习模型开发评估（3 seed × 3 时点 × 16 场景）
                   + 5 * 16)                       # 80 五条规则开发评估
RESERVE_LAUNCHES = 28
LAUNCH_CAP = NORMAL_LAUNCHES + RESERVE_LAUNCHES   # 700

#: 目标函数与对偶（spec §2）：B0=10000 B、δ=0、λ 初值=1、η=0.05。
B0_BYTES = 10000.0
DELTA = 0.0
LAMBDA_INIT = 1.0
ETA_LAMBDA = 0.05
GAMMA = 1.0                      # spec §2（旧 trainer 为 0.99，登记差异）
GAE_LAMBDA = 0.95
LOG_PROB_TOLERANCE = 1e-6        # 与 Stage B / retrain 轮同一门槛

#: PPO 超参（sc_freeze 冻结为机读配置）：全部引用 pilot 轮 train_pilot 的既有值。
CLIP = 0.2
VALUE_COEFFICIENT = 0.5          # 每个 critic 头各用 0.5（登记选择，见协议 §2）
ENTROPY_BONUS = 0.01
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5
EPOCHS_PER_BATCH = 4
MINIBATCH_STEPS = 256
#: 网络初值：每个训练 seed 从 Controller(训练种子) 构造，与 pilot/retrain 两轮同规
#: （"every arm starts at Controller(training_seed) and nothing else"），不加载任何旧权重。

#: Stage B 冻结传输配置的继承键（spec §3：其余逐字段从 Stage B 冻结配置读取）。
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

#: 六条规则对照（spec §6）：immediate 为固定参考，send_once 经源码核验为其别名（不耗运行）。
RULE_ARMS: dict[str, dict[str, Any]] = {
    "always_hold": {"policy": "always_hold", "params": {}},
    "late_discovery_gate": {"policy": "late_discovery_gate", "params": {"cut_steps": 147.0}},
    "periodic_20": {"policy": "periodic", "params": {"interval": 20}},
    "random_send": {"policy": "random_send", "params": {"send_probability": 0.5,
                                                        "seed_from_scenario": True}},
    "event": {"policy": "event", "params": {"progress_threshold": 0.2, "urgency_time": 25.0}},
}
REFERENCE_ARM = "immediate"
SEND_ONCE_ALIAS_NOTE = (
    "send_once 与 immediate 的 decide 函数体逐字相同（comm_policy.py: ImmediatePolicy 与 "
    "SendOncePolicy 均为 pending_event 谓词、orders_selection 均 False）⇒ 源码核验语义一致，"
    "按别名登记，不另耗运行（spec §6）"
)

PRODUCTION_DEPENDENCIES = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/lifecycle_metrics.py",
    "integrated_sar/task_registry.py",
)
IMPORTED_ROUND_MODULES = (
    STAGE_B_DIR / "sb_common.py",
    STAGE_B_DIR / "sb_controller.py",
    STAGE_B_DIR / "sb_collector.py",
    STAGE_B_DIR / "sb_run.py",
    STAGE_B_DIR / "sb_check.py",
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
    "sc_common.py",
    "sc_rules.py",
    "sc_controller.py",
    "sc_trainer.py",
    "sc_run.py",
    "sc_train.py",
    "sc_blocks.py",
    "sc_check.py",
    "sc_resume_probe.py",
    "sc_freeze.py",
    "sc_recover.py",
    "test_sc_math.py",
    "sc_correctness_test_support.py",
    "test_sc_cost_objective.py",
    "test_sc_cost_head_lifecycle.py",
)
CHECKPOINT_FILES = (
    ROOT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth",
    ROOT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth",
)


def enter_round() -> None:
    for directory in (str(ROOT), str(ROUND18_DIR), str(PILOT_ROUND), str(MASK_FIX_ROUND),
                      str(TIGHT_ROUND), str(STAGE_B_DIR), str(STAGE_A_DIR)):
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


def cell_paths(kind: str, seed: int, attempt: int = 1, tag: str = "") -> dict[str, Path]:
    """Every artefact of one cell; the name carries the tag (or the kind), seed and attempt."""
    stem = f"sc_{tag}_s{seed}_a{attempt}" if tag else f"sc_{kind}_s{seed}_a{attempt}"
    return {
        "lifecycle": LOGS / f"{stem}_rp.jsonl",
        "transcript": LOGS / f"{stem}_ct.jsonl",
        "tape": LOGS / f"{stem}_tape.jsonl",
        "row": ROWS / f"{stem}.json",
    }


def scenario_order() -> list[int]:
    """The frozen 128-episode scenario order (ROUND22 spec §4).

    One dedicated RNG instance seeded 20260927; the 48 training scenarios are reshuffled each
    round and taken 48 + 48 + 32. The three communication training seeds share this order.
    """
    import random

    rng = random.Random(DATA_ORDER_SEED)
    scenarios = list(TRAINING_SCENARIOS)
    order: list[int] = []
    for round_index in range(3):
        rng.shuffle(scenarios)
        take = 32 if round_index == 2 else 48
        order.extend(scenarios[:take])
    if len(order) != EPISODES_PER_SEED:
        raise RuntimeError(f"scenario order has {len(order)} episodes, expected {EPISODES_PER_SEED}")
    return order


def episode_scenario(episode_index: int, order: list[int] | None = None) -> int:
    return (order or scenario_order())[episode_index]


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
    """Book one episode launch before the driver is touched; the cap is judged on the sum."""

    def __init__(self) -> None:
        self.path = _ledger_path()

    def _book(self) -> dict[str, Any]:
        return read_json(self.path) if self.path.exists() else {"starts": 0, "entries": []}

    def reserve(self, kind: str, label: str, seed: int, attempt: int = 1) -> int:
        used = ledger_total()
        if used + 1 > LAUNCH_CAP:
            raise RuntimeError(
                f"launch cap {LAUNCH_CAP} reached at {used}; refusing a {kind} start "
                f"({label} s{seed} a{attempt})"
            )
        book = self._book()
        book["starts"] += 1
        book["entries"].append({
            "n": book["starts"], "round_total_before_this": used, "kind": kind,
            "label": label, "seed": int(seed), "attempt": int(attempt),
        })
        self.path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
        return used + 1

    def record(self, *, label: str, seed: int, attempt: int, ok: bool, detail: str) -> None:
        book = self._book()
        book.setdefault("results", []).append({
            "label": label, "seed": int(seed), "attempt": int(attempt), "ok": bool(ok),
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
    return {
        "normal_launches": NORMAL_LAUNCHES, "reserve_launches": RESERVE_LAUNCHES,
        "hard_cap": LAUNCH_CAP, "used": ledger_total(),
        "by_kind": per_kind, "by_label": per_label, "entries": entries, "results": results,
    }


# ------------------------------------------------------------------ 配置继承与 reference 缓存 -------------
def stage_b_inherited_config(seed: int) -> dict[str, Any]:
    """The scenario's world fields, inherited the way Stage B inherited them.

    Stage B inherited each seed's archived D1 A3 run_config; the training/development scenarios
    here are new, so the world fields come from the Stage B frozen override set applied to the
    archived D1 A3 record of the *configuration source* (1100000), with only the scenario seed
    changed -- the same provenance rule the round-18 runner used for its new seeds.
    """
    import sp_common as sc

    source = sc.d1_archive_path("A3", 1100000)
    fields, substituted = sc.configuration_from_archive(sc.archived_configuration(source))
    fields.update(STAGE_B_INHERITED_OVERRIDES)
    fields["seed"] = int(seed)
    substituted.append(f"Stage B frozen overrides applied: {sorted(STAGE_B_INHERITED_OVERRIDES)}")
    return fields, substituted


def _rel(path: Path) -> str:
    """Repo-relative POSIX path, so keys are stable across platforms."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


def reference_cache_key(scenario_seed: int, code_identity: dict[str, Any],
                        behaviour_fields: dict[str, Any]) -> str:
    """The cache key of one immediate reference run (spec §4).

    Keyed by model / scenario initial state / configuration / code identity. The scenario is
    reproduced from (seed, behaviour configuration) -- Stage B proved that path is byte-stable
    (R_send≡I_send, P_init≡P_repeat), which is the caching precondition the spec demands.
    """
    payload = {
        "piloc_checkpoint": code_identity["checkpoints"][_rel(CHECKPOINT_FILES[0])],
        "ctde_checkpoint": code_identity["checkpoints"][_rel(CHECKPOINT_FILES[1])],
        "scenario_seed": int(scenario_seed),
        "behaviour_fields": behaviour_fields,
        "code_identity": code_identity["freeze_sha256"],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def code_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "production": {name: sha256_file(ROOT / name) for name in PRODUCTION_DEPENDENCIES},
        "imported_round_modules": {
            _rel(path): sha256_file(path) for path in IMPORTED_ROUND_MODULES
            if path.exists()},
        "harness": {name: sha256_file(HERE / name) for name in HARNESS_MODULES
                    if (HERE / name).exists()},
        "checkpoints": {_rel(path): sha256_file(path) for path in CHECKPOINT_FILES},
    }
    payload["freeze_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return payload
