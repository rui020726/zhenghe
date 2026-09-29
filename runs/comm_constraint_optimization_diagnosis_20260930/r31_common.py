"""ROUND31 公共件：路径、行身份核验、checkpoint 参数摘要独立重算。

本轮是**只读诊断**：0 环境启动、0 新流、0 真实训练、0 optimizer.step（含合成）。
所有输入都是 ROUND30 已落盘的产物；本文件不重新定义任何训练数学，也不向 ROUND30 目录写入。

复用而非重写：常量、场景块、数据顺序、编码臂定义全部从 ROUND30/ROUND24 的冻结件 import
（`r30_common` 会经由 `v2_common.enter_round()` 装好共享 sys.path）。

行身份（执行依据 §5："用行内身份核对，不靠宽泛 glob 或文件名标签推断"）：文件名只当作
**候选键**；每次读取都断言行内 `kind/arm/training_seed/episode_index/scenario_seed/
eval_mode/policy/feature_layout/attempt` 与该键逐项一致，任何不一致都记入 problems 而不静默。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

R30_DIR = ROOT / "runs" / "comm_deadline_encoding_ablation_20260930"
R30_ROWS = R30_DIR / "rows"
R30_LOGS = R30_DIR / "logs"
R30_CHECKPOINTS = R30_DIR / "checkpoints"

for _shared in (str(R30_DIR), str(ROOT / "runs" / "comm_cost_v2_rerun_20260928")):
    if _shared not in sys.path:
        sys.path.insert(0, _shared)

import r30_common as c30  # noqa: E402  (installs the remaining shared sys.path via v2_common)
import v2_common as v2c  # noqa: E402

ARMS = ("old", "new")
TRAIN_SEEDS = tuple(c30.TRAIN_SEEDS)
DEVELOPMENT_SCENARIOS = tuple(c30.DEVELOPMENT_SCENARIOS)
TRAINING_SCENARIOS = tuple(c30.TRAINING_SCENARIOS)
B0_BYTES = c30.B0_BYTES
DELTA = c30.DELTA
ETA_LAMBDA = c30.ETA_LAMBDA
LAMBDA_INIT = c30.LAMBDA_INIT
EPISODES_PER_UPDATE = c30.EPISODES_PER_UPDATE
EPISODES_PER_SEED = c30.EPISODES_PER_SEED
LATE_RULE_ARM = c30.LATE_RULE_ARM
MODES = ("greedy", "sampled")

#: 本轮产物目录（与 ROUND30 分离，不覆盖任何历史结果）。
OUT = HERE

_ROW_PATTERNS = {
    "reference": re.compile(r"^r30_reference_s(?P<scenario>\d+)_a(?P<attempt>\d+)\.json$"),
    "rule": re.compile(r"^r30_(?P<rule>[a-z_]+)_s(?P<scenario>\d+)_a(?P<attempt>\d+)\.json$"),
    "eval": re.compile(r"^r30_eval_(?P<arm>old|new)_s(?P<seed>\d+)_final_(?P<mode>greedy|"
                       r"sampled)_s(?P<scenario>\d+)_a(?P<attempt>\d+)\.json$"),
    "train": re.compile(r"^r30_train_(?P<arm>old|new)_s(?P<seed>\d+)_e(?P<episode>\d+)"
                        r"_s(?P<scenario>\d+)_a(?P<attempt>\d+)\.json$"),
}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name: str, payload: Any) -> Path:
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = OUT / name
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ 行索引与身份核验 -------------
def filename_key(path: Path) -> dict[str, Any] | None:
    """Parse a row filename into its candidate identity key, or None if it matches no pattern."""
    for kind, pattern in _ROW_PATTERNS.items():
        match = pattern.match(path.name)
        if match:
            key = {"kind": kind}
            key.update({name: value for name, value in match.groupdict().items()})
            return key
    return None


def check_row_identity(path: Path, key: dict[str, Any]) -> list[str]:
    """Compare the row's inline identity with the filename's candidate key; return mismatches."""
    row = read_json(path)
    problems: list[str] = []

    def expect(field: str, value: Any) -> None:
        actual = row.get(field)
        if actual != value:
            problems.append(f"{path.name}: inline {field}={actual!r} != filename key {value!r}")

    kind = key["kind"]
    expect("kind", kind if kind != "rule" else "rule")
    expect("scenario_seed", int(key["scenario"]))
    expect("attempt", int(key["attempt"]))
    if kind == "reference":
        expect("policy", "immediate")
        expect("label", f"ref_s{key['scenario']}")
    elif kind == "rule":
        if key["rule"] != LATE_RULE_ARM:
            problems.append(f"{path.name}: unknown rule arm {key['rule']!r}")
        expect("policy", "learned")
        expect("label", f"rule_{key['rule']}_s{key['scenario']}")
        identity = row.get("rule_identity") or {}
        if identity.get("arm") != LATE_RULE_ARM or identity.get("policy") != LATE_RULE_ARM:
            problems.append(f"{path.name}: rule_identity={identity!r} does not name "
                            f"{LATE_RULE_ARM!r}")
    elif kind == "eval":
        expect("arm", key["arm"])
        expect("training_seed", int(key["seed"]))
        expect("eval_mode", key["mode"])
        expect("timepoint", "final")
        expect("eval_block", f"final_{key['mode']}")
        expect("policy", "learned")
        expect("feature_layout", c30.ARMS[key["arm"]]["feature_layout"])
        expect("label", f"eval_{key['arm']}_s{key['seed']}_final_{key['mode']}_s{key['scenario']}")
    elif kind == "train":
        expect("arm", key["arm"])
        expect("training_seed", int(key["seed"]))
        expect("episode_index", int(key["episode"]))
        expect("policy", "learned")
        expect("feature_layout", c30.ARMS[key["arm"]]["feature_layout"])
        # attempt == episode_index is the archived convention for training cells
        if int(key["attempt"]) != int(key["episode"]):
            problems.append(f"{path.name}: attempt {key['attempt']} != episode "
                            f"{key['episode']}")
        expect("label", f"train_{key['arm']}_s{key['seed']}_e{key['episode']}")
    return problems


def build_row_index(*, verify_identity: bool = True,
                    load_fields: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Read every ROUND30 row once and return the census plus per-cell identity problems."""
    files = sorted(R30_ROWS.glob("*.json"))
    index: dict[str, Any] = {}
    problems: list[str] = []
    unparsed: list[str] = []
    for path in files:
        key = filename_key(path)
        if key is None:
            unparsed.append(path.name)
            continue
        row = read_json(path)
        if verify_identity:
            problems.extend(check_row_identity(path, key))
        record = {"path": str(path), "key": key}
        if load_fields:
            record.update({field: row.get(field) for field in load_fields})
        else:
            record["row"] = row
        index[path.name] = record
    return {"files": len(files), "parsed": len(index), "unparsed": unparsed,
            "identity_problems": problems, "index": index}


def key_of(record: dict[str, Any]) -> tuple[str, ...]:
    key = record["key"]
    if key["kind"] == "eval":
        return ("eval", key["arm"], int(key["seed"]), key["mode"], int(key["scenario"]))
    if key["kind"] == "train":
        return ("train", key["arm"], int(key["seed"]), int(key["episode"]), int(key["scenario"]))
    if key["kind"] == "reference":
        return ("reference", int(key["scenario"]))
    return ("rule", key["rule"], int(key["scenario"]))


def index_by_key(index: dict[str, dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {key_of(record): record for record in index.values()}


# ------------------------------------------------------------------ batch 记录 -------------
def batch_path(arm: str, seed: int, batch: int) -> Path:
    return R30_CHECKPOINTS / f"r30_batch_{arm}_s{seed}_b{batch}.json"


# --------------------------------------------------- checkpoint 参数摘要独立重算 -------------
_MODULE_PREFIXES = ("trunk", "action_head", "value_task_head", "value_cost_head")


def parameter_sha256_from_payload(payload: dict[str, Any]) -> str:
    """Recompute `parameter_summary()['parameter_sha256']` from a saved payload.

    Same recipe as the frozen controller: sorted parameter names, then name bytes followed by
    the contiguous tensor bytes.  Pure computation -- no module is constructed, no weight is
    modified and nothing is written back.
    """
    saved = payload["controller"]
    tensors: dict[str, Any] = {}
    for prefix in _MODULE_PREFIXES:
        for name, tensor in saved[prefix].items():
            tensors[f"{prefix}.{name}"] = tensor
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def milestone_path(arm: str, seed: int, episodes: int) -> Path:
    return R30_CHECKPOINTS / f"r30_train_{arm}_s{seed}_ep{episodes}.pt"


def load_payload(path: Path) -> dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ 描述统计小工具 -------------
def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of an empty list")
    return sum(float(v) for v in values) / len(values)


def summary(values: list[float]) -> dict[str, Any]:
    floats = [float(v) for v in values]
    ordered = sorted(floats)
    n = len(ordered)
    average = mean(floats)

    def quantile(fraction: float) -> float:
        if n == 1:
            return ordered[0]
        position = (n - 1) * fraction
        low = int(position // 1)
        high = min(low + 1, n - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "n": n,
        "mean": round(average, 9),
        "min": ordered[0],
        "max": ordered[-1],
        "q25": round(quantile(0.25), 9),
        "median": round(quantile(0.5), 9),
        "q75": round(quantile(0.75), 9),
        "std_pop": round((sum((v - average) ** 2 for v in floats) / n) ** 0.5, 9),
        "abs_mean": round(mean([abs(v) for v in floats]), 9),
    }
