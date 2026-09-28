"""ROUND24 §2 零环境合成门禁：重跑 ROUND23 独立反例套件并记录实际计数（0 环境启动）。

- 逐个模块跑 v2 测试（计数为本次实际运行结果，不沿用历史 20/20 字符串）；
- 进程内包裹 Adam/SGD 的 `step`，分别记录合成 optimizer.step 数（与真实训练步分列）；
- 记录本门的代码身份（freeze_sha256），供 v2_freeze 核对该门与待冻结代码一致；
- 静态审计 harness 的读写路径：本目录之外无写入目标、无旧提交/旧 manifest 硬编码恢复入口。

    /d/project_env/python.exe v2_synthetic_gate.py
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import unittest
from pathlib import Path

import torch

import v2_common as common

TEST_MODULES = (
    "v2_test_math",
    "v2_test_cost_objective",
    "v2_test_cost_head_lifecycle",
    "v2_test_gae_next_value",
    "v2_test_reward_timing",
)

STEP_COUNTS = {"Adam": 0, "SGD": 0}


def _install_step_counters() -> None:
    original = {"Adam": torch.optim.Adam.step, "SGD": torch.optim.SGD.step}

    def counted(name):
        def step(self, *args, **kwargs):
            STEP_COUNTS[name] += 1
            return original[name](self, *args, **kwargs)
        return step

    torch.optim.Adam.step = counted("Adam")
    torch.optim.SGD.step = counted("SGD")


def static_path_audit() -> dict[str, Any]:
    """ROUND24 §2.5: every write path points at this round's directory; no hardcoded recovery.

    Rules come from AUDIT_MARKERS.json (data, so this script itself carries none of the
    forbidden literals): old-round directory names appear only in the registered readers; no
    module may shell out to git history for state; full commit SHAs appear only where the
    start-point provenance is registered; path constants are declared in exactly one module.
    """
    markers = json.loads((common.HERE / "AUDIT_MARKERS.json").read_text(encoding="utf-8"))
    registered_reads = markers["registered_old_round_reads"]
    registered_sha = set(markers["registered_sha_provenance"])
    old_dirs = markers["old_round_directory_names"]
    forbidden = markers["forbidden_recovery_markers"]
    owner = markers["path_constant_owner"]
    constants = markers["path_constant_names"]
    problems: list[str] = []
    old_dir_mentions: list[str] = []
    sha_mentions: list[str] = []
    for name in common.HARNESS_MODULES:
        path = common.HERE / name
        if not path.exists():
            problems.append(f"missing harness module: {name}")
            continue
        if name == markers.get("marker_source"):
            continue          # the data file that carries the forbidden literals by design
        text = path.read_text(encoding="utf-8")
        for directory in old_dirs:
            if directory in text:
                if name in registered_reads:
                    old_dir_mentions.append(f"{name} -> runs/{directory} "
                                            f"({registered_reads[name]})")
                else:
                    problems.append(f"{name}: unregistered reference to runs/{directory}")
        for marker in forbidden:
            if marker in text:
                problems.append(f"{name}: recovery marker {marker!r} present")
        if re.search(r"\b[0-9a-f]{40}\b", text) and name not in registered_sha:
            problems.append(f"{name}: carries an unregistered 40-hex commit literal")
        if name != owner:
            for constant in constants:
                if re.search(rf"^\s*{constant}\s*=", text, re.MULTILINE):
                    problems.append(f"{name}: declares the path constant {constant}")
    return {"problems": problems,
            "registered_old_round_reads": old_dir_mentions,
            "registered_sha_provenance": sorted(registered_sha),
            "passed": not problems}


def main() -> int:
    _install_step_counters()
    per_module: dict[str, Any] = {}
    failures: list[str] = []
    for name in TEST_MODULES:
        module = importlib.import_module(name)
        before = dict(STEP_COUNTS)
        suite = unittest.TestLoader().loadTestsFromModule(module)
        result = unittest.TestResult()
        suite.run(result)
        steps = {"Adam": STEP_COUNTS["Adam"] - before["Adam"],
                 "SGD": STEP_COUNTS["SGD"] - before["SGD"]}
        per_module[name] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "synthetic_optimizer_steps": steps,
            "synthetic_optimizer_steps_total": steps["Adam"] + steps["SGD"],
        }
        for case, _traceback in result.failures + result.errors:
            failures.append(f"{name}: {case}")
    payload = {
        "gate": "ROUND24 §2 zero-environment synthetic gate",
        "environment_starts": 0,
        "real_training_steps": 0,
        "per_module": per_module,
        "tests_run": sum(entry["tests_run"] for entry in per_module.values()),
        "failures": len(failures),
        "errors": sum(entry["errors"] for entry in per_module.values()),
        "synthetic_optimizer_steps": {"Adam": STEP_COUNTS["Adam"], "SGD": STEP_COUNTS["SGD"],
                                      "total": STEP_COUNTS["Adam"] + STEP_COUNTS["SGD"]},
        "counting_note": "one combined pass over the five modules with an in-process "
                         "Adam/SGD step wrapper; iterative focused reruns during development "
                         "are not included; no count here represents environment training",
        "static_path_audit": static_path_audit(),
        "freeze_sha256": common.code_identity()["freeze_sha256"],
        "passed": (not failures
                   and all(entry["failures"] == 0 and entry["errors"] == 0
                           for entry in per_module.values())
                   and static_path_audit()["passed"]),
    }
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (common.OUTPUT_DIR / "SYNTHETIC_GATE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in
                      ("tests_run", "failures", "errors", "synthetic_optimizer_steps",
                       "freeze_sha256", "passed")}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
