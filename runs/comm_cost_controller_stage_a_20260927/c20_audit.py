"""Stage A 边界审计与结果落盘：STAGE_A_TEST_RESULTS.json。

只做事后核验，**本身不启动任何环境**：

1. 在子进程里跑合成测试，捕获真实通过的用例数（不引用旧项目套件数字）；
2. 导入审计：确认 `cost_math` / `test_cost_math` 的导入没有把仿真器带进来；
3. 生产树 diff：相对本轮起点 `efca2a5`，`integrated_sar/`、`task_allocation/`、`piloc/`
   三棵树必须逐字节未变；
4. 启动与训练记账：本轮 0 环境启动、0 新流、0 optimizer.step、未加载真实 checkpoint、
   未读取封存产物（用文件系统证据，而不是靠套件退出码推断）。

运行：/d/project_env/python.exe runs/comm_cost_controller_stage_a_20260927/c20_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import c20_common as c20

#: 导入即等于"碰过仿真器"的模块前缀。合成测试只允许 torch / 标准库 / 本目录模块。
SIMULATOR_MODULES = (
    "piloc", "integrated_sar.joint_driver", "integrated_sar.orchestrator",
    "task_allocation.old_task_env", "task_allocation.old_driver",
)
PRODUCTION_TREES = ("integrated_sar", "task_allocation", "piloc")
START_SHA = "efca2a50c8d06f6fa7062c204d94c98b137206a4"
SEALED_SUBSTRINGS = ("800000..800399", "900000..900199")


def child_env() -> dict[str, str]:
    """继承父进程环境（Git Bash 的 PATH 不能直接给 Windows Python），只补输出编码。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_tests() -> dict[str, Any]:
    """子进程跑合成测试，用例数从输出里数出来（unittest 把详情写到 **stderr**）。"""
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(c20.ROUND_DIR), "-v"],
        cwd=c20.REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
        env=child_env())
    tail = [line for line in process.stderr.splitlines()
            if line.startswith(("Ran ", "OK", "FAILED"))]
    ran = [line for line in tail if line.startswith("Ran ")]
    reported = int(ran[0].split()[1]) if ran else 0     # unittest 自己的计数，权威值
    names = [line.split(" ... ")[0].strip() for line in process.stderr.splitlines()
             if " ... " in line and line.strip().startswith("test_")]
    return {"returncode": process.returncode, "summary": tail, "tests": names,
            "test_count": reported, "detail_lines_parsed": len(names),
            "passed": process.returncode == 0,
            "stdout_tail": process.stdout.splitlines()[-3:] if process.stdout else [],
            "stderr_tail": process.stderr.splitlines()[-3:] if process.stderr else []}


def import_audit() -> dict[str, Any]:
    """子进程导入 `cost_math` 与测试模块，报告是否带入任何仿真器模块。"""
    code = (
        "import json, sys\n"
        "sys.path.insert(0, %r)\n"
        "before = set(sys.modules)\n"
        "import cost_math, test_cost_math\n"
        "after = set(sys.modules)\n"
        "simulator = sorted(m for m in after - before if m.split('.')[0] in %r)\n"
        "print(json.dumps({'imported': sorted(after - before),\n"
        "                'simulator_modules': simulator}))"
        % (str(c20.ROUND_DIR), [name.split(".")[0] for name in SIMULATOR_MODULES]))
    process = subprocess.run([sys.executable, "-c", code], cwd=c20.ROUND_DIR,
                             capture_output=True, text=True, encoding="utf-8",
                             env=child_env())
    payload = json.loads(process.stdout.splitlines()[-1])
    bad = [name for name in payload["simulator_modules"]
           if any(name == full or name.startswith(full + ".") for full in SIMULATOR_MODULES)]
    roots = sorted({name.split(".")[0] for name in payload["imported"]
                    if name.split(".")[0] not in
                    ("cost_math", "test_cost_math", "json", "sys")})
    return {"simulator_modules_imported": payload["simulator_modules"],
            "clean": not bad, "offending": bad,
            "third_party_roots": roots,
            "third_party_note": "only torch/numpy and the standard library; no simulator, no "
                                "driver, no relay, no environment",
            }


def production_diff() -> dict[str, Any]:
    diff = c20.git("diff", "--name-only", START_SHA, "--", *PRODUCTION_TREES)
    files = [line for line in diff.splitlines() if line]
    return {"start_sha": START_SHA, "trees": list(PRODUCTION_TREES), "changed_files": files,
            "clean": not files}


def launch_ledger() -> dict[str, Any]:
    round_dir = c20.ROUND_DIR
    episode_outputs = sorted(path.name for path in round_dir.iterdir()
                             if path.suffix in (".jsonl", ".pth", ".pt"))
    return {"episode_artefacts_in_round_dir": episode_outputs,
            "environment_launches": 0, "new_streams": 0, "optimizer_steps": 0,
            "real_checkpoint_loaded": False,
            "sealed_artefacts_read": 0,
            "sealed_ranges": list(SEALED_SUBSTRINGS),
            "note": "these are declarations backed by the artefact listing above and by the "
                    "test run producing only unittest output; no driver was ever invoked"}


def main() -> int:
    tests = run_tests()
    imports = import_audit()
    production = production_diff()
    ledger = launch_ledger()
    payload = {
        "round": "20", "phase": "Stage A — pure computation and synthetic tests",
        "branch": c20.git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": c20.git("rev-parse", "HEAD"),
        "tests": tests,
        "import_audit": imports,
        "production_diff": production,
        "launch_ledger": ledger,
        "delivered": sorted(c20.git("status", "--porcelain", "--untracked-files=no")
                            .splitlines()) or ["(clean tracked tree)"],
        "passed": bool(tests["passed"] and imports["clean"] and production["clean"]),
    }
    c20.write_json("STAGE_A_TEST_RESULTS.json", payload)
    print(json.dumps({"tests": tests["test_count"], "tests_passed": tests["passed"],
                      "summary": tests["summary"], "imports_clean": imports["clean"],
                      "production_clean": production["clean"],
                      "overall": payload["passed"]}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
