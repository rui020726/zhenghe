"""ROUND25 批次运行器：分片执行、可断点续跑、首 6 次技术门禁。

- 分片：按冻结 cell 列表确定性取模分配，互不重叠、不会双重分片；
- resume：已有 final row 的 cell 跳过（不重复消耗启动）；重试换 attempt+1 并重新记账；
- 门禁：首两个场景 × 三臂 = 6 次，包含在 192 内，通过后才自动继续余下 186 次；
- 技术失败不删不隔离：shard 遇错即停并留下失败账本，由启动方诊断后再决定是否用储备重试。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import r25_common as common
import r25_run as runner


def _learned_once() -> tuple[Any, dict[str, Any]]:
    """Load the frozen candidate controller once per process; the same weights serve every scene."""
    controller, identity = runner.load_candidate_controller()
    return controller, identity


def run_cells(cells: list[tuple[str, int]], *, attempt: int = 1, load_learned: bool = True,
              stop_on_error: bool = True) -> dict[str, Any]:
    controller = None
    controller_parameters = None
    if load_learned:
        controller, identity = _learned_once()
        controller_parameters = identity["controller_parameter_summary"]
    ran, skipped, failed = 0, 0, []
    for arm, scenario in cells:
        if (arm, int(scenario)) in common.done_cells() and attempt == 1:
            skipped += 1
            print(f"skip {arm} s{scenario}: final record exists", flush=True)
            continue
        try:
            row = runner.run_arm(arm, int(scenario), attempt=attempt, controller=controller,
                                 controller_parameters=controller_parameters)
        except Exception as error:  # noqa: BLE001  - recorded, then the shard stops
            detail = f"{type(error).__name__}: {error}"
            print(f"FAIL {arm} s{scenario} a{attempt}: {detail}", flush=True)
            failed.append({"arm": arm, "scenario": int(scenario), "attempt": attempt,
                           "error": detail})
            if stop_on_error:
                break
            continue
        ran += 1
        print(f"[{ran}] {arm} s{scenario} a{attempt}: on_time={row['on_time']} "
              f"bytes={row['spent_bytes']} emissions={row['emissions']}", flush=True)
    return {"ran": ran, "skipped": skipped, "failed": failed,
            "ledger_used": common.launch_count()}


def gate_cells() -> list[tuple[str, int]]:
    return [(arm, int(scenario)) for scenario in common.GATE_SCENARIOS for arm in common.ARM_ORDER]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="ROUND25 three-arm evaluation runner")
    parser.add_argument("mode", choices=("gate", "shard", "missing", "all"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args(argv)

    if args.mode == "gate":
        summary = run_cells(gate_cells(), attempt=args.attempt)
    elif args.mode == "shard":
        cells = common.shard_cells(args.index, args.shards)
        print(f"shard {args.index}/{args.shards}: {len(cells)} cells", flush=True)
        summary = run_cells(cells, attempt=args.attempt)
    elif args.mode == "missing":
        done = common.done_cells()
        cells = [cell for cell in common.all_cells() if cell not in done]
        print(f"missing cells: {len(cells)}", flush=True)
        summary = run_cells(cells, attempt=args.attempt)
    else:
        summary = run_cells(common.all_cells(), attempt=args.attempt)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
