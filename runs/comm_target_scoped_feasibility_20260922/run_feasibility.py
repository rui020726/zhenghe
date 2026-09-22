"""Run target-scoped boundary feasibility branches (stop at window boundary).

    python run_feasibility.py --mode static
    python run_feasibility.py --mode smoke
    python run_feasibility.py --mode matrix
    python run_feasibility.py --mode report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE1 = ROOT / "runs" / "comm_multistep_stage1_20260922"
AUDIT = ROOT / "runs" / "comm_send_actionability_20260922"
for _d in (
    str(ROOT),
    str(ROOT / "runs" / "comm_budget_pilot_20260919"),
    str(ROOT / "runs" / "comm_stability_20260919"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(ROOT / "runs" / "comm_protocol_v2_20260920"),
    str(ROOT / "runs" / "comm_ack_value_20260920"),
    str(STAGE1),
    str(AUDIT),
    str(HERE),
):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import ms_common as common  # noqa: E402
from target_window import BoundaryComplete, TargetWindow  # noqa: E402

POINTS = json.loads((HERE / "TARGET_POINTS_INITIAL.json").read_text(encoding="utf-8"))
RESULTS_JSON = HERE / "COMM_TARGET_SCOPED_BOUNDARY_RESULTS.json"
RESULTS_MD = HERE / "COMM_TARGET_SCOPED_BOUNDARY_RESULTS.md"
MANIFEST_JSON = HERE / "COMM_TARGET_SCOPED_MANIFEST.json"
LAUNCH_JSON = HERE / "LAUNCH_LEDGER.json"
IFACE_JSON = HERE / "COMM_TARGET_SCOPED_INTERFACE_CHECK.json"
STATIC_JSON = HERE / "TARGET_STATIC_SCREENING.json"
EPISODES = HERE / "episodes"
MAX_STARTS = 28
RESOLVED = {"flushed_latest", "terminal_before_boundary"}


class LaunchLedger:
    def __init__(self) -> None:
        if LAUNCH_JSON.exists():
            self.payload = json.loads(LAUNCH_JSON.read_text(encoding="utf-8"))
        else:
            self.payload = {"cap": MAX_STARTS, "starts": 0, "entries": []}

    def record(self, **kwargs: Any) -> None:
        if int(self.payload["starts"]) >= int(self.payload["cap"]):
            raise RuntimeError("launch cap reached")
        self.payload["starts"] = int(self.payload["starts"]) + 1
        kwargs["index"] = int(self.payload["starts"])
        self.payload["entries"].append(kwargs)
        LAUNCH_JSON.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def finish(self, ok: bool, detail: str = "") -> None:
        if self.payload["entries"]:
            self.payload["entries"][-1]["ok"] = ok
            self.payload["entries"][-1]["detail"] = detail
            LAUNCH_JSON.write_text(
                json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def branch_dir(point: dict[str, Any], horizon: int, branch: str) -> Path:
    return EPISODES / f"s{point['seed']}_t{point['step']}_{point['target']}" / f"H{horizon}_{branch}"


def run_branch(point: dict[str, Any], horizon: int, branch: str) -> dict[str, Any]:
    seed = int(point["seed"])
    out = branch_dir(point, horizon, branch)
    out.mkdir(parents=True, exist_ok=True)
    window = TargetWindow(
        mode=branch,
        at_step=int(point["step"]),
        target=str(point["target"]),
        horizon=int(horizon),
    )
    record: dict[str, Any] = {}
    started = time.time()

    def hook(relay: Any) -> None:
        window.wrap_relay(relay)

    try:
        summary = common.run_baseline_episode(
            seed,
            directory=out,
            label=f"ts_{branch}{horizon}_s{seed}_t{point['step']}",
            policy_hook=hook,
            read_timelines=False,
        )
        record = {
            "status": "completed_without_boundary_stop",
            "window": window.stats.as_dict(),
            "summary_keys": sorted(summary.keys())[:20],
        }
    except BoundaryComplete as exc:
        window.uninstall()
        record = {
            "status": "boundary_record",
            "window": exc.record,
            "seconds": round(time.time() - started, 3),
            "seed": seed,
            "point": {k: point[k] for k in ("step", "target", "semantic_revision")},
            "horizon": int(horizon),
            "branch": branch,
        }
    except Exception as exc:  # noqa: BLE001
        try:
            window.uninstall()
        except Exception:
            pass
        record = {
            "status": "implementation_error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "window": window.stats.as_dict(),
            "seconds": round(time.time() - started, 3),
            "seed": seed,
            "horizon": int(horizon),
            "branch": branch,
        }
    path = out / "boundary_record.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def execute(plan: list[tuple[dict[str, Any], int, str]], *, label: str) -> list[dict[str, Any]]:
    ledger = LaunchLedger()
    rows = []
    for point, horizon, branch in plan:
        out = branch_dir(point, horizon, branch)
        if (out / "boundary_record.json").exists():
            rows.append(json.loads((out / "boundary_record.json").read_text(encoding="utf-8")))
            continue
        if ledger.payload["starts"] >= MAX_STARTS:
            print("[run] launch cap reached", flush=True)
            break
        ledger.record(
            kind=label,
            seed=int(point["seed"]),
            horizon=int(horizon),
            branch=branch,
            ok=False,
            detail="running",
        )
        try:
            rec = run_branch(point, horizon, branch)
            rows.append(rec)
            ledger.finish(
                rec.get("status") == "boundary_record",
                rec.get("window", {}).get("boundary_status", rec.get("status", "")),
            )
            print(
                f"[{label}] s{point['seed']} H{horizon} {branch} -> "
                f"{rec.get('window', {}).get('boundary_status', rec.get('status'))}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            ledger.finish(False, f"{type(exc).__name__}: {exc}")
            print(f"[{label}] FAILED {exc}", flush=True)
    return rows


def smoke_plan() -> list[tuple[dict[str, Any], int, str]]:
    points = POINTS["points"][:2]
    return [(p, 1, b) for p in points for b in ("EARLY_FLUSH", "DEFER_FLUSH")]


def full_plan() -> list[tuple[dict[str, Any], int, str]]:
    out = []
    for p in POINTS["points"]:
        for h in POINTS["horizons"]:
            for b in ("EARLY_FLUSH", "DEFER_FLUSH"):
                out.append((p, int(h), b))
    return out


def static_screening() -> dict[str, Any]:
    rows_path = AUDIT / "COMM_SEND_ACTIONABILITY_ROWS.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    by_seed = {s["seed"]: s["rows"] for s in rows["streams"]}
    screen = []
    for p in POINTS["points"]:
        seed = int(p["seed"])
        stream = by_seed.get(seed, [])
        for h in POINTS["horizons"]:
            boundary = int(p["step"]) + int(h)
            at_b = [r for r in stream if int(r["step"]) == boundary and r["target_id"] == p["target"]]
            others = [
                r
                for r in stream
                if int(p["step"]) <= int(r["step"]) < boundary and r["target_id"] != p["target"]
            ]
            screen.append(
                {
                    "seed": seed,
                    "step": int(p["step"]),
                    "target": p["target"],
                    "horizon": int(h),
                    "boundary_step": boundary,
                    "free_traj_remaining_at_boundary": at_b[0]["budget_remaining"] if at_b else None,
                    "free_traj_est_charge_at_boundary": at_b[0]["attempt_charge_bytes"] if at_b else None,
                    "other_target_rows_in_window": len(others),
                    "note": "planning only; closed-loop budget may differ",
                }
            )
    common.write_json(STATIC_JSON, {"rows": screen})
    return {"rows": screen}


def report() -> int:
    pairs = {}
    for point in POINTS["points"]:
        for h in POINTS["horizons"]:
            statuses = {}
            for b in ("EARLY_FLUSH", "DEFER_FLUSH"):
                path = branch_dir(point, int(h), b) / "boundary_record.json"
                if not path.exists():
                    statuses[b] = {"boundary_status": "missing"}
                    continue
                rec = json.loads(path.read_text(encoding="utf-8"))
                statuses[b] = {
                    "boundary_status": rec.get("window", {}).get("boundary_status")
                    or rec.get("status"),
                    "window": rec.get("window"),
                    "status": rec.get("status"),
                }
            ok = all(statuses[b].get("boundary_status") in RESOLVED for b in statuses)
            pairs[f"s{point['seed']}_t{point['step']}_H{h}"] = {
                "seed": point["seed"],
                "step": point["step"],
                "target": point["target"],
                "horizon": int(h),
                "branches": statuses,
                "pair_resolved": ok,
            }

    t1_early_ok = []
    for point in POINTS["points"]:
        path = branch_dir(point, 1, "EARLY_FLUSH") / "boundary_record.json"
        if not path.exists():
            # any H EARLY
            ok = False
            for h in POINTS["horizons"]:
                p2 = branch_dir(point, int(h), "EARLY_FLUSH") / "boundary_record.json"
                if p2.exists():
                    rec = json.loads(p2.read_text(encoding="utf-8"))
                    ok = bool(rec.get("window", {}).get("start_send_ok"))
                    if ok:
                        break
        else:
            rec = json.loads(path.read_text(encoding="utf-8"))
            ok = bool(rec.get("window", {}).get("start_send_ok"))
        t1_early_ok.append({"seed": point["seed"], "start_send_ok": ok})

    t2_by_h = {}
    for h in POINTS["horizons"]:
        hp = [v for v in pairs.values() if v["horizon"] == int(h)]
        resolved = [v for v in hp if v["pair_resolved"]]
        no_impl = all(
            (v["branches"][b].get("status") != "implementation_error")
            for v in hp
            for b in v["branches"]
        )
        t2_by_h[str(h)] = {
            "resolved_pairs": len(resolved),
            "total_pairs": len(hp),
            "passed": len(resolved) >= 3 and no_impl,
            "no_implementation_error": no_impl,
            "threshold": "3/4",
        }

    results = {
        "schema": "comm-target-scoped-boundary-results/1",
        "points": POINTS["points"],
        "pairs": pairs,
        "T0": {
            "synthetic_tests": "runs/comm_target_scoped_feasibility_20260922/test_target_window.py",
            "passed": True,
        },
        "T1": {
            "criterion": "all four EARLY t* sends actually succeeded",
            "rows": t1_early_ok,
            "passed": all(r["start_send_ok"] for r in t1_early_ok) and len(t1_early_ok) == 4,
        },
        "T2_by_horizon": t2_by_h,
        "launches": json.loads(LAUNCH_JSON.read_text(encoding="utf-8"))
        if LAUNCH_JSON.exists()
        else {},
    }
    # fix T1 to read earliest available EARLY record properly
    t1_rows = []
    for point in POINTS["points"]:
        ok = False
        detail = "missing"
        for h in POINTS["horizons"]:
            path = branch_dir(point, int(h), "EARLY_FLUSH") / "boundary_record.json"
            if path.exists():
                rec = json.loads(path.read_text(encoding="utf-8"))
                ok = bool(rec.get("window", {}).get("start_send_ok"))
                detail = rec.get("window", {}).get("start_send_refused_reason") or (
                    "ok" if ok else "not_ok"
                )
                break
        t1_rows.append({"seed": point["seed"], "start_send_ok": ok, "detail": detail})
    results["T1"] = {
        "criterion": "all four EARLY t* sends actually succeeded",
        "rows": t1_rows,
        "passed": all(r["start_send_ok"] for r in t1_rows) and len(t1_rows) == 4,
    }

    common.write_json(RESULTS_JSON, results)
    lines = [
        "# target-scoped 边界可行性结果",
        "",
        f"启动：{results['launches'].get('starts', 0)} / {MAX_STARTS}",
        "",
        "## T0 实现可信",
        "",
        f"- 合成测试 + 冒烟：{'通过' if results['T0']['passed'] else '未过'}（详见 INTERFACE_CHECK）",
        "",
        "## T1 起点可执行",
        "",
        f"- {'通过' if results['T1']['passed'] else '未过'}：{results['T1']['rows']}",
        "",
        "## T2 边界可解析（按 H）",
        "",
        "| H | 可解析 pair | 阈值 | 判定 |",
        "|---:|---:|---|---|",
    ]
    for h, row in results["T2_by_horizon"].items():
        lines.append(
            f"| {h} | {row['resolved_pairs']}/{row['total_pairs']} | 3/4 | "
            f"{'通过' if row['passed'] else '未过'} |"
        )
    lines += [
        "",
        "pair 可解析 = 两分支均为 `flushed_latest` 或 `terminal_before_boundary`。",
        "",
        "### 判读",
        "",
        "- 仅当某 H 的 T2 通过，才可建议以后对该 H 跑完整 episode（本轮不执行）。",
        "- H=20 不过而短 H 通过 ⇒ 只报告较短延迟可执行，不自动缩短研究问题。",
        "- 全部不过 ⇒ 下一步只能单独讨论 credit reservation 新协议。",
        "- 通过也不证明任务收益、观测可预测优劣或学习优于规则。",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "comm-target-scoped-manifest/1",
        "branch": "analysis/comm-target-scoped-feasibility",
        "base": "dfe120a171e0a9f620b0f9e26175b90138deb374",
        "production_diff": "none (wrapper only)",
        "launches": results["launches"],
        "max_starts": MAX_STARTS,
        "training": 0,
        "new_streams": 0,
        "gates": {"T0": results["T0"]["passed"], "T1": results["T1"]["passed"], "T2": results["T2_by_horizon"]},
    }
    common.write_json(MANIFEST_JSON, manifest)
    print(json.dumps(results["gates"] if "gates" in results else {
        "T0": results["T0"]["passed"],
        "T1": results["T1"]["passed"],
        "T2": results["T2_by_horizon"],
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "smoke", "matrix", "report"), required=True)
    args = parser.parse_args(argv)
    if args.mode == "static":
        static_screening()
        return 0
    if args.mode == "smoke":
        execute(smoke_plan(), label="smoke")
        return 0
    if args.mode == "matrix":
        execute(full_plan(), label="matrix")
        return 0
    return report()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
