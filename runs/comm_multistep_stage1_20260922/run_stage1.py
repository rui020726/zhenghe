"""Multistep diagnostic stage-one runner.

Modes:
  free          run free baselines for frozen points and write pre-action digests
  validate      freeze POINTS.json from free results (before any intervention)
  intervene     run S/H branches (first point's four branches gate the rest)
  report        write COMM_MULTISTEP_RESULTS.md/.json from per-point products

Every simulation start is counted in LAUNCH_LEDGER.json and the cap is enforced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import ms_common as common
from window_intervention import WindowIntervention, digests_equal, snapshot_relay_state


class LaunchLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self.payload = common.read_json(path)
        else:
            self.payload = {
                "schema": "comm-multistep-launch-ledger/1",
                "cap": common.MAX_SIMULATION_STARTS,
                "starts": 0,
                "entries": [],
            }

    def remaining(self) -> int:
        return int(self.payload["cap"]) - int(self.payload["starts"])

    def record(self, *, kind: str, seed: int, label: str, ok: bool, detail: str = "") -> None:
        if int(self.payload["starts"]) >= int(self.payload["cap"]):
            raise RuntimeError(
                f"simulation start cap reached ({self.payload['cap']}); refusing {label}"
            )
        self.payload["starts"] = int(self.payload["starts"]) + 1
        self.payload["entries"].append(
            {
                "index": int(self.payload["starts"]),
                "kind": kind,
                "seed": int(seed),
                "label": label,
                "ok": bool(ok),
                "detail": detail,
                "seconds": None,
            }
        )
        common.write_json(self.path, self.payload)

    def finish_last(self, seconds: float, ok: bool, detail: str = "") -> None:
        if not self.payload["entries"]:
            return
        self.payload["entries"][-1]["seconds"] = round(float(seconds), 3)
        self.payload["entries"][-1]["ok"] = bool(ok)
        if detail:
            self.payload["entries"][-1]["detail"] = detail
        common.write_json(self.path, self.payload)


def point_dir(row: dict[str, Any], horizon: int | None = None, branch: str | None = None) -> Path:
    seed = int(row["seed"])
    step = int(row["step"])
    base = common.POINTS_DIR / f"s{seed}_t{step}"
    if horizon is None:
        return base
    return base / f"H{horizon}_{branch}"


def load_points_initial() -> dict[str, Any]:
    return common.read_json(common.POINTS_INITIAL)


def load_points() -> dict[str, Any]:
    return common.read_json(common.POINTS_JSON)


def run_free(args: argparse.Namespace) -> int:
    initial = load_points_initial()
    ledger = LaunchLedger(common.EPISODES / "LAUNCH_LEDGER.json")
    selected = [row for row in initial["rows"] if row.get("status") == "selected"]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    for row in selected:
        seed = int(row["seed"])
        out_dir = point_dir(row)
        out_dir.mkdir(parents=True, exist_ok=True)
        free_path = out_dir / "free.json"
        if free_path.exists() and not args.force:
            print(f"[free] skip s{seed} (exists)", flush=True)
            continue
        if ledger.remaining() <= 0:
            print("[free] launch cap reached", flush=True)
            break
        label = f"ms_free_s{seed}_t{int(row['step'])}"
        ledger.record(kind="free", seed=seed, label=label, ok=False, detail="running")
        started = time.time()
        try:
            digest, candidate, summary = run_free_with_capture(seed, row, label=label)
            payload = {
                "seed": seed,
                "point": common.strip_outcome_fields(row),
                "free_summary": {
                    key: summary[key]
                    for key in (
                        "seed",
                        "label",
                        "on_time",
                        "targets",
                        "uplink_bytes",
                        "ack_bytes",
                        "total_bytes",
                        "emissions",
                        "spent_bytes",
                        "budget_used_fraction",
                        "return_on_time",
                        "task",
                        "ledger",
                    )
                    if key in summary
                },
                "pre_action_digest": digest,
                "candidate_at_tstar": candidate,
                "old_transcript_status": "not_used_as_byte_baseline",
            }
            # Prefer the richer keys from protocol_env when present.
            for key in ("on_time", "targets", "uplink_bytes", "ack_bytes", "total_bytes", "emissions"):
                if key not in payload["free_summary"] and key in summary:
                    payload["free_summary"][key] = summary[key]
            if "task" in summary:
                payload["free_summary"]["task"] = summary["task"]
            if "ledger" in summary:
                payload["free_summary"]["ledger"] = summary["ledger"]
            if "return_on_time" in summary:
                payload["free_summary"]["on_time"] = summary["return_on_time"]
            common.write_json(free_path, payload)
            ledger.finish_last(time.time() - started, True, "ok")
            print(f"[free] s{seed} done in {time.time() - started:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - technical failure is ledgered
            detail = f"{type(exc).__name__}: {exc}"
            common.write_json(
                out_dir / "free_error.json",
                {"seed": seed, "error": detail, "traceback": traceback.format_exc()},
            )
            ledger.finish_last(time.time() - started, False, detail)
            print(f"[free] s{seed} FAILED {detail}", flush=True)
            if args.stop_on_error:
                return 1
    return 0


def run_free_with_capture(
    seed: int, row: dict[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """One free episode that also captures the pre-action digest at t* (single launch)."""
    from integrated_sar.comm_policy import build_comm_policy

    holder: dict[str, Any] = {}
    inner = build_comm_policy(common.POLICY_NAME, common.policy_params_cli())

    class Recorder:
        name = "pre_action_recorder"
        orders_selection = True

        def __init__(self, wrapped: Any) -> None:
            self._inner = wrapped

        def parameters(self) -> dict[str, Any]:
            values = dict(self._inner.parameters())
            values["recorder"] = "pre_action"
            return values

        def decide(self, observation: Any) -> tuple[str, ...]:
            step = int(observation.step)
            chosen = self._inner.decide(observation)
            if step == int(row["step"]) and "candidate" not in holder:
                selected = set(chosen)
                for item in observation.items:
                    if str(item.target_id) == str(row["target"]):
                        holder["candidate"] = {
                            "step": step,
                            "target": str(item.target_id),
                            "confirmed_fraction": float(item.ack_confirmed_fraction),
                            "unacked": int(item.ack_unacked_broadcasts),
                            "age_steps": int(item.age_steps),
                            "changed_view": bool(item.changed_view),
                            "rule_sent": str(item.target_id) in selected,
                            "revision": int(item.revision),
                        }
                        break
                holder["observation_digest"] = snapshot_relay_state_from_observation(
                    observation
                )
            return chosen

    def hook(relay: Any) -> None:
        original = relay._emit_selected

        def wrapped() -> None:
            if int(relay._tick) == int(row["step"]) and "relay_digest" not in holder:
                holder["relay_digest"] = snapshot_relay_state(relay)
            return original()

        relay._emit_selected = wrapped

    summary = common.run_baseline_episode(
        seed,
        directory=common.EPISODES / "free",
        label=label,
        controller=Recorder(inner),
        policy_hook=hook,
        read_timelines=True,
    )
    digest = holder.get("relay_digest") or holder.get("observation_digest") or {}
    if "observation_digest" in holder and "relay_digest" in holder:
        digest = {
            "relay": holder["relay_digest"],
            "observation": holder["observation_digest"],
        }
    return digest, holder.get("candidate") or {}, summary


def snapshot_relay_state_from_observation(observation: Any) -> dict[str, Any]:
    return {
        "observation_step": int(observation.step),
        "current_time": float(observation.current_time),
        "remaining_budget_bytes": float(observation.remaining_budget_bytes),
        "capacity_remaining": observation.capacity_remaining,
        "backlog_targets": int(observation.backlog_targets),
        "subscribers": int(observation.subscribers),
        "items": [
            {
                "target": item.target_id,
                "revision": int(item.revision),
                "changed_view": bool(item.changed_view),
                "first_emission": bool(item.first_emission),
                "pending_event": bool(item.pending_event),
                "confirmed_fraction": float(item.ack_confirmed_fraction),
                "unacked": int(item.ack_unacked_broadcasts),
                "age_steps": int(item.age_steps),
                "sent_count": int(item.sent_count),
            }
            for item in observation.items
        ],
    }


def validate_points(args: argparse.Namespace) -> int:
    initial = load_points_initial()
    rows_out = []
    problems = []
    for row in initial["rows"]:
        if row.get("status") != "selected":
            rows_out.append(row)
            continue
        seed = int(row["seed"])
        free_path = point_dir(row) / "free.json"
        if not free_path.exists():
            rows_out.append(
                {
                    **row,
                    "status": "free_failed_or_missing",
                }
            )
            problems.append({"seed": row["seed"], "reason": "free_missing"})
            continue
        free = common.read_json(free_path)
        candidate = free.get("candidate_at_tstar") or {}
        ok = True
        reasons = []
        if str(candidate.get("target")) != str(row["target"]):
            ok = False
            reasons.append("target_mismatch")
        if int(candidate.get("step", -1)) != int(row["step"]):
            ok = False
            reasons.append("step_mismatch")
        if not common.eligible(
            {
                "confirmed_fraction": candidate.get("confirmed_fraction", -1),
                "changed_view": candidate.get("changed_view", False),
            }
        ):
            ok = False
            reasons.append("eligibility_failed_at_tstar")
        # Communication legality: candidate present and changed_view; hard budget is checked
        # at emit. We do not use allocation masks.
        if candidate.get("changed_view") is not True:
            ok = False
            reasons.append("not_a_communication_candidate")
        status = "validated" if ok else "invalid_at_tstar"
        rows_out.append(
            {
                **row,
                "status": status,
                "validation_reasons": reasons,
                "candidate_at_tstar": candidate,
                "pre_action_digest": free.get("pre_action_digest") or {},
            }
        )
        if not ok:
            problems.append({"seed": row["seed"], "reasons": reasons})
    payload = {
        "schema": "comm-multistep-points/1",
        "baseline_rule": common.BASELINE_RULE,
        "stream_sample_seed": common.STREAM_SAMPLE_SEED,
        "point_selection_seed": common.POINT_SELECTION_SEED,
        "horizons": list(common.HORIZONS),
        "max_simulation_starts": common.MAX_SIMULATION_STARTS,
        "rows": rows_out,
        "problems": problems,
        "validated_count": sum(1 for row in rows_out if row.get("status") == "validated"),
    }
    common.write_json(common.POINTS_JSON, payload)
    print(
        json.dumps(
            {
                "validated": payload["validated_count"],
                "problems": problems,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


def run_interventions(args: argparse.Namespace) -> int:
    points = load_points()
    ledger = LaunchLedger(common.EPISODES / "LAUNCH_LEDGER.json")
    validated = [row for row in points["rows"] if row.get("status") == "validated"]
    if not validated:
        print("[intervene] no validated points", flush=True)
        return 1
    interface_ok = True
    first = True
    for row in validated:
        if args.limit is not None and first is False:
            pass
        seed = int(row["seed"])
        step = int(row["step"])
        target = str(row["target"])
        for horizon in common.HORIZONS:
            for branch in ("S", "H"):
                out_dir = point_dir(row, horizon, branch)
                out_dir.mkdir(parents=True, exist_ok=True)
                result_path = out_dir / "result.json"
                if result_path.exists() and not args.force:
                    print(f"[intervene] skip s{seed} t{step} H{horizon} {branch}", flush=True)
                    continue
                if ledger.remaining() <= 0:
                    print("[intervene] launch cap reached", flush=True)
                    common.write_json(
                        common.HERE / "INTERFACE_CHECK.json",
                        {"interface_ok": interface_ok, "stopped_reason": "launch_cap"},
                    )
                    return 0 if interface_ok else 1
                label = f"ms_{branch}{horizon}_s{seed}_t{step}"
                ledger.record(
                    kind="intervene",
                    seed=seed,
                    label=label,
                    ok=False,
                    detail="running",
                )
                started = time.time()
                try:
                    window = WindowIntervention(
                        mode=branch, at_step=step, target=target, horizon=int(horizon)
                    )

                    def hook(relay: Any, window=window) -> None:
                        window.wrap_relay(relay)

                    summary = common.run_baseline_episode(
                        seed,
                        directory=common.EPISODES / "intervene",
                        label=label,
                        policy_hook=hook,
                        read_timelines=True,
                    )
                    finally_stats = window.finalize()
                    window.uninstall()
                    product = {
                        "seed": seed,
                        "point": common.strip_outcome_fields(row),
                        "horizon": int(horizon),
                        "branch": branch,
                        "label": label,
                        "window": finally_stats,
                        "summary": {
                            key: summary[key]
                            for key in summary
                            if key
                            in (
                                "on_time",
                                "targets",
                                "uplink_bytes",
                                "ack_bytes",
                                "total_bytes",
                                "emissions",
                                "spent_bytes",
                                "budget_used_fraction",
                                "return_on_time",
                                "task",
                                "ledger",
                                "belief",
                            )
                            or key.endswith("_bytes")
                            or key.startswith("ack_")
                        },
                    }
                    if "return_on_time" in summary:
                        product["summary"]["on_time"] = summary["return_on_time"]
                    common.write_json(result_path, product)
                    ledger.finish_last(time.time() - started, True, "ok")
                    print(f"[intervene] {label} done {time.time() - started:.1f}s", flush=True)
                    if first and horizon == common.HORIZONS[-1] and branch == "H":
                        first = False
                        check = interface_check(row, ledger)
                        common.write_json(common.HERE / "INTERFACE_CHECK.json", check)
                        if not check.get("interface_ok"):
                            print("[intervene] interface check FAILED; stopping", flush=True)
                            return 1
                        print("[intervene] interface check passed", flush=True)
                except Exception as exc:  # noqa: BLE001
                    detail = f"{type(exc).__name__}: {exc}"
                    common.write_json(
                        out_dir / "error.json",
                        {"error": detail, "traceback": traceback.format_exc()},
                    )
                    try:
                        window.uninstall()  # type: ignore[name-defined]
                    except Exception:
                        pass
                    ledger.finish_last(time.time() - started, False, detail)
                    print(f"[intervene] {label} FAILED {detail}", flush=True)
                    interface_ok = False
                    common.write_json(
                        common.HERE / "INTERFACE_CHECK.json",
                        {"interface_ok": False, "stopped_reason": detail},
                    )
                    return 1
    common.write_json(
        common.HERE / "INTERFACE_CHECK.json",
        {"interface_ok": True, "stopped_reason": "completed"},
    )
    return 0


def interface_check(first_row: dict[str, Any], ledger: LaunchLedger) -> dict[str, Any]:
    """Mechanism gate after the first validated point's four branches."""
    seed = int(first_row["seed"])
    step = int(first_row["step"])
    problems = []
    branches = {}
    for horizon in common.HORIZONS:
        for branch in ("S", "H"):
            path = point_dir(first_row, horizon, branch) / "result.json"
            if not path.exists():
                problems.append(f"missing {path.name}")
                continue
            product = common.read_json(path)
            branches[f"H{horizon}_{branch}"] = product
            window = product.get("window") or {}
            if branch == "S" and int(window.get("forced_sends", 0)) > 1:
                problems.append(f"S{horizon} forced_sends>1")
            if branch == "S" and not window.get("force_applied") and not window.get("force_refused"):
                problems.append(f"S{horizon} force neither applied nor refused")
            if branch == "H" and int(window.get("forced_sends", 0)) != 0:
                problems.append(f"H{horizon} forced a send")
    free_path = point_dir(first_row) / "free.json"
    if free_path.exists():
        free = common.read_json(free_path)
        for key, product in branches.items():
            left = free.get("pre_action_digest") or {}
            right = (product.get("window") or {}).get("pre_action_digest") or {}
            if left and right:
                ok, diffs = digests_equal(left, right)
                if not ok:
                    problems.append(f"{key} pre_action_digest mismatch: {diffs[:3]}")
    return {
        "interface_ok": not problems,
        "problems": problems,
        "first_point": {"seed": seed, "step": step, "target": first_row.get("target")},
        "branches_present": sorted(branches),
        "launch_starts": ledger.payload.get("starts"),
    }


def report(args: argparse.Namespace) -> int:
    points = load_points()
    ledger_path = common.EPISODES / "LAUNCH_LEDGER.json"
    ledger = common.read_json(ledger_path) if ledger_path.exists() else {"starts": 0, "entries": []}
    rows = []
    for row in points["rows"]:
        if row.get("status") != "validated":
            rows.append({"point": row, "status": row.get("status")})
            continue
        entry: dict[str, Any] = {
            "seed": int(row["seed"]),
            "step": int(row["step"]),
            "target": str(row["target"]),
            "free": None,
            "branches": {},
        }
        free_path = point_dir(row) / "free.json"
        if free_path.exists():
            entry["free"] = common.read_json(free_path)
        for horizon in common.HORIZONS:
            for branch in ("S", "H"):
                path = point_dir(row, horizon, branch) / "result.json"
                key = f"H{horizon}_{branch}"
                if path.exists():
                    entry["branches"][key] = common.read_json(path)
                else:
                    entry["branches"][key] = {"status": "missing"}
        rows.append(entry)
    payload = {
        "schema": "comm-multistep-results/1",
        "baseline_rule": common.BASELINE_RULE,
        "frozen_config": common.read_json(common.FROZEN_CONFIG),
        "launch_ledger": ledger,
        "points": rows,
    }
    common.write_json(common.RESULTS_JSON, payload)
    lines = [
        "# 多步发送诊断第一阶段结果",
        "",
        f"日期：2026-09-22。基线规则 `{common.BASELINE_RULE}`。总仿真启动 "
        f"{ledger.get('starts', 0)} / {common.MAX_SIMULATION_STARTS}。",
        "",
        "## 启动记账",
        "",
        f"- starts: {ledger.get('starts', 0)}",
        f"- cap: {ledger.get('cap', common.MAX_SIMULATION_STARTS)}",
        "",
        "## 逐点",
        "",
    ]
    for entry in rows:
        if "branches" not in entry:
            lines.append(f"- 非 validated 点：{entry}")
            continue
        lines.append(
            f"### s{entry['seed']} t{entry['step']} {entry['target']}"
        )
        free = entry.get("free") or {}
        fs = free.get("free_summary") or {}
        lines.append(
            f"- free: on_time={fs.get('on_time')} total_bytes={fs.get('total_bytes')}"
        )
        for key, product in entry["branches"].items():
            if product.get("status") == "missing":
                lines.append(f"- {key}: missing")
                continue
            window = product.get("window") or {}
            summary = product.get("summary") or {}
            lines.append(
                f"- {key}: on_time={summary.get('on_time')} "
                f"total={summary.get('total_bytes')} "
                f"forced={window.get('forced_sends')} "
                f"blocked={window.get('window_blocked_requests')} "
                f"truncated={window.get('window_truncated')} "
                f"next_same={window.get('next_same_view_send_step')} "
                f"next_rev={window.get('next_revision_send_step')}"
            )
        lines.append("")
    lines.extend(
        [
            "## 判读边界",
            "",
            "- 只用逐点描述；不以 8 点 p 值作方法主张。",
            "- 前向 RNG 位移与窗口截断必须与任务差异并列。",
            "- 本实验是 ACK 规则机制诊断，不单独解释 no-ACK 模型失败。",
            "",
        ]
    )
    common.RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {common.RESULTS_JSON} and {common.RESULTS_MD}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("free", "validate", "intervene", "report"),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "free":
        return run_free(args)
    if args.mode == "validate":
        return validate_points(args)
    if args.mode == "intervene":
        return run_interventions(args)
    return report(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
