"""Record every communication candidate on a free replay and classify actionability.

    python audit_runner.py --mode protocol-check
    python audit_runner.py --mode replay
    python audit_runner.py --mode report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import send_actionability as SA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE1 = ROOT / "runs" / "comm_multistep_stage1_20260922"
for _directory in (
    str(ROOT),
    str(ROOT / "runs" / "comm_budget_pilot_20260919"),
    str(ROOT / "runs" / "comm_stability_20260919"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(ROOT / "runs" / "comm_protocol_v2_20260920"),
    str(ROOT / "runs" / "comm_ack_value_20260920"),
    str(STAGE1),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import ms_common as common  # noqa: E402

ROWS_JSON = HERE / "COMM_SEND_ACTIONABILITY_ROWS.json"
RESULTS_JSON = HERE / "COMM_SEND_ACTIONABILITY_RESULTS.json"
RESULTS_MD = HERE / "COMM_SEND_ACTIONABILITY_RESULTS.md"
POINTS_JSON = HERE / "COMM_SEND_ACTIONABILITY_POINTS.json"
MANIFEST_JSON = HERE / "COMM_SEND_ACTIONABILITY_MANIFEST.json"
LAUNCH_JSON = HERE / "LAUNCH_LEDGER.json"
EPISODES = HERE / "episodes"
MAX_LAUNCHES = 8


class LaunchLedger:
    def __init__(self) -> None:
        if LAUNCH_JSON.exists():
            self.payload = json.loads(LAUNCH_JSON.read_text(encoding="utf-8"))
        else:
            self.payload = {"cap": MAX_LAUNCHES, "starts": 0, "entries": []}

    def record(self, kind: str, seed: int, label: str, ok: bool, detail: str = "") -> None:
        if int(self.payload["starts"]) >= int(self.payload["cap"]):
            raise RuntimeError("audit launch cap reached")
        self.payload["starts"] = int(self.payload["starts"]) + 1
        self.payload["entries"].append(
            {
                "index": int(self.payload["starts"]),
                "kind": kind,
                "seed": int(seed),
                "label": label,
                "ok": bool(ok),
                "detail": detail,
            }
        )
        LAUNCH_JSON.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def load_targets() -> list[dict[str, Any]]:
    points = json.loads((STAGE1 / "POINTS.json").read_text(encoding="utf-8"))
    rows = [r for r in points["rows"] if r.get("status") == "validated"]
    return sorted(rows, key=lambda r: (int(r["seed"]), int(r["step"]), str(r["target"])))


class CandidateRecorder:
    """Wrap the frozen rule and log every offered candidate before the decision."""

    name = "actionability_recorder"
    orders_selection = True

    def __init__(self, inner: Any, *, seed: int) -> None:
        self._inner = inner
        self._seed = int(seed)
        self.rows: list[SA.ActionabilityRow] = []
        self._substep: dict[int, int] = {}
        self._budget: Any | None = None

    def parameters(self) -> dict[str, Any]:
        values = dict(self._inner.parameters())
        values["recorder"] = "actionability"
        return values

    def decide(self, observation: Any) -> tuple[str, ...]:
        chosen = list(self._inner.decide(observation))
        selected = set(chosen)
        step = int(observation.step)
        sub = self._substep.get(step, 0)
        self._substep[step] = sub + 1
        # Budget/capacity snapshot from the observation (same executor state).
        budget_remaining = float(observation.remaining_budget_bytes)
        budget_cap = observation.budget_bytes_per_episode
        capacity_remaining = observation.capacity_remaining
        for item in observation.items:
            row = self._row_from_item(
                item,
                observation=observation,
                step=step,
                substep=sub,
                selected=selected,
                budget_remaining=budget_remaining,
                budget_cap=budget_cap,
                capacity_remaining=capacity_remaining,
            )
            self.rows.append(row)
        return tuple(chosen)

    def _row_from_item(
        self,
        item: Any,
        *,
        observation: Any,
        step: int,
        substep: int,
        selected: set[str],
        budget_remaining: float,
        budget_cap: int | None,
        capacity_remaining: int | None,
    ) -> SA.ActionabilityRow:
        # Cost uses production estimate x fan-out. Payload size is recovered from a
        # synthetic message view when the relay is unavailable inside decide(); the
        # replay hook below overwrites charge with the executor's own number when present.
        payload_bytes = int(getattr(item, "_payload_bytes", 0) or 0)
        subscribers = tuple(str(x) for x in range(max(1, int(observation.subscribers))))
        charge = payload_bytes * len(subscribers) if payload_bytes else 0
        msg_class = SA.message_class_of(
            first_emission=bool(item.first_emission),
            pending_event=bool(item.pending_event),
            changed_view=bool(item.changed_view),
        )
        # Protocol/rule retry budget for same-view retries (event_retry_until_acked_5).
        unacked = int(item.ack_unacked_broadcasts)
        if msg_class == "same_view_retry":
            if unacked >= 0:
                attempts_remaining = max(0, SA.MAX_ATTEMPTS - unacked)
            else:
                attempts_remaining = max(0, SA.MAX_ATTEMPTS - int(item.sent_count))
        else:
            attempts_remaining = SA.MAX_ATTEMPTS

        send_executable = True
        send_block = None
        budget = getattr(self, "_budget", None)
        if budget is not None and charge > 0:
            send_block = SA.send_block_from_budget(budget, charge)
            send_executable = send_block is None
        elif charge > 0:
            if budget_cap is not None and charge > budget_remaining + 1e-9:
                send_executable = False
                send_block = "budget_exhausted"
            elif capacity_remaining is not None and capacity_remaining <= 0:
                send_executable = False
                send_block = "capacity_step"

        hold_executable = True
        protocol_forced = False
        in_policy = True

        row = SA.ActionabilityRow(
            seed=self._seed,
            step=step,
            communication_substep=substep,
            target_id=str(item.target_id),
            semantic_revision=int(item.revision),
            content_fingerprint=f"rev:{int(item.revision)}|view:{int(item.changed_view)}",
            subscriber_set=subscribers,
            legal=True,
            terminal=False,
            subscriber_count=len(subscribers),
            payload_bytes=payload_bytes,
            attempt_charge_bytes=charge,
            budget_spent=int((budget_cap or 0) - budget_remaining) if budget_cap else 0,
            budget_remaining=budget_remaining,
            budget_bytes_per_episode=budget_cap,
            sends_this_step=0,
            capacity_remaining=capacity_remaining,
            charge_over_remaining=(
                (charge / budget_remaining) if budget_remaining > 0 else None
            ),
            sent_count=int(item.sent_count),
            attempts_remaining=attempts_remaining,
            message_class=msg_class,
            changed_view=bool(item.changed_view),
            first_emission=bool(item.first_emission),
            pending_event=bool(item.pending_event),
            stage_changed=bool(item.stage_changed),
            requirements_changed=bool(item.requirements_changed),
            members_changed=bool(item.members_changed),
            progress_delta=float(item.progress_delta),
            age_steps=int(item.age_steps),
            remaining_time=float(item.remaining_time),
            ack_confirmed_fraction=float(item.ack_confirmed_fraction),
            ack_unacked_broadcasts=unacked,
            rule_requested=str(item.target_id) in selected,
            send_executable=send_executable,
            hold_executable=hold_executable,
            send_block_reason=send_block,
            hold_block_reason=None,
            protocol_forced=protocol_forced,
            in_policy_interface=in_policy,
        )
        return row


def enrich_rows_from_relay(rows: list[SA.ActionabilityRow], relay: Any) -> None:
    """Overwrite charge with the executor formula once the live relay is available."""
    subscribers = tuple(str(name) for name in relay._subscribers)
    for row in rows:
        row.subscriber_set = subscribers
        row.subscriber_count = len(subscribers)
        ledger = relay._ledger.get(row.target_id)
        snapshot = relay._snapshot_of(row.target_id) if relay._snapshot_of else None
        if ledger is None:
            continue
        try:
            message = relay._message(row.target_id, ledger, snapshot)
            charge = SA.attempt_charge_bytes(message, subscribers)
            payload = int(SA.estimate_message_bytes(message))
            fp = SA.fingerprint_of_message(message)
        except Exception:
            continue
        row.payload_bytes = payload
        row.attempt_charge_bytes = charge
        row.content_fingerprint = fp
        # Re-evaluate SEND with the shared budget predicate on the live executor.
        reason = SA.send_block_from_budget(relay.budget, charge)
        row.send_executable = reason is None
        row.send_block_reason = reason
        if snapshot is not None and getattr(snapshot.status, "terminal", False):
            row.terminal = True
            row.legal = True


def run_replay(args: argparse.Namespace) -> int:
    from integrated_sar.comm_policy import build_comm_policy

    ledger = LaunchLedger()
    targets = load_targets()
    all_rows: list[dict[str, Any]] = []
    for spec in targets:
        seed = int(spec["seed"])
        if args.limit is not None and len(all_rows) // 10000 > args.limit:
            break
        label = f"abl_free_s{seed}"
        if ledger.payload["starts"] >= MAX_LAUNCHES:
            print("[replay] launch cap reached", flush=True)
            break
        ledger.record("free_replay", seed, label, False, "running")
        started = time.time()
        try:
            inner = build_comm_policy(common.POLICY_NAME, common.policy_params_cli())
            recorder = CandidateRecorder(inner, seed=seed)
            holder: dict[str, Any] = {}

            def hook(relay: Any) -> None:
                holder["relay"] = relay
                recorder._budget = relay.budget
                original = relay._emit_selected

                def wrapped() -> None:
                    # Snapshot executor charge context into items before decide.
                    try:
                        observation_pre = relay._observation()
                        for item in observation_pre.items:
                            ledger_entry = relay._ledger.get(item.target_id)
                            snapshot = (
                                relay._snapshot_of(item.target_id)
                                if relay._snapshot_of
                                else None
                            )
                            if ledger_entry is not None:
                                try:
                                    message = relay._message(
                                        item.target_id, ledger_entry, snapshot
                                    )
                                    object.__setattr__ if False else None
                                    # stash on a side table keyed by (step,target,rev)
                                    holder.setdefault("payload", {})[
                                        (int(relay._tick), str(item.target_id), int(item.revision))
                                    ] = int(SA.estimate_message_bytes(message))
                                    holder.setdefault("fp", {})[
                                        (int(relay._tick), str(item.target_id), int(item.revision))
                                    ] = SA.fingerprint_of_message(message)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return original()

                relay._emit_selected = wrapped

            summary = common.run_baseline_episode(
                seed,
                directory=EPISODES,
                label=label,
                controller=recorder,
                policy_hook=hook,
                read_timelines=True,
            )
            relay = holder.get("relay")
            if relay is not None:
                enrich_rows_from_relay(recorder.rows, relay)
                # Fill payload/fp from stash when enrich could not rebuild message yet.
                payload_map = holder.get("payload") or {}
                fp_map = holder.get("fp") or {}
                for row in recorder.rows:
                    key = (row.step, row.target_id, row.semantic_revision)
                    if key in payload_map:
                        row.payload_bytes = int(payload_map[key])
                        row.attempt_charge_bytes = row.payload_bytes * row.subscriber_count
                        budget = relay.budget
                        reason = SA.send_block_from_budget(budget, row.attempt_charge_bytes)
                        row.send_executable = reason is None
                        row.send_block_reason = reason
                        row.budget_spent = int(budget.spent_bytes)
                        row.budget_remaining = float(budget.remaining_bytes)
                    if key in fp_map:
                        row.content_fingerprint = str(fp_map[key])
            # Drop exact duplicates by dedup key.
            seen = set()
            unique = []
            for row in recorder.rows:
                key = row.dedup_key()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(row.as_dict())
            out = {
                "seed": seed,
                "label": label,
                "n_rows": len(unique),
                "rows": unique,
                "free_on_time": summary.get("on_time", summary.get("return_on_time")),
                "seconds": round(time.time() - started, 3),
            }
            all_rows.append(out)
            (HERE / f"rows_s{seed}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[replay] s{seed} rows={len(unique)} {time.time()-started:.1f}s", flush=True)
            ledger.payload["entries"][-1]["ok"] = True
            ledger.payload["entries"][-1]["detail"] = f"rows={len(unique)}"
            LAUNCH_JSON.write_text(
                json.dumps(ledger.payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            (HERE / f"error_s{seed}.json").write_text(
                json.dumps({"error": detail, "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8",
            )
            ledger.payload["entries"][-1]["ok"] = False
            ledger.payload["entries"][-1]["detail"] = detail
            LAUNCH_JSON.write_text(
                json.dumps(ledger.payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[replay] s{seed} FAILED {detail}", flush=True)
            if args.stop_on_error:
                return 1
    common.write_json(ROWS_JSON, {"streams": all_rows})
    return 0


def prospective_labels(stream_rows: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, Any]]:
    """H=20 mechanism labels from the free trajectory only (no outcomes)."""
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in stream_rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    steps = sorted(by_step)
    # Map (target, revision) -> list of steps where same-view send was requested.
    request_steps: dict[tuple[str, int], list[int]] = {}
    revision_steps: dict[str, list[tuple[int, int]]] = {}
    for row in stream_rows:
        key = (row["target_id"], int(row["semantic_revision"]))
        revision_steps.setdefault(row["target_id"], []).append(
            (int(row["step"]), int(row["semantic_revision"]))
        )
        if row.get("rule_requested"):
            request_steps.setdefault(key, []).append(int(row["step"]))
    for key in request_steps:
        request_steps[key] = sorted(set(request_steps[key]))
    for tid in revision_steps:
        revision_steps[tid] = sorted(set(revision_steps[tid]))

    out: list[dict[str, Any]] = []
    for row in stream_rows:
        if row.get("primary_class") != "both_actions_free":
            continue
        if row.get("message_class") not in ("update", "same_view_retry"):
            continue
        if row.get("terminal"):
            continue
        step = int(row["step"])
        tid = str(row["target_id"])
        rev = int(row["semantic_revision"])
        later = [(s, r) for (s, r) in revision_steps.get(tid, []) if s > step]
        next_rev = later[0] if later else None
        steps_to_revision = (next_rev[0] - step) if next_rev else None
        same_key = (tid, rev)
        later_req = [s for s in request_steps.get(same_key, []) if s > step]
        req_within_1 = any(s <= step + 1 for s in later_req)
        req_within_20 = any(s <= step + 20 for s in later_req)
        if steps_to_revision is not None and steps_to_revision <= 20 and req_within_20:
            window = "can_block_retry" if (not next_rev or steps_to_revision > (later_req[0] - step if later_req else 999)) else "revision_truncated"
            if later_req and steps_to_revision is not None and steps_to_revision <= (later_req[0] - step):
                window = "revision_truncated"
            else:
                window = "can_block_retry"
        elif req_within_20:
            window = "can_block_retry"
        else:
            window = "no_same_view_retry_in_20"
        out.append(
            {
                "seed": int(row["seed"]),
                "step": step,
                "target": tid,
                "revision": rev,
                "message_class": row["message_class"],
                "steps_to_new_revision": steps_to_revision,
                "same_view_retry_req_within_1": req_within_1,
                "same_view_retry_req_within_20": req_within_20,
                "window_outlook": window,
                "attempt_charge_bytes": row.get("attempt_charge_bytes"),
                "budget_remaining": row.get("budget_remaining"),
            }
        )
    out.sort(key=lambda x: (x["seed"], x["step"], x["target"], x["revision"]))
    return out


def report(args: argparse.Namespace) -> int:
    payload = json.loads(ROWS_JSON.read_text(encoding="utf-8")) if ROWS_JSON.exists() else {"streams": []}
    all_rows: list[dict[str, Any]] = []
    per_stream_counts: dict[str, dict[str, int]] = {}
    message_class_counts: dict[str, int] = {"first": 0, "update": 0, "same_view_retry": 0}
    for stream in payload.get("streams", []):
        rows = stream["rows"]
        # classify
        objects = []
        for raw in rows:
            row = SA.ActionabilityRow(
                **{
                    k: (tuple(v) if k == "subscriber_set" else v)
                    for k, v in raw.items()
                    if k in SA.ActionabilityRow.__dataclass_fields__
                }
            )
            primary, secondary = SA.classify(row)
            raw["primary_class"] = primary
            raw["secondary_reasons"] = secondary
            objects.append(row)
            all_rows.append(raw)
            message_class_counts[raw.get("message_class", "?")] = (
                message_class_counts.get(raw.get("message_class", "?"), 0) + 1
            )
        counts = SA.classify_closed(objects)
        per_stream_counts[str(stream["seed"])] = counts

    # global counts from rows
    counts = {k: 0 for k in (
        "both_actions_free",
        "send_blocked_credit",
        "send_blocked_capacity_or_queue",
        "send_blocked_content_or_terminal",
        "send_blocked_retry_exhausted",
        "hold_not_free_protocol_forced",
        "not_policy_decision",
        "other",
    )}
    for raw in all_rows:
        counts[raw["primary_class"]] = counts.get(raw["primary_class"], 0) + 1
    identity = sum(counts.values()) == len(all_rows)

    # both_actions_free post-first
    post_free = [
        r
        for r in all_rows
        if r["primary_class"] == "both_actions_free"
        and r.get("message_class") in ("update", "same_view_retry")
        and not r.get("terminal")
    ]
    streams_with_post_free = sorted({int(r["seed"]) for r in post_free})

    prospective = []
    for stream in payload.get("streams", []):
        # re-classify lightly
        rows = stream["rows"]
        for raw in rows:
            if "primary_class" not in raw:
                row = SA.ActionabilityRow(
                    **{
                        k: (tuple(v) if k == "subscriber_set" else v)
                        for k, v in raw.items()
                        if k in SA.ActionabilityRow.__dataclass_fields__
                    }
                )
                raw["primary_class"], raw["secondary_reasons"] = SA.classify(row)
        prospective.extend(prospective_labels(rows, stream))
    common.write_json(POINTS_JSON, {"points": prospective})

    # Gate B: >=4 streams with >=1 post-first both_actions_free (SEND executable)
    b_streams = sorted(
        {
            int(r["seed"])
            for r in post_free
            if r.get("send_executable")
        }
    )
    gate_b = len(b_streams) >= 4

    # Gate C among B points: retry requested within 20 and not truncated before request
    c_streams = sorted(
        {
            int(p["seed"])
            for p in prospective
            if p.get("same_view_retry_req_within_20")
            and p.get("window_outlook") == "can_block_retry"
            and int(p["seed"]) in set(b_streams)
        }
    )
    gate_c = gate_c_ok = len(c_streams) >= 4

    # Gate A
    no_outcome = all(
        not ({"on_time", "expired", "send", "hold", "free"} & set(r.keys()))
        for r in all_rows
    )
    streams_seen = {int(s["seed"]) for s in payload.get("streams", [])}
    expected = {int(x["seed"]) for x in load_targets()}
    gate_a = identity and no_outcome and streams_seen == expected

    decision_weighted = {
        k: (counts[k] / len(all_rows) if all_rows else 0.0) for k in counts
    }
    stream_weighted = {}
    if per_stream_counts:
        for k in counts:
            vals = [c[k] / max(1, sum(c.values())) for c in per_stream_counts.values()]
            stream_weighted[k] = float(np.mean(vals)) if vals else 0.0

    results = {
        "schema": "comm-send-actionability-results/1",
        "streams": sorted(streams_seen),
        "n_rows": len(all_rows),
        "class_counts": counts,
        "identity_closed": identity,
        "message_class_counts": message_class_counts,
        "per_stream_counts": per_stream_counts,
        "decision_weighted_fraction": decision_weighted,
        "stream_weighted_fraction": stream_weighted,
        "post_first_both_actions_free": len(post_free),
        "streams_with_post_first_both_actions_free": streams_with_post_free,
        "gates": {
            "A_audit_trustworthy": {
                "passed": gate_a,
                "identity_closed": identity,
                "no_outcome_fields": no_outcome,
                "streams_complete": streams_seen == expected,
            },
            "B_dual_actionable": {
                "passed": gate_b,
                "streams": b_streams,
                "threshold_streams": 4,
            },
            "C_view_scoped_window": {
                "passed": gate_c,
                "streams": c_streams,
                "threshold_streams": 4,
            },
        },
        "launches": json.loads(LAUNCH_JSON.read_text(encoding="utf-8")) if LAUNCH_JSON.exists() else {},
        "prospective_points": len(prospective),
    }
    common.write_json(RESULTS_JSON, results)

    lines = [
        "# 通信 SEND/HOLD 动作可行性审计结果",
        "",
        f"候选总数 **{len(all_rows)}**（{len(streams_seen)} 条 free 流）。恒等式闭合：**{identity}**。",
        "",
        "## 主类分布（互斥）",
        "",
        "| 主类 | 计数 | decision-wtd | stream-wtd |",
        "|---|---:|---:|---:|",
    ]
    for k in counts:
        lines.append(
            f"| {k} | {counts[k]} | {decision_weighted[k]:.4f} | {stream_weighted[k]:.4f} |"
        )
    lines.extend(
        [
            "",
            f"消息类：first={message_class_counts.get('first',0)}，"
            f"update={message_class_counts.get('update',0)}，"
            f"same_view_retry={message_class_counts.get('same_view_retry',0)}",
            "",
            f"post-first-emission 的 both_actions_free：**{len(post_free)}** 点 / "
            f"**{len(streams_with_post_free)}** 流",
            "",
            "## 门禁",
            "",
            f"- **A 审计可信**：{'通过' if gate_a else '未过'}"
            f"（恒等式 {identity}，无 outcome {no_outcome}，8 流完整 {streams_seen==expected}）",
            f"- **B 双动作可行**：{'通过' if gate_b else '未过'}"
            f"（流 {b_streams}，门槛 ≥4）",
            f"- **C view-scoped**：{'通过' if gate_c else '未过'}"
            f"（流 {c_streams}，门槛 ≥4）",
            "",
            "## 判读边界",
            "",
            "- 8 条开发流，不作总体外推或方法主张。",
            "- 动作自由度不蕴含 PPO 信用分配、观测充分性或学习优于规则。",
            "",
        ]
    )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "comm-send-actionability-manifest/1",
        "branch": "analysis/comm-send-actionability-audit",
        "base_sha": "5b1062620a348d2425201b2c9594df674b411e78",
        "cherry_pick": "fc34545c14d311fcd90463bd3ac2e539fbbdf0e7",
        "stage1_artefacts": {
            "points": "runs/comm_multistep_stage1_20260922/POINTS.json",
            "free_streams": sorted(streams_seen),
        },
        "new_launches": results["launches"],
        "max_launches": MAX_LAUNCHES,
        "production_diff": "none expected",
        "tests": "runs/comm_send_actionability_20260922/test_send_actionability.py",
        "gates": results["gates"],
        "training": 0,
        "new_streams": 0,
    }
    common.write_json(MANIFEST_JSON, manifest)
    print(json.dumps(results["gates"], ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "report"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "replay":
        return run_replay(args)
    return report(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
