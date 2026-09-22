"""Post-process stage-one products and write the final report.

No new simulation launches. Recomputes total bytes from the transport ledger and
classifies window end reasons against the horizon so a post-window revision is not
recorded as a mid-window truncation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import ms_common as common


def totals_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    ledger = summary.get("ledger") or {}
    uplink = summary.get("uplink_bytes", ledger.get("airtime_bytes"))
    ack = summary.get("ack_bytes", ledger.get("ack_attempt_bytes"))
    if uplink is None:
        uplink = ledger.get("airtime_bytes", 0)
    if ack is None:
        ack = ledger.get("ack_attempt_bytes", 0)
    return {
        "uplink_bytes": int(uplink or 0),
        "ack_bytes": int(ack or 0),
        "total_bytes": int(uplink or 0) + int(ack or 0),
        "on_time": summary.get("on_time", summary.get("return_on_time")),
        "emissions": ledger.get("emissions"),
        "spent_bytes": summary.get("spent_bytes"),
        "budget_used_fraction": summary.get("budget_used_fraction"),
    }


def classify_window(window: dict[str, Any]) -> dict[str, Any]:
    at = int(window.get("at_step") or 0)
    horizon = int(window.get("horizon") or 0)
    next_rev = window.get("next_revision_send_step")
    truncated = bool(window.get("window_truncated"))
    trunc_at = window.get("window_truncated_at")
    effective = window.get("effective_window_steps")
    end_reason = None
    if truncated and trunc_at is not None and int(trunc_at) < at + horizon:
        end_reason = "window_truncated_by_revision"
        effective = int(trunc_at) - at
    elif truncated and trunc_at is not None and int(trunc_at) >= at + horizon:
        # Revision observed only after the window already expired by time.
        end_reason = "window_expired_then_revision"
        effective = horizon
        truncated = False
    elif horizon > 0:
        end_reason = "window_expired"
        effective = horizon
    out = dict(window)
    out["window_truncated"] = truncated
    out["window_end_reason"] = end_reason
    out["effective_window_steps"] = effective
    if window.get("force_refused"):
        out["force_outcome"] = "refused_insufficient_credit"
    elif window.get("force_applied"):
        out["force_outcome"] = "forced_once"
    elif window.get("mode") == "S":
        out["force_outcome"] = "not_forced"
    else:
        out["force_outcome"] = "hold_window"
    return out


def main() -> int:
    points = common.read_json(common.POINTS_JSON)
    ledger_path = common.EPISODES / "LAUNCH_LEDGER.json"
    ledger = common.read_json(ledger_path) if ledger_path.exists() else {"starts": 0}
    rows = []
    for row in points["rows"]:
        if row.get("status") != "validated":
            rows.append({"status": row.get("status"), "point": row})
            continue
        entry: dict[str, Any] = {
            "seed": int(row["seed"]),
            "step": int(row["step"]),
            "target": str(row["target"]),
            "candidate_at_tstar": row.get("candidate_at_tstar"),
            "free": None,
            "branches": {},
        }
        free_path = common.POINTS_DIR / f"s{row['seed']}_t{row['step']}" / "free.json"
        if free_path.exists():
            free = common.read_json(free_path)
            summary = free.get("free_summary") or {}
            entry["free"] = {
                "totals": totals_from_summary(summary),
                "candidate_at_tstar": free.get("candidate_at_tstar"),
            }
        for horizon in common.HORIZONS:
            for branch in ("S", "H"):
                key = f"H{horizon}_{branch}"
                path = (
                    common.POINTS_DIR
                    / f"s{row['seed']}_t{row['step']}"
                    / key
                    / "result.json"
                )
                if not path.exists():
                    entry["branches"][key] = {"status": "missing"}
                    continue
                product = common.read_json(path)
                window = classify_window(product.get("window") or {})
                totals = totals_from_summary(product.get("summary") or {})
                entry["branches"][key] = {
                    "status": "measured",
                    "window": window,
                    "totals": totals,
                    "label": product.get("label"),
                }
        rows.append(entry)

    paired = []
    for entry in rows:
        if "branches" not in entry or not entry.get("free"):
            continue
        free_t = entry["free"]["totals"]
        row = {
            "seed": entry["seed"],
            "step": entry["step"],
            "target": entry["target"],
            "free_on_time": free_t.get("on_time"),
            "free_total_bytes": free_t.get("total_bytes"),
            "horizons": {},
        }
        for horizon in common.HORIZONS:
            s_key, h_key = f"H{horizon}_S", f"H{horizon}_H"
            s = entry["branches"].get(s_key) or {}
            h = entry["branches"].get(h_key) or {}
            if s.get("status") != "measured" or h.get("status") != "measured":
                row["horizons"][str(horizon)] = {"status": "incomplete"}
                continue
            st, ht = s["totals"], h["totals"]
            sw, hw = s["window"], h["window"]
            row["horizons"][str(horizon)] = {
                "S_on_time": st.get("on_time"),
                "H_on_time": ht.get("on_time"),
                "delta_on_time_S_minus_H": (st.get("on_time") or 0) - (ht.get("on_time") or 0),
                "S_total_bytes": st.get("total_bytes"),
                "H_total_bytes": ht.get("total_bytes"),
                "delta_total_bytes_S_minus_H": (st.get("total_bytes") or 0)
                - (ht.get("total_bytes") or 0),
                "S_force_outcome": sw.get("force_outcome"),
                "S_forced_sends": sw.get("forced_sends"),
                "H_blocked": hw.get("window_blocked_requests"),
                "S_blocked": sw.get("window_blocked_requests"),
                "S_window_end": sw.get("window_end_reason"),
                "H_window_end": hw.get("window_end_reason"),
                "S_next_same_view": sw.get("next_same_view_send_step"),
                "H_next_same_view": hw.get("next_same_view_send_step"),
                "S_next_revision": sw.get("next_revision_send_step"),
                "H_next_revision": hw.get("next_revision_send_step"),
                "delta_free_on_time_S": (st.get("on_time") or 0) - (free_t.get("on_time") or 0),
                "delta_free_on_time_H": (ht.get("on_time") or 0) - (free_t.get("on_time") or 0),
            }
        paired.append(row)

    payload = {
        "schema": "comm-multistep-results/1",
        "baseline_rule": common.BASELINE_RULE,
        "frozen_config": common.read_json(common.FROZEN_CONFIG),
        "launch_ledger": ledger,
        "launch_cap": common.MAX_SIMULATION_STARTS,
        "points_complete": sum(
            1
            for entry in rows
            if entry.get("branches")
            and all(
                (entry["branches"].get(f"H{h}_{b}") or {}).get("status") == "measured"
                for h in common.HORIZONS
                for b in ("S", "H")
            )
        ),
        "points_shortfall": sum(
            1
            for entry in rows
            if entry.get("branches")
            and any(
                (entry["branches"].get(f"H{h}_{b}") or {}).get("status") != "measured"
                for h in common.HORIZONS
                for b in ("S", "H")
            )
        ),
        "failed_attempt1_retained": True,
        "paired": paired,
        "points": rows,
    }
    common.write_json(common.RESULTS_JSON, payload)

    lines = [
        "# 多步发送诊断第一阶段结果",
        "",
        "日期：2026-09-22。基线规则 `event_retry_until_acked_5`（`max_attempts=5`）。",
        f"总仿真启动 **{ledger.get('starts', 0)} / {common.MAX_SIMULATION_STARTS}**"
        "（含 4 次接口失败后隔离的 attempt1）。",
        "",
        "## 规模",
        "",
        f"- validated 点：8；四分支齐全：{payload['points_complete']}；缺额：{payload['points_shortfall']}",
        "- 缺额原因：启动上限（s700033 的 4 分支未跑），不补抽、不换 seed。",
        "- 首轮 4 分支因视图锁定过早失败，产物保留于各 `H*/failed_attempt1/`，修复后重跑并通过接口门禁。",
        "",
        "## 逐点成对（S − H）",
        "",
        "| 点 | H | ΔonT | S total | H total | Δtotal | S 强发 | H 窗口拦 | 窗口结束 | 同视图补发 | 新 revision |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for row in paired:
        for horizon in common.HORIZONS:
            cell = row["horizons"].get(str(horizon)) or {}
            if cell.get("status") == "incomplete":
                lines.append(
                    f"| s{row['seed']} t{row['step']} | {horizon} | — | — | — | — | — | — | incomplete | — | — |"
                )
                continue
            lines.append(
                "| s{seed} t{step} | {h} | {d} | {st} | {ht} | {dt} | {fo} | {blk} | {we} | {ns} | {nr} |".format(
                    seed=row["seed"],
                    step=row["step"],
                    h=horizon,
                    d=cell.get("delta_on_time_S_minus_H"),
                    st=cell.get("S_total_bytes"),
                    ht=cell.get("H_total_bytes"),
                    dt=cell.get("delta_total_bytes_S_minus_H"),
                    fo=cell.get("S_force_outcome"),
                    blk=cell.get("H_blocked"),
                    we=cell.get("H_window_end"),
                    ns=cell.get("H_next_same_view"),
                    nr=cell.get("H_next_revision"),
                )
            )
    lines.extend(
        [
            "",
            "## 机制观察（逐点描述，不作方法主张）",
            "",
            "1. **S 强发**：见上表 `S 强发` 列。`refused_insufficient_credit` 表示干预层请求了发送，"
            "但通信执行器因预算拒绝——按协议这**不是**成功的 S 分支，不能与 H 做 S−H 因果对照。",
            "2. **H 窗口拦截**：多数点在 H=20 上有非零 `window_blocked_requests`；H=1 通常只拦 t* 一步。",
            "3. **同视图补发**：`never`/空 表示窗口内及之后未再发射锁定视图；替代发射出现在新 revision 列。",
            "4. **窗口结束**：本批多数点在窗口内或紧随其后出现新 revision（`window_truncated_by_revision` / "
            "`window_expired_then_revision`）——长窗口假设在这些点上检验时间很短，**不能**据此说发送时机无价值。",
            "5. **任务结局**：在已跑点上 S/H/free 的 `on_time` 多数相同；在 S 未真正强发时，这只说明"
            "“未能强发 + 窗口拦截”相对 free 未改变完成数，**不**等于“发送时机无跨步影响”。",
            "",
            "## 判读边界",
            "",
            "- 不以 7 点 p 值或行级 bootstrap 作方法主张；H=1 与 H=20 是不同估计对象。",
            "- 前向 RNG 位移、预算拒绝与窗口截断必须与任何任务差异并列。",
            "- 本实验是 ACK 规则（`event_retry_until_acked_5`）下的局部机制诊断，与 no-ACK 学习曲线分开解释。",
            "- **不支持**“PPO 信用分配是根因”“发送端观测可辨优劣”“学习可赢规则”。",
            "- 结论允许不确定；是否扩样或改点由使用者决定。",
            "",
        ]
    )
    common.RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {common.RESULTS_JSON} and {common.RESULTS_MD}")
    print(
        json.dumps(
            {
                "starts": ledger.get("starts"),
                "complete": payload["points_complete"],
                "shortfall": payload["points_shortfall"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
