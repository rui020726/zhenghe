"""Render the six audit questions and A/B/C verdicts into RESULTS.md/.json."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    results = json.loads((HERE / "COMM_SEND_ACTIONABILITY_RESULTS.json").read_text(encoding="utf-8"))
    rows = json.loads((HERE / "COMM_SEND_ACTIONABILITY_ROWS.json").read_text(encoding="utf-8"))
    points = json.loads((HERE / "COMM_SEND_ACTIONABILITY_POINTS.json").read_text(encoding="utf-8"))["points"]
    allr = [r for s in rows["streams"] for r in s["rows"]]
    free = [r for r in allr if r.get("primary_class") == "both_actions_free"]
    post = [r for r in free if r.get("message_class") in ("update", "same_view_retry")]
    per_stream_n = {s["seed"]: len(s["rows"]) for s in rows["streams"]}
    free_mc = Counter(r["message_class"] for r in free)
    all_mc = Counter(r["message_class"] for r in allr)
    credit = [r for r in allr if r.get("primary_class") == "send_blocked_credit"]
    c_req = [p for p in points if p.get("same_view_retry_req_within_20")]
    c_block = [p for p in points if p.get("window_outlook") == "can_block_retry"]

    q = {
        "q1_candidates_per_stream": per_stream_n,
        "q1_total_candidates": len(allr),
        "q2_both_actions_free": len(free),
        "q2_both_actions_free_streams": sorted({r["seed"] for r in free}),
        "q3_send_blocked_credit": len(credit),
        "q3_send_blocked_capacity_or_queue": results["class_counts"].get(
            "send_blocked_capacity_or_queue", 0
        ),
        "q3_send_blocked_content_or_terminal": results["class_counts"].get(
            "send_blocked_content_or_terminal", 0
        ),
        "q3_send_blocked_retry_exhausted": results["class_counts"].get(
            "send_blocked_retry_exhausted", 0
        ),
        "q3_protocol_forced": results["class_counts"].get("hold_not_free_protocol_forced", 0),
        "q4_free_first": free_mc.get("first", 0),
        "q4_free_update": free_mc.get("update", 0),
        "q4_free_same_view_retry": free_mc.get("same_view_retry", 0),
        "q4_all_message_classes": dict(all_mc),
        "q5_post_first_dual_actionable_points": len(post),
        "q5_post_first_dual_actionable_streams": sorted({r["seed"] for r in post}),
        "q5_at_least_4_streams": len({r["seed"] for r in post}) >= 4,
        "q6_same_view_retry_req_within_20_points": len(c_req),
        "q6_streams_can_block_retry": sorted({p["seed"] for p in c_block}),
        "q6_at_least_4_streams": len({p["seed"] for p in c_block}) >= 4,
    }
    results["six_questions"] = q
    (HERE / "COMM_SEND_ACTIONABILITY_RESULTS.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    g = results["gates"]
    lines = [
        "# 通信 SEND/HOLD 动作可行性审计结果",
        "",
        f"候选总数 **{len(allr)}**（8 条 free 流，重放 8/8 启动上限，短探针 0 次）。",
        f"恒等式闭合：**{results['identity_closed']}**。",
        "",
        "## 主类分布（互斥）",
        "",
        "| 主类 | 计数 | decision-wtd | stream-wtd |",
        "|---|---:|---:|---:|",
    ]
    for k, v in results["class_counts"].items():
        lines.append(
            f"| {k} | {v} | {results['decision_weighted_fraction'].get(k, 0):.4f} | "
            f"{results['stream_weighted_fraction'].get(k, 0):.4f} |"
        )
    lines += [
        "",
        "## 六个审计问题",
        "",
        f"1. **每条流通信候选**：合计 {len(allr)}；逐流 {per_stream_n}",
        f"2. **SEND/HOLD 都可执行**：**{len(free)}** 点 / **{len(q['q2_both_actions_free_streams'])}** 流（`both_actions_free`）",
        f"3. **SEND 被阻止**：预算 **{len(credit)}**；容量/队列 **{q['q3_send_blocked_capacity_or_queue']}**；"
        f"终态/内容 **{q['q3_send_blocked_content_or_terminal']}**；重试耗尽 **{q['q3_send_blocked_retry_exhausted']}**；"
        f"协议强制 **{q['q3_protocol_forced']}**",
        f"4. **自由点按消息类**：首发 **{q['q4_free_first']}**，状态更新 **{q['q4_free_update']}**，"
        f"同视图重试 **{q['q4_free_same_view_retry']}**",
        f"5. **post-first-emission 双动作可行**：**{len(post)}** 点 / **{len(q['q5_post_first_dual_actionable_streams'])}** 流"
        f"（≥4 流：{'是' if q['q5_at_least_4_streams'] else '否'}）",
        f"6. **20 步内同视图重试且不被 revision 提前截断**：**{len(c_block)}** 流"
        f"（≥4 流：{'是' if q['q6_at_least_4_streams'] else '否'}）；"
        f"其中 20 步内有同视图重试请求的点 **{q['q6_same_view_retry_req_within_20_points']}**。"
        f"几乎全部 post-first 点的 `steps_to_new_revision=1`（视图每步被新 revision 取代）。",
        "",
        "## 门禁",
        "",
        f"- **A 审计可信**：{'通过' if g['A_audit_trustworthy']['passed'] else '未过'}",
        f"- **B 双动作可行**：{'通过' if g['B_dual_actionable']['passed'] else '未过'}"
        f"（流 {g['B_dual_actionable']['streams']}）",
        f"- **C view-scoped 干预有效**：{'通过' if g['C_view_scoped_window']['passed'] else '未过'}"
        f"（流 {g['C_view_scoped_window']['streams']}）",
        "",
        "### 判定（按指令 §6）",
        "",
        "- A 过、**B 过**、**C 不过** ⇒ 不暗改 Stage1 协议冒充同一实验。",
        "- view-scoped 窗口在该消息流上几乎立刻被新 revision 截断，且 free 轨迹上少见同视图 20 步内重试请求。",
        "- 下一轮若继续，应讨论 **target-scoped**“延迟所有更新并在窗口末发送最新视图”这一**新估计对象**（本轮不设计、不执行）。",
        "",
        "## 判读边界",
        "",
        "- 8 条开发流，不作总体外推或方法主张。",
        "- 主阻塞是**预算**（decision-wtd ~95%+），不是容量或协议强制。",
        "- 动作自由度不蕴含 PPO 信用分配、观测充分性或学习优于规则。",
        "",
    ]
    (HERE / "COMM_SEND_ACTIONABILITY_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(q, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
