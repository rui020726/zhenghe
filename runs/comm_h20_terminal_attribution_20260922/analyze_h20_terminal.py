"""Offline H=20 terminal attribution from existing target-scoped + free streams.

0 new episodes. Parses only archived JSONL / boundary_record / actionability results.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
TS_EP = RUNS / "comm_target_scoped_feasibility_20260922" / "episodes"
AB_EP = RUNS / "comm_send_actionability_20260922" / "episodes"
AB_RES = RUNS / "comm_send_actionability_20260922" / "COMM_SEND_ACTIONABILITY_RESULTS.json"

SEED_DIRS = [
    ("s700010_t3_piloc-e1-target-0", "700010", "piloc-e1-target-0"),
    ("s700018_t3_piloc-e1-target-2", "700018", "piloc-e1-target-2"),
    ("s700020_t3_piloc-e1-target-0", "700020", "piloc-e1-target-0"),
    ("s700022_t3_piloc-e1-target-2", "700022", "piloc-e1-target-2"),
]
MODES = ["H20_EARLY_FLUSH", "H20_DEFER_FLUSH"]
OUT_FIELDS = (
    "status",
    "progress",
    "stage_index",
    "task_revision",
    "assigned_agents",
    "remaining_requirements",
    "failure_reason",
)


def load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def find_one(d: Path, pattern: str) -> Path | None:
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def terminal_snapshot(rp_rows: list[dict], tr_rows: list[dict]) -> dict:
    by_t: dict[str, dict] = {}
    events: list[dict] = []
    for r in rp_rows:
        if r.get("event_type") == "run_config":
            continue
        events.append(r)
        tid = r.get("target_id")
        if tid and tid != "__run__":
            by_t[tid] = r
    last_dec = None
    for r in tr_rows:
        if r.get("event_type") == "decision":
            last_dec = r
    terminal_events = [
        r
        for r in events
        if r.get("status") in ("completed", "failed", "aborted") or r.get("failure_reason")
    ]
    max_ts = max((r.get("timestamp") or 0) for r in events) if events else None
    return {
        "n_rp": len(rp_rows),
        "n_tr": len(tr_rows),
        "max_timestamp": max_ts,
        "last_decision_step": None if not last_dec else last_dec.get("step"),
        "terminal_events": [
            {
                "target_id": r.get("target_id"),
                "event_type": r.get("event_type"),
                "timestamp": r.get("timestamp"),
                "status": r.get("status"),
                "progress": r.get("progress"),
                "stage_index": r.get("stage_index"),
                "task_revision": r.get("task_revision"),
                "failure_reason": r.get("failure_reason"),
            }
            for r in terminal_events
        ],
        "targets": {
            tid: {k: r.get(k) for k in OUT_FIELDS} | {"timestamp": r.get("timestamp")}
            for tid, r in by_t.items()
        },
        "last_decision": None
        if not last_dec
        else {
            "agent_id": last_dec.get("agent_id"),
            "step": last_dec.get("step"),
            "actual_assignment": last_dec.get("actual_assignment"),
            "chosen_task": last_dec.get("chosen_task"),
            "belief_task_count": last_dec.get("belief_task_count"),
            "advice_revision": last_dec.get("advice_revision"),
            "commitment_task": last_dec.get("commitment_task"),
            "commitment_held": last_dec.get("commitment_held"),
        },
    }


def load_branch(sdir: str, mode: str) -> dict:
    d = TS_EP / sdir / mode
    br_path = d / "boundary_record.json"
    br = json.loads(br_path.read_text(encoding="utf-8")) if br_path.exists() else {}
    w = br.get("window") or br
    tr_p = find_one(d, "tr_ts_*.jsonl")
    rp_p = find_one(d, "rp_ts_*.jsonl")
    tr = load_jsonl(tr_p) if tr_p else []
    rp = load_jsonl(rp_p) if rp_p else []
    snap = terminal_snapshot(rp, tr)
    snap["boundary"] = {
        "boundary_status": w.get("boundary_status"),
        "terminal_at": w.get("terminal_at"),
        "terminal_before_boundary": w.get("terminal_before_boundary"),
        "steps_executed": w.get("steps_executed"),
        "start_send_ok": w.get("start_send_ok"),
        "start_revision": w.get("start_revision"),
        "start_send_bytes": w.get("start_send_bytes"),
        "start_fingerprint": w.get("start_fingerprint"),
        "boundary_fingerprint": w.get("boundary_fingerprint"),
        "revisions_seen": w.get("revisions_seen"),
        "policy_requests_blocked": w.get("policy_requests_blocked"),
        "other_target_emissions": w.get("other_target_emissions"),
        "forward_draws_after": w.get("forward_draws_after"),
        "forward_draws_at_fork": w.get("forward_draws_at_fork"),
    }
    # EARLY delivery/application: did target revision advance to start_revision?
    snap["early_delivery"] = {
        "start_send_requested": w.get("start_send_requested"),
        "start_send_ok": w.get("start_send_ok"),
        "start_revision": w.get("start_revision"),
        "start_send_bytes": w.get("start_send_bytes"),
        "revisions_seen": w.get("revisions_seen"),
        "applied_if_start_rev_in_seen": (
            w.get("start_revision") in (w.get("revisions_seen") or [])
            if w.get("start_revision") is not None
            else None
        ),
    }
    return snap


def field_diffs(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    out = {}
    for tid in sorted(set(a) | set(b)):
        ta, tb = a.get(tid), b.get(tid)
        if ta is None or tb is None:
            out[tid] = {"only_in": label_a if ta else label_b, label_a: ta, label_b: tb}
            continue
        dlt = {}
        for k in OUT_FIELDS:
            if ta.get(k) != tb.get(k):
                dlt[k] = {label_a: ta.get(k), label_b: tb.get(k)}
        if dlt:
            out[tid] = dlt
    return out


def classify(e: dict, d: dict) -> str:
    e_ref = e["boundary"].get("boundary_status") == "refused_credit"
    d_ref = d["boundary"].get("boundary_status") == "refused_credit"
    e_term = bool(e["boundary"].get("terminal_before_boundary"))
    d_term = bool(d["boundary"].get("terminal_before_boundary"))
    if e_ref and d_ref:
        return "boundary_refused"
    if e_term and d_term:
        if e["boundary"].get("terminal_at") != d["boundary"].get("terminal_at"):
            return "terminal_time_diff"
        if e["targets"] == d["targets"]:
            return "both_terminal_identical"
        return "both_terminal_same_time_outcome_diff"
    if e_term or d_term:
        return "one_terminal_only"
    return "neither_terminal_boundary_reached"


def main() -> None:
    free: dict[str, dict] = {}
    for _, sid, _t in SEED_DIRS:
        tr_p = find_one(AB_EP, f"tr_abl_free_s{sid}*.jsonl")
        rp_p = find_one(AB_EP, f"rp_abl_free_s{sid}*.jsonl")
        free[sid] = terminal_snapshot(
            load_jsonl(rp_p) if rp_p else [], load_jsonl(tr_p) if tr_p else []
        )

    pairs = {}
    summary_rows = []
    for sdir, sid, target in SEED_DIRS:
        snaps = {m: load_branch(sdir, m) for m in MODES}
        e, d = snaps["H20_EARLY_FLUSH"], snaps["H20_DEFER_FLUSH"]
        pair_class = classify(e, d)
        fr = free.get(sid)
        pair = {
            "seed": int(sid),
            "target": target,
            "class": pair_class,
            "EARLY_FLUSH": e,
            "DEFER_FLUSH": d,
            "early_defer_field_diffs": field_diffs(e["targets"], d["targets"], "EARLY", "DEFER"),
            "early_send": e["early_delivery"],
        }
        if fr:
            pair["free_compare"] = {
                "free_max_timestamp": fr.get("max_timestamp"),
                "free_last_decision_step": fr.get("last_decision_step"),
                "free_targets": fr.get("targets"),
                "free_terminal_events": fr.get("terminal_events"),
                "free_last_decision": fr.get("last_decision"),
            }
            pair["early_vs_free_diffs"] = field_diffs(e["targets"], fr["targets"], "EARLY", "free")
            pair["defer_vs_free_diffs"] = field_diffs(d["targets"], fr["targets"], "DEFER", "free")
        pairs[f"s{sid}_t3_H20"] = pair
        summary_rows.append(
            {
                "seed": int(sid),
                "target": target,
                "class": pair_class,
                "early_terminal_at": e["boundary"].get("terminal_at"),
                "defer_terminal_at": d["boundary"].get("terminal_at"),
                "early_boundary_status": e["boundary"].get("boundary_status"),
                "defer_boundary_status": d["boundary"].get("boundary_status"),
                "early_start_send_ok": e["boundary"].get("start_send_ok"),
                "early_start_revision": e["boundary"].get("start_revision"),
                "early_revisions_seen": e["boundary"].get("revisions_seen"),
                "defer_revisions_seen": d["boundary"].get("revisions_seen"),
                "early_defer_diffs": pair["early_defer_field_diffs"],
                "has_free": fr is not None,
                "early_vs_free_diffs": pair.get("early_vs_free_diffs"),
                "defer_vs_free_diffs": pair.get("defer_vs_free_diffs"),
            }
        )

    ab = {}
    if AB_RES.exists():
        raw = json.loads(AB_RES.read_text(encoding="utf-8"))
        ab = {
            "source": str(AB_RES.relative_to(ROOT)),
            "both_actions_free": raw.get("summary", raw).get("both_actions_free")
            if isinstance(raw.get("summary", raw), dict)
            else None,
            "q2_both_actions_free": raw.get("six_questions", raw).get("q2_both_actions_free")
            if isinstance(raw.get("six_questions", raw), dict)
            else None,
            "raw_top_keys": list(raw.keys())[:20],
        }
        # keep a few known totals if present
        for k in (
            "both_actions_free",
            "post_first_both_actions_free",
            "streams_with_post_first_both_actions_free",
            "six_questions",
            "summary",
            "B_dual_actionable",
            "gates",
        ):
            if k in raw:
                ab[k] = raw[k]

    out = {
        "round": "comm_h20_terminal_attribution_20260922",
        "constraints": {
            "new_episodes": 0,
            "training": 0,
            "new_streams": 0,
            "param_changes": 0,
            "sources": [
                "runs/comm_target_scoped_feasibility_20260922/episodes/**",
                "runs/comm_send_actionability_20260922/episodes/**",
                "runs/comm_send_actionability_20260922/COMM_SEND_ACTIONABILITY_RESULTS.json",
                "task_allocation/* training source (read-only)",
                "integrated_sar/* training/relay source (read-only)",
            ],
        },
        "classification_legend": {
            "both_terminal_identical": "双方均在边界前终态且终态快照完全一致",
            "both_terminal_same_time_outcome_diff": "双方同刻终态但 belief/work/progress 有分歧",
            "terminal_time_diff": "双方均终态但终态时间不同",
            "one_terminal_only": "仅一侧终态",
            "boundary_refused": "边界 flush 双方 refused_credit（非有效 flush 也非终态吸收）",
            "neither_terminal_boundary_reached": "双方到达边界且未终态",
        },
        "summary_rows": summary_rows,
        "pairs": pairs,
        "actionability_free_totals": ab,
        "judgment_inputs": {
            "early_success_pairs_with_any_outcome_divergence": [
                r["seed"]
                for r in summary_rows
                if r["early_start_send_ok"]
                and r["class"]
                in (
                    "both_terminal_same_time_outcome_diff",
                    "terminal_time_diff",
                    "one_terminal_only",
                )
            ],
            "both_terminal_identical_count": sum(
                1 for r in summary_rows if r["class"] == "both_terminal_identical"
            ),
            "boundary_refused_count": sum(
                1 for r in summary_rows if r["class"] == "boundary_refused"
            ),
            "note": "terminal_before_boundary 不自动解释为有效 flush；成功 EARLY 投递须 start_send_ok 且 revision 应用可核。",
        },
    }

    out_path = Path(__file__).resolve().parent / "COMM_H20_TERMINAL_ATTRIBUTION.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    # compact stdout
    print("class counts:")
    counts: dict[str, int] = {}
    for r in summary_rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(counts)
    for r in summary_rows:
        print(
            f"s{r['seed']} class={r['class']} early_term={r['early_terminal_at']} "
            f"defer_term={r['defer_terminal_at']} early_send={r['early_start_send_ok']} "
            f"diffs={list((r['early_defer_diffs'] or {}).keys())} "
            f"free_diffs_e={bool(r['early_vs_free_diffs'])} free_diffs_d={bool(r['defer_vs_free_diffs'])}"
        )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
