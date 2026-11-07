"""Per-checkpoint evidence tables for REPORT_rev02.md.

Reads an eval tag's EVAL.json plus the on-disk transcripts and emits, per cell:
the registered verdict, the send-set comparison against normal_immediate, and the
same-target-set recheck (conditioning on the targets the learned arm kept).

    python analyse_checkpoints.py --points 1200:eval_1200_e1200 ... --out tables.md
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
BASELINES = BASE / "execution" / "baselines" / "grid_v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def world_dir(root, arm, label, scene):
    compact = label.replace("_", "")
    matches = sorted((root / "world_logs").glob("eval__%s_%s__*__%d" % (arm, compact, scene)))
    if len(matches) != 1:
        raise RuntimeError("expected 1 world dir for %s %s %d, got %d" % (arm, label, scene, len(matches)))
    inner = sorted(matches[0].iterdir())
    if len(inner) != 1:
        raise RuntimeError("expected 1 inner dir under %s" % matches[0])
    return inner[0]


def send_sets(directory):
    rows = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")
    per_step = {}
    for row in rows:
        if row.get("kind") == "send":
            per_step.setdefault(int(row["step"]), set()).add(row["target"])
    return per_step, sum(len(v) for v in per_step.values())


def completed_targets(directory):
    rows = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    return {r["target_id"] for r in rows
            if r.get("event_type") == "final" and r.get("status") in ("completed", "complete", "on_time", "done")}


def verdict_word(entry):
    verdict = entry["verdict_vs_immediate"]
    if verdict["improved"]:
        return "**是**"
    reason = verdict.get("reason", "")
    if "below 0.95" in reason:
        return "否 成功率不足"
    if "bytes" in reason:
        return "否 字节不少"
    return "否 " + reason


def fmt_ci(ci):
    if not ci:
        return "—"
    return "[%.2f, %.2f]" % (ci[0], ci[1])


def analyse_point(name, tag, train_root):
    eval_dir = train_root / tag
    path = eval_dir / "EVAL.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    per_cell = data["per_cell"]
    labels = sorted(per_cell, key=lambda s: tuple(int(x) for x in re.findall(r"\d+", s)))
    scenes = data["scenes"]

    lines = ["## milestone_%s（EVAL.json sha256 `%s`）" % (name, sha256(path)), ""]
    lines += ["| cell | on_time/G | 原始 bytes | 发送数 | 不同目标 | 重复 | bytes/on_time | q vs imm | 判定 |",
              "|---|---|---|---|---|---|---|---|---|"]
    identity = {"same": 0, "subset": 0, "other": 0}
    learned_sends = immediate_sends = 0
    solid = []
    passed = []
    for label in labels:
        entry = per_cell[label]
        learned = entry["learned"]
        imm = entry["baselines"]["normal_immediate"]
        lines.append("| %s | %d/%d | %d | %d | %d | %d | %d | %d/%d | %s |" % (
            label, learned["on_time"], learned["generated_goals"], learned["spent_bytes"],
            learned["emissions"], learned["distinct_targets"], learned["duplicate_sends"],
            round(learned["bytes_per_on_time"]) if learned["on_time"] else 0,
            learned["on_time"], imm["on_time"], verdict_word(entry)))
        if entry["verdict_vs_immediate"]["improved"]:
            passed.append(label)

        overlap = 0
        for scene in scenes:
            lset, ln = send_sets(world_dir(eval_dir, "T2", label, scene))
            iset, inn = send_sets(world_dir(BASELINES, "normal_immediate", label, scene))
            learned_sends += ln
            immediate_sends += inn
            kept = {t for s in lset.values() for t in s}
            overlap += len(kept & completed_targets(world_dir(BASELINES, "normal_immediate", label, scene)))
            if lset == iset:
                identity["same"] += 1
            elif all(step in iset and targets <= iset[step] for step, targets in lset.items()) and ln < inn:
                identity["subset"] += 1
            else:
                identity["other"] += 1
        if label in passed:
            ratio = overlap / imm["on_time"] if imm["on_time"] else 0.0
            saving = 100.0 * (1 - learned["spent_bytes"] / imm["spent_bytes"]) if imm["spent_bytes"] else 0.0
            solid.append((label, overlap, imm["on_time"], ratio, saving))

    lines += ["", "改善 cell：**%d / 15**%s。" % (len(passed), ("（%s）" % "、".join(passed)) if passed else ""), ""]
    if solid:
        lines += ["同目标集合复核（分母 = 该 cell 三场景 immediate 的 on_time 合计）：", "",
                  "| cell | 保留目标在 immediate 完成数 | 比值 | 字节节省 | 复核 |", "|---|---|---|---|---|"]
        for label, overlap, denom, ratio, saving in solid:
            lines.append("| %s | %d/%d | %.3f | %.2f%% | %s |" % (
                label, overlap, denom, ratio, saving, "成立" if ratio >= 0.95 else "投递抽签"))
        lines.append("")
    lines += ["| cell | − immediate | 95% CI | − dedup | 95% CI |", "|---|---|---|---|---|"]
    for label in labels:
        entry = per_cell[label]
        pi, pd = entry["paired_vs_immediate"], entry["paired_vs_dedup"]
        lines.append("| %s | %+.2f | %s | %+.2f | %s |" % (
            label, pi["mean_delta_on_time"], fmt_ci(pi["ci95"]),
            pd["mean_delta_on_time"], fmt_ci(pd["ci95"])))
    lines.append("")

    total = len(labels) * len(scenes)
    stats = {"name": name, "tag": tag, "worlds": total, "worlds_ok": data["worlds_ok"],
             "failures": len(data.get("failures") or []),
             "identity": identity, "learned_sends": learned_sends, "immediate_sends": immediate_sends,
             "passed": passed,
             "solid": [[s[0], s[3], s[4]] for s in solid if s[3] >= 0.95],
             "pooled_success_ratio": sum(per_cell[l]["learned"]["on_time"] for l in labels)
             / max(1, sum(per_cell[l]["baselines"]["normal_immediate"]["on_time"] for l in labels)),
             "pooled_bytes_ratio": sum(per_cell[l]["learned"]["spent_bytes"] for l in labels)
             / max(1, sum(per_cell[l]["baselines"]["normal_immediate"]["spent_bytes"] for l in labels)),
             "sha": sha256(path)[:12]}
    return "\n".join(lines), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default=str(BASE / "execution" / "train_v3"))
    ap.add_argument("--points", nargs="+", required=True, help="name:tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default="")
    args = ap.parse_args()

    root = Path(args.train_root)
    chunks, stats = [], []
    for point in args.points:
        name, tag = point.split(":", 1)
        text, stat = analyse_point(name, tag, root)
        chunks.append(text)
        stats.append(stat)
        print("%s: %d/%d worlds ok, identity %s, sends %d/%d, passed %d, solid %d" % (
            name, stat["worlds_ok"], stat["worlds"], stat["identity"],
            stat["learned_sends"], stat["immediate_sends"], len(stat["passed"]), len(stat["solid"])), flush=True)
    Path(args.out).write_text("\n".join(chunks), encoding="utf-8")
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()