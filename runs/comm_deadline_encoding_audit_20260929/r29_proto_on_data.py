"""ROUND29 §6/§7 闭环：把编码原型直接作用于三来源 102 个首报点的原始期限值。

审计（r29_audit.py）给出每个首报点的原始剩余期限（来自 registry probe 的 sender-visible
deadline 与已实证的时钟规则 now = tape step - 1）。本脚本不过审计数字，而是把每个原始值
逐一送入原型 ``encoded_deadline``（含其非有限/非正显式报错逻辑），统计新列的不同值数，
与原始不同值数逐一比对：不同值数相等即"这些差异被保留"（映射为单射时的可证条件）。
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import r29_encoding_prototype as proto  # noqa: E402


def main() -> int:
    audit = json.loads((HERE / "ENCODING_AUDIT.json").read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    preserved = 0
    total = 0
    for name, stats in audit["sources"].items():
        records = stats["raw_unrecoverable_from_encoded_value"]["records"]
        raws = [round(float(r["raw_remaining"]), 9) for r in records]
        olds = [max(-1.0, min(1.0, r / proto.__dict__.get("_x", 50.0))) for r in raws]
        news = [proto.encoded_deadline(r) for r in raws]
        old_distinct = sorted(set(olds))
        new_distinct = sorted(set(news))
        raw_distinct = sorted(set(raws))
        ok = len(new_distinct) == len(raw_distinct)
        preserved += int(ok)
        total += 1
        result[name] = {
            "points": len(raws),
            "raw_distinct": len(raw_distinct),
            "old_column_distinct": len(old_distinct),
            "old_column_values": old_distinct,
            "new_column_distinct": len(new_distinct),
            "new_column_min": min(news),
            "new_column_max": max(news),
            "differences_preserved": ok,
        }
        print(f"{name}: points={len(raws)} raw_distinct={len(raw_distinct)} "
              f"old_distinct={len(old_distinct)} new_distinct={len(new_distinct)} "
              f"preserved={ok} range=[{min(news):.6f}, {max(news):.6f}]")
    print(f"sources with full preservation: {preserved}/{total}")
    result["sources_with_full_preservation"] = f"{preserved}/{total}"
    result["rule"] = ("the candidate column is an affine map remaining -> remaining/1000, which is "
                      "injective on the reals; distinct-value counts were compared per source as the "
                      "checkable witness, with every real value passed through the prototype code")
    (HERE / "ENCODING_PROTO_ON_DATA.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
