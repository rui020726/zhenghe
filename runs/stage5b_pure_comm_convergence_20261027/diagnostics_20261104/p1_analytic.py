"""P1 acceptance driver: cost-side analytic invariance, zero world replays.

Loads recovery_12/revision_01/stage5b_comm/scale_invariant.py by path so the
revision package's torch-bearing __init__ is never imported, runs the three
acceptance groups the protocol names, prints every line with its denominator,
and writes execution/scale_probe/P1_analytic.json.

Tolerance policy, stated up front because it decides two verdicts:
  * integer and identity claims are checked exactly (affordable_sends, the
    equality of all three N under proportional, budget/B0 at N=6);
  * the percentage and advantage literals in the protocol were written from a
    rounded 2,500.6 B/send, while the frozen tape gives 2,498.674 B/send, so
    those literals are compared at the protocol's own stated tolerance
    (+-0.01 pp for shares) and the reconstruction at the rounded payload is
    reported beside the measured value rather than the constant being bent to
    reproduce the literal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

DIAG = Path(__file__).resolve().parent
STAGE = DIAG.parent
REVISION = STAGE / "recovery_12" / "revision_01"
OUT = REVISION / "execution" / "scale_probe"
MODULE = REVISION / "stage5b_comm" / "scale_invariant.py"

NS = (3, 6, 12)
SHARE_TOLERANCE_PP = 0.01
ADV_TOLERANCE = 0.0005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("scale_invariant", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(name, passed, expected, actual, denominator, note=None):
    record = {"check": name, "pass": bool(passed), "expected": expected,
              "actual": actual, "denominator": denominator}
    if note:
        record["note"] = note
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    si = load_module()
    payload = si.PAYLOAD_BYTES
    grid = si.table(NS)
    checks = []

    print("=" * 78)
    print("P1  cost-side analytic invariance (zero replays)")
    print("=" * 78)
    print("module      :", MODULE.relative_to(STAGE))
    print("module sha  :", sha256(MODULE))
    print("PAYLOAD_BYTES = %d B / %d sends / %d subscribers = %.10f B"
          % (si.PAID_BYTES_TOTAL, si.EMISSIONS_TOTAL, si.SUBSCRIBERS_REF, payload))
    print("  provenance: %s" % si.PAYLOAD_PROVENANCE["source"])
    print("  cost_per_send at N=6 = %.6f B  (protocol literal 2500.6 B, delta %+.4f%%)"
          % (si.cost_per_send(6),
             100.0 * (si.cost_per_send(6) - si.SPEC_LITERAL_COST_PER_SEND)
             / si.SPEC_LITERAL_COST_PER_SEND))
    print()

    # ---- group 1: fixed, the main calibre ------------------------------------
    print("--- group 1: mode='fixed' (main calibre), denominators named per line ---")
    expect_afford = {3: 46, 6: 23, 12: 11}
    expect_share = {3: 2.134, 6: 4.268, 12: 8.536}
    expect_adv = {3: 0.125, 6: 0.250, 12: 0.500}
    for n in NS:
        cell = grid["fixed"][n]
        print("  N=%-2d affordable_sends = %d   [budget %.0f B // cost %.4f B]"
              % (n, cell["affordable_sends"], cell["budget_bytes"],
                 cell["cost_per_send_bytes"]))
        print("       share_per_send     = %.6f%%  [cost %.4f B / budget %.0f B]"
              % (cell["share_per_send_pct"], cell["cost_per_send_bytes"],
                 cell["budget_bytes"]))
        print("       advantage_cost     = %.7f    [cost %.4f B / B0 %.0f B]"
              % (cell["advantage_cost_per_send"], cell["cost_per_send_bytes"],
                 cell["b0_bytes"]))
        checks.append(check(
            "fixed.affordable_sends[N=%d]" % n,
            cell["affordable_sends"] == expect_afford[n],
            expect_afford[n], cell["affordable_sends"],
            "budget_bytes // cost_per_send = %.0f // %.6f"
            % (cell["budget_bytes"], cell["cost_per_send_bytes"])))
        checks.append(check(
            "fixed.share_per_send_pct[N=%d]" % n,
            abs(cell["share_per_send_pct"] - expect_share[n]) <= SHARE_TOLERANCE_PP,
            expect_share[n], round(cell["share_per_send_pct"], 6),
            "100 * cost_per_send / budget_bytes, tolerance +-%.2f pp"
            % SHARE_TOLERANCE_PP,
            "protocol literal derived from a rounded 2500.6 B/send; measured is %.4f B/send"
            % si.cost_per_send(6)))
        checks.append(check(
            "fixed.advantage_cost_per_send[N=%d]" % n,
            abs(cell["advantage_cost_per_send"] - expect_adv[n]) <= ADV_TOLERANCE,
            expect_adv[n], round(cell["advantage_cost_per_send"], 7),
            "cost_per_send / b0_bytes, tolerance +-%.4f" % ADV_TOLERANCE))
    print()

    # ---- group 2: proportional, analytic control -----------------------------
    print("--- group 2: mode='proportional' (analytic control; never replayed) ---")
    shares = [grid["proportional"][n]["share_per_send_pct"] for n in NS]
    affords = [grid["proportional"][n]["affordable_sends"] for n in NS]
    advs = [grid["proportional"][n]["advantage_cost_per_send"] for n in NS]
    for n in NS:
        cell = grid["proportional"][n]
        print("  N=%-2d budget=%.0f B  B0=%.0f B  cost=%.4f B  afford=%d  "
              "share=%.6f%%  adv=%.7f"
              % (n, cell["budget_bytes"], cell["b0_bytes"],
                 cell["cost_per_send_bytes"], cell["affordable_sends"],
                 cell["share_per_send_pct"], cell["advantage_cost_per_send"]))
    spread = lambda xs: max(xs) - min(xs)
    checks.append(check("proportional.share_equal_across_N", spread(shares) == 0.0,
                        0.0, spread(shares), "max-min over N=3/6/12, exact float equality"))
    checks.append(check("proportional.share_equals_4.268pct",
                        all(abs(s - 4.268) <= SHARE_TOLERANCE_PP for s in shares),
                        "4.268 +-0.01 pp", [round(s, 6) for s in shares],
                        "each of N=3/6/12 against the protocol literal"))
    checks.append(check("proportional.affordable_equal_across_N",
                        len(set(affords)) == 1 and affords[0] == 23,
                        23, affords, "budget_bytes // cost_per_send per N"))
    checks.append(check("proportional.advantage_equal_across_N", spread(advs) == 0.0,
                        0.0, spread(advs), "max-min over N=3/6/12"))
    checks.append(check("proportional.advantage_equals_0.25",
                        all(abs(a - 0.25) <= ADV_TOLERANCE for a in advs),
                        "0.25 +-%.4f" % ADV_TOLERANCE, [round(a, 7) for a in advs],
                        "cost_per_send / b0_bytes per N"))
    print("  share spread across N   = %.3e pp   (must be exactly 0)" % spread(shares))
    print("  advantage spread across N = %.3e      (must be exactly 0)" % spread(advs))
    print()

    # ---- group 3: N=6 must equal the frozen literals in both modes -----------
    print("--- group 3: N=6 equals the frozen literals in both modes ---")
    for mode in ("fixed", "proportional"):
        cell = grid[mode][6]
        print("  mode=%-13s budget=%.0f  B0=%.0f  affordable=%d  share=%.6f%%"
              % (mode, cell["budget_bytes"], cell["b0_bytes"],
                 cell["affordable_sends"], cell["share_per_send_pct"]))
        checks.append(check("%s.budget_bytes[N=6]" % mode,
                            cell["budget_bytes"] == 58600.0, 58600,
                            cell["budget_bytes"], "frozen BUDGET_BYTES"))
        checks.append(check("%s.b0_bytes[N=6]" % mode, cell["b0_bytes"] == 10000.0,
                            10000, cell["b0_bytes"], "frozen B0_BYTES"))
        checks.append(check("%s.affordable_sends[N=6]" % mode,
                            cell["affordable_sends"] == 23, 23,
                            cell["affordable_sends"],
                            "58600 // %.6f" % cell["cost_per_send_bytes"]))
        checks.append(check("%s.share_per_send_pct[N=6]" % mode,
                            abs(cell["share_per_send_pct"] - 4.268) <= SHARE_TOLERANCE_PP,
                            "4.268 +-0.01 pp", round(cell["share_per_send_pct"], 6),
                            "100 * cost_per_send / 58600"))
    print()

    # ---- reconstruction at the protocol's rounded payload --------------------
    print("--- reconstruction: what the protocol literals imply about the payload ---")
    reconstruction = {}
    for n in NS:
        cost = si.SPEC_LITERAL_COST_PER_SEND * n / 6
        reconstruction[n] = {
            "payload_bytes": si.SPEC_LITERAL_PAYLOAD_BYTES,
            "cost_per_send_bytes": cost,
            "affordable_sends_fixed": int(58600 // cost),
            "share_per_send_pct_fixed": 100.0 * cost / 58600,
            "advantage_cost_per_send_fixed": cost / 10000,
        }
        print("  N=%-2d payload=%.4f cost=%.4f afford=%d share=%.4f%% adv=%.5f"
              % (n, si.SPEC_LITERAL_PAYLOAD_BYTES, cost,
                 reconstruction[n]["affordable_sends_fixed"],
                 reconstruction[n]["share_per_send_pct_fixed"],
                 reconstruction[n]["advantage_cost_per_send_fixed"]))
    print("  => the protocol's 2.134/4.268/8.536%% and 0.125/0.250/0.500 are reproduced by")
    print("     payload=416.7667 B; the frozen tape gives %.4f B, so the measured shares"
          % payload)
    print("     read 2.132/4.264/8.528%% instead (delta %+.4f%% on cost_per_send)."
          % (100.0 * (si.cost_per_send(6) - si.SPEC_LITERAL_COST_PER_SEND)
             / si.SPEC_LITERAL_COST_PER_SEND))
    print("     The integer claims (46/23/11, 23, 58600, 10000) are unaffected.")
    print()

    passed = sum(1 for c in checks if c["pass"])
    verdict = "pass" if passed == len(checks) else "fail"
    print("=" * 78)
    print("P1 verdict: %s   (%d/%d checks pass)" % (verdict, passed, len(checks)))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c["check"], "expected", c["expected"], "actual", c["actual"])
    print("=" * 78)

    report = {
        "schema": "scale-probe-p1-analytic/1",
        "stage": "P1",
        "replay_quota_used": 0,
        "identity": "local experiment level, no identity built",
        "module": {"path": str(MODULE.relative_to(STAGE)), "sha256": sha256(MODULE)},
        "payload_bytes": payload,
        "payload_provenance": si.PAYLOAD_PROVENANCE,
        "spec_literal_reconstruction": {str(k): v for k, v in reconstruction.items()},
        "tolerances": {"share_pp": SHARE_TOLERANCE_PP, "advantage": ADV_TOLERANCE},
        "grid": {mode: {str(n): grid[mode][n] for n in NS} for mode in ("fixed", "proportional")},
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "verdict": verdict,
    }
    path = OUT / "P1_analytic.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True),
                    encoding="utf-8")
    print("wrote", path)
    print("sha256", sha256(path))


if __name__ == "__main__":
    main()
