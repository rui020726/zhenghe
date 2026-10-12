# ROUND67 Errata 2 - the 5000 forward-row cap was exceeded (append-only)

**Task:** ROUND67 final reconciliation, second addendum
**Branch:** `analysis/round67-late-adaptation-diagnosis`
**Base HEAD (Errata 1):** `be08c5495fd69f7fd7b88ca712900f32cfa54bd7`
**Nature:** read-only. 0 forward, 0 training, 0 simulation, 0 test, no new experiment. Evidence is
taken from the ORIGINAL session commandExecution tool records (turn
`01a13038-88e9-79e0-bb70-c1cea769613b`), which persist independently of any on-disk log; the absence
of a disk invocation log does not erase an executed forward.
**Action:** RETRACT the "cap not breached" statement in Errata 1 and the "4711 of 5000 permitted
rows" framing in the original report / execution basis. Record the true forward-row consumption.
The original ledger, Errata 1, and every R67 deliverable are preserved unchanged.

## What actually ran (from the original session tool records)

| Exec ID | Exit | Guard charge at the time | Actual forwards | Persisted? |
|---|---|---|---|---|
| `chatcmpl-tool-ab6eca2d9f24c9c6` (run A) | 1 | `guard.charge(mat.shape[0], ...)` counted ONE arm | loop forwarded BOTH NEW and BC = 2 x 2354 = **4708** | no (crashed after forwarding) |
| `chatcmpl-tool-844962e1339d71a9` (run B) | 0 | `guard.charge(2 * mat.shape[0], ...)` (fixed) | **4708** | yes, `r67_probability_panel.json` |
| `chatcmpl-tool-9ddcba19c384d656` (run C) | 0 | focal probe | **3** | yes, `r67_focal_attribution_probe.json` |

Run A evidence (verbatim from the tool record): `planned forward rows = 4708`;
`forward rows used: 2354 in 9 charged batches`; `recompute-vs-saved records 2232 matched 2232
mismatched 0`; then `main` line 371 failed on the controllers output (`None.items`). The
`guard.charge(mat.shape[0])` bug charged one arm while the `for arm in ("NEW","BC")` loop forwarded
both, so run A really forwarded 4708 rows while the guard recorded only 2354. All 4708 forwards
completed (the 2232-record recompute ran and matched); the crash happened afterward, at output time.
Run A did not land its output JSON, but the forwards were real and are not refundable.

Run B re-ran the identical 374 + 1860 + 120 input panel with the corrected charge and persisted the
4708-row `r67_probability_panel.json`. Run C forwarded the 3 focal rows.

## Corrected cumulative accounting

- Documented forward rows = 4708 (A) + 4708 (B) + 3 (C) = **9419**.
- Cap = 5000, so the budget was **exceeded by at least 4419**. Any further undocumented execution
  stays `unknown` and can only raise the floor, never lower it.
- "cap not breached (4708 and 4711 <= 5000)" in Errata 1 is RETRACTED. The 4708 / 4711 figures were
  the persisted, deduplicated accounting; they under-count real consumption because run A forwarded
  the full panel without persisting and before the guard bug was fixed. The on-disk ledger
  (`r67_forward_row_ledger.json`, 4711) is therefore also incomplete: it misses run A entirely.

## Resource violation vs scientific eligibility (kept separate)

- Resource accounting: this is a real over-consumption of the 5000-row budget (>= 9419). The round
  must NOT be described as "4711 of 5000 permitted rows" or otherwise compliant.
- Scientific content: runs A and B are the SAME deterministic panel (identical 2354 distinct inputs,
  CPU `no_grad` forward), so run B persisted the same numbers run A computed; the duplicate cost
  extra compute but changed no measured value. The over-consumption does not by itself invalidate the
  action-equivalence (192/192), saturation, recompute (2232/2232 within 1.19e-07) or focal-probe
  records; those stand as measured observations.
- Withdrawn is only the within-budget / compliance framing and any implication that 9419 rows can be
  relabelled a compliant 4711. Errata 1 corrections 1 (clip) and 2 (sampling) are unaffected.

## Consumption of THIS addendum (zero side effects)

0 forward, 0 environment, 0 simulation, 0 training, 0 optimizer, 0 PPO, 0 checkpoint write, 0 test.
Read-only parsing of persisted R67 artifacts plus the cited original-session tool records; no new
forward was run to prove anything.

Protected ROUND46 files re-verified byte-unchanged (cf3ad198..., 93a2f69f...). No push, no merge,
no deploy.
