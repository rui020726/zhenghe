# Paired expansion: legacy 100 + fresh 60, both arms, zero regressions

The small-scale gate passed 6/6, so per the brief's conditional expansion both existing
analysis blocks ran paired, two arms each, on the frozen commit
`6915d6383fb61b37e1652fca0c4eb73b4b5b32a2` (verified in every run's `run_signature` by
the preflight; 320 runs total, single main process, 8 worker shards, atomic renames).
Both blocks were read during development: **neither is a held-out test set**.

## analysis_legacy (seeds 20000-20099)

| arm | completed | expired | coverage resets | lost work-steps |
|---|---|---|---|---|
| baseline `coalition_planner`+sticky | 579 | 9 | 1 | 1.2 |
| candidate `coalition_redundant_release` | **588** | **0** | **0** | **0** |

All nine baseline expiries became on-time completions (+9/−9), across five seeds:
20003 (five targets, including the never-covered `piloc-e1-target-0` and the
finished-after-deadline `piloc-e1-target-4`), 20008 (`target-2`), 20052 (`target-3`),
20070 (`target-2`), 20083 (`target-4`). **Zero regressed targets in any of the 100
paired episodes.**

## analysis_fresh (seeds 21000-21059)

| arm | completed | expired | coverage resets | lost work-steps |
|---|---|---|---|---|
| baseline | 355 | 0 | 0 | 0.0 |
| candidate | 355 | 0 | 0 | 0.0 |

The fresh block had no baseline expiries, so there was nothing to rescue; the candidate
is outcome-identical, which is the no-harm side of the design: a mechanism gated on
redundant-presence-with-alternative does not disturb episodes without the disease.

## Reading

* Measured, closed loop, not an estimate: the release eliminates every expiry in both
  blocks and every rescue is a per-target transition from not-on-time to on-time.
* The blocks are development-read (the legacy block chose arms in earlier rounds, the
  fresh block confirmed the learnability gate). Neither may be reported as held-out; a
  genuinely untouched block would be needed for that claim.
* Paired per-seed rows are in `paired/expansion.json`; raw `paired/*/*.jsonl` are
  uncommitted by design.

Reproduce:
```
D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/paired_expansion.py
```
It resumes (skips existing outputs), re-checks the preflight, and re-aggregates.
