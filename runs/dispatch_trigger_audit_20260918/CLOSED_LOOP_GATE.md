# Closed-loop gate: the redundant-member release rescues all six, and nothing else moves

**Verdict: every gate PASSES.** Baseline `coalition_planner` (binding, sticky) vs
candidate `coalition_redundant_release`, same five failure seeds, same recorded detection
streams, 1000 steps each, full-episode replay. The baseline reproduces the committed
868d68a arm exactly (per-target finals identical; the probe replays of the same arm were
verified byte-identical to the committed traces).

| seed | arm | completed | expired | resets | lost work | override | rejected | re-commands |
|---|---|---|---|---|---|---|---|---|
| 20003 | baseline | 1 | 4 | 0 | 0.0 | 0 | 0 | 0 |
| 20003 | candidate | 6 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| 20008 | baseline | 5 | 1 | 1 | 1.2 | 0 | 0 | 0 |
| 20008 | candidate | 6 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| 20052 | baseline | 5 | 1 | 0 | 0.0 | 0 | 0 | 0 |
| 20052 | candidate | 6 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| 20070 | baseline | 5 | 1 | 0 | 0.0 | 0 | 0 | 0 |
| 20070 | candidate | 6 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| 20083 | baseline | 5 | 1 | 0 | 0.0 | 0 | 0 | 0 |
| 20083 | candidate | 6 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| **total** | baseline | **21** | **8** | 1 | 1.2 | 0 | 0 | 0 |
| **total** | candidate | **30** | **0** | 0 | 0.0 | 0 | 0 | 0 |

Rescued (candidate on time, baseline not): all six — `20003/target-1`, `20003/target-0`
(never covered → completed), `20008/target-2`, `20052/target-3`, `20070/target-2`,
`20083/target-4`. Five of the six were *finished-after-deadline* cases; the release
assembled their coalitions early enough to finish inside the deadline. New failures:
none. Deadlocks: none. First-run churn (98 own-task re-commands) was removed by the
allocator's own-task suppression, which is part of the candidate arm; the table above is
post-suppression.

| gate | result |
|---|---|
| rescued at least 3 | **pass** — 6 |
| no new failure | pass — 0 |
| no new execution break | pass — `executed_override_unjustified` 0 both arms |
| zero invalid commands | pass — 0 rejected |
| zero deadlocks | pass — longest idle streak 0.0 |
| no command churn | pass — 0 own-task re-commands both arms |

Reproduce: `D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/closed_loop_gate.py`
(writes `CLOSED_LOOP_GATE.json`; raw `closedloop/*.jsonl` uncommitted by design).
