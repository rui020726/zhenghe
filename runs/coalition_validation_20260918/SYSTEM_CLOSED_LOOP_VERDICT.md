# Live-system closed loop: real PILOC discovery, real CTDE allocation, 60 fresh seeds

The replay in `HELDOUT_VERDICT.md` fixes the detections and measures allocation. This
stage removes that simplification: PILOC searches live, so what gets discovered and when
is a consequence of the episode itself. 60 seeds (31000-31059, drawn against every
historical manifest **and** the held-out 400 before any run), two arms, same scenario
seed per pair, no shared stream, source `1f4a598` (`integrated_sar` identical to the
replay freeze).

## Results

| metric | baseline | candidate |
|---|---|---|
| goals placed | 360 | 360 |
| discovered by search | 356 (98.89 %) | 356 (98.89 %) |
| on-time completions | 350 | **356** |
| expired | 6 | **0** |
| completion rate of discovered | 98.31 % | **100 %** |
| zero-marginal-contribution working ticks | 14 511 | **13 450** |
| coverage resets / discarded work | 1 / 3.26 | **0 / 0.0** |
| release cycles / replan attempts | 0 / — | 843 / — |
| accepted reassignments / declines | — | 315 / 528 |
| own-task re-commands | 0 | **0** |
| rejected assignments | 0 | **0** |
| unjustified overrides | 0 | 0 |
| rescue travel distance (sum over robots) | 350.3 | 360.5 (+2.9 %) |
| wall-clock cost (sum, 60 episodes) | 1076 s | 1000 s |

Paired completion-rate-of-discovered (same estimator as the protocol: seed-level
bootstrap, B=10 000, RNG 20260918): mean **+0.0167**, 95 % CI **[0.0000, +0.0389]** —
the lower bound touches zero, so by the pre-fixed rule this is an **improvement trend,
not a demonstrated gain**, on 60 episodes (3 episodes better, 0 worse). The held-out
replay with 400 episodes is where the significance claim lives; this stage's job was to
look for *new* behaviour, and it found none.

## The feedback question, answered directly

**In every one of the 60 episodes the two arms discovered the same targets** — search
coverage identical, no episode where discovery differed. That is not a bug and not luck:
invariant 10 keeps the search team and the rescue team as separate entities sharing only
the scene and the clock, and allocation decisions move rescue robots, which PILOC's
observation does not read. So allocation cannot feed back into discovery *in the current
architecture*, and this loop confirms that empirically rather than assuming it. If the
two teams are ever merged or the search side starts reading task state, this stage must be
re-run: the replay would then understate coupling.

## Costs and safety

* Rescue travel rises 2.9 % — reassigned robots drive a little further. That is the
  mechanism's real price, paid in kilometres, not in failures.
* Zero rejected assignments (unlike the replay, where the class exists but is tiny), zero
  unjustified overrides, zero duplicate commands, zero coverage resets, zero discarded
  work.
* Runtime is unchanged within noise.

## Verdict

The full system produces no new failure mode from the redundant-member release, converts
all six live expiries into on-time completions, and costs a measured 2.9 % extra rescue
travel. The performance claim from this stage is a trend (CI touching zero at n=60); the
demonstrated claim comes from the 400-seed replay.

Reproduce: `system_seeds.py` then `system_closed_loop.py` (resume-aware, single parent
owns shards and aggregation); numbers in `SYSTEM_LOOP.json`.
