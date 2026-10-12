# ROUND64 Resume-Seam Readiness

## Result

Candidate execution identity is `/7`, supplied by one separate candidate binding consumed by both executor and guard. The old `ROUND64_FREEZE.json` remains byte-for-byte unchanged (`cea674...a080e`); it is referenced by hash, not rewritten. New model-row version, executor version, and guard ledger version now resolve from that binding. The guard still accepts historical `/2` through `/6` records for read-only recovery, including the six completed cells; it does not require their old labels to equal `/7`.

An exact `/6` executor + `/4` guard counterexample is rejected in `run_cell` before terminal lookup, attempt context, slot claim, or builder. A consistent `/7` executor/guard/model row validates against the separate candidate freeze and immutable historical freeze. The candidate preserves scene 1700027, target-5, R60 seed33 ep128 model hashes, deltas `-30/0/30/60`, and the original 10 normal + 2 reserve = 12 hard cap. It grants no run authority.

The real raw ledger was read-only verified: 9 slots consumed (normal 7, reserve 2), 3 total slots of headroom (normal 3, reserve 0), 6 qualified cells. Four matrix cells remain unresolved; normal-07 d+30 FORCE_SEND has no terminal receipt. The new recovery gate refuses before any new slot: it reports four additional attempts needed, three normal slots available, zero reserve slots, and the nonterminal normal-07. No history was repaired or synthesized.

## Collector Test

The tested collector is created by `build_cell_collector`, the same factory called from `run_cell`, and derives from the inherited `OneShotSendSkipCollector`. Tests invoke its actual patched `_choose_masked` with synthetic `SendObservation`/`SendItem` and a deterministic stub controller. FORCE_SEND and FORCE_SKIP enter the free-action branch; a sequentially unaffordable candidate remains masked; the selected target closes after its one override while another target remains greedy; disabled L0 actions/tape equal the parent collector. Relay `_emit`, the real JointRun builder, and executor write path are patched to fail if reached.

The final narrow command was:

```text
python -m unittest ...test_round64_zero_env ...test_round64_resume_seam -v
Ran 30 tests; exit code 0
```

This includes 23 existing zero-environment guard/runtime tests and 7 new seam tests. Earlier pure-test iterations failed on a wrong collector fixture, missing import path, target fixture mismatch, and one guard indentation regression; each was corrected before the final 30/30 run. They consumed no R64 raw slot or environment attempt.

## Scope and Counts

No simulation, actual builder, real `JointRun.run`, claim, R64 raw slot, PPO/optimizer update, RNG restore, training restore, checkpoint load/write, or tape replay occurred in this readiness pass. R64 raw ledger SHA remained `0f36b1a1...8779e`; raw slot count remained 9. The two completed `/4`/`/6` qualification records and original strict failed gate were untouched. Existing protected ROUND46 hashes still match.

Pure tests used seven stub-controller forwards and two isolated fake `run()` returns in the pre-existing guard instrumentation tests; those fakes do not construct or step an environment. Temporary synthetic guard fixtures are separate from the persistent R64 raw ledger and are cleaned by the test suite. No synthetic optimizer was created or updated.

Modified code hashes: `r64_execute.py` `66ea6574...137c30`, `r64_guard.py` `8d1b207e...3f0fa`, `r64_identity.py` `a2f80d91...6b3af5`, `test_round64_resume_seam.py` `378ecb6b...f60d01`, candidate freeze `ROUND64_RESUME_SEAM_CANDIDATE.json` `28a0845c...d9b5dea5`. Unchanged: `r64_runtime.py` `3ae654d2...5bc099`, `r64_intervention.py` `eb77f35d...d8c46`, historical `ROUND64_FREEZE.json`, raw ledger, and raw strict gate. Full hashes and machine-readable counters are in `ROUND64_RESUME_SEAM_READINESS.json`.

Candidate code HEAD: `016a512bb7768fd28149916d19e87ecca77e9de5`. This establishes only zero-environment seam readiness. The four remaining cells still exceed the three available slots; a new authorization is required before any further episode. No 13th attempt is approved or implied.
