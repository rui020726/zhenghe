# ROUND42-R gate repair, exact resume, and fixed-plan completion

- Status: **complete**
- Resume execution version: `round42-gate-and-exact-resume/2`
- Gate path identity version: `round42-frozen-checkpoint-path-map/2`
- Training version: `constrained-cost-v2/one-shot-send-skip/1/published-completion-event-at-relay-tick/2`
- Action semantics / runner: `one-shot-send-skip/1` / `counted-jointrun-proxy/1`
- Execution source HEAD: `0369408dec2cfabffb623e0445174725f9dde0d5`
- Previous reviewed HEAD: `2825069eaf9acf0fff87a9a02e69ae2f8e88c785`; latest preceding delivery HEAD: `5cf41b5edd1b5e13f41b15abc08ae5876da4cd11`; original prelaunch HEAD: `f46fe4703aa03ce6400f63c601c3203ef741dd9d`.

## Gate and exact recovery

The seed 11 batch-0 gate was rerun through its complete original audit path. The real batch record and frozen checkpoint entries differed only in slash spelling; the repaired comparison retained the canonical root and full directories, rejected collisions, and rehashed both frozen files. All action, likelihood, reward tick, cost, GAE, dual, optimizer-scope, parameter-chain, version, and count assertions passed.

The approved seed 11 after-batch checkpoint was reloaded from its registered SHA256 `a85722e0864456ea3eb735195c7ffcde4be9d42301bdf2bc0781db6747a851ef`. Model modules, all Adam steps/moments, lambda/history/cursor, training order, Python/NumPy/Torch CPU RNG state, and next random outputs matched exactly. Diagnostics reapplied the registered RNG state before continuation.

Both checks were read-only: 0 environment constructions, 0 environment starts, 0 optimizer.step calls, and 0 checkpoint writes. The old failed gate and old stop state remain intact.

The first versioned resume attempt separately passed these same read-only checks, then stopped before creating a claim because frozen ROUND40 resume code referenced parameter_identity after importing it only in its fresh-start branch. Its stop record, traceback, gate and recovery outputs remain preserved. The attempt had zero claim/quota delta. This execution uses a separately frozen resume-only ROUND40 adapter that moves that import to function scope; after removing only that import from each AST, the adapter and frozen trainer modules are identical. The frozen external ROUND40 source is unchanged.

## Fixed plan and cumulative accounting

- Completed before this continuation: 68 successful logical jobs.
- Completed in this continuation: 604 logical jobs.
- Final total: 672 / 672 successful logical jobs.
- Claims: 672 normal + 1 reserve = 673 / 700 total claims.
- Environment constructions: 673; run calls: 672; completed steps: 672,000.
- Preserved original technical failure: 1; new technical failures: 0.
- PPO updates: 96; Adam steps: 1092; checkpoint writes: 297.
- Technical reserve remaining: 27 / 28; hard cap remains 700.
- 64 immediate references and seed 11 episodes 0–3 were reused from the original records; batch 0 was not replayed.

## Registered development results

Greedy and sampled comparisons are separate, on the previously used 16-scene development block. These results do not establish generalization or an overall zero-loss guarantee.

| Seed / mode | OLD tasks / bytes | NEW tasks / bytes | NEW−OLD tasks / bytes |
|---|---:|---:|---:|
| 11 / greedy | 88 / 0 | 88 / 0 | +0 / +0 |
| 11 / sampled | 94 / 99276 | 94 / 94710 | +0 / -4566 |
| 22 / greedy | 88 / 0 | 88 / 0 | +0 / +0 |
| 22 / sampled | 95 / 110304 | 93 / 55896 | -2 / -54408 |
| 33 / greedy | 96 / 239028 | 88 / 0 | -8 / -239028 |
| 33 / sampled | 95 / 151374 | 92 / 50250 | -3 / -101124 |

Task-count and communication-cost verdict indicator: **False**.
This is a descriptive check of the frozen development block. The OLD and NEW policies were evaluated under the same one-shot SEND/SKIP protocol. A full-SEND policy is not additional savings over immediate; an all-SKIP policy that loses tasks is not a success.

## Preservation and deliverables

- Original run inventory: 370 files; all hashes unchanged (`5756513d565efbafcbfd65a1187f3262222e2dcb1e8a877309ffe269156adbce`).
- The original prelaunch identity, failed gate, stop state, claims, ledgers, batch-0 rows/tapes/logs, and checkpoints were not rewritten.
- Machine-readable outputs: `ROUND42_FIRST_BATCH_GATES.json`, `ROUND42_EXACT_RESUME_VERIFICATION.json` (in runtime), `ROUND42R_EXECUTION_LEDGER.json`, `ROUND42_RESUME_FINAL_RESULT.json`, `ROUND42_CHECKPOINT_INDEX.json`, `ROUND42_RAW_ARTIFACT_MANIFEST.json`, and `ROUND42_VERDICT.json`.
- No push, deployment, extra scene, or confirmation experiment was performed.
