# ROUND42-R Report — one-shot SEND/SKIP first-report selection ablation

## Execution identity

- Original ROUND42 prelaunch HEAD: `f46fe4703aa03ce6400f63c601c3203ef741dd9d`
- ROUND42-R execution HEAD: `0369408dec2cfabffb623e0445174725f9dde0d5`
- ROUND42-R execution branch: `fix/comm-one-shot-gate-resume` (the preceding ROUND42 branch was `fix/comm-one-shot-runner-counting`).
- ROUND41 delivery HEAD: `cc28a28fc5fb13e31ab9ed0fc3456797aadcc115`
- ROUND40 delivery HEAD: `8f8245a665b15960e70194fc6e245221b70db90e`
- Training version: `constrained-cost-v2/one-shot-send-skip/1/published-completion-event-at-relay-tick/2`
- Runner version: `counted-jointrun-proxy/1`
- Action semantics: `one-shot-send-skip/1`
- Runtime imports were pinned and path-checked against the ROUND40 canonical input HEAD `4e989a9088111ce6d9171cb1a21629a4b2b1e861`. The ROUND41-derived checkout has 17 committed dependency snapshots with different content; those snapshots were not imported. See `ROUND42_CONFIGURATION_DIFF.json` for both hashes and active-source paths.
- Frozen execution duration: 7795.1 seconds

## Launches and updates

- Normal claims, including the preserved failed claim: **672 / 672**
- Cumulative constructor calls, including the old failed attempt: **673**
- Cumulative `JointRun` constructions: **673**
- `run()` invocations / underlying method calls: **672 / 672**
- Completed steps / successful logical jobs: **672000 / 672**
- Simulations started: **672 / 672**
- Technical reserve used: **1 / 28**
- Total launch claims: **673 / 700**
- Preserved old technical failure / new technical failures: **1 / 0**
- Failed scene 1300000 was retried as reserve attempt 2; retry gate: **passed**.
- The preserved failed attempt is excluded from scientific outcome rows and paired analysis.
- PPO update calls / actual Adam optimizer steps: **96 / 1092**
- Committed batch-record step sum matches the Adam hook: **True**
- ROUND42 checkpoint write calls: **297**
- Synthetic optimizer steps: **0** (the pure-compute gate guarded this before launch).
- No extra smoke, independent stream, or throughput probe was launched.

## Prelaunch and first-batch gates

- Pure-compute gate: 20 tests passed; environment constructors=0, optimizer steps=0, checkpoint writes=0.
- Each seed's first four episodes produced one real PPO batch before continuation.
- The actual batch gate independently checked action-set closure, SEND/SKIP likelihood, event ticks, cost bundling, GAE targets, paired `g_hat`, lambda update, optimizer parameter scope, both critics, and checkpoint identity.
- Gate artifacts: `ROUND42_FIRST_BATCH_GATES.json`.

## Training outcomes

| Seed | On-time tasks / 128 | Bytes | First-report emissions | Free SEND | Free SKIP | Final λ | Adam steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 739 | 762114 | 305 | 305 | 457 | 1.287500 | 352 |
| 22 | 732 | 449154 | 180 | 180 | 582 | 1.375000 | 400 |
| 33 | 741 | 737754 | 296 | 296 | 466 | 1.262500 | 340 |

Training actions are sampled. All 128 episodes use the frozen ROUND40 scenario order, and no best-seed or best-checkpoint selection was performed.

## Development comparisons

Each row is paired on the same 16 registered development scenes. Greedy and sampled results remain separate.

| Seed / mode | OLD tasks / bytes | NEW tasks / bytes | Immediate tasks / bytes | NEW−OLD tasks / bytes | NEW SEND / SKIP |
|---|---:|---:|---:|---:|---:|
| 11 / greedy | 88 / 0 | 88 / 0 | 96 / 239028 | +0 / +0 | 0 / 96 |
| 11 / sampled | 94 / 99276 | 94 / 94710 | 96 / 239028 | +0 / -4566 | 38 / 58 |
| 22 / greedy | 88 / 0 | 88 / 0 | 96 / 239028 | +0 / +0 | 0 / 96 |
| 22 / sampled | 95 / 110304 | 93 / 55896 | 96 / 239028 | -2 / -54408 | 22 / 74 |
| 33 / greedy | 96 / 239028 | 88 / 0 | 96 / 239028 | -8 / -239028 | 0 / 96 |
| 33 / sampled | 95 / 151374 | 92 / 50250 | 96 / 239028 | -3 / -101124 | 20 / 76 |

The machine-readable paired file lists scene-level task deltas, recovered/lost target IDs, target swaps, maximum single-scene loss, cost, emissions, packets delivered/lost, and mask/refusal counts: `ROUND42_PAIRED_RESULTS.json`.
The accepted-emission records in that file are relay-accepted send records (cost charged at emission); recipient delivery and packet loss counts are reported separately, and each raw transcript is linked by path and hash.

Rule baselines on the same 16 scenes:

| Rule arm | On-time tasks | Bytes | First reports | Packets delivered | Packets lost |
|---|---:|---:|---:|---:|---:|
| late_discovery_gate | 96 | 77412 | 31 | 125 | 61 |
| always_hold | 88 | 0 | 0 | 0 | 0 |


## Verdict indicators

- Task counts match immediate per scene, with fewer first reports and lower bytes, in all six seed/mode pairs: **False**.
- All free NEW choices are SEND: **False**.
- All free NEW choices are SKIP: **False**.
- OLD exactly matches NEW tasks and bytes in every scene and pair: **False**.
- Greedy and sampled task/cost outcomes differ by seed: `{"11": true, "22": true, "33": true}`.

These are descriptive checks on the registered development block. They do not support significance, population noninferiority, or generalization claims. The policy offers one immediate SEND/SKIP decision; later fixed HOLDs close an already chosen target and do not represent a learned wait duration.

## Checkpoints and source artifacts

- NEW ep0, ep64, ep128 and every batch parameter/checkpoint link are in `ROUND42_CHECKPOINT_INDEX.json`.
- ROUND40 ep0 identities match each NEW ep0 initialization; OLD ROUND40 ep128 files were only loaded read-only and their byte hashes were checked after evaluation.
- Raw local artifact inventory contains 4349 files with absolute paths, sizes and SHA256: `ROUND42_RAW_ARTIFACT_MANIFEST.json`.
- Raw lifecycles, transcripts, tapes, rows, checkpoints and launch claims remain Git-ignored.

## Limitations and next-step boundary

- ROUND42 uses the registered 16-scene development block, not an independent test.
- OLD's historical first-report-only weights interpret HOLD as episode SKIP under this explicit one-shot evaluation protocol; historical ROUND40 behavior is not equivalent.
- Results do not establish that all communication timing or full communication-frequency control is solved.
- Any additional smoke, tuning, training, or confirmation experiment requires separate authorization.

## Deliverable commit

The post-run report and manifests are committed locally on the ROUND42-R branch. No push, merge or deployment was performed.
