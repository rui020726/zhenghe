# ROUND75 S1G Phase A close-out

Request ID: `R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015`  
Branch: `exp/round75-native-ppo-qualification`  
Base HEAD: `ff5a1409812633dfcce00288858576960b77ec39`  
Base parent: `d2a634facbc25bea1acd7be3e01474ddca0a5a08`  
Output root: `runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/`

This report covers Phase A only. It records the already-approved source changes, the new
S1G plan and outer receipt wiring, and three narrow pure-test Python attempts. It does not
launch Phase B and does not treat the historical `7eff4a4` S0 preparation as this request.

## Scope and classification

The scientific replay remains A1 + A2 + A3: one critic-only control step, one native update
of four physical optimizer steps, and the same update replay from the disk bundle with four
physical optimizer steps. A1 and A2 physical steps are normal. A3 physical steps are
technical and use the dedicated registration prefix. The scientific arithmetic is unchanged:
4 physical optimizer steps and 70 trunk calls / 840 target rows per native update.

Only physical optimizer steps carry the technical classification. Every new trunk target
row and call remains normal: A2 840 normal, A3 840 normal, total 1,680 normal. The two new
batch-read events also remain normal. There is no authorization or route that reclassifies
trunk rows as technical. Existing production counters and caps are not rewritten.

The S1G baseline is the old production state, not a zero ledger: 5 normal physical steps,
840 normal trunk rows across 70 calls, one normal batch read, and zero environment
construction. After the planned S1G attempt, the physical category totals are 10 normal /
4 technical / 14 hard; the trunk totals remain 2,520 normal / 0 technical / 2,520.

## Source identity and wiring

- `r75_frozen.py` adds `physical_step_category` and
  `physical_step_registration_prefix` to `r75_native_ppo_update`. It rejects a technical
  category without the prefix and a prefix on a normal step. The category and prefixed
  `what` are passed to `resources.charge_physical_step`; no optimizer, loss, epoch, clip,
  or trunk-meter arithmetic changes.
- `r75_s1_runner.py` adds the S1G request, plan, attempt directory, technical prefix, and
  `--run-s1g` CLI. In the S1G path, A3 calls
  `pre_register_technical_attempt(S1G_TECHNICAL_PREFIX)` before `r75_native_ppo_update`,
  then passes `physical_step_category="technical"` and the same prefix. A2 remains normal.
  The S1G request is accepted only with its frozen request ID and private attempt
  directory.
- `r75_s1g_outer_exec.py` is a new pure-stdlib parent. It separates prepare and launch,
  preserves the failed historical attempt, writes an O_EXCL one-time attempt marker, runs
  one real CPU child exactly once, captures binary stdout and stderr, and records both the
  real child return code and the wrapper return separately. It contains final SHA pins and
  no placeholder hashes.
- `ROUND75_S1G_RUN_PLAN.json` freezes the S1G contract, the non-zero production baseline,
  the normal/technical accounting above, child receipt requirements, and Phase B as one
  authorized attempt after Phase A source review.

## Frozen source and plan SHA256

| File | SHA256 | Bytes |
|---|---|---:|
| `r75_frozen.py` | `6531152014bf6698b5c474abee1576c8d9cf82ba69f128a553f2f38a90a70f28` | 42,667 |
| `r75_s1_runner.py` | `5699fadd6c7fc3eae5a58a3687285938e519f434d6e8cd7a061c193d5d09dfac` | 100,622 |
| `r75_s1g_outer_exec.py` | `1f463ff60bea129b2b0fd7001e0ad1bfd3db5bcdbfc27045da14072eb62bf1cb` | 36,513 |
| `ROUND75_S1G_RUN_PLAN.json` | `8d1d46acfe6633d7c02bbeae7377984323e4157f030e1d4e4802a9bbdb1b364e` | 24,248 |
| `test_r75_s1g_phase_a.py` | `5f5e87504b59aa324e5ded84b0864d09d8131b1489e1a09a55a43fd7832770ff` | 5,567 |

## Narrow pure-test attempts

The suite has six distinct cases and no Torch or NumPy import. It uses AST checks for the
frozen fee route and runner/outer wiring, reads the frozen plan, and exercises the real
guard pre-registration API only in a disposable fixture under the repo. It also proves that
the canonical production root is refused as a fixture root. No canonical production counter
or ledger was used.

The final 6-case suite did not pass as a whole. The first three attempts executed six cases
each (18 case executions): attempt 1 had two errors, attempt 2 had one assertion failure, and
attempt 3 passed 6/6. Attempt 4 failed at import before executing any of the six cases, so the
total is 18 case executions, not 24. The four real exit codes are 1, 1, 0, 1.

1. Attempt 1, exit code 1. It had two errors: the default OS temp directory was not writable
   under the execution sandbox, and the plan test looked for `cumulative_caps_after_attempt`
   at the wrong JSON level. Receipt: `test_r75_s1g_phase_a_exit.json`.
2. Attempt 2, exit code 1. One assertion failed because the expected dictionary omitted the
   plan's `contract_caps` field. Receipt:
   `test_r75_s1g_phase_a_attempt2_exit.json`.
3. Attempt 3, exit code 0. All six cases passed. Receipt:
   `test_r75_s1g_phase_a_attempt3_exit.json`.
4. Attempt 4, exit code 1. After the incremental source-review edits, the process failed at
   import on `r75_s1_runner.py:1613` with `IndentationError: unexpected indent`; none of the
   six cases ran. Receipt: `test_r75_s1g_phase_a_attempt4_exit.json`.

All three attempts and logs remain as Git-external originals. No test process was run after
the passing attempt.

This sentence is superseded by the final close-out: the passing attempt 3 no longer qualifies
the later source revision. Attempt 4 binds the final source to an import failure. The final
source is unverified and Phase B is forbidden.

## Read-only close-out findings

The frozen S1G accounting is internally coherent, but the existing precheck is not S1G-bound:

- `r75_s1_runner.py` always calls `r75_s1_precheck.main()` inside `run_s1()`.
- `r75_s1_precheck.py` hardcodes module-level `PLAN_PATH = HERE / "ROUND75_S1_RUN_PLAN.json"`.
  It does not read `S1G_PLAN_PATH`, and the S1G runner path does not hand it a new plan.
- The old plan and the S1G plan both pin `r75_frozen.py`
  `83813d6e06971a1e1fc1e25d9e9f94b0fa72310e300e037c0fa73d539c540b80`, while the final working
  `r75_frozen.py` is `bb4bae98518cc482f4b733794ee1c3f93097c56c5b3e85d6fed2c65f2a72198e`.
  Therefore a real S1G launch would fail identity verification even if the source imported.
- The precheck's ledger check also asserts a zero ledger. The real S1G contract starts from the
  non-zero old production baseline. This is not safe to bypass; the precheck must be revised in
  a later authorized source revision before any launch.

The final runner also contains the known `IndentationError`. It is retained unchanged here.
No syntax repair or fifth test was performed. The outer receipt path and accounting plan are
evidence, not a license to launch Phase B.

## Resource boundary

Phase A performed zero Torch imports, zero NumPy imports, zero model load or construction,
zero forward or backward, zero real or synthetic optimizer step, zero PPO update, zero
environment construction, zero sampling, and zero checkpoint write. It did not restore RNG
state and did not launch the outer runner. No technical attempt was registered in the
canonical production ledger.

The pure guard fixture test made one technical charge and one unregistered-technical refusal
inside its disposable temporary root. This is diagnostic, not production or Phase B
consumption. The canonical total ledger remains:

`r75_total_ledger.jsonl` SHA256
`fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d`, 19,980 bytes.

## Handoff

Phase A is complete and locally committed. Phase B remains not started and is allowed only
as the already-approved one real CPU NN attempt after source review, through the frozen
outer receipt path. A failed B attempt is a final case result, not an authorization to
retry, rewrite the plan, or borrow unused normal headroom.
