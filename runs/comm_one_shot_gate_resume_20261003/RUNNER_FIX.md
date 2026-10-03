# ROUND42-R runner fix

## Previous stop and immutable attempt

ROUND42 stopped at the first `immediate_reference_scene1300000` launch. The original branch's final delivery is `70faab3f451b3c17690c189216c315cccaf4a6d6`; the failed run was frozen at `3c0b61cbdb702c3e4d8b1a4668030bda2db102ab`, based on `ffa87ceb1a9d48abe27fc124ef1a963470878604`. The preserved claim records one normal launch and one builder invocation, while the traceback shows the original builder returned a `JointRun` before assignment to its read-only, slotted `run` method failed. No call to `run()` or simulation step followed.

The old claim, slot, empty log, exception, stop state, and inventory remain untouched in the original ROUND42 directory. `ROUND42R_CARRYOVER.json` records their content hashes and imports the old failed claim and normal slot byte-for-byte into the isolated ROUND42-R ledger. The retry is attempt 2 under reserve slot 1. The old claim remains failed; the logical job points to the successful reserve attempt only after its retry gate passes.

## Repair

The defect was in `r42_run.configure()`'s `counted_build` closure. It attempted `run.run = counted_run` on the production `JointRun`, which is a `@dataclass(slots=True)` with a read-only method attribute. ROUND42-R replaces that assignment with `_make_counted_builder()` and `_CountedJointRunProxy` in `r42_run.py`.

The builder calls the original production builder once and wraps the returned object without copying it. Attribute reads and writes go to that same object. Its `run(*args, **kwargs)` forwards the original arguments once and returns the original return object; it preserves raised exception identity. The one-shot policy refusal guard remains around the same underlying call. The proxy records separately:

- builder invocation and returned `JointRun` construction;
- proxy `run()` invocation and underlying `JointRun.run()` call;
- run return or exception;
- completed orchestrator steps, read from `orchestrator.step_index` after the call.

`simulation_started` is true only when at least one completed orchestrator step is observed. A builder error has no run object to inspect and is recorded without a `finally` access to one. Failure filenames include the attempt number so a retry cannot replace earlier evidence.

The change does not modify the production `JointRun`, event order, environment transition, SEND/SKIP protocol, observations, costs, budget, rewards, GAE, PPO, dual critics, optimizer set, initial weights, training order, or evaluation decoding. Training semantics remain `one-shot-send-skip/1` and the existing ROUND42 training version. The separate execution identity is `counted-jointrun-proxy/1`.

## Regression and execution identity

`test_r42_runner_counting.py` passes through the same `_make_counted_builder()` factory installed by production `configure()`. A slots-only fixture first demonstrates that the previous instance assignment fails. The regression checks single construction, zero execution while wrapping, exactly one delegated call, argument and return identity, exception identity, attribute delegation, build-versus-run failure classification, completed-step accounting, and zero real `JointRun`, SAR environment, optimizer, and checkpoint activity.

The ROUND42 original 12 pure-compute tests are freshly rerun with the runner seam regression before the new freeze is written. The frozen identity records hashes for the ROUND42-R code files, inherited production inputs, task/search checkpoints, continuation basis, carryover manifest, and unchanged ROUND40 scientific configuration. The freeze records the actual repair-basis commit and the subsequent committed prelaunch HEAD is written before any environment launch.

The retry gate is `ROUND42R_REFERENCE_RETRY_GATE.json`. It requires the reserve attempt to use scenario 1300000, the immediate policy, the frozen search and allocation checkpoint hashes, one builder return, one `run()` call, exactly the frozen episode steps, and closed task and airtime ledgers. Passing this gate allows the executor to continue automatically with the remaining 63 immediate references and the original fixed plan.

## Quota carryover

The original plan remains 672 normal logical jobs plus 28 technical reserve launches under a 700-launch hard cap. ROUND42-R carries forward the old failed normal slot 1, uses one reserve launch for the required retry, and has 671 new normal claims remaining. If every logical job succeeds, the final ledger is 672 normal claims (including the old failed claim), 1 reserve claim, 673 cumulative builder invocations/constructions, 672 completed logical jobs and successful run calls, and 27 unused reserve launches. The old failure is excluded from scientific outcome rows and paired analysis.

## Remote and scope

The requested delivery is local only. No Git push, deployment, additional stream, smoke episode, extra environment, or work outside the fixed 672 logical jobs is part of this continuation.
