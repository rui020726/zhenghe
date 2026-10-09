# ROUND58 gen3 execution basis

- Task: ROUND58-GEN3-FIXED-PLAN-EXECUTION
- Source: direct user task instruction in the current Codex conversation, received 2026-10-09; this is the source of the present authorization, not an inference from the old R58 authorization or a synthetic test fixture.
- Approval status: the user supplied the conditional execution prompt as the direct task and explicitly requests execution. The prompt's own approval clause states that, subject to the verified history count of 16, the approved new-generation accounting is 16 historical calls retained + 1024 normal tasks + up to 24 technical reserve, hard cumulative cap 1064.
- Explicit task scope: wire the new cumulative quota only in the private adapter; bind and commit an identity-complete gen3 freeze; run first-four correctness gates; if they pass, complete the fixed 384 training + 576 model-evaluation + 64 rule-evaluation plan; report statistics and commit locally. No push, merge, deployment, parameter changes, or expansion.
- Scientific invariants: beta=1.00; R53 critic isolation; positive cost and lambda*A_task-A_cost; 19D one-shot SEND/SKIP and sequential budget mask; frozen reward timing, GAE, PPO, dual updates, search/allocation weights, and training order.
- Historical evidence contract: preserve all old claims and category fields; 16 actual calls count once in cumulative hard cap; historical category conflict remains unresolved; 48 old optimizer calls remain non-effective and outside the gen3 training chain.
- Source artifact: `ROUND58_GEN3_EXECUTION_BASIS.json` (hash-bound by the private launcher and execution receipts). The direct prompt is the approval source; this file records its scope and provenance for an auditable execution handoff.

This basis does not authorize any change beyond the task above. Any change to science, seeds, task count, scenario block, or caps requires new approval.
