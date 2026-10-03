# ROUND42-R resume attempt 1 stopped before a new environment claim

Attempt 1 reached and passed both required read-only checks: the complete seed 11 batch-0 gate,
including both frozen PILOC/CTDE file hashes, and exact in-memory recovery of the registered
episode-4 checkpoint. It then stopped in the frozen ROUND40 resume branch before the first new
episode claim.

`r40_train.train_seed` imports `parameter_identity` only inside its fresh-start branch. The
resume branch loads the registered controller, optimizer, and state, then reaches a reference to
that function-local name and raises `UnboundLocalError`. This is a resume plumbing defect. It does
not change a scientific result.

No attempt-1 claim was created. The cumulative ledger remains at 68 normal slots, one reserve
slot, 69 environment constructions, 68,000 completed steps, one committed PPO update, 12 Adam
steps, and four checkpoint writes. The isolated claim directory still matches the source
directory's 69 claim files; all six checkpoint files match byte-for-byte. The exact episode-4
source remains SHA256 `a85722e0864456ea3eb735195c7ffcde4be9d42301bdf2bc0781db6747a851ef`.

The failed start marker, stop traceback, gate result, exact-resume result, and cumulative ledger
remain under `runtime/`; their hashes and the attempt delta are recorded in
`ROUND42_RESUME_ATTEMPT01_STOP.json`. This attempt is stopped. Any continuation uses a distinct
versioned resume-only adapter and a separate output directory, with the same original training
version and episode-4 checkpoint. The earlier version is not patched or reused for execution.
