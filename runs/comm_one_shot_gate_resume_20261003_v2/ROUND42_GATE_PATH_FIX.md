# ROUND42 first-batch checkpoint path gate repair

## Cause

The stopped seed 11 gate compared the batch record checkpoint-key set with keys made by
Path.as_posix(). The batch record contains Windows backslashes, so the same two registered
files were represented differently and the set assertion stopped before the per-file SHA256
checks. The original failed gate record and stop state remain in the prior run directory and
are also copied under source_snapshot/previous_resume_attempt01 for this isolated continuation.

## Repair

The copied, versioned gate routes the actual first-batch path comparison through
_validate_frozen_checkpoint_path_map. It normalizes Windows slash variants relative to the
explicit canonical input root, rejects drive/root escapes and normalized-key collisions,
compares the complete relative directory and filename set, checks each record hash against the
freeze, and re-hashes the file at its frozen location. It does not accept basename-only matches.

The gate repair is limited to this copied audit implementation. The separate ROUND42-R attempt-2
resume adapter is documented in ROUND42_RESUME_ATTEMPT02_ADAPTER.md; it changes only the scope
of the parameter_identity import needed by the resumed ROUND40 trainer. The original scientific
training runner, optimizer, objective, policy, training-version files, and external frozen
ROUND40 source remain unchanged.

## Regression evidence

test_r42_gate_resume.py exercises the helper used by the real gate. The real seed 11 batch-0
record and frozen config reproduce the old raw-set mismatch while the repaired helper resolves
both exact frozen files and verifies their content hashes. Synthetic tests also cover slash
equivalence, another directory with the same basename, wrong digest, missing file, duplicate
normalized record/frozen keys, and a path outside the declared drive/root.

The nine targeted tests passed in Python 3.10.19 / NumPy 2.2.6 / PyTorch 2.9.0+cu128 using
runs/comm_one_shot_gate_resume_20261003_v2/test_r42_gate_resume.py. These tests used temporary
files and read-only frozen inputs: 0 environment constructions, 0 environment starts, 0 optimizer
steps, 0 training-checkpoint writes. The full real gate and read-only exact-resume validation
remain part of the v2 prelaunch gate.
