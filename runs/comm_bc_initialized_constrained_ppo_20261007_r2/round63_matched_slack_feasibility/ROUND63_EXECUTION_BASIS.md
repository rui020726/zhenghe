# ROUND63 Execution Basis

## Objective and boundary

Audit whether the saved R60/R61 early-first-report cases support a covariate-matched comparison of ready time versus deadline slack. This is a read-only, zero-environment analysis. It does not create an environment, claim, episode, model instance, forward pass, RNG restore, training update, checkpoint, or evaluation.

The output is a feasibility result, not a new intervention or a causal estimate. R61 outcomes are attached only after candidate extraction, grouping, and covariate ranking have been completed.

## Frozen inputs

- Starting source revision: R62 HEAD `62dcb2419f323461b6aa07bb79ce39e3d17a4ac7` on `analysis/round63-matched-slack-feasibility`.
- Model identity reused from the R62 audit: R60 compliant seed33 ep128 greedy; file SHA256 `573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a`. Model bytes are not loaded by this analysis.
- Frozen focus: scene `1700027`, target `piloc-e1-target-5`, ready step 134.
- Candidate source: 32 saved R60 seed33 greedy scene tapes and their paired lifecycle records. Each tape SHA and each candidate's tape/lifecycle path and SHA are recorded in `MATCHED_PANEL.json`.
- Post-hoc effect source: R61 `raw/ROUND61_PAIRED_ANALYSIS.json`, SHA256 `4b303efa39563a92c5bb8fab8f74ccd84e6bec7e0eab26b09f1745c9d5446bbd`.
- R61 case-selection source: `ROUND61_CASE_FREEZE.json`, SHA256 `cba8d5b1a086630021a5955c55931d1cc9506ef7ce2143b75646320f5060ca4e`.

## Frozen extraction and matching

Read only saved sender-side tapes. Retain free first-report choices with `ready_step <= 147`, first-emission feature set, first-emission audit true, sequential affordability true, and initial send executable with reason `allowed`. Compute ready as `round(feature[18] * 1000) - round(feature[0])`; verify it equals the recorded decision step. Keep the full 19D vector.

Use exact `(ready_step, backlog)` groups for primary support and exact `(ready_step, backlog, has_future_requirements, budget_fraction)` groups as a stricter descriptive view. Missing remaining time remains null, is not replaced by zero, and is excluded from slack ranges. Remaining time is derived from the paired lifecycle deadline as `detection timestamp + deadline_offset - (decision_step - 1)`.

Cross-scene candidate order is lexicographic by absolute ready difference, absolute backlog difference, future-requirement mismatch, absolute budget-fraction difference, integer scene, then target ID. Exclude the focus scene. This order is descriptive and does not loosen the exact-support test.

Recompute the R61 case-selection distance for audit only: squared differences divided by candidate range on legal columns 13, 14, 15, and 18; constant columns contribute zero. This is not a 19D Euclidean metric and is not used as proof of common ready/backlog support.

Only after the candidate set, groups, and covariate rankings are materialized, join the eight R61 effects by `(scene, target)` from the raw paired-analysis JSON. No outcome field enters selection or ranking, and no outcome is imputed for other candidates.

## Validation and side-effect limits

Five standard-library pure-computation tests cover exact group separation, null slack handling, exclusion of same-scene candidates, deterministic ordering/tie-breaks, and invariance of matching output to mutated outcome fields. The analysis reads only the identified saved R60/R61 artifacts and writes the derived panel/report/verdict in this directory.

Hard-zero counters for this round: environment construction/run/steps, claims/slots, model loads/forwards, RNG restore, optimizer/PPO updates, checkpoint writes, and new evaluations. No R60/R61/R62 source or protected ROUND46 file is modified.
