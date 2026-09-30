# ROUND33 checkpoint mapping erratum

ROUND33 reported only 6/18 checkpoint metadata comparisons as matching, with conflicts at every b15 and b31 point. The original ROUND33 outputs remain unchanged. This erratum corrects the sidecar mapping used for that comparison; it does not verify serialized checkpoint tensors.

## Cause

In `runs/comm_temporal_credit_audit_20260930/r33_temporal_credit_audit.js`, the fixed-point loop reads:

```js
const recordPath = group.batch_records[String(batchIndex)];
```

`group.batch_records` is a 32-element filename-sorted array, not a mapping keyed by numeric batch. For every arm and seed, positional entry 15 is the sidecar named `b22`, and positional entry 31 is `b9`. The subsequent checkpoint comparison therefore compared the b15/b31 checkpoint index entry to a different batch record and tape set.

This is visible in the saved ROUND33 diagnostic. For `ACTIVE_ROWS_s11/b15`, it reports episode rows 88–91, which are batch 22. The correctly named `r32_batch_ACTIVE_ROWS_s11_b15.json` contains episodes 60–63. For `ACTIVE_ROWS_s11/b31`, it reports episodes 36–39, which are batch 9; the correctly named `r32_batch_ACTIVE_ROWS_s11_b31.json` contains episodes 124–127. The same positional mapping is used for the other four arm/seed pairs.

## Corrected mapping and evidence

ROUND34 joins the sidecars by their explicit JSON `batch` value and independently verifies the filename suffix. It then checks the referenced episode rows and tape files, matching scenarios and sample counts against the sidecar and `UPDATE_MECHANISM.json`.

- Corrected JSON metadata alignment: 18/18 points.
- Checkpoint file-byte SHA256 matches its index entry: 18/18.
- Parameter digest claim in the index, `controller_parameters_before`, and mechanism record: 18/18 equal as strings.
- Lambda, episodes-done, update count, optimizer-step history, batch scenarios, episode IDs, and tape/sample counts: 18/18 internally consistent after the corrected join.
- The current 1,040-row inventory and 768-training-tape inventory hashes equal the ROUND33 audit's recorded inventory hashes.

The correction resolves the reported JSON mapping conflicts. It does not establish that a `.pt` payload contains the indexed weights: PyTorch is unavailable, and no checkpoint was deserialized. Thus all 18 file hashes pass, the corrected index/batch/mechanism/tape claims coordinate, but actual payload metadata, parameter contents, module contents, forward, and native likelihood remain unavailable or unrun.

The per-point mapping and validation layers are in `CHECKPOINT_IDENTITY_RECONCILIATION.json`. `CHECKPOINT_INDEX.json`, all batch records, all `.pt` files, and every ROUND33 artifact are preserved byte-for-byte.
