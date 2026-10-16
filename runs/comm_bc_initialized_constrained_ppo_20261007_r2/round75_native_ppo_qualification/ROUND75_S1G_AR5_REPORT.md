# ROUND75 S1G A-R5 T closeout

## Result

- Status: PASS
- Authorized pure stdlib Python process: 1/1
- Actual PID: 46272
- Actual OS exit: 0
- Tests: 6 run, 0 failures, result `OK`
- Raw Python stderr: `......\n----------------------------------------------------------------------\nRan 6 tests in 0.281s\n\nOK\n`
- Raw Python stdout contains the real case1 production B2 parser snapshot, 240 events and reconciliation OK.

## Receipt diagnostic errata and independent facts

Original claim, receipt, stdout and stderr were not rewritten. The receipt contains two derived PowerShell probe errors:

1. `production_nine_unchanged=false` is false. In the probe, `$x = $after | Where-Object ...` unwrapped the single matched record into a hashtable; `$x.Count` became key count 3 and `$x[0]` did not mean the first array item. Independent read-only comparison of all nine names, bytes and SHA256 values confirms 9/9 unchanged.
2. Fixture count `1/1` is false. It counted the parent fixture root instead of the task fixture root, and `Measure-Object` output was counted as objects. Independent read-only enumeration of `raw/ar5_ledger_schema_fixture/R75_S1G_AR5_LEDGER_SCHEMA_READONLY_REPAIR_20261015` gives 5 case directories and 25 files.

The non-empty stderr is normal unittest reporting; the original text explicitly contains `Ran 6 tests in 0.281s` and `OK`.

## Key reconciliation

- old B2 original_outer_reconciliation=false: preserved, not rewritten.
- new_parser_reanalysis_on_original_B2=PASS, with 240 total ledger events and all four units reconciled.
- parent_prelaunch_claim_missing=true: preserved, not backfilled.
- AR5_T_prelaunch_claim=raw/ar5_t/R75_S1G_AR5_T_LEDGER_SCHEMA_REPAIR_20261016/claim.json
- no_NN_rerun=true.
- No optimizer, PPO, environment, sampling, checkpoint write or B2 rerun occurred.

## Files

- Original claim: `raw/ar5_t/R75_S1G_AR5_T_LEDGER_SCHEMA_REPAIR_20261016/claim.json`
- Original receipt: `raw/ar5_t/R75_S1G_AR5_T_LEDGER_SCHEMA_REPAIR_20261016/receipt.json`
- Original stdout: `raw/ar5_t/R75_S1G_AR5_T_LEDGER_SCHEMA_REPAIR_20261016/stdout.log`
- Original stderr: `raw/ar5_t/R75_S1G_AR5_T_LEDGER_SCHEMA_REPAIR_20261016/stderr.log`
- Derived report: `ROUND75_S1G_AR5_REPORT.md`
- Derived verdict: `ROUND75_S1G_AR5_VERDICT.json`
- Derived manifest: `ROUND75_S1G_AR5_MANIFEST.json`

These original evidence files stay outside Git.
