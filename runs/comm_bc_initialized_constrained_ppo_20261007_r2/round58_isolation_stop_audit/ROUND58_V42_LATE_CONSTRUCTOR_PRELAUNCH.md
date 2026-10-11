# ROUND58 V42 late-rule constructor-only repair

V41 completed 992/1024 fixed logical jobs (384 training, 576 model evaluations,
32 immediate-rule evaluations) and stopped **before a late claim or environment
construction**. Its exact terminal error is `LateDiscoveryGatePolicy.__init__()
got an unexpected keyword argument 'cut'` at the archived R58 evaluator's
`LateDiscoveryGatePolicy(cut=147.0)` expression. All five prior technical
failures and 1013 cumulative attempt slots remain preserved; the V41 error
added none.

The actual imported rule source is the frozen canonical
`D:\code\整合\zhenghe\runs\comm_cost_v2_rerun_20260928\v2_rules.py`, SHA256
`a7ce36de2d465cadaba90bd70b36539b5e865a73be8c4e075b9faf5a86c53a57`.
Its real constructor takes `cut_steps`, default 147.0; its inherited decision
uses `first_emission and ready_step > 147.0`. V42's private import bridge
accepts only the exact frozen 147 alias, calls the original class with
`cut_steps=147.0`, and retains its `decide` method. Other threshold values,
ambiguous keywords or wrong module provenance are rejected. Neither this
canonical source nor the archived `r58_execute.py` is edited.

Two zero-environment constructor tests and one synthetic controlled-supervisor
transaction test passed. `ROUND58_V42C_ACTUAL_EXISTING_CELLS_GATE.json` passed
through real `__main__` and archived evaluation order: 608 already-complete
cells (576 model + 32 immediate) were returned via original receipts; the
actual late-rule object was constructed with `cut_steps=147.0` and inherited
unchanged `decide`; the gate stopped before the first late provider, with zero
environment, claim, optimizer or checkpoint writes. The historical V41 error
and V42 source identities are separate, append-only records.

Only the missing 32 late-rule cells may now execute under the unchanged
1024-normal/24-reserve/history16 cumulative hard cap1064 fixed plan.
