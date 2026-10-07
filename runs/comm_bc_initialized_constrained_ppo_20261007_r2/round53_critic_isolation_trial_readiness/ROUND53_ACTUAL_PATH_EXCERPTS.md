# ROUND53 actual future-path source excerpts

Generated from current source; every code segment is limited to at most 120 lines. These are actual wired functions, not pseudocode.

## R53 frozen plan and registered RNG identity

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_execute.py`:46-129

```python
def plan_from_frozen_sources() -> dict[str, Any]:
    frozen = read_json(RUN_ROOT / "ROUND46_R2_V6_FROZEN_CONFIG.json")
    key_plan = read_json(RUN_ROOT / "ROUND46_R2_V6_KEY_PLAN.json")
    training_manifest = read_json(RUN_ROOT / "ROUND46_R2_V6_TRAINING_VERSION_MANIFEST.json")
    r44_freeze = read_json(R44_DIR / "ROUND44_FROZEN_CONFIG.json")
    r42 = r44_freeze["round42"]
    objective = r42["objective"]
    frozen_checks = {
        "steps_per_episode": r42.get("steps_per_episode") == 1000,
        "observation_width": r42.get("actor_feature_width") == 19,
        "b0_bytes": objective.get("b0_bytes") == 10000.0,
        "delta": objective.get("delta") == 0.0,
        "gamma": objective.get("gamma") == 1.0,
        "gae_lambda": objective.get("gae_lambda") == 1.0,
        "eta_lambda": objective.get("eta_lambda") == 0.5,
        "lambda_init": objective.get("lambda_init") == 1.0,
        "ALL_ROWS": "ALL_ROWS" in training_manifest.get("row_filtering", ""),
        "one_shot": training_manifest.get("action_semantics") == "one-shot-send-skip/1",
        "training_episodes": training_manifest.get("training_episodes") == 384,
        "updates": training_manifest.get("updates") == 96,
    }
    if not all(frozen_checks.values()):
        raise RuntimeError(f"frozen scientific configuration mismatch: {frozen_checks}")
    if sha256_file(SOURCE_RUNNER) != frozen.get("runner_source_sha256"):
        raise RuntimeError("actual ROUND46 runner source bytes differ from its frozen identity")
    order = [int(v) for v in key_plan["training_scenario_order"]]
    r44_order = [int(v) for v in read_json(R44_DIR / "ROUND44_FROZEN_CONFIG.json")["round42"]["training_order"]]
    if order != r44_order:
        raise RuntimeError("ROUND53 frozen training sequence differs from ROUND44")
    plan = build_fixed_plan(order)
    plan["training_order"] = order
    plan["training_order_sha256"] = hashlib.sha256(json.dumps(order).encode("utf-8")).hexdigest()
    plan["source_freeze_sha256"] = sha256_file(RUN_ROOT / "ROUND46_R2_V6_FROZEN_CONFIG.json")
    plan["source_key_plan_sha256"] = sha256_file(RUN_ROOT / "ROUND46_R2_V6_KEY_PLAN.json")
    plan["authorization"] = "PENDING_USER_APPROVAL"
    input_identity = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")
    registered = input_identity.get("round44_registered_ep0_rng", {})
    frozen_registered = r44_freeze.get("round43", {}).get("registered_ep0_inputs", {})
    if set(registered) != {str(seed) for seed in SEEDS} or set(frozen_registered) != set(registered):
        raise RuntimeError("registered R44 ep0 RNG identities are not exactly the three frozen seeds")
    rng_sources = {}
    for seed in SEEDS:
        entry = registered[str(seed)]
        frozen_entry = frozen_registered[str(seed)]
        path = Path(entry["path"])
        expected_sha = entry["sha256"]
        if (Path(frozen_entry["path"]).resolve() != path.resolve()
                or frozen_entry.get("file_sha256") != expected_sha
                or path.name != f"v2_train_s{seed}_ep0.pt"
                or not path.is_file() or sha256_file(path) != expected_sha):
            raise RuntimeError(f"seed {seed}: registered R44 ep0 RNG source path/hash mismatch")
        rng_sources[str(seed)] = {"path": str(path.resolve()), "sha256": expected_sha,
                                  "rng_components": ["python", "numpy", "torch"],
                                  "load_role": "rng_only_after_same_seed_BC_POST_model_init",
                                  "model_or_optimizer_restored": False}
    plan["registered_r44_ep0_rng_only"] = rng_sources
    plan["source_content_identities"] = {
        "round46_runner_sha256": sha256_file(SOURCE_RUNNER),
        "canonical_trainer_sha256": sha256_file(CANON / "v2_trainer.py"),
        "round44_freeze_sha256": sha256_file(R44_DIR / "ROUND44_FROZEN_CONFIG.json"),
        "round44_reuse_manifest_sha256": sha256_file(R44_DIR / "ROUND44_BASELINE_REUSE.json"),
        "round46_input_identity_sha256": sha256_file(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json"),
        "round46_v10_manifest_sha256": sha256_file(RUN_ROOT / "v10_eval_final_close" / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json"),
        "round52_prototype_contract_sha256": sha256_file(RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "PROTOTYPE_CONTRACT.json"),
        "round52_gradient_evidence_sha256": sha256_file(RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "FORWARD_AND_GRADIENT_COMPARISON.json"),
        "r53_executor_sha256": sha256_file(HERE / "r53_execute.py"),
        "r53_contracts_sha256": sha256_file(HERE / "r53_contracts.py"),
        "r53_policy_adapter_sha256": sha256_file(HERE / "round53_trial_adapter.py"),
    }
    plan["scientific_config_identity"] = {
        "objective": "min E[c] s.t. g=E[Yref]-E[Ypi]-delta<=0",
        "actor_advantage": "lambda_before*A_task-A_positive_cost",
        "frozen_round44_objective": objective,
        "frozen_checks": frozen_checks,
        "search_and_allocation": "reuse exact ROUND44 frozen runtime; no edits",
        "ack": "disabled per frozen immediate/training protocol",
        "timing": "policy_owned_timing",
        "actions": "one-shot SEND/SKIP; original masks and greedy tie rule",
        "changed_training_factor_only": "critic representation stop-gradient at task/cost critic-head inputs",
        "fresh_rng_scheme": "restore only the exact registered same-seed ROUND44 ep0 Python/NumPy/Torch RNG payload after BC_POST model-only initialization via frozen R42 train_seed; do not restore its model or optimizer",
    }
    return plan


```

## R53 live runtime binding before training-route check

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_execute.py`:453-508

```python
def bind_live_runtime(base: Any, runtime: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Explicitly bind all output-owning live modules and actual trainer call path."""
    raw = Path(base.V6_DIR).resolve()
    required = ("common", "r40_common", "r42_run", "r40_train", "trainer",
                "r42_versions", "controller_module", "runner")
    missing = [key for key in required if key not in runtime]
    if missing: raise RuntimeError(f"runtime missing required live objects: {missing}")
    common, r40 = runtime["common"], runtime["r40_common"]
    output_fields = {
        common: {"OUTPUT_DIR": raw, "LOGS": raw / "logs", "ROWS": raw / "rows",
                 "REJECTED": raw / "rejected", "CHECKPOINTS": raw / "checkpoints",
                 "NORMAL_LAUNCHES": NORMAL_CAP, "RESERVE_LAUNCHES": RESERVE_CAP,
                 "LAUNCH_CAP": HARD_CAP},
        r40: {"ROUND_DIR": raw, "NORMAL_CAP": NORMAL_CAP,
              "RESERVE_CAP": RESERVE_CAP, "HARD_CAP": HARD_CAP},
    }
    for obj, fields in output_fields.items():
        for name, value in fields.items(): setattr(obj, name, value)
    r40.plan = lambda: {"normal_total": NORMAL_CAP,
                        "technical_reserve_max": RESERVE_CAP, "hard_cap": HARD_CAP,
                        "training_keys": 384, "evaluation_keys": 144,
                        "task_id": TASK_ID}
    run = runtime["r42_run"]
    run.RUN_DIR = raw
    # These are the modules actually imported by R45/R44/R42; bind each named
    # owner explicitly, including r44_runtime (the prior leak point).
    r45_module = getattr(base, "_R45_MODULE", None)
    r44_runtime = getattr(r45_module, "R44_RUNTIME_MODULE", None) if r45_module else None
    analysis_module = runtime.get("r42_analysis") or sys.modules.get("r42_analysis")
    output_modules = {
        "r42_run": run,
        "r44_runtime": r44_runtime,
        "r44_prepare": sys.modules.get("r44_prepare"),
        "r42_analysis": analysis_module,
    }
    for label, module in output_modules.items():
        if module is None or not hasattr(module, "RUN_DIR"):
            raise RuntimeError(f"live output-owning module lacks explicit RUN_DIR binding: {label}")
        module.RUN_DIR = raw
    # R42 obtains cache rows through this function at train_seed call time.
    immediate = {int(row["scene"]): read_json(Path(row["path"]))
                 for row in verify_baseline_reuse()["training_immediate_cache"]["rows_by_scene"]}
    run.reference_cache = lambda: immediate
    r40_train = runtime["r40_train"]
    r40_train.common = common
    r40_train.runner = runtime["runner"]
    r40_train.trainer = runtime["trainer"]
    original_r40_train_seed = r40_train.train_seed
    def verified_r40_train_seed(*args, **kwargs):
        if (r40_train.trainer is not runtime["trainer"]
                or r40_train.common is not common
                or r40_train.runner is not runtime["runner"]):
            raise RuntimeError("actual r42_run -> r40_train binding does not use the verified live objects")
        return original_r40_train_seed(*args, **kwargs)
    r40_train.train_seed = verified_r40_train_seed

```

## R53 exact RNG route and upstream function identity guards

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_execute.py`:509-541

```python
    # Preserve the frozen R46/R45/R42 fresh-start route.  R45 injects the
    # same-seed BC_POST model-only state; the actual R42 train_seed then loads
    # only the registered R44 ep0 RNG and restores it after fresh_controller.
    # Same-R53 continuation is handled by the unchanged R40 resume adapter.
    frozen_rng = runtime["frozen"]["round43"]["registered_ep0_inputs"]
    expected_rng = plan["registered_r44_ep0_rng_only"]
    for seed in SEEDS:
        actual = frozen_rng.get(str(seed), {})
        expected = expected_rng[str(seed)]
        if (Path(actual.get("path", "")).resolve() != Path(expected["path"]).resolve()
                or actual.get("file_sha256") != expected["sha256"]
                or sha256_file(Path(expected["path"])) != expected["sha256"]):
            raise RuntimeError(f"runtime frozen R44 ep0 RNG binding mismatch for seed {seed}")
    legacy_train = runtime["round46_r2_legacy_call_training_seed"]
    train_seed = run.train_seed
    if (not callable(legacy_train) or legacy_train.__name__ != "_call_training_seed"
            or Path(legacy_train.__code__.co_filename).resolve()
               != (R45_DIR / "r45_r2_execute_r3.py").resolve()
            or not callable(train_seed) or train_seed.__name__ != "train_seed"
            or Path(train_seed.__code__.co_filename).resolve()
               != (R43_DIR / "r42_run.py").resolve()):
        raise RuntimeError("frozen R46->R45 BC initializer->R42 registered-RNG training route is not live")
    r46_training_call = runtime["module"]._call_training_seed
    if (not callable(r46_training_call)
            or Path(r46_training_call.__code__.co_filename).resolve() != SOURCE_RUNNER.resolve()):
        raise RuntimeError("ROUND46 factory-scope wrapper is not installed around the R45 training call")
    rng_route_identity = {"R46_training_wrapper": str(Path(r46_training_call.__code__.co_filename).resolve()),
                          "R45_model_only_initializer": str(Path(legacy_train.__code__.co_filename).resolve()),
                          "R42_train_seed": str(Path(train_seed.__code__.co_filename).resolve()),
                          "registered_sources": expected_rng,
                          "restore_order": "BC_POST fresh_controller -> exact registered R44 ep0 RNG restore -> R40 first episode",
                          "restores_old_model_optimizer": False}

```

## R53 planned claims and reserve precondition

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_execute.py`:542-581

```python
    # A pre-claim guard validates every logical key before the actual durable claim.
    jobs = {item["logical_job"]: item for item in plan["jobs"]}
    original_claim = run._claim
    def planned_claim(logical_job, *args, **kwargs):
        if logical_job not in jobs:
            raise RuntimeError(f"unplanned ROUND53 claim refused: {logical_job}")
        attempt = int(kwargs.get("attempt", args[0] if args else 1))
        job = jobs[logical_job]
        slug = runtime["r40_common"]._slug(str(logical_job))
        attempt_one = raw / "launch_claims" / slug / "attempt_0001.json"
        if attempt == 1:
            if attempt_one.exists(): raise RuntimeError(f"duplicate normal claim refused: {logical_job}")
        elif attempt == 2:
            original = read_json(attempt_one) if attempt_one.is_file() else {}
            if (original.get("logical_job") != logical_job
                    or int(original.get("attempt", -1)) != 1
                    or original.get("category") != "normal"
                    or int(original.get("launch_slot", -1)) != int(job["normal_slot"])
                    or original.get("status") != "failed_technical"):
                raise RuntimeError("technical reserve retry requires an actual failed_technical attempt 1")
        else:
            raise RuntimeError(f"attempt number outside frozen one-retry contract: {attempt}")
        claim_path = original_claim(logical_job, *args, **kwargs)
        claim = read_json(claim_path)
        expected_category = "normal" if attempt == 1 else "reserve"
        slot = int(claim.get("launch_slot", 0))
        expected_slot = int(job["normal_slot"]) if attempt == 1 else slot
        slot_path = raw / "launch_slots" / expected_category / f"{slot:04d}.json"
        slot_record = read_json(slot_path) if slot_path.is_file() else {}
        if (claim.get("logical_job") != logical_job or int(claim.get("attempt", -1)) != attempt
                or claim.get("category") != expected_category
                or (attempt == 1 and slot != expected_slot)
                or slot_record.get("category") != expected_category
                or int(slot_record.get("slot", 0)) != slot
                or slot_record.get("logical_job") != logical_job
                or int(slot_record.get("attempt", 0)) != attempt):
            raise RuntimeError("persisted claim/slot identity differs from the frozen ROUND53 plan")
        return claim_path
    run._claim = planned_claim

```

## R53 strict closure and immutable trainer route

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_execute.py`:596-649

```python
    # Route the exact physical trainer object; the r42_run -> r40_train adapter
    # assigns this same object immediately before entering train_seed.
    undo_update = install_on_verified_trainer(
        runtime["trainer"], source_sha256=sha256_file(CANON / "v2_trainer.py"),
        source_path=CANON / "v2_trainer.py")
    runtime["r40_train"].trainer = runtime["trainer"]
    validate_live_binding(runtime, runtime["trainer"])

    # New freeze identity and semantic version make old PPO payloads incompatible.
    freeze = runtime["round42_freeze"]
    freeze["training_version"] = TRAINING_VERSION
    freeze["critic_gradient_route_version"] = "critic_representation_stop_gradient/1"
    versions = runtime["r42_versions"]
    versions.TRAINING_VERSION = TRAINING_VERSION
    base.TRAINING_VERSION = TRAINING_VERSION
    base.ROUND46_RUNNER_VERSION = R53_RUNNER_VERSION
    runtime["trainer"].TRAINING_VERSION = TRAINING_VERSION
    run.TRAINING_VERSION = TRAINING_VERSION
    runtime["runner"].TRAINING_VERSION = TRAINING_VERSION
    runtime["round42_freeze"] = freeze
    # The R40 adapter scans only its private checkpoint root. It must be empty.
    checkpoint_dir = Path(common.CHECKPOINTS)
    stale = list(checkpoint_dir.glob("*.pt"))
    if stale: raise RuntimeError(f"private fresh-training checkpoint root is not empty: {stale}")
    bindings = {label: str(Path(module.RUN_DIR).resolve())
                for label, module in output_modules.items()}
    expected = str(raw)
    if any(value != expected for value in bindings.values()):
        raise RuntimeError(f"not every live output module is bound to R53: {bindings}")
    save = runtime["trainer"].save_checkpoint
    save_cells = dict(zip(save.__code__.co_freevars, save.__closure__ or (), strict=True))
    journal_cell = save_cells.get("checkpoint_journal")
    if (journal_cell is None or Path(journal_cell.cell_contents).resolve()
            != Path(base.CHECKPOINT_JOURNAL_PATH).resolve()):
        raise RuntimeError("actual save_checkpoint closure is not routed to the R53 journal")
    bc_module = getattr(base, "_R45_MODULE", None)
    bc_entry = getattr(bc_module, "_bc_entry_for_seed", None)
    if not callable(bc_entry): raise RuntimeError("live R45 BC_POST model-only identity accessor is missing")
    bc_roles = {}
    expected_bc = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["bc_models_by_seed"]
    for seed in SEEDS:
        role = bc_entry(seed); expected_role = expected_bc[str(seed)]
        if (Path(role.get("checkpoint_path", "")).resolve() != Path(expected_role["checkpoint_path"]).resolve()
                or role.get("checkpoint_sha256") != expected_role["checkpoint_sha256"]
                or role.get("parameter_sha256") != expected_role["parameter_sha256"]):
            raise RuntimeError(f"live R45 BC_POST role identity mismatch for seed {seed}")
        bc_roles[str(seed)] = {"path": role["checkpoint_path"],
                               "checkpoint_sha256": role["checkpoint_sha256"],
                               "parameter_sha256": role["parameter_sha256"]}
    for path in (common.OUTPUT_DIR, common.LOGS, common.ROWS, common.REJECTED, common.CHECKPOINTS,
                 r40.ROUND_DIR, run.RUN_DIR, base.CHECKPOINT_JOURNAL_PATH,
                 raw / "launch_claims", raw / "launch_slots"):
        if Path(path).resolve() != raw:
            require_private_output_root(raw, [Path(path)])
```

## R53 authorized execution, train/evaluation order, final ledger

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_execute.py`:672-759

```python
def execute_authorized(auth_path: Path = AUTH_PATH) -> dict[str, Any]:
    """Actual execution entry. No auth receipt => refuse before runtime setup."""
    plan = plan_from_frozen_sources()
    plan_bytes_sha = canonical_sha(plan)
    receipt = read_json(auth_path)  # Missing file is a hard pre-runtime stop.
    authorize_receipt(receipt, plan_sha256=plan_bytes_sha)
    baseline = verify_baseline_reuse()
    # Verify code/source identity, independently of documentation commit HEAD.
    expected_trainer = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
    if sha256_file(CANON / "v2_trainer.py") != expected_trainer:
        raise RuntimeError("canonical trainer byte identity changed")
    RAW.mkdir(parents=True, exist_ok=False)
    plan_path = RAW / "ROUND53_FIXED_PLAN.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base = load_base_runner()
    bind_runner_paths(base, RAW)
    install_private_configure_seams(base)
    # Configure the exact frozen runtime, but never call the monolithic R46 execute.
    base._R45_MODULE = None; base._RUNTIME = None
    base._configure_runtime_module()
    runtime = base._RUNTIME
    if runtime is None: raise RuntimeError("R53 private runtime did not configure")
    live = bind_live_runtime(base, runtime, plan)
    module = runtime["module"]
    first4 = base._run_first_four_and_all_gates(module, runtime)
    trained = base._continue_fixed_training(module, runtime, first4)
    rows = []
    run = runtime["r42_run"]
    for seed in SEEDS:
        controller, identity = run.load_eval_model("NEW", seed)
        for mode in ("greedy", "sampled"):
            for scene in SCENES:
                rows.append(run.run_model_eval(arm="NEW", training_seed=seed, mode=mode,
                                               scenario=scene, controller=controller,
                                               identity=identity))
    # BC_POST model-only fresh factory; no optimizer/RNG state is restored.
    bc_entries = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["bc_models_by_seed"]
    import torch
    for seed in SEEDS:
        controller = runtime["controller_module"].fresh_controller(seed)
        entry = bc_entries[str(seed)]
        if controller.parameter_summary().get("parameter_sha256") != entry["parameter_sha256"]:
            raise RuntimeError(f"BC_POST model-only initialization mismatch for seed {seed}")
        identity = {"arm": "BC", "training_seed": seed, "episodes_done": 0,
                    "load_mode": "bc_model_only_four_modules", "optimizer_restored": False,
                    "rng_restored": False, "file_sha256": entry["checkpoint_sha256"],
                    "parameter_identity": {"parameter_sha256": entry["parameter_sha256"]}}
        for scene in SCENES:
            rows.append(run.run_model_eval(arm="BC", training_seed=seed, mode="sampled",
                                           scenario=scene, controller=controller,
                                           identity=identity))
    if len(rows) != 144: raise RuntimeError(f"ROUND53 eval closure {len(rows)}/144")
    claim_paths = sorted((RAW / "launch_claims").glob("*/*.json"))
    claims = [read_json(path) for path in claim_paths]
    for path, claim in zip(claim_paths, claims, strict=True):
        expected_slug = runtime["r40_common"]._slug(str(claim.get("logical_job", "")))
        expected_name = f"attempt_{int(claim.get('attempt', 0)):04d}.json"
        if path.parent.name != expected_slug or path.name != expected_name:
            raise RuntimeError(f"claim file path/core identity mismatch: {path}")
    core_closure = validate_claims(plan, claims, strict=True)
    normal_slots = list((RAW / "launch_slots" / "normal").glob("*.json"))
    reserve_slots = list((RAW / "launch_slots" / "reserve").glob("*.json"))
    normal_records = [read_json(p) for p in normal_slots]
    reserve_records = [read_json(p) for p in reserve_slots]
    for category, paths, records in (("normal", normal_slots, normal_records),
                                     ("reserve", reserve_slots, reserve_records)):
        for path, record in zip(paths, records, strict=True):
            if path.name != f"{int(record.get('slot', 0)):04d}.json":
                raise RuntimeError(f"slot file path/core identity mismatch: {path}")
    slot_closure = validate_slot_records(
        plan, claims, normal_records, reserve_records, strict=True)
    ledger = run.write_execution_ledger(strict=True)
    if ledger.get("status") == "deferred":
        raise RuntimeError("final derived ledger did not close; authoritative claims are preserved")
    result = {"kind": "round53-execution-result/1", "status": "complete",
              "task_id": TASK_ID, "runner_version": R53_RUNNER_VERSION,
              "training_version": TRAINING_VERSION, "training": trained,
              "first_four_and_gates": first4, "evaluation_rows": len(rows),
              "ledger": ledger, "claim_closure": core_closure,
              "slot_closure": slot_closure,
              "normal_slots": len(normal_slots), "reserve_slots": len(reserve_slots),
              "baseline_reuse": baseline,
              "live_bindings": {k: v for k, v in live.items() if k != "route_undo"}}
    (RAW / "ROUND53_EXECUTION_RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


```

## R42 post-controller RNG restore helper

`D:\code\整合\zhenghe-round42\runs\comm_one_shot_full_trace_ablation_20261004\r42_run.py`:775-797

```python
def _registered_ep0_rng_after_init(runtime: dict[str, Any], training_seed: int,
                                   rng_state: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Apply the registered training RNG after the same-seed fresh model is constructed."""
    module = runtime["controller_module"]
    trainer = runtime["trainer"]
    original = module.fresh_controller
    status: dict[str, Any] = {"done": False}

    def fresh_controller_with_registered_rng(seed: int) -> Any:
        controller = original(seed)
        if not status["done"] and int(seed) == int(training_seed):
            status["fresh_controller_rng"] = trainer.rng_state()
            trainer.restore_rng(rng_state)
            status["done"] = True
        return controller

    module.fresh_controller = fresh_controller_with_registered_rng
    try:
        yield status
    finally:
        module.fresh_controller = original


```

## R42 actual train_seed ep0 payload verification and restore

`D:\code\整合\zhenghe-round42\runs\comm_one_shot_full_trace_ablation_20261004\r42_run.py`:798-881

```python
def train_seed(training_seed: int, *, episodes: int, frozen: dict[str, Any]) -> dict[str, Any]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    r40_train = runtime["r40_train"]
    checkpoints = runtime["common"].CHECKPOINTS
    rolling = checkpoints / f"v2_train_s{int(training_seed)}.pt"
    prior_updates = list(checkpoints.glob(f"v2_train_s{int(training_seed)}_after_b*.pt"))
    fresh_start = not rolling.exists() and not prior_updates
    start_payload: dict[str, Any] | None = None
    if fresh_start:
        registered = frozen["round43"]["registered_ep0_inputs"][str(int(training_seed))]
        path = Path(registered["path"])
        expected_path = ROUND42_OUTPUT / "checkpoints" / f"v2_train_s{int(training_seed)}_ep0.pt"
        if path.resolve() != expected_path.resolve() or not path.is_file():
            raise RuntimeError(f"seed {training_seed}: registered ROUND42 ep0 RNG input path changed")
        if sha256(path) != registered["file_sha256"]:
            raise RuntimeError(f"seed {training_seed}: registered ROUND42 ep0 RNG input hash changed")
        payload = torch_load_cpu(path)
        state = payload.get("training_state", {})
        if (int(state.get("episodes_done", -1)) != 0
                or int(state.get("updates", -1)) != 0
                or int(state.get("optimizer_steps", -1)) != 0
                or int(state.get("training_seed", -1)) != int(training_seed)
                or state.get("training_version") != frozen["round42"]["source_training_version"]
                or payload.get("training_version") != frozen["round42"]["source_training_version"]
                or float(state.get("gae_lambda", float("nan"))) != 0.95
                or float(state.get("gamma", float("nan"))) != float(
                    frozen["round43"]["gamma"])
                or float(state.get("lambda", float("nan"))) != 1.0
                or payload.get("comm_action_semantics") != "one-shot-send-skip/1"
                or state.get("comm_action_semantics") != "one-shot-send-skip/1"):
            raise RuntimeError(f"seed {training_seed}: registered RNG input is not ROUND42 ep0")
        # The model and optimizer are freshly built by the frozen ROUND40 adapter. Restore the
        # registered ep0 RNG immediately after fresh_controller(seed), so initialization cannot
        # overwrite the registered rollout/update RNG before the first episode.
        start_payload = payload
        runtime.setdefault("rng_priming", {})[str(int(training_seed))] = {
            "source": str(path), "source_sha256": registered["file_sha256"],
            "loaded_model": False, "loaded_optimizer": False,
            "episodes_done": 0, "updates": 0, "optimizer_steps": 0,
        }
    cache_rows = reference_cache()
    cache = {scene: row for scene, row in cache_rows.items()}
    scenario_order = runtime["common"].scenario_order()
    missing = sorted(set(scenario_order[:episodes]) - set(cache))
    if missing:
        raise RuntimeError(f"ROUND42 immediate reference cache is incomplete: {missing}")
    r40_train.common = runtime["common"]
    r40_train.runner = runtime["runner"]
    r40_train.trainer = runtime["trainer"]
    if fresh_start:
        assert start_payload is not None
        prime_context = _registered_ep0_rng_after_init(
            runtime, int(training_seed), start_payload["rng"])
    else:
        prime_context = contextlib.nullcontext({"done": True})
    try:
        with prime_context as priming:
            report = r40_train.train_seed(
                int(training_seed), episodes=int(episodes),
                freeze=runtime["round42_freeze"], reference_cache=cache)
    except BaseException:
        # The durable PPO/Adam call journals can be ahead of the last environment-ledger
        # flush when training fails between batches. Persist their reconciled state without
        # masking the original training exception.
        try:
            write_execution_ledger(strict=False)
        except BaseException:
            pass
        raise
    if fresh_start and not priming["done"]:
        raise RuntimeError(f"seed {training_seed}: registered ep0 RNG was not applied after initialization")
    if fresh_start:
        runtime["rng_priming"][str(int(training_seed))]["applied_after_fresh_controller"] = True
    report["comm_action_semantics"] = "one-shot-send-skip/1"
    report["training_version"] = frozen["round42"]["training_version"]
    report["rng_priming"] = dict(runtime.get("rng_priming", {}).get(
        str(int(training_seed)), {"source": "same_round43_process_resume",
                                  "loaded_model": False, "loaded_optimizer": False}))
    write_execution_ledger(strict=True)
    return report


```

## R42 fixed evaluation model loading

`D:\code\整合\zhenghe-round42\runs\comm_one_shot_full_trace_ablation_20261004\r42_run.py`:882-935

```python
def load_eval_model(arm: str, training_seed: int) -> tuple[Any, dict[str, Any]]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    trainer = runtime["trainer"]
    if int(training_seed) not in SEEDS:
        raise ValueError(f"unregistered training seed {training_seed}")
    if arm == "OLD":
        source = _json_read(ROUND42_OUTPUT / "ROUND42_CHECKPOINT_INDEX.json")[
            "new_checkpoints"][str(training_seed)]["128"]
        path = Path(source["path"])
        if sha256(path) != source["file_sha256_after"]:
            raise RuntimeError(f"ROUND42 OLD read-only checkpoint hash changed: {path}")
        controller, state, identity = trainer.load_checkpoint_read_only(path)
        expected = source["parameter_identity"]["parameter_sha256"]
        if controller.parameter_summary()["parameter_sha256"] != expected:
            raise RuntimeError(f"ROUND42 OLD parameter identity changed: seed {training_seed}")
        if (state.get("training_version") != runtime["frozen"]["round42"][
                "source_training_version"]
                or int(state.get("episodes_done", -1)) != 128):
            raise RuntimeError(f"ROUND42 OLD checkpoint is not NEW ep128: seed {training_seed}")
    elif arm == "NEW":
        path = runtime["common"].CHECKPOINTS / f"v2_train_s{training_seed}_ep128.pt"
        if not path.is_file():
            raise FileNotFoundError(f"ROUND42 NEW ep128 checkpoint missing: {path}")
        payload = torch_load_cpu(path)
        if (payload.get("training_version") != runtime["r42_versions"].TRAINING_VERSION
                or payload.get("comm_action_semantics") != "one-shot-send-skip/1"):
            raise RuntimeError(f"ROUND42 NEW checkpoint identity mismatch: {path}")
        controller, state, identity = trainer.load_checkpoint_read_only(
            path, freeze=runtime["round42_freeze"])
        if (int(state.get("episodes_done", -1)) != 128
                or float(state.get("gae_lambda", -1)) != 1.0
                or float(state.get("gamma", -1)) != 1.0):
            raise RuntimeError(f"ROUND42 NEW model is not ep128: {path}")
    else:
        raise ValueError(f"unknown ROUND42 model arm {arm!r}")
    identity.update({
        "arm": arm,
        "path": str(path),
        "file_sha256": sha256(path),
        "training_seed": int(training_seed),
        "episodes_done": int(state.get("episodes_done", -1)),
        "updates": int(state.get("updates", -1)),
        "optimizer_steps": int(state.get("optimizer_steps", -1)),
        "lambda": float(state.get("lambda", float("nan"))),
        "parameter_identity": controller.parameter_summary(),
        "evaluation_semantics": "one-shot-send-skip/1",
    })
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError(f"{arm} seed {training_seed}: read-only evaluation restored state")
    return controller, identity


```

## R42 evaluation cell path

`D:\code\整合\zhenghe-round42\runs\comm_one_shot_full_trace_ablation_20261004\r42_run.py`:941-973

```python
def run_model_eval(*, arm: str, training_seed: int, mode: str, scenario: int,
                   controller: Any, identity: dict[str, Any]) -> dict[str, Any]:
    from r42_one_shot import OneShotSendSkipCollector

    if mode not in ("greedy", "sampled"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    runtime = _RUNTIME
    runner = runtime["runner"]
    collector = OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
    logical = (f"model_eval_{arm.lower()}_s{training_seed}_{mode}_scene{int(scenario)}")
    label = f"eval_{arm.lower()}_s{training_seed}_{mode}_s{int(scenario)}_r43"
    action_seed = runner.eval_action_seed(training_seed, scenario, mode)
    return _execute_cell(
        label, logical, "eval", int(scenario),
        {"arm": arm, "training_seed": int(training_seed), "mode": mode,
         "scenario_seed": int(scenario)},
        collector=collector,
        runner_kwargs={
            "policy": "learned", "attach": collector.policy(), "controller": controller,
            "collector": collector, "action_rng_seed": action_seed,
            "tag": f"r43_eval_{arm.lower()}_s{training_seed}_{mode}",
            "checkpoint_load_identity": identity,
        },
        row_updates={
            "model_arm": arm, "training_seed": int(training_seed),
            "eval_mode": mode, "eval_block": f"one_shot_{mode}",
            "action_rng_seed": action_seed,
            "checkpoint_parameter_identity": identity["parameter_identity"],
        },
        identity=identity,
    )


```

## R45 BC model-only injection before actual R42 train_seed

`D:\code\整合\zhenghe-round42\runs\comm_rule_warmstart_cost_learning_20261006_r2\r45_r2_execute_r3.py`:1485-1546

```python
def _call_training_seed(runtime: dict[str, Any], seed: int, *, episodes: int) -> dict[str, Any]:
    """Call the R43 entry and R40 trainer with BC model-only initialization."""
    import r45_bc
    model = _bc_entry_for_seed(seed)
    module = runtime["controller_module"]
    base_fresh_controller = module.fresh_controller
    injection: dict[str, Any] = {"seed": int(seed), "applied": False}

    def fresh_bc_controller(model_seed: int):
        controller = base_fresh_controller(model_seed)
        if int(model_seed) != int(seed):
            return controller
        if injection["applied"]:
            raise RuntimeError(f"seed {seed}: fresh BC controller injection was invoked twice")
        if _sha256(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]:
            raise RuntimeError(f"seed {seed}: BC checkpoint changed before fresh controller load")
        payload = r45_bc.read_bc_checkpoint(Path(model["checkpoint_path"]))
        before_rng = runtime["trainer"].rng_state()
        r45_bc.load_bc_controller_state(controller, payload)
        after_rng = runtime["trainer"].rng_state()
        parameter = controller.parameter_summary()["parameter_sha256"]
        modules = _module_parameter_digests(controller, r45_bc)
        if (parameter != model["parameter_sha256"]
                or modules != model["module_parameter_sha256"]
                or not _same_value(before_rng, after_rng)
                or int(payload.get("seed", -1)) != int(seed)
                or payload.get("post_bc_parameter_sha256") != model["parameter_sha256"]):
            raise RuntimeError(f"seed {seed}: model-only BC initialization identity failed")
        injection.update({
            "applied": True,
            "bc_checkpoint_path": model["checkpoint_path"],
            "bc_checkpoint_sha256": model["checkpoint_sha256"],
            "bc_parameter_sha256": parameter,
            "module_parameter_sha256": modules,
            "bc_load_changed_rng": False,
            "bc_optimizer_loaded": False,
            "registered_ep0_rollout_rng_applied_by_r42_entry": False,
        })
        return controller

    module.fresh_controller = fresh_bc_controller
    try:
        report = runtime["r42_run"].train_seed(
            int(seed), episodes=int(episodes), frozen=runtime["frozen"])
    finally:
        module.fresh_controller = base_fresh_controller
    if injection["applied"]:
        priming = report.get("rng_priming", {})
        if (priming.get("applied_after_fresh_controller") is not True
                or priming.get("loaded_model") is not False
                or priming.get("loaded_optimizer") is not False):
            raise RuntimeError(f"seed {seed}: R42 did not apply only its registered ep0 RNG")
        injection["registered_ep0_rollout_rng_applied_by_r42_entry"] = True
    elif not (Path(runtime["common"].CHECKPOINTS)
              / f"v2_train_s{seed}.pt").is_file():
        raise RuntimeError(f"seed {seed}: resumed train call has neither BC init nor R2 rolling state")
    report["round45_bc_initialization"] = injection
    report["training_version"] = runtime["r42_versions"].TRAINING_VERSION
    report["runner_version"] = RUNNER_VERSION
    return report


```

## R46 first-four/cross-seed gates and fixed continuation

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\r46_execute_r2.py`:3239-3335

```python
def _run_first_four_and_all_gates(module: Any, runtime: dict[str, Any]) -> dict[int, dict[str, Any]]:
    first_four: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        module._task_state(
            f"ROUND46 seed {seed}: fixed first four training episodes",
            active_training_seed=seed, phase="first_four",
        )
        report = module._call_training_seed(runtime, seed, episodes=4)
        _check_first_four_report(seed, report)
        first_four_path = R46_DIR / f"ROUND46_FIRST_FOUR_s{seed}.json"
        if first_four_path.exists():
            saved = _json(first_four_path)
            if (saved.get("kind") != "round46-first-four-report/1"
                    or int(saved.get("seed", -1)) != seed
                    or saved.get("training_version") != TRAINING_VERSION):
                raise RuntimeError(f"ROUND46 first-four receipt identity changed for seed {seed}")
            first_four[seed] = saved["report"]
        else:
            first_four[seed] = report
            _write_json(first_four_path, {
                "kind": "round46-first-four-report/1", "seed": seed,
                "runner_version": ROUND46_RUNNER_VERSION,
                "training_version": TRAINING_VERSION, "report": report,
            })

    gates = {}
    for seed in SEEDS:
        module._task_state(
            f"ROUND46 seed {seed}: actual R43/R44 first-batch correctness gate",
            active_training_seed=seed, phase="first_batch_gate",
        )
        gate = module._run_first_batch_gate(runtime, seed)
        if gate.get("status") != "passed":
            raise RuntimeError(f"ROUND46 seed {seed} actual batch0 gate failed")
        gate_round46 = dict(gate)
        gate_round46.update({
            "kind": "round46-actual-first-batch-gate/1",
            "round": 46, "runner_version": ROUND46_RUNNER_VERSION,
            "training_version": TRAINING_VERSION,
            "actual_gate_implementation": "R43 first_batch_gate via R44/R42 runtime",
        })
        _write_json(R46_DIR / f"ROUND46_FIRST_BATCH_GATE_s{seed}.json", gate_round46)
        gates[seed] = gate_round46
    if set(gates) != set(SEEDS) or any(item["status"] != "passed" for item in gates.values()):
        raise RuntimeError("ROUND46 cross-seed gate did not pass for all three seeds")
    module._task_state(
        "ROUND46 all three first-batch gates passed; fixed continuation authorized",
        all_seed_first_batch_gates="passed", gates_completed=3,
    )
    return first_four


def _continue_fixed_training(module: Any, runtime: dict[str, Any],
                             first_four: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        module._task_state(
            f"ROUND46 seed {seed}: fixed continuation to ep128",
            active_training_seed=seed, phase="continuation",
        )
        full = module._call_training_seed(runtime, seed, episodes=128)
        if (int(full.get("episodes_done", -1)) != 128
                or int(full.get("updates", -1)) != 32
                or int(full.get("optimizer_steps", -1)) <= 0
                or full.get("training_version") != TRAINING_VERSION):
            raise RuntimeError(f"ROUND46 seed {seed} fixed ep128 training failed closure")
        checkpoint = V6_CHECKPOINTS_DIR / f"v2_train_s{seed}_ep128.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        summary = {
            "kind": "round46-seed-training-result/1", "status": "complete",
            "seed": seed, "training_version": TRAINING_VERSION,
            "runner_version": ROUND46_RUNNER_VERSION,
            "first_four_report": first_four[seed],
            "fixed_ep128_report": full,
            "ep128_checkpoint_path": str(checkpoint.resolve()),
            "ep128_checkpoint_sha256": _sha256(checkpoint),
            "bc_checkpoint": module._bc_entry_for_seed(seed),
        }
        summary_path = R46_DIR / f"ROUND46_TRAINING_s{seed}.json"
        if summary_path.exists():
            saved = _json(summary_path)
            if (saved.get("status") != "complete"
                    or saved.get("ep128_checkpoint_sha256") != summary["ep128_checkpoint_sha256"]):
                raise RuntimeError(f"ROUND46 seed {seed} ep128 summary conflicts with checkpoint")
            summary = saved
        else:
            _write_json(summary_path, summary)
        summaries[seed] = summary
        module._task_state(
            f"ROUND46 seed {seed}: ep128 complete",
            seed_episodes_done=128, seed_ppo_updates=32,
            completed_training_seeds=sorted(summaries),
        )
    return summaries


```

## R40 checkpoint save milestone

`D:\code\整合\zhenghe-round40\runs\comm_first_report_training_ablation_20261003\r40_train.py`:115-149

```python
def _save_milestone(controller: Any, optimizer: torch.optim.Optimizer,
                    state: dict[str, Any], training_seed: int, episodes: int,
                    freeze: dict[str, Any]) -> Path:
    path = _milestone_path(training_seed, episodes, freeze)
    if path.exists():
        restored, restored_state, identity = trainer.load_checkpoint_read_only(path,
                                                                                freeze=freeze)
        expected = controller.parameter_summary()["parameter_sha256"]
        actual = restored.parameter_summary()["parameter_sha256"]
        _assert_same = (
            int(restored_state.get("training_seed", -1)) == int(training_seed)
            and int(restored_state.get("episodes_done", -1)) == int(episodes)
            and int(restored_state.get("updates", -1)) == int(state["updates"])
            and int(restored_state.get("optimizer_steps", -1)) == int(state["optimizer_steps"])
            and float(restored_state.get("lambda", float("nan"))) == float(state["lambda"])
            and actual == expected
            and identity.get("read_only_for_round40") is True
        )
        if not _assert_same:
            raise RuntimeError(f"immutable milestone differs from current R40 state: {path}")
        return path

    temporary = path.with_name(path.name + f".{os.getpid()}.milestone-tmp")
    if temporary.exists():
        raise RuntimeError(f"stale milestone temporary requires inspection: {temporary}")
    trainer.save_checkpoint(controller, optimizer, state, temporary, freeze=freeze)
    try:
        # Windows os.rename refuses to replace an existing destination, preserving milestones.
        os.rename(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return _save_milestone(controller, optimizer, state, training_seed, episodes, freeze)
    return path


```

## R40 same-version resume source selection

`D:\code\整合\zhenghe-round40\runs\comm_first_report_training_ablation_20261003\r40_train.py`:276-317

```python
def _latest_resume_source(training_seed: int, freeze: dict[str, Any]) -> Path | None:
    rolling = _rolling_path(training_seed)
    after_paths = list(common.CHECKPOINTS.glob(f"v2_train_s{training_seed}_after_b*.pt"))
    def _batch_number(path: Path) -> int:
        match = re.match(rf"v2_train_s{training_seed}_after_b(\d+)", path.stem)
        if match is None:
            raise RuntimeError(f"malformed immutable post-update checkpoint name: {path.name}")
        return int(match.group(1))
    candidates = ([rolling] if rolling.exists() else []) + sorted(after_paths,
                                                                key=_batch_number)
    valid: list[tuple[int, int, Path, str]] = []
    for priority, path in enumerate(candidates):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            if path == rolling:
                # A later immutable post-update snapshot can still establish the committed
                # cursor; this path is retained and reported as a damaged rolling cache.
                continue
            raise RuntimeError(f"immutable post-update checkpoint is unreadable: {path}") from error
        state = payload.get("training_state")
        if (payload.get("training_version") != trainer.TRAINING_VERSION
                or payload.get("comm_action_eligibility") != "first-report-only/1"
                or not isinstance(state, dict)
                or state.get("training_version") != trainer.TRAINING_VERSION
                or state.get("comm_action_eligibility") != "first-report-only/1"
                or payload.get("freeze") != freeze):
            raise RuntimeError(f"checkpoint is outside the ROUND40 resume identity: {path}")
        controller = trainer._restore_controller(payload, state)
        parameter = controller.parameter_summary()["parameter_sha256"]
        valid.append((int(state["updates"]), priority, path, parameter))
    if not valid:
        return None
    max_update = max(item[0] for item in valid)
    latest = [item for item in valid if item[0] == max_update]
    if len({item[3] for item in latest}) != 1:
        raise RuntimeError("ROUND40 checkpoints conflict at the same committed update count")
    # Prefer rolling when it agrees with the newest immutable post-update snapshot.
    latest.sort(key=lambda item: (item[2] != rolling, item[1]))
    return latest[0][2]


```

## R40 fresh BC controller/Adam/dual initialization and first rollout

`D:\code\整合\zhenghe-round40\runs\comm_first_report_training_ablation_20261003\r40_train.py`:342-403

```python
def train_seed(training_seed: int, *, episodes: int, freeze: dict[str, Any],
               reference_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if training_seed not in common.TRAIN_SEEDS:
        raise ValueError(f"unregistered training seed {training_seed}")
    if episodes not in (4, common.EPISODES_PER_SEED) and episodes % common.EPISODES_PER_UPDATE:
        raise ValueError("ROUND40 training limit must end on a 4-episode PPO batch")
    if not isinstance(reference_cache, dict):
        raise TypeError("immediate cache must be a scenario-keyed dictionary")

    checkpoint = _rolling_path(training_seed)
    milestone_paths: dict[str, str] = {}
    resume_source = _latest_resume_source(training_seed, freeze)
    if resume_source is not None:
        controller, optimizer, state = trainer.load_checkpoint(resume_source, freeze=freeze)
        if resume_source != checkpoint:
            # A completed immutable snapshot is the source of truth if a crash happened after
            # its save and before the rolling checkpoint/commit receipt.
            trainer.save_checkpoint(controller, optimizer, state, checkpoint, freeze=freeze)
        _recover_final_batch_commit(training_seed, state, controller, checkpoint)
        print(f"R40 resumed seed {training_seed} at episode {state['episodes_done']} "
              f"lambda={state['lambda']:.4f} updates={state['updates']}", flush=True)
    else:
        from v2_controller import fresh_controller, parameter_identity

        controller = fresh_controller(training_seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        state = trainer.fresh_training_state(training_seed)
        initial_milestone = _save_milestone(controller, optimizer, state, training_seed, 0, freeze)
        milestone_paths["0"] = str(initial_milestone)
        print(f"R40 seed {training_seed}: fresh Controller({training_seed}); "
              "no old weights loaded", flush=True)

    if int(state["training_seed"]) != int(training_seed):
        raise RuntimeError("rolling checkpoint seed does not match this training job")
    if state.get("comm_action_eligibility") != "first-report-only/1":
        raise RuntimeError("rolling checkpoint is missing ROUND40 action eligibility")
    order = common.scenario_order()
    needed_scenarios = set(order[:episodes])
    missing = sorted(needed_scenarios - set(reference_cache))
    if missing:
        raise RuntimeError(f"immediate reference cache is incomplete: {missing}")
    references = {int(scenario): float(row["on_time"])
                  for scenario, row in reference_cache.items()}

    before = parameter_identity(controller)
    module_before = _module_digests(controller)
    history: list[dict[str, Any]] = []
    started = time.time()

    # Record only immutable milestones whose content carries the current freeze identity.
    for milestone in (0, 64, 128):
        if milestone > int(state["episodes_done"]):
            continue
        milestone_path = _milestone_path(training_seed, milestone, freeze)
        if milestone_path.is_file():
            milestone_paths[str(milestone)] = str(milestone_path)
    # Repair only a missing ep64/ep128 immutable copy from a same-version rolling checkpoint.
    if int(state["episodes_done"]) in common.MILESTONE_EPISODES:
        milestone_path = _save_milestone(controller, optimizer, state, training_seed,
                                         int(state["episodes_done"]), freeze)
        milestone_paths[str(int(state["episodes_done"]))] = str(milestone_path)

```

## R40 PPO update and checkpoint commit path

`D:\code\整合\zhenghe-round40\runs\comm_first_report_training_ablation_20261003\r40_train.py`:448-556

```python
        update_report = trainer.ppo_update(controller, optimizer, episodes_in_batch,
                                           update_index=batch_index)
        g_record = trainer.batch_g_hat(episodes_in_batch, references)
        dual_record = trainer.dual_update(state["lambda"], g_record["g_hat"])
        state["lambda"] = dual_record["lambda_next"]
        state["lambda_history"].append({
            "batch": batch_index,
            "g_hat": dual_record["g_hat"],
            "lambda_before": dual_record["lambda_before"],
            "lambda_next": dual_record["lambda_next"],
        })
        state["updates"] += 1
        state["optimizer_steps"] += int(update_report.get("optimizer_steps", 0))
        after = parameter_identity(controller)
        if after["parameter_sha256"] == before["parameter_sha256"]:
            raise RuntimeError(f"seed {training_seed} batch {batch_index}: parameters did not move")

        batch_record = {
            "kind": "round40-training-batch-record/1",
            "batch": batch_index,
            "training_seed": int(training_seed),
            "episodes": [int(row["episode_index"]) for row in episodes_in_batch],
            "scenarios": [int(row["scenario_seed"]) for row in episodes_in_batch],
            "episode_artifacts": [
                {
                    "episode_index": int(row["episode_index"]),
                    "scenario_seed": int(row["scenario_seed"]),
                    "attempt": int(row["attempt"]),
                    "row_path": row["artefacts"]["row"],
                    "row_sha256": _sha256(Path(row["artefacts"]["row"])),
                    "tape_path": row["artefacts"]["tape"],
                    "tape_sha256": _sha256(Path(row["artefacts"]["tape"])),
                }
                for row in episodes_in_batch
            ],
            "lambda_at_rollout": lambda_at_rollout,
            "update": update_report,
            "constraint": g_record,
            "dual": dual_record,
            "controller_parameters_before": before,
            "controller_parameters_after": after,
            "preupdate_checkpoint": preupdate_checkpoint.name,
            "preupdate_checkpoint_sha256": preupdate_sha,
            "optimizer_state_sha256_before": optimizer_state_before_hash,
            "episodes_done_after": int(state["episodes_done"]),
            "updates_after": int(state["updates"]),
            "optimizer_steps_after": int(state["optimizer_steps"]),
            "postupdate_checkpoint": _postupdate_checkpoint_path(training_seed,
                                                                   batch_index, freeze).name,
            "controller_module_digests_before": module_before,
            "controller_module_digests_after": _module_digests(controller),
            "frozen_checkpoints": {
                str(path.relative_to(common.ROOT)): common.sha256_file(path)
                for path in common.CHECKPOINT_FILES
            },
            "optimizer_parameter_shapes": [
                list(parameter.shape) for group in optimizer.param_groups
                for parameter in group["params"]
            ],
            "controller_parameter_shapes": [list(parameter.shape)
                                            for parameter in controller.parameters],
            "training_version": state["training_version"],
            "comm_action_eligibility": state["comm_action_eligibility"],
            "record_status": "computed_before_checkpoint_commit",
        }
        batch_record_path = _batch_record_path(training_seed, batch_index)
        _write_json_exclusive(batch_record_path, batch_record)
        postupdate_checkpoint = _postupdate_checkpoint_path(training_seed, batch_index, freeze)
        _write_immutable_snapshot(postupdate_checkpoint, controller, optimizer, state, freeze)
        # The immutable post-update checkpoint is the recovery commit point. The rolling copy
        # and receipt follow; a prior batch record is never overwritten on recovery.
        trainer.save_checkpoint(controller, optimizer, state, checkpoint, freeze=freeze)
        _write_batch_commit(training_seed, batch_index, batch_record_path, checkpoint,
                            postupdate_checkpoint, state, after)
        if int(state["episodes_done"]) in common.MILESTONE_EPISODES:
            milestone_path = _save_milestone(controller, optimizer, state, training_seed,
                                             int(state["episodes_done"]), freeze)
            milestone_paths[str(int(state["episodes_done"]))] = str(milestone_path)
        _save_progress(training_seed, state, status="batch_checkpoint_committed",
                       parameter_identity=after)
        before = after
        module_before = _module_digests(controller)
        print(f"R40 seed {training_seed} batch {batch_index}: "
              f"samples={update_report['samples']} "
              f"optimizer_steps={update_report['optimizer_steps']} "
              f"g_hat={dual_record['g_hat']:+.4f} "
              f"lambda {dual_record['lambda_before']:.4f} -> "
              f"{dual_record['lambda_next']:.4f}", flush=True)

    final_identity = parameter_identity(controller)
    _save_progress(training_seed, state,
                   status="first_four_gate_ready" if episodes == 4 else "requested_limit_complete",
                   parameter_identity=final_identity)
    return {
        "training_seed": int(training_seed),
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "lambda": float(state["lambda"]),
        "lambda_history": state["lambda_history"],
        "seconds": round(time.time() - started, 2),
        "checkpoint": str(checkpoint),
        "controller_parameters": final_identity,
        "history": history,
        "training_version": state["training_version"],
        "comm_action_eligibility": state["comm_action_eligibility"],
    }


```

## R53 order-independent strict claims

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_contracts.py`:68-133

```python
def validate_claims(plan: dict[str, Any], claims: list[dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
    """Validate authoritative claims independent of directory iteration order.

    Reserve attempts are accepted only when the persisted attempt-1 core claim
    for the same planned key is actually technical-failed.  A copied
    ``original_status`` field is corroboration, never the authority.
    """
    planned = {job["logical_job"]: job for job in plan["jobs"]}
    by_job: dict[str, dict[int, dict[str, Any]]] = {}
    counts = {"normal": 0, "reserve": 0}
    completed: set[str] = set()
    used_slots = {"normal": set(), "reserve": set()}
    for claim in claims:
        logical = str(claim.get("logical_job", ""))
        try:
            attempt = int(claim.get("attempt", 1))
            slot = int(claim.get("launch_slot", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("claim attempt/launch_slot must be integers") from exc
        category = str(claim.get("category", ""))
        status = claim.get("status")
        if logical not in planned:
            raise ValueError(f"unplanned logical job: {logical}")
        if attempt not in (1, 2) or (category, attempt) not in (("normal", 1), ("reserve", 2)):
            raise ValueError(f"claim category/attempt mismatch: {(logical, attempt)} {category}")
        attempts = by_job.setdefault(logical, {})
        if attempt in attempts:
            raise ValueError(f"duplicate claim attempt: {(logical, attempt)}")
        if status not in ("claimed_before_environment", "complete", "failed_technical",
                          "not_started_quota_exhausted"):
            raise ValueError(f"invalid claim status: {status}")
        if category == "normal":
            if slot != int(planned[logical]["normal_slot"]):
                raise ValueError(f"normal claim slot mismatch: {logical}")
        elif not 1 <= slot <= RESERVE_CAP:
            raise ValueError("reserve slot outside the frozen technical reserve")
        if slot in used_slots[category]:
            raise ValueError(f"duplicate {category} launch slot")
        used_slots[category].add(slot)
        counts[category] += 1
        attempts[attempt] = claim
        if status == "complete":
            completed.add(logical)

    for logical, attempts in by_job.items():
        retry = attempts.get(2)
        if retry is None:
            continue
        original = attempts.get(1)
        if (original is None or original.get("category") != "normal"
                or int(original.get("attempt", -1)) != 1
                or original.get("status") != "failed_technical"
                or int(original.get("launch_slot", -1)) != int(planned[logical]["normal_slot"])):
            raise ValueError("reserve may retry only a persisted technical-failed normal core claim")
        if retry.get("original_status") not in (None, "failed_technical"):
            raise ValueError("reserve original_status contradicts the persisted core claim")

    if counts["normal"] > NORMAL_CAP or counts["reserve"] > RESERVE_CAP:
        raise ValueError("claim budget exceeds cap")
    if strict and (counts["normal"] != NORMAL_CAP or completed != set(planned)):
        raise ValueError("strict final ledger requires exact full plan closure")
    return {"counts": counts, "completed_logical_jobs": len(completed),
            "claim_count": len(claims), "strict": strict,
            "reserve_slots_required_to_be_full": False}


```

## R53 durable quota slot cross-closure

`D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_contracts.py`:134-171

```python
def validate_slot_records(plan: dict[str, Any], claims: list[dict[str, Any]],
                          normal_slots: list[dict[str, Any]],
                          reserve_slots: list[dict[str, Any]], *,
                          strict: bool = False) -> dict[str, Any]:
    """Close durable slot files against the authoritative normal/reserve claims."""
    expected: dict[str, dict[int, tuple[str, int]]] = {"normal": {}, "reserve": {}}
    for claim in claims:
        category = str(claim.get("category", ""))
        if category not in expected:
            raise ValueError(f"unknown claim slot category: {category}")
        slot = int(claim.get("launch_slot", 0))
        logical = str(claim.get("logical_job", ""))
        attempt = int(claim.get("attempt", 0))
        if slot in expected[category]:
            raise ValueError(f"duplicate claim slot: {category}/{slot}")
        expected[category][slot] = (logical, attempt)

    observed: dict[str, dict[int, tuple[str, int]]] = {"normal": {}, "reserve": {}}
    for category, records in (("normal", normal_slots), ("reserve", reserve_slots)):
        for record in records:
            slot = int(record.get("slot", 0))
            pair = (str(record.get("logical_job", "")), int(record.get("attempt", 0)))
            if record.get("category") != category or slot <= 0:
                raise ValueError(f"invalid durable slot record: {category}/{slot}")
            if slot in observed[category]:
                raise ValueError(f"duplicate durable slot record: {category}/{slot}")
            observed[category][slot] = pair
        if observed[category] != expected[category]:
            raise ValueError(f"durable {category} slots do not match authoritative claims")
    if strict and (len(observed["normal"]) != NORMAL_CAP
                   or len(observed["reserve"]) > RESERVE_CAP):
        raise ValueError("strict durable slot ledger exceeds or fails frozen slot closure")
    return {"normal_slots": len(observed["normal"]),
            "reserve_slots": len(observed["reserve"]),
            "reserve_capacity_required_to_be_full": False,
            "claims_and_slots_match": True}


```
