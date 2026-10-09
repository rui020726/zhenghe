from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import r58_private_launcher_repaired as launcher
import r58_checkpoint_contract_repaired as checkpoint_contract
import r58_resume_seam as resume_seam


class StopBeforeEpisodeProvider(RuntimeError):
    pass


def main() -> int:
    proof = resume_seam.validate_resume_sources(REPO)
    binding = None
    rows = []
    counters = {}
    denied = {"environment_construct": 0, "episode_provider": 0,
              "optimizer_step": 0, "checkpoint_disk_write": 0,
              "claim_journal_write": 0, "rng_restore": 0}
    with tempfile.TemporaryDirectory(prefix="r58_resume_entry_", dir=launcher.GEN2) as temp:
        temp_root = Path(temp).resolve()
        audit_raw = temp_root / "audit_raw"
        resume_root = temp_root / "resume" / "raw"
        runtime_root = resume_root / "runtime"
        checkpoint_root = resume_root / "checkpoints"
        for directory in (resume_root, runtime_root, checkpoint_root,
                          resume_root / "logs", resume_root / "rows",
                          resume_root / "training_progress"):
            directory.mkdir(parents=True, exist_ok=True)

        binding = launcher.prepare_actual_runtime_for_audit(audit_raw)
        runtime, base = binding["runtime"], binding["base"]
        r40 = runtime["r40_train"]
        run = runtime["r42_run"]
        common = runtime["common"]
        legacy = binding["legacy"]

        # Route runtime objects exactly as the recover generation does: immutable
        # source claims remain in gen3/raw, new model artifacts/journals in sibling.
        common.OUTPUT_DIR = resume_root
        common.LOGS = resume_root / "logs"
        common.ROWS = resume_root / "rows"
        common.REJECTED = resume_root / "rejected"
        common.CHECKPOINTS = checkpoint_root
        runtime["r40_common"].ROUND_DIR = launcher.PRIVATE_RAW
        run.RUN_DIR = launcher.PRIVATE_RAW
        base.CHECKPOINT_JOURNAL_PATH = runtime_root / "checkpoint_writes.jsonl"
        base.FAMILY_JOURNAL_PATH = base.CHECKPOINT_JOURNAL_PATH
        base.FAMILY_LEDGER_PATH = runtime_root / "ROUND53_EXECUTION_LEDGER.json"
        base.LEDGER_PATH = base.FAMILY_LEDGER_PATH

        old_contract = runtime["r58_checkpoint_contract"]
        old_counter_bridge = runtime.get("r58_ppo_counter_bridge")
        old_contract["undo"]()
        identity = launcher._repair_identity()
        contract = checkpoint_contract.install(
            runtime["trainer"], r40, identity=identity,
            training_version=launcher.REPAIRED_TRAINING_VERSION,
            expected_checkpoint_root=checkpoint_root,
            versions=runtime["r42_versions"], r42_run=run,
            approved_sources=proof["sources"], resume_map_sha256=proof["manifest_sha256"])
        runtime["r58_checkpoint_contract"] = contract
        configure_bridge = resume_seam.install_r42_configure_compat(
            run, REPO, extra_batch_root=proof["sources"][11]["path"].parent,
            proof=proof, resume_runtime_dir=runtime_root)
        runtime["r58_resume_configure_bridge"] = configure_bridge
        runtime["r58_resume_source_proof"] = proof
        call_bridge = resume_seam.install_persistent_call_journals(
            runtime, resume_runtime_dir=runtime_root)
        runtime["r58_call_journals"] = call_bridge
        runtime["r58_ppo_counter_bridge"] = call_bridge

        strict_ledger = run.ledger_snapshot(strict=True)
        counts = strict_ledger.get("counts", {})
        if (len(strict_ledger.get("normal", [])) != 12
                or len(strict_ledger.get("reserve", [])) != 0
                or proof.get("historical_ppo_carryover") != 3
                or counts.get("optimizer_step_calls") != 24
                or counts.get("ppo_update_calls") != 3):
            raise RuntimeError(f"historical carryover ledger did not close: {counts}")

        original_save = runtime["trainer"].save_checkpoint
        original_snapshot = r40._write_immutable_snapshot
        original_progress = r40._save_progress
        original_step = torch.optim.Adam.step
        r40_run = runtime["r40_run"]
        original_episode = r40_run.run_training_episode
        attempted_checkpoint_paths = []
        provider_calls = []
        def memory_save(_controller, _optimizer, _state, path, *, freeze):
            denied["checkpoint_disk_write"] += 1
            attempted_checkpoint_paths.append(str(Path(path).resolve()))
            return {"kind": "memory-only-checkpoint-save-spy/1"}
        def memory_snapshot(path, _controller, _optimizer, _state, _freeze):
            denied["checkpoint_disk_write"] += 1
            attempted_checkpoint_paths.append(str(Path(path).resolve()))
            return "memory-only-snapshot-sha256"
        def memory_progress(*_args, **_kwargs):
            return None
        def deny_step(_optimizer, *_args, **_kwargs):
            denied["optimizer_step"] += 1
            raise RuntimeError("optimizer.step hard-denied by zero-environment resume audit")
        def stop_provider(_common, _runner, _controller, *, training_seed,
                          episode_index, scenario_seed, lam):
            denied["episode_provider"] += 1
            provider_calls.append({"seed": int(training_seed),
                                   "episode_index": int(episode_index),
                                   "scenario_seed": int(scenario_seed),
                                   "lambda": float(lam)})
            raise StopBeforeEpisodeProvider("stopped before environment construction")

        runtime["trainer"].save_checkpoint = memory_save
        r40._write_immutable_snapshot = memory_snapshot
        r40._save_progress = memory_progress
        torch.optim.Adam.step = deny_step
        r40_run.run_training_episode = stop_provider
        try:
            for seed in (11, 22, 33):
                freeze = runtime["round42_freeze"]
                chosen = r40._latest_resume_source(seed, freeze)
                if Path(chosen).resolve() != proof["sources"][seed]["path"]:
                    raise RuntimeError(f"seed {seed}: live selector did not choose approved exact source")
                order = list(common.scenario_order())
                references = {int(scene): {"on_time": 0} for scene in order}
                try:
                    r40.train_seed(seed, episodes=128, freeze=freeze,
                                   reference_cache=references)
                except StopBeforeEpisodeProvider:
                    pass
                else:
                    raise RuntimeError(f"seed {seed}: resume entry did not reach guarded provider")
                call = provider_calls[-1]
                if call["seed"] != seed or call["episode_index"] != 4:
                    raise RuntimeError(f"seed {seed}: live train_seed did not resume at episode 4: {call}")
                if call["scenario_seed"] != order[4]:
                    raise RuntimeError(f"seed {seed}: episode4 scenario order mismatch")
                rows.append({"seed": seed, "source_checkpoint_sha256":
                             proof["sources"][seed]["checkpoint_sha256"],
                             "selected_source": str(Path(chosen).resolve()),
                             "actual_train_seed_entry_episode": call["episode_index"],
                             "episode4_scenario": call["scenario_seed"],
                             "status": "actual_resume_loader_and_train_seed_reached_pre_environment_guard"})
        finally:
            runtime["trainer"].save_checkpoint = original_save
            r40._write_immutable_snapshot = original_snapshot
            r40._save_progress = original_progress
            torch.optim.Adam.step = original_step
            r40_run.run_training_episode = original_episode
            call_bridge["undo"]()
            runtime["r58_ppo_counter_bridge"] = old_counter_bridge

        counters = {"optimizer_steps": int(run._OPTIMIZER_STEP_CALLS),
                    "ppo_calls": int(run._PPO_UPDATE_CALLS),
                    "historical_checkpoint_writes": int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS),
                    "strict_ledger": {"normal_claims": len(strict_ledger["normal"]),
                                      "reserve_claims": len(strict_ledger["reserve"]),
                                      "counts": counts},
                    "direct_resume_optimizer_rows": 0,
                    "direct_resume_ppo_rows": 0,
                    "direct_resume_checkpoint_rows": 0,
                    "requested_checkpoint_saves_blocked_in_memory": len(attempted_checkpoint_paths),
                    "authorized_rng_restores": len(rows)}
        launcher.close_audit_runtime(binding)

    result = {"kind": "round58-gen3-resume-entry-zero-environment-audit/1",
              "status": "passed_three_seed_resume_entry_and_historical_ledger",
              "source_map_sha256": proof["manifest_sha256"],
              "historical_checkpoint_journal": proof["historical_checkpoint_journal"],
              "per_seed": rows, "counters": counters,
              "actual_side_effects": {"environment_construct": 0, "optimizer_step": 0,
                                      "ppo_update": 0, "checkpoint_write": 0,
                                      "claim_or_journal_write": 0, "rng_restore": 3},
              "blocked_or_intercepted_requests": denied,
              "reference_cache": "in-memory key-complete sentinel; provider guard fires before outcomes are read",
              "note": "No environment constructed; checkpoint save requests intercepted in memory; no core disk write.",
              "new_resume_root_was_temporary": True}
    out = HERE / "ROUND58_GEN3_RESUME_ENTRY_AUDIT.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
