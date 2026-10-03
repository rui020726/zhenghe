"""Read-only ROUND40 provenance check and immutable ROUND41 launch freeze."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

RUN_DIR = Path(__file__).resolve().parent
R41_ROOT = RUN_DIR.parents[1]
R40_WORKTREE = Path(r"D:\code\整合\zhenghe-round40")
R40_DIR = R40_WORKTREE / "runs" / "comm_first_report_training_ablation_20261003"
CANONICAL_INPUT_ROOT = Path(r"D:\code\整合\zhenghe")
MANIFEST = R40_DIR / "GIT_OUTSIDE_ARTIFACTS.json"
ROUND40_HEAD = "8f8245a665b15960e70194fc6e245221b70db90e"
EXPECTED_R40_BASE = "4e989a9088111ce6d9171cb1a21629a4b2b1e861"
EXPECTED_R41_BRANCH = "exp/comm-temporal-decoder-diagnostic"
SEEDS = (11, 22, 33)
SCENARIOS = tuple(range(1_301_000, 1_301_016))
MODES = ("greedy", "sampled")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest(path)}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False, capture_output=True, text=True,
    )
    return result.returncode == 0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("/", "\\").casefold()


def claim_path(logical: str, attempt: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", logical).strip("_.")
    return R40_DIR / "launch_claims" / slug / f"attempt_{attempt:04d}.json"


def check_claim(logical: str) -> dict[str, Any]:
    directory = claim_path(logical, 1).parent
    completed = []
    for path in sorted(directory.glob("attempt_*.json")) if directory.exists() else []:
        claim = load_json(path)
        if claim.get("status") == "complete":
            completed.append((path, claim))
    if len(completed) != 1:
        raise RuntimeError(f"expected exactly one completed ROUND40 claim for {logical}, got {len(completed)}")
    path, claim = completed[0]
    if int(claim.get("attempt", -1)) < 1:
        raise RuntimeError(f"invalid attempt on {path}")
    return claim


def validate_row_closure(row: dict[str, Any], *, path: Path) -> list[str]:
    if int(row.get("spent_bytes", -1)) != int(row.get("airtime_tap_total", -2)):
        raise RuntimeError(f"cost tap does not close at {path}")
    stream_total = sum(int(value.get("airtime_bytes", 0))
                       for value in row.get("cost_stream", {}).values())
    if stream_total != int(row["spent_bytes"]):
        raise RuntimeError(f"cost stream does not close at {path}")
    if int(row.get("task_reward_total", -1)) != int(row.get("on_time", -2)):
        raise RuntimeError(f"task reward does not close at {path}")
    reward_total = sum(float(value) for value in row.get("task_stream", {}).values())
    if reward_total != int(row["on_time"]):
        raise RuntimeError(f"task stream does not close at {path}")
    if int(row.get("steps", -1)) != 1000:
        raise RuntimeError(f"wrong episode step count at {path}")
    ids = sorted({str(event["task_id"]) for event in row.get("task_reward_events", [])
                  if event.get("on_time") and event.get("credited")})
    if len(ids) != int(row["on_time"]):
        raise RuntimeError(f"credited target ids do not close at {path}")
    if int(row.get("emissions", -1)) != len(row.get("emission_sequence", [])):
        raise RuntimeError(f"emissions do not close at {path}")
    return ids


def world_configuration(row: dict[str, Any]) -> dict[str, Any]:
    lifecycle = Path(row["artefacts"]["lifecycle"])
    with lifecycle.open("r", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    config = first.get("detail", {}).get("configuration")
    if not isinstance(config, dict):
        raise RuntimeError(f"lifecycle row has no serialized run configuration: {lifecycle}")
    return {key: value for key, value in config.items()
            if key not in {"output", "comm_transcript", "record_stream", "replay_stream"}}


def main() -> None:
    if not R40_DIR.is_dir() or not MANIFEST.is_file():
        raise FileNotFoundError(f"registered ROUND40 artifact directory is missing: {R40_DIR}")
    if git(R40_WORKTREE, "branch", "--show-current") != "exp/comm-first-report-training-ablation":
        raise RuntimeError("ROUND40 source branch changed")
    if git(R40_WORKTREE, "rev-parse", "HEAD") != ROUND40_HEAD:
        raise RuntimeError("ROUND40 source HEAD changed after provenance review")
    if git(R40_WORKTREE, "status", "--porcelain"):
        raise RuntimeError("ROUND40 source worktree is dirty")
    if git(CANONICAL_INPUT_ROOT, "rev-parse", "HEAD") != EXPECTED_R40_BASE:
        raise RuntimeError("canonical ROUND40 input checkout is not the recorded base HEAD")
    if git(R41_ROOT, "branch", "--show-current") != EXPECTED_R41_BRANCH:
        raise RuntimeError("not on the isolated ROUND41 branch")
    if not is_ancestor(R41_ROOT, ROUND40_HEAD, "HEAD"):
        raise RuntimeError("ROUND41 branch is not based on the verified ROUND40 head")

    manifest = load_json(MANIFEST)
    if int(manifest["raw_artifact_count"]) != len(manifest["raw_artifacts"]):
        raise RuntimeError("ROUND40 Git-external artifact count is internally inconsistent")
    artifact_index = {normalized(item["path"]): item for item in manifest["raw_artifacts"]}

    def verified_external(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"ROUND40 raw input is missing: {path}")
        entry = artifact_index.get(normalized(path))
        if entry is None:
            raise RuntimeError(f"ROUND40 raw input is absent from the frozen manifest: {path}")
        actual = file_record(path)
        if actual["size_bytes"] != int(entry["size_bytes"]):
            raise RuntimeError(f"ROUND40 raw size changed: {path}")
        if actual["sha256"] != entry["sha256"]:
            raise RuntimeError(f"ROUND40 raw SHA256 changed: {path}")
        return actual

    frozen_doc = load_json(R40_DIR / "ROUND40_FROZEN_CONFIG.json")
    round40 = frozen_doc["round40"]
    freeze = round40["freeze_identity"]
    if freeze["base_head"] != EXPECTED_R40_BASE:
        raise RuntimeError("ROUND40 frozen base identity disagrees with the actual source")
    if round40["unexpected_research_configuration_differences"]:
        raise RuntimeError("ROUND40 freeze records unexpected configuration differences")
    for name, expected in freeze["round40_code_sha256"].items():
        path = R40_DIR / name
        if digest(path) != expected:
            raise RuntimeError(f"ROUND40 frozen code changed: {path}")

    # Load all three final NEW models through the recorded project's read-only API.  The
    # read-only loader does not restore optimizer or RNG and returns parameter-level identity.
    dependency_dirs = [
        CANONICAL_INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928",
        CANONICAL_INPUT_ROOT / "runs" / "comm_cost_stage_b_20260927",
        CANONICAL_INPUT_ROOT / "runs" / "comm_budget_action_mask_fix_20260922",
        CANONICAL_INPUT_ROOT / "runs" / "comm_budget_pilot_20260919",
        CANONICAL_INPUT_ROOT / "runs" / "comm_cost_controller_stage_a_20260927",
        CANONICAL_INPUT_ROOT / "runs" / "comm_tight_budget_20260919",
        CANONICAL_INPUT_ROOT / "runs" / "search_trace_and_policy_20260925",
        R40_DIR,
        CANONICAL_INPUT_ROOT,
    ]
    for directory in reversed(dependency_dirs):
        sys.path.insert(0, str(directory))
    import torch
    import r40_common

    common = r40_common.bootstrap(CANONICAL_INPUT_ROOT)
    common.DEVICE = torch.device("cpu")
    runner = importlib.import_module("v2_run")
    trainer = importlib.import_module("v2_trainer")
    importlib.import_module("v2_controller")
    versions = importlib.import_module("r40_versions")
    versions.install(trainer, runner, freeze=freeze)

    paired = load_json(R40_DIR / "PAIRED_RESULTS.json")
    action_diag = load_json(R40_DIR / "TRAINING_ACTION_DIAGNOSTICS.json")
    training_diag = load_json(R40_DIR / "TRAINING_DIAGNOSTICS.json")
    version_manifest = load_json(R40_DIR / "TRAINING_VERSION_MANIFEST.json")
    if action_diag.get("environment_constructed") is not False:
        raise RuntimeError("ROUND40 action diagnostics unexpectedly constructed an environment")
    if int(action_diag.get("optimizer_steps", -1)) != 0:
        raise RuntimeError("ROUND40 passive action diagnostic unexpectedly performed optimizer steps")
    historical_optimizer_steps = sum(
        int(paired["checkpoint_results"]["NEW_training_outputs"][str(seed)]["ep128"]["optimizer_steps"])
        for seed in SEEDS
    )
    if historical_optimizer_steps != 556:
        raise RuntimeError("ROUND40 ep128 optimizer counters no longer sum to the reported 556")
    for name, expected in action_diag["source_hashes"].items():
        if digest(R40_DIR / name) != expected:
            raise RuntimeError(f"ROUND40 diagnostic source hash mismatch: {name}")

    checkpoint_freeze = paired["checkpoint_results"]["NEW_training_outputs"]
    model_records: dict[str, Any] = {}
    before_hashes: dict[str, str] = {}
    for seed in SEEDS:
        seed_key = str(seed)
        expected = checkpoint_freeze[seed_key]["ep128"]
        checkpoint = Path(expected["path"])
        artifact = verified_external(checkpoint)
        if artifact["sha256"] != expected["file_sha256"]:
            raise RuntimeError(f"ep128 checkpoint byte identity changed for seed {seed}")
        before_hashes[seed_key] = artifact["sha256"]
        controller, state, read_identity = trainer.load_checkpoint_read_only(
            checkpoint, freeze=freeze
        )
        parameter_identity = controller.parameter_summary()
        if parameter_identity["parameter_sha256"] != expected["parameter_sha256"]:
            raise RuntimeError(f"ep128 parameter identity changed for seed {seed}")
        if (read_identity.get("load_mode") != "read_only"
                or read_identity.get("optimizer_restored") is not False
                or read_identity.get("rng_restored") is not False):
            raise RuntimeError(f"seed {seed} was not loaded through a read-only boundary")
        if int(state.get("training_seed", -1)) != seed:
            raise RuntimeError(f"seed field mismatch in ep128 checkpoint {seed}")
        if int(state.get("episodes_done", -1)) != 128 or int(state.get("updates", -1)) != 32:
            raise RuntimeError(f"ep128 progress identity mismatch for seed {seed}")
        if int(state.get("optimizer_steps", -1)) != int(expected["optimizer_steps"]):
            raise RuntimeError(f"optimizer counter mismatch for seed {seed}")
        if float(state.get("lambda", float("nan"))) != float(expected["lambda"]):
            raise RuntimeError(f"lambda mismatch for seed {seed}")
        if state.get("training_version") != version_manifest["training_version"]:
            raise RuntimeError(f"training version mismatch for seed {seed}")
        if state.get("comm_action_eligibility") != version_manifest["comm_action_eligibility"]:
            raise RuntimeError(f"action-space identity mismatch for seed {seed}")
        model_records[seed_key] = {
            **artifact,
            "arm": "NEW",
            "training_seed": seed,
            "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]),
            "optimizer_steps_at_training_completion": int(state["optimizer_steps"]),
            "lambda": float(state["lambda"]),
            "training_version": state["training_version"],
            "reward_timing_version": state["reward_timing_version"],
            "comm_action_eligibility": state["comm_action_eligibility"],
            "parameter_identity": parameter_identity,
            "read_only_load_identity": read_identity,
        }
        del controller

    baseline_rows: dict[str, Any] = {}
    baseline_totals: dict[str, Any] = {}
    world_configs: dict[str, Any] = {}

    def add_row(key: str, row_path: Path, claim_logical: str, expected: dict[str, Any]) -> dict[str, Any]:
        claim = check_claim(claim_logical)
        if claim.get("status") != "complete":
            raise RuntimeError(f"ROUND40 claim did not complete: {claim_logical}")
        if int(claim["attempt"]) != int(row_path.stem.rsplit("_a", 1)[1]):
            raise RuntimeError(f"attempt mismatch for {claim_logical}")
        artifact = verified_external(row_path)
        row = load_json(row_path)
        ids = validate_row_closure(row, path=row_path)
        lifecycle_record = verified_external(Path(row["artefacts"]["lifecycle"]))
        transcript_record = verified_external(Path(row["artefacts"]["transcript"]))
        actual_world_config = world_configuration(row)
        world_hash = hashlib.sha256(json.dumps(
            actual_world_config, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")).hexdigest()
        if int(row["scenario_seed"]) != int(expected["scenario_seed"]):
            raise RuntimeError(f"scenario identity mismatch at {row_path}")
        for name, value in expected.items():
            if name == "scenario_seed":
                continue
            if row.get(name) != value:
                raise RuntimeError(f"{name} identity mismatch at {row_path}")
        row_identity = {
            "row": artifact,
            "scenario_seed": int(row["scenario_seed"]),
            "on_time": int(row["on_time"]),
            "on_time_target_ids": ids,
            "spent_bytes": int(row["spent_bytes"]),
            "emissions": int(row["emissions"]),
            "refresh_emissions": sum(not bool(emission[5])
                                      for emission in row["emission_sequence"]),
            "emission_sequence": row["emission_sequence"],
            "refusal_sequence": row["refusal_sequence"],
            "behaviour_sha256": row["behaviour_sha256"],
            "config_hash": row["config_hash"],
            "world_configuration_sha256": world_hash,
            "world_configuration": actual_world_config,
            "lifecycle": lifecycle_record,
            "transcript": transcript_record,
            "action_rng_seed": row.get("action_rng_seed"),
            "round40_claim": {
                "logical_job": claim_logical,
                "attempt": int(claim["attempt"]),
                "status": claim["status"],
            },
        }
        if row.get("artefacts", {}).get("tape"):
            tape_path = Path(row["artefacts"]["tape"])
            row_identity["tape"] = verified_external(tape_path)
        baseline_rows[key] = row_identity
        if key.startswith("NEW/s") and key.endswith("/greedy/" + str(row["scenario_seed"])):
            scene_key = str(row["scenario_seed"])
            prior = world_configs.get(scene_key)
            if prior is not None and prior != actual_world_config:
                raise RuntimeError(f"ROUND40 model-arm world configuration mismatch at {scene_key}")
            world_configs[scene_key] = actual_world_config
        return row

    model_totals: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            matching_rows = []
            expected_total = action_diag["final_ep128_evaluation_by_arm_seed_mode"][
                f"NEW/s{seed}/{mode}"
            ]
            expected_scenes = {int(item["scenario_seed"]): item
                               for item in expected_total["per_scene"]}
            if set(expected_scenes) != set(SCENARIOS):
                raise RuntimeError(f"ROUND40 action diagnostics have wrong scenes for {seed}/{mode}")
            for scene in SCENARIOS:
                logical = f"model_eval_new_s{seed}_{mode}_scene{scene}"
                claim = check_claim(logical)
                attempt = int(claim["attempt"])
                row_path = R40_DIR / "rows" / (
                    f"v2_r40_eval_new_s{seed}_ep128_{mode}_s{scene}_a{attempt}.json"
                )
                row = add_row(
                    f"NEW/s{seed}/{mode}/{scene}", row_path, logical,
                    {"scenario_seed": scene, "training_seed": seed, "model_arm": "NEW",
                     "timepoint": "ep128", "eval_mode": mode,
                     "eval_block": f"ep128_{mode}",
                     "comm_action_eligibility": version_manifest["comm_action_eligibility"]},
                )
                checkpoint_id = row.get("checkpoint_identity", {})
                if (checkpoint_id.get("file_sha256") != model_records[str(seed)]["sha256"]
                        or checkpoint_id.get("parameter_identity")
                        != model_records[str(seed)]["parameter_identity"]):
                    raise RuntimeError(f"ROUND40 row/checkpoint identity mismatch {seed}/{mode}/{scene}")
                if checkpoint_id.get("optimizer_restored") is not False or checkpoint_id.get("rng_restored") is not False:
                    raise RuntimeError(f"ROUND40 row was not read-only {seed}/{mode}/{scene}")
                if row.get("collector_summary", {}).get("timing", {}).get("decoder") != mode:
                    raise RuntimeError(f"ROUND40 eval mode label mismatch {seed}/{mode}/{scene}")
                if row.get("optimizer_steps") != 0:
                    raise RuntimeError(f"ROUND40 evaluation row records optimizer work {seed}/{mode}/{scene}")
                if mode == "greedy":
                    world_configs.setdefault(str(scene), baseline_rows[
                        f"NEW/s{seed}/{mode}/{scene}"
                    ]["world_configuration"])
                elif baseline_rows[f"NEW/s{seed}/greedy/{scene}"]["world_configuration"] != baseline_rows[
                    f"NEW/s{seed}/{mode}/{scene}"
                ]["world_configuration"]:
                    raise RuntimeError(f"ROUND40 mode changed world configuration at {seed}/{scene}")
                if row["config_hash"] != expected_scenes[scene].get("config_hash", row["config_hash"]):
                    raise RuntimeError(f"ROUND40 action diagnostic config hash mismatch {seed}/{mode}/{scene}")
                if (int(row["on_time"]) != int(expected_scenes[scene]["on_time"])
                        or int(row["spent_bytes"]) != int(expected_scenes[scene]["spent_bytes"])):
                    raise RuntimeError(f"ROUND40 per-scene summary seed binding mismatch {seed}/{mode}/{scene}")
                if row_identity := baseline_rows[f"NEW/s{seed}/{mode}/{scene}"]:
                    matching_rows.append(row)
            observed = {
                "scenes": len(matching_rows),
                "on_time_total": sum(int(row["on_time"]) for row in matching_rows),
                "bytes_total": sum(int(row["spent_bytes"]) for row in matching_rows),
                "rows": [baseline_rows[f"NEW/s{seed}/{mode}/{scene}"]["row"]
                         for scene in SCENARIOS],
            }
            if observed["scenes"] != 16:
                raise RuntimeError(f"wrong ROUND40 baseline row count for seed {seed}/{mode}")
            expected_round40 = paired["absolute_vs_immediate"][f"NEW/s{seed}/{mode}"]
            if (observed["on_time_total"] != int(expected_round40["policy_on_time_total"])
                    or observed["bytes_total"] != int(expected_round40["policy_bytes_total"])):
                raise RuntimeError(f"ROUND40 paired summary mismatch for seed {seed}/{mode}")
            totals = expected_total["totals"]
            if (observed["on_time_total"] != sum(int(x["on_time"]) for x in expected_scenes.values())
                    or observed["bytes_total"] != sum(int(x["spent_bytes"]) for x in expected_scenes.values())):
                raise RuntimeError(f"ROUND40 training action summary mismatch for seed {seed}/{mode}")
            model_totals[f"NEW/s{seed}/{mode}"] = {
                key: value for key, value in observed.items() if key != "rows"
            }

    rule_totals: dict[str, Any] = {}
    for arm in ("immediate", "late_discovery_gate", "always_hold"):
        rows = []
        for scene in SCENARIOS:
            if arm == "immediate":
                logical = f"reference_immediate_s{scene}"
                claim = check_claim(logical)
                path = R40_DIR / "rows" / f"v2_reference_s{scene}_a{int(claim['attempt'])}.json"
                expected = {"scenario_seed": scene, "reference_arm": "immediate"}
            else:
                logical = f"rule_{arm}_scene{scene}"
                claim = check_claim(logical)
                path = R40_DIR / "rows" / (
                    f"v2_r40_rule_{arm}_s{scene}_a{int(claim['attempt'])}.json"
                )
                expected = {"scenario_seed": scene, "model_arm": arm}
            row = add_row(f"{arm}/{scene}", path, logical, expected)
            if arm == "late_discovery_gate":
                identity = row.get("rule_identity", {})
                if (identity.get("arm") != arm
                        or float(identity.get("built_parameters", {}).get("cut_steps", -1)) != 147.0):
                    raise RuntimeError(f"late147 baseline identity mismatch at {scene}")
            elif arm == "always_hold":
                if row.get("rule_identity", {}).get("arm") != arm:
                    raise RuntimeError(f"always_hold baseline identity mismatch at {scene}")
            elif row.get("reference_arm") != "immediate":
                raise RuntimeError(f"immediate reference identity mismatch at {scene}")
            rows.append(row)
        rule_totals[arm] = {
            "scenes": len(rows),
            "on_time_total": sum(int(row["on_time"]) for row in rows),
            "bytes_total": sum(int(row["spent_bytes"]) for row in rows),
            "rows": [baseline_rows[f"{arm}/{scene}"]["row"] for scene in SCENARIOS],
        }
    if rule_totals["immediate"]["on_time_total"] != 96 or rule_totals["immediate"]["bytes_total"] != 239028:
        raise RuntimeError("ROUND40 immediate baseline does not match its reported absolute totals")
    if rule_totals["late_discovery_gate"]["on_time_total"] != 96 or rule_totals["late_discovery_gate"]["bytes_total"] != 77412:
        raise RuntimeError("ROUND40 late147 baseline does not match its reported identity/totals")
    if rule_totals["always_hold"]["on_time_total"] != 88 or rule_totals["always_hold"]["bytes_total"] != 0:
        raise RuntimeError("ROUND40 always_hold baseline does not match its reported totals")

    world_config_hashes: dict[str, str] = {}
    effective_world_hashes: dict[str, str] = {}
    for scene in SCENARIOS:
        fields, _substituted = common.stage_b_inherited_config(scene)
        serialized = json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)
        world_config_hashes[str(scene)] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        fields["comm_policy"] = "learned"
        fields["output"] = "<round-local-output>"
        fields["comm_transcript"] = "<round-local-transcript>"
        config = runner.RunConfiguration(**fields)
        config.validate()
        actual_config = asdict(config)
        actual_config = {key: value for key, value in actual_config.items()
                         if key not in {"output", "comm_transcript", "record_stream", "replay_stream"}}
        effective_serialized = json.dumps(
            actual_config, ensure_ascii=False, sort_keys=True, default=str
        )
        effective_world_hashes[str(scene)] = hashlib.sha256(
            effective_serialized.encode("utf-8")
        ).hexdigest()
        if effective_world_hashes[str(scene)] != hashlib.sha256(json.dumps(
                world_configs[str(scene)], ensure_ascii=False, sort_keys=True,
                default=str).encode("utf-8")).hexdigest():
            raise RuntimeError(f"inherited ROUND40 config does not reconstruct scenario {scene}")
        for seed in SEEDS:
            greedy = baseline_rows[f"NEW/s{seed}/greedy/{scene}"]["config_hash"]
            sampled = baseline_rows[f"NEW/s{seed}/sampled/{scene}"]["config_hash"]
            if greedy != sampled:
                raise RuntimeError(f"ROUND40 greedy/sample run config mismatch at {seed}/{scene}")

    for seed in SEEDS:
        path = R40_DIR / "checkpoints" / f"v2_train_s{seed}_ep128.pt"
        if digest(path) != before_hashes[str(seed)]:
            raise RuntimeError(f"read-only load changed ep128 checkpoint {seed}")

    # Rerun the same A-J pure-compute tests under the resolved project runtime; they import
    # only the decoder, policy contracts, and production budget shadow object.
    test_path = RUN_DIR / "test_r41_hazard_median.py"
    test = subprocess.run([sys.executable, str(test_path)], check=False,
                          capture_output=True, text=True)
    test_output = test.stdout + test.stderr
    if test.returncode != 0 or "Ran 10 tests" not in test_output or "OK" not in test_output:
        raise RuntimeError("ROUND41 A-J pure-compute gate failed:\n" + test_output[-6000:])
    test_names = re.findall(r"test_[A-J]_[A-Za-z0-9_]+ \(__main__\.HazardMathTests\) \.\.\. ok",
                            test_output)
    pure_gate = {
        "kind": "round41-pure-compute-gate/1",
        "status": "passed",
        "command": [sys.executable, str(test_path)],
        "test_count": 10,
        "test_names": [line.split(" (")[0] for line in test_names],
        "stdout_stderr": test_output,
        "environment_constructed": False,
        "joint_run_constructed": False,
        "simulator_started": False,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": 0,
        "decoder_rng_calls": 0,
    }

    tracked = [
        "r41_common.py", "r41_hazard_median.py", "r41_preflight.py", "r41_run.py",
        "test_r41_hazard_median.py",
    ]
    code_hashes = {name: digest(RUN_DIR / name) for name in tracked}
    if git(R41_ROOT, "status", "--porcelain"):
        raise RuntimeError("prelaunch source tree must be committed and clean before freeze")
    r41_head = git(R41_ROOT, "rev-parse", "HEAD")

    pure_path = RUN_DIR / "ROUND41_PURE_COMPUTE_GATE.json"
    pure_path.write_text(json.dumps(pure_gate, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    # The run file is authored before the freeze commit, so include its stable script hash
    # after writing the pure-gate artifact.
    all_code_hashes = {name: digest(RUN_DIR / name) for name in tracked}
    freeze_doc = {
        "kind": "round41-prelaunch-freeze/1",
        "status": "passed",
        "execution_identity": {
            "round40_branch": "exp/comm-first-report-training-ablation",
            "round40_head": ROUND40_HEAD,
            "round40_worktree": str(R40_WORKTREE),
            "round40_input_root": manifest["input_root"],
            "round40_canonical_input_head": EXPECTED_R40_BASE,
            "round41_branch": EXPECTED_R41_BRANCH,
            "round41_worktree": str(R41_ROOT),
            "round41_source_head_before_freeze_commit": r41_head,
            "round41_prelaunch_environment_launches": 0,
        },
        "method": {
            "mode": "hazard_median",
            "decoder_schema": "round41-cumulative-risk-decoder/1",
            "hazard_formula": "H_new = H_old - log1p(-p_send)",
            "threshold": float(__import__("math").log(2.0)),
            "threshold_origin": "fixed cumulative probability boundary 0.5",
            "eligible_only": "unemitted first report, legal and sequentially affordable candidate",
            "real_first_emission_closes_state": True,
            "request_or_shadow_charge_closes_state": False,
            "same_target_revisions_reset_state": False,
            "same_target_same_relay_tick_updates_at_most_once": True,
            "episode_reset": "new collector instance per episode",
            "post_first_emission_action": "HOLD",
            "actor_feature_width": 19,
            "H_in_actor_input": False,
            "action_rng_calls": 0,
            "p_one_representation": "finite H saturates at tau; threshold flag avoids JSON Infinity",
            "log_probability_semantics": "not a sampling likelihood for hazard_median",
            "training_version_or_weight_change": False,
            "production_default_change": False,
        },
        "models": model_records,
        "world_and_scenarios": {
            "scenarios": list(SCENARIOS),
            "scene_count": 16,
            "seeds": list(SEEDS),
            "inherited_world_config_sha256_by_scenario": world_config_hashes,
            "effective_round40_world_config_by_scenario": world_configs,
            "effective_world_config_sha256_by_scenario": effective_world_hashes,
            "only_policy_decoder_difference": True,
            "configuration_source": str(CANONICAL_INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_common.py"),
        },
        "evaluation_quota": {
            "new_hazard_median_formal": 48,
            "new_greedy_scene_1301000_reproduction": 3,
            "normal_launches": 51,
            "technical_reserve_max": 5,
            "hard_cap": 56,
            "first_gate_cells": [
                {"seed": seed, "mode": "greedy_reproduction", "scenario": SCENARIOS[0]}
                for seed in SEEDS
            ] + [
                {"seed": seed, "mode": "hazard_median", "scenario": SCENARIOS[0]}
                for seed in SEEDS
            ],
            "remaining_hazard_cells_after_gate": 45,
        },
        "baseline_reuse": {
            "new_greedy": 48,
            "new_sampled": 48,
            "immediate": 16,
            "late_discovery_gate_cut_147": 16,
            "always_hold": 16,
            "old_models_used_as_candidates": False,
            "rows": baseline_rows,
            "totals": model_totals,
            "rule_totals": {
                key: {name: value for name, value in record.items() if name != "rows"}
                for key, record in rule_totals.items()
            },
            "round40_action_diagnostics_source_hashes": action_diag["source_hashes"],
            "round40_training_diagnostics_seed_keys": sorted(training_diag["by_seed"].keys()),
            "round40_paired_results_kind": paired["kind"],
            "round40_ep128_optimizer_steps_by_seed": {
                str(seed): int(paired["checkpoint_results"]["NEW_training_outputs"][str(seed)]["ep128"]["optimizer_steps"])
                for seed in SEEDS
            },
            "round40_ep128_optimizer_steps_total": historical_optimizer_steps,
            "seed_mapping_method": "direct raw row, claim, tape and checkpoint joins; ROUND40 summary used only as cross-check",
            "round40_summary_binding_correction_reapplied": True,
        },
        "source_artifacts": {
            name: file_record(R40_DIR / name)
            for name in (
                "ROUND40_REPORT.md", "PAIRED_RESULTS.json", "TRAINING_ACTION_DIAGNOSTICS.json",
                "TRAINING_DIAGNOSTICS.json", "ROUND40_FROZEN_CONFIG.json",
                "ROUND40_PREFLIGHT.json", "TRAINING_VERSION_MANIFEST.json",
                "GIT_OUTSIDE_ARTIFACTS.json", "SCENARIO_ORDER.json",
            )
        },
        "round41_code_sha256": all_code_hashes,
        "prelaunch_tests": {
            "pure_compute": "ROUND41_PURE_COMPUTE_GATE.json",
            "environment_constructed": False,
            "joint_run_constructed": False,
            "optimizer_step_calls": 0,
            "checkpoint_write_calls": 0,
        },
    }
    (RUN_DIR / "ROUND41_PURE_COMPUTE_GATE.json").write_text(
        json.dumps(pure_gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (RUN_DIR / "ROUND41_FROZEN_CONFIG.json").write_text(
        json.dumps(freeze_doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "passed",
        "round40_head": ROUND40_HEAD,
        "round41_source_head": r41_head,
        "model_hashes": {seed: model_records[str(seed)]["sha256"] for seed in SEEDS},
        "baseline_row_count": len(baseline_rows),
        "pure_tests": pure_gate["test_count"],
        "environment_constructed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
