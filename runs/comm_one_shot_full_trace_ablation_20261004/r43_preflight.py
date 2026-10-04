"""ROUND43 read-only baseline audit and full-trace correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from r42_run import ROUND42_OUTPUT, RUN_DIR, configure, verify_frozen_source


FROZEN_PATH = RUN_DIR / "ROUND43_FROZEN_CONFIG.json"
CONFIG_DIFF_PATH = RUN_DIR / "ROUND43_CONFIGURATION_DIFF.json"
RECOVERY_GATE_PATH = RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"
ARTIFACT_DIRS = (
    "launch_claims", "launch_slots", "logs", "rows", "checkpoints",
    "training_progress", "runtime", "failures", "rejected",
)
R42_MANIFEST_PATH = ROUND42_OUTPUT / "ROUND42_RAW_ARTIFACT_MANIFEST.json"
R42_INDEX_PATH = ROUND42_OUTPUT / "ROUND42_CHECKPOINT_INDEX.json"
R42_PAIRED_PATH = ROUND42_OUTPUT / "ROUND42_PAIRED_RESULTS.json"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_existing_artifacts() -> dict[str, tuple[int, str]]:
    """Hash the pre-existing execution state so this gate can run during exact resume."""
    result: dict[str, tuple[int, str]] = {}
    for name in ARTIFACT_DIRS:
        root = RUN_DIR / name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                result[path.relative_to(RUN_DIR).as_posix()] = (
                    path.stat().st_size, _sha256(path))
    return result


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_close(actual: float, expected: float, *, tol: float = 1e-8,
                  message: str = "values differ") -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(f"{message}: {actual!r} != {expected!r}")


def _same_tree(actual: Any, expected: Any) -> bool:
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
                and actual.dtype == expected.dtype and actual.shape == expected.shape
                and torch.equal(actual, expected))
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        return (isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
                and actual.dtype == expected.dtype and actual.shape == expected.shape
                and np.array_equal(actual, expected))
    if isinstance(actual, dict) or isinstance(expected, dict):
        return (isinstance(actual, dict) and isinstance(expected, dict)
                and actual.keys() == expected.keys()
                and all(_same_tree(actual[key], expected[key]) for key in actual))
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        return (type(actual) is type(expected) and len(actual) == len(expected)
                and all(_same_tree(a, b) for a, b in zip(actual, expected)))
    try:
        equal = actual == expected
        return bool(equal) if not isinstance(equal, np.ndarray) else bool(equal.all())
    except (TypeError, ValueError):
        return False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest_rows(prefix: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = {item["round_relative_path"].replace("\\", "/"): item
               for item in manifest["files"]}
    rows: list[dict[str, Any]] = []
    for path in sorted((ROUND42_OUTPUT / "rows").glob(prefix)):
        relative = path.relative_to(ROUND42_OUTPUT).as_posix()
        entry = entries.get(relative)
        _assert(entry is not None, f"ROUND42 raw manifest has no row entry: {relative}")
        actual_sha = _sha256(path)
        _assert(actual_sha == entry["sha256"], f"ROUND42 row SHA256 mismatch: {relative}")
        row = _read(path)
        rows.append({"path": str(path), "relative_path": relative,
                     "sha256": actual_sha, "row": row})
    return rows


def audit_reusable_baselines(frozen: dict[str, Any], runtime: dict[str, Any]
                             ) -> dict[str, Any]:
    """Verify exact Round42 reference, rule, and OLD-evaluation cells before any launch."""
    manifest = _read(R42_MANIFEST_PATH)
    index = _read(R42_INDEX_PATH)
    paired = _read(R42_PAIRED_PATH)
    import r42_analysis

    refs = _manifest_rows("v2_reference_s*_a*.json", manifest)
    rules = _manifest_rows("v2_r42_rule_*.json", manifest)
    evals = _manifest_rows("v2_r42_eval_new_*.json", manifest)
    _assert(len(refs) == 64, f"expected 64 reference rows, found {len(refs)}")
    _assert(len(rules) == 32, f"expected 32 fixed-rule rows, found {len(rules)}")
    _assert(len(evals) == 96, f"expected 96 Round42 NEW ep128 evaluation rows, found {len(evals)}")

    expected_scenes = set(range(1_300_000, 1_300_048)) | set(range(1_301_000, 1_301_016))
    _assert({int(item["row"]["scenario_seed"]) for item in refs} == expected_scenes,
            "ROUND42 immediate reference scene set differs from the frozen plan")
    _assert(all(item["row"].get("reference_arm") == "immediate" for item in refs),
            "ROUND42 reference set contains a non-immediate row")

    rule_counts = Counter(item["row"].get("model_arm") for item in rules)
    _assert(rule_counts == Counter({"late_discovery_gate": 16, "always_hold": 16}),
            f"ROUND42 rule arms/counts differ: {rule_counts}")
    for item in rules:
        row = item["row"]
        ident = row.get("rule_identity", {})
        _assert(ident.get("arm") == row.get("model_arm"),
                f"rule identity mismatch in {item['relative_path']}")
        if row["model_arm"] == "late_discovery_gate":
            _assert(float(ident.get("params", {}).get("cut_steps", -1)) == 147.0,
                    f"late_discovery_gate cut changed in {item['relative_path']}")

    eval_groups: Counter[tuple[int, str]] = Counter()
    eval_keys: set[tuple[int, str, int]] = set()
    old_eval_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
    training_version = frozen["round42"]["source_training_version"]
    for item in evals:
        row = item["row"]
        seed, mode, scene = (int(row["training_seed"]), str(row["eval_mode"]),
                             int(row["scenario_seed"]))
        key = (seed, mode)
        eval_groups[key] += 1
        _assert((seed, mode, scene) not in eval_keys,
                f"duplicate ROUND42 OLD evaluation cell: {seed}/{mode}/{scene}")
        eval_keys.add((seed, mode, scene))
        _assert(row.get("model_arm") == "NEW", "OLD baseline must be ROUND42 NEW")
        _assert(row.get("training_version") == training_version,
                f"OLD row has wrong Round42 training version: {item['relative_path']}")
        _assert(row.get("comm_action_semantics") == "one-shot-send-skip/1",
                f"OLD row has wrong one-shot semantics: {item['relative_path']}")
        _assert(row.get("eval_block") == f"one_shot_{mode}"
                and row.get("policy") == "learned"
                and row.get("collector_summary", {}).get("timing", {}).get("decoder") == mode,
                f"OLD row's frozen one-shot decoder/mode identity changed: {item['relative_path']}")
        _assert(row.get("runner_version") == "counted-jointrun-proxy/1",
                f"OLD row has an unexpected runner identity: {item['relative_path']}")
        ident = row.get("checkpoint_load_identity", {})
        expected_model = index["new_checkpoints"][str(seed)]["128"]
        _assert(ident.get("load_mode") == "read_only"
                and ident.get("optimizer_restored") is False
                and ident.get("rng_restored") is False,
                f"OLD evaluation was not strictly read-only: {item['relative_path']}")
        _assert(Path(ident.get("path", "")).resolve() == Path(expected_model["path"]).resolve()
                and ident.get("file_sha256") == expected_model["file_sha256_after"],
                f"OLD evaluation checkpoint identity mismatch: {item['relative_path']}")
        _assert(row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
                == expected_model["parameter_identity"]["parameter_sha256"],
                f"OLD evaluation parameter identity mismatch: {item['relative_path']}")
        _assert(row.get("execution_identity", {}).get("execution_head")
                == frozen["execution_identity"]["source_execution_head"],
                f"OLD row source execution head mismatch: {item['relative_path']}")
        old_eval_rows.setdefault(key, []).append(row)

    expected_eval_groups = {(seed, mode): 16 for seed in (11, 22, 33)
                            for mode in ("greedy", "sampled")}
    _assert(dict(eval_groups) == expected_eval_groups,
            f"ROUND42 OLD evaluation groups differ: {dict(eval_groups)}")
    for item in refs + rules + evals:
        row = item["row"]
        _assert(int(row.get("steps", -1)) == 1000,
                f"baseline episode horizon changed: {item['relative_path']}")
        _assert(float(row.get("b0_bytes", -1)) == 10_000.0,
                f"baseline B0 changed: {item['relative_path']}")
        _assert(row.get("reward_timing_version") == "published-completion-event-at-relay-tick/2",
                f"baseline reward timing identity changed: {item['relative_path']}")
        r42_analysis._validate_reward_timing(row)

    # Read-only content verification covers all six same-seed ep0/ep128 files. Checkpoint
    # bytes, training metadata, parameter summaries, and the OLD evaluation references are
    # kept in separate namespaces in the resulting inventory.
    checkpoints: dict[str, Any] = {}
    trainer = runtime["trainer"]
    for seed in (11, 22, 33):
        checkpoints[str(seed)] = {}
        for episode in (0, 128):
            record = index["new_checkpoints"][str(seed)][str(episode)]
            path = Path(record["path"])
            _assert(path.is_file() and _sha256(path) == record["file_sha256_after"],
                    f"ROUND42 checkpoint file SHA256 mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload["training_state"]
            controller, read_state, identity = trainer.load_checkpoint_read_only(path)
            actual_parameter = controller.parameter_summary()
            _assert(actual_parameter["parameter_sha256"]
                    == record["parameter_identity"]["parameter_sha256"],
                    f"ROUND42 parameter content mismatch: seed {seed} ep{episode}")
            _assert(int(state["training_seed"]) == seed
                    and int(state["episodes_done"]) == episode
                    and state["training_version"] == training_version
                    and state["comm_action_semantics"] == "one-shot-send-skip/1"
                    and float(state["gae_lambda"]) == 0.95
                    and float(state["gamma"]) == 1.0,
                    f"ROUND42 checkpoint metadata mismatch: seed {seed} ep{episode}")
            _assert(identity.get("load_mode") == "read_only"
                    and identity.get("optimizer_restored") is False
                    and identity.get("rng_restored") is False,
                    f"ROUND42 checkpoint reader restored training state: seed {seed} ep{episode}")
            _assert(int(read_state["episodes_done"]) == episode,
                    f"ROUND42 readonly state cursor mismatch: seed {seed} ep{episode}")
            if episode == 0:
                _assert(int(state["updates"]) == 0 and int(state["optimizer_steps"]) == 0
                        and len(payload["optimizer"]["state"]) == 0
                        and float(state["lambda"]) == 1.0,
                        f"ROUND42 ep0 state is not an untrained initialization: seed {seed}")
            else:
                _assert(int(state["updates"]) == 32 and int(state["optimizer_steps"]) > 0,
                        f"ROUND42 ep128 cursor mismatch: seed {seed}")
            checkpoints[str(seed)][str(episode)] = {
                "path": str(path), "file_sha256": _sha256(path),
                "parameter_sha256": actual_parameter["parameter_sha256"],
                "episodes_done": int(state["episodes_done"]),
                "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda_dual": float(state["lambda"]),
                "gae_lambda": float(state["gae_lambda"]),
                "gamma": float(state["gamma"]),
                "training_version": state["training_version"],
                "read_only": identity,
            }

    return {
        "kind": "round43-baseline-reuse-audit/1",
        "status": "passed",
        "round42_delivery_head": frozen["execution_identity"]["source_delivery_head"],
        "round42_execution_head": frozen["execution_identity"]["source_execution_head"],
        "raw_manifest_sha256": _sha256(R42_MANIFEST_PATH),
        "paired_results_sha256": _sha256(R42_PAIRED_PATH),
        "checkpoint_index_sha256": _sha256(R42_INDEX_PATH),
        "reused": {
            "immediate_references": {"count": len(refs), "rows": refs},
            "fixed_rules": {"count": len(rules), "rows": rules},
            "old_round42_new_ep128_evaluations": {
                "count": len(evals), "groups": {f"{s}_{m}": n
                                                   for (s, m), n in sorted(eval_groups.items())},
                "rows": evals,
            },
        },
        "checkpoint_content_verification": checkpoints,
        "interpretation": (
            "ROUND42 NEW ep128 is OLD for ROUND43; no ROUND42-trained checkpoint is used to "
            "initialize ROUND43 NEW. The 16-scene block is previously used development data."),
    }


def _make_tape(step: int, *, value: float, cost_value: float, action: int,
               free: list[str], targets: list[str] | None = None) -> dict[str, Any]:
    targets = targets or ["target"]
    return {
        "step": step,
        "features": np.zeros((len(targets), 19), dtype=np.float32),
        "targets": targets,
        "actions": {target: action for target in targets},
        "actionable": list(free),
        "free": list(free),
        "log_prob": (len(free) * math.log(0.5)),
        "value": value,
        "cost_value": cost_value,
    }


def _episode(episode_index: int, tapes: list[dict[str, Any]], *, steps: int,
             task_stream: dict[int, float], cost_stream: dict[int, dict[str, float]],
             dual: float = 1.0) -> dict[str, Any]:
    return {
        "collector": SimpleNamespace(records=tapes), "steps": steps,
        "task_stream": task_stream, "cost_stream": cost_stream,
        "lambda": dual, "episode_index": episode_index,
        "scenario_seed": 1_300_000 + episode_index, "on_time": int(sum(task_stream.values())),
    }


def _hand_gae(rewards: list[float], values: list[float], gamma: float,
              trace: float, terminal: list[bool], bootstrap: list[float]
              ) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        alive = 0.0 if terminal[i] else 1.0
        delta = rewards[i] + gamma * bootstrap[i] * alive - values[i]
        advantages[i] = delta + gamma * trace * alive * next_advantage
        next_advantage = advantages[i]
    return advantages, [a + v for a, v in zip(advantages, values)]


def _test_batch_samples_and_gae(runtime: dict[str, Any]) -> dict[str, Any]:
    trainer = runtime["trainer"]
    common = runtime["common"]
    gamma = float(common.GAMMA)
    _assert(gamma == 1.0, "ROUND43 changed gamma")
    delayed_episode = _episode(
        0,
        [
            _make_tape(1, value=0.2, cost_value=0.05, action=1, free=["target"]),
            _make_tape(4, value=0.4, cost_value=0.03, action=0, free=[]),
        ],
        steps=5,
        task_stream={5: 1.0},
        cost_stream={1: {"airtime_bytes": 1000.0},
                     5: {"airtime_bytes": 500.0}},
        dual=1.25,
    )
    rows: dict[str, Any] = {}
    for trace in (0.95, 1.0):
        common.GAE_LAMBDA = trace
        samples = trainer.batch_samples([delayed_episode])
        _assert([int(s["step"]) for s in samples] == [1, 4],
                "actual batch_samples changed decision-row/time structure")
        expected_task_a, expected_task_ret = _hand_gae(
            [0.0, 1.0], [0.2, 0.4], gamma, trace, [False, True], [0.4, 0.0])
        expected_cost_a, expected_cost_ret = _hand_gae(
            [0.1, 0.05], [0.05, 0.03], gamma, trace, [False, True], [0.03, 0.0])
        expected_combined = [1.25 * a - c for a, c in zip(expected_task_a, expected_cost_a)]
        independent = []
        for i, sample in enumerate(samples):
            _assert_close(sample["return_task"], expected_task_ret[i],
                          message=f"task return, trace={trace}, row={i}")
            _assert_close(sample["return_cost"], expected_cost_ret[i],
                          message=f"positive-cost return, trace={trace}, row={i}")
            _assert_close(sample["advantage"], expected_combined[i],
                          message=f"combined advantage, trace={trace}, row={i}")
            independent.append({
                "step": int(sample["step"]), "free_action_row": bool(sample["free"]),
                "task_advantage_hand": expected_task_a[i],
                "cost_advantage_hand": expected_cost_a[i],
                "combined_advantage_hand": expected_combined[i],
                "actual_return_task": float(sample["return_task"]),
                "actual_return_cost": float(sample["return_cost"]),
                "actual_combined_advantage": float(sample["advantage"]),
            })
        rows[str(trace)] = {"rows": independent}

    _assert_close(rows["1.0"]["rows"][0]["actual_return_task"], 1.0,
                  message="lambda=1 full terminal trace task return")
    _assert_close(rows["1.0"]["rows"][0]["actual_return_cost"], 0.15,
                  message="lambda=1 full terminal trace positive cost return")
    _assert(rows["0.95"]["rows"][0]["actual_return_task"] != 1.0,
            "0.95 trace was incorrectly treated as an implementation failure")

    # Exercise the actual GAE helper's terminal and bootstrap-truncation branches with
    # nonzero values; expected returns are calculated as direct finite-horizon sums.
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    terminal = torch.tensor([False, False, True])
    bootstrap = torch.tensor([0.5, 0.75, 0.0], dtype=torch.float64)
    actual_a, actual_ret = trainer.cm.gae(
        rewards, values, gamma=1.0, gae_lambda=1.0,
        terminal=terminal, bootstrap=bootstrap)
    expected_returns = [6.0, 5.0, 3.0]
    expected_advantages = [expected_returns[i] - float(values[i]) for i in range(3)]
    for i in range(3):
        _assert_close(actual_ret[i], expected_returns[i], message=f"true terminal return row {i}")
        _assert_close(actual_a[i], expected_advantages[i], message=f"true terminal advantage row {i}")

    trunc_rewards = torch.tensor([0.5, 0.25], dtype=torch.float64)
    trunc_values = torch.tensor([0.2, 0.3], dtype=torch.float64)
    trunc_terminal = torch.tensor([False, False])
    trunc_bootstrap = torch.tensor([0.3, 0.7], dtype=torch.float64)
    trunc_a, trunc_ret = trainer.cm.gae(
        trunc_rewards, trunc_values, gamma=1.0, gae_lambda=1.0,
        terminal=trunc_terminal, bootstrap=trunc_bootstrap)
    for i, expected in enumerate(([1.45, 0.95], [1.25, 0.65])):
        observed = trunc_ret if i == 0 else trunc_a
        for j in range(2):
            _assert_close(observed[j], expected[j], message=f"truncation output {i}/{j}")

    common.GAE_LAMBDA = 1.0
    return {
        "kind": "round43-batch-samples-gae-regression/1",
        "status": "passed",
        "gamma": gamma,
        "trace_comparison": rows,
        "full_terminal_nonzero_values": {
            "returns": expected_returns, "advantages": expected_advantages,
            "actual_returns": [float(x) for x in actual_ret],
            "actual_advantages": [float(x) for x in actual_a],
        },
        "bootstrap_truncation_nonzero_values": {
            "boundary_bootstrap_value": 0.7,
            "actual_returns": [float(x) for x in trunc_ret],
            "actual_advantages": [float(x) for x in trunc_a],
            "not_described_as_pure_monte_carlo": True,
        },
        "cost_convention": "positive measured cost; combined advantage lambda_dual*A_task-A_cost",
    }


def _test_live_actor_path(runtime: dict[str, Any]) -> dict[str, Any]:
    trainer = runtime["trainer"]
    common = runtime["common"]
    ambient = trainer.rng_state()
    controller = runtime["controller_module"].fresh_controller(11)
    with torch.no_grad():
        controller.inner.action_head.weight.zero_()
        controller.inner.action_head.bias.zero_()
        controller.inner.value_head.weight.zero_()
        controller.inner.value_head.bias.zero_()
        controller.cost_head.weight.zero_()
        controller.cost_head.bias.zero_()

    send = _episode(
        0, [_make_tape(1, value=0.0, cost_value=0.0, action=1, free=["target"])],
        steps=3, task_stream={3: 1.0},
        cost_stream={1: {"airtime_bytes": 2000.0}}, dual=1.0)
    skip = _episode(
        1, [_make_tape(1, value=0.0, cost_value=0.0, action=0, free=["target"])],
        steps=3, task_stream={}, cost_stream={}, dual=1.0)
    closed = _episode(
        2, [_make_tape(3, value=0.0, cost_value=0.0, action=0, free=[])],
        steps=3, task_stream={}, cost_stream={}, dual=1.0)
    episodes = [send, skip, closed]
    samples = trainer.batch_samples(episodes)
    _assert(len(samples) == 3 and samples[1]["free"] == {"target"}
            and samples[2]["free"] == set(),
            "a first SKIP lost actor eligibility or a fixed HOLD row was removed")
    _assert_close(samples[0]["advantage"], 0.8,
                  message="SEND's one-task minus 0.2 normalized cost advantage")
    _assert_close(samples[1]["advantage"], 0.0, message="SKIP synthetic advantage")
    _assert_close(samples[2]["advantage"], 0.0, message="closed HOLD synthetic advantage")
    alignment = trainer.likelihood_alignment(controller, samples)
    _assert(alignment.get("passed") is True
            and float(alignment.get("max_abs_gap", math.inf)) <= 1e-6,
            "actual likelihood alignment rejected SEND/SKIP or the closed critic row")

    parameters_before = [p.detach().clone() for p in controller.parameters]
    actual_clip = torch.nn.utils.clip_grad_norm_
    clip_norms: list[float] = []

    def capture_clip(parameters, max_norm, *args, **kwargs):
        total = actual_clip(parameters, max_norm, *args, **kwargs)
        clip_norms.append(float(total))
        return total

    class BlockBeforeUpdate(RuntimeError):
        pass

    class BlockedOptimizer:
        def __init__(self) -> None:
            self.step_attempts = 0
            self.gradients: dict[str, list[float]] = {}

        def zero_grad(self) -> None:
            for parameter in controller.parameters:
                parameter.grad = None

        def step(self) -> None:
            self.step_attempts += 1
            self.gradients["action_head.bias"] = controller.inner.action_head.bias.grad.detach().cpu().tolist()
            raise BlockBeforeUpdate("preflight blocks immediately before optimizer state mutation")

    optimizer = BlockedOptimizer()
    torch.nn.utils.clip_grad_norm_ = capture_clip
    try:
        try:
            # The original implementation is invoked directly so this synthetic gradient probe
            # does not enter the experiment's PPO-update counter.
            runtime["restore_ppo_update"](controller, optimizer, episodes, update_index=0)
        except BlockBeforeUpdate:
            pass
        else:
            raise RuntimeError("actual actor path did not reach the guarded optimizer boundary")
    finally:
        torch.nn.utils.clip_grad_norm_ = actual_clip
        trainer.restore_rng(ambient)

    _assert(optimizer.step_attempts == 1 and len(clip_norms) == 1,
            "guarded actor path made zero or multiple optimizer-boundary attempts")
    _assert(all(torch.equal(before, after.detach())
                for before, after in zip(parameters_before, controller.parameters)),
            "guarded gradient gate changed model parameters")
    _assert(int(__import__("r42_run")._OPTIMIZER_STEP_CALLS) == 0,
            "synthetic actor gate invoked an Adam optimizer update")
    actual_gradient = optimizer.gradients["action_head.bias"]
    n_rows = len(samples)  # ALL_ROWS denominator includes the critic-only closed-HOLD row.
    preclip_class1 = -0.8 / (2.0 * n_rows)
    preclip_class0 = 0.8 / (2.0 * n_rows)
    scale = min(1.0, float(common.MAX_GRAD_NORM) / (clip_norms[0] + 1e-6))
    _assert_close(actual_gradient[0], preclip_class0 * scale, tol=1e-5,
                  message="SEND/SKIP ALL_ROWS actor gradient, class 0")
    _assert_close(actual_gradient[1], preclip_class1 * scale, tol=1e-5,
                  message="SEND/SKIP ALL_ROWS actor gradient, SEND class")
    _assert(actual_gradient[0] > 0.0 and actual_gradient[1] < 0.0,
            "actor gradient does not increase the relative SEND logit for A_actor=0.8")

    logits = torch.zeros((1, 2), dtype=torch.float32)
    send_skip_ll = float(trainer.joint_log_prob(
        logits, torch.tensor([0]), torch.tensor([1.0])))
    free_entropy = float(trainer.action_entropy(logits, torch.tensor([1.0])))
    closed_ll = float(trainer.joint_log_prob(
        logits, torch.tensor([0]), torch.tensor([0.0])))
    closed_entropy = float(trainer.action_entropy(logits, torch.tensor([0.0])))
    _assert_close(send_skip_ll, -math.log(2.0), message="first SKIP log-likelihood")
    _assert_close(free_entropy, math.log(2.0), message="first SKIP entropy")
    _assert_close(closed_ll, 0.0, message="fixed HOLD likelihood")
    _assert_close(closed_entropy, 0.0, message="fixed HOLD entropy")
    return {
        "kind": "round43-live-actor-path-regression/1",
        "status": "passed",
        "actual_path": "trainer.ppo_update -> batch_samples -> joint_log_prob/action_entropy -> loss.backward -> clip_grad_norm_",
        "synthetic_rows": 3,
        "send_row_advantage": 0.8,
        "skip_row_advantage": 0.0,
        "closed_hold_row_advantage": 0.0,
        "closed_hold_in_all_rows_denominator": True,
        "action_head_bias_gradient_after_clip": actual_gradient,
        "preclip_expected_gradient": [preclip_class0, preclip_class1],
        "clip_norm_before_clip": clip_norms[0],
        "clip_scale": scale,
        "send_class": 1,
        "gradient_moves_toward_send": True,
        "first_skip_log_probability": send_skip_ll,
        "first_skip_entropy": free_entropy,
        "fixed_hold_log_probability": closed_ll,
        "fixed_hold_entropy": closed_entropy,
        "optimizer_update_calls": 0,
        "guarded_step_boundary_attempts": optimizer.step_attempts,
        "parameters_unchanged": True,
    }


def _test_registered_ep0_initialization(runtime: dict[str, Any], frozen: dict[str, Any]
                                        ) -> dict[str, Any]:
    """Prove the actual fresh model/Adam start matches each registered ROUND42 ep0 identity."""
    trainer = runtime["trainer"]
    common = runtime["common"]
    versions = runtime["r42_versions"]
    index = _read(R42_INDEX_PATH)
    ambient_rng = trainer.rng_state()
    by_seed: dict[str, Any] = {}
    import r42_run
    try:
        for seed in (11, 22, 33):
            registered = frozen["round43"]["registered_ep0_inputs"][str(seed)]
            path = Path(registered["path"])
            _assert(path.is_file() and _sha256(path) == registered["file_sha256"],
                    f"registered ep0 RNG input changed for seed {seed}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected = index["new_checkpoints"][str(seed)]["0"]
            with r42_run._registered_ep0_rng_after_init(
                    runtime, seed, payload["rng"]) as prime:
                controller = runtime["controller_module"].fresh_controller(seed)
                optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
                state = trainer.fresh_training_state(seed)
                active_rng = trainer.rng_state()
            parameter_sha = controller.parameter_summary()["parameter_sha256"]
            _assert(parameter_sha == expected["parameter_identity"]["parameter_sha256"],
                    f"fresh same-seed model does not match registered ep0 parameters: seed {seed}")
            _assert(parameter_sha == registered["parameter_sha256"],
                    f"registered ep0 parameter identity disagrees with content index: seed {seed}")
            _assert(_same_tree(optimizer.state_dict(), payload["optimizer"]),
                    f"fresh Adam state does not match registered ep0 optimizer: seed {seed}")
            _assert(prime.get("done") is True,
                    f"registered ep0 RNG adapter did not run after initialization: seed {seed}")
            _assert(_same_tree(active_rng, payload["rng"]),
                    f"registered ep0 RNG was not exactly applied to all captured generators: seed {seed}")
            fresh_rng = prime["fresh_controller_rng"]
            natural_stream_mismatches = sorted(
                key for key in payload["rng"]
                if not _same_tree(fresh_rng[key], payload["rng"][key]))
            _assert(int(state["training_seed"]) == seed
                    and int(state["episodes_done"]) == 0
                    and int(state["updates"]) == 0
                    and int(state["optimizer_steps"]) == 0
                    and float(state["lambda"]) == 1.0
                    and float(state["gamma"]) == 1.0
                    and float(state["gae_lambda"]) == 1.0
                    and state["training_version"] == versions.TRAINING_VERSION,
                    f"fresh ROUND43 state cursor/config mismatch: seed {seed}")
            by_seed[str(seed)] = {
                "registered_ep0_path": str(path),
                "registered_ep0_file_sha256": registered["file_sha256"],
                "parameter_sha256": parameter_sha,
                "optimizer_empty_and_matching": True,
                "registered_rng_exactly_matches_ep0_after_fresh_init": True,
                "generator_keys": sorted(payload["rng"]),
                "fresh_initializer_natural_stream_differences": natural_stream_mismatches,
                "natural_initializer_streams_replaced_before_first_episode": True,
                "fresh_state": {key: state[key] for key in (
                    "training_seed", "episodes_done", "updates", "optimizer_steps",
                    "lambda", "gamma", "gae_lambda", "training_version")},
                "loaded_model_or_optimizer": False,
            }
    finally:
        trainer.restore_rng(ambient_rng)
    return {
        "kind": "round43-registered-ep0-initialization/1",
        "status": "passed",
        "by_seed": by_seed,
        "actual_rng_sources": [
            {
                "stream": "Python/NumPy global",
                "source": "registered ROUND42 ep0 RNG restored after fresh controller init; the real PILOC constructor then seeds these globals from RunConfiguration.seed, and reset_piloc_environment reseeds the environment's own streams from the same scenario seed",
                "call_path": [
                    "r42_run._registered_ep0_rng_after_init",
                    "v2_run._run_episode -> build_joint_run(configuration.seed=scenario_seed)",
                    "integrated_sar.piloc_runtime.build_piloc_search_side",
                    "integrated_sar.piloc_runtime.reset_piloc_environment",
                ],
            },
            {
                "stream": "Torch CPU global for one-shot policy sampling",
                "source": "v2_run._run_episode applies the unchanged per-episode action_seed after environment construction and before the first step",
                "derivation": "sha256('sc-action:{training_seed}:{episode_index}')[:16]",
            },
            {
                "stream": "PPO minibatch ordering",
                "source": "dedicated random.Random instance per update; does not consume the Python global stream",
                "derivation": "sha256(f'{DATA_ORDER_SEED}:{update_index}')[:16]",
            },
            {
                "stream": "scenario, PILOC environment, and forward communication channel",
                "source": "dedicated deterministic generators seeded from the registered scenario_seed; environment reseed is performed by the original run path",
                "independent_of_global_rng": True,
            },
        ],
        "interpretation": (
            "Fresh same-seed controller and empty Adam were constructed locally; only after "
            "controller initialization does the training runner apply the registered ep0 RNG. "
            "No ROUND42 model weights or optimizer state are used to train ROUND43."),
    }


def _batch0_recompute(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    trainer = runtime["trainer"]
    common = runtime["common"]
    directory = ROUND42_OUTPUT / "checkpoints"
    result: dict[str, Any] = {}
    for seed in (11, 22, 33):
        commit_path = directory / f"r40_batch_commit_s{seed}_b0.json"
        commit = _read(commit_path)
        record_path = directory / commit["batch_record"]
        _assert(_sha256(record_path) == commit["batch_record_sha256"],
                f"ROUND42 batch0 record hash mismatch: seed {seed}")
        record = _read(record_path)
        _assert(record["episodes"] == [0, 1, 2, 3]
                and int(record["training_seed"]) == seed,
                f"ROUND42 batch0 cursor mismatch: seed {seed}")
        pre_path = directory / str(record["preupdate_checkpoint"])
        _assert(_sha256(pre_path) == record["preupdate_checkpoint_sha256"],
                f"ROUND42 batch0 preupdate checkpoint hash mismatch: seed {seed}")
        pre_controller, pre_state, pre_identity = trainer.load_checkpoint_read_only(pre_path)
        _assert(int(pre_state["episodes_done"]) == 0
                and pre_identity["optimizer_restored"] is False
                and pre_identity["rng_restored"] is False,
                f"ROUND42 batch0 critic checkpoint is not readonly ep0: seed {seed}")

        episodes: list[dict[str, Any]] = []
        artifact_inputs = []
        for artifact in record["episode_artifacts"]:
            row_path = Path(artifact["row_path"])
            tape_path = Path(artifact["tape_path"])
            _assert(_sha256(row_path) == artifact["row_sha256"]
                    and _sha256(tape_path) == artifact["tape_sha256"],
                    f"ROUND42 batch0 raw input hash mismatch: seed {seed}")
            row = _read(row_path)
            tapes = _load_jsonl(tape_path)
            _assert(row["training_version"] == frozen["round42"]["source_training_version"]
                    and int(row["training_seed"]) == seed,
                    f"ROUND42 batch0 raw row identity mismatch: seed {seed}")
            for tape in tapes:
                features = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
                with torch.no_grad():
                    _logits, value_task = pre_controller.forward(features)
                    value_cost = pre_controller.cost_values(features)
                _assert_close(tape["value"], float(value_task.mean()), tol=1e-6,
                              message=f"ROUND42 tape task critic output seed {seed}")
                _assert_close(tape["cost_value"], float(value_cost.mean()), tol=1e-6,
                              message=f"ROUND42 tape cost critic output seed {seed}")
            episodes.append({
                "collector": SimpleNamespace(records=tapes),
                "steps": int(row["steps"]),
                "task_stream": {int(k): float(v) for k, v in row["task_stream"].items()},
                "cost_stream": {int(k): v for k, v in row["cost_stream"].items()},
                "lambda": float(row["lambda_at_rollout"]),
                "episode_index": int(row["episode_index"]),
                "scenario_seed": int(row["scenario_seed"]),
                "on_time": int(row["on_time"]),
            })
            artifact_inputs.append({
                "episode_index": int(row["episode_index"]),
                "scenario_seed": int(row["scenario_seed"]),
                "row_path": str(row_path), "row_sha256": artifact["row_sha256"],
                "tape_path": str(tape_path), "tape_sha256": artifact["tape_sha256"],
            })

        common.GAE_LAMBDA = 0.95
        old_samples = trainer.batch_samples(episodes)
        common.GAE_LAMBDA = 1.0
        new_samples = trainer.batch_samples(episodes)
        _assert(len(old_samples) == len(new_samples),
                f"ROUND42 batch0 sample count changed across trace values: seed {seed}")
        old_by_key = {(int(s["episode_index"]), int(s["step"])): s for s in old_samples}
        new_by_key = {(int(s["episode_index"]), int(s["step"])): s for s in new_samples}
        _assert(old_by_key.keys() == new_by_key.keys(),
                f"ROUND42 batch0 decision rows changed across trace values: seed {seed}")
        free_rows = []
        for key in sorted(old_by_key):
            old, new = old_by_key[key], new_by_key[key]
            if not old["free"]:
                continue
            free_rows.append({
                "episode_index": key[0], "decision_step": key[1],
                "free_target_ids": sorted(old["free"]),
                "task_advantage_0_95": float(old["advantage"]),
                "task_return_0_95": float(old["return_task"]),
                "positive_cost_return_0_95": float(old["return_cost"]),
                "combined_advantage_1_0": float(new["advantage"]),
                "task_return_1_0": float(new["return_task"]),
                "positive_cost_return_1_0": float(new["return_cost"]),
                "combined_advantage_delta": float(new["advantage"] - old["advantage"]),
                "task_return_delta": float(new["return_task"] - old["return_task"]),
                "positive_cost_return_delta": float(new["return_cost"] - old["return_cost"]),
            })
        _assert(bool(free_rows), f"ROUND42 batch0 has no free one-shot rows: seed {seed}")
        result[str(seed)] = {
            "batch_record_path": str(record_path),
            "batch_record_sha256": _sha256(record_path),
            "preupdate_checkpoint_path": str(pre_path),
            "preupdate_checkpoint_sha256": _sha256(pre_path),
            "critic_parameter_sha256": pre_controller.parameter_summary()["parameter_sha256"],
            "episodes": artifact_inputs,
            "decision_rows": len(new_samples),
            "free_action_decision_rows": len(free_rows),
            "free_action_rows": free_rows,
            "mean_task_return_delta": sum(x["task_return_delta"] for x in free_rows) / len(free_rows),
            "mean_positive_cost_return_delta": sum(
                x["positive_cost_return_delta"] for x in free_rows) / len(free_rows),
            "mean_combined_advantage_delta": sum(
                x["combined_advantage_delta"] for x in free_rows) / len(free_rows),
        }
    common.GAE_LAMBDA = 1.0
    return {
        "kind": "round43-round42-batch0-readonly-recompute/1",
        "status": "passed",
        "critic_values": "recomputed read-only from each frozen ROUND42 pre-update controller and matched to stored tape",
        "reward_streams": "same stored task and positive-cost streams passed through the actual ROUND43 batch_samples function",
        "causal_claim": "none; this is a trace-coefficient counterfactual, not message causal value",
        "by_seed": result,
    }


def _test_version_boundary(runtime: dict[str, Any], frozen: dict[str, Any]
                           ) -> dict[str, Any]:
    trainer = runtime["trainer"]
    versions = runtime["r42_versions"]
    new_freeze = runtime["round42_freeze"]
    import r42_run
    old_index = _read(R42_INDEX_PATH)
    old_path = Path(old_index["new_checkpoints"]["11"]["128"]["path"])
    old_controller, old_runner_identity = r42_run.load_eval_model("OLD", 11)
    _assert(old_controller.parameter_summary()["parameter_sha256"]
            == old_index["new_checkpoints"]["11"]["128"][
                "parameter_identity"]["parameter_sha256"]
            and old_runner_identity["training_version"]
            == frozen["round42"]["source_training_version"]
            and old_runner_identity["optimizer_restored"] is False
            and old_runner_identity["rng_restored"] is False,
            "actual ROUND43 OLD loader did not preserve read-only ROUND42 ep128 identity")
    before_writes = int(versions.CHECKPOINT_WRITE_CALLS)
    original_restore = trainer._restore_controller
    restore_calls = 0

    def counted_restore(*args, **kwargs):
        nonlocal restore_calls
        restore_calls += 1
        return original_restore(*args, **kwargs)

    trainer._restore_controller = counted_restore
    try:
        try:
            trainer.load_checkpoint(old_path, freeze=new_freeze)
        except RuntimeError as error:
            error_text = str(error)
        else:
            raise RuntimeError("ROUND42 trained checkpoint was accepted for ROUND43 resume")
    finally:
        trainer._restore_controller = original_restore
    _assert(restore_calls == 0,
            "cross-version rejection happened after controller/optimizer restore")
    _assert("training_version mismatch" in error_text,
            f"cross-version rejection did not identify its boundary: {error_text}")

    old_controller, _old_state, old_identity = trainer.load_checkpoint_read_only(old_path)
    _assert(old_identity["load_mode"] == "read_only"
            and old_identity["optimizer_restored"] is False
            and old_identity["rng_restored"] is False,
            "ROUND42 model cannot be loaded explicitly read-only")

    ambient = trainer.rng_state()
    controller = runtime["controller_module"].fresh_controller(11)
    optimizer = torch.optim.Adam(controller.parameters, lr=float(runtime["common"].LEARNING_RATE))
    state = trainer.fresh_training_state(11)
    state["episodes_done"] = 0
    state["updates"] = 0
    state["optimizer_steps"] = 0
    state["training_seed"] = 11
    freeze = dict(new_freeze)
    payload = {
        "schema": trainer.STATE_SCHEMA,
        "trainer_tag": trainer.TRAINER_TAG,
        "training_version": versions.TRAINING_VERSION,
        "reward_timing_version": versions.REWARD_TIMING_VERSION,
        "comm_action_eligibility": versions.COMM_ACTION_ELIGIBILITY,
        "comm_action_semantics": versions.COMM_ACTION_SEMANTICS,
        "loss_version": trainer.LOSS_VERSION,
        "gae_schema": trainer.GAE_SCHEMA,
        "optimizer_schema": trainer.OPTIMIZER_SCHEMA,
        "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "freeze": freeze,
        "controller": controller.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_state": state,
        "rng": trainer.rng_state(),
    }
    expected_parameters = controller.parameter_summary()["parameter_sha256"]
    expected_state = json.loads(json.dumps(state))
    expected_optimizer = optimizer.state_dict()
    expected_rng = payload["rng"]
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    try:
        restored_controller, restored_optimizer, restored_state = trainer.load_checkpoint(
            buffer, freeze=freeze)
        _assert(restored_controller.parameter_summary()["parameter_sha256"]
                == expected_parameters, "same-version in-memory restore changed parameters")
        _assert(restored_state == expected_state,
                "same-version in-memory restore changed training cursor/state")
        _assert(restored_optimizer.state_dict() == expected_optimizer,
                "same-version in-memory restore changed the empty optimizer state")
        _assert(trainer.rng_state() == expected_rng,
                "same-version in-memory restore did not restore its exact RNG snapshot")
    finally:
        trainer.restore_rng(ambient)
    _assert(int(versions.CHECKPOINT_WRITE_CALLS) == before_writes,
            "version-boundary preflight wrote a training checkpoint")
    return {
        "kind": "round43-training-version-restore-boundary/1",
        "status": "passed",
        "old_round42_training_restore": "rejected_before_controller_optimizer_or_rng_restore",
        "old_round42_read_only_load": old_identity,
        "old_round42_runner_eval_loader": old_runner_identity,
        "new_round43_same_version_in_memory_restore": "exact parameter/state/optimizer/RNG match",
        "synthetic_checkpoint_file_writes": 0,
        "optimizer_steps": 0,
    }


def run_preflight(*, draft: bool = False) -> dict[str, Any]:
    frozen = _read(FROZEN_PATH)
    artifacts_before = _snapshot_existing_artifacts()
    if not draft:
        verify_frozen_source(frozen)
    round40_path = Path(frozen["execution_identity"]["round40_worktree"]) / "runs" / (
        "comm_first_report_training_ablation_20261003") / "ROUND40_FROZEN_CONFIG.json"
    round40_frozen = _read(round40_path)
    runtime = configure(frozen, round40_frozen["round40"]["freeze_identity"])
    if not draft:
        verify_frozen_source(frozen)
    import r42_run

    prior_ledger = r42_run.ledger_snapshot(strict=True)
    prior_counters = {
        "environment_constructors": prior_ledger["counts"][
            "environment_constructor_calls"],
        "ppo_update_calls": r42_run._PPO_UPDATE_CALLS,
        "optimizer_step_calls": r42_run._OPTIMIZER_STEP_CALLS,
        "checkpoint_write_calls": runtime["r42_versions"].CHECKPOINT_WRITE_CALLS,
        "normal_claims": prior_ledger["counts"]["normal_used"],
        "reserve_claims": prior_ledger["counts"]["reserve_used"],
    }
    recovery_gate = _read(RECOVERY_GATE_PATH)
    recovery_gate_identity = frozen["execution_identity"].get("resume_recovery_gate", {})
    _assert(recovery_gate.get("status") == "passed"
            and _sha256(RECOVERY_GATE_PATH) == recovery_gate_identity.get("sha256")
            and recovery_gate.get("source_code_sha256_for_next_freeze")
            == frozen["execution_identity"].get("source_code_sha256"),
            "ROUND43 exact-resume recovery gate differs from the committed source identity")
    common = runtime["common"]
    _assert(float(common.GAE_LAMBDA) == 1.0 and float(common.GAMMA) == 1.0,
            "active real training path has the wrong GAE/gamma after configuration")
    order = list(common.scenario_order())
    _assert(order == [int(x) for x in frozen["round42"]["training_order"]],
            "active scenario_order differs from ROUND42's registered sequence")
    order_sha = hashlib.sha256(json.dumps(order).encode("utf-8")).hexdigest()
    _assert(order_sha == frozen["round42"]["training_order_sha256"],
            "active scenario_order hash differs from ROUND42")
    frozen_search = frozen["round42"]["frozen_search_allocation_checkpoints"]
    checkpoint_inputs: dict[str, Any] = {}
    for name, record in frozen_search.items():
        path = Path(record["path"])
        _assert(path.is_file() and _sha256(path) == record["sha256"]
                and path.stat().st_size == int(record["size_bytes"]),
                f"frozen search/allocation checkpoint changed: {path}")
        checkpoint_inputs[name] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path), "read_only": True,
        }
    _assert({str(path.resolve()) for path in common.CHECKPOINT_FILES}
            == {str(Path(item["path"]).resolve()) for item in frozen_search.values()},
            "active runner checkpoint path set differs from the frozen search/allocation pair")

    code_status = {
        "real_trainer_module": str(Path(runtime["trainer"].__file__).resolve()),
        "common_module": str(Path(common.__file__).resolve()),
        "active_gae_lambda": float(runtime["trainer"].common.GAE_LAMBDA),
        "active_gamma": float(runtime["trainer"].common.GAMMA),
        "task_and_cost_share_common": runtime["trainer"].common is common,
        "runner_device": str(common.DEVICE),
        "scenario_order_sha256": order_sha,
        "scenario_order_count": len(order),
        "round40_normal_cap": int(runtime["r40_common"].NORMAL_CAP),
        "round43_reserve_cap": int(runtime["r40_common"].RESERVE_CAP),
        "round43_hard_cap": int(runtime["r40_common"].HARD_CAP),
        "frozen_search_allocation_checkpoint_files": checkpoint_inputs,
    }
    import torch
    environment_status = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "execution_device": str(common.DEVICE),
        "cuda_available_but_unused": bool(torch.cuda.is_available()),
    }
    _assert(Path(sys.executable).resolve()
            == Path(frozen["execution_identity"]["python_executable"]).resolve(),
            "active Python executable differs from the prelaunch freeze")
    _assert(code_status["task_and_cost_share_common"],
            "task and cost GAE do not share the active common configuration")
    baselines = audit_reusable_baselines(frozen, runtime)
    batch0 = _batch0_recompute(runtime, frozen)
    ep0_initialization = _test_registered_ep0_initialization(runtime, frozen)
    gae = _test_batch_samples_and_gae(runtime)
    actor = _test_live_actor_path(runtime)
    version = _test_version_boundary(runtime, frozen)
    common.GAE_LAMBDA = 1.0

    after_ledger = r42_run.ledger_snapshot(strict=True)
    after_counters = {
        "environment_constructors": after_ledger["counts"][
            "environment_constructor_calls"],
        "ppo_update_calls": r42_run._PPO_UPDATE_CALLS,
        "optimizer_step_calls": r42_run._OPTIMIZER_STEP_CALLS,
        "checkpoint_write_calls": runtime["r42_versions"].CHECKPOINT_WRITE_CALLS,
        "normal_claims": after_ledger["counts"]["normal_used"],
        "reserve_claims": after_ledger["counts"]["reserve_used"],
    }
    artifacts_after = _snapshot_existing_artifacts()
    _assert(prior_counters == after_counters
            and json.dumps(prior_ledger, sort_keys=True, default=str)
            == json.dumps(after_ledger, sort_keys=True, default=str)
            and artifacts_before == artifacts_after,
            "ROUND43 read-only preflight changed a persisted counter or execution artifact")

    result = {
        "kind": "round43-prelaunch-correctness-gate/2",
        "status": "passed",
        "draft_mode": bool(draft),
        "active_call_path": code_status,
        "frozen_search_allocation_inputs": checkpoint_inputs,
        "runtime_environment": environment_status,
        "reusable_baseline_audit": baselines,
        "round42_batch0_readonly_recompute": batch0,
        "registered_ep0_initialization": ep0_initialization,
        "gae_and_batch_samples_regression": gae,
        "actual_actor_path_regression": actor,
        "version_and_restore_boundary": version,
        "pre_environment_counters": {
            "environment_constructor_calls": 0,
            "environment_starts": 0,
            "ppo_update_calls": 0,
            "optimizer_step_calls": 0,
            "checkpoint_write_calls": 0,
            "meaning": "deltas during this no-environment recovery preflight",
            "cumulative_before_preflight": prior_counters,
            "cumulative_after_preflight": after_counters,
            "synthetic_blocked_actor_step_attempts": actor["guarded_step_boundary_attempts"],
        },
        "resume_recovery_gate": {
            "path": str(RECOVERY_GATE_PATH),
            "sha256": _sha256(RECOVERY_GATE_PATH),
            "status": recovery_gate["status"],
        },
        "persisted_execution_artifacts_unchanged": True,
        "baseline_repair_cells": 0,
        "normal_plan": 480,
        "technical_reserve": 24,
        "hard_cap": 600,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true",
                        help="run all read-only gates before the final source commit")
    args = parser.parse_args()
    result = run_preflight(draft=args.draft)
    prefix = "DRAFT_" if args.draft else ""
    gate_path = RUN_DIR / f"{prefix}ROUND43_PREFLIGHT.json"
    baseline_path = RUN_DIR / f"{prefix}ROUND43_BASELINE_REUSE.json"
    batch0_path = RUN_DIR / f"{prefix}ROUND43_BATCH0_RECOMPUTE.json"
    gae_path = RUN_DIR / f"{prefix}ROUND43_GAE_REGRESSION.json"
    actor_path = RUN_DIR / f"{prefix}ROUND43_ACTOR_PATH_REGRESSION.json"
    version_path = RUN_DIR / f"{prefix}ROUND43_VERSION_BOUNDARY.json"
    ep0_path = RUN_DIR / f"{prefix}ROUND43_EP0_INITIALIZATION.json"
    products = ((gate_path, result),
                (baseline_path, result["reusable_baseline_audit"]),
                (batch0_path, result["round42_batch0_readonly_recompute"]),
                (gae_path, result["gae_and_batch_samples_regression"]),
                (actor_path, result["actual_actor_path_regression"]),
                (version_path, result["version_and_restore_boundary"]),
                (ep0_path, result["registered_ep0_initialization"]))
    for path, value in products:
        body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"preflight evidence already exists with different content: {path}")
        if not path.exists():
            path.write_text(body, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": result["status"],
        "active_gae_lambda": result["active_call_path"]["active_gae_lambda"],
        "active_gamma": result["active_call_path"]["active_gamma"],
        "baseline_reuse": {
            "references": result["reusable_baseline_audit"]["reused"]["immediate_references"]["count"],
            "rules": result["reusable_baseline_audit"]["reused"]["fixed_rules"]["count"],
            "old_evaluations": result["reusable_baseline_audit"]["reused"]["old_round42_new_ep128_evaluations"]["count"],
        },
        "batch0_seeds": sorted(result["round42_batch0_readonly_recompute"]["by_seed"]),
        "pure_optimizer_updates": 0,
        "environment_starts": 0,
        "evidence": [path.name for path, _ in products],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
