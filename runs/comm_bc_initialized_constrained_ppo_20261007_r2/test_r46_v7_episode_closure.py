from __future__ import annotations

import importlib.util
import json
import tempfile
import subprocess
import unittest
from collections import Counter
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("r46_v7_eval_only.py")
SPEC = importlib.util.spec_from_file_location("r46_v7_eval_only_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
V7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V7)

PLACED = tuple(f"piloc-e1-target-{index}" for index in range(6))
COST_FIELDS = (
    "airtime_bytes", "transmission_attempts", "emissions", "packets_lost",
    "packets_delivered", "deliveries_applied", "budget_refused_sends",
    "capacity_refused_sends",
)


def episode_fixture(specs: dict[int, tuple[str, int, float | None]]) -> tuple[dict, list, list, list]:
    """Build an explicit hand-counted window: status, deadline tick, completion tick."""
    lifecycle: list[dict] = []
    status_counts: Counter[str] = Counter()
    completed = expired = active = unassigned = on_time = 0
    task_reward_events: list[dict] = []
    for index, (status, deadline, completion) in specs.items():
        target = PLACED[index]
        lifecycle.append({
            "target_id": target, "event_type": "detection", "timestamp": 1.0,
            "status": "detected", "detail": {"deadline_offset": float(deadline - 1)},
        })
        if completion is not None:
            lifecycle.append({
                "target_id": target, "event_type": "progress", "timestamp": float(completion),
                "status": "completed",
            })
            completed += 1
            is_on_time = completion <= deadline
            if is_on_time:
                on_time += 1
            task_reward_events.append({
                "event_id": f"reward:{target}", "task_id": target,
                "credited": is_on_time, "on_time": is_on_time,
            })
        elif deadline <= 1000:
            expired += 1
        elif status == "detected":
            unassigned += 1
        else:
            active += 1
            lifecycle.append({
                "target_id": target, "event_type": "assignment", "timestamp": 20.0,
            })
        lifecycle.append({
            "target_id": target, "event_type": "final", "timestamp": 1000.0,
            "status": status,
        })
        status_counts[status] += 1

    discovered = len(specs)
    metrics = {
        "steps": 1000,
        "targets_discovered": discovered,
        "targets_active": active + unassigned,
        "remaining_search_targets": len(PLACED) - discovered,
        "status_counts": dict(status_counts),
    }
    lifecycle.append({
        "target_id": "__run__", "event_type": "final", "timestamp": 1000.0,
        "status": "run_finished", "detail": {"step": 1000, "metrics": metrics},
    })
    cost_stream = {str(tick): {field: 0 for field in COST_FIELDS}
                   for tick in range(1, 1001)}
    row = {
        "steps": 1000, "completed": completed, "expired": expired,
        "active": active, "unassigned": unassigned, "discovered": discovered,
        "on_time": on_time, "task_reward_total": on_time,
        "task_reward_events": task_reward_events, "cost_stream": cost_stream,
        "relay_counters": {field: 0 for field in COST_FIELDS},
        "spent_bytes": 0, "airtime_tap_total": 0,
        "collector_summary": {
            "decision_steps": 0,
            "one_shot_counts": {"free_choices": 0, "send_now_choices": 0,
                                 "skip_episode_choices": 0},
        },
    }
    return row, lifecycle, [], []


class EpisodeClosureTests(unittest.TestCase):
    def close(self, fixture):
        row, lifecycle, transcript, tape = fixture
        return V7.validate_episode_closure(
            row, lifecycle, transcript, tape, placed_target_ids=PLACED)

    def test_six_completed_targets_close_without_becoming_an_adoption_threshold(self):
        fixture = episode_fixture({i: ("completed", 900, 100 + i) for i in range(6)})
        result = self.close(fixture)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["terminal_task_count_description"], 6)
        self.assertEqual(result["window_task_counts"]["on_time"], 6)

    def test_full_window_with_active_and_never_discovered_target_is_valid(self):
        fixture = episode_fixture({
            0: ("completed", 900, 100),
            1: ("completed", 900, 200),
            2: ("completed", 100, 200),  # completed late; still a valid completion
            3: ("expired", 100, None),
            4: ("in_progress", 1200, None),
            # target 5 remains placed and unseen in the search side
        })
        result = self.close(fixture)
        self.assertEqual(result["placed_target_count"], 6)
        self.assertEqual(result["registry_target_count"], 5)
        self.assertEqual(result["never_discovered_count"], 1)
        self.assertEqual(result["window_task_counts"], {
            "discovered": 5, "completed": 3, "expired": 1,
            "active": 1, "unassigned": 0, "on_time": 2,
        })

    def test_full_window_with_unassigned_discovered_target_is_valid(self):
        fixture = episode_fixture({
            0: ("completed", 900, 100),
            1: ("expired", 100, None),
            2: ("detected", 1200, None),
            # targets 3..5 are never discovered
        })
        result = self.close(fixture)
        self.assertEqual(result["window_task_counts"], {
            "discovered": 3, "completed": 1, "expired": 1,
            "active": 0, "unassigned": 1, "on_time": 1,
        })
        self.assertEqual(result["never_discovered_count"], 3)

    def test_missing_run_final_is_rejected(self):
        fixture = episode_fixture({i: ("completed", 900, 100) for i in range(6)})
        fixture[1][:] = [x for x in fixture[1] if x.get("target_id") != "__run__"]
        with self.assertRaisesRegex(V7.V7Error, "exactly one run final"):
            self.close(fixture)

    def test_duplicate_registry_final_identity_is_rejected(self):
        fixture = episode_fixture({i: ("completed", 900, 100) for i in range(6)})
        duplicate = dict(next(x for x in fixture[1]
                              if x.get("event_type") == "final"
                              and x.get("target_id") == PLACED[0]))
        fixture[1].append(duplicate)
        with self.assertRaisesRegex(V7.V7Error, "duplicate target identity"):
            self.close(fixture)

    def test_nonclosing_status_counters_are_rejected(self):
        fixture = episode_fixture({
            0: ("completed", 900, 100), 1: ("in_progress", 1200, None),
        })
        fixture[0]["active"] = 0
        with self.assertRaisesRegex(V7.V7Error, "row window state counters"):
            self.close(fixture)

    def test_registry_status_census_mismatch_is_rejected(self):
        fixture = episode_fixture({0: ("completed", 900, 100)})
        run_final = next(x for x in fixture[1] if x.get("target_id") == "__run__")
        run_final["detail"]["metrics"]["status_counts"] = {"completed": 0}
        with self.assertRaisesRegex(V7.V7Error, "registry status counts"):
            self.close(fixture)

    def test_adoption_identity_is_performance_neutral(self):
        candidate = {
            "key": list(V7.OLD_KEY),
            "claim_sha256": V7.OLD_CLAIM_SHA,
            "row_sha256": V7.OLD_ROW_SHA,
            "lifecycle_sha256": V7.OLD_ARTIFACT_SHAS["lifecycle"],
            "transcript_sha256": V7.OLD_ARTIFACT_SHAS["transcript"],
            "tape_sha256": V7.OLD_ARTIFACT_SHAS["tape"],
            "model_file_sha256": V7.MODEL_FILE_SHA,
            "model_parameter_sha256": V7.MODEL_PARAMETER_SHA,
            "old_executor_sha256": V7.OLD_EXECUTOR_SHA,
            "old_source_head": V7.OLD_HEAD,
            "old_input_head": V7.OLD_INPUT_HEAD,
            "old_claim_status": "failed_technical",
            "old_claim_category": "normal", "old_slot": 386,
            "row_identity": {
                "model_arm": "NEW", "training_seed": 11, "eval_mode": "greedy",
                "scenario_seed": 1_301_000, "attempt": 1, "steps": 1000,
            },
            "episode_final_status": "passed", "event_closure_status": "passed",
            "cost_closure_status": "passed", "spent_bytes": 9798,
            "cost_stream_airtime_bytes": 9798, "model_load_mode": "read_only",
            "optimizer_restored": False, "rng_restored": False,
            "runtime_forward_status": "passed", "configuration_status": "passed",
            "source_gate_status": "passed", "performance_used_for_adoption": False,
            "placed_target_count": 6, "registry_target_count": 6,
            "never_discovered_count": 0,
        }
        for on_time, expired in ((4, 2), (6, 0)):
            with self.subTest(on_time=on_time, expired=expired):
                candidate["on_time_count"] = on_time
                candidate["expired_count"] = expired
                decision = V7.validate_adoption_manifest(candidate)
                self.assertEqual(decision["status"], "accepted")
                self.assertFalse(decision["performance_used_for_adoption"])


class HistoricalSourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = MODULE_PATH.parents[2]
        cls.repo = repo
        prefix = "92bf455ea3a85cae727110a9552e92a5804bcd25:runs/comm_bc_initialized_constrained_ppo_20261007_r2/"

        def git_object(name: str) -> bytes:
            return subprocess.run(
                ["git", "-C", str(repo), "show", prefix + name],
                check=True, capture_output=True,
            ).stdout

        cls.gate_bytes = git_object("ROUND46_R2_V6_PRELAUNCH_GATE.json")
        cls.gate = json.loads(cls.gate_bytes)
        cls.preflight_head = cls.gate["head_at_preflight"]

        def preflight_object(name: str) -> bytes:
            return subprocess.run(
                ["git", "-C", str(repo), "show",
                 f"{cls.preflight_head}:runs/comm_bc_initialized_constrained_ppo_20261007_r2/{name}"],
                check=True, capture_output=True,
            ).stdout

        cls.executor = preflight_object("r46_execute_r2.py")
        cls.frozen = preflight_object("ROUND46_R2_V6_FROZEN_CONFIG.json")
        cls.inputs = preflight_object("ROUND46_R2_V6_INPUT_IDENTITY.json")
        cls.synthetic_executor = b"registered executor source bytes\n"
        cls.synthetic_frozen = b"registered original freeze bytes\n"
        cls.synthetic_inputs = b'{"registered_input":"source"}\n'
        cls.synthetic_gate = {
            "status": "passed",
            "branch": "fix/comm-bc-initialization-readback",
            "head_at_preflight": "8275ff9290a265521bf91efc7a365a0b640f587d",
            "runner_source_sha256": V7.sha256_bytes(cls.synthetic_executor),
            "frozen_config_sha256": V7.sha256_bytes(cls.synthetic_frozen),
            "input_identity_sha256": V7.sha256_bytes(cls.synthetic_inputs),
        }

    def test_historical_and_current_branches_are_separate_and_both_valid(self):
        audit = V7.validate_old_source_gate(
            self.synthetic_gate, self.synthetic_executor,
            self.synthetic_frozen, self.synthetic_inputs,
            current_branch=V7.BRANCH,
            preflight_head=self.synthetic_gate["head_at_preflight"],
            actual_input_head=V7.OLD_INPUT_HEAD,
            expected_executor_sha=V7.sha256_bytes(self.synthetic_executor),
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["historical_branch"], "fix/comm-bc-initialization-readback")
        self.assertEqual(audit["current_v7_branch"], V7.BRANCH)

    def test_actual_old_git_objects_report_the_stale_freeze_identity(self):
        audit = V7.verify_old_source_identity(self.repo)
        self.assertEqual(audit["status"], "failed_source_authenticity")
        self.assertEqual(audit["historical_branch"], "fix/comm-bc-initialization-readback")
        # The authorized eval-only lineage continues on the V8 branch; the audit
        # must still report the actual current branch of this task family.
        self.assertIn(audit["current_v7_branch"],
                      (V7.BRANCH, "fix/round46-r2-v8-eval-booking-close",
                       "fix/round46-r2-v9-eval-runtime-close"))
        self.assertFalse(audit["checks"]["original_frozen_config"])
        self.assertFalse(audit["checks"]["original_input_identity"])
        self.assertEqual(audit["preflight_git_object_sha256"]["frozen_config"],
                         V7.sha256_bytes(self.frozen))

    def test_wrong_old_source_or_current_branch_is_rejected(self):
        with self.assertRaises(V7.V7Error):
            V7.validate_old_source_gate(
                self.synthetic_gate, self.synthetic_executor,
                self.synthetic_frozen, self.synthetic_inputs,
                current_branch=V7.BRANCH, source_head="0" * 40,
                preflight_head=self.synthetic_gate["head_at_preflight"],
                actual_input_head=V7.OLD_INPUT_HEAD,
                expected_executor_sha=V7.sha256_bytes(self.synthetic_executor),
            )
        with self.assertRaises(V7.V7Error):
            V7.validate_old_source_gate(
                self.synthetic_gate, self.synthetic_executor,
                self.synthetic_frozen, self.synthetic_inputs,
                current_branch="fix/comm-bc-initialization-readback",
                preflight_head=self.synthetic_gate["head_at_preflight"],
                actual_input_head=V7.OLD_INPUT_HEAD,
                expected_executor_sha=V7.sha256_bytes(self.synthetic_executor),
            )

    def test_executor_freeze_and_input_object_mismatches_are_rejected(self):
        variants = (
            (self.executor + b"x", self.frozen, self.inputs),
            (self.executor, self.frozen + b"x", self.inputs),
            (self.executor, self.frozen, self.inputs + b"x"),
        )
        for executor, frozen, inputs in variants:
            with self.subTest(
                executor=executor != self.executor,
                frozen=frozen != self.frozen,
                inputs=inputs != self.inputs,
            ):
                with self.assertRaises(V7.V7Error):
                    V7.validate_old_source_gate(
                        self.synthetic_gate, executor, frozen, inputs,
                        current_branch=V7.BRANCH,
                        preflight_head=self.synthetic_gate["head_at_preflight"],
                        actual_input_head=V7.OLD_INPUT_HEAD,
                        expected_executor_sha=V7.sha256_bytes(self.synthetic_executor))


class QuotaSeamTests(unittest.TestCase):
    def test_fixed_adoption_and_retry_plans_have_exact_capacity(self):
        adoption = V7.build_quota_plan(adoption_passed=True)
        retry = V7.build_quota_plan(adoption_passed=False)
        self.assertEqual((len(adoption["entries"]), adoption["new_claim_count"]), (95, 95))
        self.assertEqual(sum(x["category"] == "normal" for x in adoption["entries"]), 94)
        self.assertEqual(sum(x["category"] == "reserve" for x in adoption["entries"]), 1)
        self.assertEqual(adoption["adopted_key"], list(V7.OLD_KEY))
        self.assertEqual((len(retry["entries"]), retry["new_claim_count"]), (96, 96))
        self.assertEqual(sum(x["category"] == "normal" for x in retry["entries"]), 94)
        self.assertEqual(sum(x["category"] == "reserve" for x in retry["entries"]), 2)
        self.assertEqual(retry["entries"][0]["key"], list(V7.OLD_KEY))
        self.assertEqual((retry["entries"][0]["attempt"], retry["entries"][0]["slot"]), (2, 1))
        self.assertEqual((retry["entries"][-1]["category"], retry["entries"][-1]["slot"]),
                         ("reserve", 2))

    def test_real_r40_claim_seam_books_retry_normal_and_capacity_reserve_once(self):
        source = MODULE_PATH.parents[2] / "runs/comm_first_report_training_ablation_20261003/r40_common.py"
        spec = importlib.util.spec_from_file_location("r40_common_v7_quota_test", source)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        r40 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r40)
        plan = V7.build_quota_plan(adoption_passed=False)
        with tempfile.TemporaryDirectory(dir=V7.HERE) as directory:
            root = Path(directory)
            (root / "launch_slots" / "normal").mkdir(parents=True)
            (root / "launch_slots" / "reserve").mkdir(parents=True)
            old = {name: getattr(r40, name) for name in (
                "ROUND_DIR", "NORMAL_CAP", "RESERVE_CAP", "HARD_CAP",
                "_reserve_slot", "claim_launch", "next_attempt",
            )}
            try:
                r40.ROUND_DIR = root
                r40.NORMAL_CAP, r40.RESERVE_CAP, r40.HARD_CAP = 480, 24, 504
                adapter = V7.install_quota_adapter(
                    {"r40_common": r40}, plan, output_root=root)
                retry = plan["entries"][0]
                retry_logical = retry["logical_job"]
                self.assertEqual(r40.next_attempt(retry_logical), 2)
                retry_claim_path = r40.claim_launch(
                    retry_logical, category="reserve", attempt=2,
                    metadata={"retry_of": "preserved attempt 1"})
                retry_claim = V7.read_json(retry_claim_path)
                self.assertEqual((retry_claim["status"], retry_claim["category"],
                                  retry_claim["launch_slot"]),
                                 ("claimed_before_environment", "reserve", 1))

                normal = plan["entries"][1]
                self.assertEqual(r40.next_attempt(normal["logical_job"]), 1)
                normal_claim_path = r40.claim_launch(
                    normal["logical_job"], category="normal", attempt=1)
                normal_claim = V7.read_json(normal_claim_path)
                self.assertEqual((normal_claim["category"], normal_claim["launch_slot"]),
                                 ("normal", 387))

                capacity = plan["entries"][-1]
                capacity_claim_path = r40.claim_launch(
                    capacity["logical_job"], category="normal", attempt=1)
                capacity_claim = V7.read_json(capacity_claim_path)
                self.assertEqual((capacity_claim["category"], capacity_claim["launch_slot"]),
                                 ("reserve", 2))

                with self.assertRaises(RuntimeError):
                    r40.claim_launch(retry_logical, category="reserve", attempt=2)
                self.assertEqual(len(adapter["claim_records"]), 3)
                self.assertEqual(len(list((root / "launch_claims").glob("*/*.json"))), 3)
                self.assertEqual(len(list((root / "launch_slots" / "normal").glob("*.json"))), 1)
                self.assertEqual(len(list((root / "launch_slots" / "reserve").glob("*.json"))), 2)
            finally:
                for name, value in old.items():
                    setattr(r40, name, value)


class V7FreezeGateSeamTests(unittest.TestCase):
    """The zero-environment regression field must be a live freeze-gate check."""

    def setUp(self):
        self._old_v7_root = V7.V7_ROOT
        self._old_protected = V7.PROTECTED
        self._tmp = tempfile.TemporaryDirectory(dir=V7.HERE)
        root = Path(self._tmp.name)
        self.sandbox = root / "sandbox_repo"
        self.sandbox.mkdir()
        self.v7_root = root / "v7_eval_only"
        self.v7_root.mkdir()

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(self.sandbox), "-c", "user.email=v7@test",
                 "-c", "user.name=v7", "-c", "commit.gpgsign=false", *args],
                check=True, capture_output=True, text=True).stdout.strip()

        git("init", "-b", V7.BRANCH)
        git("commit", "--allow-empty", "-m", "frozen code identity")
        self.source_commit = git("rev-parse", "HEAD")

        V7.V7_ROOT = self.v7_root
        V7.PROTECTED = {}
        self.journal = root / "checkpoint_journal.jsonl"
        self.journal.write_bytes(b"{}\n")
        self.base = type("Base", (), {"REPO": self.sandbox,
                                      "CHECKPOINT_JOURNAL_PATH": self.journal})()
        self.adapter_sha = V7.sha256_file(MODULE_PATH)

    def tearDown(self):
        V7.V7_ROOT = self._old_v7_root
        V7.PROTECTED = self._old_protected
        self._tmp.cleanup()

    def _write_freeze(self, *, regression_field):
        freeze = {
            "task_id": V7.TASK_ID, "status": "frozen_before_environment",
            "source_commit": self.source_commit, "quota_branch": "retry_attempt2",
            "new_claim_count": 96, "checkpoint_files": {},
        }
        plan = {"branch": "retry_attempt2", "new_claim_count": 96}
        gate = {
            "task_id": V7.TASK_ID, "status": "passed",
            "frozen_config_sha256": V7.sha256_bytes(
                (json.dumps(freeze, ensure_ascii=False, indent=2) + "\n").encode("utf-8")),
            "quota_plan_sha256": V7.sha256_bytes(
                (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")),
            "quota_plan_canonical_sha256": V7.sha256_bytes(
                V7.canonical_json_bytes(plan)),
            "adapter_source_sha256": self.adapter_sha,
            "source_hashes": {"r46_v7_eval_only.py": self.adapter_sha},
            "checkpoint_journal_sha256": V7.sha256_file(self.journal),
        }
        if regression_field is not None:
            gate["zero_environment_regression"] = regression_field
        freeze_bytes = (json.dumps(freeze, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        gate_bytes = (json.dumps(gate, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        (self.v7_root / "ROUND46_R2_V7_FROZEN_CONFIG.json").write_bytes(freeze_bytes)
        (self.v7_root / "ROUND46_R2_V7_QUOTA_PLAN.json").write_bytes(plan_bytes)
        (self.v7_root / "ROUND46_R2_V7_PRELAUNCH_GATE.json").write_bytes(gate_bytes)

    def test_gate_without_zero_environment_regression_is_rejected(self):
        self._write_freeze(regression_field=None)
        with self.assertRaises(V7.V7Error):
            V7._verify_v7_freeze(self.base, {}, require_gate=True)

    def test_gate_with_unpassed_regression_is_rejected(self):
        self._write_freeze(regression_field="failed")
        with self.assertRaises(V7.V7Error):
            V7._verify_v7_freeze(self.base, {}, require_gate=True)

    def test_gate_with_passed_regression_and_consistent_identity_passes(self):
        self._write_freeze(regression_field="passed")
        V7._verify_v7_freeze(self.base, {}, require_gate=True)

    def test_require_gate_false_bypasses_only_the_regression_field(self):
        self._write_freeze(regression_field=None)
        V7._verify_v7_freeze(self.base, {}, require_gate=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
