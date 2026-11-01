import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from stage5b_comm import recovery_descendant as D
from stage5b_comm import recovery_protocol as R
from stage5b_comm import recovery_runtime as RT


class ReconstructedParentProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root).resolve()
        run_dir = root / "execution/recovery_train_run_01"
        rid = "recovery-train-4321-1793210000000000000"
        argv = ["stage5b.py", "recovery-train", "--root", str(root), "--workers", "3"]
        base = {"path": str(root / "protocol/RECOVERY_RUNTIME_CONTEXT.json"),
                "bytes": 12, "sha256": "a" * 64}
        g3 = {"path": str(root / "execution/recovery_runtime/G3_REPORT.json"),
              "bytes": 13, "sha256": "b" * 64}
        auth = {"path": str(root / "execution/recovery_world_authorization_01/WORLD_AUTHORIZATION.json"),
                "bytes": 14, "sha256": "c" * 64}
        source = {"path": str(root / "protocol/RECOVERY_SOURCE_MANIFEST.json"),
                  "bytes": 15, "sha256": "d" * 64}
        slot_plan = {"path": str(root / "protocol/RECOVERY_WORLD_SLOT_PLAN.json"),
                     "bytes": 16, "sha256": "e" * 64}
        identity = {"path": str(root / "protocol/RECOVERY_EXPERIMENT_IDENTITY.json"),
                    "bytes": 17, "sha256": "f" * 64}
        filenames = ["stage5b.py", "recovery_runtime.py", "runner.py", "trainer.py", "guard.py"]
        code = [{"category": "code", "path": str(root / ("stage5b.py" if name == "stage5b.py"
                    else "stage5b_comm/" + name)), "bytes": 100 + i,
                 "sha256": str(i + 1) * 64} for i, name in enumerate(filenames)]
        context = SimpleNamespace(
            is_descendant=True, stage_index=2, context_pin=base,
            payload={"source_manifest": source, "slot_plan": slot_plan, "identity": identity},
            source_manifest={"files": code},
            runs=[{"arm": arm, "training_seed": seed, "active": True,
                   "start_episode": 1000, "end_episode_cap": 1120}
                  for arm, seed, _ in R.RUN_CURSORS])
        bindings = {"base_context": base, "g3_report": g3, "world_authorization": auth}
        started = {"schema": "stage5b-phase-receipt/1", "phase": "recovery-train",
                   "stage": "started", "receipt_id": rid, "argv": argv,
                   "detail": {"execution_bindings": bindings}}
        results = [{"slot": "recovery:%s:%d" % (arm, seed), "status": "ok",
                    "episodes": 120, "terminal_episode": 1120, "converged": False}
                   for arm, seed, _ in R.RUN_CURSORS]
        detail = {"status": "FINITE_RECOVERY_STAGE_COMPLETE", "worlds": 720,
                  "slot_counts": {"claimed": 720, "ok": 720, "results": 720},
                  "execution_bindings": bindings, "results": results}
        receipt = {"schema": "stage5b-phase-receipt/1", "phase": "recovery-train",
                   "stage": "terminal", "receipt_id": rid, "returncode": 0,
                   "argv": argv, "detail": detail}
        observer = {"schema": "stage5b-training-observer-preparation/1", "tool_session_id": 9876,
                    "train_started_receipt": {"receipt_id": rid, "argv": argv}}
        preceding = {"tool_return": {"session_id": 9876, "chunk_id": "live-6",
                            "output": ""}}
        poll = {"observed_utc": "2026-10-29T01:00:00Z",
                "tool_return": {"chunk_id": "terminal-1",
                                "exit_code": 0, "output": ""}}
        marker = {"schema": D.GENERIC_RECONSTRUCTION_SCHEMA,
                  "status": "POST_RUN_RECONSTRUCTION_NOT_CONTEMPORANEOUS"}
        summary = {"schema": "stage5b-recovery-train-terminal-summary/1",
                   "reconstruction_basis": marker, "stage_complete": True,
                   "global_stop": False, "convergence_complete": False,
                   "claims": 720, "results": 720, "update_journals": 180,
                   "checkpoints": [{"run": "%s_%d" % (a, s), "episode": 1120,
                                    "bytes": 100, "sha256": "1" * 64}
                                   for a, s, _ in R.RUN_CURSORS],
                   "certificates": [{"run": "%s_%d" % (a, s), "episode": 1120,
                                     "bytes": 100, "sha256": "2" * 64}
                                    for a, s, _ in R.RUN_CURSORS]}
        expected = {Path(x["path"]).resolve().relative_to(root).as_posix():
                    (x["bytes"], x["sha256"]) for x in code}
        for pin in (source, slot_plan, identity, base, g3, auth):
            expected[Path(pin["path"]).resolve().relative_to(root).as_posix()] = (pin["bytes"], pin["sha256"])
        before = {"reconstruction_basis": marker,
                  "files": [{"path": path, "bytes": size, "sha256": digest}
                            for path, (size, digest) in expected.items()]}
        after = {"reconstruction_basis": marker, "files": copy.deepcopy(before["files"]),
                 "source_context_auth_drift": "UNKNOWN_NOT_COMPUTABLE_FROM_POSTHOC_PINS"}
        poll_pin = {"path": str(root / "execution/recovery_train_observer_20261029/" 
                                  "poll_0007_terminal.json"), "bytes": 200, "sha256": "3" * 64}
        previous_pin = {"path": str(root / "execution/recovery_train_observer_20261029/"
                                      "poll_0006_live.json"), "bytes": 190, "sha256": "4" * 64}
        pins = {"observer_preparation": {"path": str(root / "execution/recovery_train_observer_20261029/observer_preparation.json"),
                                     "bytes": 180, "sha256": "5" * 64},
                "terminal_poll": poll_pin, "preceding_live_poll": previous_pin}
        provenance = {"schema": D.GENERIC_RECONSTRUCTION_SCHEMA,
                      "status": "POSTHOC_EVIDENCE_INDEX", "parent_root": str(root),
                      "run_directory": str(run_dir), "receipt_id": rid,
                      "observer_mode": D.PREPARED_OBSERVER_MODE,
                      "training_interpreter_history": D.TRAINING_INTERPRETER_UNKNOWN,
                      "training_session_id": 9876,
                      "pins": pins}
        cli = {"phase": "recovery-train", "returncode": 0, "receipt_id": rid}
        stdout = json.dumps(cli) + "\n"
        return (root, run_dir, context, provenance, started, receipt, observer, preceding, poll,
                stdout, summary, before, after, pins)

    def validate(self, fixture):
        provenance = fixture[3]
        return D.validate_reconstructed_parent_provenance(
            *fixture, observer_mode=provenance["observer_mode"],
            training_interpreter_history=provenance["training_interpreter_history"],
            training_session_id=provenance["training_session_id"])

    def test_real_poll_shape_and_posthoc_pins_preserve_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.validate(self.fixture(tmp))
        detail, bindings, checkpoints, certificates, provenance = result
        self.assertEqual(detail["worlds"], 720)
        self.assertEqual(len(checkpoints), len(certificates), 6)
        self.assertEqual(provenance["outer_tool_exit"]["raw_tool_return"],
                         "PERSISTED_IN_PINNED_TERMINAL_POLL")
        self.assertEqual(provenance["source_context_auth_drift"],
                         "UNKNOWN_NOT_COMPUTABLE_FROM_POSTHOC_PINS")
        self.assertEqual(provenance["child_exit_code"], "UNKNOWN_NOT_SEPARATELY_RECORDED")
        self.assertEqual(provenance["terminal_checkpoint_payload_to_final_state"],
                         "UNKNOWN_NEXT_G3_REQUIRED")

    def test_nonzero_or_unrelated_terminal_poll_and_receipt_mismatch_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.fixture(tmp)
            for mutate in (
                lambda f: f[8]["tool_return"].update(exit_code=1),
                lambda f: f[7]["tool_return"].update(session_id=777),
                lambda f: f[5].update(receipt_id="recovery-train-other"),
                lambda f: f.__setitem__(9, f[9].replace('"returncode": 0', '"returncode": 1')),
            ):
                bad = list(copy.deepcopy(base))
                mutate(bad)
                with self.assertRaises(RT.RecoveryRuntimeError):
                    self.validate(bad)

    def test_unmarked_or_drifted_posthoc_source_pins_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.fixture(tmp)
            for mutate in (
                lambda f: f[11]["reconstruction_basis"].update(status="CONTEMPORANEOUS"),
                lambda f: f[12].update(source_context_auth_drift=[]),
                lambda f: f[11]["files"][0].update(sha256="0" * 64),
            ):
                bad = copy.deepcopy(base)
                mutate(bad)
                with self.assertRaises(RT.RecoveryRuntimeError):
                    self.validate(bad)


    def observerless_fixture(self, root):
        fixture = list(self.fixture(root))
        fixture[6] = None
        observer_dir = Path(root).resolve() / "execution/recovery_train_observer_01"
        previous_pin = {"path": str(observer_dir / "poll_raw_098.json"),
                        "bytes": 178, "sha256": "4" * 64}
        poll_pin = {"path": str(observer_dir / "poll_raw_099_terminal.json"),
                    "bytes": 259, "sha256": "3" * 64}
        pins = {"observer_preparation": None, "preceding_live_poll": previous_pin,
                "terminal_poll": poll_pin}
        fixture[7] = {"chunk_id": "463b58", "session_id": 38844,
                     "output": "actor_type:discrete\r\n"}
        fixture[8] = {"chunk_id": "afa5c2", "exit_code": 0,
                     "output": fixture[9]}
        provenance = copy.deepcopy(fixture[3])
        provenance.update({"observer_mode": D.OBSERVERLESS_POSTHOC_MODE,
                           "training_interpreter_history": D.TRAINING_INTERPRETER_UNKNOWN,
                           "training_session_id": 38844,
                           "pins": pins})
        fixture[3], fixture[13] = provenance, pins
        return tuple(fixture)

    def test_observerless_posthoc_poll_shape_preserves_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.validate(self.observerless_fixture(tmp))
        detail, _bindings, checkpoints, certificates, provenance = result
        self.assertEqual(detail["worlds"], 720)
        self.assertEqual(len(checkpoints), len(certificates), 6)
        self.assertEqual(provenance["observer_preparation"], "UNKNOWN_NOT_CAPTURED")
        self.assertEqual(provenance["training_interpreter_history"],
                         D.TRAINING_INTERPRETER_UNKNOWN)
        self.assertEqual(provenance["terminal_poll_session_id"], 38844)

    def test_observerless_mode_rejects_missing_markers_wrong_pins_or_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.observerless_fixture(tmp)
            mutations = (
                lambda f: f[3].update(observer_mode=D.PREPARED_OBSERVER_MODE),
                lambda f: f[3].update(training_interpreter_history="RECORDED"),
                lambda f: f[13].update(observer_preparation={"path": "not-null"}),
                lambda f: f[8].update(exit_code=1),
                lambda f: f[7].update(session_id=777),
                lambda f: f[3]["pins"]["terminal_poll"].update(
                    path=str(Path(f[0]) / "execution/recovery_train_observer_01/poll_raw_100_terminal.json")),
            )
            for mutate in mutations:
                bad = list(copy.deepcopy(base))
                mutate(bad)
                with self.assertRaises(RT.RecoveryRuntimeError):
                    self.validate(bad)

if __name__ == "__main__":
    unittest.main()
