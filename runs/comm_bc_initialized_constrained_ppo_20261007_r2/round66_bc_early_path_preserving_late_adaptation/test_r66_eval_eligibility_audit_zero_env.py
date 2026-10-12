"""Zero-environment tests for r66_eval_eligibility_audit.py.

No environment is constructed, no optimizer runs, nothing is written outside a per-test scratch
directory UNDER THE ROUND DIR (the sandbox AC temp rejects rmtree cleanup, so OS temp is avoided).
Each _row() call namespaces its artifact files uniquely, so two rows in one scratch dir never
clobber each other. These tests pin the run-class-scoped companion contract the decision side
required, including the counter-examples the adapter's broad companion rule would have let through:
  * a MODEL row missing its tape does NOT pass;
  * a MODEL row missing its lifecycle does NOT pass;
  * a MODEL tape whose line count != decision_steps does NOT pass;
  * a zero-byte companion is reported distinctly from a missing one;
  * a collector-less RULE row with an absent tape DOES pass.
"""
import contextlib
import hashlib
import itertools
import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r66_eval_eligibility_audit as aud  # noqa: E402

_ROW_SEQ = itertools.count()


@contextlib.contextmanager
def _scratch_dir():
    base = HERE / "_test_scratch"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"t_{uuid.uuid4().hex[:12]}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield str(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _w(path, text=""):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return str(path)


def _artefact(tmp, name, spec):
    """spec: "present" | "zero" | "missing" | ("lines", n) | None (not declared)."""
    p = os.path.join(tmp, name)
    if spec is None:
        return None
    if spec == "missing":
        return p
    if spec == "zero":
        return _w(p, "")
    if isinstance(spec, tuple) and spec[0] == "lines":
        return _w(p, "\n".join("x" for _ in range(spec[1])) + "\n")
    return _w(p, "content\n")


def _row(tmp, *, run="model", ds=3, tape=("lines", 3), lifecycle="present",
         transcript="present", arm="NEW", seed=11, mode="greedy", scene=1800000):
    tag = f"r{next(_ROW_SEQ)}_"
    art = {"lifecycle": _artefact(tmp, tag + "lifecycle.jsonl", lifecycle),
           "transcript": _artefact(tmp, tag + "transcript.jsonl", transcript),
           "tape": _artefact(tmp, tag + "tape.jsonl", tape)}
    cs = {"decision_steps": ds} if run == "model" else None
    if run == "unknown":
        cs = "corrupt-not-a-dict"
    return {"model_arm": arm, "training_seed": (None if run == "rule" else seed),
            "eval_mode": mode, "scenario_seed": scene, "steps": 1000, "on_time": 3,
            "collector_summary": cs, "artefacts": art}


class RunClassContract(unittest.TestCase):
    def test_model_sound_tape_passes(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="model", ds=3, tape=("lines", 3)))
            self.assertEqual(ev["run_class"], "model")
            self.assertTrue(ev["eligible"], ev["reasons"])

    def test_model_missing_tape_fails(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="model", ds=3, tape="missing"))
            self.assertFalse(ev["eligible"])
            self.assertTrue(any("tape" in r and "missing" in r for r in ev["reasons"]), ev["reasons"])

    def test_model_missing_lifecycle_fails(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="model", lifecycle="missing"))
            self.assertFalse(ev["eligible"])
            self.assertTrue(any("lifecycle" in r and "missing" in r for r in ev["reasons"]), ev["reasons"])

    def test_model_missing_transcript_fails(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="model", transcript="missing"))
            self.assertFalse(ev["eligible"])
            self.assertTrue(any("transcript" in r for r in ev["reasons"]), ev["reasons"])

    def test_model_tape_line_mismatch_fails(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="model", ds=3, tape=("lines", 2)))
            self.assertFalse(ev["eligible"])
            self.assertTrue(any("decision_steps" in r for r in ev["reasons"]), ev["reasons"])

    def test_model_no_decision_steps_fails(self):
        with _scratch_dir() as t:
            row = _row(t, run="model", ds=3, tape=("lines", 3))
            row["collector_summary"] = {}  # model run but no integer decision_steps
            ev = aud.check_row_evidence(row)
            self.assertFalse(ev["eligible"])
            self.assertTrue(any("decision_steps" in r for r in ev["reasons"]), ev["reasons"])

    def test_zero_byte_distinct_from_missing(self):
        with _scratch_dir() as t:
            zero = aud.check_row_evidence(_row(t, run="model", lifecycle="zero"))
            miss = aud.check_row_evidence(_row(t, run="model", lifecycle="missing"))
            self.assertEqual(zero["artifacts"]["lifecycle"], "zero_byte")
            self.assertEqual(miss["artifacts"]["lifecycle"], "missing")
            self.assertNotEqual(zero["artifacts"]["lifecycle"], miss["artifacts"]["lifecycle"])
            self.assertFalse(zero["eligible"])
            self.assertFalse(miss["eligible"])

    def test_rule_tape_absent_passes(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="rule", tape="missing"))
            self.assertEqual(ev["run_class"], "rule")
            self.assertTrue(ev["eligible"], ev["reasons"])
            self.assertEqual(ev["artifacts"]["tape"], "missing")

    def test_rule_tape_not_declared_passes(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="rule", tape=None))
            self.assertTrue(ev["eligible"], ev["reasons"])
            self.assertEqual(ev["artifacts"]["tape"], "not_declared")

    def test_rule_tape_zero_byte_fails(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="rule", tape="zero"))
            self.assertFalse(ev["eligible"])
            self.assertTrue(any("tape" in r and "zero-byte" in r for r in ev["reasons"]), ev["reasons"])

    def test_rule_missing_lifecycle_fails(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="rule", lifecycle="missing"))
            self.assertFalse(ev["eligible"])

    def test_unknown_run_class_fails(self):
        with _scratch_dir() as t:
            ev = aud.check_row_evidence(_row(t, run="unknown"))
            self.assertEqual(ev["run_class"], "unknown")
            self.assertFalse(ev["eligible"])
            self.assertTrue(any("unknown" in r for r in ev["reasons"]), ev["reasons"])


class ReceiptAudit(unittest.TestCase):
    def _persist(self, tmp, row, logical, *, corrupt_sha=False, row_name="row.json"):
        rp = _w(os.path.join(tmp, "rows", row_name), json.dumps(row))
        sha = aud._sha256_file(Path(rp))
        if corrupt_sha:
            sha = "0" * 64
        claim = {"logical_job": logical, "status": "complete",
                 "result_receipt": {"row_path": rp, "row_sha256": sha,
                                    "companions": {}, "declared_but_not_produced": []}}
        return claim, rp

    def test_sound_model_claim_passes(self):
        with _scratch_dir() as t:
            row = _row(t, run="model", ds=3, tape=("lines", 3))
            claim, _ = self._persist(t, row, "r66eval2_model_new_s11_greedy_scene1800000")
            res = aud.audit_receipt(claim)
            self.assertTrue(res["admissible"], res["reasons"])
            self.assertEqual(res["run_class"], "model")

    def test_row_missing_fails(self):
        with _scratch_dir() as t:
            claim = {"logical_job": "r66eval2_model_new_s11_greedy_scene1800000",
                     "status": "complete",
                     "result_receipt": {"row_path": os.path.join(t, "nope.json"),
                                        "row_sha256": "0" * 64}}
            self.assertFalse(aud.audit_receipt(claim)["admissible"])

    def test_hash_mismatch_fails(self):
        with _scratch_dir() as t:
            row = _row(t, run="model", ds=3, tape=("lines", 3))
            claim, _ = self._persist(t, row, "r66eval2_model_new_s11_greedy_scene1800000",
                                     corrupt_sha=True)
            res = aud.audit_receipt(claim)
            self.assertFalse(res["admissible"])
            self.assertTrue(any("sha256" in r for r in res["reasons"]), res["reasons"])

    def test_model_missing_tape_claim_fails(self):
        with _scratch_dir() as t:
            row = _row(t, run="model", ds=3, tape="missing")
            claim, _ = self._persist(t, row, "r66eval2_model_new_s11_greedy_scene1800000")
            res = aud.audit_receipt(claim)
            self.assertFalse(res["admissible"])
            self.assertTrue(any("tape" in r for r in res["reasons"]), res["reasons"])

    def test_identity_mismatch_fails(self):
        with _scratch_dir() as t:
            row = _row(t, run="model", ds=3, tape=("lines", 3), scene=1800005)
            claim, _ = self._persist(t, row, "r66eval2_model_new_s11_greedy_scene1800000")
            res = aud.audit_receipt(claim)
            self.assertFalse(res["admissible"])
            self.assertTrue(any("identity" in r for r in res["reasons"]), res["reasons"])

    def test_no_receipt_fails(self):
        claim = {"logical_job": "r66eval2_rule_immediate_greedy_scene1800000", "status": "complete"}
        self.assertFalse(aud.audit_receipt(claim)["admissible"])


class ClaimsAudit(unittest.TestCase):
    def _write_claim(self, tmp, logical, row, row_name, *, corrupt_sha=False):
        rp = _w(os.path.join(tmp, "rows", row_name), json.dumps(row))
        sha = aud._sha256_file(Path(rp))
        if corrupt_sha:
            sha = "1" * 64
        d = os.path.join(tmp, "launch_claims", logical)
        Path(d).mkdir(parents=True, exist_ok=True)
        _w(os.path.join(d, "attempt_001.json"),
           json.dumps({"logical_job": logical, "status": "complete",
                       "result_receipt": {"row_path": rp, "row_sha256": sha}}))

    def test_orphan_excluded(self):
        with _scratch_dir() as t:
            good = _row(t, run="model", ds=3, tape=("lines", 3))
            self._write_claim(t, "r66eval2_model_new_s11_greedy_scene1800000", good,
                              "v2_r66eval2_model_new_s11_greedy_scene1800000.json")
            orphan = _row(t, run="model", ds=3, tape=("lines", 3), scene=1800001)
            _w(os.path.join(t, "rows", "v2_r66eval2_model_new_s11_greedy_scene1800001.json"),
               json.dumps(orphan))
            res = aud.audit_claims(Path(t))
            self.assertIn("r66eval2_model_new_s11_greedy_scene1800000", res["admissible_jobs"])
            self.assertEqual(res["summary"]["orphan_row_files"], 1)
            self.assertTrue(any("scene1800001" in o for o in res["orphan_rows"]))

    def test_ambiguous_refused(self):
        with _scratch_dir() as t:
            logical = "r66eval2_model_new_s11_greedy_scene1800000"
            r1 = _row(t, run="model", ds=3, tape=("lines", 3))
            self._write_claim(t, logical, r1, "v2_r66eval2_model_new_s11_greedy_scene1800000.json")
            d = os.path.join(t, "launch_claims", logical)
            r2 = _row(t, run="model", ds=4, tape=("lines", 4))
            r2p = _w(os.path.join(t, "rows", "v2_r66eval2_model_new_s11_greedy_scene1800000_b.json"),
                     json.dumps(r2))
            _w(os.path.join(d, "attempt_002.json"),
               json.dumps({"logical_job": logical, "status": "complete",
                           "result_receipt": {"row_path": r2p,
                                              "row_sha256": aud._sha256_file(Path(r2p))}}))
            res = aud.audit_claims(Path(t))
            self.assertEqual(res["summary"]["ambiguous_logical_jobs"], 1)
            self.assertNotIn(logical, res["admissible_jobs"])

    def test_incomplete_claim_not_admissible(self):
        with _scratch_dir() as t:
            logical = "r66eval2_rule_immediate_greedy_scene1800000"
            rp = _w(os.path.join(t, "rows", "v2_r66eval2_rule_immediate_greedy_scene1800000.json"),
                    json.dumps(_row(t, run="rule", tape="missing")))
            d = os.path.join(t, "launch_claims", logical)
            Path(d).mkdir(parents=True, exist_ok=True)
            _w(os.path.join(d, "attempt_001.json"),
               json.dumps({"logical_job": logical, "status": "failed_technical",
                           "result_receipt": {"row_path": rp,
                                              "row_sha256": aud._sha256_file(Path(rp))}}))
            res = aud.audit_claims(Path(t))
            self.assertNotIn(logical, res["admissible_jobs"])
            self.assertEqual(res["summary"]["orphan_row_files"], 1)

    def test_out_of_scope_training_not_counted(self):
        with _scratch_dir() as t:
            rp = _w(os.path.join(t, "rows", "v2_train_s11_ep0.json"), json.dumps(_row(t)))
            d = os.path.join(t, "launch_claims", "v2_train_s11_ep0")
            Path(d).mkdir(parents=True, exist_ok=True)
            _w(os.path.join(d, "attempt_001.json"),
               json.dumps({"logical_job": "v2_train_s11_ep0", "status": "complete",
                           "result_receipt": {"row_path": rp,
                                              "row_sha256": aud._sha256_file(Path(rp))}}))
            res = aud.audit_claims(Path(t))
            self.assertEqual(res["summary"]["out_of_scope_complete_claims"], 1)
            self.assertEqual(res["summary"]["complete_eval_claims_audited"], 0)
            self.assertEqual(res["summary"]["admissible_logical_jobs"], 0)


class Independence(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        with _scratch_dir() as t:
            p = os.path.join(t, "b.bin")
            Path(p).write_bytes(b"hello world\n")  # binary: no Windows newline translation
            self.assertEqual(aud._sha256_file(Path(p)), hashlib.sha256(b"hello world\n").hexdigest())

    def test_auditor_does_not_import_or_call_adapter(self):
        source = (HERE / "r66_eval_eligibility_audit.py").read_text(encoding="utf-8")
        for banned in ("import r66_eval_fixed2", "import r66_execute",
                       "from r66_eval_fixed2", "from r66_execute",
                       "_receipt_is_sound(", "_qualified_row_paths(",
                       "_load_qualified_rows(", "import torch"):
            self.assertNotIn(banned, source, f"auditor must stay independent of {banned!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
