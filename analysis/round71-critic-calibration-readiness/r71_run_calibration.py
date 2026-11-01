#!/usr/bin/env python
"""ROUND71 calibration CLI (ZERO-EXECUTION preparation).

Phase order is fixed and fail-closed: licence gate -> phase requirements -> model-read approval
-> runtime configuration, and the model read-in point is a controlled stop boundary.  Nothing
before the gates touches the filesystem: no raw root is created, no module is loaded and no
model is read.  ``binding-acceptance`` is the only phase that runs this round; every other
phase fails closed because the candidate plan is APPROVAL_PENDING and no real licence exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r71_runtime_binding as binding  # noqa: E402
from r71_critic_calibration_seam import LicenseInvalid, LicenseMissing  # noqa: E402
from r71_runtime_binding import ModelReadBoundary  # noqa: E402

PHASES = ("binding-acceptance", "offline-fit", "train", "eval")
SEEDS = (11, 22, 33)


def _licence_mapping(text):
    if text is None:
        return None
    if text == "-":
        text = sys.stdin.read()
    path = Path(text)
    mapping = json.loads(path.read_text(encoding="utf-8") if path.is_file() else text)
    if not isinstance(mapping, dict):
        raise LicenseInvalid("a licence must be a JSON object")
    return mapping


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ROUND71 critic-calibration CLI (preparation)")
    parser.add_argument("--raw-root", default=None,
                        help="output root a licensed run would own; never created before the gates")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--licence", default=None,
                        help="path to or inline JSON of a declared licence mapping")
    parser.add_argument("--model-read-approval", default=None,
                        help="explicit approval id for the model read-in point")
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS, default=None,
                        help="frozen seed(s); defaults to all three for train/eval")
    parser.add_argument("--run-id", default="r71-candidate-unassigned")
    parser.add_argument("--receipt", default=None,
                        help="where to write the machine receipt")
    parser.add_argument("--overwrite-receipt", action="store_true")
    return parser.parse_args(argv)


def _write_receipt(path, receipt, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite the existing receipt {path}; pass "
                         f"--overwrite-receipt to replace it explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    args = parse_args(argv)
    receipt_path = Path(args.receipt) if args.receipt else (
        HERE / ("ROUND71_RUNTIME_BINDING_ACCEPTANCE.json" if args.phase == "binding-acceptance"
                else "ROUND71_RUNTIME_BINDING_RECEIPT.json"))
    base = {"kind": "round71-critic-calibration-cli/1", "task_id": binding.TASK_ID,
            "branch": binding.BRANCH, "phase": args.phase, "run_id": args.run_id,
            "consumption": 0}

    # ---- gate 1: the licence (a raw token is never a licence)
    try:
        gate = binding.make_license_gate(_licence_mapping(args.licence))
    except LicenseInvalid as exc:
        receipt = dict(base, status="fail_closed", refusal=f"LicenseInvalid: {exc}")
        _write_receipt(receipt_path, receipt, args.overwrite_receipt)
        print(json.dumps({"status": receipt["status"], "refusal": receipt["refusal"]}))
        return 2

    # ---- the only phase that runs this round: zero-consumption static/live acceptance
    if args.phase == "binding-acceptance":
        receipt = dict(base, acceptance=binding.binding_acceptance())
        receipt["status"] = receipt["acceptance"]["status"]
        _write_receipt(receipt_path, receipt, args.overwrite_receipt)
        print(json.dumps({"status": receipt["status"],
                          "receipt": str(receipt_path)}))
        return 0 if receipt["status"] == "passed" else 1

    # ---- gate 2: phase requirements (nothing has been created or loaded yet)
    problems = []
    if args.phase in ("train", "eval") and not args.raw_root:
        problems.append(f"phase {args.phase} requires --raw-root")
    seeds = tuple(args.seed) if args.seed else SEEDS
    if args.phase == "offline-fit" and not seeds:
        problems.append("phase offline-fit requires at least one frozen seed")
    if problems:
        receipt = dict(base, status="usage_refused", problems=problems)
        _write_receipt(receipt_path, receipt, args.overwrite_receipt)
        print(json.dumps({"status": receipt["status"], "problems": problems}))
        return 5

    # ---- gates 3+4 (inside configure): model-read approval, then the stop boundary.
    # No raw root exists yet and none is created before the licence/approval gates pass.
    run_identity = binding.registered_run_identity(args.run_id)
    try:
        cfg = binding.configure_r71_runtime(Path(args.raw_root), gate, run_identity,
                                            model_read_approval=args.model_read_approval)
    except (LicenseMissing, LicenseInvalid) as exc:
        receipt = dict(base, status="fail_closed",
                       refusal=f"{type(exc).__name__}: {exc}")
        _write_receipt(receipt_path, receipt, args.overwrite_receipt)
        print(json.dumps({"status": receipt["status"], "refusal": receipt["refusal"]}))
        return 3
    except ModelReadBoundary as exc:
        receipt = dict(base, status="stopped_at_model_read_boundary",
                       refusal=f"ModelReadBoundary: {exc}")
        _write_receipt(receipt_path, receipt, args.overwrite_receipt)
        print(json.dumps({"status": receipt["status"], "refusal": receipt["refusal"]}))
        return 3

    # ---- a licensed runtime would stop HERE: the phase bodies are not implemented this round
    receipt = dict(base, status="stopped_at_model_read_boundary",
                   note=("the runtime stood up under a verified licence and an explicit "
                         "model-read approval; the phase body is deliberately not implemented "
                         "in this zero-execution preparation round"),
                   r71_identity=cfg["r71"]["identity"], seeds=list(seeds))
    _write_receipt(receipt_path, receipt, args.overwrite_receipt)
    print(json.dumps({"status": receipt["status"]}))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())