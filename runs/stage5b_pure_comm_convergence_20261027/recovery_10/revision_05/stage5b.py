"""Single CLI entry. Phases are never chained: one authorised phase per invocation.

Stage 2 is a draft pending parent admission. The evaluation, analysis and finalisation phases exist in the parser
so an unauthorised call is rejected by name with a non-zero status instead of running a
half-trained plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage5b_comm import constants as C
from stage5b_comm import runner
from stage5b_comm import guard, parent_admission
from stage5b_comm.paths import Layout, root_dir

PHASES = ("prepare", "puretests", "freeze-base-models", "resume-preflight", "train",
          "convergence-report", "monitor", "recovery-g3", "recovery-train",
          "freeze-eval-models", "eval", "analysis", "finalize")
RECOVERY_PHASES = ("recovery-g3", "recovery-train")

#: Phases that may not start before the frozen identity exists.
NEEDS_IDENTITY = ("freeze-base-models", "resume-preflight", "train", "convergence-report")


def parse(argv):
    parser = argparse.ArgumentParser(
        description=("Stage 5B command surface: normal stages remain parent-gated; "
                     "explicit recovery G3/training require --recovery-context"))
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--root", default=str(ROOT),
                        help="this revision root; the default is the directory of this file")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--recovery-context", default=None,
        help="explicit FROZEN recovery runtime context; required by recovery phases")
    parser.add_argument(
        "--recovery-g3-report", default=None,
        help="validated G3 report sidecar; required only by recovery-train")
    parser.add_argument(
        "--recovery-world-authorization", default=None,
        help="validated main world-authorization sidecar; required only by recovery-train")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > C.WORKERS_MAX:
        parser.error("--workers must be between 1 and %d" % C.WORKERS_MAX)
    if args.phase in RECOVERY_PHASES and not args.recovery_context:
        parser.error("--recovery-context is required for " + args.phase)
    if args.phase not in RECOVERY_PHASES and args.recovery_context:
        parser.error("--recovery-context is accepted only by explicit recovery phases")
    sidecars = (args.recovery_g3_report, args.recovery_world_authorization)
    if args.phase == "recovery-train" and (not sidecars[0] or not sidecars[1]):
        parser.error("recovery-train requires --recovery-g3-report and "
                     "--recovery-world-authorization")
    if args.phase != "recovery-train" and any(value is not None for value in sidecars):
        parser.error("execution sidecars are accepted only by recovery-train")
    return args


def main(argv):
    args = parse(argv)
    layout = Layout(root_dir(args.root))
    if args.phase in ("prepare", *NEEDS_IDENTITY):
        try:
            guard.assert_experiment_root(layout)
            guard.assert_parent_admission(layout)
        except parent_admission.PendingParentAdmission as error:
            print(json.dumps({"phase": args.phase, "status": "PENDING_PARENT_ADMISSION",
                              "error": str(error)}, ensure_ascii=False))
            return 2
    if args.phase in NEEDS_IDENTITY and not layout.identity.is_file():
        print(json.dumps({"phase": args.phase, "status": "BLOCKED_NO_FREEZE"},
                         ensure_ascii=False))
        return 2
    try:
        outcome = runner.run_phase(
            args.phase, layout, workers=args.workers,
            recovery_context=args.recovery_context,
            recovery_g3_report=args.recovery_g3_report,
            recovery_world_authorization=args.recovery_world_authorization)
    except Exception as error:
        # A phase that wrote GLOBAL_STOP or saw a failed world must not exit 0. The error
        # receipt is written by runner.run_phase; this boundary turns the exception into a
        # real non-zero process exit without inventing an outer/wrapper code.
        print(json.dumps({"phase": args.phase, "status": "ERROR",
                          "error": type(error).__name__ + ": " + str(error),
                          "workers": int(args.workers)}, ensure_ascii=False))
        return 1
    print(json.dumps({"phase": args.phase, "returncode": int(outcome["code"]),
                      "receipt_id": outcome["receipt_id"],
                      "workers": int(args.workers)},
                     ensure_ascii=False))
    return int(outcome["code"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
