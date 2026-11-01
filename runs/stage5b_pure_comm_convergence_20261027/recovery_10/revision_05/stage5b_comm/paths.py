"""Every path this experiment may read or write, resolved from one root.

Two roots exist and they are never mixed: this stage's root (writable, one frozen launch plan)
and the frozen parent root (read-only: parent identity, plan, telemetry, six milestone_1320
checkpoints and the parent training rows used as pre-resume convergence history).
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import constants as C

#: <revision>/stage5b_comm/paths.py -> <revision>, resolved from this file.
REVISION_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = REVISION_ROOT.parent

#: Every earlier Stage 5B tree is read-only evidence for this stage. The list exists so a
#: mistaken --root fails closed instead of writing into a frozen run.
LEGACY_ROOTS = (
    Path(C.BASE_ROOT),
    Path("D:/code/整合/zhenghe-round42/runs/stage5b_pure_comm_pacing_and_ppo_20261026"),
    Path("D:/code/整合/zhenghe-round42/runs/stage5b_pure_comm_budget58600_20261026"),
    Path("D:/code/整合/zhenghe-round42/runs/stage5b_pure_comm_longer_training_20261026"),
)


def _find_repo_root(start):
    """Find zhenghe-round42 from the real integrated_sar anchor, not a fixed depth."""
    for candidate in (Path(start), *Path(start).parents):
        if (candidate / "integrated_sar").is_dir():
            return candidate
    raise RuntimeError("cannot locate the repository root (no integrated_sar under %s)"
                       % (Path(start),))


REPO_ROOT = _find_repo_root(REVISION_ROOT)


def ensure_repo_path():
    """Put the frozen source repository on sys.path from the file location alone."""
    value = str(REPO_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return REPO_ROOT


def root_dir(explicit=None):
    """Resolve the single allowed run root; never invent one."""
    if explicit:
        path = Path(explicit).resolve()
    else:
        path = REVISION_ROOT
    if not path.is_dir():
        raise RuntimeError("run root does not exist: %s" % (path,))
    return path


class Layout:
    """Directory and file layout inside the single allowed run root."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocol = self.root / "protocol"
        self.package = self.root / "stage5b_comm"
        self.tests = self.root / "tests"
        self.execution = self.root / "execution"
        self.world_logs = self.execution / "world_logs"
        self.rows = self.execution / "rows"
        self.ledger = self.execution / "ledger"
        self.receipts = self.execution / "receipts"
        self.checkpoints = self.execution / "checkpoints"
        self.outputs = self.execution / "outputs"
        self.telemetry = self.execution / "telemetry"
        self.monitor = self.execution / "monitor"
        self.eval_models = self.execution / "eval_model_freeze"
        self.base_models = self.execution / "base_model_freeze"
        self.resume_preflight = self.execution / "resume_preflight"
        self.convergence = self.execution / "convergence"

    def ensure(self):
        for path in (self.protocol, self.execution, self.world_logs, self.rows,
                     self.ledger, self.receipts, self.checkpoints, self.outputs,
                     self.telemetry, self.monitor, self.base_models, self.resume_preflight,
                     self.convergence):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def preregistration(self):
        return self.protocol / "STAGE5B_PREREGISTRATION.json"

    @property
    def identity(self):
        return self.protocol / "EXPERIMENT_IDENTITY.json"

    @property
    def slot_plan(self):
        return self.protocol / "WORLD_SLOT_PLAN.json"

    @property
    def collaboration_roles(self):
        return self.protocol / "COLLABORATION_ROLES.json"

    @property
    def parent_admission(self):
        return self.protocol / "PARENT_ADMISSION.json"

    @property
    def eval_model_freeze(self):
        return self.eval_models / "EVAL_MODEL_FREEZE.json"

    @property
    def base_model_freeze(self):
        return self.base_models / "BASE_MODEL_FREEZE.json"

    @property
    def resume_preflight_receipt(self):
        return self.resume_preflight / "RESUME_PREFLIGHT.json"

    @property
    def train_rows(self):
        return self.rows / "train"

    def train_row(self, slot):
        return self.train_rows / (slot.replace(":", "_") + ".json")

    def checkpoint(self, arm, seed, episode_number):
        """Resume checkpoint of one run at one 120-episode milestone."""
        return (Path(self.checkpoints) / ("%s_s%d" % (arm, int(seed)))
                / ("milestone_%04d.pth" % int(episode_number)))

    def convergence_check(self, arm, seed, end_episode):
        return (Path(self.convergence)
                / ("CHECK_%s_s%d_e%04d.json" % (arm, int(seed), int(end_episode))))

    def convergence_certificate(self, arm, seed):
        return Path(self.convergence) / ("CERTIFICATE_%s_s%d.json" % (arm, int(seed)))

    @property
    def convergence_report(self):
        return Path(self.root) / "CONVERGENCE_REPORT.json"

    def deliverable(self, name):
        if name not in C.DELIVERABLES:
            raise ValueError("%r is not one of the registered deliverables" % (name,))
        return self.root / name

    @property
    def manifest(self):
        return self.root / "STAGE5B_RUN_MANIFEST.json"

    def assert_inside(self, path):
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("%s escapes the run root %s" % (resolved, self.root)) from error
        return resolved


def base_root_dir():
    """The frozen parent root; read-only for this stage, never a writable root."""
    path = Path(C.BASE_ROOT).resolve()
    if not path.is_dir():
        raise RuntimeError("the frozen parent root is missing: %s" % (path,))
    return path


class BaseLayout:
    """Read-only layout of the frozen parent revision this stage continues from.

    Every path here is outside this stage's root and no method returns a writable location of
    this run. The runner and the guard only ever read these files.
    """

    def __init__(self, root=None):
        self.root = Path(root).resolve() if root is not None else base_root_dir()

    @property
    def protocol(self):
        return self.root / "protocol"

    @property
    def identity(self):
        return self.protocol / "EXPERIMENT_IDENTITY.json"

    @property
    def slot_plan(self):
        return self.protocol / "WORLD_SLOT_PLAN.json"

    @property
    def preregistration(self):
        return self.protocol / "STAGE5B_PREREGISTRATION.json"

    @property
    def execution(self):
        return self.root / "execution"

    @property
    def telemetry(self):
        return self.execution / "telemetry"

    @property
    def rows(self):
        return self.execution / "rows"

    @property
    def train_rows(self):
        return self.rows / "train"

    @property
    def checkpoints(self):
        return self.execution / "checkpoints"

    @property
    def stop_marker(self):
        return self.execution / "ledger" / "GLOBAL_STOP.json"

    def checkpoint(self, arm, seed, name):
        return self.checkpoints / ("%s_s%d" % (arm, int(seed))) / name

    def train_row(self, arm, seed, episode_index):
        return self.train_rows / ("train_%s_%d_%04d.json"
                                  % (arm, int(seed), int(episode_index)))

    def history_rows(self, arm, seed, start, stop):
        """Parent training rows for episodes [start, stop), in episode order."""
        return [self.train_row(arm, seed, index) for index in range(int(start), int(stop))]

