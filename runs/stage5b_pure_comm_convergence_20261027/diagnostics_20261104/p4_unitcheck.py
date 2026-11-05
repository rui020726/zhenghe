"""P4 offline unit check: zero worlds, zero training, zero replays.

P3 installed the scale transform by shadowing ``controller.forward``.  That covers the
sampling path only.  Three independent places read the 19-dim row:

  1. budget_masked.BudgetMaskedCollector._choose_masked  -> self.controller.forward(matrix)
  2. sb_collector.StageBCollector.decide                 -> self.controller.cost_values(matrix)
  3. trainer.build_samples -> controller.inner.trunk(stacked) on tape["features"]

(1) is an instance attribute lookup, so it can be shadowed.  (2) is too.  (3) re-reads the
*tape*, which the pinned collector wrote from the untransformed rows.  If only (1) is
shadowed, the stored behaviour log-prob was produced under the rewritten row while the
update recomputes it under the raw row, so the PPO ratio is not 1 at the first epoch even
though no parameter has moved.  With clip 0.2 those samples get zero gradient, and the ones
inside the band get a gradient pointing at the wrong objective.  Any "cannot learn" verdict
from such a run would be an artefact of the seam, not a property of the policy.

This file proves the situation with a stub controller that mirrors StageBController's two
relevant lines, then re-checks after the fix:

    forward(batch)     -> self.inner.forward(batch)
    cost_values(batch) -> self.cost_head(self.inner.trunk(batch)).squeeze(-1)

Run:  D:/project_env/python.exe -B diagnostics_20261104/p4_unitcheck.py
Exit code 0 = every check passed.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
ROOT = STAGE / "recovery_12" / "revision_01"

import numpy as np
import torch

sys.path.insert(0, str(ROOT))
from stage5b_comm import scale_features as sf  # noqa: E402

N_ROWS = 5
INDEX_SENT_COUNT = 2
INDEX_BACKLOG = 14
INDEX_SUBSCRIBERS = 17


class _StubInner(torch.nn.Module):
    """Mirrors train_pilot.Controller's forward: trunk -> action_head / value_head."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.trunk = torch.nn.Sequential(torch.nn.Linear(sf.SUBSCRIBERS_REF * 3 + 1, 16),
                                         torch.nn.Tanh())
        self.action_head = torch.nn.Linear(16, 2)
        self.value_head = torch.nn.Linear(16, 1)

    def forward(self, batch):
        hidden = self.trunk(batch)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)


class _StubController:
    """The two lines StageBController adds on top of the pilot controller."""

    def __init__(self) -> None:
        self.inner = _StubInner()
        self.cost_head = torch.nn.Linear(16, 1)
        self.seen = []

    def forward(self, batch):
        return self.inner.forward(batch)

    def cost_values(self, batch):
        return self.cost_head(self.inner.trunk(batch)).squeeze(-1)


class _PlainInner:
    """Mirrors train_pilot.Controller: a *plain class*, not an nn.Module.

    ``trunk`` lives in the instance ``__dict__``, so restore must write the real object back
    rather than deleting the key. Getting this wrong deletes the trunk and every later world
    dies with ``'Controller' object has no attribute 'trunk'``.
    """

    def __init__(self) -> None:
        self._inner = _StubInner()
        self.trunk = self._inner.trunk
        self.action_head = self._inner.action_head
        self.value_head = self._inner.value_head
        self.parameters = [*self.trunk.parameters(), *self.action_head.parameters(),
                           *self.value_head.parameters()]

    def forward(self, batch):
        hidden = self.trunk(batch)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)


class _PlainController(_StubController):
    def __init__(self) -> None:
        self.inner = _PlainInner()
        self.cost_head = torch.nn.Linear(16, 1)


def raw_rows(subscribers=12):
    """A 19-dim comm-send-state/1 row with distinct, realistic magnitudes."""
    rows = []
    for index in range(N_ROWS):
        row = [0.0] * 19
        row[0] = float(index)                 # age_steps
        row[INDEX_SENT_COUNT] = float(index)  # sent_count: must never be rewritten
        row[10] = float(index * 3)            # revision_delta
        row[INDEX_BACKLOG] = float(3 + index * 5)   # queue_backlog_targets
        row[15] = 1.0 - 0.1 * index           # budget_remaining_fraction
        row[INDEX_SUBSCRIBERS] = float(subscribers)
        row[18] = 0.5
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


class Checks:
    def __init__(self) -> None:
        self.failures = []

    def check(self, name, condition, detail=""):
        ok = bool(condition)
        print("  %-52s %s%s" % (name, "PASS" if ok else "FAIL",
                                ("" if ok else "   " + str(detail))))
        if not ok:
            self.failures.append(name)
        return ok


def main() -> int:
    checks = Checks()
    rows = raw_rows()
    matrix = torch.as_tensor(rows)

    print("A  off is the identity")
    off = sf.ScaleInvariantFeatures(mode="off", subscribers=12)
    out = off.rewrite(matrix)
    checks.check("off returns the caller's own object", out is matrix)
    off.install(_StubController())
    checks.check("off rewrote no value", sum(off.changed.values()) == 0, off.changed)

    print("B  norm_only rewrites exactly #17 and #14")
    norm = sf.ScaleInvariantFeatures(mode="norm_only", subscribers=12)
    rewritten = norm.rewrite(matrix)
    expect17 = rows[:, INDEX_SUBSCRIBERS] / sf.SUBSCRIBERS_REF
    expect14 = rows[:, INDEX_BACKLOG] / float(sf.NUM_GOALS)
    checks.check("#17 == n/6", np.allclose(rewritten[:, INDEX_SUBSCRIBERS].numpy(), expect17))
    checks.check("#17 at n=6 lands on exactly 1.0",
                 np.allclose(
                     sf.ScaleInvariantFeatures(mode="norm_only", subscribers=6)
                     .rewrite(raw_rows(subscribers=6))[:, INDEX_SUBSCRIBERS].numpy(), 1.0))
    checks.check("#14 == backlog/24", np.allclose(rewritten[:, INDEX_BACKLOG].numpy(), expect14))
    untouched = [c for c in range(19) if c not in (INDEX_BACKLOG, INDEX_SUBSCRIBERS)]
    checks.check("#2 sent_count and every other column untouched",
                 np.allclose(rewritten[:, untouched].numpy(), rows[:, untouched]))
    checks.check("returns a torch tensor",
                 isinstance(rewritten, torch.Tensor), type(rewritten).__name__)
    checks.check("dtype preserved", rewritten.dtype == matrix.dtype,
                 (rewritten.dtype, matrix.dtype))

    print("C  every read site must see the same row")
    controller = _StubController()
    target = torch.as_tensor(rewritten)

    trunk_seen = []
    handle = controller.inner.trunk.register_forward_pre_hook(
        lambda _module, inputs: trunk_seen.append(torch.as_tensor(inputs[0]).detach().clone()))

    transform = sf.ScaleInvariantFeatures(mode="norm_only", subscribers=12)
    transform.install(controller)

    # The three real read paths, in the order the runtime uses them.
    controller.forward(matrix)                       # budget_masked sampling
    controller.cost_values(torch.as_tensor(rows))    # sb_collector cost critic
    controller.inner.trunk(torch.as_tensor(rows))    # trainer.build_samples
    handle.remove()

    checks.check("all three read sites hit the wrapped trunk", len(trunk_seen) == 3,
                 len(trunk_seen))
    for name, observed in zip(("sampling", "cost critic", "PPO update"), trunk_seen):
        checks.check("%s reached the trunk rewritten" % name,
                     torch.allclose(observed, target),
                     float((observed - target).abs().max()))

    print("D  importance ratio at update 0 must be exactly 1")
    old_logits, _ = controller.forward(matrix)
    hidden = controller.inner.trunk(torch.as_tensor(rows))
    new_logits = controller.inner.action_head(hidden)
    delta = float((torch.log_softmax(new_logits, dim=-1)
                   - torch.log_softmax(old_logits, dim=-1)).abs().max().detach())
    checks.check("max |new_logp - old_logp| < 1e-6", delta < 1e-6, delta)

    print("E  restore puts the real module back")
    sf.ScaleInvariantFeatures.restore({"scale_features": transform})
    checks.check("trunk is an nn.Module again",
                 isinstance(controller.inner.trunk, torch.nn.Module),
                 type(controller.inner.trunk).__name__)
    checks.check("parameter set unchanged by install/restore",
                 not any(isinstance(m, torch.nn.Module) is False
                         for _, m in controller.inner.named_modules()))
    before = sorted(name for name, _ in controller.inner.trunk.named_parameters())
    sf.ScaleInvariantFeatures.restore({"scale_features": transform})
    checks.check("restore is idempotent",
                 sorted(name for name, _ in controller.inner.trunk.named_parameters()) == before)

    print("F  inference path is unchanged by moving the seam forward -> trunk")
    # P3 measured its on-time numbers with the forward shadow. train_pilot.Controller.forward
    # is ``action_head(trunk(batch))`` with nothing in between, so the two placements must
    # emit the same logits. If this ever fails, P3's archived rows no longer describe the
    # code the P4 run is executing.
    plain = _StubController()
    baseline = plain.forward(matrix)

    old_style = _StubController()
    original_forward = old_style.forward

    def forward_shadow(batch):                       # exactly the pre-fix implementation
        return original_forward(
            sf.ScaleInvariantFeatures(mode="norm_only", subscribers=12).rewrite(batch))

    old_style.forward = forward_shadow

    new_style = _StubController()
    sf.ScaleInvariantFeatures(mode="norm_only", subscribers=12).install(new_style)
    checks.check("norm_only forward-shadow == trunk-wrap (logits)",
                 torch.allclose(old_style.forward(matrix)[0], new_style.forward(matrix)[0]))
    checks.check("norm_only forward-shadow == trunk-wrap (values)",
                 torch.allclose(old_style.forward(matrix)[1], new_style.forward(matrix)[1]))

    off_style = _StubController()
    sf.ScaleInvariantFeatures(mode="off", subscribers=12).install(off_style)
    checks.check("off forward-shadow == trunk-wrap == no transform at all",
                 torch.allclose(off_style.forward(matrix)[0], baseline[0])
                 and torch.allclose(off_style.forward(matrix)[1], baseline[1]))

    print("G  a plain-class inner (train_pilot.Controller) survives install/restore")
    # This is the shape the runtime actually loads via v2_controller.fresh_controller, and the
    # first smoke run died here: restore popped 'trunk' out of a plain instance __dict__ and
    # deleted the real trunk.
    plain_ctrl = _PlainController()
    real_trunk = plain_ctrl.inner.trunk
    checks.check("real trunk starts as a module", isinstance(real_trunk, torch.nn.Module))
    check = sf.ScaleInvariantFeatures(mode="norm_only", subscribers=12)
    check.install(plain_ctrl)
    checks.check("shadow is callable, trunk attribute still exists",
                 callable(plain_ctrl.inner.trunk))
    sf.ScaleInvariantFeatures.restore({"scale_features": check})
    checks.check("restore gives the real module back, not a deletion",
                 plain_ctrl.inner.trunk is real_trunk,
                 type(plain_ctrl.inner.trunk).__name__)
    checks.check("restored controller still forwards",
                 torch.allclose(plain_ctrl.forward(matrix)[0], baseline[0]))
    checks.check("optimiser parameter list still resolves",
                 len(list(plain_ctrl.inner.parameters)) == 6,
                 len(list(plain_ctrl.inner.parameters)))

    print()
    if checks.failures:
        print("FAILED %d check(s): %s" % (len(checks.failures), ", ".join(checks.failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
