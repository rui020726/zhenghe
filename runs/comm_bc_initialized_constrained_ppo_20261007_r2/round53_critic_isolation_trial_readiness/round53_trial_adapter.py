"""Non-launching mechanics for the proposed ROUND53 critic-gradient route.

This module deliberately does not import project runtime modules, configure a
runner, build environments, create claims, or touch checkpoints.  A future
authorized harness may use ``critic_input_detached`` around the exact
source-bound ``ppo_update`` call after its own authorization and namespace
gates pass.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator
from pathlib import Path


TRAINER_SHA256 = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
ROUTE_VERSION = "critic_representation_stop_gradient/1"


@contextmanager
def critic_input_detached(controller: Any) -> Iterator[None]:
    """Detach only the shared representation entering task/cost value heads."""
    modules = (controller.inner.value_head, controller.cost_head)
    before = [set(module._forward_pre_hooks) for module in modules]
    handles = []

    def detach_first(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args or not hasattr(args[0], "detach"):
            raise RuntimeError("critic head input is not a detachable tensor")
        return (args[0].detach(), *args[1:])

    try:
        for module in modules:
            handles.append(module.register_forward_pre_hook(detach_first))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        if any(set(module._forward_pre_hooks) != old
               for module, old in zip(modules, before)):
            raise AssertionError("temporary critic hooks were not fully removed")


def install_on_verified_trainer(trainer: Any, *, source_sha256: str) -> Callable[[], None]:
    """Install one scoped wrapper on a verified trainer object; no call is made.

    The returned undo function restores the exact method object.  Callers must
    separately establish the live runtime binding and output namespace.
    """
    if source_sha256 != TRAINER_SHA256:
        raise RuntimeError("canonical trainer source identity mismatch")
    original = getattr(trainer, "ppo_update", None)
    if not callable(original):
        raise RuntimeError("verified trainer has no callable ppo_update")

    @wraps(original)
    def isolated_ppo_update(controller: Any, *args: Any, **kwargs: Any) -> Any:
        with critic_input_detached(controller):
            return original(controller, *args, **kwargs)

    trainer.ppo_update = isolated_ppo_update

    def undo() -> None:
        if getattr(trainer, "ppo_update", None) is not isolated_ppo_update:
            raise RuntimeError("trainer binding changed before adapter removal")
        trainer.ppo_update = original

    return undo


def validate_live_binding(runtime: dict[str, Any], trainer: Any) -> None:
    """Fail closed unless the exact trainer object is on the actual adapter path."""
    if runtime.get("trainer") is not trainer:
        raise RuntimeError("runtime trainer role/object mismatch")
    r40 = runtime.get("r40_train")
    if r40 is None:
        raise RuntimeError("runtime lacks the ROUND40 training adapter")
    # R42 assigns this live object immediately before calling r40.train_seed.
    # This check must be repeated after that assignment in an authorized harness.
    if getattr(r40, "trainer", None) is not trainer:
        raise RuntimeError("ROUND40 adapter is not bound to the verified trainer")


def require_explicit_authorization(authorization: dict[str, Any]) -> None:
    """Reject readiness documents and proposals as launch authorization."""
    if (not isinstance(authorization, dict)
            or authorization.get("status") != "explicitly_authorized"
            or authorization.get("task_id") != "ROUND53-ZERO-LAUNCH-ISOLATED-TRIAL-READINESS"
            or authorization.get("hard_cap") != 552
            or authorization.get("branch") != "feat/round53-critic-isolation-trial-readiness"):
        raise PermissionError("ROUND53 requires a separate explicit task/branch/quota authorization")


def require_private_output_root(root: Path, targets: list[Path]) -> None:
    """Pure path check; reject any output that escapes a fresh private root."""
    base = Path(root).resolve()
    for target in targets:
        resolved = Path(target).resolve()
        if resolved == base or base not in resolved.parents:
            raise RuntimeError(f"output escapes the isolated root: {resolved}")

