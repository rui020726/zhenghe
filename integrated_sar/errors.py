"""Explicit failure modes for the joint inference driver.

Every error carries the concrete path, module or tensor shape that caused it, so a
missing asset can never degrade silently into a heuristic run with wrong weights.
"""

from __future__ import annotations


class IntegratedSARError(RuntimeError):
    """Base class for every error raised by the integration layer."""


class ScenarioError(IntegratedSARError):
    """The obstacle map or the derived search scenario is unusable."""


class AssetMissingError(IntegratedSARError):
    """A checkpoint, map or scenario file was requested but cannot be read."""


class DependencyMissingError(IntegratedSARError):
    """A Python package required by the requested mode is not installed."""


class CheckpointIncompatibleError(IntegratedSARError):
    """Weights were found but do not match the architecture or produce unusable output."""


class ConfigurationError(IntegratedSARError):
    """Command-line settings contradict each other or cannot be honoured on this machine."""
