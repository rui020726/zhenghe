"""Pure-communication Stage 5B package; importing it loads no world and no model."""

import sys as _sys

# Frozen dependencies are read-only: importing them must not create __pycache__ in old paths.
_sys.dont_write_bytecode = True

from . import paths as _paths

_paths.ensure_repo_path()

__all__ = ["constants", "paths"]

