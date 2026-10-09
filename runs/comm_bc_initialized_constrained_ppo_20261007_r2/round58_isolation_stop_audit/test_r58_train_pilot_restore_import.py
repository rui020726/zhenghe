from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
CANON = REPO.parent / "zhenghe"
PILOT = CANON / "runs" / "comm_budget_pilot_20260919"
STAGE_B = CANON / "runs" / "comm_cost_stage_b_20260927"


class TrainPilotRestoreImportTests(unittest.TestCase):
    def _child(self, body: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["R58_REPO"] = str(REPO)
        env["R58_AUDIT"] = str(AUDIT)
        env["R58_PILOT"] = str(PILOT)
        env["R58_STAGE_B"] = str(STAGE_B)
        return subprocess.run([sys.executable, "-c", body], cwd=REPO, env=env,
                              text=True, capture_output=True, timeout=90)

    def test_clean_old_restore_path_reproduces_missing_train_pilot(self) -> None:
        code = r"""
import os, sys
repo, audit, pilot, stage_b = (os.environ[k] for k in
    ('R58_REPO', 'R58_AUDIT', 'R58_PILOT', 'R58_STAGE_B'))
sys.path[:0] = [repo, stage_b, audit]
import sb_common  # configure-time import installs PILOT_ROUND
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p)) !=
               os.path.normcase(os.path.abspath(pilot))]
sys.modules.pop('train_pilot', None)
from sb_controller import StageBController
try:
    StageBController(11)
except ModuleNotFoundError as exc:
    assert exc.name == 'train_pilot', repr(exc)
    print('OLD_PATH_REPRODUCED')
else:
    raise AssertionError('old configure-cleanup path unexpectedly imported train_pilot')
"""
        result = self._child(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OLD_PATH_REPRODUCED", result.stdout)

    def test_verified_private_scope_loads_only_frozen_module_in_clean_child(self) -> None:
        code = r"""
import hashlib, os, sys
from pathlib import Path
repo, audit, pilot, stage_b = (Path(os.environ[k]) for k in
    ('R58_REPO', 'R58_AUDIT', 'R58_PILOT', 'R58_STAGE_B'))
sys.path[:0] = [str(repo), str(stage_b), str(audit)]
import sb_common
sys.path[:] = [p for p in sys.path if Path(p).resolve() != pilot.resolve()]
sys.modules.pop('train_pilot', None)
from r58_checkpoint_contract_repaired import (
    TRAIN_PILOT_SHA256, _verified_train_pilot_import_scope)
from sb_controller import StageBController
with _verified_train_pilot_import_scope(repo):
    controller = StageBController(11)
    import train_pilot
    assert Path(train_pilot.__file__).resolve() == (pilot / 'train_pilot.py').resolve()
    assert hashlib.sha256(Path(train_pilot.__file__).read_bytes()).hexdigest() == TRAIN_PILOT_SHA256
assert controller.parameter_summary()['parameter_count'] > 0
assert str(pilot) not in sys.path  # scope does not leak configure-time path state
print('FROZEN_TRAIN_PILOT_BOUND', train_pilot.__file__)
"""
        result = self._child(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FROZEN_TRAIN_PILOT_BOUND", result.stdout)

    def test_clean_old_stage_b_import_fails_after_configure_path_cleanup(self) -> None:
        code = r"""
import os, sys
from pathlib import Path
repo = Path(os.environ['R58_REPO'])
audit = Path(os.environ['R58_AUDIT'])
canon = repo.parent / 'zhenghe' / 'runs' / 'comm_cost_v2_rerun_20260928'
round18 = repo.parent / 'zhenghe' / 'runs' / 'search_trace_and_policy_20260925'
sys.path[:0] = [str(repo), str(canon), str(audit)]
import v2_common
assert Path(v2_common.__file__).resolve() == (canon / 'v2_common.py').resolve()
assert Path(v2_common.ROUND18_DIR).resolve() == round18.resolve()
sys.path[:] = [p for p in sys.path if Path(p).resolve() != round18.resolve()]
sys.modules.pop('sp_common', None)
try:
    v2_common.stage_b_inherited_config(1300003)
except ModuleNotFoundError as exc:
    assert exc.name == 'sp_common', repr(exc)
    print('OLD_STAGE_B_IMPORT_FAILURE_REPRODUCED')
else:
    raise AssertionError('old lazy Stage-B import unexpectedly resolved sp_common')
"""
        result = self._child(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OLD_STAGE_B_IMPORT_FAILURE_REPRODUCED", result.stdout)

    def test_verified_sp_common_binding_survives_path_cleanup_and_rejects_wrong_cache(self) -> None:
        code = r"""
import builtins, hashlib, os, sys, types
from pathlib import Path
repo = Path(os.environ['R58_REPO'])
audit = Path(os.environ['R58_AUDIT'])
canon = repo.parent / 'zhenghe' / 'runs' / 'comm_cost_v2_rerun_20260928'
round18 = repo.parent / 'zhenghe' / 'runs' / 'search_trace_and_policy_20260925'
sys.path[:0] = [str(repo), str(canon), str(audit)]
import v2_common
sys.path[:] = [p for p in sys.path if Path(p).resolve() != round18.resolve()]
sys.modules.pop('sp_common', None)
from r58_private_launcher_repaired import (
    FROZEN_SP_COMMON_SHA256, _verified_sp_common_module)
sys.path[:] = [p for p in sys.path if not Path(p).exists() or Path(p).resolve() != round18.resolve()]
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'sp_common' and level == 0:
        return _verified_sp_common_module(v2_common)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
try:
    fields, substituted = v2_common.stage_b_inherited_config(1300003)
finally:
    builtins.__import__ = original
sys.path[:] = [p for p in sys.path if not Path(p).exists() or Path(p).resolve() != round18.resolve()]
module = sys.modules['sp_common']
assert Path(module.__file__).resolve() == (round18 / 'sp_common.py').resolve()
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == FROZEN_SP_COMMON_SHA256
assert fields['seed'] == 1300003 and fields['steps'] == 1000
assert all(Path(p).resolve() != round18.resolve() for p in sys.path if Path(p).exists())
sys.modules.pop('sp_common', None)
sys.modules['sp_common'] = types.ModuleType('sp_common')
sys.modules['sp_common'].__file__ = str(repo / 'unapproved' / 'sp_common.py')
try:
    _verified_sp_common_module(v2_common)
except PermissionError as exc:
    assert 'unapproved provenance' in str(exc)
    print('FROZEN_SOURCE_BOUND_AND_WRONG_CACHE_REJECTED')
else:
    raise AssertionError('wrong cached sp_common was accepted')
"""
        result = self._child(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FROZEN_SOURCE_BOUND_AND_WRONG_CACHE_REJECTED", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
