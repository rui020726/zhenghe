from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def _without_parameter_identity_imports(tree: ast.AST) -> ast.AST:
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            rewritten = []
            for item in value:
                if isinstance(item, ast.ImportFrom) and item.module == "v2_controller":
                    item.names = [alias for alias in item.names
                                  if alias.name != "parameter_identity"]
                    if not item.names:
                        continue
                rewritten.append(item)
            setattr(parent, field, rewritten)
    return tree


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise AssertionError(f"expected one function {name}")
    return matches[0]


class ResumeAdapterRegression(unittest.TestCase):
    def test_only_moves_parameter_identity_import(self) -> None:
        frozen = json.loads((RUN_DIR / "ROUND43_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
        round40 = Path(frozen["execution_identity"]["round40_worktree"])
        original_path = (round40 / "runs" / "comm_first_report_training_ablation_20261003"
                         / "r40_train.py")
        adapter_path = RUN_DIR / "r40_train_resume_adapter.py"
        original = ast.parse(original_path.read_text(encoding="utf-8"))
        adapter = ast.parse(adapter_path.read_text(encoding="utf-8"))
        self.assertEqual(
            ast.dump(_without_parameter_identity_imports(original), include_attributes=False),
            ast.dump(_without_parameter_identity_imports(adapter), include_attributes=False),
        )

    def test_import_is_available_before_resume_branch(self) -> None:
        adapter = ast.parse((RUN_DIR / "r40_train_resume_adapter.py").read_text(encoding="utf-8"))
        train_seed = _function(adapter, "train_seed")
        first = train_seed.body[0]
        self.assertIsInstance(first, ast.ImportFrom)
        self.assertEqual(first.module, "v2_controller")
        self.assertEqual([alias.name for alias in first.names], ["parameter_identity"])
        self.assertTrue(any(
            isinstance(node, ast.ImportFrom) and node.module == "v2_controller"
            and [alias.name for alias in node.names] == ["fresh_controller"]
            for node in ast.walk(train_seed)
        ))


if __name__ == "__main__":
    unittest.main()
