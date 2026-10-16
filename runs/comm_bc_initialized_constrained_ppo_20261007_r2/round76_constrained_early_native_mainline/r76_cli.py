"""R76 CLI boundary: refuse before lazy real imports unless a frozen real permit exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import r76_contracts as contracts


class ExecutionRefused(RuntimeError):
    """Raised before real Torch/NumPy/model/config/directory/resource work."""


REQUIRED_PERMIT_FIELDS = (
    "kind", "status", "request_id", "task_id", "branch", "source_head", "scope",
    "real_contract_sha256", "resource_root", "contract_path",
)


def validate_permit_document(document: Any, *, request_id: str,
                             source_head: str) -> dict[str, Any]:
    contracts.validate_exact_keys(document, REQUIRED_PERMIT_FIELDS, "real execution permit")
    if document["kind"] != "round76-real-training-permission/1":
        raise ExecutionRefused("the permit kind is not the R76 real-training permit")
    if document["status"] != "issued":
        raise ExecutionRefused("the permit is not issued")
    if document["request_id"] != request_id or request_id != contracts.REQUEST_ID:
        raise ExecutionRefused("the request id cannot bypass a missing or mismatched permit")
    if document["task_id"] != contracts.TASK_ID:
        raise ExecutionRefused("the permit task id is not R76 A")
    if document["branch"] != contracts.BRANCH:
        raise ExecutionRefused("the permit branch is not the R76 branch")
    if document["source_head"] != source_head:
        raise ExecutionRefused("the permit source head does not match the requested source")
    if document["scope"] != "real_training_frozen_contract":
        raise ExecutionRefused("A0 and A1 do not authorize a real training launch")
    if not isinstance(document["real_contract_sha256"], str) \
            or len(document["real_contract_sha256"]) != 64:
        raise ExecutionRefused("the real training contract SHA is missing")
    if not document["resource_root"]:
        raise ExecutionRefused("the permit has no canonical resource root")
    return document


def load_permit(path, *, request_id: str, source_head: str) -> dict[str, Any]:
    permit_path = Path(path)
    if not permit_path.is_file():
        raise ExecutionRefused(
            "real execution refused before any lazy import, model, directory, or charge: "
            "no real-training permit file was supplied")
    try:
        document = json.loads(permit_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ExecutionRefused("the supplied permit is not valid JSON") from error
    return validate_permit_document(document, request_id=request_id, source_head=source_head)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROUND76 R71-INIT native early mainline")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--permit", required=True)
    parser.add_argument("--attempt-dir", required=True)
    return parser


def preflight(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any]]:
    args = build_parser().parse_args(argv)
    permit = load_permit(args.permit, request_id=args.request_id,
                         source_head=args.source_head)
    return args, permit


def main(argv: list[str] | None = None) -> int:
    # Keep this boundary deliberately first: no r76_native/r76_runtime import may occur above.
    args, permit = preflight(argv)
    contracts.validate_real_execution_contract(
        permit["contract_path"], source_head=args.source_head,
        driver_path=Path(__file__).resolve().parent / "r76_execute.py")
    try:
        import r76_execute
    except Exception as error:
        raise ExecutionRefused(
            f"the frozen real training driver is not source-accepted yet: {error}") from error
    return r76_execute.run_real_training(args, permit)
