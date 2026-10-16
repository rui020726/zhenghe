"""Pure ROUND76 B entry boundary; real work is reached only through the external B2 bundle."""

from __future__ import annotations

import argparse
import json
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROUND76 B entry boundary")
    parser.add_argument("--show-b0-state", action="store_true",
                        help="print the pure B0 state")
    parser.add_argument("--execute", action="store_true",
                        help="run the B2 driver only with a valid contract and permit")
    parser.add_argument("--contract-path")
    parser.add_argument("--permit-path")
    parser.add_argument("--source-head")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def b0_state() -> dict[str, Any]:
    import r76_b_contracts as contracts

    state = contracts.build_b0_state()
    contracts.validate_stage_caps(state["caps"])
    return state


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.show_b0_state:
        print(json.dumps(b0_state(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.execute:
        import r76_b_driver

        return r76_b_driver.main(argv)
    raise SystemExit("B entry requires --execute or --show-b0-state")


if __name__ == "__main__":
    raise SystemExit(main())
