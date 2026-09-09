"""Build the deterministic registry-derived A10G support planning contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.dataset_pack.operation_support import (
    DEFAULT_GENERATED_CONTRACT_PATH,
    build_operation_support_contract,
    write_operation_support_contract,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_GENERATED_CONTRACT_PATH)
    args = parser.parse_args(argv)
    output = write_operation_support_contract(args.output)
    contract = build_operation_support_contract()
    print(
        f"operations={len(contract.operations)} training_approved={contract.training_approved} "
        f"contract_sha256={contract.sha256} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
