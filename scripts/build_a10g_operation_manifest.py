#!/usr/bin/env python3
"""Build or check deterministic A10G operation/composite registry manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.dataset_pack.composite_sampler import (  # noqa: E402
    build_composite_block_registry,
    plan_composite_corpus,
)
from perfseer_v3.dataset_pack.operation_sampler import (  # noqa: E402
    build_operation_generator_registry,
    plan_operation_corpus,
)


DEFAULT_OPERATION_REGISTRY = (
    SRC / "perfseer_v3" / "registries" / "operation_benchmark_registry.yaml"
)
DEFAULT_COMPOSITE_REGISTRY = (
    SRC / "perfseer_v3" / "registries" / "composite_block_registry.yaml"
)


def _yaml(payload: dict[str, object]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False, width=120)


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def _write_or_check(path: Path, rendered: str, *, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale or missing generated manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in registries are stale")
    parser.add_argument("--operation-registry", type=Path, default=DEFAULT_OPERATION_REGISTRY)
    parser.add_argument("--composite-registry", type=Path, default=DEFAULT_COMPOSITE_REGISTRY)
    parser.add_argument("--operation-plan", type=Path, help="optional JSON path for the 8K candidate plan")
    parser.add_argument("--composite-plan", type=Path, help="optional JSON path for the 2K candidate plan")
    args = parser.parse_args()

    operation_registry = build_operation_generator_registry()
    composite_registry = build_composite_block_registry()
    _write_or_check(
        args.operation_registry,
        _yaml(operation_registry.to_dict()),
        check=args.check,
    )
    _write_or_check(
        args.composite_registry,
        _yaml(composite_registry.to_dict()),
        check=args.check,
    )
    if args.operation_plan:
        _write_or_check(
            args.operation_plan,
            _json(plan_operation_corpus(generator_registry=operation_registry).to_dict()),
            check=args.check,
        )
    if args.composite_plan:
        _write_or_check(
            args.composite_plan,
            _json(plan_composite_corpus(registry=composite_registry).to_dict()),
            check=args.check,
        )
    print(
        json.dumps(
            {
                "operation_generator_registry_sha256": operation_registry.sha256,
                "operation_generator_count": len(operation_registry.generators),
                "composite_block_registry_sha256": composite_registry.sha256,
                "composite_block_count": len(composite_registry.blocks),
                "checked": args.check,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
