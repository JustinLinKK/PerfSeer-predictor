"""Build deterministic runtime registry and feature schema assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.op_registry import DEFAULT_REGISTRY_PATH, OperationRegistry
from perfseer_v3.schema import build_feature_schema, validate_feature_schema


def build_assets(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    schema_path: Path = SRC / "perfseer_v3" / "schemas" / "perfseer_graph_v3.json",
    runtime_registry_path: Path = SRC / "perfseer_v3" / "op_registry_v3.json",
) -> tuple[Path, Path]:
    registry = OperationRegistry.load(registry_path)
    schema = build_feature_schema(registry)
    validate_feature_schema(schema, registry)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime_registry_path.write_text(
        json.dumps(registry.runtime_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return schema_path, runtime_registry_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--schema-out",
        type=Path,
        default=SRC / "perfseer_v3" / "schemas" / "perfseer_graph_v3.json",
    )
    parser.add_argument(
        "--runtime-registry-out",
        type=Path,
        default=SRC / "perfseer_v3" / "op_registry_v3.json",
    )
    args = parser.parse_args(argv)
    schema_path, registry_path = build_assets(args.registry, args.schema_out, args.runtime_registry_out)
    print(f"schema={schema_path}")
    print(f"runtime_registry={registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

