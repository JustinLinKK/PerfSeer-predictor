#!/usr/bin/env python3
"""Build a profiling manifest from PyTorch model source files."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def literal_assignment(line: str, name: str) -> Any | None:
    if not line.startswith(f"{name} "):
        return None
    try:
        tree = ast.parse(line)
        stmt = tree.body[0]
        if not isinstance(stmt, ast.Assign):
            return None
        return ast.literal_eval(stmt.value)
    except Exception:
        return None


def read_source_metadata(path: Path, default_shape: list[int]) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {"model_id": path.stem, "input_shape": default_shape, "original_stem": path.stem}
    saw_make_model = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("NODE_SPECS"):
                break
            if stripped.startswith("def make_model"):
                saw_make_model = True
            for key, field in (("MODEL_ID", "model_id"), ("INPUT_SHAPE", "input_shape"), ("ORIGINAL_STEM", "original_stem")):
                value = literal_assignment(stripped, key)
                if value is not None:
                    metadata[field] = value
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 4096))
        text_tail = handle.read().decode("utf-8", errors="ignore")
        if "def make_model" in text_tail:
            saw_make_model = True
    if not saw_make_model:
        return None
    metadata["input_shape"] = input_shape(metadata.get("input_shape"), default_shape)
    return metadata


def input_shape(value: Any, default_shape: list[int]) -> list[int]:
    if value is None:
        value = default_shape
    return [int(dim) for dim in value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a model-source profiling manifest.")
    parser.add_argument("--models-dir", required=True, help="Directory with PyTorch source files.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--default-input-shape", default="1,3,224,224")
    parser.add_argument("--precision-config", default="fp32_ieee")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    default_shape = [int(part) for part in args.default_input_shape.split(",")]
    rows = []
    for path in sorted(models_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        metadata = read_source_metadata(path, default_shape)
        if metadata is None:
            continue
        model_id = metadata["model_id"]
        shape = metadata["input_shape"]
        rows.append(
            {
                "model_id": str(model_id),
                "original_stem": str(metadata.get("original_stem", path.stem)),
                "model_file": path.name,
                "input_shape": shape,
                "precision_config": args.precision_config,
                "label_file": f"label/label/{model_id}_{args.precision_config}.txt",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"models": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
