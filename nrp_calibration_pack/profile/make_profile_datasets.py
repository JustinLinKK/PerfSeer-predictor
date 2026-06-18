"""Create per-model synthetic profile specs for calibration labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_REPEATS = 50
DEFAULT_INFER_REPEATS = 50
DEFAULT_SEED = 20260605


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-model profile input/repeat specs from a calibration manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest/subset_manifest.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for <model_id>.json specs plus index.jsonl.")
    parser.add_argument("--train-repeats", type=positive_int, default=DEFAULT_TRAIN_REPEATS)
    parser.add_argument("--infer-repeats", type=positive_int, default=DEFAULT_INFER_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=positive_int, help="Optional number of unique models to emit.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing specs.")
    return parser.parse_args(argv)


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def load_unique_model_rows(manifest_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with manifest_path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            model_id = str(row["model_id"])
            if model_id in seen:
                continue
            seen.add(model_id)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def make_spec(row: dict[str, Any], args: argparse.Namespace, model_index: int) -> dict[str, Any]:
    input_specs = row.get("input_specs") or [{"name": "input0", "shape": row["input_shape"], "dtype": "float32", "kind": "float"}]
    input_specs = [
        {
            "name": str(spec.get("name", f"input{idx}")),
            "shape": [int(dim) for dim in spec.get("shape", [])],
            "dtype": str(spec.get("dtype", "float32")),
            "kind": str(spec.get("kind", "float")),
        }
        for idx, spec in enumerate(input_specs)
    ]
    input_shape = [int(dim) for dim in input_specs[0]["shape"]]
    batch_size = int(input_shape[0]) if input_shape else 1
    model_seed = int(args.seed) + model_index
    return {
        "profile_dataset_format_version": 1,
        "model_id": row["model_id"],
        "graph_id": row.get("graph_id", row["model_id"]),
        "original_stem": row.get("original_stem", row.get("stem")),
        "input_shape": input_shape,
        "input_specs": input_specs,
        "batch_size": batch_size,
        "train_repeats": int(args.train_repeats),
        "train_samples": batch_size * int(args.train_repeats),
        "infer_repeats": int(args.infer_repeats),
        "infer_samples": batch_size * int(args.infer_repeats),
        "seed": model_seed,
        "dtype": "float32",
        "target": "zeros",
        "purpose": "resource_utilization_and_step_time",
    }


def write_specs(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        spec = make_spec(row, args, idx)
        path = out_dir / f"{spec['model_id']}.json"
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; pass --force to overwrite")
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        specs.append(spec)

    index_path = out_dir / "index.jsonl"
    if index_path.exists() and not args.force:
        raise FileExistsError(f"{index_path} exists; pass --force to overwrite")
    with index_path.open("w") as fh:
        for spec in specs:
            fh.write(json.dumps(spec, sort_keys=True) + "\n")

    summary = {
        "profile_dataset_format_version": 1,
        "models": len(specs),
        "train_repeats": int(args.train_repeats),
        "infer_repeats": int(args.infer_repeats),
        "seed": int(args.seed),
    }
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"{summary_path} exists; pass --force to overwrite")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_unique_model_rows(Path(args.manifest), args.limit)
    summary = write_specs(rows, args)
    print(
        "wrote "
        f"{summary['models']} profile dataset specs to {Path(args.output_dir)} "
        f"({summary['train_repeats']} train repeats, "
        f"{summary['infer_repeats']} inference repeats)",
        flush=True,
    )


if __name__ == "__main__":
    main()
