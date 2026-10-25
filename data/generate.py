#!/usr/bin/env python3
"""Direct, deterministic workload construction; never reads a labeled dataset.

The transfer design asks the original family sampler for fewer configurations.
It does not generate a full pretraining set and then select dataset rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import itertools

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))
from _model_catalog import FAMILY_SPECS, sample_family_configs


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inputs(config, batch):
    # The exact original function is packaged with its runtime. Import lazily so
    # CLI --help and basic hashing do not import Torch.
    from _model_runtime import input_specs_from_config
    return input_specs_from_config(config, batch)


def render_source(identifier, config):
    # Byte-compatible with the original corpus generator, including final LF.
    return '''"""Generated multimodal PerfSeer model source."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch.nn as nn

_SOURCE_DIR = Path(__file__).resolve().parent
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))
_RUNTIME_PATH = _SOURCE_DIR / '_model_runtime.py'
_RUNTIME_SPEC = importlib.util.spec_from_file_location('_model_runtime', _RUNTIME_PATH)
if _RUNTIME_SPEC is None or _RUNTIME_SPEC.loader is None:
    raise ImportError(f'cannot load generated runtime at {_RUNTIME_PATH}')
_RUNTIME = importlib.util.module_from_spec(_RUNTIME_SPEC)
sys.modules[_RUNTIME_SPEC.name] = _RUNTIME
_RUNTIME_SPEC.loader.exec_module(_RUNTIME)

MODEL_ID = %r
MODEL_CONFIG = json.loads(%r)
INPUT_SPECS = json.loads(%r)
INPUT_SHAPE = tuple(INPUT_SPECS[0]['shape'])


def make_model() -> nn.Module:
    return _RUNTIME.make_model_from_config(MODEL_CONFIG)
''' % (identifier, canonical(config), canonical(inputs(config, 1)))


def configurations(mode="transfer", seed=42, per_family=16, families=None):
    for family, spec in FAMILY_SPECS.items():
        if families and family not in families:
            continue
        count = spec.quota if mode == "pretrain" else per_family
        for config in sample_family_configs(family, count, seed):
            yield config


def group_key(config):
    architecture = {k: v for k, v in config.items() if k not in ("batch_sizes", "sample_rank", "family_index")}
    return f"{config['family_index']:02d}:" + digest(architecture)


def render_variant(config, batch, precision, optimizer):
    model_id = f"model_{config['family_index']:02d}_{config['sample_rank']:04d}"
    identifier = f"{model_id}::bs{batch}::{precision}::{optimizer}"
    text = (HERE / f"specs/variant_templates/{config['family_index']:02d}.py.in").read_text()
    values = {"MODEL_ID": model_id, "PROFILE_POINT_ID": identifier, "MODEL_CONFIG": json.loads(canonical(config)),
              "BATCH_SIZE": batch, "PRECISION_CONFIG": precision, "OPTIMIZER_NAME": optimizer,
              "INPUT_SPECS": json.loads(canonical(inputs(config, batch)))}
    for key, value in values.items():
        text = text.replace("@" + key + "@", repr(value))
    return text.replace("@MODEL_ID_TEXT@", model_id)


def design(mode, seed, per_family, families, train_budget, splits):
    """Construct variants directly from architecture specs, without a row pool.

    In the budgeted transfer design, all TRAIN configurations receive one variant
    before any receive a second. A hash rotation balances the eight precision,
    optimizer, and batch choices. The only external input is frozen group splits.
    """
    configs = list(configurations("pretrain" if train_budget else mode, seed, per_family, families))
    if train_budget:
        configs = [c for c in configs if splits[group_key(c)] == 0]
        # Interleave families so partial second rounds do not privilege early families.
        grouped = [[c for c in configs if c["family_id"] == f] for f in FAMILY_SPECS]
        configs = [c for group in itertools.zip_longest(*grouped) for c in group if c is not None]
        if train_budget < len(configs) or train_budget > len(configs) * 8:
            raise ValueError(f"budget must cover every TRAIN config and be <= 8 variants/config: {len(configs)}..{len(configs)*8}")
        count = 0
        for round_number in range(8):
            for config in configs:
                variants = list(itertools.product(config["batch_sizes"], ("fp32_ieee", "fp16_amp"), ("adam", "muon")))
                offset = int(digest(config)[:8], 16) % 8
                yield config, variants[(round_number + offset) % 8]
                count += 1
                if count == train_budget:
                    return
    else:
        for config in configs:
            for variant in itertools.product(config["batch_sizes"], ("fp32_ieee", "fp16_amp"), ("adam", "muon")):
                yield config, variant


def generate(output, *, mode="transfer", seed=42, per_family=16, families=None, train_budget=None, split_file=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {output}")
    models = output / "models"
    models.mkdir(parents=True, exist_ok=True)
    (output / "variants").mkdir()
    for name in ("_model_catalog.py", "_model_runtime.py"):
        shutil.copyfile(HERE / "runtime" / name, models / name)
    rows, config_rows, seen = [], [], set()
    split_file = Path(split_file or HERE / "specs/frozen_group_splits.json")
    splits = json.loads(split_file.read_text())
    runtime_hash = sha(HERE / "runtime/_model_runtime.py")
    for config, (batch, precision, optimizer) in design(mode, seed, per_family, families, train_budget, splits):
        model_id = f"model_{config['family_index']:02d}_{config['sample_rank']:04d}"
        source = render_source(model_id, config)
        compile(source, model_id, "exec")
        if model_id not in seen:
            (models / (model_id + ".py")).write_text(source)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        if model_id not in seen:
            config_rows.append({"model_id": model_id, "config": config, "config_sha256": digest(config),
                               "source_sha256": source_hash})
            seen.add(model_id)
        architecture = {k: v for k, v in config.items() if k not in ("batch_sizes", "sample_rank")}
        variant = render_variant(config, batch, precision, optimizer)
        variant_file = f"variants/{model_id}__bs{batch}__{precision}__{optimizer}.py"
        (output / variant_file).write_text(variant)
        identity = {"architecture": architecture, "batch_size": batch,
                                "input_specs": inputs(config, batch), "precision_config": precision,
                                "optimizer": optimizer, "source_sha256": source_hash,
                                "runtime_sha256": runtime_hash,
                                "variant_source_sha256": hashlib.sha256(variant.encode()).hexdigest()}
        rows.append({**identity, "profile_point_id": f"{model_id}::bs{batch}::{precision}::{optimizer}",
                                 "model_id": model_id, "model_file": f"models/{model_id}.py",
                                 "variant_file": variant_file,
                                 "family_id": config["family_id"], "modality": config["modality"],
                                 "base_module": config["base_module"], "config_sha256": digest(config),
                                 "workload_sha256": digest(identity),
                                 "configuration_group": group_key(config), "frozen_split": splits.get(group_key(config)),
                                 "steps_per_epoch": 1, "dataset_kind": "synthetic"})
    for name, values in (("workloads.jsonl", rows), ("configurations.jsonl", config_rows)):
        (output / name).write_text("".join(canonical(row) + "\n" for row in values))
    actual_families = {r["family_id"] for r in rows}
    coverage = {family: {"base_modules": sorted({r["base_module"] for r in rows if r["family_id"] == family}),
                         "required_base_modules": sorted(FAMILY_SPECS[family].base_modules)}
                for family in sorted(actual_families)}
    summary = {"schema": "perfseer-direct-workloads-v1", "mode": mode, "seed": seed,
               "configs_per_family": None if mode == "pretrain" else per_family,
               "train_budget": train_budget, "split_file_sha256": sha(split_file),
               "variant_design": "one variant per TRAIN config, then balanced additional rounds" if train_budget else "full eight-variant factorial",
               "algorithm": "original deterministic family sampler; smaller count requested directly",
               "reads_pretraining_bundle": False, "labels_generated": False,
               "source_count": len(config_rows), "workload_count": len(rows),
               "families": dict(Counter(r["family_id"] for r in rows)),
               "modalities": dict(Counter(r["modality"] for r in rows)), "coverage": coverage,
               "all_base_modules_covered": all(v["base_modules"] == v["required_base_modules"] for v in coverage.values()),
               "files": {name: sha(output / name) for name in ("workloads.jsonl", "configurations.jsonl")},
               "generator_sha256": sha(__file__),
               "runtime_files": {p.name: sha(p) for p in sorted((HERE / "runtime").glob("*.py"))}}
    summary["variant_templates"] = {p.name: sha(p) for p in sorted((HERE / "specs/variant_templates").glob("*.in"))}
    summary["source_lineages"] = {"A10G_historical": "original base program + shared runtime", "A10_A100_historical": "standalone variant program"}
    (output / "generation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pretrain", "transfer"), default="transfer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs-per-family", type=int, default=16)
    parser.add_argument("--train-budget", type=int, help="Directly generate this many TRAIN variants, covering all TRAIN configurations first (release: 8192).")
    parser.add_argument("--split-file", type=Path, default=HERE / "specs/frozen_group_splits.json")
    parser.add_argument("--family", action="append", choices=tuple(FAMILY_SPECS))
    args = parser.parse_args()
    if args.configs_per_family < 1:
        parser.error("--configs-per-family must be positive")
    if args.train_budget and args.mode != "transfer":
        parser.error("--train-budget requires --mode transfer")
    result = generate(args.output, mode=args.mode, seed=args.seed, per_family=args.configs_per_family, families=args.family,
                      train_budget=args.train_budget, split_file=args.split_file)
    print(json.dumps({k: result[k] for k in ("mode", "source_count", "workload_count", "all_base_modules_covered")}))


if __name__ == "__main__":
    main()
