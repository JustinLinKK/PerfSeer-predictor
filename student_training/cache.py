"""Create verified TRAIN-only teacher shards; never read heldout/test keys."""
import argparse
import json
from pathlib import Path
import torch
from support import sha, signature, write_json, atomic_save, read_config, provenance, load_data, split_rows, legacy
from teacher import load_teacher, Tap

FIELDS = ("prediction", "busy_logit", "busy_positive", "embedding")


def cache_binding(config, prov, rows):
    return {"teacher_checkpoint_sha256": prov["teacher_checkpoint_sha256"],
            "dataset_sha256": prov["dataset_sha256"], "normalization_sha256": prov["normalization_sha256"],
            "architecture_sha256": prov["architecture_sha256"], "hardware": config["hardware"],
            "teacher_identity": config["teacher"], "train_indices_sha256": signature(rows.tolist())}


def load_cache(directory, binding, rows, size):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["binding"] != binding or not all(manifest["checks"].values()):
        raise ValueError("Cache binding or verification mismatch")
    parts = []
    for name, digest in sorted(manifest["shards"].items()):
        path = directory / name
        if path.parent != directory or sha(path) != digest:
            raise ValueError("Cache shard path/checksum mismatch")
        part = torch.load(path, weights_only=True, map_location="cpu")
        if set(part) != {"indices", *FIELDS} or any(not torch.isfinite(v).all() for v in part.values()):
            raise ValueError("Invalid cache tensors")
        if any(len(part[k]) != len(part["indices"]) for k in FIELDS):
            raise ValueError("Unaligned cache tensors")
        parts.append(part)
    indices = torch.cat([p["indices"] for p in parts])
    if indices.dtype != torch.long or not torch.equal(indices, rows):
        raise ValueError("Cache must cover exactly the ordered TRAIN rows")
    positions = torch.full((size,), -1, dtype=torch.long)
    positions[indices] = torch.arange(len(indices))
    return {k: torch.cat([p[k] for p in parts]) for k in FIELDS}, positions, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = read_config(args.config)
    prov = provenance(config)
    base, data = load_data(config)
    partitions, _ = split_rows(config, data, prov["dataset_sha256"])
    rows = partitions["train"][:8] if args.smoke else partitions["train"]
    directory = Path(config["cache"])
    if args.smoke:
        directory = directory.with_name(directory.name + "-smoke")
    directory.mkdir(parents=True, exist_ok=True)
    binding = cache_binding(config, prov, rows)
    if (directory / "manifest.json").exists():
        load_cache(directory, binding, rows, len(data["sample_ids"]))
        legacy.log(event="cache_already_verified", directory=str(directory))
        return
    if (directory / "binding.json").exists():
        if json.loads((directory / "binding.json").read_text()) != binding:
            raise ValueError("Partial cache belongs to another teacher/dataset/split")
    else:
        if list(directory.glob("shard-*.pt")):
            raise ValueError("Unbound partial shards")
        write_json(binding, directory / "binding.json")
    base, teacher, _ = load_teacher(config, args.device)
    tap, shards = Tap(teacher), {}
    try:
        for start in range(0, len(rows), 1024):
            selected = rows[start:start + 1024]
            name = f"shard-{start:07d}.pt"
            path = directory / name
            receipt = directory / (name + ".json")
            if path.exists() and receipt.exists():
                record = json.loads(receipt.read_text())
                if record != {"sha256": sha(path), "binding": signature(binding), "indices": signature(selected.tolist())}:
                    raise ValueError("Partial shard receipt changed")
            else:
                values = {k: [] for k in ("indices", *FIELDS)}
                for chunk in selected.split(args.batch_size):
                    with torch.no_grad(), base.deterministic_evaluation():
                        result = tap(base.to_batch(data["features"], chunk, args.device),
                                     torch.as_tensor(data["raw_steps_per_epoch"])[chunk].to(args.device), data["epoch_protocol"])
                    result["indices"] = chunk
                    for key, value in result.items():
                        values[key].append(value.detach().cpu())
                values = {key: torch.cat(value) for key, value in values.items()}
                if any(not torch.isfinite(value).all() for value in values.values()):
                    raise FloatingPointError("Nonfinite teacher cache")
                atomic_save(values, path)
                write_json({"sha256": sha(path), "binding": signature(binding), "indices": signature(selected.tolist())}, receipt)
            shards[name] = sha(path)
            legacy.log(event="cache_shard", rows=min(start + 1024, len(rows)), total=len(rows))
        if provenance(config) != prov:
            raise ValueError("Teacher/data source changed while caching")
        manifest = {"binding": binding, "shards": shards, "rows": len(rows),
                    "scope": "TRAIN only; no validation or heldout predictions",
                    "checks": {"finite": True, "source_unchanged": True, "hash_receipts": True}}
        write_json(manifest, directory / "manifest.json")
        load_cache(directory, binding, rows, len(data["sample_ids"]))
    finally:
        tap.close()


if __name__ == "__main__":
    main()
