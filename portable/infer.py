#!/usr/bin/env python3
"""Input-only inference for the released P1, Adapted and T7 teacher designs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from teachers import ROOT, bootstrap, build, predict, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, default=ROOT / "architecture.pt")
    parser.add_argument("--kind", choices=("P1", "Adapted", "T7"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    bootstrap()
    base, model, metadata = build(args.kind, args.architecture, args.checkpoint, args.device)
    data = base.load_normalized_inputs(args.input, metadata["source_normalization"])
    forbidden = {"targets", "raw_targets_v4", "families", "splits", "configuration_groups", "release_partition"}
    if forbidden.intersection(data):
        raise ValueError("inference requires an input-only bundle, without labels or dataset partitions")
    values = predict(base, model, data, device=args.device, batch_size=args.batch_size)
    buckets = np.maximum(1, np.ceil(values[:, 4:6] * 1048576 / 100000000)).astype(np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_ids=np.asarray(data["sample_ids"]), predictions=values,
                        target_names=np.asarray(base.TARGET_NAMES),
                        target_units=np.asarray(["ms", "ms", "ms", "percent", "MiB", "MiB"]),
                        vram_100mb_buckets=buckets, vram_decimal_gb=buckets / 10,
                        raw_steps_per_epoch=np.asarray(data["raw_steps_per_epoch"]),
                        epoch_protocol=np.asarray(data["epoch_protocol"]),
                        normalization_sha256=np.asarray(metadata["source_normalization"].sha256),
                        checkpoint_sha256=np.asarray(sha(args.checkpoint)),
                        architecture_sha256=np.asarray(sha(args.architecture)), kind=np.asarray(args.kind),
                        uncertainty_status=np.asarray("NOT_CALIBRATED"))
    print(json.dumps({"rows": len(values), "kind": args.kind, "device": args.device,
                      "parameter_count": sum(p.numel() for p in model.parameters()), "finite": True}))


if __name__ == "__main__":
    main()
