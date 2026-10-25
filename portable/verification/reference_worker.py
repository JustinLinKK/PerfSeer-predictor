#!/usr/bin/env python3
"""Development verifier: read an explicitly supplied historical factory on CPU.

This is never imported by the release runtime and is not needed for inference.
The caller supplies all reference paths; no checkout location is assumed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", type=Path, required=True)
    parser.add_argument("--kind", choices=("P1", "Adapted", "T7"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("historical_reference_factory", args.factory)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.kind == "T7":
        base, model, metadata = module.build("T7", device="cpu", checkpoint=args.checkpoint)
        prediction_function = module.previous.predict
    else:
        base, model, metadata = module.build(device="cpu", checkpoint=args.checkpoint)
        prediction_function = module.predict
    data = torch.load(args.input, weights_only=False, map_location="cpu")
    if any(key in data for key in ("targets", "raw_targets_v4", "splits", "families")):
        raise ValueError("verification probe must be input-only")
    predictions = prediction_function(base, model, data, list(range(len(data["features"]))), device="cpu", batch=2)
    np.savez_compressed(args.output, predictions=predictions, sample_ids=np.asarray(data["sample_ids"]))
    print(json.dumps({"kind": args.kind, "parameter_count": sum(p.numel() for p in model.parameters()),
                      "state_tensors": len(model.state_dict()), "rows": len(predictions),
                      "cpu_only": not torch.cuda.is_initialized()}))


if __name__ == "__main__":
    main()
