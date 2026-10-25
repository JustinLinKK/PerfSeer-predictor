#!/usr/bin/env python3
"""Build an input-only verification probe from explicit TRAIN bundle rows."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teachers import bootstrap
bootstrap()
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.bundle, weights_only=False, mmap=True, map_location="cpu")
    train = torch.nonzero(data["splits"] == 0).flatten().tolist()
    # Two distinct family structures, bounded graph sizes. Labels never inspected.
    order = sorted(train, key=lambda i: len(data["features"][i].x_cont))
    chosen, families = [], set()
    for index in order:
        family = data["sample_ids"][index].split("_")[1]
        if family not in families:
            chosen.append(index)
            families.add(family)
        if len(chosen) == 2:
            break
    output = {"features": [data["features"][i] for i in chosen], "sample_ids": [data["sample_ids"][i] for i in chosen],
              "raw_steps_per_epoch": torch.as_tensor(data["raw_steps_per_epoch"])[chosen],
              "epoch_protocol": data["epoch_protocol"], "normalization": data["normalization"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({"rows": len(chosen), "sample_ids": output["sample_ids"], "source_indices": chosen,
                      "source_split": "TRAIN", "input_only": True}))


if __name__ == "__main__":
    main()
