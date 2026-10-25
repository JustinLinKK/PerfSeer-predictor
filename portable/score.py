#!/usr/bin/env python3
"""Score already-generated predictions with the pinned v5 metric contract."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from v5_objectives import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as prediction, np.load(args.key, allow_pickle=False) as key:
        if not np.array_equal(prediction["sample_ids"], key["sample_ids"]):
            raise ValueError("prediction and key sample identities/order differ")
        result = metrics(prediction["predictions"], key["targets"],
                         raw_targets=key["raw_targets_v4"] if "raw_targets_v4" in key else None,
                         families=key["families"] if "families" in key else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"score": result["score"]}))


if __name__ == "__main__":
    main()
