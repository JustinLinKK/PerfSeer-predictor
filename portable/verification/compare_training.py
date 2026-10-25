#!/usr/bin/env python3
"""Verify exact CPU interrupted/resumed training, including optimizer and RNG."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from io_utils import sha


def equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError("tensor mismatch: " + path)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise ValueError("array mismatch: " + path)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("mapping keys mismatch: " + path)
        for key in a:
            if key != "seconds":
                equal(a[key], b[key], path + "." + str(key))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError("sequence length mismatch: " + path)
        for index, (x, y) in enumerate(zip(a, b)):
            equal(x, y, path + "." + str(index))
    elif a != b:
        raise ValueError("value mismatch: " + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    a = torch.load(args.uninterrupted / "last.pt", weights_only=False, map_location="cpu", mmap=True)
    b = torch.load(args.recovered / "last.pt", weights_only=False, map_location="cpu", mmap=True)
    equal(a, b)
    selected_a = torch.load(args.uninterrupted / "best_model.pt", weights_only=False, map_location="cpu", mmap=True)
    selected_b = torch.load(args.recovered / "best_model.pt", weights_only=False, map_location="cpu", mmap=True)
    equal(selected_a, selected_b)
    initial = selected_a["state"]
    changed = [name for name, value in a["state"].items() if not torch.equal(value, initial[name])]
    if not changed:
        raise ValueError("no learned state change in the smoke trajectory")
    report_a = json.loads((args.uninterrupted / "result.json").read_text())
    report_b = json.loads((args.recovered / "result.json").read_text())
    if not (report_a["finite_nonzero_gradients_verified"] and report_b["finite_nonzero_gradients_verified"]):
        raise ValueError("missing real-gradient proof")
    if report_a["best_validation_score"] != report_b["best_validation_score"]:
        raise ValueError("selected scores differ")
    if report_a["candidate"] == "T7":
        illegal = [name for name in changed if not (name.startswith(("p1.timing.", "p1.sm.", "p1.vram9.base.memory_head."))
                   or name == "p1.vram9.offset" or "lora_" in name)]
        if illegal:
            raise ValueError("T7 changed frozen source parameters: " + str(illegal[:3]))
    result = {"passed": True, "candidate": report_a["candidate"], "device": "cpu", "cuda_used": False,
              "epochs": a["cursor"]["epoch"], "updates": a["global_step"], "state_tensors": len(a["state"]),
              "state_optimizer_rng_history_lr_cursor_exact": True, "selected_state_exact": True,
              "finite_nonzero_gradients_verified": True, "changed_tensors_from_selected_baseline": len(changed),
              "source_unchanged": report_a["source_unchanged"] and report_b["source_unchanged"],
              "uninterrupted_result_sha256": sha(args.uninterrupted / "result.json"),
              "recovered_result_sha256": sha(args.recovered / "result.json"),
              "portable_code_sha256": report_a["config"]["code_sha256"],
              "scope": "bounded real-data smoke; not evidence of production model quality"}
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
