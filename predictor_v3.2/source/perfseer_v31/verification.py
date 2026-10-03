"""Bounded, real-data CUDA checks. These never replace production training."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
import time

import torch

from perfseer_v3.features import fit_normalization, apply_normalization
from perfseer_v3.op_registry import OperationRegistry

from .dataset import verify
from .features import build_features
from .inference import export_model, predict
from .io import atomic_write, read_json
from .model import SeerNetV31, SeerNetV31Config
from .runner import Samples
from .training import (T1, S1, build_optimizers, checkpoint_payload, evaluate,
                       regression_loss, restore_model, to_batch, train_batch)
from .version import TARGET_NAMES


def _verify_features(path):
    torch.set_num_threads(1)
    features = build_features(read_json(path))
    features.validate()
    tensor_bytes = sum(getattr(features, field.name).numel() * getattr(features, field.name).element_size()
                       for field in fields(features) if isinstance(getattr(features, field.name), torch.Tensor))
    return features.x_cont.size(0), features.edge_cont.size(0), tensor_bytes


def run(dataset, output, *, mode="smoke", steps=200, device="cuda"):
    verified = verify(dataset)
    manifest = read_json(dataset / "dataset_manifest.json")
    if mode == "features":
        paths = sorted({row["input_path"] for split in manifest["split_files"].values()
                        for row in read_json(dataset / split["path"])})
        with ProcessPoolExecutor(max_workers=8) as pool:
            sizes = list(pool.map(_verify_features, [dataset / path for path in paths]))
        report = {"passed": True, "dataset_fingerprint": manifest["fingerprint"],
                  "unique_inputs": len(paths), "max_nodes": max(size[0] for size in sizes),
                  "max_edges": max(size[1] for size in sizes), "total_nodes": sum(size[0] for size in sizes),
                  "total_feature_tensor_bytes": sum(size[2] for size in sizes)}
        atomic_write(output / "features-report.json", report)
        return report
    rows = read_json(dataset / manifest["split_files"]["train"]["path"])
    # Distinct small graphs bound the experiment's time and memory, not the
    # production corpus. Their IDs and selection rule are persisted explicitly.
    selected, seen = [], set()
    for row in sorted(rows, key=lambda row: ((dataset / row["input_path"]).stat().st_size, row["sample_id"])):
        if row["group_id"] not in seen:
            selected.append(row)
            seen.add(row["group_id"])
        if len(selected) == (8 if mode == "overfit" else 2):
            break
    raw = Samples(dataset, selected)
    normalization = fit_normalization([raw[i]["features"] for i in range(len(raw))], split_name="train", split_fingerprint=manifest["fingerprint"])
    samples = [{**raw[i], "features": apply_normalization(raw[i]["features"], normalization)} for i in range(len(raw))]
    scales = torch.stack([sample["target"] for sample in samples]).median(0).values
    scales[1] = 100.
    report = {"mode": mode, "torch": torch.__version__, "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "dataset_fingerprint": manifest["fingerprint"], "sample_ids": [row["sample_id"] for row in selected],
              "selection": "distinct architecture groups, smallest compressed graphs, training split only", "models": {}}
    for role, preset in (("teacher", T1), ("student", S1)):
        torch.manual_seed(42)
        config = SeerNetV31Config.from_registry(OperationRegistry.load(), samples[0]["features"].layout, **preset)
        model = SeerNetV31(config, scales).to(device)
        optimizers, _ = build_optimizers(model, 5e-4 if role == "teacher" else 1e-3)
        initial = evaluate(model, samples, len(samples))
        best = initial
        started = time.monotonic()
        history = []
        for step in range(steps if mode == "overfit" else 2):
            model.train()
            loss, microbatch = train_batch(model, samples, optimizers, len(samples))
            metrics = evaluate(model, samples, microbatch)
            if max(metrics["errors"]) < max(best["errors"]):
                best = metrics
            history.append({"step": step + 1, "loss": loss, "errors": metrics["errors"]})
            if step % 10 == 0:
                print(f"{mode} {role} step={step + 1} errors={metrics['errors']}", flush=True)
        checkpoint = checkpoint_payload(model, role=role, epoch=0, normalization=asdict(normalization), dataset_fingerprint=manifest["fingerprint"])
        artifact_path = output / f"{mode}-{role}-export.pt"
        export_model(checkpoint, artifact_path)
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        reloaded = restore_model(artifact, dataset_fingerprint=manifest["fingerprint"], device=device).eval()
        expected = model.eval().predict_batch(to_batch(samples, device)).prediction
        actual = reloaded.predict_batch(to_batch(samples, device)).prediction
        torch.testing.assert_close(expected, actual)
        assert actual.shape == (len(samples), 3)
        del reloaded
        report["models"][role] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                                   "initial": initial, "best": best, "last": metrics, "history": history,
                                   "elapsed_seconds": time.monotonic() - started, "checkpoint_reload": True}
        del model, optimizers, checkpoint, artifact, expected, actual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["passed"] = mode == "smoke" or all(value["best"]["gate_passed"] for value in report["models"].values())
    atomic_write(output / f"{mode}-report.json", report)
    if not report["passed"]:
        raise RuntimeError("bounded overfit gate failed; inspect report before production submission")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "overfit", "features"), default="smoke")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    print(run(args.dataset, args.output, mode=args.mode, steps=args.steps))


if __name__ == "__main__":
    main()
