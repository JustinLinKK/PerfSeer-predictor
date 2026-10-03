"""Check actual T1/S1 distillation together in the immutable CUDA image."""

import argparse
from pathlib import Path

import torch

from perfseer_v31.io import atomic_write, read_json
from perfseer_v31.runner import Samples, normalization_from_dict
from perfseer_v31.training import build_optimizers, restore_model, train_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--largest", action="store_true")
    args = parser.parse_args()
    teacher_state = torch.load(args.artifacts / "smoke-teacher-export.pt", map_location="cpu", weights_only=False)
    student_state = torch.load(args.artifacts / "smoke-student-export.pt", map_location="cpu", weights_only=False)
    teacher = restore_model(teacher_state, device="cuda").eval().requires_grad_(False)
    student = restore_model(student_state, device="cuda").train()
    manifest = read_json(args.dataset / "dataset_manifest.json")
    assert teacher_state["dataset_fingerprint"] == student_state["dataset_fingerprint"] == manifest["fingerprint"]
    ids = set(read_json(args.artifacts / "smoke-report.json")["sample_ids"])
    training_rows = read_json(args.dataset / manifest["split_files"]["train"]["path"])
    audit = read_json(args.dataset / "conversion-audit.json")
    if args.largest:
        key = max(audit["captures"], key=lambda key: audit["captures"][key]["nodes"])
        # Probe input shapes only, with synthetic targets; held-out labels never
        # enter this disposable memory/gradient check or production training.
        rows = [{"input_path": f"models/{key}.json.gz", "sample_id": f"shape-only:{key}",
                 "targets": dict(zip(manifest["target_names"], (1000., 50., 1000.), strict=True))}]
    else:
        rows = [row for row in training_rows if row["sample_id"] in ids]
    samples = Samples(args.dataset, rows, normalization_from_dict(student_state["normalization"]))
    teacher_loss = None
    torch.cuda.reset_peak_memory_stats()
    if args.largest:
        teacher.train().requires_grad_(True)
        teacher_optimizers, _ = build_optimizers(teacher, .0005)
        teacher_loss, _ = train_batch(teacher, samples[:], teacher_optimizers, 1)
        teacher.zero_grad(set_to_none=True)
        del teacher_optimizers
        teacher.eval().requires_grad_(False)
        torch.cuda.empty_cache()
    optimizers, _ = build_optimizers(student, .001)
    loss, microbatch = train_batch(student, samples[:], optimizers, len(samples), teacher=teacher)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    name = "largest-graph-distillation.json" if args.largest else "paired-distillation.json"
    atomic_write(args.artifacts / name, {"passed": True, "loss": loss, "microbatch": microbatch,
                 "teacher_frozen": True, "teacher_training_loss": teacher_loss,
                 "shape_only": args.largest, "uses_heldout_labels": False,
                 "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                 "sample_ids": [row["sample_id"] for row in rows],
                 "nodes": sum(sample["features"].x_cont.size(0) for sample in samples[:]),
                 "torch": torch.__version__, "dataset_fingerprint": manifest["fingerprint"]})


if __name__ == "__main__":
    main()
