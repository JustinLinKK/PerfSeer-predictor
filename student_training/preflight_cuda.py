#!/usr/bin/env python3
"""Deferred parent-lock-owned CUDA feasibility check. Never acquires/creates a GPU lock."""
import argparse
import copy
import gc
import json
import tempfile
from pathlib import Path
import torch
from support import (HERE, read_config, load_data, split_rows, initialized_metadata, create_student,
                     packed, sha, signature, code_hash, write_json, legacy)
from design import SELECTED, PRETRAIN, WINNER_ID
from losses import loss, task_reliability
from teacher import load_teacher, Tap
from export import export_verified


def largest(data, rows, count):
    # Scores use only input graph shape, never labels or validation membership.
    return torch.tensor(sorted(map(int, rows), key=lambda i:
        (-(len(data["features"][i].x_cont) + data["features"][i].edge_index.shape[1]), i))[:count])


def targets(config, base, data, rows, device):
    _, teacher, _ = load_teacher(config, device)
    tap = Tap(teacher)
    failures = []
    try:
        for micro in (8, 4, 2, 1):
            try:
                values = {}
                for chunk in rows.split(micro):
                    with torch.no_grad(), base.deterministic_evaluation():
                        outputs = tap(base.to_batch(data["features"], chunk, device),
                                      torch.as_tensor(data["raw_steps_per_epoch"])[chunk].to(device),
                                      data["epoch_protocol"])
                    for name, value in outputs.items():
                        values.setdefault(name, []).append(value.detach().cpu())
                return {name: torch.cat(value) for name, value in values.items()}, micro, failures
            except torch.cuda.OutOfMemoryError as error:
                if device != "cuda":
                    raise
                failures.append({"teacher_micro_batch": micro, "error": str(error)})
                tap.values.clear()
                values = {}
                outputs = None
                gc.collect()
                torch.cuda.empty_cache()
        raise RuntimeError("Teacher preflight OOM at every allowed microbatch")
    finally:
        tap.close()
        del teacher, tap
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def probe_case(config, data, base, rows, candidate, teacher, device, output):
    design = PRETRAIN if candidate == "PRETRAIN" else SELECTED
    metadata = initialized_metadata(config, data, rows)
    torch.manual_seed(0)
    model = create_student(metadata, design, config.get("initial_student"), sha(config["normalization"])).to(device)
    frozen = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if not value.requires_grad}
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=design["lr"])
    labels = torch.as_tensor(data["targets"])[rows].float().to(device)
    cache = {k: v[:len(rows)].to(device) for k, v in teacher.items()}
    reliability = task_reliability(cache["prediction"], labels) if design["objective"] == "task_pair" else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prediction = model(*packed(base, data, rows, device))
    objective, terms = loss(prediction, cache, labels, design["objective"], reliability)
    if not torch.isfinite(objective):
        raise FloatingPointError("Nonfinite CUDA preflight objective")
    objective.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any((g != 0).any() for g in gradients):
        raise AssertionError("Missing/nonfinite/zero CUDA gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    if not any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters()
               if value.requires_grad):
        raise AssertionError("No actual optimizer update")
    for name, value in model.named_parameters():
        if name in frozen and not torch.equal(value.detach().cpu(), frozen[name]):
            raise AssertionError("Frozen source parameter changed")
    for name, value in model.named_buffers():
        if not torch.equal(value.detach().cpu(), buffers[name]):
            raise AssertionError("Frozen/source buffer changed")
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    result = {"candidate": candidate, "batch_size": len(rows), "rows_sha256": signature(rows.tolist()),
              "largest_nodes": max(len(data["features"][int(i)].x_cont) for i in rows),
              "finite_nonzero_gradients": True, "optimizer_update": True, "frozen_parameters_unchanged": True,
              "buffers_unchanged": True, "peak_cuda_allocated_bytes": peak, "loss": float(objective.detach())}
    del optimizer, prediction, gradients, cache, objective
    with torch.no_grad():
        before_export = model(*packed(base, data, rows[:4], device))["prediction"].detach().cpu()
    result["export"] = export_verified(model, base, data, rows[:4], output,
                                        {"candidate": candidate, "preflight": True})
    with torch.no_grad():
        after_export = model(*packed(base, data, rows[:4], "cpu"))["prediction"]
    torch.testing.assert_close(after_export, before_export, rtol=1e-4, atol=1e-3)
    result["export"]["training_device_cpu_parity"] = True
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-config", required=True)
    parser.add_argument("--pretrain-config", required=True, help="existing full-data A10G teacher config")
    parser.add_argument("--teacher-probe-checkpoint", required=True, help="existing T7; shape probe only, not production teacher")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu" and not args.smoke:
        parser.error("CPU preflight is smoke mechanics only; production requires CUDA")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    transfer = read_config(args.transfer_config)
    transfer["teacher"] = {**transfer["teacher"], "checkpoint": str(Path(args.teacher_probe_checkpoint).resolve())}
    pretrain = read_config(args.pretrain_config)
    bindings = {"code_sha256": code_hash(), "transfer_subset_sha256": sha(transfer["subset"]),
                "initial_student_sha256": sha(transfer["initial_student"]),
                "probe_teacher_sha256": sha(args.teacher_probe_checkpoint), "hardware": {}}
    for config in (transfer, pretrain):
        bindings["hardware"][config["hardware"]] = {k + "_sha256": sha(config[k])
            for k in ("dataset", "normalization", "architecture")}
    failures = []
    batches = (2,) if args.smoke else (32, 16, 8, 4, 2, 1)
    passed = False
    for batch in batches:
        results, teacher_micro = {}, {}
        try:
            with tempfile.TemporaryDirectory(prefix="cuda-preflight-", dir=Path(args.output).resolve().parent) as scratch:
                for config in (transfer, pretrain):
                    base, data = load_data(config)
                    data_sha = bindings["hardware"][config["hardware"]]["dataset_sha256"]
                    parts, _ = split_rows(config, data, data_sha)
                    scope_rows = {"transfer": largest(data, parts["train"], batch)} if config is transfer else {}
                    full_train = (torch.as_tensor(data["splits"]) == 0).nonzero().flatten()
                    scope_rows["pretrain"] = largest(data, full_train, batch)
                    for scope, rows in scope_rows.items():
                        teacher, micro, teacher_failures = targets(config, base, data, rows, args.device)
                        teacher_micro[config["hardware"] + "/" + scope] = micro
                        failures.extend(teacher_failures)
                        candidates = [WINNER_ID] if scope == "transfer" else ["PRETRAIN"]
                        for candidate in candidates:
                            name = config["hardware"] + "_" + candidate
                            result = probe_case(config, data, base, rows, candidate, teacher, args.device,
                                                Path(scratch) / name)
                            results[name] = result
                            legacy.log(event="student_preflight", name=name, batch_size=batch, passed=True)
                    del data, teacher
                    gc.collect()
                passed = True
                break
        except torch.cuda.OutOfMemoryError as error:
            if args.device != "cuda":
                raise
            failures.append({"student_batch_size": batch, "error": str(error)})
            # Discard the entire attempted comparison: fallback reruns every design at the same batch.
            data = teacher = None
            gc.collect()
            torch.cuda.empty_cache()
    if not passed:
        write_json({"passed": False, "bindings": bindings, "failures": failures}, args.output)
        raise RuntimeError("Student preflight OOM at every allowed batch")
    if bindings["code_sha256"] != code_hash() or bindings["probe_teacher_sha256"] != sha(args.teacher_probe_checkpoint):
        raise ValueError("Code or probe teacher changed during preflight")
    for config in (transfer, pretrain):
        if any(bindings["hardware"][config["hardware"]][key + "_sha256"] != sha(config[key])
               for key in ("dataset", "normalization", "architecture")):
            raise ValueError("Data/normalization/architecture changed during preflight")
    receipt = {"passed": True, "device": args.device, "smoke": args.smoke, "bindings": bindings,
               "batch_size": batch, "teacher_micro_batches": teacher_micro, "failures": failures,
               "cases": results, "scope": "TRAIN input shape/loss/export feasibility; proxy T7 is not final transfer teacher",
               "source_unchanged": True,
               "requires_explicit_config_batch_update": batch != 32, "caller_owns_shared_gpu_lock": True,
               "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" else None}
    write_json(receipt, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
