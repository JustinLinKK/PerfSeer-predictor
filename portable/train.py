"""Portable selected-teacher recipes: full target continuation or T7 transfer.

All paths are supplied explicitly. Production defaults preserve the release
recipes; --smoke is a bounded CPU-capable verification mode, never a full study.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from teachers import ROOT, initialize_target, predict, bootstrap
from io_utils import atomic_save, write_json, log, sha, summary, rng_state, restore_rng
from training_support import (BUCKET_MIB, metrics, pairwise_loss, log_interval_loss, memory_loss,
                              pretrain_parameter_groups, t7_parameter_groups)


def code_hash():
    digest = hashlib.sha256()
    for name in ("train.py", "teachers.py", "teacher_components.py", "training_support.py", "io_utils.py",
                 "base_loader.py", "aba3_training_support.py", "v5_objectives.py", "model/starter.py"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def score(base, model, data, rows, device):
    prediction = predict(base, model, data, rows, device=device, batch_size=16)
    return metrics(prediction, torch.as_tensor(data["targets"])[rows].double().numpy(),
                   raw_targets=np.asarray(data["raw_targets_v4"], dtype=np.float64)[rows],
                   families=np.asarray(data["families"])[rows])


def partitions(data, subset=None):
    if subset is None:
        result = [torch.nonzero(data["splits"] == n).flatten().tolist() for n in (0, 1)] + [[]]
        if {data["configuration_groups"][i] for i in result[0]} & {data["configuration_groups"][i] for i in result[1]}:
            raise ValueError("configuration-group leakage")
        return result
    specification = json.loads(Path(subset).read_text())
    lookup = {identifier: i for i, identifier in enumerate(data["sample_ids"])}
    if len(lookup) != len(data["sample_ids"]):
        raise ValueError("duplicate sample identities")
    result = []
    for name in ("train", "validation", "holdout"):
        if name + "_ids" not in specification:
            raise ValueError("portable transfer requires explicit sample-ID lists in the subset manifest")
        result.append([lookup[x] for x in specification[name + "_ids"]])
    for indices, expected_split in zip(result, (0, 1, 1)):
        if len(indices) != len(set(indices)) or any(int(data["splits"][i]) != expected_split for i in indices):
            raise ValueError("subset duplicates or frozen split mismatch")
    groups = [{data["configuration_groups"][i] for i in indices} for indices in result]
    if groups[0] & (groups[1] | groups[2]) or groups[1] & groups[2]:
        raise ValueError("configuration-group leakage")
    return result


def main(recipe):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Raw P_A100/P1 warm-start checkpoint")
    parser.add_argument("--architecture", type=Path, default=ROOT / "architecture.pt")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=recipe == "T7")
    parser.add_argument("--hardware", default="A10", help="A10/A10G/A100 for historical packs, or exact measured_hardware_id for a fresh pack")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--crash-after-epoch", type=int, default=0)
    parser.add_argument("--keep-resume-for-verification", action="store_true")
    parser.add_argument("--test-inputs", type=Path)
    parser.add_argument("--test-key", type=Path)
    args = parser.parse_args()
    if args.keep_resume_for_verification and not args.smoke:
        parser.error("--keep-resume-for-verification is restricted to smoke runs")
    if args.micro_batch not in (1, 2, 4, 8):
        parser.error("micro batch must be 1, 2, 4 or 8")
    if bool(args.test_inputs) != bool(args.test_key):
        parser.error("test inputs and test key must be supplied together")
    if recipe == "T7" and args.hardware != "A10":
        parser.error("the released transfer route is A100 to A10 only")
    if recipe == "full" and args.subset:
        parser.error("full target continuation uses the complete TRAIN split")
    bootstrap()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out, device = args.output, args.device
    out.mkdir(parents=True, exist_ok=True)
    source_hash, architecture_hash = sha(args.source), sha(args.architecture)
    base, model, metadata = initialize_target(args.source, args.architecture, transfer=recipe == "T7", device=device)
    data = torch.load(args.dataset, weights_only=False, mmap=True, map_location="cpu")
    measured_hardware = data.get("manifest", {}).get("measured_hardware_id")
    if measured_hardware and args.hardware != measured_hardware:
        raise ValueError("fresh dataset hardware differs from the declared training target")
    expected_normalization = metadata["source_normalization"].sha256
    if data["normalization"].sha256 != expected_normalization or any(
            f.metadata.get("normalization_sha256") != expected_normalization for f in data["features"]):
        raise ValueError("training features must use the exact frozen source normalization")
    if recipe == "full" and "release_partition" in data and not args.smoke:
        raise ValueError("full continuation requires the full target dataset, not a compact transfer cache")
    train, validation, holdout = partitions(data, args.subset)
    if args.smoke:
        train = sorted(train, key=lambda i: len(data["features"][i].x_cont))[:4]
        validation = sorted(validation, key=lambda i: len(data["features"][i].x_cont))[:4]
        holdout = []
    if not train or not validation:
        raise ValueError("nonempty TRAIN and validation splits are required")
    epochs = (2 if recipe == "full" else 3) if args.smoke else (30 if recipe == "full" else 60)
    micro = min(args.micro_batch, 2) if args.smoke else args.micro_batch
    effective = 2 if args.smoke else (64 if recipe == "full" else 32)
    accumulation = effective // micro
    batches = math.ceil(len(train) / effective)
    targets = torch.as_tensor(data["targets"]).float()
    raw_train_memory = torch.as_tensor(np.asarray([data["raw_targets_v4"][i][4:6] for i in train]), dtype=torch.float32)
    raw_index = torch.full((len(targets),), -1, dtype=torch.long)
    raw_index[torch.tensor(train)] = torch.arange(len(train))
    class_weights = base.balanced_binary_weights((targets[train, 3] > 0).float().mean(), device)
    name = "T7" if recipe == "T7" else "P_" + args.hardware
    config = {"candidate": name, "recipe": recipe, "hardware": args.hardware, "epochs": epochs,
              "micro_batch": micro, "accumulate": accumulation, "seed": 0, "smoke": args.smoke, "device": device,
              "train_rows": len(train), "validation_rows": len(validation), "holdout_rows": len(holdout),
              "source_sha256": source_hash, "architecture_sha256": architecture_hash,
              "dataset_sha256": sha(args.dataset), "subset_sha256": sha(args.subset) if args.subset else None,
              "code_sha256": code_hash(), "initialization": "Explicit P_A100/P1 warm-start; not from scratch",
              "selection": "validation v5 score; ties retain earlier epoch", "hardware_identity_override": 190,
              "normalization": "frozen A10 source normalization; original internal A100 memory renormalization",
              "resume": "atomic completed-epoch cursor, optimizer, RNG, history and best state",
              "test_inputs_sha256": sha(args.test_inputs) if args.test_inputs else None,
              "test_key_sha256": sha(args.test_key) if args.test_key else None}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
        raise ValueError("refusing incompatible configuration/resume")
    if (out / "result.json").exists():
        done = json.loads((out / "result.json").read_text())
        if done["checkpoint_sha256"] != sha(out / "best_model.pt"):
            raise ValueError("completed checkpoint hash differs")
        log(event="already_complete", candidate=name)
        return
    write_json(config, out / "config.json")
    reference = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.startswith("p1.vram9.")} if recipe == "full" else {}
    anchor, correct = None, None
    if recipe == "full":
        anchor_path = out / "anchor.pt"
        if anchor_path.exists():
            cached = torch.load(anchor_path, weights_only=False, map_location="cpu")
            if cached["config_hash"] != config_hash:
                raise ValueError("anchor configuration changed")
            anchor = cached["prediction"]
        else:
            anchor = torch.zeros((len(targets), 2))
            anchor[train] = torch.tensor(predict(base, model, data, train, device=device)[:, 4:6], dtype=torch.float32)
            atomic_save({"config_hash": config_hash, "prediction": anchor}, anchor_path)
        correct = (torch.ceil(anchor / BUCKET_MIB).clamp_min(1) * BUCKET_MIB - targets[:, 4:6]).abs() < 1e-3
    last, best_path = out / "last.pt", out / "best_model.pt"
    saved, start_epoch, step, best_score, best_epoch = None, 0, 0, -1., 0
    history = []
    if last.exists():
        saved = torch.load(last, weights_only=False, map_location="cpu")
        if saved["config_hash"] != config_hash or saved["source_sha256"] != source_hash:
            raise ValueError("resume lineage/configuration mismatch")
        if saved["cursor"]["next_batch"] != 0:
            raise ValueError("only completed-epoch checkpoints are supported")
        model.load_state_dict(saved["state"], strict=True)
        start_epoch, step = saved["cursor"]["epoch"], saved["global_step"]
        history, best_score, best_epoch = saved["history"], saved["best_score"], saved["best_epoch"]
        restore_rng(saved["rng"])
        log(event="resumed", candidate=name, epoch=start_epoch, global_step=step)
    else:
        report = score(base, model, data, validation, device)
        best_score = report["score"]
        history = [{"epoch": 0, **summary(report)}]
        atomic_save({"state": model.state_dict(), "config": config, "validation_score": best_score, "epoch": 0}, best_path)
        write_json(history, out / "history.json")
        log(event="baseline", candidate=name, **summary(report))
    last_stage, optimizer, refs = None, None, {}
    for epoch in range(start_epoch, epochs):
        calibrate = recipe == "T7" and epoch < (1 if args.smoke else 5)
        stage = "full" if recipe == "full" else f"calibrate={calibrate}"
        if stage != last_stage:
            groups = pretrain_parameter_groups(model) if recipe == "full" else t7_parameter_groups(model, calibrate)
            optimizer = torch.optim.AdamW(groups)
            initial_lrs, stage_step = [g["lr"] for g in groups], 0
            if saved is not None and saved["stage"] == stage:
                optimizer.load_state_dict(saved["optimizer"])
                groups, initial_lrs, stage_step = optimizer.param_groups, saved["initial_lrs"], saved["stage_step"]
            refs = {n: reference[n].to(device) for n, p in model.named_parameters() if p.requires_grad and n in reference}
            last_stage = stage
            log(event="stage", candidate=name, epoch=epoch, stage=stage,
                trainable={g["name"]: sum(p.numel() for p in g["params"]) for g in groups})
        generator = torch.Generator().manual_seed((20261023 if recipe == "full" else 20261024) + epoch)
        order = torch.tensor(train)[torch.randperm(len(train), generator=generator)]
        started, updates, sums = time.monotonic(), 0, {}
        for begin in range(0, len(order), effective):
            optimizer.zero_grad(set_to_none=True)
            chunks = list(order[begin:begin + effective].split(micro))
            term_sums = {}
            for idx in chunks:
                batch = base.to_batch(data["features"], idx, device)
                y = targets[idx].to(device)
                model.eval()
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
                with context:
                    pred, corr, tp, bl, bp, sp = model(batch, torch.as_tensor(data["raw_steps_per_epoch"])[idx].to(device), data["epoch_protocol"], True)
                pred, bl, bp = pred.float(), bl.float(), bp.float()
                lt = pred[:, :2].clamp_min(1e-8).log()
                terms = {"timing": base.timing_loss(pred[:, :2], y[:, :2], corr.float(), tp.float()) + .5 * (pairwise_loss(lt[:, 0], y[:, 0]) + pairwise_loss(lt[:, 1], y[:, 1])),
                         "busy": base.sm_loss(bl, bp, y[:, 3], sp.float(), class_weights) + .5 * base.busy_pair_loss(torch.sigmoid(bl) * bp / 10, y[:, 3])}
                if recipe == "full":
                    terms["vram"] = log_interval_loss(pred[:, 4:6], y[:, 4:6])
                    mask, ref = correct[idx].to(device).float(), anchor[idx].to(device)
                    gap = F.smooth_l1_loss(pred[:, 4:6].clamp_min(1).log(), ref.clamp_min(1).log(), beta=.01, reduction="none")
                    terms["anchor"] = (mask * gap).sum() / mask.sum().clamp_min(1)
                    terms["l2sp"] = .01 * sum((param - refs[n]).square().sum() for n, param in model.named_parameters() if n in refs)
                else:
                    terms["memory"] = memory_loss(pred[:, 4:6], y[:, 4:6], raw_train_memory[raw_index[idx]].to(device))
                loss = sum(terms.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss / len(chunks)).backward()
                for key, value in terms.items():
                    term_sums[key] = term_sums.get(key, 0.) + float(value.detach()) / len(chunks)
            finite_nonzero = False
            for group, lr in zip(groups, initial_lrs):
                warm_epochs = (5 if group["name"].startswith("vram") else 2) if recipe == "full" else (3 if group["name"].startswith("memory") else 2)
                fraction = (epoch + updates / max(1, batches)) / epochs
                group["lr"] = lr * min(1., (stage_step + 1) / max(1, warm_epochs * batches)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
                norm = torch.nn.utils.clip_grad_norm_(group["params"], 1., error_if_nonfinite=True)
                finite_nonzero |= bool(norm > 0)
            if not finite_nonzero:
                raise RuntimeError("no group received a real gradient")
            optimizer.step(); stage_step += 1; step += 1; updates += 1
            for key, value in term_sums.items():
                sums[key] = sums.get(key, 0.) + value
            if step % 100 == 0:
                log(event="train", candidate=name, epoch=epoch + 1, step=step, loss=sum(term_sums.values()))
        report = score(base, model, data, validation, device)
        row = {"epoch": epoch + 1, "stage": stage, **summary(report), "train": {k: v / updates for k, v in sums.items()},
               "seconds": time.monotonic() - started}
        history.append(row)
        if report["score"] > best_score:
            best_score, best_epoch = report["score"], epoch + 1
            atomic_save({"state": model.state_dict(), "config": config, "validation_score": best_score, "epoch": best_epoch}, best_path)
        write_json(history, out / "history.json")
        atomic_save({"state": model.state_dict(), "optimizer": optimizer.state_dict(), "cursor": {"epoch": epoch + 1, "next_batch": 0},
                     "global_step": step, "history": history, "best_score": best_score, "best_epoch": best_epoch, "rng": rng_state(),
                     "source_sha256": source_hash, "config_hash": config_hash, "stage": stage,
                     "stage_step": stage_step, "initial_lrs": initial_lrs}, last)
        log(event="epoch", candidate=name, **row, best_score=best_score, best_epoch=best_epoch)
        if args.crash_after_epoch == epoch + 1:
            os._exit(99)
        saved = None
    selected = torch.load(best_path, weights_only=False, map_location="cpu", mmap=True)
    model.load_state_dict(selected["state"], strict=True)
    replay = score(base, model, data, validation, device)
    if abs(replay["score"] - best_score) >= 1e-9 or sha(args.source) != source_hash:
        raise ValueError("selected checkpoint replay or source immutability failed")
    result = {"candidate": name, "best_validation_score": best_score, "best_epoch": best_epoch, "validation": replay,
              "replay_verified": True, "source_unchanged": True, "config": config, "completed_epochs": epochs,
              "checkpoint_sha256": sha(best_path), "finite_nonzero_gradients_verified": True}
    if holdout:
        result["historical_validation_holdout"] = score(base, model, data, holdout, device)
    if args.test_inputs and not args.smoke:
        test = base.load_normalized_inputs(args.test_inputs, torch.load(args.architecture, weights_only=False)["source_normalization"])
        with np.load(args.test_key, allow_pickle=False) as key:
            if not np.array_equal(np.asarray(test["sample_ids"]), key["sample_ids"]):
                raise ValueError("test identities differ")
            result["sealed_test"] = metrics(predict(base, model, test, device=device), key["targets"],
                                            raw_targets=key["raw_targets_v4"], families=key["families"])
    write_json(result, out / "result.json")
    if not args.keep_resume_for_verification:
        last.unlink(missing_ok=True)
        (out / "anchor.pt").unlink(missing_ok=True)
    log(event="complete", candidate=name, best_validation_score=best_score, best_epoch=best_epoch)
