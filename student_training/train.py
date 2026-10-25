import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
#!/usr/bin/env python3
"""Full-data D3 pretraining with validation-only checkpoint selection."""
import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from support import (read_config, provenance, load_data, split_rows, validation_key, initialized_metadata,
                     create_student, packed, predict, report, signature, code_hash, sha, atomic_save,
                     write_json, random_state, restore_random, legacy)
from cache import cache_binding, load_cache
from design import PRETRAIN
from losses import loss
from export import export_verified
from preflight_contract import verify_receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--crash-after-epoch", type=int, default=0)
    parser.add_argument("--preflight", help="required successful CUDA verifier receipt for CUDA production")
    args = parser.parse_args()
    config = read_config(args.config)
    candidate = config["candidate"]
    design = PRETRAIN
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(config["output"])
    if args.smoke:
        output = output.with_name(output.name + "-smoke")
    output.mkdir(parents=True, exist_ok=True)
    prov = provenance(config)
    base, data = load_data(config)
    parts, split_receipt = split_rows(config, data, prov["dataset_sha256"])
    train, validation = parts["train"], parts["validation"]
    if args.smoke:
        train, validation = train[:8], validation[:8]
    key = validation_key(config, data, validation)
    cache_dir = Path(config["cache"])
    if args.smoke:
        cache_dir = cache_dir.with_name(cache_dir.name + "-smoke")
    cache, positions, _ = load_cache(cache_dir, cache_binding(config, prov, train), train, len(data["sample_ids"]))
    prov["cache_manifest_sha256"] = sha(cache_dir / "manifest.json")
    metadata = initialized_metadata(config, data, train)
    batch = 4 if args.smoke else int(config.get("batch_size", 32))
    preflight_hash = verify_receipt(args.preflight, config, prov, batch) if args.device == "cuda" and not args.smoke else None
    model = create_student(metadata, design, config.get("initial_student"), prov["normalization_sha256"]).to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    targets = torch.as_tensor(data["targets"]).float()
    reliability = None
    epochs = 2 if args.smoke else int(config.get("epochs", 40))
    decay = float(config.get("ema_decay", .999))
    lr = float(config.get("lr", design["lr"]))
    resolved = {**config, "design": design, "epochs": epochs, "batch_size": batch, "ema_decay": decay,
                "lr": lr, "seed": 0, "smoke": args.smoke, "device": args.device, "code_sha256": code_hash(),
                "preflight_sha256": preflight_hash,
                "provenance": prov, "splits": split_receipt, "initialization": metadata["initialization"],
                "selection": "maximum validation v5 composite; ties earlier epoch; heldout excluded",
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "task_reliability_train_only": None if reliability is None else reliability.tolist()}
    config_hash = signature(resolved)
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != resolved:
        raise ValueError("Output configuration/provenance changed")
    write_json(resolved, output / "config.json")
    if (output / "result.json").exists():
        done = json.loads((output / "result.json").read_text())
        if done["config_hash"] != config_hash or any(sha(output / p) != h for p, h in done["files"].items()):
            raise ValueError("Completed artifact changed")
        legacy.log(event="already_complete", candidate=candidate)
        return
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    total_steps = epochs * math.ceil(len(train) / batch)
    warmup = max(1, int(.03 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(1., (step + 1) / warmup) * .5 * (1 + math.cos(math.pi * min(1., step / total_steps))))
    state = {"epoch": 0, "global_step": 0, "best_score": -1., "best_epoch": 0, "history": []}
    last, selected = output / "last.pt", output / "best_model.pt"
    best = None
    if last.exists():
        saved = torch.load(last, weights_only=False, map_location="cpu")
        if saved["config_hash"] != config_hash:
            raise ValueError("Resume provenance changed")
        model.load_state_dict(saved["student"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state, best = saved["state"], saved["best"]
        restore_random(saved["rng"])
        atomic_save(best, selected)
        del saved
        legacy.log(event="resumed", candidate=candidate, epoch=state["epoch"], step=state["global_step"])

    def evaluate():
        return report(predict(ema, base, data, validation, args.device, batch), key)

    def capture(value, epoch):
        return {"state": {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()},
                "config": resolved, "metadata": {k: metadata[k] for k in ("vram_config", "physics_config", "initialization")}, "model_kind": design["model"],
                "epoch": epoch, "validation_score": value["score"], "validation": value}

    if best is None:
        value = evaluate()
        best, state["best_score"] = capture(value, 0), value["score"]
        state["history"].append({"epoch": 0, "score": value["score"]})
        atomic_save(best, selected)
    while state["epoch"] < epochs:
        epoch, started = state["epoch"], time.monotonic()
        model.train()
        order = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(20261024 + epoch))]
        totals, updates = {}, 0
        for rows in order.split(batch):
            teacher = {k: v[positions[rows]].to(args.device) for k, v in cache.items()}
            prediction = model(*packed(base, data, rows, args.device))
            value, terms = loss(prediction, teacher, targets[rows].to(args.device), design["objective"], reliability)
            if not torch.isfinite(value) or not all(math.isfinite(x) for x in terms.values()):
                raise FloatingPointError("Nonfinite distillation objective")
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters(), strict=True):
                    average.lerp_(current, 1 - decay)
                for average, current in zip(ema.buffers(), model.buffers(), strict=True):
                    average.copy_(current)
            updates += 1
            state["global_step"] += 1
            for name, number in {"loss": float(value.detach()), "grad_norm": float(norm), **terms}.items():
                totals[name] = totals.get(name, 0.) + number
            if state["global_step"] % 100 == 0:
                legacy.log(event="train", candidate=candidate, epoch=epoch + 1, step=state["global_step"], loss=float(value.detach()))
        validation_report = evaluate()
        if not math.isfinite(validation_report["score"]):
            raise FloatingPointError("Nonfinite validation score")
        entry = {"epoch": epoch + 1, "score": validation_report["score"],
                 "train": {k: v / updates for k, v in totals.items()}, "lr": scheduler.get_last_lr()[0],
                 "seconds": time.monotonic() - started}
        state["history"].append(entry)
        if validation_report["score"] > state["best_score"]:
            state["best_score"], state["best_epoch"] = validation_report["score"], epoch + 1
            best = capture(validation_report, epoch + 1)
        state["epoch"] = epoch + 1
        atomic_save({"student": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "state": state, "best": best, "rng": random_state(),
                     "config_hash": config_hash}, last)
        atomic_save(best, selected)
        write_json(state["history"], output / "history.json")
        legacy.log(event="epoch", candidate=candidate, **entry, best_score=state["best_score"])
        if args.crash_after_epoch == epoch + 1:
            os._exit(99)
    ema.load_state_dict(best["state"])
    replay = evaluate()
    if abs(replay["score"] - best["validation_score"]) > 1e-12:
        raise AssertionError("Selected checkpoint validation replay changed")
    current = provenance(config)
    if any(current[k] != prov[k] for k in current):
        raise ValueError("Source changed while training")
    verification = export_verified(ema, base, data, validation, output,
                                   {"version": "release_v1_student", "candidate": candidate, "hardware": config["hardware"],
                                    "config_hash": config_hash, "best_epoch": state["best_epoch"]})
    result = {"candidate": candidate, "config_hash": config_hash, "config": resolved,
              "best_epoch": state["best_epoch"], "best_validation_score": replay["score"], "validation": replay,
              "completed_epochs": state["epoch"], "optimizer_updates": state["global_step"],
              "replay_verified": True, "source_unchanged": True, "verification": verification,
              "heldout_evaluated": False, "files": {p: sha(output / p) for p in ("best_model.pt", "student.ts")}}
    write_json(result, output / "result.json")
    # Keep the transaction until winner export/cleanup: enables full resume-state auditing.
    legacy.log(event="complete", candidate=candidate, best_validation_score=replay["score"])


if __name__ == "__main__":
    main()
