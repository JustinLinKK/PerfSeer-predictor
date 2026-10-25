"""Shared, hash-bound study I/O. No heldout labels are read by training or cache code."""
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path[:0] = [str(HERE), str(RELEASE)]
from types import SimpleNamespace
from portable import teachers as portable_teachers
from portable.v5_objectives import metrics
def _log(**values):
    print(json.dumps(values, default=float), flush=True)
legacy = SimpleNamespace(load_base=portable_teachers.load_base, metrics=metrics, log=_log)
from runtime.student import student_from_teacher, pack_batch, INPUT_NAMES


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, default=float) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def code_hash():
    files = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob("*.py"))
             if "tests" not in p.parts and "receipts" not in p.parts}
    for path in sorted((RELEASE / "portable").rglob("*.py")):
        files["portable/" + str(path.relative_to(RELEASE / "portable"))] = sha(path)
    return signature(files)


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_random(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def read_config(path):
    config = json.loads(Path(path).read_text())
    required = ("candidate", "hardware", "teacher", "dataset", "normalization", "architecture",
                "validation_key", "cache", "output")
    if any(k not in config for k in required):
        raise ValueError(f"Required fields: {required}")
    teacher = config["teacher"]
    if set(("checkpoint", "kind", "hardware", "identity_override")) - teacher.keys():
        raise ValueError("Teacher needs checkpoint, kind, hardware and explicit identity_override (or null)")
    if teacher["hardware"] != config["hardware"]:
        raise ValueError("Teacher and target hardware differ")
    if teacher["kind"] not in ("P1", "Adapted", "T7", "transfer_T7"):
        raise ValueError("Teacher kind must be P1, Adapted or transfer_T7")
    if config["candidate"] != "PRETRAIN" and not config.get("subset"):
        raise ValueError("Transfer candidates require immutable subset.json")
    if config["candidate"] not in ('PRETRAIN', 'TS2'):
        raise ValueError("Unknown candidate; only the published transfer recipe is supported")
    if config["candidate"] != "PRETRAIN" and (config.get("route") != "A10_A100" or config["hardware"] != "A10"):
        raise ValueError("The authorized transfer route is A100 to A10")
    return config


def provenance(config):
    values = {name + "_sha256": sha(config[name])
              for name in ("dataset", "normalization", "architecture", "validation_key")}
    values.update(teacher_checkpoint_sha256=sha(config["teacher"]["checkpoint"]),
                  teacher_identity=config["teacher"], hardware=config["hardware"])
    for key in ("subset", "initial_student"):
        if config.get(key):
            values[key + "_sha256"] = sha(config[key])
    return values


def load_data(config):
    base = legacy.load_base()
    data = torch.load(config["dataset"], weights_only=False, map_location="cpu", mmap=True)
    if data["epoch_protocol"] != "measured_single_step":
        raise ValueError("D3 physical epoch contract requires measured_single_step")
    return base, data


def split_rows(config, data, data_sha):
    split = torch.as_tensor(data["splits"])
    if config.get("subset"):
        fixed = json.loads(Path(config["subset"]).read_text())
        if fixed["source_dataset_sha256"] != data_sha or fixed["hardware"] != config["hardware"]:
            raise ValueError("Subset dataset/hardware mismatch")
        parts = {k: torch.as_tensor(fixed[k + "_indices"], dtype=torch.long)
                 for k in ("train", "validation", "holdout")}
    else:
        fixed = {}
        parts = {"train": (split == 0).nonzero().flatten(), "validation": (split == 1).nonzero().flatten(),
                 "holdout": torch.empty(0, dtype=torch.long)}
    groups, ids = {}, {}
    for name, rows in parts.items():
        if len(rows) != len(rows.unique()) or (rows < 0).any() or (rows >= len(split)).any():
            raise ValueError("Partition has duplicate or invalid indices")
        if name != "holdout" and not len(rows):
            raise ValueError("Empty training/selection partition")
        if not (split[rows] == (0 if name == "train" else 1)).all():
            raise ValueError("Subset crosses frozen source train/validation boundary")
        ids[name] = [str(data["sample_ids"][int(i)]) for i in rows]
        groups[name] = sorted({str(data["configuration_groups"][int(i)]) for i in rows})
        if len(set(ids[name])) != len(ids[name]):
            raise ValueError("Duplicate sample identity")
        for suffix, observed in (("ids", ids[name]), ("groups", groups[name])):
            if name + "_" + suffix in fixed and observed != fixed[name + "_" + suffix]:
                raise ValueError("Immutable subset ID/group membership changed")
    names = list(parts)
    for i, name in enumerate(names):
        for other in names[:i]:
            if set(ids[name]) & set(ids[other]) or set(groups[name]) & set(groups[other]):
                raise ValueError("Partition sample/configuration groups overlap")
    return parts, {k: {"indices_sha256": signature(v.tolist()), "ids_sha256": signature(ids[k]),
                      "groups_sha256": signature(groups[k]), "rows": len(v)} for k, v in parts.items()}


def validation_key(config, data, rows):
    with np.load(config["validation_key"], allow_pickle=False) as key:
        names = list(map(str, key["sample_ids"].tolist()))
        if len(names) != len(set(names)):
            raise ValueError("Duplicate validation-key IDs")
        lookup = {s: i for i, s in enumerate(names)}
        indices = np.asarray([lookup[str(data["sample_ids"][int(i)])] for i in rows])
        value = {k: key[k][indices] for k in ("targets", "raw_targets_v4", "families")}
    torch.testing.assert_close(torch.as_tensor(data["targets"])[rows].double(),
                               torch.as_tensor(value["targets"]).double(), rtol=0, atol=1e-4)
    return value


def initialized_metadata(config, data, rows):
    metadata = copy.deepcopy(torch.load(config["architecture"], weights_only=False, map_location="cpu"))
    target = torch.as_tensor(data["targets"])[rows].float()
    positive = target[:, 3] > 0
    metadata["initialization"] = {"target_medians": target.median(0).values.tolist(),
        "mean_node_count": sum(len(data["features"][int(i)].x_cont) for i in rows) / len(rows),
        "sm_positive_fraction": float(positive.float().mean()),
        "sm_positive_median": float(target[positive, 3].median()) if positive.any() else 50.}
    if any(metadata["initialization"]["target_medians"][i] <= 0 for i in (0, 1, 4, 5)):
        raise ValueError("TRAIN timing/memory scales must be positive")
    return metadata


def create_student(metadata, design, initial=None, normalization_hash=None):
    student = student_from_teacher(metadata, 192, 4)
    if design["initialization"] == "source_student":
        if not initial:
            raise ValueError("Warmstart design requires initial_student")
        saved = torch.load(initial, weights_only=False, map_location="cpu")
        if saved["config"]["hardware"] != "A100":
            raise ValueError("Transfer warmstart requires the A100 pretrained student")
        source_hash = saved["config"]["provenance"]["normalization_sha256"]
        if source_hash != normalization_hash:
            raise ValueError("Source student uses another feature normalization")
        student.load_state_dict(saved["state"], strict=True)
    return student


def restore_student(path):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    student = student_from_teacher(saved["metadata"], 192, 4)
    student.load_state_dict(saved["state"], strict=True)
    return student.eval(), saved


def packed(base, data, rows, device):
    return tuple(t.to(device) for t in pack_batch(base.to_batch(data["features"], rows, "cpu"),
                 torch.as_tensor(data["raw_steps_per_epoch"])[rows]))


@torch.no_grad()
def predict(model, base, data, rows, device="cpu", batch_size=32):
    model.eval()
    values = []
    for selected in torch.as_tensor(rows).split(batch_size):
        with base.deterministic_evaluation():
            values.append(model(*packed(base, data, selected, device))["prediction"].double().cpu())
    return torch.cat(values).numpy()


def report(prediction, key):
    return legacy.metrics(prediction, key["targets"], raw_targets=key["raw_targets_v4"], families=key["families"])
