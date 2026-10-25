"""Static native workload inputs and six measured training targets."""

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .native_static import Reader, VERSION as MODULE_FEATURE_VERSION

VERSION = "perfseer_v4_native_six_dataset_v1"
FEATURE_VERSION = "perfseer_v4_native_static_hash_v1"
TARGET_NAMES = ("train_step_wall_ms", "train_step_gpu_ms", "train_epoch_ms",
                "train_avg_sm_util_percent", "train_avg_vram_mib", "train_peak_vram_mib")
HEAD_GROUPS = ((0, 1, 2), (3,), (4, 5))
FEATURE_DIM = 4096
SOURCES = ("A100", "A10")
SPLIT_POLICIES = ("held-out-families", "known-families-90-5-5")
LITERALS = ("MODEL_ID", "PROFILE_POINT_ID", "MODEL_CONFIG", "BATCH_SIZE",
            "PRECISION_CONFIG", "OPTIMIZER_NAME", "INPUT_SPECS")
TRAINING_FIELDS = ("batch_size", "effective_batch_size", "grad_accumulation_steps", "steps_per_epoch",
                   "precision", "optimizer")


def static_description(text, config, templates, *, static_modules=False):
    """Read literal assignments and AST counts; never import workload modules."""
    literals = {}
    template = text
    for name in LITERALS:
        pattern = rf"^{name}\s*=\s*(.+)$"
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"expected one single-line literal assignment: {name}")
        literals[name] = ast.literal_eval(matches[0])
        template = re.sub(pattern, "", template, flags=re.MULTILINE)
    for name, key in (("MODEL_ID", "model_id"), ("PROFILE_POINT_ID", "profile_point_id"),
                      ("BATCH_SIZE", "batch_size"), ("PRECISION_CONFIG", "precision_config"),
                      ("OPTIMIZER_NAME", "optimizer")):
        if literals[name] != config[key]:
            raise ValueError(f"source literal differs from configuration: {name}")
    template = re.sub(r'\A\s*"""[^\n]*"""\s*\n', "", template)
    template_sha = hashlib.sha256(template.encode()).hexdigest()
    if template_sha not in templates:
        tree = ast.parse(template)
        counts = Counter()
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.If, ast.BinOp, ast.Return, ast.ClassDef, ast.FunctionDef)):
                counts["syntax." + type(node).__name__] += 1
            if isinstance(node, ast.Call):
                function = node.func
                parts = []
                while isinstance(function, ast.Attribute):
                    parts.append(function.attr)
                    function = function.value
                if isinstance(function, ast.Name):
                    parts.append(function.id)
                counts["call." + ".".join(reversed(parts))] += 1
        templates[template_sha] = dict(counts)
    model = {key: value for key, value in literals["MODEL_CONFIG"].items()
             if key not in {"family_index", "sample_rank", "batch_sizes"}}
    result = {"model": model, "inputs": literals["INPUT_SPECS"], "source": templates[template_sha]}
    if static_modules:
        reader_key = ("reader", template_sha)
        if reader_key not in templates:
            templates[reader_key] = Reader(template)
        key = ("modules", template_sha, fingerprint({"model": model, "inputs": literals["INPUT_SPECS"]}))
        if key not in templates:
            templates[key] = templates[reader_key].describe(model, literals["INPUT_SPECS"])
        result["modules"] = templates[key]
    return result


def encode(description):
    """Signed hashing with separate numeric/categorical bins and no fitted vocabulary."""
    result = np.zeros(FEATURE_DIM, dtype=np.float32)
    def visit(key, value):
        if isinstance(value, dict):
            for name, item in sorted(value.items()):
                visit(key + "." + name, item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(key + "." + str(index), item)
        else:
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if numeric and not math.isfinite(value):
                raise ValueError("nonfinite workload descriptor")
            token = key if numeric else key + "=" + json.dumps(value, sort_keys=True)
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:8], "little") % (FEATURE_DIM // 2)
            if not numeric:
                index += FEATURE_DIM // 2
            weight = math.copysign(math.log1p(abs(value)), value) if numeric else 1.
            result[index] += weight if digest[8] & 1 else -weight
            if numeric:
                visit(key + ".present", True)
    visit("workload", description)
    return result


def native_rows(repo, source, configs, splits):
    path = repo / "labels" / source / "labels/labels.jsonl"
    raw = [json.loads(line) for line in path.open()]
    counts = Counter(row["profile_point_id"] for row in raw)
    rows, quarantined = [], []
    for row in raw:
        identifier = row["profile_point_id"]
        if counts[identifier] != 1:
            quarantined.append(identifier)
            continue
        config = configs[identifier]
        if row["status"] != "ok" or row["hardware_id"] != "nvidia_" + source.lower():
            raise ValueError("native label status or hardware differs")
        for key in ("model_id", "batch_size", "precision_config", "optimizer"):
            if row[key] != config[key]:
                raise ValueError("native label configuration differs")
        if (row["variant_source_sha256"] != config["source_sha256"] or
                row["variant_source_file"] != config["model_file"]):
            raise ValueError("native source identity differs")
        training = row["training"]
        if (training["batch_size"] != config["batch_size"] or training["optimizer"] != config["optimizer"]
                or training["precision"] != config["precision_config"]):
            raise ValueError("measured protocol differs from workload")
        values = [row["targets"].get(name) for name in TARGET_NAMES]
        if values[2] is None:
            if training.get("epoch_time_source") != "measured_epochs" or training.get("measured_epochs", 0) <= 0:
                raise ValueError("measured epoch target required")
            values[2] = training.get("measured_epoch_wall_mean_ms")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or (not 0 <= v <= 100 if i == 3 else v <= 0) for i, v in enumerate(values)):
            raise ValueError("six measured physical targets required; imputation is disabled")
        family = config["model_id"].split("_")[1]
        rows.append({"sample_id": identifier, "group_id": family, "split": splits[family],
                     "targets": values, "training": {key: training[key] for key in TRAINING_FIELDS}})
    return rows, quarantined, file_sha256(path)


def known_family_splits(groups):
    """Split equivalent configuration groups within each family, independent of labels."""
    result = {}
    for family in sorted({group.split(":", 1)[0] for group in groups}):
        ordered = sorted({group for group in groups if group.startswith(family + ":")},
                         key=lambda group: hashlib.sha256(("42:" + group).encode()).hexdigest())
        if len(ordered) < 3:
            raise ValueError("each family needs at least three distinct configuration groups")
        held = max(1, round(len(ordered) * .05))
        train = len(ordered) - 2 * held
        result.update({group: 0 if i < train else 1 if i < train + held else 2
                       for i, group in enumerate(ordered)})
    return result


def prepare(repo, output, *, static_modules=False, split_policy="held-out-families"):
    if split_policy not in SPLIT_POLICIES:
        raise ValueError("unsupported native split policy")
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = repo / "labels/A10/model_config.jsonl"
    config_rows = [json.loads(line) for line in config_path.open()]
    configs = {row["profile_point_id"]: row for row in config_rows}
    if len(configs) != len(config_rows):
        raise ValueError("duplicate workload configurations")
    families = sorted({row["model_id"].split("_")[1] for row in config_rows},
                      key=lambda value: hashlib.sha256(("42:" + value).encode()).hexdigest())
    if len(families) != 32:
        raise ValueError("expected the delivered 32 workload families")
    splits = {family: 0 if i < 22 else 1 if i < 27 else 2 for i, family in enumerate(families)}
    native = {source: native_rows(repo, source, configs, splits) for source in SOURCES}
    required = {row["sample_id"] for rows, _, _ in native.values() for row in rows}
    descriptions, templates, inventory = {}, {}, {}
    for index, identifier in enumerate(sorted(required)):
        config = configs[identifier]
        path = (repo / "labels/A10" / config["model_file"]).resolve()
        if not path.is_relative_to(repo / "labels/A10"):
            raise ValueError("workload path escapes dataset")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != config["source_sha256"]:
            raise ValueError("workload source checksum differs")
        descriptions[identifier] = static_description(content.decode(), config, templates, static_modules=static_modules)
        inventory[config["model_file"]] = digest
        if index % 2000 == 0:
            print(json.dumps({"event": "static_inputs", "completed": index, "total": len(required)}), flush=True)
    configuration_groups = {identifier: configs[identifier]["model_id"].split("_")[1] + ":" + fingerprint(description["model"])
                            for identifier, description in descriptions.items()}
    known = split_policy == "known-families-90-5-5"
    group_splits = known_family_splits(set(configuration_groups.values())) if known else {}
    receipts = {}
    for source, (rows, quarantine, label_sha) in native.items():
        root = output / source
        root.mkdir(parents=True, exist_ok=True)
        if known:
            for row in rows:
                row["split"] = group_splits[configuration_groups[row["sample_id"]]]
        features = np.stack([encode({**descriptions[row["sample_id"]], "training": row["training"],
                                    "hardware": "nvidia_" + source.lower()}) for row in rows])
        path = root / "dataset.npz"
        temporary = root / "dataset.tmp.npz"
        np.savez_compressed(temporary, features=features, targets=np.asarray([r["targets"] for r in rows], dtype=np.float32),
                            splits=np.asarray([r["split"] for r in rows], dtype=np.int8),
                            sample_ids=np.asarray([r["sample_id"] for r in rows]),
                            groups=np.asarray([r["group_id"] for r in rows]),
                            **({"configuration_groups": np.asarray([configuration_groups[r["sample_id"]] for r in rows])} if known else {}))
        temporary.replace(path)
        manifest = {"version": VERSION, "feature_version": MODULE_FEATURE_VERSION if static_modules else FEATURE_VERSION,
                    "feature_dim": FEATURE_DIM,
                    "source": source, "target_names": list(TARGET_NAMES), "total_rows": len(rows),
                    "splits": {name: sum(r["split"] == i for r in rows) for i, name in enumerate(("train", "validation", "test"))},
                    "family_splits": splits, "quarantined_rows": quarantine, "labels_sha256": label_sha,
                    "config_sha256": file_sha256(config_path), "source_inventory_sha256": fingerprint(inventory),
                    "preparation_code_sha256": file_sha256(Path(__file__)), "dataset_sha256": file_sha256(path),
                    "workloads_executed": False, "prediction_hardware": "nvidia_" + source.lower()}
        if static_modules:
            manifest["module_reader_sha256"] = file_sha256(Path(__file__).with_name("native_static.py"))
        if known:
            del manifest["family_splits"]
            manifest.update(split_policy=split_policy, split_seed=42, split_fractions=[.90, .05, .05],
                            configuration_group_splits=group_splits)
        manifest["fingerprint"] = fingerprint(manifest)
        atomic_write(root / "dataset_manifest.json", manifest)
        verify(root, repo=repo)
        receipts[source] = manifest
        del features
        print(json.dumps({"event": "native_dataset_ready", "source": source, "rows": len(rows)}), flush=True)
    atomic_write(output / "preparation.json", {"status": "verified", "datasets": receipts, "workloads_executed": False})
    return receipts


def verify(root, *, repo=None):
    root = Path(root)
    manifest = read_json(root / "dataset_manifest.json")
    policy = manifest.get("split_policy", "held-out-families")
    if policy not in SPLIT_POLICIES:
        raise ValueError("unsupported native split policy")
    if (manifest["version"] != VERSION or manifest["feature_version"] not in {FEATURE_VERSION, MODULE_FEATURE_VERSION}
            or manifest["target_names"] != list(TARGET_NAMES) or manifest["feature_dim"] != FEATURE_DIM
            or manifest["fingerprint"] != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
            or file_sha256(root / "dataset.npz") != manifest["dataset_sha256"]):
        raise ValueError("native dataset contract/checksum differs")
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        features, targets, splits, ids, groups = (data[k] for k in ("features", "targets", "splits", "sample_ids", "groups"))
        n = manifest["total_rows"]
        if (features.shape != (n, FEATURE_DIM) or targets.shape != (n, 6) or len(set(ids)) != n
                or ids.shape != (n,) or groups.shape != (n,) or splits.shape != (n,)
                or features.dtype != np.float32 or targets.dtype != np.float32
                or not np.isfinite(features).all() or not np.isfinite(targets).all()
                or (targets[:, [0, 1, 2, 4, 5]] <= 0).any() or (targets[:, 3] < 0).any() or (targets[:, 3] > 100).any()):
            raise ValueError("native tensor contract differs")
        for i, name in enumerate(("train", "validation", "test")):
            if int((splits == i).sum()) != manifest["splits"][name] or not (splits == i).any():
                raise ValueError("native split count differs")
        if policy == "known-families-90-5-5":
            assignments = manifest["configuration_group_splits"]
            configuration_groups = data["configuration_groups"]
            if (manifest["split_seed"] != 42 or manifest["split_fractions"] != [.90, .05, .05]
                    or assignments != known_family_splits(set(assignments))
                    or configuration_groups.shape != (n,)
                    or any(assignments.get(group) != int(split) or group.split(":", 1)[0] != family
                           for group, family, split in zip(configuration_groups, groups, splits))):
                raise ValueError("native configuration group split leakage")
            expected = {f"{i:02d}" for i in range(1, 33)}
            if any(set(groups[splits == i]) != expected for i in range(3)):
                raise ValueError("every native split must cover all 32 families")
        elif any(manifest["family_splits"][group] != int(split) for group, split in zip(groups, splits)):
            raise ValueError("source family split leakage")
        if repo is not None:
            repo = Path(repo)
            config_path = repo / "labels/A10/model_config.jsonl"
            if file_sha256(config_path) != manifest["config_sha256"]:
                raise ValueError("native configuration changed")
            configs = {r["profile_point_id"]: r for r in (json.loads(line) for line in config_path.open())}
            family_splits = manifest.get("family_splits", dict.fromkeys(groups, 0))
            rows, quarantine, sha = native_rows(repo, manifest["source"], configs, family_splits)
            if sha != manifest["labels_sha256"] or quarantine != manifest["quarantined_rows"]:
                raise ValueError("native labels or quarantine changed")
            if [r["sample_id"] for r in rows] != ids.tolist() or not np.array_equal(
                    targets, np.asarray([r["targets"] for r in rows], dtype=np.float32)):
                raise ValueError("prepared labels differ from measured native values")
            if policy == "known-families-90-5-5":
                for identifier, family, group in zip(ids, groups, configuration_groups):
                    config = configs[identifier]
                    path = (repo / "labels/A10" / config["model_file"]).resolve()
                    if not path.is_relative_to(repo / "labels/A10") or file_sha256(path) != config["source_sha256"]:
                        raise ValueError("native workload source changed")
                    values = re.findall(r"^MODEL_CONFIG\s*=\s*(.+)$", path.read_text(), flags=re.MULTILINE)
                    if len(values) != 1:
                        raise ValueError("expected one literal model configuration")
                    model = {k: v for k, v in ast.literal_eval(values[0]).items()
                             if k not in {"family_index", "sample_rank", "batch_sizes"}}
                    if group != family + ":" + fingerprint(model):
                        raise ValueError("native configuration group identity differs")
    return manifest
