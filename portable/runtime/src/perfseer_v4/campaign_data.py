"""Native-label campaign preparation, strict graph capture, and portable audits."""

import ast
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import gzip
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import tempfile
import traceback
from types import SimpleNamespace

import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .campaign_objective import target_tensors
from .capture import capture, graph_from_design
from .dataset import checked_path
from .features import build_features, normalize_features
from .version import TARGET_NAMES

VERSION = "perfseer_v4_masked_campaign_dataset_v1"
EXPECTED = {"A100": 47296, "A10": 43698, "RTX5090": 4128}
TARGET_HARDWARE = "nvidia_geforce_rtx_5090_32gb_local"


def family_splits(configs):
    families = sorted({r["model_id"].split("_")[1] for r in configs},
                      key=lambda x: fingerprint_dummy(x))
    if len(families) != 32:
        raise ValueError("expected the delivered 32 source families")
    return {f: "train" if i < 22 else "validation" if i < 27 else "test" for i, f in enumerate(families)}


def fingerprint_dummy(value):
    # Preserve the existing ABA preparation's split assignment byte for byte.
    import hashlib
    return hashlib.sha256(("42:" + value).encode()).hexdigest()


def native_rows(repo, hardware, configs, splits):
    path = repo / "labels" / hardware / "labels/labels.jsonl"
    raw = [json.loads(line) for line in path.open()]
    counts = Counter(r["profile_point_id"] for r in raw)
    quarantine = [r for r in raw if counts[r["profile_point_id"]] != 1]
    result = []
    for number, row in enumerate(raw, 1):
        if counts[row["profile_point_id"]] != 1:
            continue
        config = configs[row["profile_point_id"]]
        if row["status"] != "ok" or row["hardware_id"] != "nvidia_" + hardware.lower():
            raise ValueError("source status or hardware differs")
        if any(config[k] != row[k] for k in ("model_id", "batch_size", "precision_config", "optimizer")):
            raise ValueError("native label and workload configuration differ")
        if config["source_sha256"] != row["variant_source_sha256"] or config["model_file"] != row["variant_source_file"]:
            raise ValueError("native workload source identity differs")
        values = {name: row["targets"].get(name) for name in TARGET_NAMES}
        provenance = {name: "targets." + name if values[name] is not None else "unmeasured" for name in TARGET_NAMES}
        training = row["training"]
        if values[TARGET_NAMES[2]] is None and training.get("epoch_time_source") == "measured_epochs" and training.get("measured_epochs", 0) > 0:
            values[TARGET_NAMES[2]] = training.get("measured_epoch_wall_mean_ms")
            provenance[TARGET_NAMES[2]] = "training.measured_epoch_wall_mean_ms"
        family = row["model_id"].split("_")[1]
        item = {"sample_id": row["profile_point_id"], "group_id": "source_family_" + family,
                "split": splits[family], "hardware_id": row["hardware_id"], "target_names": list(TARGET_NAMES),
                "targets": values, "target_mask": {k: v is not None for k, v in values.items()},
                "target_provenance": provenance, "native_targets": row["targets"], "measured_training": training,
                "source_line": number, "source_sha256": config["source_sha256"], "source_file": config["model_file"],
                "source_configuration": config}
        target_tensors([item])
        result.append(item)
    if len(result) != EXPECTED[hardware] or len(quarantine) != (10 if hardware == "A10" else 0):
        raise ValueError("delivered counts differ from the reviewed corpus")
    return result, quarantine, {"path": str(path.relative_to(repo)), "sha256": file_sha256(path)}


def _module(path):
    name = "_perfseer_campaign_workload_" + file_sha256(path)[:20]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def structural_source(path):
    """Ignore identifiers and optimizer constants only; keep executable structure."""
    tree = ast.parse(path.read_text())
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    ignorable = {"MODEL_ID", "PROFILE_POINT_ID", "OPTIMIZER_NAME"} - referenced
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
        ignorable for t in node.targets))]
    # Generated module docstrings contain IDs, not executable semantics.
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        tree.body.pop(0)
    return fingerprint(ast.dump(tree, include_attributes=False))


def make_inputs(specs, device):
    values = []
    for spec in specs:
        shape = tuple(spec["shape"])
        if spec.get("dtype") in {"int64", "long"} or spec.get("kind") in {"tokens", "token_ids"}:
            value = torch.zeros(shape, dtype=torch.long, device=device)
        elif spec.get("kind") == "adjacency":
            base = torch.eye(shape[-1], device=device)
            value = (torch.roll(base, 1, 1) + torch.roll(base, -1, 1)).expand(shape).clone()
        else:
            value = torch.randn(shape, device=device)
        values.append(value)
    return tuple(values)


@contextmanager
def workload_precision(precision):
    """Mirror native profiler precision controls, then restore caller settings."""
    enabled = precision in {"tf32", "fp32_tf32"}
    controls = [(torch.backends, "fp32_precision"), (torch.backends.cuda.matmul, "fp32_precision"),
                (torch.backends.cudnn, "fp32_precision"), (torch.backends.cudnn.conv, "fp32_precision"),
                (torch.backends.cudnn.rnn, "fp32_precision")]
    if not all(hasattr(obj, key) for obj, key in controls):
        controls = [(torch.backends.cuda.matmul, "allow_tf32"), (torch.backends.cudnn, "allow_tf32")]
    previous = [(obj, key, getattr(obj, key)) for obj, key in controls]
    benchmark = torch.backends.cudnn.benchmark
    try:
        for obj, key in controls:
            setattr(obj, key, ("tf32" if enabled else "ieee") if key == "fp32_precision" else enabled)
        torch.backends.cudnn.benchmark = True
        yield {"tf32": enabled, "cudnn_benchmark": True, "api": controls[0][1]}
    finally:
        for obj, key, value in previous:
            setattr(obj, key, value)
        torch.backends.cudnn.benchmark = benchmark


def checkpoint_segments(model):
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, torch.nn.Sequential) and len(blocks) >= 10 and all(hasattr(model, n) for n in ("stem", "pool", "head")):
        return min(4, len(blocks))
    network = getattr(model, "network", None)
    if (isinstance(network, torch.nn.Sequential) and len(network) >= 10 and hasattr(model, "head") and
            getattr(model, "family", None) in {"panns_cnn14", "m5_waveform_cnn", "temporal_convolutional_network"}):
        return min(4, len(network))
    return 0


class ProfileModel(torch.nn.Module):
    """Mirror the delivered profiler's non-reentrant activation checkpointing."""
    def __init__(self, model, segments):
        super().__init__()
        self.model, self.segments = model, segments

    def forward(self, *inputs):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils.checkpoint import checkpoint
        model = self.model
        if self.segments < 2 or len(inputs) != 1:
            return model(*inputs)
        vision = all(hasattr(model, key) for key in ("stem", "blocks", "pool", "head"))
        features = model.stem(inputs[0]) if vision else inputs[0]
        blocks = list((model.blocks if vision else model.network).children())
        size = math.ceil(len(blocks) / self.segments)
        chunks = [blocks[i:i + size] for i in range(0, len(blocks), size)]
        class BufferContext(TorchDispatchMode):
            def __init__(self, segment=None):
                super().__init__()
                self.segment = segment

            def __enter__(self):
                self.states = [] if self.segment is None else [
                    (layer, {key: value.clone() for key, value in layer.named_buffers(recurse=False)})
                    for layer in self.segment.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
                return super().__enter__()

            def __exit__(self, *args):
                result = super().__exit__(*args)
                with torch.no_grad():
                    for layer, state in self.states:
                        for key, value in state.items():
                            getattr(layer, key).copy_(value)
                return result

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))
        for chunk in chunks[:-1]:
            segment = torch.nn.Sequential(*chunk)
            features = checkpoint(segment, features, use_reentrant=False,
                                  context_fn=lambda segment=segment: (BufferContext(), BufferContext(segment)))
        features = torch.nn.Sequential(*chunks[-1])(features)
        return model.head(model.pool(features).flatten(1)) if vision else model.head(features.mean(tuple(range(2, features.ndim))))


def optimizer_variant(design, name):
    """Reuse verified forward/backward structure with a fresh optimizer summary."""
    if design["training"]["TRAINING_CONFIG"]["optimizer"]["name"] == name:
        return design
    from perfseer_v3.capture_training import _append_optimizer_summary
    from perfseer_v3.op_registry import OperationRegistry
    from perfseer_v31.liveness import apply_liveness
    from .version import INPUT_SCHEMA_VERSION
    graph = graph_from_design(design)
    optimizer_nodes = [node for node in graph.nodes if node.phase == "optimizer"]
    if len(optimizer_nodes) != 1 or optimizer_nodes[0] != graph.nodes[-1]:
        raise ValueError("expected one terminal analytical optimizer summary")
    last = optimizer_nodes[0]
    nodes = graph.nodes[:-1]
    edges = tuple(edge for edge in graph.tensor_edges if last.node_id not in (edge.producer_node_id, edge.consumer_node_id))
    training = {**graph.training_config, "optimizer": {**graph.training_config["optimizer"], "name": name}}
    optimizer = {**training["optimizer"], "gradient_accumulation_steps": training["gradient_accumulation_steps"]}
    base = replace(graph, precision="fp32", nodes=nodes, tensor_edges=edges, training_config=training,
                   global_features=replace(graph.global_features, operation_nodes=len(nodes), tensor_edges=len(edges),
                       total_flops=graph.global_features.total_flops - last.flops.value,
                       total_optimizer_state_bytes=0, critical_path_length=last.depth))
    changed = apply_liveness(replace(_append_optimizer_summary(base, OperationRegistry.load(), name, optimizer), precision=graph.precision))
    payload = changed.to_dict()
    converted = {"version": design["training"]["version"], "INPUT_SPECS": payload.pop("input_signature"),
                 "NODE_SPECS": payload.pop("nodes"), "EDGE_SPECS": payload.pop("tensor_edges"),
                 "TRAINING_CONFIG": payload.pop("training_config"), "graph": payload}
    result = {"version": INPUT_SCHEMA_VERSION, "training": converted}
    graph_from_design(result).validate()
    return result


def capture_source(repo, row, cache, device):
    source = checked_path(repo / "labels/A10", row["source_file"])
    if file_sha256(source) != row["source_sha256"]:
        raise ValueError("workload source hash differs")
    training = row["measured_training"]
    # Measurements are excluded from capture/cache keys and model inputs.
    settings = {k: training[k] for k in ("batch_size", "effective_batch_size", "grad_accumulation_steps", "optimizer", "precision", "steps_per_epoch")}
    identity = {"source": structural_source(source), "training": {k: v for k, v in settings.items() if k != "optimizer"},
                "initialization_seed": 42, "torch": str(torch.__version__),
                "capture_device": str(device), "implementation": file_sha256(Path(__file__)),
                "execution_replay_implementation": file_sha256(Path(sys.modules["perfseer_v31.execution_replay"].__file__)),
                "capture_implementation": file_sha256(Path(sys.modules["perfseer_v31.capture_training"].__file__))}
    key = fingerprint(identity)
    path, audit_path = cache / (key + ".json.gz"), cache / (key + ".audit.json")
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit["identity"] != identity or audit["sha256"] != file_sha256(path):
            raise ValueError("capture cache identity differs")
        design = optimizer_variant(read_json(path), settings["optimizer"])
        return design, {**audit, "requested_optimizer": settings["optimizer"], "reused_verified_workload": True}
    torch.manual_seed(42)
    module = _module(source)
    model = module.make_model().to(device).train()
    segments = checkpoint_segments(model)
    config = {"version": "perfseer_v31_training_config_v1", "precision": settings["precision"],
              "microbatch_size": settings["batch_size"], "gradient_accumulation_steps": settings["grad_accumulation_steps"],
              "optimizer": {"name": settings["optimizer"], "learning_rate": .001},
              "scheduler": {"name": "none"}, "backend": "cuda_eager", "loss": "mse_to_zero",
              "steps_per_epoch": settings["steps_per_epoch"], "checkpointing": segments > 0,
              "checkpoint_segments": segments}
    began = time.monotonic()
    with workload_precision(settings["precision"]) as precision_controls:
        inputs = make_inputs(module.INPUT_SPECS, device)
        numerical_failure = None
        try:
            design, capture_audit = capture(ProfileModel(model, segments), inputs, None, None,
                                            config, architecture_key=identity["source"])
        except (AssertionError, ValueError) as error:
            if not isinstance(error, AssertionError) and "nonfinite" not in str(error):
                raise
            numerical_failure = str(error)
            error.__traceback__ = None
        if numerical_failure is not None:
            import gc
            gc.collect()
            # A measured workload may overflow or use nondeterministic CUDA
            # reductions. Require an independent exact execution-DAG match;
            # never alter its code, precision, weights, inputs or native labels.
            design, capture_audit = capture(ProfileModel(model, segments), inputs, None, None,
                                            config, architecture_key=identity["source"], verify_dataflow=True)
            capture_audit["native_numerical_failure"] = numerical_failure
    audit = {"identity": identity, "capture": capture_audit, "seconds": time.monotonic() - began,
             "captured_device": torch.cuda.get_device_name() if str(device).startswith("cuda") else "cpu",
             "prediction_hardware_from_labels": True, "optimizer_summary_is_analytical": True,
             "precision_controls": precision_controls}
    atomic_write(path, design, compress=True)
    audit["sha256"] = file_sha256(path)
    atomic_write(audit_path, audit)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return design, {**audit, "requested_optimizer": settings["optimizer"], "reused_verified_workload": False}


def read_features(path):
    path = Path(path)
    source = io.BytesIO(gzip.decompress(path.read_bytes())) if path.suffix == ".gz" else path
    return torch.load(source, map_location="cpu", weights_only=False)


def write_features(path, features):
    """Lossless compression keeps large unrolled execution graphs practical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".features-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=1, mtime=0) as compressed:
                torch.save(features, compressed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_design(root, row, design, audit):
    from copy import deepcopy
    design = deepcopy(design)
    metadata = design["training"]["graph"]["metadata"]
    if row["hardware_id"] != TARGET_HARDWARE:
        for key in ("hardware_profile", "hardware_profile_sha256", "hardware_features", "hardware_environment"):
            metadata.pop(key, None)
        metadata["target_hardware_id"] = row["hardware_id"]
        metadata["hardware_id"] = row["hardware_id"]
    graph = graph_from_design(design)
    if graph.metadata["target_hardware_id"] != row["hardware_id"]:
        raise ValueError("capture hardware binding differs")
    key = fingerprint(design)
    path = root / "graphs" / (key + ".json.gz")
    feature_path = root / "features" / (key + ".pt.gz")
    if not path.exists():
        atomic_write(path, design, compress=True)
    if not feature_path.exists():
        features = build_features(design, include_resources=True)
        write_features(feature_path, features)
    else:
        features = read_features(feature_path)
    resources = {"resources": features.resources.tolist(), "hardware_cont": features.training.hardware_cont.tolist(),
                 "hardware_missing_mask": features.training.hardware_missing_mask.tolist()}
    return {**row, "input_path": str(path.relative_to(root)), "input_sha256": file_sha256(path),
            "feature_path": str(feature_path.relative_to(root)), "feature_sha256": file_sha256(feature_path),
            "resource_inputs": resources,
            "capture_audit": audit, "graph_nodes": len(graph.nodes), "graph_edges": len(graph.tensor_edges),
            "workload_sha256": fingerprint({k: v for k, v in design["training"].items() if k != "graph"})}


def _smoke_rows(rows, limit):
    if not limit:
        return rows
    result = []
    for split in ("train", "validation", "test"):
        chosen = [row for row in rows if row["split"] == split]
        # Small files first for bounded integration checks; labels never rank rows.
        chosen.sort(key=lambda r: (r.get("source_configuration", {}).get("bytes", 0), r["sample_id"]))
        result.extend(chosen[:limit])
    return result


def _feature_worker():
    torch.set_num_threads(1)


def _capture_worker():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.85)


class CaptureWorker:
    """Retire CUDA contexts periodically and retry fatal CUDA errors in isolation."""
    def __init__(self):
        self.pool = None
        self.tasks = 0

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.tasks = 0

    def __call__(self, repo, row, cache, device):
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        import multiprocessing
        if self.tasks >= 64:
            self.close()
        for attempt in range(4):
            if self.pool is None:
                # Retire the idle pool ourselves: max_tasks_per_child can hang
                # at worker replacement in older container Python versions.
                self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=_capture_worker)
            try:
                result = self.pool.submit(capture_source, repo, row, cache, device).result()
                self.tasks += 1
                return result
            except Exception as error:
                # A poisoned CUDA context cannot be recovered with empty_cache.
                # Numerical/contract failures still fail closed without retry.
                fatal = isinstance(error, BrokenProcessPool) or any(message in str(error).lower() for message in (
                    "illegal memory access", "device-side assert", "unspecified launch failure", "cuda error: misaligned address"))
                failure = {"sample_id": row["sample_id"], "attempt": attempt + 1, "error": repr(error),
                           "traceback": traceback.format_exc(), "retry_in_fresh_process": fatal and attempt < 3}
                atomic_write(Path(cache) / "failures" / (fingerprint(row["sample_id"]) + f"-{attempt + 1}.json"), failure)
                self.close()
                if not fatal or attempt == 3:
                    raise


def prepare(repo, output, *, device="cpu", smoke_rows=0, feature_workers=0):
    """Capture serially on GPU while bounded CPU workers prepare features."""
    if not 0 <= feature_workers <= 8:
        raise ValueError("feature workers must be between zero and eight")
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    import multiprocessing
    pool = ProcessPoolExecutor(max_workers=feature_workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_feature_worker) if feature_workers else nullcontext(None)
    capture_worker = CaptureWorker() if str(device).startswith("cuda") else None
    try:
        with pool as executor:
            return _prepare(repo, output, device=device, smoke_rows=smoke_rows, executor=executor,
                            pending_limit=max(1, feature_workers * 2), capture_worker=capture_worker)
    finally:
        if capture_worker is not None:
            capture_worker.close()


def _prepare(repo, output, *, device, smoke_rows, executor, pending_limit, capture_worker=None):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if repo == output or not (repo / "labels/A10/model_config.jsonl").is_file():
        raise ValueError("expected repository and separate prepared output")
    output.mkdir(parents=True, exist_ok=True)
    configs_path = repo / "labels/A10/model_config.jsonl"
    config_list = [json.loads(line) for line in configs_path.open()]
    configs = {r["profile_point_id"]: r for r in config_list}
    if len(configs) != 48000 or len(config_list) != len(configs):
        raise ValueError("expected 48,000 unique workload configurations")
    splits = family_splits(config_list)
    manifests = {}
    for hardware in ("A100", "A10", "RTX5090"):
        root = output / hardware
        manifest_path = root / "dataset_manifest.json"
        if manifest_path.exists():
            existing = verify_dataset(root)
            if existing["smoke_rows_per_split"] != smoke_rows:
                raise ValueError("cannot mix smoke and full preparation")
            manifests[hardware] = existing
            continue
        if hardware == "RTX5090":
            target_root = repo / "labels/RTX5090/ready_for_train"
            from .dataset import verify
            native_manifest = verify(target_root, workers=8 if executor else 1)
            rows = []
            for split, meta in native_manifest["split_files"].items():
                for row in read_json(target_root / meta["path"]):
                    rows.append({**row, "hardware_id": TARGET_HARDWARE,
                                 "target_mask": {name: True for name in TARGET_NAMES}})
            provenance = {"path": str(target_root.relative_to(repo)), "sha256": file_sha256(target_root / "dataset_manifest.json")}
            quarantine = []
        else:
            rows, quarantine, provenance = native_rows(repo, hardware, configs, splits)
        selected = _smoke_rows(rows, smoke_rows)
        atomic_write(root / "provenance.json.gz", {"source": provenance, "rows": rows, "quarantined_rows": quarantine}, compress=True)
        captured = []
        pending = []

        def collect_one():
            item_index, item_row, future = pending.pop(0)
            try:
                captured.append(future.result())
            except Exception as error:
                atomic_write(root / "preparation-failure.json", {"sample_id": item_row["sample_id"], "error": repr(error),
                             "completed": item_index, "expected": len(selected), "status": "FAILED_NOT_READY"})
                raise

        for index, row in enumerate(selected):
            try:
                if hardware == "RTX5090":
                    design = read_json(checked_path(target_root, row["input_path"]))
                    audit = {"kind": "existing_verified_target", "source_input_sha256": row["input_sha256"]}
                else:
                    design, audit = (capture_worker or capture_source)(repo, row, output / "capture-cache", device)
                if executor:
                    # A bounded queue prevents large unrolled graphs from
                    # accumulating in host memory while CPU features are built.
                    if len(design["training"]["NODE_SPECS"]) > 20000:
                        while pending:
                            collect_one()
                    pending.append((index, row, executor.submit(publish_design, root, row, design, audit)))
                    if len(pending) >= pending_limit:
                        collect_one()
                else:
                    captured.append(publish_design(root, row, design, audit))
            except Exception as error:
                atomic_write(root / "preparation-failure.json", {"sample_id": row["sample_id"], "error": repr(error), "completed": index,
                             "expected": len(selected), "status": "FAILED_NOT_READY"})
                raise
            if index % 25 == 0 or index + 1 == len(selected):
                print(json.dumps({"event": "prepare", "hardware": hardware, "completed": len(captured),
                                  "captured": index + 1, "total": len(selected)}), flush=True)
                atomic_write(root / "preparation-progress.json", {"completed": len(captured), "captured": index + 1, "total": len(selected)})
        while pending:
            collect_one()
        atomic_write(root / "preparation-progress.json", {"completed": len(captured), "captured": len(captured), "total": len(selected)})
        files = {}
        for split in ("train", "validation", "test"):
            path = root / split / "samples.json.gz"
            split_rows = [row for row in captured if row["split"] == split]
            atomic_write(path, split_rows, compress=True)
            files[split] = {"path": str(path.relative_to(root)), "sha256": file_sha256(path), "rows": len(split_rows)}
        manifest = {"version": VERSION, "hardware": hardware, "prediction_hardware": captured[0]["hardware_id"],
                    "target_names": list(TARGET_NAMES), "split_files": files, "total_rows": len(captured),
                    "full_accepted_rows": len(rows), "quarantined_rows": len(quarantine), "smoke_rows_per_split": smoke_rows,
                    "source_family_splits": splits, "provenance_sha256": file_sha256(root / "provenance.json.gz"),
                    "preparation_code_sha256": file_sha256(Path(__file__)), "torch": str(torch.__version__)}
        manifest["fingerprint"] = fingerprint(manifest)
        atomic_write(manifest_path, manifest)
        manifests[hardware] = verify_dataset(root)
        (root / "preparation-failure.json").unlink(missing_ok=True)
    overlap = overlap_audit(output)
    atomic_write(output / "preparation-receipt.json", {"status": "smoke_prepared" if smoke_rows else "prepared", "datasets":
                 {key: value["fingerprint"] for key, value in manifests.items()}, "overlap": overlap})
    return manifests


def load_rows(root, split):
    root = Path(root)
    manifest = read_json(root / "dataset_manifest.json")
    meta = manifest["split_files"][split]
    path = checked_path(root, meta["path"])
    if file_sha256(path) != meta["sha256"]:
        raise ValueError("split hash differs")
    rows = read_json(path)
    if len(rows) != meta["rows"]:
        raise ValueError("split count differs")
    return rows


def verify_dataset(root):
    root = Path(root)
    manifest = read_json(root / "dataset_manifest.json")
    if manifest["version"] != VERSION or manifest["fingerprint"] != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("campaign dataset contract differs")
    if file_sha256(root / "provenance.json.gz") != manifest["provenance_sha256"]:
        raise ValueError("native provenance hash differs")
    originals = {row["sample_id"]: row for row in read_json(root / "provenance.json.gz")["rows"]}
    seen, groups, hashes, checked = set(), {}, {}, set()
    counts = Counter()
    for split in ("train", "validation", "test"):
        rows = load_rows(root, split)
        if not rows:
            raise ValueError("empty campaign split")
        target_tensors(rows)
        for row in rows:
            if row["sample_id"] in seen or row["split"] != split or row["hardware_id"] != manifest["prediction_hardware"]:
                raise ValueError("duplicate row, split, or hardware mismatch")
            seen.add(row["sample_id"])
            for mapping, key in ((groups, row["group_id"]), (hashes, row["input_sha256"])):
                if mapping.setdefault(key, split) != split:
                    raise ValueError("group or graph crosses splits")
            original = originals[row["sample_id"]]
            if any(row[key] != original[key] for key in ("targets", "target_mask", "native_targets", "split", "group_id")):
                raise ValueError("native labels or split changed")
            for name in TARGET_NAMES:
                counts[name] += row["target_mask"][name]
            for path_key, hash_key in (("input_path", "input_sha256"), ("feature_path", "feature_sha256")):
                path = checked_path(root, row[path_key])
                if str(path) not in checked:
                    if file_sha256(path) != row[hash_key]:
                        raise ValueError("graph or feature cache hash differs")
                    checked.add(str(path))
            graph = graph_from_design(read_json(root / row["input_path"]))
            if graph.metadata["target_hardware_id"] != row["hardware_id"]:
                raise ValueError("graph hardware differs")
            if manifest["hardware"] != "RTX5090":
                measured = row["measured_training"]
                if (graph.precision != measured["precision"] or
                        graph.training_config["microbatch_size"] != measured["batch_size"] or
                        graph.training_config["gradient_accumulation_steps"] != measured["grad_accumulation_steps"] or
                        graph.optimizer_config["name"] != measured["optimizer"]):
                    raise ValueError("prepared execution differs from the measured protocol")
                audit = row["capture_audit"]["capture"]["training"]
                if audit.get("numerical_mismatches"):
                    proof = audit.get("execution_dataflow") or {}
                    if (proof.get("version") != "perfseer_execution_dataflow_v1" or len(proof.get("sha256", "")) != 64 or
                            proof.get("operations_checked", 0) <= 0 or
                            proof.get("outputs_checked") != audit["gradient_parameters_checked"] + audit["forward_outputs_checked"] + 1):
                        raise ValueError("numerically unstable profiling capture lacks execution proof")
        if not manifest["smoke_rows_per_split"] and manifest["hardware"] == "RTX5090" and len(rows) != {"train": 2988, "validation": 576, "test": 564}[split]:
            raise ValueError("target frozen split changed")
    if len(seen) != manifest["total_rows"] or (not manifest["smoke_rows_per_split"] and len(seen) != EXPECTED[manifest["hardware"]]):
        raise ValueError("campaign case coverage differs")
    return manifest


def overlap_audit(root):
    datasets = {gpu: [row for split in ("train", "validation", "test") for row in load_rows(Path(root) / gpu, split)]
                for gpu in EXPECTED}
    reports = {}
    for source in ("A100", "A10"):
        identities = {r["workload_sha256"]: r["split"] for r in datasets[source]}
        overlaps = [r["sample_id"] for r in datasets["RTX5090"] if r["workload_sha256"] in identities]
        reports[source] = {"exact_structural_overlaps": overlaps, "protocol_matched_cohort": False,
                           "claim": "combined hardware/workload/protocol transfer; unequal IDs do not prove independence"}
    return reports


class CampaignSamples:
    def __init__(self, root, rows, normalization=None, *, resource_only=False):
        self.root, self.rows, self.normalization = Path(root), rows, normalization
        self.resource_only = resource_only
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        if self.resource_only and "resource_inputs" in row:
            compact = row["resource_inputs"]
            features = SimpleNamespace(resources=torch.tensor(compact["resources"], dtype=torch.float32), training=SimpleNamespace(
                hardware_cont=torch.tensor(compact["hardware_cont"], dtype=torch.float32),
                hardware_missing_mask=torch.tensor(compact["hardware_missing_mask"], dtype=torch.float32)))
            targets, mask = target_tensors([row])
            return {"features": features, "target": targets[0], "mask": mask[0], "sample_id": row["sample_id"],
                    "group_id": row["group_id"], "split": row["split"], "row": row}
        key = row["feature_path"]
        if key not in self.cache:
            path = checked_path(self.root, key)
            if file_sha256(path) != row["feature_sha256"]:
                raise ValueError("feature cache checksum differs")
            features = read_features(path)
            features.validate()
            self.cache[key] = normalize_features(features, self.normalization) if self.normalization else features
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        targets, mask = target_tensors([row])
        return {"features": self.cache[key], "target": targets[0], "mask": mask[0], "sample_id": row["sample_id"],
                "group_id": row["group_id"], "split": row["split"], "row": row}


class LogicalBatch:
    """Load only the active microbatch, even for enormous execution graphs."""
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
        self.rows = [dataset.rows[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.dataset[i] for i in self.indices[index]]
        return self.dataset[self.indices[index]]
