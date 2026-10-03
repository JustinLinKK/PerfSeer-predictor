"""Acquire the nine original A10 datasets and reproduce their tiny subset keys.

Ported from scripts/manage_dataset_sources.py in the original real-a10 workflow.
Archives stay compressed: the profiler consumes keys and byte fingerprints, not
decoded examples. OGB node count is read from its raw archive without loading a
multi-gigabyte PyG graph or initializing CUDA.
"""

from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import heapq
import io
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "record/perfseer-v32/transfer-source-data"
VERSION = "perfseer_v32_a10_source_data_v1"
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".dcm", ".ogg", ".wav", ".flac"}
# Source ids/slugs and full sample counts are recovered from the original registry
# and all 40,020 native A10 labels. Archive names are part of the input RNG seed.
SOURCES = {
    "cassava_leaf_disease": dict(kind="competition", slug="cassava-leaf-disease-classification",
        samples=21397, archive_bytes=6185662420,
        sha256="25e1106760b771db44a52ea68c5ec95c22067119e9b882363791fc3a6d3c56ae"),
    "pothole_image_segmentation": dict(kind="dataset", slug="farzadnekouei/pothole-image-segmentation-dataset",
        samples=780, archive_bytes=62132662,
        sha256="5f637c1bba3787629b836ce5f49d177104de2cfa96a67dc869cab7c1f140faf1"),
    "taco_yolo_object_detection": dict(kind="dataset", slug="vencerlanz09/taco-dataset-yolo-format",
        samples=6004, archive_bytes=243202317,
        sha256="c977ef94ab7843124260887a42788da48b46c5d0424c43930827675cb8ad7f68"),
    "jigsaw_toxic_comment": dict(kind="competition", slug="jigsaw-toxic-comment-classification-challenge",
        samples=159571, archive_bytes=55201987, csv="train.csv.zip::train.csv",
        sha256="85ca047827488005dc0aa7c8bb9204372d937118e7ef79fc828abae3f30599fd"),
    "cnn_dailymail_summarization": dict(kind="dataset", slug="gowrishankarp/newspaper-text-summarization-cnn-dailymail",
        samples=287113, archive_bytes=527738644, csv="cnn_dailymail/train.csv",
        sha256="8f9e0cad39333d1c9f8902be6d846000ef1ccd5e994ad3c42f53336270ab8611"),
    "animal_audio_classification": dict(kind="dataset", slug="warcoder/cats-vs-dogs-vs-birds-audio-classification",
        samples=610, archive_bytes=13878900,
        sha256="62e52d6e3936635ec68da297d695441d897e1673f5bcdf70b33b90b7219a55af"),
    "store_sales_time_series": dict(kind="competition", slug="store-sales-time-series-forecasting",
        samples=3000888, archive_bytes=22416355, csv="train.csv",
        sha256="12e9c1dc4833cc804b3ff1515bd7b688f45fd8216fb2b340525036b006d625be"),
    "credit_card_default": dict(kind="dataset", slug="uciml/default-of-credit-card-clients-dataset",
        samples=30000, archive_bytes=1025318, csv="UCI_Credit_Card.csv",
        sha256="841af51be5c1e60383a4d8e8ef03b90f619fda362af871455d1c1d326b8e7a2d"),
    "ogbn_products": dict(kind="ogb", slug="ogbn-products", samples=2449029,
        url="https://snap.stanford.edu/ogb/data/nodeproppred/products.zip"),
}


def archive_name(source):
    return "products.zip" if source["kind"] == "ogb" else source["slug"].split("/")[-1] + ".zip"


@contextmanager
def csv_stream(archive, member):
    """Stream the native CSV, retaining the nested Jigsaw ZIP convention."""
    with zipfile.ZipFile(archive) as outer:
        if "::" in member:
            nested, inner = member.split("::")
            with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as bundle:
                with bundle.open(inner) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8", errors="ignore", newline="") as text:
                        yield text
        else:
            with outer.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", errors="ignore", newline="") as text:
                    yield text


def sample_keys(dataset_id, source, archive):
    prefix = archive_name(source)
    if source["kind"] == "ogb":
        with zipfile.ZipFile(archive) as bundle:
            count = int(gzip.decompress(bundle.read("products/raw/num-node-list.csv.gz")))
        if count != source["samples"]:
            raise ValueError("OGBN-Products raw node count differs from A10")
        yield from (f"node:{index:012d}" for index in range(count))
    elif source.get("csv"):
        with csv_stream(archive, source["csv"]) as stream:
            reader = csv.reader(stream)
            if not next(reader, None):
                raise ValueError("source CSV header is missing")
            for index, _ in enumerate(reader):
                yield f"{prefix}::{source['csv']}:row:{index:012d}"
    else:
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in MEDIA_SUFFIXES:
                    continue
                if dataset_id == "cassava_leaf_disease" and not info.filename.startswith("train_images/"):
                    continue
                yield f"{prefix}::{info.filename}"


def tiny_mask(dataset_id, source, archive):
    count = 0

    def counted():
        nonlocal count
        for key in sample_keys(dataset_id, source, archive):
            count += 1
            yield key

    keys = heapq.nsmallest(1024, counted(), key=lambda key: hashlib.sha256(f"{dataset_id}:{key}".encode()).hexdigest())
    if count != source["samples"] or len(set(keys)) != min(count, 1024):
        raise ValueError(f"original sample count/keys differ for {dataset_id}: {count} != {source['samples']}")
    return dict(subset_id="tiny", num_samples=len(keys), sample_keys=keys)


def check_archive(path, source):
    if not path.is_file():
        raise FileNotFoundError(f"missing source archive: {path}; run prepare-data")
    size = path.stat().st_size
    if source.get("archive_bytes") is not None and size != source["archive_bytes"]:
        raise ValueError(f"source archive size differs from A10: {path.name}")
    digest = file_sha256(path)
    if source.get("sha256") is not None and digest != source["sha256"]:
        raise ValueError(f"source archive checksum differs: {path.name}")
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names) or any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            raise ValueError(f"unsafe or duplicate source archive entries: {path.name}")
        if source.get("sha256") is None and bundle.testzip() is not None:
            raise ValueError(f"source archive CRC differs: {path.name}")
    return dict(bytes=size, sha256=digest)


def acquire(root, dataset_id, source, cache_root=None):
    """Promote only a fully verified download; interrupted staging is resumable."""
    target = root / "raw" / dataset_id / archive_name(source)
    if target.exists():
        return target, "existing", check_archive(target, source)
    staging = root / ".downloads" / dataset_id
    staging.mkdir(parents=True, exist_ok=True)
    candidate = staging / archive_name(source)
    method = "download"
    cached = Path(cache_root) / "raw" / dataset_id / target.name if cache_root else None
    if cached is not None and cached.is_file() and not candidate.exists():
        check_archive(cached, source)
        partial = staging / (target.name + ".copying")
        shutil.copyfile(cached, partial)
        partial.replace(candidate)
        method = "verified_cache_copy"
    if not candidate.exists():
        print(f"Downloading {dataset_id} from {source['slug']}", flush=True)
        if source["kind"] == "ogb":
            partial = staging / (target.name + ".part")
            with urllib.request.urlopen(source["url"], timeout=60) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
                stream.flush()
            check_archive(partial, source)
            partial.replace(candidate)
        else:
            command = [sys.executable, "-m", "kaggle"]
            command += (["competitions", "download", "-c", source["slug"]] if source["kind"] == "competition"
                        else ["datasets", "download", source["slug"]])
            with (staging / "download.log").open("a") as log:
                completed = subprocess.run([*command, "-p", str(staging), "-q"], stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"download failed for {dataset_id}; see {staging / 'download.log'}")
    try:
        evidence = check_archive(candidate, source)
    except (ValueError, zipfile.BadZipFile):
        # Preserve corrupt/partial evidence and allow a subsequent prepare-data retry.
        candidate.replace(staging / (candidate.name + ".invalid"))
        raise
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate.replace(target)
    return target, method, evidence


def verify_source_contract(aliases):
    """Check every source label, including counts absent from canonical anchors."""
    seen = set()
    for alias in aliases:
        native = alias["native_label"]["dataset"]
        name = native["dataset_id"]
        if name not in SOURCES:
            raise ValueError(f"unexpected native A10 dataset: {name}")
        source = SOURCES[name]
        if native["sample_count"] != source["samples"] or native["num_samples"] != min(1024, source["samples"]) or native["subset_id"] != "tiny":
            raise ValueError(f"native A10 dataset counts differ: {name}")
        if source.get("archive_bytes") is not None and native["sample_bytes_mean"] != source["archive_bytes"]:
            raise ValueError(f"native A10 archive size differs: {name}")
        seen.add(name)
    if seen != set(SOURCES):
        raise ValueError("source aliases do not cover all nine datasets")


def verify_data(root, anchors, *, rebuild_masks=False):
    """Hash every raw archive and reconcile persisted masks and input materials."""
    from .transfer_labeling import source_materials
    root = Path(root).resolve()
    manifest = read_json(root / "source-data-manifest.json")
    if manifest.get("version") != VERSION or manifest.get("catalog_fingerprint") != fingerprint(SOURCES):
        raise ValueError("source-data catalog/version differs; run prepare-data")
    if manifest.get("fingerprint") != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("source-data manifest fingerprint differs")
    if set(manifest["datasets"]) != set(SOURCES):
        raise ValueError("source-data manifest is incomplete")
    required = {(a["identity"]["dataset_id"], a["identity"]["num_samples"]) for a in anchors}
    if required != {(name, min(1024, s["samples"])) for name, s in SOURCES.items()}:
        raise ValueError("campaign/source-data requirements differ")
    for name, source in SOURCES.items():
        entry = manifest["datasets"][name]
        path = root / "raw" / name / archive_name(source)
        if check_archive(path, source) != entry["archive"]:
            raise ValueError(f"source archive manifest differs: {name}")
        mask_path = root / "prepared" / name / "subset_masks/tiny.json"
        mask = read_json(mask_path)
        if file_sha256(mask_path) != entry["mask_sha256"] or mask["num_samples"] != min(1024, source["samples"]):
            raise ValueError(f"source subset mask differs: {name}")
        if rebuild_masks and mask != tiny_mask(name, source, path):
            raise ValueError(f"source subset selection differs: {name}")
    materials = source_materials(root, anchors)
    if fingerprint(materials) != manifest["material_fingerprint"]:
        raise ValueError("source input replay fingerprint differs")
    return dict(status="verified", source_data_root=str(root), datasets=len(SOURCES),
                subset_samples=sum(min(1024, s["samples"]) for s in SOURCES.values()),
                source_data_fingerprint=manifest["fingerprint"], material_fingerprint=fingerprint(materials),
                rebuilt_masks=rebuild_masks, gpu_execution_performed=False)


def prepare_data(root, anchors, aliases, *, cache_root=None):
    from .transfer_labeling import source_materials
    root = Path(root).resolve()
    verify_source_contract(aliases)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "source-data-manifest.json").exists():
            return verify_data(root, anchors, rebuild_masks=True)
        entries = {}
        for name, source in SOURCES.items():
            print(f"Preparing source data: {name}", flush=True)
            archive, method, evidence = acquire(root, name, source, cache_root)
            mask = tiny_mask(name, source, archive)
            mask_path = root / "prepared" / name / "subset_masks/tiny.json"
            if mask_path.exists() and read_json(mask_path) != mask:
                raise ValueError(f"existing subset mask differs: {mask_path}")
            atomic_write(mask_path, mask)
            entries[name] = dict(source=source, archive=evidence, acquisition=method,
                                 mask_sha256=file_sha256(mask_path), num_samples=mask["num_samples"])
            atomic_write(root / "prepared" / name / "source-receipt.json", entries[name])
        materials = source_materials(root, anchors)
        manifest = dict(version=VERSION, catalog_fingerprint=fingerprint(SOURCES), datasets=entries,
                        subset_policy="sha256(dataset_id + ':' + key); first min(1024, full_count)",
                        input_adapter="local_subset_key_tensor", material_fingerprint=fingerprint(materials))
        manifest["fingerprint"] = fingerprint(manifest)
        atomic_write(root / "source-data-manifest.json", manifest)
        report = verify_data(root, anchors, rebuild_masks=True)
        atomic_write(root / "source-data-verification.json", report)
        return report
