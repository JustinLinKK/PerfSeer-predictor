import copy
import gzip
import hashlib
import io
import json
import zipfile

import pytest
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from perfseer_v32 import transfer_labeling as labeling
from perfseer_v32 import transfer_sources as sources


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source preparation must not initialize CUDA")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def write_zip(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def test_original_dataset_catalog():
    assert {name: min(1024, row["samples"]) for name, row in sources.SOURCES.items()} == {
        "cassava_leaf_disease": 1024, "pothole_image_segmentation": 780,
        "taco_yolo_object_detection": 1024, "jigsaw_toxic_comment": 1024,
        "cnn_dailymail_summarization": 1024, "animal_audio_classification": 610,
        "store_sales_time_series": 1024, "credit_card_default": 1024, "ogbn_products": 1024,
    }
    assert labeling.DEFAULT_SOURCE_ROOT == sources.DEFAULT_SOURCE_ROOT


def test_media_selection_matches_full_sort_and_preserves_archive_keys(tmp_path):
    source = dict(kind="dataset", slug="owner/images", samples=1100)
    path = tmp_path / "images.zip"
    members = {f"train/{i}.jpg": b"image" for i in range(1100)}
    write_zip(path, {**members, "labels.csv": b"header\n"})
    result = sources.tiny_mask("fixture", source, path)
    expected = sorted([f"images.zip::{name}" for name in members],
                      key=lambda key: hashlib.sha256(("fixture:" + key).encode()).hexdigest())[:1024]
    assert result == dict(subset_id="tiny", num_samples=1024, sample_keys=expected)
    with pytest.raises(ValueError, match="sample count"):
        sources.tiny_mask("fixture", {**source, "samples": 1099}, path)
    source = dict(kind="competition", slug="cassava", samples=1100)
    write_zip(path, {**{f"train_images/{i}.jpg": b"image" for i in range(1100)}, "test_images/x.jpg": b"test"})
    assert all("train_images/" in key for key in sources.tiny_mask("cassava_leaf_disease", source, path)["sample_keys"])


def test_nested_csv_keys_count_records_and_replay_exact_bytes(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as bundle:
        bundle.writestr("train.csv", 'id,text\n1,"two\nlines"\n2,last\n')
    path = tmp_path / "jigsaw.zip"
    write_zip(path, {"train.csv.zip": inner.getvalue()})
    source = dict(kind="competition", slug="jigsaw", samples=2, csv="train.csv.zip::train.csv")
    keys = list(sources.sample_keys("fixture", source, path))
    assert keys == [f"jigsaw.zip::train.csv.zip::train.csv:row:{i:012d}" for i in range(2)]
    assert labeling.sample_bytes(tmp_path, keys[0]) == b"1\x1ftwo\nlines"
    assert labeling.sample_bytes(tmp_path, keys[1]) == b"2\x1flast"


def test_graph_keys_come_from_raw_node_count(tmp_path):
    path = tmp_path / "products.zip"
    write_zip(path, {"products/raw/num-node-list.csv.gz": gzip.compress(b"12\n")})
    source = dict(kind="ogb", samples=12)
    assert list(sources.sample_keys("ogbn_products", source, path)) == [f"node:{i:012d}" for i in range(12)]
    with pytest.raises(ValueError, match="raw node count"):
        list(sources.sample_keys("ogbn_products", {**source, "samples": 13}, path))


@pytest.fixture
def prepared_data(tmp_path, monkeypatch):
    source = dict(kind="dataset", slug="owner/fixture", samples=8, csv="train.csv")
    cache, root = tmp_path / "cache", tmp_path / "data"
    archive = cache / "raw/fixture/fixture.zip"
    write_zip(archive, {"train.csv": "id,value\n" + "".join(f"{i},value-{i}\n" for i in range(8))})
    source.update(sha256=file_sha256(archive), archive_bytes=archive.stat().st_size)
    monkeypatch.setattr(sources, "SOURCES", {"fixture": source})
    anchors = [dict(identity=dict(dataset_id="fixture", subset_id="tiny", num_samples=8))]
    aliases = [dict(native_label=dict(dataset=dict(dataset_id="fixture", subset_id="tiny", num_samples=8,
                         sample_count=8, sample_bytes_mean=source["archive_bytes"])))]
    return root, cache, anchors, aliases, source


def test_prepare_verify_and_resume_without_network_or_cuda(prepared_data, monkeypatch):
    root, cache, anchors, aliases, source = prepared_data
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected download"))
    report = sources.prepare_data(root, anchors, aliases, cache_root=cache)
    assert report["status"] == "verified" and report["gpu_execution_performed"] is False
    assert sources.prepare_data(root, anchors, aliases) == report
    assert report == sources.verify_data(root, anchors, rebuild_masks=True)
    assert (cache / "raw/fixture/fixture.zip").is_file()
    assert read_json(root / "source-data-manifest.json")["datasets"]["fixture"]["acquisition"] == "verified_cache_copy"
    with pytest.raises(ValueError, match="counts differ"):
        sources.verify_source_contract([dict(native_label=dict(dataset=dict(aliases[0]["native_label"]["dataset"], sample_count=9)))])


def test_tampered_masks_and_archives_rejected(prepared_data):
    root, cache, anchors, aliases, source = prepared_data
    sources.prepare_data(root, anchors, aliases, cache_root=cache)
    mask_path = root / "prepared/fixture/subset_masks/tiny.json"
    original = read_json(mask_path)
    changed = copy.deepcopy(original)
    changed["sample_keys"].reverse()
    atomic_write(mask_path, changed)
    with pytest.raises(ValueError, match="subset mask differs"):
        sources.verify_data(root, anchors)
    # Recomputed local hashes cannot make a different selection pass a full audit.
    manifest_path = root / "source-data-manifest.json"
    manifest = read_json(manifest_path)
    manifest["datasets"]["fixture"]["mask_sha256"] = file_sha256(mask_path)
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    atomic_write(manifest_path, manifest)
    with pytest.raises(ValueError, match="subset selection differs"):
        sources.verify_data(root, anchors, rebuild_masks=True)
    atomic_write(mask_path, original)
    archive = root / "raw/fixture/fixture.zip"
    archive.write_bytes(archive.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksum differs"):
        sources.verify_data(root, anchors)


def test_interrupted_download_is_not_promoted_and_retry_is_possible(tmp_path, monkeypatch):
    source = dict(kind="dataset", slug="owner/fixture", samples=1)
    candidate = tmp_path / ".downloads/fixture/fixture.zip"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"incomplete ZIP")
    with pytest.raises(zipfile.BadZipFile):
        sources.acquire(tmp_path, "fixture", source)
    assert not (tmp_path / "raw/fixture/fixture.zip").exists()
    assert candidate.with_name("fixture.zip.invalid").exists()

    def download(command, **kwargs):
        assert command[1:5] == ["-m", "kaggle", "datasets", "download"]
        write_zip(candidate, {"one.jpg": b"actual input"})
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(sources.subprocess, "run", download)
    path, method, evidence = sources.acquire(tmp_path, "fixture", source)
    assert method == "download" and evidence["sha256"] == file_sha256(path)


def test_missing_data_rejected_before_gpu_preflight(tmp_path, monkeypatch):
    from perfseer_v32 import transfer_verification
    monkeypatch.setattr(transfer_verification, "verify", lambda *a, **kw: None)
    monkeypatch.setattr(labeling, "Nvml", lambda: pytest.fail("GPU preflight reached without source data"))
    for name, value in [("manifest.json", {}), ("anchors.json.gz", []),
                        ("configurations.json.gz", []), ("experiments.json.gz", {})]:
        atomic_write(tmp_path / name, value, compress=name.endswith(".gz"))
    with pytest.raises(FileNotFoundError):
        labeling.label(tmp_path, tmp_path / "missing")


def test_label_cli_uses_prepared_default_root(tmp_path, monkeypatch):
    monkeypatch.setattr(labeling, "DEFAULT_SOURCE_ROOT", tmp_path)
    atomic_write(tmp_path / "source-data-manifest.json", {})
    calls = []

    def fake_label(output, root, data, *, resume, campaign_profile):
        calls.append((root, resume, campaign_profile))
        return dict(status="mocked")

    monkeypatch.setattr(labeling, "label", fake_label)
    assert labeling.main(["label", "--resume"]) == 0
    assert calls == [(tmp_path, True, "full")]


def test_label_cli_reports_clean_pause(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(labeling, "DEFAULT_SOURCE_ROOT", tmp_path)
    atomic_write(tmp_path / "source-data-manifest.json", {})

    def paused(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(labeling, "label", paused)
    assert labeling.main(["label", "--campaign-profile", "24h", "--resume"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "paused" and result["campaign_profile"] == "24h"
    assert result["resume_from"] == "last verified configuration; the interrupted configuration will be retried"
