#!/usr/bin/env python
"""Package verified local RTX 5090 labels, paired graphs, teacher, and Nautilus launcher."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import run_perfseer_v32_transfer_training
import torch
from perfseer_v31.io import file_sha256, read_json
from perfseer_v32.transfer_training import HARDWARE_ID, load_labels, verify
from perfseer_v32.training import restore_model


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "record/perfseer-v32/transfer-rtx5090")
    parser.add_argument("--dataset", type=Path, default=ROOT / "record/perfseer-v32/transfer-training-rtx5090/dataset")
    parser.add_argument("--teacher", type=Path, default=ROOT / "predictor_v3.2/teacher_model/teacher_v3.2_epoch26.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/perfseer-v32-rtx5090-a100-transfer-20260916")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".zip").exists():
        parser.error("use a fresh package output path")
    torch.set_num_threads(1)
    verification = verify(args.dataset, args.source)
    base = torch.load(args.teacher, map_location="cpu", weights_only=False, mmap=True)
    if base["role"] != "teacher":
        raise ValueError("base checkpoint must be a v3.2 teacher")
    restore_model(base)
    labels = load_labels(args.source)
    repeat_panel = set(read_json(args.source / "campaign-scope-24h.json.gz")["repeat_panel"])
    files = {"pyproject.toml": ROOT / "pyproject.toml", "weights/base-teacher.pt": args.teacher}
    for package in ("perfseer_v3", "perfseer_v3.1", "perfseer_v3.2"):
        folder = ROOT / "src" / package
        for path in folder.rglob("*"):
            relative = path.relative_to(folder)
            if any(part in {"dataset_with_label", "checkpoints", "__pycache__", "containers"} for part in relative.parts):
                continue
            if path.is_file() and path.suffix in {".py", ".json", ".yaml"}:
                files[str(path.relative_to(ROOT))] = path
    for path in args.dataset.rglob("*"):
        if path.is_file():
            files["dataset/" + str(path.relative_to(args.dataset))] = path
    for name in ("labels-24h.json.gz", "anchors.json.gz", "source_materials.json.gz", "manifest.json",
                 "execution.json", "execution-migration.json", "configurations.json.gz", "campaign-scope-24h.json.gz",
                 "labeling-report-24h.json", "source-data-verification.json"):
        if (args.source / name).is_file():
            files["native/" + name] = args.source / name
    for row in labels:
        files["native/" + row["graph_path"]] = args.source / row["graph_path"]
        for repetition in range(3 if row["profile_point_id"] in repeat_panel else 1):
            attempt = args.source / "attempts" / f"{row['profile_point_id']}.{repetition}.json.gz"
            files["native/attempts/" + attempt.name] = attempt
    for name in ("run_perfseer_v32_transfer_labeling.py", "run_perfseer_v32_transfer_training.py"):
        files["scripts/" + name] = ROOT / "scripts" / name
    for path in (ROOT / "src/perfseer_v3.2/containers/transfer").iterdir():
        if path.is_file():
            files[path.name] = path
    for path in (ROOT / "tests").glob("test_perfseer_v32*.py"):
        files["tests/" + path.name] = path
    files["tests/test_perfseer_v31.py"] = ROOT / "tests/test_perfseer_v31.py"
    inventory = {}
    for relative, source in sorted(files.items()):
        if source.is_symlink():
            raise ValueError(f"unexpected package symlink: {source}")
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = file_sha256(source)
        shutil.copyfile(source, target)
        if file_sha256(target) != expected:
            raise ValueError(f"package copy differs: {relative}")
        inventory[relative] = {"sha256": expected, "bytes": target.stat().st_size}
    manifest = {"version": "perfseer_v32_rtx5090_a100_package_v1", "files": inventory,
                "dataset_fingerprint": verification["dataset_fingerprint"], "prediction_hardware": HARDWARE_ID,
                "base_teacher_epoch": base["epoch"], "base_teacher_sha256": file_sha256(args.teacher),
                "base_dataset_fingerprint": base["dataset_fingerprint"]}
    (args.output / "PACKAGE-MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "SHA256SUMS").write_text("".join(f"{item['sha256']}  {name}\n" for name, item in inventory.items()) +
                                          f"{file_sha256(args.output / 'PACKAGE-MANIFEST.json')}  PACKAGE-MANIFEST.json\n")
    subprocess.run([sys.executable, str(args.output / "verify_package.py"), "--runtime"], check=True)
    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                bundle.write(path, str(path.relative_to(args.output.parent)))
    archive.with_suffix(".zip.sha256").write_text(f"{file_sha256(archive)}  {archive.name}\n")
    print(f"Verified package: {archive}", flush=True)


if __name__ == "__main__":
    main()
