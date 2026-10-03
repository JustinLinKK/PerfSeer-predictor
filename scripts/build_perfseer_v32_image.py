#!/usr/bin/env python3
"""Stage and verify a minimal v3.2 image context before invoking Docker."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--context", type=Path, default=Path("record/perfseer-v32/build-context"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    context = args.context.resolve()
    if context.exists():
        raise ValueError("use a fresh build context directory")
    files = [root / "pyproject.toml"]
    for package in ("perfseer_v3.1", "perfseer_v3.2"):
        folder = root / "src" / package
        files.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in {".py", ".json", ".md"})
    folder = root / "src/perfseer_v3"
    for directory, children, names in os.walk(folder):
        children[:] = [name for name in children if name not in {"dataset_with_label", "__pycache__"}]
        files.extend(Path(directory) / name for name in names)
    files.extend(p for p in (root / "src/perfseer_v3.2/dataset_with_label/ready_for_train_12").rglob("*") if p.is_file())
    hashes = {}
    for source in sorted(files):
        if source.is_symlink() or source.suffix in {".pyc", ".pt", ".pth", ".zip", ".key", ".pem"} or source.name == "kaggle.json":
            raise ValueError(f"unexpected image input: {source}")
        relative = source.relative_to(root)
        target = context / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = digest(source)
        if digest(target) != hashes[str(relative)]:
            raise ValueError(f"staged input hash differs: {relative}")
    for name in ("Dockerfile", "Dockerfile.dockerignore"):
        source = root / "src/perfseer_v3.2/containers/training" / name
        shutil.copyfile(source, context / name)
        hashes[name] = digest(source)
        if digest(context / name) != hashes[name]:
            raise ValueError(f"staged {name} differs")
    actual = {str(p.relative_to(context)) for p in context.rglob("*") if p.is_file()}
    if actual != set(hashes):
        raise ValueError("unexpected staged files")
    (context.parent / (context.name + "-manifest.json")).write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Verified {len(hashes)} image input files", flush=True)
    subprocess.run(["docker", "build", "--provenance=false", "--progress=plain", "-t", args.tag, str(context)], check=True)


if __name__ == "__main__":
    main()
