#!/usr/bin/env python3
"""Verify the release asset manifest and optionally extract its exact members."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or not path.parts:
        raise ValueError('Unsafe archive path')
    return path

def verify_extract(asset, root, destination=None):
    archive = root / safe_relative(asset['path'])
    if archive.stat().st_size != asset['bytes'] or sha(archive) != asset['sha256']:
        raise ValueError(f"Asset hash mismatch: {asset['path']}. Run git lfs pull first.")
    expected = {m['path']: m for m in asset['members']}
    if len(expected) != len(asset['members']):
        raise ValueError('Duplicate manifest members')
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            relative = safe_relative(member.name)
            if not member.isfile() or member.name in seen or member.name not in expected:
                raise ValueError('Unexpected or duplicate archive member')
            spec = expected[member.name]
            if member.size != spec['bytes']:
                raise ValueError('Archive member size changed')
            target = destination / relative if destination else None
            existing = False
            if target:
                for component in (target, *target.parents):
                    if component.is_symlink():
                        raise ValueError('Extraction target contains symlink')
                if target.exists():
                    if not target.is_file() or target.stat().st_size != spec['bytes'] or sha(target) != spec['sha256']:
                        raise ValueError('Refusing to overwrite a different extracted file')
                    existing = True
                target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting') if target and not existing else None
            output = None
            digest = hashlib.sha256()
            count = 0
            try:
                if temporary:
                    output = temporary.open('xb')
                stream = tar.extractfile(member)
                while block := stream.read(8 * 1024 * 1024):
                    digest.update(block)
                    count += len(block)
                    if output:
                        output.write(block)
                if count != spec['bytes'] or digest.hexdigest() != spec['sha256']:
                    raise ValueError('Archive member checksum mismatch')
                if output:
                    output.close()
                    output = None
                    temporary.replace(target)
            finally:
                if output:
                    output.close()
                if temporary:
                    temporary.unlink(missing_ok=True)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Missing archive members')
    return {'asset': asset['path'], 'verified': True, 'members': len(seen), 'extracted': destination is not None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('assets/manifest.json'))
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--extract', type=Path)
    parser.add_argument('--asset', action='append', help='Optional archive basename; repeat to select assets')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    chosen = [a for a in manifest['assets'] if not args.asset or Path(a['path']).name in args.asset]
    if not chosen or (args.asset and set(args.asset) != {Path(a['path']).name for a in chosen}):
        raise ValueError('Requested assets are absent from manifest')
    for item in chosen:
        print(json.dumps(verify_extract(item, args.root.resolve(), args.extract.resolve() if args.extract else None)), flush=True)

if __name__ == '__main__':
    main()
