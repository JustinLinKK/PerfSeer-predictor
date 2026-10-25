#!/usr/bin/env python3
"""Build a reviewable source allowlist and integrity manifest from explicit roots."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pinned-package', type=Path, required=True)
    parser.add_argument('--architecture-source', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime_paths = sorted(p for p in (root / 'runtime').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for path in runtime_paths:
        assert sha(path) == sha(args.pinned_package / path.relative_to(root)), path.name
    for name in ('aba3_training_support.py', 'v5_objectives.py', 'model/starter.py'):
        assert sha(root / name) == sha(args.pinned_package / name), name
    assert sha(root / 'architecture.pt') == sha(args.architecture_source)
    receipt_names = ['environment.json', 'recipe-parity.json', 'fresh-pipeline.json',
                     'P1-portable-parity.json', 'Adapted-portable-parity.json', 'T7-portable-parity.json',
                     'P1-resume-verification.json', 'T7-resume-verification.json']
    portable = [p for p in root.iterdir() if p.is_file() and p.name not in ('SOURCE_ALLOWLIST.json', 'SOURCE_MANIFEST.json', 'FINAL_VALIDATION.json')]
    portable += runtime_paths
    portable += [root / 'model/starter.py']
    portable += [root / 'verification' / name for name in receipt_names]
    portable += sorted((root / 'verification').glob('*.py'))
    portable = sorted(set(portable))
    data = sorted(args.data_root.glob('*.py'))
    data += sorted((args.data_root / 'runtime').glob('*.py'))
    data += sorted((args.data_root / 'specs').rglob('*.json'))
    data += sorted((args.data_root / 'specs').rglob('*.py.in'))
    data += sorted((args.data_root / 'tests').glob('*.py'))
    # The archived README is historical provenance; the maintained clean guide is FRESH_DATA.md.
    data += [args.data_root / 'README.md', args.data_root / 'RELEASE_STATUS.json']
    def entry(path, base, prefix):
        destination = prefix + '/' + path.relative_to(base).as_posix()
        return {'source': destination, 'destination': destination, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    allowlist = {'schema': 'perfseer-source-copy-allowlist-v1', 'source_root': 'release_v1 staging root',
                 'destination_root': 'clean release branch root',
                 'data_files': [entry(p, args.data_root, 'data') for p in sorted(set(data))],
                 'portable_files': [entry(p, root, 'portable') for p in portable],
                 'control_files': ['portable/SOURCE_ALLOWLIST.json', 'portable/SOURCE_MANIFEST.json'],
                 'excluded': ['generated per-workload programs', 'label records', 'feature packs', 'teacher checkpoints',
                              'temporary smoke checkpoints', '__pycache__', '*.pyc'],
                 'guide': 'portable/FRESH_DATA.md'}
    (root / 'SOURCE_ALLOWLIST.json').write_text(json.dumps(allowlist, indent=2) + '\n')
    digest = hashlib.sha256()
    for name in ('train.py', 'teachers.py', 'teacher_components.py', 'training_support.py', 'io_utils.py',
                 'base_loader.py', 'aba3_training_support.py', 'v5_objectives.py', 'model/starter.py'):
        digest.update(name.encode()); digest.update((root / name).read_bytes())
    manifest = {'schema': 'perfseer-portable-source-manifest-v1',
                'trainer_code_sha256': digest.hexdigest(), 'architecture_sha256': sha(root / 'architecture.pt'),
                'pinned_runtime_files': len(runtime_paths), 'pinned_runtime_byte_identical': True,
                'copied_support_byte_identical': True, 'architecture_byte_identical': True,
                'extraction_proof': 'verification/recipe-parity.json',
                'required_receipts': ['verification/' + name for name in receipt_names],
                'files': [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                          for p in sorted(portable + [root / 'SOURCE_ALLOWLIST.json'])]}
    (root / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'portable_files': len(manifest['files']), 'data_source_files': len(allowlist['data_files']),
                      'pinned_runtime_files': len(runtime_paths), 'trainer_code_sha256': digest.hexdigest(),
                      'manifest_sha256': sha(root / 'SOURCE_MANIFEST.json')}))


if __name__ == '__main__':
    main()
