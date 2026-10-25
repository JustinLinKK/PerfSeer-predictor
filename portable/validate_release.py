#!/usr/bin/env python3
"""Validate the frozen portable source package without importing Torch or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import re

TRAIN_FILES = ('train.py', 'teachers.py', 'teacher_components.py', 'training_support.py', 'io_utils.py',
               'base_loader.py', 'aba3_training_support.py', 'v5_objectives.py', 'model/starter.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(root, name):
    path = root / name
    require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe manifest path')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source-root', type=Path, help='Optional release root containing data/ and portable/')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:
        path = safe(root, item['path'])
        require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'changed or missing portable file: ' + item['path'])
        if path.suffix == '.py':
            source = path.read_text()
            for forbidden in ('/' + 'home/', '/' + 'mnt/', 'C:' + '\\\\Users\\\\'):
                require(forbidden not in source, 'absolute developer path: ' + item['path'])
            require(not re.search(r'^\s*(?:from|import)\s+(?:common|transfer_models|train_continue|train_p)(?:\s|\.)', source, re.M),
                    'historical trainer import: ' + item['path'])
    digest = hashlib.sha256()
    for name in TRAIN_FILES:
        digest.update(name.encode()); digest.update((root / name).read_bytes())
    require(digest.hexdigest() == manifest['trainer_code_sha256'], 'trainer hash differs')
    for kind, count in [('P1', 253644481), ('Adapted', 253644481), ('T7', 255692483)]:
        receipt = json.loads((root / 'verification' / (kind + '-portable-parity.json')).read_text())
        require(receipt['passed'] and receipt['bit_identical'] and receipt['prediction_max_absolute_difference'] == 0,
                'missing exact inference parity: ' + kind)
        require(receipt['stdout']['parameter_count'] == count and receipt['stdout']['finite'], 'parameter count/finite check failed')
        require(all(receipt[key] for key in ['python_isolated_mode', 'working_directory_outside_repository',
                                            'PYTHONPATH_empty', 'CUDA_VISIBLE_DEVICES_empty']), 'isolation proof failed')
    for kind in ('P1', 'T7'):
        receipt = json.loads((root / 'verification' / (kind + '-resume-verification.json')).read_text())
        require(receipt['passed'] and not receipt['cuda_used'] and receipt['portable_code_sha256'] == digest.hexdigest(),
                'training proof does not bind final code: ' + kind)
        require(all(receipt[key] for key in ['state_optimizer_rng_history_lr_cursor_exact', 'selected_state_exact',
                                            'finite_nonzero_gradients_verified', 'source_unchanged']), 'resume proof failed')
        require(receipt['changed_tensors_from_selected_baseline'] > 0, 'no real parameter update')
    recipe = json.loads((root / 'verification/recipe-parity.json').read_text())
    require(recipe['passed'] and recipe['T7_optimizer_group_names_order_rates_parameters_exact']
            and recipe['T7_memory_loss_and_gradient_exact'], 'recipe parity failed')
    pipeline = json.loads((root / 'verification/fresh-pipeline.json').read_text())
    require(pipeline['passed'] and pipeline['packer_sha256'] == sha(root / 'pack_fresh.py')
            and pipeline['input_only_inference_finite_and_physical'] and pipeline['epoch_contract_verified']
            and pipeline['hardware'] == 'nvidia_geforce_rtx_5090_local', 'fresh pipeline proof failed')
    data_files = 0
    if args.source_root:
        allowlist = json.loads((root / 'SOURCE_ALLOWLIST.json').read_text())
        for item in allowlist['data_files']:
            path = safe(args.source_root, item['destination'])
            require(path.is_file() and sha(path) == item['sha256'], 'missing/changed data source: ' + item['destination'])
            data_files += 1
    result = {'passed': True, 'portable_files': len(manifest['files']), 'data_source_files_checked': data_files,
              'manifest_sha256': sha(root / 'SOURCE_MANIFEST.json'), 'trainer_code_sha256': digest.hexdigest(),
              'torch_imported': False, 'gpu_accessed': False, 'scope': 'Source integrity and frozen verification receipts'}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
