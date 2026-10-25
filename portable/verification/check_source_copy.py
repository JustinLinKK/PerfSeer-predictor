#!/usr/bin/env python3
"""Verify the allowlisted clean release tree in an isolated temporary directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    allowlist = json.loads((source / 'portable/SOURCE_ALLOWLIST.json').read_text())
    environment = {**os.environ, 'PYTHONPATH': '', 'CUDA_VISIBLE_DEVICES': ''}
    with tempfile.TemporaryDirectory(prefix='perfseer-release-source-') as directory:
        root = Path(directory)
        for item in allowlist['data_files'] + allowlist['portable_files']:
            destination = root / item['destination']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / item['source'], destination)
        for name in allowlist['control_files']:
            shutil.copy2(source / name, root / name)
        def execute(script, *arguments, check=True, isolated=True):
            command = [sys.executable] + (['-I'] if isolated else []) + [str(root / script), *arguments]
            return subprocess.run(command,
                                  cwd=root, env=environment, text=True, capture_output=True, check=check)
        validation = execute('portable/validate_release.py', '--source-root', str(root))
        tests = execute('data/tests/test_data_release.py')
        for script in ['portable/infer.py', 'portable/train_pretrain.py', 'portable/train_transfer.py',
                       'portable/pack_fresh.py', 'portable/score.py', 'data/generate.py', 'data/label.py']:
            execute(script, '--help', isolated=script != 'data/label.py')
        changed = root / 'portable/pack_fresh.py'
        changed.write_text(changed.read_text() + '\n# integrity test\n')
        rejected = execute('portable/validate_release.py', check=False)
        assert rejected.returncode != 0 and 'changed or missing portable file' in rejected.stderr
        result = {'passed': True, 'portable_python_isolated_mode': True, 'data_labeler_documented_normal_python': True, 'PYTHONPATH_empty': True,
                  'CUDA_VISIBLE_DEVICES_empty': True, 'working_directory_outside_repository': True,
                  'allowlisted_copy_verified': True, 'generator_identity_tests': tests.stderr.strip(),
                  'seven_cli_entrypoints_load': True, 'tampered_source_rejected': True,
                  'validation': json.loads(validation.stdout),
                  'source_manifest_sha256': hashlib.sha256((source / 'portable/SOURCE_MANIFEST.json').read_bytes()).hexdigest()}
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
