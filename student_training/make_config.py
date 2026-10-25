import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""Generate runnable configs from verified, extracted release asset layouts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

def make_config(assets, output):
    assets = Path(assets).resolve()
    if (assets / 'PerfSeer-release-v1').is_dir():
        assets = assets / 'PerfSeer-release-v1'
    checkpoint = assets / 'checkpoints/P_A10/best_model.pt'
    architecture = checkpoint.with_name('architecture.pt')
    if not architecture.is_file():
        architecture = Path(__file__).resolve().parent.parent / 'portable/architecture.pt'
    output = Path(output).resolve()
    config = {'candidate': 'PRETRAIN', 'hardware': 'A10', 'teacher': {'checkpoint': str(checkpoint), 'kind': 'Adapted', 'hardware': 'A10', 'identity_override': 190}, 'dataset': str(assets / 'A10/data/A10/dataset.pt'), 'normalization': str(assets / 'A10/normalization.pt'), 'architecture': str(architecture), 'validation_key': str(assets / 'A10/A10-validation-key.npz'), 'cache': str(output.parent / 'cache/P_A10'), 'output': str(output), 'epochs': 40, 'batch_size': 32, 'ema_decay': 0.999, 'cpu_threads': 4}
    for name in ('dataset', 'normalization', 'architecture', 'validation_key'):
        if not Path(config[name]).is_file():
            raise FileNotFoundError('Missing extracted asset: ' + config[name])
    if not checkpoint.is_file():
        raise FileNotFoundError('Missing extracted teacher: ' + str(checkpoint))
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True, help='extraction directory or PerfSeer-release-v1 itself')
    parser.add_argument('--run-output', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    value = make_config(args.assets, args.run_output)
    path = Path(args.config)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Refusing to replace a different training configuration')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))
if __name__ == '__main__':
    main()
