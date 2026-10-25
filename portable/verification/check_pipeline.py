#!/usr/bin/env python3
"""Verify a fresh feature pack and its input-only prediction output on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teachers import bootstrap
bootstrap()
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.pack / 'verification.json').read_text())
    assert receipt['passed'] and receipt['all_targets_recomputed_from_observations']
    assert receipt['all_generated_source_identities_verified']
    assert not receipt['normalizer_refitted'] and not receipt['cuda_used_for_feature_capture']
    for name, expected in receipt['files'].items():
        assert sha(args.pack / name) == expected, name
    bundle = torch.load(args.pack / 'all-inputs.pt', weights_only=False, map_location='cpu')
    data = np.load(args.predictions, allow_pickle=False)
    prediction = data['predictions']
    assert data['sample_ids'].tolist() == bundle['sample_ids']
    assert prediction.shape == (receipt['accepted_rows'], 6) and np.isfinite(prediction).all()
    assert (prediction[:, [0, 1, 2, 4, 5]] > 0).all()
    assert ((prediction[:, 3] >= 0) & (prediction[:, 3] <= 100)).all()
    assert np.array_equal(data['raw_steps_per_epoch'], bundle['raw_steps_per_epoch'].numpy())
    assert np.allclose(prediction[:, 2], prediction[:, 0] * data['raw_steps_per_epoch'], rtol=1e-7)
    assert data['normalization_sha256'].item() == bundle['normalization'].sha256
    assert not torch.cuda.is_initialized()
    result = {**{k: v for k, v in receipt.items() if k != 'files'},
              'input_only_inference_finite_and_physical': True, 'sample_ids_exact': True,
              'epoch_contract_verified': True, 'prediction_device': 'cpu',
              'prediction_kind': data['kind'].item(), 'prediction_sha256': sha(args.predictions),
              'input_bundle_sha256': sha(args.pack / 'all-inputs.pt'),
              'packer_sha256': sha(Path(__file__).resolve().parents[1] / 'pack_fresh.py'),
              'scope': 'Five real local GPU observations; static packing/inference verification only, not model accuracy or a complete new GPU dataset'}
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
