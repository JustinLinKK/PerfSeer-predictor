"""Final evaluation of one preselected recipe after validation checkpoint selection."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from support import (provenance, load_data, split_rows, validation_key, predict, report,
                     restore_student, sha, atomic_save, write_json, signature)
from design import WINNER_ID

def final_evaluation(config, directory, selected_hash, selection_hash, device='cpu'):
    directory = Path(directory)
    receipt = directory / 'final_evaluation.json'
    binding = {'selected_checkpoint_sha256': selected_hash, 'selection_sha256': selection_hash}
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if any((saved.get(k) != v for k, v in binding.items())):
            raise ValueError('Final evaluation already exists for another selection')
        return saved
    base, data = load_data(config)
    prov = provenance(config)
    if any((config['provenance'].get(k) != v for k, v in prov.items())):
        raise ValueError('Training source changed before final evaluation')
    parts, _ = split_rows(config, data, prov['dataset_sha256'])
    if len(parts['holdout']):
        rows = parts['holdout']
        key = validation_key(config, data, rows)
        scope = 'historical_validation_holdout'
        binding.update(dataset_sha256=prov['dataset_sha256'], key_sha256=prov['validation_key_sha256'], indices_sha256=signature(rows.tolist()), subset_sha256=prov['subset_sha256'])
    elif config.get('test_dataset') and config.get('test_key'):
        data = torch.load(config['test_dataset'], weights_only=False, map_location='cpu', mmap=True)
        rows = torch.arange(len(data['sample_ids']))
        with np.load(config['test_key'], allow_pickle=False) as sealed:
            names = list(map(str, sealed['sample_ids'].tolist()))
            if len(names) != len(set(names)) or set(names) != set(map(str, data['sample_ids'])):
                raise ValueError('Test key/input identities differ')
            lookup = {name: i for i, name in enumerate(names)}
            order = [lookup[str(name)] for name in data['sample_ids']]
            key = {k: sealed[k][order] for k in ('targets', 'raw_targets_v4', 'families')}
        scope = 'sealed_test'
        binding.update(dataset_sha256=sha(config['test_dataset']), key_sha256=sha(config['test_key']))
    else:
        return {**binding, 'scope': 'no_independent_test_available', 'evaluated': False, 'note': 'A10 full-data pretraining has no independent sealed test; do not invent a holdout.'}
    if sha(directory / 'best_model.pt') != selected_hash:
        raise ValueError('Selected checkpoint changed before final evaluation')
    prediction_path = directory / 'final_predictions.pt'
    if prediction_path.exists():
        saved = torch.load(prediction_path, weights_only=True, map_location='cpu')
        if saved['binding'] != binding:
            raise ValueError('Final predictions belong to another checkpoint/key')
        prediction = saved['prediction'].numpy()
    else:
        model, _ = restore_student(directory / 'best_model.pt')
        prediction = predict(model.to(device), base, data, rows, device)
        atomic_save({'binding': binding, 'prediction': torch.from_numpy(prediction)}, prediction_path)
    value = {**binding, 'scope': scope, 'evaluated': True, 'rows': len(rows), 'used_for_selection': False, 'report': report(prediction, key), 'prediction_sha256': sha(prediction_path)}
    write_json(value, receipt)
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    directory, output = Path(args.run), Path(args.output)
    result = json.loads((directory / "result.json").read_text())
    if result["config"]["smoke"] or result["candidate"] not in ("PRETRAIN", WINNER_ID):
        raise ValueError("Finalization requires a full run of a published recipe")
    if signature(result["config"]) != result["config_hash"]:
        raise ValueError("Training configuration changed")
    for name, digest in result["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("Selected artifact changed")
    saved = torch.load(directory / "best_model.pt", weights_only=False, map_location="cpu")
    if saved["config"] != result["config"] or saved["validation_score"] != result["best_validation_score"]:
        raise ValueError("Selected checkpoint/result mismatch")
    del saved
    # The published recipe was fixed outside this run. Only checkpoint selection remains.
    lock = {"candidate": result["candidate"], "config_hash": result["config_hash"],
            "checkpoint_sha256": result["files"]["best_model.pt"], "used_heldout_for_selection": False}
    lock_path = directory / "recipe_selection.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Cannot replace a fixed selected checkpoint")
    write_json(lock, lock_path)
    final_evaluation(result["config"], directory, lock["checkpoint_sha256"], sha(lock_path), args.device)
    output.mkdir(parents=True, exist_ok=True)
    names = ["best_model.pt", "student.ts", "result.json", "recipe_selection.json"]
    if (directory / "final_evaluation.json").exists():
        names.append("final_evaluation.json")
    for name in names:
        target = output / name
        if target.exists() and sha(target) != sha(directory / name):
            raise ValueError("Another artifact already occupies output")
        shutil.copyfile(directory / name, target)
    write_json({"candidate": result["candidate"], "files": {name: sha(output / name) for name in names}},
               output / "manifest.json")

if __name__ == "__main__":
    main()
