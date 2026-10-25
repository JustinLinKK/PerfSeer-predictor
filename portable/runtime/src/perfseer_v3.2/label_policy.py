"""Apply the explicitly requested shorter-time reference policy to native rows."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json

from .capture import inference_settings
from .version import HEAD_GROUPS, MODES, TARGET_NAMES, validate_targets

VERSION = "perfseer_v32_shorter_timing_v1"
HEAD_NAMES = ("train_timing", "train_sm", "train_memory", "infer_timing", "infer_sm", "infer_memory")
POLICY = {"version": VERSION, "conflict": "no_common_5pct_interval_in_any_timing_target",
          "training_selection": ["train_epoch_ms", "train_step_wall_ms", "train_step_gpu_ms", "sample_id"],
          "inference_selection": ["infer_step_wall_ms", "infer_step_gpu_ms", "sample_id"],
          "scope": "within_split_and_exact_mode_input_and_dataset_subset",
          "reference_semantics": "user_selected_shorter_time_not_remeasured_truth"}


def _mode_identity(item):
    from .verification import _input_identity

    path, identities = _input_identity(item)
    settings = read_json(path)["training"]["TRAINING_CONFIG"]
    return path, {mode: fingerprint([identities[mode], settings if mode == "training" else inference_settings(settings)])
                  for mode in MODES}


def input_identities(root, rows, workers):
    paths = {str(root / row["input_path"]): row["input_sha256"] for row in rows}
    result = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (path, identity) in enumerate(pool.map(_mode_identity, paths.items()), 1):
            result[str(Path(path).relative_to(root))] = identity
            if index % 100 == 0:
                print(f"label policy input verification: {index}/{len(paths)}", flush=True)
    return result


def group_key(row, mode, identities):
    parts = row["sample_id"].split("::")
    if len(parts) != 7 or parts[0] != row["provenance"]["model_id"]:
        raise ValueError("label policy requires native workload sample identities")
    return fingerprint([row["split"], row["hardware_id"], row["provenance"]["modality"],
                        parts[1:3], mode, identities[row["input_path"]][mode]])


def timing_conflict(values):
    # The intervals [0.95*y, 1.05*y] have a common point iff 19*max <= 21*min.
    return any(19 * max(column) > 21 * min(column) for column in zip(*values, strict=True))


def revise_rows(rows, identities, consensus_changes):
    originals = {row["sample_id"]: row for row in rows}
    if len(originals) != len(rows):
        raise ValueError("duplicate policy sample")
    for row in rows:
        validate_targets([row["targets"][name] for name in TARGET_NAMES])
        if row["targets"] != {name: row["native_targets"][name] for name in TARGET_NAMES}:
            raise ValueError("label policy must start with native measurements")
    revised = deepcopy(rows)
    by_id = {row["sample_id"]: row for row in revised}
    donors, ledger, decisions = {}, [], []
    groups = defaultdict(list)
    for row in revised:
        for mode in MODES:
            groups[(mode, group_key(row, mode, identities))].append(row)
    for change in consensus_changes:
        head = HEAD_NAMES.index(change["head"])
        names = [TARGET_NAMES[i] for i in HEAD_GROUPS[head]]
        row, donor = by_id[change["sample_id"]], originals[change["donor_sample_id"]]
        mode = "training" if head < 3 else "inference"
        key = (row["sample_id"], head)
        if key in donors or row["split"] != "train" or donor["split"] != "train":
            raise ValueError("duplicate or non-training consensus edit")
        if group_key(row, mode, identities) != group_key(donor, mode, identities):
            raise ValueError("consensus donor workload differs")
        before = [row["targets"][name] for name in names]
        after = [donor["targets"][name] for name in names]
        if (change["target_names"] != names or change["original_values"] != before or
                change["estimated_values"] != after or change["verified_measurement_correction"] is not False):
            raise ValueError("consensus donor provenance differs")
        row["targets"].update(zip(names, after, strict=True))
        donors[key] = donor["sample_id"]
        ledger.append({"stage": "previous_consensus", "split": "train", **change})
    for (mode, key), members in sorted(groups.items()):
        head = 0 if mode == "training" else 3
        names = [TARGET_NAMES[i] for i in HEAD_GROUPS[head]]
        values = [[row["targets"][name] for name in names] for row in members]
        if not timing_conflict(values):
            continue
        order = POLICY[f"{mode}_selection"][:-1]
        selected = min(members, key=lambda row: (*[row["targets"][name] for name in order], row["sample_id"]))
        donor_id = donors.get((selected["sample_id"], head), selected["sample_id"])
        after = [originals[donor_id]["targets"][name] for name in names]
        if after != [selected["targets"][name] for name in names]:
            raise ValueError("selected timing tuple has no original donor")
        changed = 0
        for row in members:
            before = [row["targets"][name] for name in names]
            if before == after:
                continue
            row["targets"].update(zip(names, after, strict=True))
            ledger.append({"stage": "shorter_time", "split": row["split"], "sample_id": row["sample_id"],
                           "head": HEAD_NAMES[head], "target_names": names, "previous_values": before,
                           "estimated_values": after, "donor_sample_id": donor_id, "group": key,
                           "verified_measurement_correction": False})
            changed += 1
        decisions.append({"split": selected["split"], "mode": mode, "group": key, "rows": len(members),
                          "changed_rows": changed, "donor_sample_id": donor_id, "selected_values": after})
    for row in revised:
        validate_targets([row["targets"][name] for name in TARGET_NAMES])
        row["target_policy"] = VERSION
    return revised, ledger, decisions


def _checked(root, entry):
    path = root / entry["path"]
    if not path.resolve().is_relative_to(root.resolve()) or file_sha256(path) != entry["sha256"]:
        raise ValueError("label policy artifact hash or path differs")
    return read_json(path)


def verify_policy(root, manifest, workers=4):
    root = Path(root)
    ref = manifest["label_policy"]
    policy = _checked(root, ref)
    if (ref["version"] != VERSION or policy["policy"] != POLICY or
            ref.get("reference_semantics") != POLICY["reference_semantics"]):
        raise ValueError("unsupported label policy")
    if policy["fingerprint"] != fingerprint({k: v for k, v in policy.items() if k != "fingerprint"}):
        raise ValueError("label policy fingerprint differs")
    base = _checked(root, policy["original_manifest"])
    if ("label_policy" in base or base["fingerprint"] != policy["source_dataset_fingerprint"] or
            base["fingerprint"] != fingerprint({k: v for k, v in base.items() if k != "fingerprint"})):
        raise ValueError("original dataset identity differs")
    for name in base:
        if name not in {"fingerprint", "split_files"} and base[name] != manifest[name]:
            raise ValueError("label policy changed non-label dataset contract")
    originals = [row for split in ("train", "validation", "test") for row in _checked(root, base["split_files"][split])]
    identities = _checked(root, policy["input_identities"])
    if identities != input_identities(root, originals, workers):
        raise ValueError("label policy mode input identities differ")
    seed = _checked(root, policy["consensus_changes"])
    seed_manifest = _checked(root, policy["consensus_manifest"])
    if (seed_manifest["fingerprint"] != fingerprint({k: v for k, v in seed_manifest.items() if k != "fingerprint"}) or
            seed_manifest["source_dataset_fingerprint"] != base["fingerprint"] or
            seed_manifest["source_manifest_sha256"] != policy["original_manifest"]["sha256"] or
            seed_manifest["artifacts"]["changes.json.gz"] != policy["consensus_changes"]["sha256"]):
        raise ValueError("consensus seed identity differs")
    revised, ledger, decisions = revise_rows(originals, identities, seed)
    if _checked(root, policy["changes"]) != ledger or _checked(root, policy["decisions"]) != decisions:
        raise ValueError("label policy donor reconstruction differs")
    for split, meta in manifest["split_files"].items():
        expected = [row for row in revised if row["split"] == split]
        if _checked(root, meta) != expected or len(expected) != meta["rows"]:
            raise ValueError("revised labels differ from reconstructed policy")
    return {row["sample_id"]: row for row in revised}


def activate(root, consensus, workers=4):
    from .dataset import verify

    root, consensus = Path(root).resolve(), Path(consensus).resolve()
    manifest = read_json(root / "dataset_manifest.json")
    if "label_policy" in manifest:
        return verify(root, workers)
    verify(root, workers)
    seed_manifest = read_json(consensus / "manifest.json")
    if (seed_manifest["source_dataset_fingerprint"] != manifest["fingerprint"] or
            file_sha256(consensus / "changes.json.gz") != seed_manifest["artifacts"]["changes.json.gz"]):
        raise ValueError("consensus estimates belong to a different source")
    originals = [row for split in ("train", "validation", "test") for row in read_json(root / manifest["split_files"][split]["path"])]
    identities = input_identities(root, originals, workers)
    seed = read_json(consensus / "changes.json.gz")
    revised, ledger, decisions = revise_rows(originals, identities, seed)
    folder = root / "label-policies" / VERSION
    if folder.exists():
        raise ValueError("label policy staging directory already exists; preserve and review it before retrying")
    folder.mkdir(parents=True)

    def save(name, value):
        path = folder / name
        atomic_write(path, value, compress=name.endswith(".gz"))
        return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}

    policy = {"policy": POLICY, "source_dataset_fingerprint": manifest["fingerprint"],
              "original_manifest": save("original-dataset-manifest.json", manifest),
              "consensus_manifest": save("consensus-manifest.json", seed_manifest),
              "consensus_changes": save("consensus-changes.json.gz", seed),
              "input_identities": save("input-identities.json.gz", identities),
              "changes": save("changes.json.gz", ledger), "decisions": save("decisions.json.gz", decisions)}
    candidate = deepcopy(manifest)
    for split in manifest["split_files"]:
        values = [row for row in revised if row["split"] == split]
        candidate["split_files"][split] = {**save(f"{split}-samples.json.gz", values), "rows": len(values)}
    policy["fingerprint"] = fingerprint(policy)
    candidate["label_policy"] = {**save("policy.json", policy), "version": VERSION,
                                 "reference_semantics": POLICY["reference_semantics"]}
    candidate["fingerprint"] = fingerprint({k: v for k, v in candidate.items() if k != "fingerprint"})
    verify_policy(root, candidate, workers)
    if read_json(root / "dataset_manifest.json") != manifest:
        raise ValueError("active dataset changed during label preparation")
    atomic_write(root / "dataset_manifest.json", candidate)
    report = verify(root, workers)
    report.update(changed_rows=sum(a["targets"] != b["targets"] for a, b in zip(originals, revised, strict=True)),
                  changed_values=sum(a["targets"][name] != b["targets"][name] for a, b in zip(originals, revised, strict=True) for name in TARGET_NAMES),
                  conflict_groups=len(decisions), conflict_groups_by_split=dict(Counter(x["split"] for x in decisions)),
                  original_dataset_fingerprint=manifest["fingerprint"], accuracy_improvement_verified=False,
                  reference_semantics=POLICY["reference_semantics"])
    atomic_write(folder / "verification-report.json", report)
    return report


def main():
    from .dataset import DATA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA / "ready_for_train_12")
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(activate(args.dataset, args.consensus, args.workers), flush=True)


if __name__ == "__main__":
    main()
