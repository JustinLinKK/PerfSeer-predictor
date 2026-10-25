"""Fail closed when a CUDA trainer has no matching successful feasibility receipt."""
import json
from pathlib import Path
from support import sha, code_hash


def verify_receipt(path, config, prov, batch):
    if not path:
        raise ValueError("CUDA student training requires --preflight from the parent-owned CUDA verifier")
    receipt = json.loads(Path(path).read_text())
    if not receipt.get("passed") or receipt.get("smoke") or receipt.get("device") != "cuda":
        raise ValueError("A successful production CUDA receipt is required")
    binding = receipt["bindings"]
    if binding["code_sha256"] != code_hash() or receipt["batch_size"] != batch:
        raise ValueError("CUDA receipt code/batch mismatch; explicitly apply verified batch choice")
    hardware = binding["hardware"][config["hardware"]]
    for key in ("dataset", "normalization", "architecture"):
        if hardware[key + "_sha256"] != prov[key + "_sha256"]:
            raise ValueError("CUDA receipt input contract mismatch: " + key)
    name = config["hardware"] + "_" + config["candidate"]
    case = receipt["cases"][name]
    if case["batch_size"] != batch or not all(case[k] for k in (
            "finite_nonzero_gradients", "optimizer_update", "frozen_parameters_unchanged", "buffers_unchanged")):
        raise ValueError("CUDA candidate mechanics were not verified")
    if config["candidate"] != "PRETRAIN":
        if binding["transfer_subset_sha256"] != prov["subset_sha256"]:
            raise ValueError("CUDA transfer subset changed")
        if binding["initial_student_sha256"] != prov["initial_student_sha256"]:
            raise ValueError("CUDA source student changed")
    return sha(path)
