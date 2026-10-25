"""Generate runnable configs from verified, extracted release asset layouts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from design import WINNER_ID, SELECTED

def make_config(assets, mode, hardware, output):
    assets = Path(assets).resolve()
    if (assets / "PerfSeer-release-v1").is_dir():
        assets = assets / "PerfSeer-release-v1"
    if mode == "transfer" and hardware != "A10":
        raise ValueError("The published transfer route is A100 to A10")
    if hardware not in ("A10", "A10G", "A100"):
        raise ValueError("Unknown released hardware")
    identifier = "T_A10_A100" if mode == "transfer" else "P_" + hardware
    kind = "transfer_T7" if mode == "transfer" else ("P1" if hardware == "A100" else "Adapted")
    checkpoint = assets / "checkpoints" / identifier / "best_model.pt"
    architecture = checkpoint.with_name("architecture.pt")
    if not architecture.exists():
        architecture = Path(__file__).resolve().parent.parent / "portable/architecture.pt"
    output = Path(output).resolve()
    config = {"candidate": WINNER_ID if mode == "transfer" else "PRETRAIN", "hardware": hardware,
        "teacher": {"checkpoint": str(checkpoint), "kind": kind, "hardware": hardware,
                    "identity_override": None if kind == "P1" else 190},
        "dataset": str(assets / hardware / "data" / hardware / "dataset.pt"),
        "normalization": str(assets / hardware / "normalization.pt"), "architecture": str(architecture),
        "validation_key": str(assets / hardware / (hardware + "-validation-key.npz")),
        "cache": str(output.parent / "cache" / identifier), "output": str(output),
        "epochs": 40, "batch_size": 32, "ema_decay": .999, "cpu_threads": 4}
    if mode == "transfer":
        config.update(route="A10_A100", subset=str(assets / "transfer/A10/selection.json"))
        if SELECTED["initialization"] == "source_student":
            config["initial_student"] = str(assets / "checkpoints/S_A100/best_model.pt")
    elif hardware != "A10":
        config.update(test_dataset=str(assets / hardware / "data" / hardware / "test-inputs.pt"),
                      test_key=str(assets / hardware / (hardware + "-test-key.npz")))
    for name in ("dataset", "normalization", "architecture", "validation_key", "subset", "initial_student", "test_dataset", "test_key"):
        if name in config and not Path(config[name]).is_file():
            raise FileNotFoundError("Required extracted asset missing: " + config[name])
    if not checkpoint.is_file():
        raise FileNotFoundError("Required extracted checkpoint missing: " + str(checkpoint))
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="extraction directory or PerfSeer-release-v1 itself")
    parser.add_argument("--mode", choices=("pretrain", "transfer"), required=True)
    parser.add_argument("--hardware", choices=("A10", "A10G", "A100"), required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    value = make_config(args.assets, args.mode, args.hardware, args.run_output)
    path = Path(args.config)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Refusing to overwrite a different training config")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))

if __name__ == "__main__":
    main()
