"""Generate the CV model-architecture corpus defined by dataset/cv_model_families.md.

The corpus is the predictor's training data: one entry is one model architecture
file to be generated and profiled. It is independent of the GPU used for labeling.

Each entry = family x architecture-field grid x operator substitutions:
  ffn         baseline MLP / conv1x1 expansion, or SwiGLU
  conv_block  baseline conv stack, or ConvNeXt (V1 for small, V2 with GRN for large)
  upsample    conv_transpose, conv + Upsample, or conv + PixelShuffle
  attention   mha, gqa, or mla

Size target: 6,320 architectures, matching the original CV corpus
(6,320 of the 10,005-template calibration pack; x4 precisions = 25,280 labels).

Output: dataset/cv_corpus/architectures.jsonl + per-family summary on stdout.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset/cv_corpus"

# ConvNeXt version rule from the family document: V1 for small models, V2 with
# global response normalization for large ones. Threshold in millions of params,
# estimated from the family's own size knobs.
CONVNEXT_V2_PARAM_THRESHOLD_M = 25.0


def field_grids() -> dict[str, dict[str, list]]:
    """Architecture-field grid per family, using the fields each family declares."""
    return {
        "resnet50": {
            "width_multiplier": [0.5, 0.75, 1.0, 1.25, 1.5],
            "stage_depths": [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 12, 3],
                             [3, 4, 23, 3], [3, 8, 36, 3], [2, 3, 4, 2]],
            "input_resolution": [128, 160, 224, 288, 320],
            "act": ["relu", "mish", "prelu", "silu"],
        },
        "efficientnet_b0_b4": {
            "width_multiplier": [0.8, 1.0, 1.1, 1.2, 1.4, 1.6],
            "depth_multiplier": [0.8, 1.0, 1.1, 1.2, 1.4, 1.8],
            "input_resolution": [224, 240, 260, 300, 380],
        },
        "mobilenet_v3_large": {
            "width_multiplier": [0.35, 0.5, 0.75, 1.0, 1.25],
            "input_resolution": [128, 160, 192, 224, 288, 320],
        },
        "inception_v3": {
            "aux_logits": [True, False],
            "input_resolution": [224, 256, 299, 320, 384],
        },
        "vit_s16": {
            "depth": [4, 6, 8, 12, 16, 18, 24],
            "heads": [3, 6, 8, 12],
            "patch_size": [8, 14, 16, 32],
            "input_resolution": [160, 224, 320, 384],
        },
        "swin_t": {
            "window_size": [7, 8, 12, 16],
            "depths": [[2, 2, 6, 2], [2, 2, 18, 2], [2, 2, 2, 2], [2, 6, 12, 2]],
            "heads": [[3, 6, 12, 24], [4, 8, 16, 32], [2, 4, 8, 16]],
            "input_resolution": [160, 224, 320, 384],
        },
        "mlp_mixer_s": {
            "depth": [8, 12, 16, 24],
            "channel_dim": [384, 512, 768, 1024],
            "token_dim": [128, 256, 384, 512],
            "patch_size": [8, 16, 32],
        },
        "stn_cnn": {
            "localization_width": [16, 32, 64, 128, 256],
            "input_resolution": [128, 160, 224, 288, 320],
        },
        "unet": {
            "depth": [3, 4, 5, 6],
            "base_channels": [16, 32, 48, 64, 96],
            "input_resolution": [192, 256, 384, 512],
            "norm": ["batchnorm", "groupnorm", "layernorm"],
        },
        "pix2pix": {
            "generator_width": [32, 64, 96, 128],
            "discriminator_width": [32, 64, 96],
            "input_resolution": [128, 256, 512],
        },
        "restormer": {
            "depths": [[2, 3, 3, 4], [4, 6, 6, 8], [1, 2, 2, 3]],
            "heads": [[1, 2, 4, 8], [2, 4, 8, 16], [1, 1, 2, 4]],
            "width": [16, 24, 32, 48],
            "input_resolution": [96, 128, 192, 256],
        },
    }


def substitution_axes() -> dict[str, dict[str, list[str]]]:
    """Applicable operator substitutions per family, from cv_model_families.md."""
    ffn = ["baseline", "swiglu"]
    conv = ["baseline", "convnext"]
    up = ["conv_transpose", "conv_upsample", "conv_pixelshuffle"]
    attn = ["mha", "gqa", "mla"]
    return {
        "resnet50": {"conv_block": conv},
        "efficientnet_b0_b4": {"ffn": ffn},
        "mobilenet_v3_large": {"ffn": ffn},
        "inception_v3": {"conv_block": conv},
        "vit_s16": {"ffn": ffn, "attention": attn},
        "swin_t": {"ffn": ffn, "attention": attn},
        "mlp_mixer_s": {"ffn": ffn},
        "stn_cnn": {"conv_block": conv},
        "unet": {"conv_block": conv, "upsample": up},
        "pix2pix": {"upsample": up},
        "restormer": {"ffn": ffn, "attention": attn, "upsample": up},
    }


ADAPTERS = {
    "resnet50": ["histopathologic-cancer", "dogs-vs-cats"],
    "efficientnet_b0_b4": ["dog-breed", "plant-pathology"],
    "mobilenet_v3_large": ["aerial-cactus", "dog-breed"],
    "inception_v3": ["dogs-vs-cats", "aptos2019"],
    "vit_s16": ["siim-isic-melanoma", "ranzcr-clip"],
    "swin_t": ["siim-isic-melanoma", "ranzcr-clip"],
    "mlp_mixer_s": ["leaf-classification", "dog-breed"],
    "stn_cnn": ["aerial-cactus", "dogs-vs-cats"],
    "unet": ["denoising-dirty-documents"],
    "pix2pix": ["denoising-dirty-documents"],
    "restormer": ["denoising-dirty-documents"],
}


def estimate_params_m(family: str, fields: dict) -> float:
    """Rough parameter count in millions, used only to pick ConvNeXt V1 vs V2."""
    if family == "resnet50":
        return 25.6 * fields["width_multiplier"] ** 2 * (sum(fields["stage_depths"]) / 16)
    if family == "inception_v3":
        return 23.8
    if family == "stn_cnn":
        return 0.5 + fields["localization_width"] / 32.0
    if family == "unet":
        return (fields["base_channels"] / 64.0) ** 2 * 31.0 * (fields["depth"] / 4.0)
    return 0.0


def expand(family: str, grid: dict, subs: dict, cap: int) -> list[dict]:
    """Cartesian product of field grid and substitution axes, evenly subsampled to cap."""
    field_names = list(grid)
    field_combos = list(itertools.product(*(grid[k] for k in field_names)))
    sub_names = list(subs)
    sub_combos = list(itertools.product(*(subs[k] for k in sub_names))) or [()]

    entries = []
    for fc in field_combos:
        fields = dict(zip(field_names, fc))
        for sc in sub_combos:
            substitutions = dict(zip(sub_names, sc))
            if substitutions.get("conv_block") == "convnext":
                big = estimate_params_m(family, fields) >= CONVNEXT_V2_PARAM_THRESHOLD_M
                substitutions["conv_block"] = "convnext_v2_grn" if big else "convnext_v1"
            entries.append({"family_id": family, "modality": "vision",
                            "fields": fields, "substitutions": substitutions,
                            "adapter_ids": ADAPTERS[family]})

    if len(entries) <= cap:
        return entries
    stride = len(entries) / cap  # even subsample keeps every axis represented
    return [entries[int(i * stride)] for i in range(cap)]


# Quotas scaled from the original CV corpus proportions (resnet-heavy, then
# efficientnet, then transformers), summing to 6,320.
QUOTAS = {
    "resnet50": 1300,
    "efficientnet_b0_b4": 900,
    "vit_s16": 900,
    "swin_t": 700,
    "unet": 700,
    "restormer": 600,
    "mobilenet_v3_large": 400,
    "mlp_mixer_s": 320,
    "pix2pix": 250,
    "inception_v3": 150,
    "stn_cnn": 100,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=6320,
                    help="total architectures; original CV corpus was 6,320")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    grids, subs = field_grids(), substitution_axes()

    # Full grid size per family bounds how many distinct architectures exist.
    capacity = {}
    for fam, grid in grids.items():
        n_fields = 1
        for v in grid.values():
            n_fields *= len(v)
        n_subs = 1
        for v in subs.get(fam, {}).values():
            n_subs *= len(v)
        capacity[fam] = n_fields * n_subs

    # Desired quotas scaled to the target, then clipped to capacity; the shortfall
    # is redistributed to families that still have headroom, so the total is met.
    scale = args.target / sum(QUOTAS.values())
    alloc = {f: min(round(q * scale), capacity[f]) for f, q in QUOTAS.items()}
    while sum(alloc.values()) < args.target:
        headroom = {f: capacity[f] - alloc[f] for f in alloc if capacity[f] > alloc[f]}
        if not headroom:
            break
        deficit = args.target - sum(alloc.values())
        total_head = sum(headroom.values())
        for f, h in headroom.items():
            alloc[f] += min(h, max(1, round(deficit * h / total_head)))
        overshoot = sum(alloc.values()) - args.target
        for f in sorted(alloc, key=lambda k: -alloc[k]):
            if overshoot <= 0:
                break
            cut = min(overshoot, alloc[f])
            alloc[f] -= cut
            overshoot -= cut

    corpus, summary = [], {}
    for family, cap in alloc.items():
        entries = expand(family, grids[family], subs.get(family, {}), cap)
        for i, e in enumerate(entries):
            e["model_id"] = f"{family}_{i:05d}"
        corpus.extend(entries)
        summary[family] = (len(entries), capacity[family])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "architectures.jsonl", "w") as f:
        for e in corpus:
            f.write(json.dumps(e) + "\n")

    print(f"total architectures: {len(corpus)}  (x4 precisions = {4*len(corpus)} labels)\n")
    print(f"{'family':22s} {'sampled':>8s} {'full grid':>10s} substitution axes")
    for fam, (n, raw) in sorted(summary.items(), key=lambda kv: -kv[1][0]):
        axes = ", ".join(subs.get(fam, {})) or "none"
        print(f"{fam:22s} {n:8d} {raw:10d} {axes}")


if __name__ == "__main__":
    main()
